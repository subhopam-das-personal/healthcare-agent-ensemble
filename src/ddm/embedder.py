"""DDM Embedder — Phase 3: generate Gemini text-embedding-004 vectors for each patient.

Builds a plain-English clinical narrative per patient from their indexed conditions,
medications, and observations, then calls the Google Generative AI embedding API
(text-embedding-004, 768-dim) and stores the result in patients.embedding.

Usage:
    python -m src.ddm.embedder                  # embed all patients with NULL embedding
    python -m src.ddm.embedder --batch 50       # process up to 50 patients
"""

import asyncio
import logging
import os
import sys
from datetime import date
from typing import Optional

import httpx
from sqlalchemy import select, text as sa_text
from sqlalchemy.orm import sessionmaker

from .db import get_session_factory
from .schema import Patient, PatientCondition, PatientMedication, PatientObservation

logger = logging.getLogger(__name__)

GEMINI_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "text-embedding-004:embedContent"
)
MAX_CONCURRENCY = 4   # simultaneous Gemini API calls
EMBED_DIM = 768


def _compute_age(birth_date: Optional[date]) -> Optional[int]:
    if not birth_date:
        return None
    today = date.today()
    return today.year - birth_date.year - (
        (today.month, today.day) < (birth_date.month, birth_date.day)
    )


def build_patient_narrative(
    patient: Patient,
    conditions: list,
    medications: list,
    observations: list,
) -> str:
    """Build a plain-English clinical narrative suitable for embedding.

    Compact enough to stay well under Gemini's token limit (~2048 tokens),
    rich enough to capture the clinical fingerprint of the patient.
    """
    parts = []

    # Demographics
    age = _compute_age(patient.birth_date)
    gender = patient.gender or "unknown gender"
    if age:
        parts.append(f"{age}-year-old {gender}.")
    else:
        parts.append(f"Patient, {gender}.")

    # Active conditions
    active = [c for c in conditions if c.clinical_status == "active"]
    if active:
        cond_names = [c.display for c in active if c.display][:15]
        parts.append("Conditions: " + "; ".join(cond_names) + ".")

    # Active medications
    active_meds = [m for m in medications if m.status == "active"]
    if active_meds:
        med_names = [m.display for m in active_meds if m.display][:10]
        parts.append("Medications: " + "; ".join(med_names) + ".")

    # Key observations (most recent per LOINC)
    if observations:
        seen_loinc: set[str] = set()
        obs_parts = []
        for o in sorted(observations, key=lambda x: x.observation_date or date.min, reverse=True):
            if o.loinc_code in seen_loinc:
                continue
            seen_loinc.add(o.loinc_code)
            if o.display and (o.value_quantity is not None or o.value_string):
                val = f"{o.value_quantity} {o.value_unit}" if o.value_quantity is not None else o.value_string
                obs_parts.append(f"{o.display}: {val}")
            if len(obs_parts) >= 10:
                break
        if obs_parts:
            parts.append("Labs/Vitals: " + "; ".join(obs_parts) + ".")

    return " ".join(parts)


async def embed_text(
    client: httpx.AsyncClient,
    api_key: str,
    text: str,
) -> list[float]:
    """Call Gemini text-embedding-004 and return the 768-dim vector."""
    resp = await client.post(
        GEMINI_EMBED_URL,
        params={"key": api_key},
        json={
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_DOCUMENT",
        },
        timeout=20.0,
    )
    resp.raise_for_status()
    values = resp.json()["embedding"]["values"]
    if len(values) != EMBED_DIM:
        raise ValueError(f"Expected {EMBED_DIM}-dim vector, got {len(values)}")
    return values


async def _embed_patient(
    session_factory: sessionmaker,
    client: httpx.AsyncClient,
    api_key: str,
    semaphore: asyncio.Semaphore,
    patient_id: str,
) -> None:
    """Fetch patient data, build narrative, embed, and store in patients.embedding."""
    async with semaphore:
        async with session_factory() as session:
            patient = await session.get(Patient, patient_id)
            if patient is None:
                return

            cond_result = await session.execute(
                select(PatientCondition).where(PatientCondition.patient_id == patient_id)
            )
            conditions = cond_result.scalars().all()

            med_result = await session.execute(
                select(PatientMedication).where(PatientMedication.patient_id == patient_id)
            )
            medications = med_result.scalars().all()

            obs_result = await session.execute(
                select(PatientObservation).where(PatientObservation.patient_id == patient_id)
            )
            observations = obs_result.scalars().all()

        narrative = build_patient_narrative(patient, conditions, medications, observations)
        if not narrative.strip():
            logger.debug(f"patient {patient_id}: empty narrative, skipping")
            return

        vector = await embed_text(client, api_key, narrative)

        # Store vector using raw SQL — pgvector needs the array cast
        async with session_factory() as session:
            async with session.begin():
                await session.execute(
                    sa_text(
                        "UPDATE patients SET embedding = CAST(:vec AS vector) WHERE id = :pid"
                    ),
                    {"vec": str(vector), "pid": patient_id},
                )

    logger.debug(f"patient {patient_id}: embedded ({len(narrative)} chars)")


async def run_embedder(batch_size: Optional[int] = None) -> dict:
    """Embed all patients whose embedding column is NULL.

    Returns a summary dict with counts for success, skipped, and error.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set")

    session_factory = get_session_factory()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # Fetch patients with NULL embedding
    async with session_factory() as session:
        query = sa_text(
            "SELECT id FROM patients WHERE embedding IS NULL ORDER BY indexed_at"
        )
        if batch_size:
            query = sa_text(
                "SELECT id FROM patients WHERE embedding IS NULL ORDER BY indexed_at LIMIT :lim"
            ).bindparams(lim=batch_size)
        result = await session.execute(query)
        patient_ids = [row[0] for row in result.all()]

    logger.info(f"Embedding {len(patient_ids)} patients (batch_size={batch_size})...")

    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(
                _embed_patient(session_factory, client, api_key, semaphore, pid)
            )
            for pid in patient_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        logger.warning(f"{len(errors)}/{len(patient_ids)} patients failed to embed")
        for e in errors[:3]:
            logger.warning(f"  embed error: {e}")

    summary = {
        "total": len(patient_ids),
        "success": len(patient_ids) - len(errors),
        "errors": len(errors),
    }
    logger.info(f"Embedding complete: {summary}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    _batch = None
    if "--batch" in sys.argv:
        idx = sys.argv.index("--batch")
        if idx + 1 < len(sys.argv):
            _batch = int(sys.argv[idx + 1])
    asyncio.run(run_embedder(_batch))
