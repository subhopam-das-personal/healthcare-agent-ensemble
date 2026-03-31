"""DDM Embedder — Phase 3: generate Voyage AI voyage-3 vectors for each patient.

Builds a plain-English clinical narrative per patient from their indexed conditions,
medications, and observations, then calls the Voyage AI embedding API
(voyage-3, output_dimension=768) and stores the result in patients.embedding.

Voyage AI supports batching — 200 patients run in 2 API calls, not 200.

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

VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_MODEL = "voyage-3.5-lite"
EMBED_DIM = 512          # matches patients.embedding vector(512)
BATCH_SIZE = 100         # Voyage supports up to 128 inputs per request


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
    """Build a plain-English clinical narrative suitable for embedding."""
    parts = []

    age = _compute_age(patient.birth_date)
    gender = patient.gender or "unknown gender"
    parts.append(f"{age}-year-old {gender}." if age else f"Patient, {gender}.")

    active_conds = [c for c in conditions if c.clinical_status == "active"]
    if active_conds:
        names = [c.display for c in active_conds if c.display][:15]
        parts.append("Conditions: " + "; ".join(names) + ".")

    active_meds = [m for m in medications if m.status == "active"]
    if active_meds:
        names = [m.display for m in active_meds if m.display][:10]
        parts.append("Medications: " + "; ".join(names) + ".")

    if observations:
        seen: set[str] = set()
        obs_parts = []
        for o in sorted(observations, key=lambda x: x.observation_date or date.min, reverse=True):
            if o.loinc_code in seen:
                continue
            seen.add(o.loinc_code)
            if o.display and (o.value_quantity is not None or o.value_string):
                val = f"{o.value_quantity} {o.value_unit}" if o.value_quantity is not None else o.value_string
                obs_parts.append(f"{o.display}: {val}")
            if len(obs_parts) >= 10:
                break
        if obs_parts:
            parts.append("Labs/Vitals: " + "; ".join(obs_parts) + ".")

    return " ".join(parts)


async def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Call Voyage AI and return a list of 768-dim vectors (one per input text).

    Args:
        texts: List of strings to embed (max 128 per call).
        input_type: "document" for indexing, "query" for retrieval queries.
    """
    api_key = os.environ.get("VOYAGE_EMBEDDING_API_KEY", "")
    if not api_key:
        raise RuntimeError("VOYAGE_EMBEDDING_API_KEY environment variable is not set")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            VOYAGE_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "input": texts,
                "model": VOYAGE_MODEL,
                "input_type": input_type,
                # voyage-3.5-lite native dim is 512 — no truncation needed
                # "output_dimension" omitted to use model default
            },
        )
        resp.raise_for_status()
        data = resp.json()

    # Voyage returns {"data": [{"embedding": [...], "index": 0}, ...]}
    items = sorted(data["data"], key=lambda x: x["index"])
    vectors = [item["embedding"] for item in items]

    for i, v in enumerate(vectors):
        if len(v) != EMBED_DIM:
            raise ValueError(f"Expected {EMBED_DIM}-dim vector at index {i}, got {len(v)}")

    return vectors


async def run_embedder(batch_size: Optional[int] = None) -> dict:
    """Embed all patients whose embedding column is NULL.

    Fetches patient narratives in batches of BATCH_SIZE, calls Voyage AI once
    per batch, and writes results to patients.embedding.

    Returns a summary dict with counts for success, skipped, and error.
    """
    api_key = os.environ.get("VOYAGE_EMBEDDING_API_KEY", "")
    if not api_key:
        raise RuntimeError("VOYAGE_EMBEDDING_API_KEY environment variable is not set")

    session_factory = get_session_factory()

    # Fetch patients with NULL embedding
    async with session_factory() as session:
        sql = "SELECT id FROM patients WHERE embedding IS NULL ORDER BY indexed_at"
        if batch_size:
            sql += f" LIMIT {batch_size}"
        result = await session.execute(sa_text(sql))
        patient_ids = [row[0] for row in result.all()]

    if not patient_ids:
        logger.info("No patients need embedding — all done.")
        return {"total": 0, "success": 0, "errors": 0}

    logger.info(f"Building narratives for {len(patient_ids)} patients...")

    # Build all narratives first
    narratives: dict[str, str] = {}
    async with session_factory() as session:
        for pid in patient_ids:
            patient = await session.get(Patient, pid)
            if patient is None:
                continue

            cond_r = await session.execute(
                select(PatientCondition).where(PatientCondition.patient_id == pid)
            )
            med_r = await session.execute(
                select(PatientMedication).where(PatientMedication.patient_id == pid)
            )
            obs_r = await session.execute(
                select(PatientObservation).where(PatientObservation.patient_id == pid)
            )
            narrative = build_patient_narrative(
                patient,
                cond_r.scalars().all(),
                med_r.scalars().all(),
                obs_r.scalars().all(),
            )
            if narrative.strip():
                narratives[pid] = narrative

    ids_to_embed = list(narratives.keys())
    texts_to_embed = [narratives[pid] for pid in ids_to_embed]
    logger.info(f"Embedding {len(ids_to_embed)} patients in batches of {BATCH_SIZE}...")

    success = 0
    errors = 0

    # Process in batches
    for i in range(0, len(ids_to_embed), BATCH_SIZE):
        batch_ids = ids_to_embed[i : i + BATCH_SIZE]
        batch_texts = texts_to_embed[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(ids_to_embed) + BATCH_SIZE - 1) // BATCH_SIZE

        try:
            logger.info(f"Batch {batch_num}/{total_batches}: embedding {len(batch_ids)} patients...")
            vectors = await embed_texts(batch_texts, input_type="document")

            # Write all vectors in this batch to DB
            async with session_factory() as session:
                async with session.begin():
                    for pid, vector in zip(batch_ids, vectors):
                        await session.execute(
                            sa_text(
                                "UPDATE patients SET embedding = CAST(:vec AS vector) WHERE id = :pid"
                            ),
                            {"vec": str(vector), "pid": pid},
                        )
            success += len(batch_ids)
            logger.info(f"Batch {batch_num}/{total_batches}: wrote {len(batch_ids)} embeddings.")

        except Exception as e:
            logger.warning(f"Batch {batch_num}/{total_batches} failed: {e}")
            errors += len(batch_ids)

    summary = {"total": len(patient_ids), "success": success, "errors": errors}
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
