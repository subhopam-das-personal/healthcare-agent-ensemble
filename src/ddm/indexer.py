"""DDM FHIR Indexer — paginate FHIR sources into Railway Postgres.

Usage:
    python -m src.ddm.indexer               # index all active sources
    python -m src.ddm.indexer 1             # index source_id=1
    python -m src.ddm.indexer 1 100         # stop after 100 patients (smoke test)

Requires DATABASE_URL env var (postgres:// or postgresql+asyncpg://).

Resume support: if a 'running' job with last_page_url exists for a source,
the indexer resumes from that URL rather than starting over.
"""

import asyncio
import logging
import os
import sys
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from .db import get_engine, get_session_factory
from .schema import (
    IndexJob, Patient, PatientCondition, PatientMedication,
    PatientObservation, FhirSource,
)
from .sources import fetch_patient_resources, paginate_patients

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unit normalization — applied at index time so queries never see mixed units.
# (loinc_code → (canonical_unit, converter(value, raw_unit) → float))
# ---------------------------------------------------------------------------
_UNIT_CONVERTERS: dict[str, tuple[str, object]] = {
    "2160-0": ("mg/dL", lambda v, u: v * 0.0113 if u and "umol" in u.lower() else v),
    "4548-4": ("%",     lambda v, u: v * 0.0915 if u and "mmol" in u.lower() else v),
    "2345-7": ("mg/dL", lambda v, u: v),
    "2093-3": ("mg/dL", lambda v, u: v),
    "2085-9": ("mg/dL", lambda v, u: v),
    "2089-1": ("mg/dL", lambda v, u: v),
    "8480-6": ("mmHg",  lambda v, u: v),
    "8462-4": ("mmHg",  lambda v, u: v),
    "8867-4": ("/min",  lambda v, u: v),
    "2947-0": ("mEq/L", lambda v, u: v),
    "6298-4": ("mEq/L", lambda v, u: v),
}


# ---------------------------------------------------------------------------
# FHIR resource parsers
# ---------------------------------------------------------------------------

def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s[:10]).date()
    except ValueError:
        return None


def _icd10_chapter(code: Optional[str]) -> Optional[str]:
    return code[0].upper() if code else None


def _parse_condition(resource: dict) -> Optional[dict]:
    codings = resource.get("code", {}).get("coding", [])
    icd10, snomed, display = None, None, None

    for c in codings:
        sys_lower = c.get("system", "").lower()
        code_val = c.get("code")
        disp = c.get("display")
        if "icd" in sys_lower:
            icd10 = icd10 or code_val
            display = display or disp
        elif "snomed" in sys_lower:
            snomed = snomed or code_val
            display = display or disp

    display = display or resource.get("code", {}).get("text", "Unknown")

    clinical_status = (
        resource.get("clinicalStatus", {})
        .get("coding", [{}])[0]
        .get("code", "active")
    )
    onset_raw = resource.get("onsetDateTime") or resource.get("onsetString")

    return {
        "icd10_code": icd10,
        "snomed_code": snomed,
        "display": display,
        "onset_date": _parse_date(onset_raw),
        "clinical_status": clinical_status,
        "snomed_ancestors": None,  # filled by enricher
        "icd10_chapter": _icd10_chapter(icd10),
    }


def _parse_medication(resource: dict) -> Optional[dict]:
    med = resource.get("medicationCodeableConcept", {})
    codings = med.get("coding", [])

    rxnorm, display = None, None
    for c in codings:
        if "rxnorm" in c.get("system", "").lower():
            rxnorm = rxnorm or c.get("code")
            display = display or c.get("display")
    if not rxnorm and codings:
        rxnorm = codings[0].get("code")
        display = display or codings[0].get("display")

    display = display or med.get("text", "Unknown")

    return {
        "rxnorm_code": rxnorm,
        "display": display,
        "status": resource.get("status", "active"),
        "drug_class": None,       # filled by enricher
        "drug_class_rxcui": None,
    }


def _parse_observation(resource: dict) -> Optional[dict]:
    codings = resource.get("code", {}).get("coding", [])
    loinc_code, display = None, None

    for c in codings:
        if "loinc" in c.get("system", "").lower():
            loinc_code = loinc_code or c.get("code")
            display = display or c.get("display")
    if not loinc_code and codings:
        loinc_code = codings[0].get("code")
        display = display or codings[0].get("display")

    display = display or resource.get("code", {}).get("text", "Unknown")

    vq = resource.get("valueQuantity", {})
    raw_value = vq.get("value")
    raw_unit = vq.get("unit") or vq.get("code")
    value_string = resource.get("valueString") or (
        resource.get("valueCodeableConcept", {}).get("text")
    )

    # Normalize to canonical unit
    canon_value, canon_unit = raw_value, raw_unit
    if loinc_code and loinc_code in _UNIT_CONVERTERS and raw_value is not None:
        target_unit, converter = _UNIT_CONVERTERS[loinc_code]
        try:
            canon_value = converter(float(raw_value), raw_unit)
            canon_unit = target_unit
        except (TypeError, ValueError):
            pass

    obs_date = resource.get("effectiveDateTime") or (
        resource.get("effectivePeriod", {}).get("start")
    )

    return {
        "loinc_code": loinc_code,
        "display": display,
        "value_quantity": float(canon_value) if canon_value is not None else None,
        "value_unit": canon_unit,
        "value_string": value_string,
        "observation_date": _parse_date(obs_date),
    }


# ---------------------------------------------------------------------------
# DB write helpers
# ---------------------------------------------------------------------------

async def _upsert_patient(
    session: AsyncSession,
    patient_resource: dict,
    source_id: int,
    resources: dict,
) -> bool:
    """Write patient + conditions/medications/observations. Returns True on success."""
    patient_id = patient_resource.get("id")
    if not patient_id:
        return False

    now = datetime.now(timezone.utc)

    name_parts = (patient_resource.get("name") or [{}])[0]
    given = " ".join(name_parts.get("given", [])) or None
    family = name_parts.get("family") or None

    stmt = pg_insert(Patient).values(
        id=patient_id,
        source_id=source_id,
        given_name=given,
        family_name=family,
        birth_date=_parse_date(patient_resource.get("birthDate")),
        gender=patient_resource.get("gender"),
        raw_fhir=patient_resource,
        indexed_at=now,
    ).on_conflict_do_update(
        index_elements=["id"],
        set_={
            "source_id": source_id,
            "given_name": given,
            "family_name": family,
            "birth_date": _parse_date(patient_resource.get("birthDate")),
            "gender": patient_resource.get("gender"),
            "raw_fhir": patient_resource,
            "indexed_at": now,
        },
    )
    await session.execute(stmt)

    # Replace child rows entirely on re-index
    for table in ("patient_conditions", "patient_medications", "patient_observations"):
        await session.execute(
            text(f"DELETE FROM {table} WHERE patient_id = :pid"),
            {"pid": patient_id},
        )

    for cond in resources.get("conditions", []):
        data = _parse_condition(cond)
        if data:
            session.add(PatientCondition(patient_id=patient_id, indexed_at=now, **data))

    for med in resources.get("medications", []):
        data = _parse_medication(med)
        if data:
            session.add(PatientMedication(patient_id=patient_id, indexed_at=now, **data))

    for obs in resources.get("observations", [])[:50]:
        data = _parse_observation(obs)
        if data:
            session.add(PatientObservation(patient_id=patient_id, indexed_at=now, **data))

    return True


# ---------------------------------------------------------------------------
# Page processor
# ---------------------------------------------------------------------------

async def _process_page(
    session_factory: sessionmaker,
    source: FhirSource,
    patients: list[dict],
    semaphore: asyncio.Semaphore,
) -> int:
    """Fetch details for a page of patients concurrently, then write in one transaction."""

    async def _fetch_one(patient_resource: dict) -> Optional[tuple[dict, dict]]:
        pid = patient_resource.get("id")
        if not pid:
            return None
        try:
            async with semaphore:
                resources = await fetch_patient_resources(source, pid)
            return patient_resource, resources
        except Exception as e:
            logger.warning(f"Skipping patient {pid}: {e}")
            return None

    results = await asyncio.gather(*[_fetch_one(p) for p in patients])
    valid = [r for r in results if r is not None]

    indexed = 0
    async with session_factory() as session:
        async with session.begin():
            for patient_resource, resources in valid:
                ok = await _upsert_patient(session, patient_resource, source.id, resources)
                if ok:
                    indexed += 1

    return indexed


# ---------------------------------------------------------------------------
# Source indexer
# ---------------------------------------------------------------------------

async def _index_source(
    engine,  # unused; kept for call-site compatibility
    source: FhirSource,
    max_patients: Optional[int] = None,
) -> None:
    session_factory = get_session_factory()
    now = datetime.now(timezone.utc)
    semaphore = asyncio.Semaphore(5)

    # Create or resume job
    async with session_factory() as session:
        async with session.begin():
            result = await session.execute(
                select(IndexJob)
                .where(IndexJob.source_id == source.id)
                .where(IndexJob.status == "running")
                .order_by(IndexJob.started_at.desc())
                .limit(1)
            )
            job = result.scalar_one_or_none()

            if job and job.last_page_url:
                resume_url: Optional[str] = job.last_page_url
                logger.info(f"Resuming job {job.id} for {source.name!r} from {resume_url}")
            else:
                job = IndexJob(
                    source_id=source.id,
                    status="running",
                    patients_fetched=0,
                    patients_indexed=0,
                    started_at=now,
                )
                session.add(job)
                await session.flush()
                resume_url = None
                logger.info(f"Starting new index job {job.id} for {source.name!r}")

            job_id = job.id
            total_fetched = job.patients_fetched
            total_indexed = job.patients_indexed

    try:
        async for page_url, patients, next_url in paginate_patients(source, resume_url):
            logger.info(
                f"[job {job_id}] page {page_url!r}: {len(patients)} patients, "
                f"next={next_url is not None}"
            )

            page_indexed = await _process_page(
                session_factory, source, patients, semaphore
            )
            total_fetched += len(patients)
            total_indexed += page_indexed

            # Checkpoint job state
            async with session_factory() as session:
                async with session.begin():
                    result = await session.execute(
                        select(IndexJob).where(IndexJob.id == job_id)
                    )
                    job = result.scalar_one()
                    job.patients_fetched = total_fetched
                    job.patients_indexed = total_indexed
                    job.last_page_url = next_url  # None on last page

            if max_patients and total_indexed >= max_patients:
                logger.info(f"Reached max_patients={max_patients}, stopping early.")
                break

        # Mark done
        async with session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    select(IndexJob).where(IndexJob.id == job_id)
                )
                job = result.scalar_one()
                job.status = "done"
                job.finished_at = datetime.now(timezone.utc)

        logger.info(
            f"[job {job_id}] Done. Fetched={total_fetched}, Indexed={total_indexed}"
        )

    except Exception as exc:
        async with session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    select(IndexJob).where(IndexJob.id == job_id)
                )
                job = result.scalar_one()
                job.status = "error"
                job.error_message = str(exc)
                job.finished_at = datetime.now(timezone.utc)

        logger.error(f"[job {job_id}] Failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_indexer(
    source_id: Optional[int] = None,
    max_patients: Optional[int] = None,
) -> None:
    """Index all active FHIR sources (or a specific one by ID)."""
    engine = get_engine()
    session_factory = get_session_factory()

    async with session_factory() as session:
        query = select(FhirSource).where(FhirSource.active.is_(True))
        if source_id is not None:
            query = query.where(FhirSource.id == source_id)
        result = await session.execute(query)
        sources = result.scalars().all()

    if not sources:
        logger.warning("No active FHIR sources found. Did you run the migration?")
        await engine.dispose()
        return

    for source in sources:
        await _index_source(engine, source, max_patients)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    _source_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
    _max_p = int(sys.argv[2]) if len(sys.argv) > 2 else None
    asyncio.run(run_indexer(_source_id, _max_p))
