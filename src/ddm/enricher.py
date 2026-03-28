"""DDM Enricher — Phase 2 post-index enrichment.

For each patient already in the DB, fills in:
  - patient_conditions.snomed_ancestors via NLM tx.fhir.org (cached in ontology_cache)
  - patient_medications.drug_class / drug_class_rxcui via RxNav (cached in drug_class_map)

Unit normalization happens at index time in indexer.py; the enricher handles
ontological and pharmacological lookups that are too slow to do inline.

Usage:
    python -m src.ddm.enricher                  # enrich everything unenriched
    python -m src.ddm.enricher --batch 50       # process up to 50 conditions/meds
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .db import get_engine, get_session_factory
from .schema import (
    DrugClassMap, OntologyCache, PatientCondition, PatientMedication,
)

logger = logging.getLogger(__name__)

NLM_FHIR_BASE = "https://tx.fhir.org/r4"
RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
HTTP_TIMEOUT = 10.0
MAX_SNOMED_DEPTH = 3   # BFS levels up the SNOMED hierarchy
MAX_CONCURRENCY = 4    # simultaneous external API calls


# ---------------------------------------------------------------------------
# SNOMED ancestor lookup via NLM tx.fhir.org
# ---------------------------------------------------------------------------

async def _nlm_lookup_parents(
    client: httpx.AsyncClient, snomed_code: str
) -> list[dict]:
    """Call tx.fhir.org $lookup and extract immediate parent codes + displays."""
    try:
        resp = await client.get(
            f"{NLM_FHIR_BASE}/CodeSystem/$lookup",
            params={
                "system": "http://snomed.info/sct",
                "code": snomed_code,
                "property": "parent",
            },
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug(f"SNOMED lookup failed for {snomed_code}: {e}")
        return []

    parents = []
    for param in data.get("parameter", []):
        if param.get("name") == "property":
            parts = {p["name"]: p for p in param.get("part", [])}
            if parts.get("code", {}).get("valueCode") == "parent":
                parent_code = parts.get("value", {}).get("valueCode")
                parent_display = parts.get("valueDisplay", {}).get("valueString", "")
                if parent_code:
                    parents.append({"code": parent_code, "display": parent_display})
    return parents


async def fetch_snomed_ancestors(
    client: httpx.AsyncClient, snomed_code: str
) -> list[dict]:
    """BFS up the SNOMED hierarchy, up to MAX_SNOMED_DEPTH levels.

    Returns [{code, display, depth}, ...] sorted by depth ascending.
    """
    visited: dict[str, dict] = {}  # code → {code, display, depth}
    queue = [(snomed_code, 0)]

    while queue:
        code, depth = queue.pop(0)
        if depth >= MAX_SNOMED_DEPTH:
            continue
        if code in visited:
            continue

        parents = await _nlm_lookup_parents(client, code)
        for p in parents:
            if p["code"] not in visited:
                entry = {"code": p["code"], "display": p["display"], "depth": depth + 1}
                visited[p["code"]] = entry
                queue.append((p["code"], depth + 1))

    return sorted(visited.values(), key=lambda x: x["depth"])


async def _enrich_condition(
    session_factory: sessionmaker,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    condition_id: int,
    snomed_code: str,
) -> None:
    """Look up SNOMED ancestors, cache, and update patient_conditions row."""
    async with semaphore:
        # Check ontology_cache first
        async with session_factory() as session:
            result = await session.execute(
                select(OntologyCache).where(OntologyCache.snomed_code == snomed_code)
            )
            cached = result.scalar_one_or_none()

        if cached:
            ancestors = cached.ancestors or []
        else:
            ancestors = await fetch_snomed_ancestors(client, snomed_code)
            # Store in cache
            now = datetime.now(timezone.utc)
            async with session_factory() as session:
                async with session.begin():
                    stmt = pg_insert(OntologyCache).values(
                        snomed_code=snomed_code,
                        ancestors=ancestors,
                        fetched_at=now,
                    ).on_conflict_do_update(
                        index_elements=["snomed_code"],
                        set_={"ancestors": ancestors, "fetched_at": now},
                    )
                    await session.execute(stmt)

        # Update the condition row
        ancestor_codes = [a["code"] for a in ancestors] if ancestors else None
        async with session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(PatientCondition)
                    .where(PatientCondition.id == condition_id)
                    .values(snomed_ancestors=ancestor_codes)
                )

    logger.debug(f"condition {condition_id}: {len(ancestors)} ancestors for SNOMED {snomed_code}")


# ---------------------------------------------------------------------------
# Drug class lookup via RxNav rxclass API
# ---------------------------------------------------------------------------

async def fetch_drug_class(
    client: httpx.AsyncClient, rxnorm_code: str
) -> tuple[Optional[str], Optional[str]]:
    """Return (drug_class, drug_class_rxcui) for an RxNorm code via RxNav.

    Tries ATC classes first (broadest coverage), falls back to NDFRT VA classes.
    Returns (None, None) if no class found.
    """
    for rela_source in ("ATC", "NDFRT", "MESH"):
        try:
            resp = await client.get(
                f"{RXNAV_BASE}/rxclass/class/byRxcui.json",
                params={"rxcui": rxnorm_code, "relaSource": rela_source},
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.debug(f"RxNav drug class lookup failed for {rxnorm_code} ({rela_source}): {e}")
            continue

        drug_info_list = (
            data.get("rxclassDrugInfoList", {})
            .get("rxclassDrugInfo", [])
        )
        if drug_info_list:
            concept = drug_info_list[0].get("rxclassMinConceptItem", {})
            return concept.get("className"), concept.get("classId")

    return None, None


async def _enrich_medication(
    session_factory: sessionmaker,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    med_id: int,
    rxnorm_code: str,
) -> None:
    """Look up drug class, cache, and update patient_medications row."""
    async with semaphore:
        # Check drug_class_map cache first
        async with session_factory() as session:
            result = await session.execute(
                select(DrugClassMap).where(DrugClassMap.rxnorm_code == rxnorm_code)
            )
            cached = result.scalar_one_or_none()

        if cached:
            drug_class, drug_class_rxcui = cached.drug_class, cached.drug_class_rxcui
        else:
            drug_class, drug_class_rxcui = await fetch_drug_class(client, rxnorm_code)
            now = datetime.now(timezone.utc)
            async with session_factory() as session:
                async with session.begin():
                    stmt = pg_insert(DrugClassMap).values(
                        rxnorm_code=rxnorm_code,
                        drug_class=drug_class,
                        drug_class_rxcui=drug_class_rxcui,
                        updated_at=now,
                    ).on_conflict_do_update(
                        index_elements=["rxnorm_code"],
                        set_={
                            "drug_class": drug_class,
                            "drug_class_rxcui": drug_class_rxcui,
                            "updated_at": now,
                        },
                    )
                    await session.execute(stmt)

        async with session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(PatientMedication)
                    .where(PatientMedication.id == med_id)
                    .values(drug_class=drug_class, drug_class_rxcui=drug_class_rxcui)
                )

    logger.debug(f"medication {med_id}: drug_class={drug_class!r} for RxNorm {rxnorm_code}")


# ---------------------------------------------------------------------------
# Main enricher
# ---------------------------------------------------------------------------

async def run_enricher(batch_size: Optional[int] = None) -> None:
    """Enrich all unenriched conditions and medications in the DB."""
    session_factory = get_session_factory()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        # --- Conditions: fetch those with a SNOMED code but no ancestors yet ---
        async with session_factory() as session:
            query = (
                select(PatientCondition.id, PatientCondition.snomed_code)
                .where(PatientCondition.snomed_code.isnot(None))
                .where(PatientCondition.snomed_ancestors.is_(None))
            )
            if batch_size:
                query = query.limit(batch_size)
            result = await session.execute(query)
            conditions = result.all()

        logger.info(f"Enriching {len(conditions)} conditions (SNOMED ancestors)...")
        cond_tasks = [
            asyncio.create_task(
                _enrich_condition(session_factory, client, semaphore, row.id, row.snomed_code)
            )
            for row in conditions
        ]
        results = await asyncio.gather(*cond_tasks, return_exceptions=True)
        cond_errors = sum(1 for r in results if isinstance(r, Exception))
        if cond_errors:
            logger.warning(f"{cond_errors}/{len(conditions)} condition enrichments failed")

        # --- Medications: fetch those with an RxNorm code but no drug class ---
        async with session_factory() as session:
            query = (
                select(PatientMedication.id, PatientMedication.rxnorm_code)
                .where(PatientMedication.rxnorm_code.isnot(None))
                .where(PatientMedication.drug_class.is_(None))
            )
            if batch_size:
                query = query.limit(batch_size)
            result = await session.execute(query)
            medications = result.all()

        logger.info(f"Enriching {len(medications)} medications (drug class)...")
        med_tasks = [
            asyncio.create_task(
                _enrich_medication(session_factory, client, semaphore, row.id, row.rxnorm_code)
            )
            for row in medications
        ]
        results = await asyncio.gather(*med_tasks, return_exceptions=True)
        med_errors = sum(1 for r in results if isinstance(r, Exception))
        if med_errors:
            logger.warning(f"{med_errors}/{len(medications)} medication enrichments failed")

    logger.info(
        f"Enrichment complete. Conditions={len(conditions) - cond_errors}, "
        f"Medications={len(medications) - med_errors}"
    )


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
    asyncio.run(run_enricher(_batch))
