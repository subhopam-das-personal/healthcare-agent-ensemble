"""DDM Hybrid Query Router — Phase 3.

Pipeline:
  1. Claude Haiku extracts medical entities from the NL question
  2. DB ontology lookups expand entities to ICD-10/SNOMED codes and drug classes
  3. Claude generates a validated SQL query → execute against Postgres
  4. If SQL returns 0 rows or fails: text-similarity fallback (ILIKE across displays)

The expansion dict is returned alongside results so the UI can show the
ontological expansion panel ("your query matched these 6 drug classes").
"""

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Allow running standalone (python -m src.ddm.query_engine)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.ddm.db import get_session_factory

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# Schema context fed to Claude for SQL generation
# ---------------------------------------------------------------------------

_SCHEMA = """
Tables (Railway Postgres):

patients(id TEXT PK, given_name TEXT, family_name TEXT, birth_date DATE, gender TEXT)

patient_conditions(id SERIAL PK, patient_id TEXT FK→patients.id,
    icd10_code TEXT, snomed_code TEXT, display TEXT,
    clinical_status TEXT, snomed_ancestors TEXT[], icd10_chapter TEXT)

patient_medications(id SERIAL PK, patient_id TEXT FK→patients.id,
    rxnorm_code TEXT, display TEXT, status TEXT,
    drug_class TEXT, drug_class_rxcui TEXT)

patient_observations(id SERIAL PK, patient_id TEXT FK→patients.id,
    loinc_code TEXT, display TEXT,
    value_quantity FLOAT, value_unit TEXT, observation_date DATE)
"""

# Columns allowed in WHERE / SELECT — guards against hallucinated column names
_VALID_COLS: dict[str, set[str]] = {
    "patients": {"id", "given_name", "family_name", "birth_date", "gender"},
    "patient_conditions": {
        "patient_id", "icd10_code", "snomed_code", "display",
        "clinical_status", "snomed_ancestors", "icd10_chapter",
    },
    "patient_medications": {
        "patient_id", "rxnorm_code", "display", "status",
        "drug_class", "drug_class_rxcui",
    },
    "patient_observations": {
        "patient_id", "loinc_code", "display",
        "value_quantity", "value_unit", "observation_date",
    },
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    patients: list[dict] = field(default_factory=list)
    mode: str = "none"           # "structured" | "vector" | "text_fallback" | "empty"
    sql: Optional[str] = None    # SQL that ran (shown in transparency panel)
    expansion: dict = field(default_factory=dict)
    count: int = 0


# ---------------------------------------------------------------------------
# Entity extraction (Claude Haiku)
# ---------------------------------------------------------------------------

_NER_PROMPT = """\
Extract medical entities from the clinical query below.
Return ONLY a JSON object with these keys (arrays of strings, empty if none found):
{
  "conditions": [],   // e.g. "heart failure", "type 2 diabetes"
  "drugs": [],        // e.g. "metformin", "ACE inhibitor", "anticoagulant"
  "labs": [],         // e.g. "creatinine", "HbA1c", "potassium"
  "organs": []        // e.g. "kidney", "liver", "heart"
}

Query: {question}"""


async def _extract_entities(question: str) -> dict:
    """Use Claude Haiku to extract structured medical entities from the NL query."""
    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": _NER_PROMPT.format(question=question)}],
        )
        raw = resp.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"NER extraction failed: {e}")
        return {"conditions": [], "drugs": [], "labs": [], "organs": []}


# ---------------------------------------------------------------------------
# Ontology expansion (DB lookups)
# ---------------------------------------------------------------------------

async def _expand_entities(entities: dict, session: AsyncSession) -> dict:
    """Expand NL entities to structured codes via the DB ontology/drug class tables."""
    expansion: dict[str, list] = {
        "icd10_codes": [],
        "snomed_codes": [],
        "drug_classes": [],
        "loinc_codes": [],
        "condition_displays": entities.get("conditions", []),
        "drug_displays": entities.get("drugs", []),
    }

    conditions = entities.get("conditions", [])
    drugs = entities.get("drugs", [])
    labs = entities.get("labs", [])

    # Expand conditions → ICD-10 + SNOMED via patient_conditions display matches
    if conditions:
        placeholders = " OR ".join(
            f"display ILIKE :cond{i}" for i in range(len(conditions))
        )
        params = {f"cond{i}": f"%{c}%" for i, c in enumerate(conditions)}
        rows = await session.execute(
            text(f"SELECT DISTINCT icd10_code, snomed_code FROM patient_conditions WHERE {placeholders}"),
            params,
        )
        for row in rows:
            if row.icd10_code and row.icd10_code not in expansion["icd10_codes"]:
                expansion["icd10_codes"].append(row.icd10_code)
            if row.snomed_code and row.snomed_code not in expansion["snomed_codes"]:
                expansion["snomed_codes"].append(row.snomed_code)

    # Expand drugs → drug_class via patient_medications + drug_class_map
    if drugs:
        placeholders = " OR ".join(
            f"display ILIKE :drug{i}" for i in range(len(drugs))
        )
        params = {f"drug{i}": f"%{d}%" for i, d in enumerate(drugs)}
        rows = await session.execute(
            text(
                f"SELECT DISTINCT drug_class FROM patient_medications "
                f"WHERE drug_class IS NOT NULL AND ({placeholders})"
            ),
            params,
        )
        for row in rows:
            if row.drug_class and row.drug_class not in expansion["drug_classes"]:
                expansion["drug_classes"].append(row.drug_class)

        # Also check drug_class_map for drug class names matching drug terms
        placeholders2 = " OR ".join(
            f"drug_class ILIKE :dc{i}" for i in range(len(drugs))
        )
        params2 = {f"dc{i}": f"%{d}%" for i, d in enumerate(drugs)}
        rows2 = await session.execute(
            text(f"SELECT DISTINCT drug_class FROM drug_class_map WHERE {placeholders2}"),
            params2,
        )
        for row in rows2:
            if row.drug_class and row.drug_class not in expansion["drug_classes"]:
                expansion["drug_classes"].append(row.drug_class)

    # Expand labs → LOINC codes via patient_observations display matches
    if labs:
        placeholders = " OR ".join(
            f"display ILIKE :lab{i}" for i in range(len(labs))
        )
        params = {f"lab{i}": f"%{l}%" for i, l in enumerate(labs)}
        rows = await session.execute(
            text(
                f"SELECT DISTINCT loinc_code FROM patient_observations "
                f"WHERE loinc_code IS NOT NULL AND ({placeholders})"
            ),
            params,
        )
        for row in rows:
            if row.loinc_code and row.loinc_code not in expansion["loinc_codes"]:
                expansion["loinc_codes"].append(row.loinc_code)

    return expansion


# ---------------------------------------------------------------------------
# SQL generation + validation (Claude Sonnet)
# ---------------------------------------------------------------------------

_SQL_PROMPT = """\
You are a SQL expert for a clinical database. Write a single read-only SELECT query.

{schema}

Expanded entities from the user's question:
{expansion_json}

User question: {question}

Rules:
- Return ONLY the SQL query, no explanation, no markdown fences.
- SELECT only from: patients, patient_conditions, patient_medications, patient_observations
- Always return patient id, given_name, family_name, birth_date, gender
- Use DISTINCT to avoid duplicate patients
- Use ILIKE for text matching (case-insensitive)
- Use ANY(array_col) for snomed_ancestors array lookups
- Limit to 50 results
- No INSERT, UPDATE, DELETE, DROP, or CTEs referencing external tables
- If no structured codes are available, use ILIKE on display columns"""


async def _generate_sql(question: str, expansion: dict) -> Optional[str]:
    """Ask Claude to generate a SQL query for the expanded entities."""
    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        # Only include non-empty expansion fields in the prompt
        compact_expansion = {k: v for k, v in expansion.items() if v}

        resp = await client.messages.create(
            model=SONNET_MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": _SQL_PROMPT.format(
                    schema=_SCHEMA,
                    expansion_json=json.dumps(compact_expansion, indent=2),
                    question=question,
                ),
            }],
        )
        sql = resp.content[0].text.strip()
        # Strip markdown fences
        sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.MULTILINE).strip()
        return sql
    except Exception as e:
        logger.warning(f"SQL generation failed: {e}")
        return None


def _validate_sql(sql: str) -> bool:
    """Basic safety checks: SELECT only, no dangerous keywords, known columns only."""
    upper = sql.upper().strip()

    # Must be a SELECT statement
    if not upper.startswith("SELECT"):
        return False

    # Block dangerous keywords
    for kw in ("INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE", "GRANT"):
        if re.search(r"\b" + kw + r"\b", upper):
            return False

    return True


async def _try_sql_path(
    sql: str, session: AsyncSession
) -> Optional[list[dict]]:
    """EXPLAIN the SQL (syntax check), then execute it. Returns None on any failure."""
    if not _validate_sql(sql):
        logger.warning("SQL failed safety validation")
        return None
    try:
        await session.execute(text(f"EXPLAIN {sql}"))
    except Exception as e:
        logger.warning(f"SQL EXPLAIN failed: {e}")
        return None
    try:
        result = await session.execute(text(sql))
        rows = result.mappings().all()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"SQL execution failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Vector similarity search (Gemini embedding → pgvector HNSW)
# ---------------------------------------------------------------------------

async def _embed_question(question: str) -> Optional[list[float]]:
    """Embed the NL question via Voyage AI voyage-3 (768-dim).

    Returns None if VOYAGE_API_KEY is not set or the API call fails.
    """
    if not os.environ.get("VOYAGE_EMBEDDING_API_KEY"):
        return None
    try:
        from ddm.embedder import embed_texts
        vectors = await embed_texts([question], input_type="query")
        return vectors[0]
    except Exception as e:
        logger.warning(f"Question embedding failed: {e}")
        return None


async def _vector_search(
    vector: list[float], session: AsyncSession, limit: int = 20
) -> list[dict]:
    """Search patients by cosine similarity to the query vector.

    Skips gracefully if no patients have embeddings yet.
    """
    # Check whether any embeddings exist first (avoids full seqscan on empty column)
    check = await session.execute(
        text("SELECT 1 FROM patients WHERE embedding IS NOT NULL LIMIT 1")
    )
    if not check.fetchone():
        logger.info("No patient embeddings found — skipping vector search")
        return []

    try:
        result = await session.execute(
            text("""
                SELECT id, given_name, family_name, birth_date, gender,
                       1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                FROM patients
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :lim
            """),
            {"vec": str(vector), "lim": limit},
        )
        rows = result.mappings().all()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Text fallback (ILIKE across condition/medication displays)
# ---------------------------------------------------------------------------

async def _text_fallback(
    entities: dict, session: AsyncSession
) -> list[dict]:
    """Simple ILIKE search when SQL path fails. Returns up to 20 patients."""
    all_terms = (
        entities.get("conditions", [])
        + entities.get("drugs", [])
        + entities.get("labs", [])
        + entities.get("organs", [])
    )
    if not all_terms:
        return []

    cond_clauses = " OR ".join(f"pc.display ILIKE :t{i}" for i in range(len(all_terms)))
    med_clauses = " OR ".join(f"pm.display ILIKE :t{i}" for i in range(len(all_terms)))
    params = {f"t{i}": f"%{t}%" for i, t in enumerate(all_terms)}

    sql = f"""
        SELECT DISTINCT p.id, p.given_name, p.family_name, p.birth_date, p.gender
        FROM patients p
        LEFT JOIN patient_conditions pc ON pc.patient_id = p.id
        LEFT JOIN patient_medications pm ON pm.patient_id = p.id
        WHERE ({cond_clauses}) OR ({med_clauses})
        LIMIT 20
    """
    try:
        result = await session.execute(text(sql), params)
        return [dict(r) for r in result.mappings().all()]
    except Exception as e:
        logger.warning(f"Text fallback failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Row → patient dict (serialisable)
# ---------------------------------------------------------------------------

def _serialize_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def query_patients(question: str) -> QueryResult:
    """Run the full hybrid query pipeline for a natural language clinical question."""
    session_factory = get_session_factory()

    # Step 1 — entity extraction
    entities = await _extract_entities(question)
    logger.info(f"Entities extracted: {entities}")

    async with session_factory() as session:
        # Step 2 — ontology expansion
        expansion = await _expand_entities(entities, session)
        logger.info(
            f"Expansion: {len(expansion['icd10_codes'])} ICD10, "
            f"{len(expansion['drug_classes'])} drug classes, "
            f"{len(expansion['loinc_codes'])} LOINC"
        )

        # Step 3 — SQL path
        sql = await _generate_sql(question, expansion)
        patients = None
        if sql:
            patients = await _try_sql_path(sql, session)

        if patients:
            serialised = [_serialize_row(p) for p in patients]
            return QueryResult(
                patients=serialised,
                mode="structured",
                sql=sql,
                expansion=expansion,
                count=len(serialised),
            )

        # Step 4 — vector similarity search
        logger.info("SQL path returned no results; trying vector similarity search")
        query_vector = await _embed_question(question)
        if query_vector:
            patients = await _vector_search(query_vector, session)
            if patients:
                serialised = [_serialize_row(p) for p in patients]
                return QueryResult(
                    patients=serialised,
                    mode="vector",
                    sql=None,
                    expansion=expansion,
                    count=len(serialised),
                )

        # Step 5 — text fallback
        logger.info("Vector search returned no results; falling back to text search")
        patients = await _text_fallback(entities, session)
        serialised = [_serialize_row(p) for p in patients]
        return QueryResult(
            patients=serialised,
            mode="text_fallback",
            sql=None,
            expansion=expansion,
            count=len(serialised),
        )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    q = sys.argv[1] if len(sys.argv) > 1 else "patients with diabetes on metformin"
    result = asyncio.run(query_patients(q))
    print(f"\nMode: {result.mode}  Count: {result.count}")
    print(f"Expansion: {result.expansion}")
    if result.sql:
        print(f"SQL:\n{result.sql}")
    for p in result.patients[:5]:
        print(p)
