"""Clinical Intelligence MCP Server — 4 core tools + 1 stretch goal."""

import json
import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP, Context

# Add parent to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.fhir_client import get_patient_data, DEFAULT_FHIR_BASE_URL
from shared.claude_client import run_ddx_reasoning, run_drug_interaction_reasoning, run_synthesis
from shared.rxnav_client import get_interactions, resolve_medications_to_rxcuis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("ClinicalIntelligence")


@mcp.tool()
async def get_patient_summary(
    patient_id: str,
    fhir_base_url: str = DEFAULT_FHIR_BASE_URL,
    access_token: str = "",
    ctx: Context = None,
) -> str:
    """Fetch comprehensive patient summary from FHIR server.

    Retrieves Patient demographics, Conditions (SNOMED), MedicationRequests (RxNorm),
    Observations (LOINC), and AllergyIntolerances for a given patient.

    Args:
        patient_id: FHIR Patient resource ID
        fhir_base_url: FHIR server base URL (default: SMART Health IT sandbox)
        access_token: SMART on FHIR bearer token (optional, from SHARP context)
    """
    if ctx:
        ctx.info(f"Fetching FHIR data for patient {patient_id}")

    token = access_token if access_token else None
    data = await get_patient_data(patient_id, fhir_base_url, token)

    if "error" in data:
        return json.dumps({"error": data["error"]}, indent=2)

    if ctx:
        ctx.info(
            f"Found: {len(data['conditions'])} conditions, "
            f"{len(data['medications'])} medications, "
            f"{len(data['observations'])} observations, "
            f"{len(data['allergies'])} allergies"
        )

    return json.dumps(data, indent=2)


@mcp.tool()
async def generate_differential_diagnosis(
    patient_id: str,
    fhir_base_url: str = DEFAULT_FHIR_BASE_URL,
    symptoms: str = "",
    access_token: str = "",
    ctx: Context = None,
) -> str:
    """Generate AI-powered differential diagnosis from FHIR patient data.

    Analyzes patient conditions, labs, medications, and optional symptoms to produce
    ranked differential diagnoses with supporting evidence and recommended workup.

    Args:
        patient_id: FHIR Patient resource ID
        fhir_base_url: FHIR server base URL
        symptoms: Optional free-text symptom description to supplement FHIR data
        access_token: SMART on FHIR bearer token (optional)
    """
    if ctx:
        ctx.info(f"Building clinical vignette for patient {patient_id}")

    token = access_token if access_token else None
    patient_data = await get_patient_data(patient_id, fhir_base_url, token)

    if "error" in patient_data:
        return json.dumps({"error": patient_data["error"]}, indent=2)

    if not patient_data["conditions"] and not symptoms:
        return json.dumps({
            "error": "No conditions found and no symptoms provided. Cannot generate differential.",
            "suggestion": "Provide symptoms parameter with clinical presentation details."
        }, indent=2)

    if ctx:
        ctx.info("Running AI differential diagnosis reasoning...")

    result = run_ddx_reasoning(patient_data, symptoms)
    return json.dumps(result, indent=2)


@mcp.tool()
async def check_drug_interactions(
    patient_id: str,
    fhir_base_url: str = DEFAULT_FHIR_BASE_URL,
    proposed_medications: str = "",
    access_token: str = "",
    ctx: Context = None,
) -> str:
    """Check drug-drug interactions for a patient's medication list with AI clinical interpretation.

    Fetches current medications from FHIR, checks interactions via RxNav database,
    then uses AI to interpret clinical significance considering the patient's specific
    conditions and allergies.

    Args:
        patient_id: FHIR Patient resource ID
        fhir_base_url: FHIR server base URL
        proposed_medications: Optional comma-separated new medications to check against existing regimen
        access_token: SMART on FHIR bearer token (optional)
    """
    if ctx:
        ctx.info(f"Checking drug interactions for patient {patient_id}")

    token = access_token if access_token else None
    patient_data = await get_patient_data(patient_id, fhir_base_url, token)

    if "error" in patient_data:
        return json.dumps({"error": patient_data["error"]}, indent=2)

    medications = patient_data.get("medications", [])
    if not medications:
        return json.dumps({
            "interactions": [],
            "note": "No medications found for this patient.",
            "overall_risk_level": "N/A"
        }, indent=2)

    # Resolve medications to RxCUIs
    if ctx:
        ctx.info(f"Resolving {len(medications)} medications to RxCUI codes...")

    enriched_meds = await resolve_medications_to_rxcuis(medications)
    rxcuis = [m["rxcui"] for m in enriched_meds if m.get("rxcui")]

    # Check RxNav interactions
    rxnav_results = None
    if len(rxcuis) >= 2:
        if ctx:
            ctx.info(f"Checking {len(rxcuis)} RxCUIs against RxNav interaction database...")
        rxnav_results = await get_interactions(rxcuis)

    # Determine if we need AI-only fallback
    has_db_interactions = (
        rxnav_results
        and rxnav_results.get("interactions")
        and len(rxnav_results["interactions"]) > 0
    )

    if not has_db_interactions and len(medications) >= 3:
        if ctx:
            ctx.info("No database interactions found for 3+ medications. Running AI-only analysis...")

    # Run Claude reasoning over interactions
    if ctx:
        ctx.info("Running AI clinical significance analysis...")

    proposed = [m.strip() for m in proposed_medications.split(",") if m.strip()] if proposed_medications else None
    result = run_drug_interaction_reasoning(patient_data, rxnav_results, proposed)
    return json.dumps(result, indent=2)


@mcp.tool()
async def synthesize_clinical_assessment(
    patient_summary_json: str,
    ddx_results_json: str,
    interaction_results_json: str,
    care_gaps_json: str = "",
    ctx: Context = None,
) -> str:
    """Synthesize differential diagnosis and drug interaction analyses into a unified clinical briefing.

    This is the crown jewel tool — it identifies cross-cutting insights that individual
    analyses miss, such as a suspected diagnosis that contraindicates a current medication,
    or a drug side effect that could explain a symptom in the differential.

    Args:
        patient_summary_json: JSON string from get_patient_summary
        ddx_results_json: JSON string from generate_differential_diagnosis
        interaction_results_json: JSON string from check_drug_interactions
        care_gaps_json: Optional JSON string from analyze_care_gaps
    """
    if ctx:
        ctx.info("Synthesizing cross-cutting clinical assessment...")

    try:
        patient_summary = json.loads(patient_summary_json)
        ddx_results = json.loads(ddx_results_json)
        interaction_results = json.loads(interaction_results_json)
        care_gaps = json.loads(care_gaps_json) if care_gaps_json else None
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON input: {str(e)}"}, indent=2)

    result = run_synthesis(patient_summary, ddx_results, interaction_results, care_gaps)

    if ctx:
        ctx.info("Clinical assessment synthesis complete.")

    return json.dumps(result, indent=2)


if __name__ == "__main__":
    port = int(os.environ.get("MCP_PORT", 8000))
    logger.info(f"Starting Clinical Intelligence MCP Server on port {port}")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
