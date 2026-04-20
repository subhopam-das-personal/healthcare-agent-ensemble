"""Z.AI API client for clinical reasoning."""

import os
import json
import logging
from collections.abc import AsyncGenerator
from zai import ZaiClient, ZaiAsyncClient

logger = logging.getLogger(__name__)

ZAI_MODEL = "glm-4.7"
ZAI_TIMEOUT = 60.0


def get_client() -> ZaiClient:
    """Get a synchronous ZaiClient instance."""
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        raise ValueError("ZAI_API_KEY environment variable is required")
    base_url = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/")
    return ZaiClient(api_key=api_key, base_url=base_url, timeout=ZAI_TIMEOUT)


def get_async_client() -> ZaiAsyncClient:
    """Get an asynchronous ZaiAsyncClient instance."""
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        raise ValueError("ZAI_API_KEY environment variable is required")
    base_url = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/")
    return ZaiAsyncClient(api_key=api_key, base_url=base_url, timeout=ZAI_TIMEOUT)


DDX_SYSTEM_PROMPT = """You are a clinical decision support assistant specializing in differential diagnosis.

Given patient data from FHIR resources (conditions, medications, labs, allergies) and optional symptom descriptions, generate a ranked differential diagnosis.

Structure your response as JSON with this exact format:
{
  "differentials": [
    {
      "rank": 1,
      "diagnosis": "Diagnosis name",
      "confidence": "High/Medium/Low",
      "supporting_evidence": ["evidence 1", "evidence 2"],
      "against_evidence": ["counter-evidence if any"],
      "recommended_workup": ["test 1", "test 2"]
    }
  ],
  "red_flags": ["urgent finding 1"],
  "key_findings": ["notable finding from patient data"],
  "reasoning_summary": "Brief narrative of clinical reasoning"
}

Rules:
- Rank by likelihood given ALL available data (conditions, labs, meds, symptoms)
- Include 3-7 differentials
- Flag any red flags requiring immediate attention
- Reference specific lab values, vitals, or conditions from the data
- DISCLAIMER: All outputs require review by a licensed provider."""


DRUG_INTERACTION_SYSTEM_PROMPT = """You are a clinical pharmacology decision support assistant.

Given a patient's medication list, conditions, allergies, and any drug interaction data from RxNav, provide a comprehensive drug interaction analysis.

Structure your response as JSON with this exact format:
{
  "interactions": [
    {
      "drug_pair": ["Drug A", "Drug B"],
      "severity": "High/Moderate/Low",
      "description": "What happens when these drugs interact",
      "clinical_significance": "Why this matters for THIS patient specifically",
      "recommendation": "What to do about it",
      "evidence_basis": "database-confirmed or AI-generated"
    }
  ],
  "patient_specific_concerns": ["concern considering this patient's conditions/allergies"],
  "medication_summary": "Brief overview of the medication regimen",
  "overall_risk_level": "High/Moderate/Low"
}

Rules:
- Consider the patient's specific conditions and allergies when assessing significance
- Flag drug-allergy cross-reactivity risks
- Note any medications that may worsen existing conditions
- Distinguish between database-confirmed and AI-generated interaction assessments
- DISCLAIMER: All outputs require review by a licensed provider."""


SYNTHESIS_SYSTEM_PROMPT = """You are a senior clinical decision support assistant performing an integrated clinical assessment.

You are given the results of a differential diagnosis analysis, a drug interaction analysis, and a patient summary. Your job is to SYNTHESIZE these into one unified clinical briefing, identifying cross-cutting insights that individual analyses miss.

Key cross-cutting patterns to look for:
- A suspected diagnosis that contraindicates a current medication
- A drug side effect that could explain a symptom in the differential
- A medication interaction that changes the likelihood of a differential
- Missing medications for confirmed conditions (care gaps)

Structure your response as JSON with this exact format:
{
  "key_findings": ["Most important integrated finding 1", "finding 2"],
  "differential_diagnosis": {
    "ranked_list": [
      {"rank": 1, "diagnosis": "name", "confidence": "High/Med/Low", "rationale": "brief"}
    ],
    "medication_implications": "How the DDx affects medication decisions"
  },
  "medication_safety": {
    "critical_alerts": ["urgent medication issue"],
    "interaction_summary": "Key interactions and their clinical impact",
    "ddx_drug_connections": "How suspected diagnoses relate to current medications"
  },
  "care_gaps": ["identified gap 1"],
  "recommended_next_steps": ["step 1", "step 2"],
  "red_flags": ["anything requiring immediate attention"],
  "assessment_summary": "2-3 sentence integrated clinical narrative"
}

Rules:
- Prioritize CROSS-CUTTING insights over restating individual analyses
- The value is in the CONNECTIONS between DDx and drug safety
- Flag anything where one analysis changes the interpretation of another
- DISCLAIMER: All outputs require review by a licensed provider."""


def run_ddx_reasoning(patient_data: dict, symptoms: str = "") -> dict:
    """Generate differential diagnosis using Z.AI."""
    user_content = f"Patient Data:\n{json.dumps(patient_data, indent=2)}"
    if symptoms:
        user_content += f"\n\nAdditional Symptoms Reported:\n{symptoms}"

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=ZAI_MODEL,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": DDX_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        text = response.choices[0].message.content
        # Try to parse as JSON, fall back to raw text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Extract JSON from markdown code block if present
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            return {"raw_response": text}
    except Exception as e:
        logger.error(f"Z.AI DDx reasoning failed: {e}")
        return {"error": f"AI reasoning failed: {str(e)}"}


def run_drug_interaction_reasoning(
    patient_data: dict,
    rxnav_interactions: dict | None = None,
    proposed_medications: list[str] | None = None,
) -> dict:
    """Analyze drug interactions using Z.AI."""
    user_content = f"Patient Data:\n{json.dumps(patient_data, indent=2)}"
    if rxnav_interactions:
        user_content += f"\n\nRxNav Database Interactions:\n{json.dumps(rxnav_interactions, indent=2)}"
    else:
        user_content += "\n\nNo database interactions found. Please analyze based on pharmacological knowledge (label as AI-generated)."
    if proposed_medications:
        user_content += f"\n\nProposed New Medications to Check:\n{json.dumps(proposed_medications)}"

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=ZAI_MODEL,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": DRUG_INTERACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        text = response.choices[0].message.content
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            return {"raw_response": text}
    except Exception as e:
        logger.error(f"Z.AI drug interaction reasoning failed: {e}")
        return {"error": f"AI reasoning failed: {str(e)}"}


def run_synthesis(
    patient_summary: dict,
    ddx_results: dict,
    interaction_results: dict,
    care_gaps: dict | None = None,
) -> dict:
    """Synthesize all analyses into unified clinical assessment."""
    user_content = f"""Patient Summary:
{json.dumps(patient_summary, indent=2)}

Differential Diagnosis Analysis:
{json.dumps(ddx_results, indent=2)}

Drug Interaction Analysis:
{json.dumps(interaction_results, indent=2)}"""

    if care_gaps:
        user_content += f"\n\nCare Gap Analysis:\n{json.dumps(care_gaps, indent=2)}"

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=ZAI_MODEL,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        text = response.choices[0].message.content
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            return {"raw_response": text}
    except Exception as e:
        logger.error(f"Z.AI synthesis failed: {e}")
        return {"error": f"AI reasoning failed: {str(e)}"}


# ── Async streaming generators ──────────────────────────────────────────────


async def stream_ddx_tokens(
    patient_data: dict, symptoms: str = ""
) -> AsyncGenerator[str, None]:
    """Async generator yielding differential diagnosis text tokens as they arrive."""
    user_content = f"Patient Data:\n{json.dumps(patient_data, indent=2)}"
    if symptoms:
        user_content += f"\n\nAdditional Symptoms Reported:\n{symptoms}"
    client = get_async_client()
    stream = client.chat.completions.create(
        model=ZAI_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": DDX_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def stream_drug_interaction_tokens(
    patient_data: dict,
    rxnav_interactions: dict | None = None,
    proposed_medications: list[str] | None = None,
) -> AsyncGenerator[str, None]:
    """Async generator yielding drug interaction analysis tokens as they arrive."""
    user_content = f"Patient Data:\n{json.dumps(patient_data, indent=2)}"
    if rxnav_interactions:
        user_content += f"\n\nRxNav Database Interactions:\n{json.dumps(rxnav_interactions, indent=2)}"
    else:
        user_content += "\n\nNo database interactions found. Please analyze based on pharmacological knowledge (label as AI-generated)."
    if proposed_medications:
        user_content += f"\n\nProposed New Medications to Check:\n{json.dumps(proposed_medications)}"
    client = get_async_client()
    stream = client.chat.completions.create(
        model=ZAI_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": DRUG_INTERACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def stream_synthesis_tokens(
    patient_summary: dict,
    ddx_results: dict,
    interaction_results: dict,
    care_gaps: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Async generator yielding integrated clinical assessment tokens as they arrive."""
    user_content = f"""Patient Summary:
{json.dumps(patient_summary, indent=2)}

Differential Diagnosis Analysis:
{json.dumps(ddx_results, indent=2)}

Drug Interaction Analysis:
{json.dumps(interaction_results, indent=2)}"""
    if care_gaps:
        user_content += f"\n\nCare Gap Analysis:\n{json.dumps(care_gaps, indent=2)}"
    client = get_async_client()
    stream = client.chat.completions.create(
        model=ZAI_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
