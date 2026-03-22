"""Claude API client for clinical reasoning."""

import os
import json
import logging
from anthropic import Anthropic

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_TIMEOUT = 60.0


def get_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    return Anthropic(api_key=api_key, timeout=CLAUDE_TIMEOUT)


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
    """Generate differential diagnosis using Claude."""
    client = get_client()

    user_content = f"Patient Data:\n{json.dumps(patient_data, indent=2)}"
    if symptoms:
        user_content += f"\n\nAdditional Symptoms Reported:\n{symptoms}"

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=DDX_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        text = next(b.text for b in response.content if hasattr(b, "text"))
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
        logger.error(f"Claude DDx reasoning failed: {e}")
        return {"error": f"AI reasoning failed: {str(e)}"}


def run_drug_interaction_reasoning(
    patient_data: dict,
    rxnav_interactions: dict | None = None,
    proposed_medications: list[str] | None = None,
) -> dict:
    """Analyze drug interactions using Claude."""
    client = get_client()

    user_content = f"Patient Data:\n{json.dumps(patient_data, indent=2)}"
    if rxnav_interactions:
        user_content += f"\n\nRxNav Database Interactions:\n{json.dumps(rxnav_interactions, indent=2)}"
    else:
        user_content += "\n\nNo database interactions found. Please analyze based on pharmacological knowledge (label as AI-generated)."
    if proposed_medications:
        user_content += f"\n\nProposed New Medications to Check:\n{json.dumps(proposed_medications)}"

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=DRUG_INTERACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        text = next(b.text for b in response.content if hasattr(b, "text"))
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
        logger.error(f"Claude drug interaction reasoning failed: {e}")
        return {"error": f"AI reasoning failed: {str(e)}"}


def run_synthesis(
    patient_summary: dict,
    ddx_results: dict,
    interaction_results: dict,
    care_gaps: dict | None = None,
) -> dict:
    """Synthesize all analyses into unified clinical assessment."""
    client = get_client()

    user_content = f"""Patient Summary:
{json.dumps(patient_summary, indent=2)}

Differential Diagnosis Analysis:
{json.dumps(ddx_results, indent=2)}

Drug Interaction Analysis:
{json.dumps(interaction_results, indent=2)}"""

    if care_gaps:
        user_content += f"\n\nCare Gap Analysis:\n{json.dumps(care_gaps, indent=2)}"

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=SYNTHESIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        text = next(b.text for b in response.content if hasattr(b, "text"))
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
        logger.error(f"Claude synthesis failed: {e}")
        return {"error": f"AI reasoning failed: {str(e)}"}
