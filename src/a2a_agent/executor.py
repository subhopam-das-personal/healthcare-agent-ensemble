"""CDS Agent Executor — orchestrates clinical tools via MCP protocol."""

import asyncio
import json
import logging
import re

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message, new_task

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.fhir_client import DEFAULT_FHIR_BASE_URL
from shared.mcp_client import (
    make_mcp_session,
    mcp_get_patient_summary,
    mcp_generate_differential_diagnosis,
    mcp_check_drug_interactions,
    mcp_synthesize_clinical_assessment,
)

logger = logging.getLogger(__name__)

# UUID pattern for FHIR patient IDs
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)

# Skill keywords for natural language routing
_SKILL_KEYWORDS = {
    "drug": "quick-drug-check",
    "interaction": "quick-drug-check",
    "medication": "quick-drug-check",
    "diagnosis": "differential-diagnosis",
    "differential": "differential-diagnosis",
    "ddx": "differential-diagnosis",
}


def _extract_skill_from_text(text: str) -> str:
    lower = text.lower()
    for keyword, skill in _SKILL_KEYWORDS.items():
        if keyword in lower:
            return skill
    return "comprehensive-clinical-review"


def _extract_fhir_url_from_text(text: str) -> str:
    """Extract FHIR base URL if explicitly mentioned in the text."""
    match = re.search(r"https?://[^\s/]+(?:/[^\s]*)?/(?=Patient|Observation|Condition)", text)
    if match:
        # Return just the base up to the resource type
        url = match.group(0).rstrip("/")
        return url
    return DEFAULT_FHIR_BASE_URL


def _status_msg(text: str):
    return new_agent_text_message(text)


def _parse_user_input(context: RequestContext) -> dict:
    parts = context.message.parts if context.message else []
    text = ""
    for part in parts:
        inner = part.root if hasattr(part, "root") else part
        if hasattr(inner, "text"):
            text += inner.text

    # 1. Try JSON first (structured input from API callers)
    try:
        parsed = json.loads(text)
        return {
            "patient_id": parsed.get("patient_id", ""),
            "fhir_base_url": parsed.get("fhir_base_url", DEFAULT_FHIR_BASE_URL),
            "symptoms": parsed.get("symptoms", ""),
            "skill": parsed.get("skill", "comprehensive-clinical-review"),
            "proposed_medications": parsed.get("proposed_medications", ""),
            "access_token": parsed.get("access_token", ""),
        }
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Extract UUID patient ID from natural language (e.g. Prompt Opinion platform input)
    uuids = _UUID_RE.findall(text)
    if uuids:
        patient_id = uuids[0]
        logger.info(f"[_parse_user_input] Extracted patient_id from natural language: {patient_id!r}")
        return {
            "patient_id": patient_id,
            "fhir_base_url": _extract_fhir_url_from_text(text),
            "symptoms": "",
            "skill": _extract_skill_from_text(text),
            "proposed_medications": "",
            "access_token": "",
        }

    # 3. Last resort: treat as bare patient ID only if it looks like one (no spaces)
    stripped = text.strip()
    if stripped and " " not in stripped and len(stripped) < 128:
        logger.info(f"[_parse_user_input] Using bare text as patient_id: {stripped!r}")
        return {
            "patient_id": stripped,
            "fhir_base_url": DEFAULT_FHIR_BASE_URL,
            "symptoms": "",
            "skill": "comprehensive-clinical-review",
            "proposed_medications": "",
            "access_token": "",
        }

    # 4. Cannot extract a patient ID — return empty so the executor fails gracefully
    logger.warning(f"[_parse_user_input] Could not extract patient_id from: {text[:100]!r}")
    return {
        "patient_id": "",
        "fhir_base_url": DEFAULT_FHIR_BASE_URL,
        "symptoms": text.strip(),
        "skill": _extract_skill_from_text(text),
        "proposed_medications": "",
        "access_token": "",
    }


class CDSAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        params = _parse_user_input(context)

        patient_id = params["patient_id"]
        if not patient_id:
            await updater.update_status(
                TaskState.failed,
                message=_status_msg("Error: No patient_id provided. Send a patient_id as text or JSON."),
            )
            return

        skill = params["skill"]

        async with make_mcp_session() as session:
            try:
                if skill == "quick-drug-check":
                    await self._quick_drug_check(updater, event_queue, params, session)
                elif skill == "differential-diagnosis":
                    await self._differential_diagnosis(updater, event_queue, params, session)
                else:
                    await self._comprehensive_review(updater, event_queue, params, session)
            except Exception as e:
                logger.error(f"Agent execution failed: {e}", exc_info=True)
                await updater.update_status(
                    TaskState.failed,
                    message=_status_msg(f"Error: {str(e)}"),
                )

    async def _comprehensive_review(self, updater, event_queue, params, session):
        patient_id = params["patient_id"]
        fhir_base_url = params["fhir_base_url"]
        token = params["access_token"] or ""

        # Step 1: Fetch patient data
        await updater.update_status(TaskState.working,
            message=_status_msg("Step 1/4: Fetching patient data from FHIR..."))
        await updater.update_status(TaskState.working,
            message=_status_msg(f"→ Calling MCP: get_patient_summary(patient_id={patient_id!r})"))

        patient_data = await mcp_get_patient_summary(session, patient_id, fhir_base_url, token)
        if "error" in patient_data:
            await updater.update_status(TaskState.failed,
                message=_status_msg(f"FHIR fetch failed: {patient_data['error']}"))
            return

        pt = patient_data.get("patient", {})
        await updater.update_status(TaskState.working, message=_status_msg(
            f"← Patient data: {pt.get('name', patient_id)}, "
            f"{len(patient_data.get('conditions', []))} conditions, "
            f"{len(patient_data.get('medications', []))} medications"
        ))

        patient_json_str = json.dumps(patient_data)

        # Step 2: Parallel DDx + drug interactions
        await updater.update_status(TaskState.working,
            message=_status_msg("Step 2/4: Running differential diagnosis + drug interaction analysis in parallel..."))
        await updater.update_status(TaskState.working,
            message=_status_msg("→ Calling MCP: generate_differential_diagnosis(...)"))
        await updater.update_status(TaskState.working,
            message=_status_msg("→ Calling MCP: check_drug_interactions(...)"))

        ddx_results, interaction_results = await asyncio.gather(
            mcp_generate_differential_diagnosis(
                session, patient_id, fhir_base_url,
                symptoms=params.get("symptoms", ""),
                access_token=token,
                patient_json=patient_json_str,
            ),
            mcp_check_drug_interactions(
                session, patient_id, fhir_base_url,
                proposed_medications=params.get("proposed_medications", ""),
                access_token=token,
                patient_json=patient_json_str,
            ),
        )

        top_ddx = ""
        if isinstance(ddx_results, dict) and ddx_results.get("differentials"):
            d = ddx_results["differentials"][0]
            top_ddx = f"{d.get('diagnosis', '?')} ({d.get('confidence', '?')})"
        await updater.update_status(TaskState.working,
            message=_status_msg(
                f"← DDx: {len(ddx_results.get('differentials', []))} differentials"
                + (f", top: {top_ddx}" if top_ddx else "")
            ))

        risk = interaction_results.get("overall_risk_level", "?") if isinstance(interaction_results, dict) else "?"
        n_interactions = len(interaction_results.get("interactions", [])) if isinstance(interaction_results, dict) else 0
        await updater.update_status(TaskState.working,
            message=_status_msg(f"← Drug interactions: {n_interactions} found, risk level: {risk}"))

        # Step 3: Synthesize
        await updater.update_status(TaskState.working,
            message=_status_msg("Step 3/4: Synthesizing cross-cutting clinical assessment..."))
        await updater.update_status(TaskState.working,
            message=_status_msg("→ Calling MCP: synthesize_clinical_assessment(...)"))

        synthesis = await mcp_synthesize_clinical_assessment(
            session,
            patient_summary_json=patient_json_str,
            ddx_results_json=json.dumps(ddx_results),
            interaction_results_json=json.dumps(interaction_results),
        )

        await updater.update_status(TaskState.working,
            message=_status_msg("← Synthesis complete"))

        # Step 4: Return
        await updater.update_status(TaskState.working,
            message=_status_msg("Step 4/4: Preparing clinical briefing..."))

        final_output = {
            "assessment_type": "Comprehensive Clinical Review",
            "patient_id": patient_id,
            "patient_summary": patient_data.get("patient", {}),
            "synthesis": synthesis,
            "detailed_analyses": {
                "differential_diagnosis": ddx_results,
                "drug_interactions": interaction_results,
            },
            "disclaimer": "All outputs require review by a licensed healthcare provider.",
        }
        await event_queue.enqueue_event(new_agent_text_message(json.dumps(final_output, indent=2)))
        await updater.complete()

    async def _quick_drug_check(self, updater, event_queue, params, session):
        patient_id = params["patient_id"]
        fhir_base_url = params["fhir_base_url"]
        token = params["access_token"] or ""

        await updater.update_status(TaskState.working,
            message=_status_msg("Fetching patient medications..."))
        await updater.update_status(TaskState.working,
            message=_status_msg(f"→ Calling MCP: check_drug_interactions(patient_id={patient_id!r})"))

        result = await mcp_check_drug_interactions(
            session, patient_id, fhir_base_url,
            proposed_medications=params.get("proposed_medications", ""),
            access_token=token,
        )

        if "error" in result:
            await updater.update_status(TaskState.failed,
                message=_status_msg(f"Drug check failed: {result['error']}"))
            return

        risk = result.get("overall_risk_level", "?")
        n = len(result.get("interactions", []))
        await updater.update_status(TaskState.working,
            message=_status_msg(f"← Drug interactions: {n} found, risk level: {risk}"))

        await event_queue.enqueue_event(new_agent_text_message(json.dumps(result, indent=2)))
        await updater.complete()

    async def _differential_diagnosis(self, updater, event_queue, params, session):
        patient_id = params["patient_id"]
        fhir_base_url = params["fhir_base_url"]
        token = params["access_token"] or ""

        await updater.update_status(TaskState.working,
            message=_status_msg("Fetching patient data for differential diagnosis..."))
        await updater.update_status(TaskState.working,
            message=_status_msg(f"→ Calling MCP: generate_differential_diagnosis(patient_id={patient_id!r})"))

        result = await mcp_generate_differential_diagnosis(
            session, patient_id, fhir_base_url,
            symptoms=params.get("symptoms", ""),
            access_token=token,
        )

        if "error" in result:
            await updater.update_status(TaskState.failed,
                message=_status_msg(f"DDx failed: {result['error']}"))
            return

        top = ""
        if result.get("differentials"):
            d = result["differentials"][0]
            top = f", top: {d.get('diagnosis', '?')} ({d.get('confidence', '?')})"
        await updater.update_status(TaskState.working,
            message=_status_msg(f"← DDx: {len(result.get('differentials', []))} differentials{top}"))

        await event_queue.enqueue_event(new_agent_text_message(json.dumps(result, indent=2)))
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("Cancel not supported")
