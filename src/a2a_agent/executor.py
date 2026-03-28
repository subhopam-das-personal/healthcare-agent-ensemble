"""CDS Agent Executor — direct FHIR + streaming Claude reasoning."""

import json
import logging
import re
import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TaskState, TextPart
from a2a.utils import new_agent_text_message, new_task

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.fhir_client import get_patient_data, DEFAULT_FHIR_BASE_URL
from shared.rxnav_client import resolve_medications_to_rxcuis, get_interactions
from shared.claude_client import (
    stream_ddx_tokens,
    stream_drug_interaction_tokens,
    stream_synthesis_tokens,
)
from shared.trials_client import search_trials_by_conditions

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
        url = match.group(0).rstrip("/")
        return url
    return DEFAULT_FHIR_BASE_URL


def _status_msg(text: str):
    return new_agent_text_message(text)


def _parse_json_text(text: str) -> dict:
    """Parse JSON from accumulated streaming text, handling markdown code blocks."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```json" in text:
        try:
            return json.loads(text.split("```json")[1].split("```")[0].strip())
        except (json.JSONDecodeError, IndexError):
            pass
    if "```" in text:
        try:
            return json.loads(text.split("```")[1].split("```")[0].strip())
        except (json.JSONDecodeError, IndexError):
            pass
    return {"raw_response": text}


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
            "patient_json": parsed.get("patient_json", ""),
        }
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Extract UUID patient ID from natural language (e.g. PromptOpinion platform input)
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
        try:
            if skill == "quick-drug-check":
                await self._quick_drug_check(updater, event_queue, params)
            elif skill == "differential-diagnosis":
                await self._differential_diagnosis(updater, event_queue, params)
            else:
                await self._comprehensive_review(updater, event_queue, params)
        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            await updater.update_status(
                TaskState.failed,
                message=_status_msg(f"Error: {str(e)}"),
            )

    async def _fetch_patient(self, patient_id: str, fhir_base_url: str, token: str | None, patient_json: str = "") -> dict:
        return await get_patient_data(patient_id, fhir_base_url, token, patient_json)

    async def _fetch_rxnav(self, medications: list) -> dict | None:
        if len(medications) < 2:
            return None
        enriched = await resolve_medications_to_rxcuis(medications)
        rxcuis = [m["rxcui"] for m in enriched if m.get("rxcui")]
        if len(rxcuis) < 2:
            return None
        return await get_interactions(rxcuis)

    async def _comprehensive_review(self, updater, event_queue, params):
        patient_id = params["patient_id"]
        fhir_base_url = params["fhir_base_url"]
        token = params["access_token"] or None
        patient_json = params.get("patient_json", "")

        # Step 1: Fetch patient data
        await updater.update_status(TaskState.working,
            message=_status_msg("Step 1/5: Fetching patient data from FHIR..."))

        patient_data = await self._fetch_patient(patient_id, fhir_base_url, token, patient_json)
        if "error" in patient_data:
            await updater.update_status(TaskState.failed,
                message=_status_msg(f"FHIR fetch failed: {patient_data['error']}"))
            return

        pt = patient_data.get("patient", {})
        await updater.update_status(TaskState.working, message=_status_msg(
            f"Patient: {pt.get('name', patient_id)}, "
            f"{len(patient_data.get('conditions', []))} conditions, "
            f"{len(patient_data.get('medications', []))} medications"
        ))

        # Step 2: RxNav drug interaction database lookup
        await updater.update_status(TaskState.working,
            message=_status_msg("Step 2/5: Looking up drug interaction database..."))
        rxnav_results = await self._fetch_rxnav(patient_data.get("medications", []))

        # Step 3: Match clinical trials
        await updater.update_status(TaskState.working,
            message=_status_msg("Step 3/5: Searching for matching clinical trials..."))

        conditions = patient_data.get("conditions", [])
        condition_names = [c.get("display", "") for c in conditions if c.get("display")]
        age = None
        if pt.get("birthDate"):
            try:
                from datetime import date as _date
                birth = _date.fromisoformat(pt["birthDate"])
                age = (_date.today() - birth).days // 365
            except Exception:
                pass
        gender = pt.get("gender")
        trials = await search_trials_by_conditions(condition_names, age=age, gender=gender)

        await updater.update_status(TaskState.working, message=_status_msg(
            f"Found {len(trials)} recruiting trial{'s' if len(trials) != 1 else ''} "
            f"matching patient profile"
        ))

        # Start artifact stream with trials section
        artifact_id = str(uuid.uuid4())
        import json as _json
        await updater.add_artifact(
            parts=[Part(root=TextPart(text="\n## Clinical Trials\n\n" + _json.dumps(trials, indent=2)))],
            artifact_id=artifact_id, append=False, last_chunk=False,
        )

        # Step 4: Stream differential diagnosis
        await updater.update_status(TaskState.working,
            message=_status_msg("Step 4/5: Streaming differential diagnosis..."))

        await updater.add_artifact(
            parts=[Part(root=TextPart(text="\n\n## Differential Diagnosis\n\n"))],
            artifact_id=artifact_id, append=True, last_chunk=False,
        )

        ddx_text = ""
        async for chunk in stream_ddx_tokens(patient_data, params.get("symptoms", "")):
            ddx_text += chunk
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=chunk))],
                artifact_id=artifact_id, append=True, last_chunk=False,
            )
        ddx_results = _parse_json_text(ddx_text)

        # Stream drug interactions
        proposed = [m.strip() for m in params.get("proposed_medications", "").split(",")
                    if m.strip()] if params.get("proposed_medications") else None
        await updater.add_artifact(
            parts=[Part(root=TextPart(text="\n\n## Drug Interaction Analysis\n\n"))],
            artifact_id=artifact_id, append=True, last_chunk=False,
        )

        drug_text = ""
        async for chunk in stream_drug_interaction_tokens(patient_data, rxnav_results, proposed):
            drug_text += chunk
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=chunk))],
                artifact_id=artifact_id, append=True, last_chunk=False,
            )
        interaction_results = _parse_json_text(drug_text)

        # Step 5: Stream integrated synthesis
        await updater.update_status(TaskState.working,
            message=_status_msg("Step 5/5: Streaming integrated clinical assessment..."))
        await updater.add_artifact(
            parts=[Part(root=TextPart(text="\n\n## Integrated Clinical Assessment\n\n"))],
            artifact_id=artifact_id, append=True, last_chunk=False,
        )

        async for chunk in stream_synthesis_tokens(patient_data, ddx_results, interaction_results):
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=chunk))],
                artifact_id=artifact_id, append=True, last_chunk=False,
            )

        await updater.complete()

    async def _quick_drug_check(self, updater, event_queue, params):
        patient_id = params["patient_id"]
        fhir_base_url = params["fhir_base_url"]
        token = params["access_token"] or None
        patient_json = params.get("patient_json", "")

        await updater.update_status(TaskState.working,
            message=_status_msg("Fetching patient medications from FHIR..."))

        patient_data = await self._fetch_patient(patient_id, fhir_base_url, token, patient_json)
        if "error" in patient_data:
            await updater.update_status(TaskState.failed,
                message=_status_msg(f"Drug check failed: {patient_data['error']}"))
            return

        medications = patient_data.get("medications", [])
        if not medications:
            artifact_id = str(uuid.uuid4())
            no_med_text = json.dumps({"interactions": [], "note": "No medications found.", "overall_risk_level": "N/A"}, indent=2)
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=no_med_text))],
                artifact_id=artifact_id, append=False, last_chunk=True,
            )
            await updater.complete()
            return

        await updater.update_status(TaskState.working,
            message=_status_msg(f"{len(medications)} medications found. Looking up interactions..."))
        rxnav_results = await self._fetch_rxnav(medications)

        proposed = [m.strip() for m in params.get("proposed_medications", "").split(",")
                    if m.strip()] if params.get("proposed_medications") else None

        await updater.update_status(TaskState.working,
            message=_status_msg("Streaming drug interaction analysis..."))

        artifact_id = str(uuid.uuid4())
        await updater.add_artifact(
            parts=[Part(root=TextPart(text="\n## Drug Interaction Analysis\n\n"))],
            artifact_id=artifact_id, append=False, last_chunk=False,
        )

        async for chunk in stream_drug_interaction_tokens(patient_data, rxnav_results, proposed):
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=chunk))],
                artifact_id=artifact_id, append=True, last_chunk=False,
            )

        await updater.complete()

    async def _differential_diagnosis(self, updater, event_queue, params):
        patient_id = params["patient_id"]
        fhir_base_url = params["fhir_base_url"]
        token = params["access_token"] or None
        patient_json = params.get("patient_json", "")

        await updater.update_status(TaskState.working,
            message=_status_msg("Fetching patient data from FHIR..."))

        patient_data = await self._fetch_patient(patient_id, fhir_base_url, token, patient_json)
        if "error" in patient_data:
            await updater.update_status(TaskState.failed,
                message=_status_msg(f"DDx failed: {patient_data['error']}"))
            return

        if not patient_data.get("conditions") and not params.get("symptoms"):
            await updater.update_status(TaskState.failed,
                message=_status_msg("No conditions found and no symptoms provided."))
            return

        await updater.update_status(TaskState.working,
            message=_status_msg("Streaming differential diagnosis..."))

        artifact_id = str(uuid.uuid4())
        await updater.add_artifact(
            parts=[Part(root=TextPart(text="\n## Differential Diagnosis\n\n"))],
            artifact_id=artifact_id, append=False, last_chunk=False,
        )

        async for chunk in stream_ddx_tokens(patient_data, params.get("symptoms", "")):
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=chunk))],
                artifact_id=artifact_id, append=True, last_chunk=False,
            )

        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("Cancel not supported")
