"""CDS Agent Executor — orchestrates MCP tools via direct function calls."""

import asyncio
import json
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message, new_task

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.fhir_client import get_patient_data, DEFAULT_FHIR_BASE_URL
from shared.claude_client import run_ddx_reasoning, run_drug_interaction_reasoning, run_synthesis
from shared.rxnav_client import get_interactions, resolve_medications_to_rxcuis

logger = logging.getLogger(__name__)


def _parse_user_input(context: RequestContext) -> dict:
    """Extract patient_id, fhir_base_url, symptoms, and skill from user message."""
    parts = context.message.parts if context.message else []
    text = ""
    for part in parts:
        if hasattr(part, "text"):
            text += part.text

    # Try to parse as JSON first
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

    # Fall back to treating text as patient_id
    return {
        "patient_id": text.strip(),
        "fhir_base_url": DEFAULT_FHIR_BASE_URL,
        "symptoms": "",
        "skill": "comprehensive-clinical-review",
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
            await updater.update_status(TaskState.failed, "No patient_id provided.")
            await event_queue.enqueue_event(
                new_agent_text_message("Error: Please provide a patient_id.")
            )
            return

        skill = params["skill"]
        token = params["access_token"] or None

        try:
            if skill == "quick-drug-check":
                await self._quick_drug_check(updater, event_queue, params, token)
            elif skill == "differential-diagnosis":
                await self._differential_diagnosis(updater, event_queue, params, token)
            else:
                await self._comprehensive_review(updater, event_queue, params, token)
        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            await updater.update_status(TaskState.failed, f"Execution error: {str(e)}")
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: {str(e)}")
            )

    async def _comprehensive_review(
        self, updater: TaskUpdater, event_queue: EventQueue, params: dict, token
    ):
        patient_id = params["patient_id"]
        fhir_base_url = params["fhir_base_url"]

        # Step 1: Get patient summary
        await updater.update_status(TaskState.working, "Step 1/4: Fetching patient data from FHIR...")
        patient_data = await get_patient_data(patient_id, fhir_base_url, token)
        if "error" in patient_data:
            await updater.update_status(TaskState.failed, f"FHIR fetch failed: {patient_data['error']}")
            await event_queue.enqueue_event(new_agent_text_message(json.dumps(patient_data, indent=2)))
            return

        patient_summary = json.dumps(patient_data, indent=2)

        # Step 2: Run DDx and drug interactions in parallel
        await updater.update_status(
            TaskState.working,
            "Step 2/4: Running differential diagnosis + drug interaction analysis in parallel..."
        )

        # Prepare drug interaction data
        medications = patient_data.get("medications", [])
        enriched_meds = await resolve_medications_to_rxcuis(medications)
        rxcuis = [m["rxcui"] for m in enriched_meds if m.get("rxcui")]

        async def _run_ddx():
            return run_ddx_reasoning(patient_data, params.get("symptoms", ""))

        async def _run_interactions():
            rxnav_results = None
            if len(rxcuis) >= 2:
                rxnav_results = await get_interactions(rxcuis)
            proposed = (
                [m.strip() for m in params["proposed_medications"].split(",") if m.strip()]
                if params.get("proposed_medications") else None
            )
            return run_drug_interaction_reasoning(patient_data, rxnav_results, proposed)

        ddx_results, interaction_results = await asyncio.gather(
            _run_ddx(),
            _run_interactions(),
        )

        # Step 3: Synthesize
        await updater.update_status(TaskState.working, "Step 3/4: Synthesizing cross-cutting clinical assessment...")
        synthesis = run_synthesis(patient_data, ddx_results, interaction_results)

        # Step 4: Return results
        await updater.update_status(TaskState.working, "Step 4/4: Preparing clinical briefing...")

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

        await event_queue.enqueue_event(
            new_agent_text_message(json.dumps(final_output, indent=2))
        )
        await updater.complete()

    async def _quick_drug_check(
        self, updater: TaskUpdater, event_queue: EventQueue, params: dict, token
    ):
        patient_id = params["patient_id"]

        await updater.update_status(TaskState.working, "Fetching patient medications...")
        patient_data = await get_patient_data(patient_id, params["fhir_base_url"], token)
        if "error" in patient_data:
            await updater.update_status(TaskState.failed, f"FHIR fetch failed: {patient_data['error']}")
            return

        medications = patient_data.get("medications", [])
        enriched_meds = await resolve_medications_to_rxcuis(medications)
        rxcuis = [m["rxcui"] for m in enriched_meds if m.get("rxcui")]

        await updater.update_status(TaskState.working, "Checking drug interactions...")
        rxnav_results = await get_interactions(rxcuis) if len(rxcuis) >= 2 else None

        proposed = (
            [m.strip() for m in params["proposed_medications"].split(",") if m.strip()]
            if params.get("proposed_medications") else None
        )
        result = run_drug_interaction_reasoning(patient_data, rxnav_results, proposed)

        await event_queue.enqueue_event(
            new_agent_text_message(json.dumps(result, indent=2))
        )
        await updater.complete()

    async def _differential_diagnosis(
        self, updater: TaskUpdater, event_queue: EventQueue, params: dict, token
    ):
        patient_id = params["patient_id"]

        await updater.update_status(TaskState.working, "Fetching patient data for differential diagnosis...")
        patient_data = await get_patient_data(patient_id, params["fhir_base_url"], token)
        if "error" in patient_data:
            await updater.update_status(TaskState.failed, f"FHIR fetch failed: {patient_data['error']}")
            return

        await updater.update_status(TaskState.working, "Generating differential diagnosis...")
        result = run_ddx_reasoning(patient_data, params.get("symptoms", ""))

        await event_queue.enqueue_event(
            new_agent_text_message(json.dumps(result, indent=2))
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("Cancel not supported")
