"""CDS Orchestrator A2A Agent Server."""

import logging
import os
import sys
import uuid
from collections.abc import AsyncGenerator
from typing import Union

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard, AgentSkill, AgentCapabilities, AgentProvider,
    Message, MessageSendParams, Task, TaskArtifactUpdateEvent, TaskStatusUpdateEvent,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a2a_agent.executor import CDSAgentExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

A2A_PORT = int(os.environ.get("PORT", os.environ.get("A2A_PORT", 9999)))
A2A_HOST = os.environ.get("A2A_HOST", "0.0.0.0")
A2A_PUBLIC_URL = os.environ.get("A2A_PUBLIC_URL", f"http://localhost:{A2A_PORT}")

agent_card = AgentCard(
    name="Clinical Decision Support Orchestrator",
    description=(
        "AI-powered CDS agent that orchestrates differential diagnosis, "
        "drug interaction analysis, and clinical assessment synthesis using "
        "FHIR patient data. Combines multiple clinical analyses into unified "
        "briefings that identify cross-cutting insights."
    ),
    url=A2A_PUBLIC_URL.rstrip("/"),
    version="1.0.0",
    provider=AgentProvider(
        organization="HealthcareAgentEnsemble",
        url="https://github.com/subhopam-das-personal/healthcare-agent-ensemble",
    ),
    capabilities=AgentCapabilities(streaming=True, pushNotifications=False),
    defaultInputModes=["text/plain", "application/json"],
    defaultOutputModes=["text/plain", "application/json"],
    skills=[
        AgentSkill(
            id="comprehensive-clinical-review",
            name="Comprehensive Clinical Review",
            description=(
                "Full orchestrated clinical review: extracts FHIR patient data, "
                "runs differential diagnosis and drug interaction analysis in parallel, "
                "then synthesizes into a unified clinical briefing with cross-cutting insights."
            ),
            tags=["diagnosis", "medications", "clinical", "fhir", "synthesis"],
            examples=[
                "Run a comprehensive clinical review for patient 1234",
                '{"patient_id": "1234", "symptoms": "chest pain, shortness of breath"}',
            ],
        ),
        AgentSkill(
            id="quick-drug-check",
            name="Quick Drug Interaction Check",
            description=(
                "Rapid drug interaction analysis for a patient's current medications. "
                "Checks RxNav database and provides AI-powered clinical significance interpretation."
            ),
            tags=["medications", "safety", "pharmacology", "interactions"],
            examples=[
                '{"patient_id": "1234", "skill": "quick-drug-check"}',
                '{"patient_id": "1234", "skill": "quick-drug-check", "proposed_medications": "warfarin,amiodarone"}',
            ],
        ),
        AgentSkill(
            id="differential-diagnosis",
            name="Differential Diagnosis Generator",
            description=(
                "AI-powered differential diagnosis from FHIR patient data and optional symptoms. "
                "Produces ranked differentials with confidence levels and recommended workup."
            ),
            tags=["diagnosis", "clinical", "fhir"],
            examples=[
                '{"patient_id": "1234", "skill": "differential-diagnosis"}',
                '{"patient_id": "1234", "skill": "differential-diagnosis", "symptoms": "chest pain and elevated troponin"}',
            ],
        ),
    ],
)


class RobustRequestHandler(DefaultRequestHandler):
    """DefaultRequestHandler with compatibility fixes for clients (e.g. PromptOpinion)
    that omit task_id in their A2A messages."""

    def _ensure_task_id(self, params: MessageSendParams) -> None:
        if not params.message.task_id:
            params.message.task_id = str(uuid.uuid4())

    async def on_message_send(
        self, params: MessageSendParams, context: ServerCallContext | None = None
    ) -> Union[Message, Task]:
        self._ensure_task_id(params)
        return await super().on_message_send(params, context)

    async def on_message_send_stream(
        self, params: MessageSendParams, context: ServerCallContext | None = None
    ) -> AsyncGenerator[Union[Message, Task, TaskStatusUpdateEvent, TaskArtifactUpdateEvent], None]:
        # Do NOT inject task_id here: the SDK creates its own task for new streams.
        # Injecting a random task_id causes "Task X does not exist" because the
        # generated ID was never saved to InMemoryTaskStore before the lookup.
        async for event in super().on_message_send_stream(params, context):
            yield event


def create_app():
    handler = RobustRequestHandler(
        agent_executor=CDSAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )
    return a2a_app.build()


if __name__ == "__main__":
    logger.info(f"Starting CDS Orchestrator A2A Agent on {A2A_HOST}:{A2A_PORT}")
    app = create_app()
    uvicorn.run(app, host=A2A_HOST, port=A2A_PORT)
