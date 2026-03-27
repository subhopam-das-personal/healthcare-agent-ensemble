"""Tests for ui_server/server.py — event parsing and URL resolution."""

import sys, os
import importlib
import types
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Import helpers (avoid loading FastAPI app at module level) ────────────────

def _load_ui_server(env_overrides: dict | None = None):
    """Import server module with optional env overrides, isolated each call."""
    env_overrides = env_overrides or {}
    saved = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update(env_overrides)
    # Force reimport so _resolve_a2a_url() re-runs with new env
    if "ui_server.server" in sys.modules:
        del sys.modules["ui_server.server"]
    try:
        import ui_server.server as m
        return m
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── _resolve_a2a_url ──────────────────────────────────────────────────────────

class TestResolveA2aUrl:
    def _url(self, env: dict):
        m = _load_ui_server(env)
        return m.A2A_AGENT_URL

    def test_explicit_a2a_agent_url_wins(self):
        url = self._url({
            "A2A_AGENT_URL": "https://custom.example.com",
            "RAILWAY_SERVICE_A2A_AGENT_URL": "should-be-ignored.railway.app",
        })
        assert url == "https://custom.example.com"

    def test_railway_hostname_gets_https(self):
        url = self._url({
            "A2A_AGENT_URL": "",
            "RAILWAY_SERVICE_A2A_AGENT_URL": "a2a-agent-prod.up.railway.app",
        })
        assert url == "https://a2a-agent-prod.up.railway.app"

    def test_non_railway_hostname_gets_http(self):
        url = self._url({
            "A2A_AGENT_URL": "",
            "RAILWAY_SERVICE_A2A_AGENT_URL": "localhost:9999",
        })
        assert url == "http://localhost:9999"

    def test_fallback_when_no_env_vars(self):
        url = self._url({
            "A2A_AGENT_URL": "",
            "RAILWAY_SERVICE_A2A_AGENT_URL": "",
        })
        assert url == "http://localhost:9999"


# ── _text_from_a2a_event ──────────────────────────────────────────────────────

import uuid as _uuid
from a2a.types import (
    Artifact, Message, Part, TextPart, Role, Task, TaskState, TaskStatus,
    TaskArtifactUpdateEvent, TaskStatusUpdateEvent, SendStreamingMessageResponse,
    SendStreamingMessageSuccessResponse, JSONRPCErrorResponse,
    JSONRPCError,
)


def _msg_response(text: str) -> SendStreamingMessageResponse:
    msg = Message(
        role=Role.agent,
        message_id=str(_uuid.uuid4()),
        parts=[Part(root=TextPart(text=text))],
    )
    return SendStreamingMessageResponse(
        root=SendStreamingMessageSuccessResponse(id="1", result=msg)
    )


def _status_response() -> SendStreamingMessageResponse:
    status_msg = Message(
        role=Role.agent,
        message_id=str(_uuid.uuid4()),
        parts=[Part(root=TextPart(text="Step 1/4: Fetching FHIR data..."))],
    )
    event = TaskStatusUpdateEvent(
        task_id=str(_uuid.uuid4()),
        context_id=str(_uuid.uuid4()),
        status=TaskStatus(state=TaskState.working, message=status_msg),
        final=False,
    )
    return SendStreamingMessageResponse(
        root=SendStreamingMessageSuccessResponse(id="1", result=event)
    )


def _artifact_response(text: str, append: bool = False) -> SendStreamingMessageResponse:
    event = TaskArtifactUpdateEvent(
        task_id=str(_uuid.uuid4()),
        context_id=str(_uuid.uuid4()),
        artifact=Artifact(
            artifact_id=str(_uuid.uuid4()),
            parts=[Part(root=TextPart(text=text))],
        ),
        append=append,
    )
    return SendStreamingMessageResponse(
        root=SendStreamingMessageSuccessResponse(id="1", result=event)
    )


def _error_response() -> SendStreamingMessageResponse:
    return SendStreamingMessageResponse(
        root=JSONRPCErrorResponse(id="1", error=JSONRPCError(code=-32001, message="not found"))
    )


class TestTextFromA2aEvent:
    @pytest.fixture(autouse=True)
    def _fn(self):
        m = _load_ui_server()
        self.fn = m._text_from_a2a_event

    def test_message_event_returns_text(self):
        assert self.fn(_msg_response("## Differential Diagnosis\n\n")) == "## Differential Diagnosis\n\n"

    def test_message_event_returns_first_non_empty_part(self):
        msg = Message(
            role=Role.agent,
            message_id=str(_uuid.uuid4()),
            parts=[
                Part(root=TextPart(text="")),
                Part(root=TextPart(text="hello")),
                Part(root=TextPart(text="world")),
            ],
        )
        resp = SendStreamingMessageResponse(
            root=SendStreamingMessageSuccessResponse(id="1", result=msg)
        )
        assert self.fn(resp) == "hello"

    def test_task_status_update_working_returns_status_prefix(self):
        assert self.fn(_status_response()) == "status:Step 1/4: Fetching FHIR data..."

    def test_task_status_update_working_empty_message_returns_none(self):
        event = TaskStatusUpdateEvent(
            task_id=str(_uuid.uuid4()),
            context_id=str(_uuid.uuid4()),
            status=TaskStatus(state=TaskState.working),
            final=False,
        )
        resp = SendStreamingMessageResponse(
            root=SendStreamingMessageSuccessResponse(id="1", result=event)
        )
        assert self.fn(resp) is None

    def test_error_response_returns_none(self):
        assert self.fn(_error_response()) is None

    def test_empty_parts_returns_none(self):
        msg = Message(role=Role.agent, message_id=str(_uuid.uuid4()), parts=[])
        resp = SendStreamingMessageResponse(
            root=SendStreamingMessageSuccessResponse(id="1", result=msg)
        )
        assert self.fn(resp) is None

    def test_empty_text_parts_returns_none(self):
        msg = Message(
            role=Role.agent,
            message_id=str(_uuid.uuid4()),
            parts=[Part(root=TextPart(text="")), Part(root=TextPart(text=""))],
        )
        resp = SendStreamingMessageResponse(
            root=SendStreamingMessageSuccessResponse(id="1", result=msg)
        )
        assert self.fn(resp) is None

    def test_artifact_event_returns_text(self):
        assert self.fn(_artifact_response("## Differential Diagnosis\n\n")) == "## Differential Diagnosis\n\n"

    def test_artifact_event_append_chunk_returns_text(self):
        assert self.fn(_artifact_response("token chunk", append=True)) == "token chunk"

    def test_artifact_event_empty_text_returns_none(self):
        event = TaskArtifactUpdateEvent(
            task_id=str(_uuid.uuid4()),
            context_id=str(_uuid.uuid4()),
            artifact=Artifact(
                artifact_id=str(_uuid.uuid4()),
                parts=[Part(root=TextPart(text=""))],
            ),
        )
        resp = SendStreamingMessageResponse(
            root=SendStreamingMessageSuccessResponse(id="1", result=event)
        )
        assert self.fn(resp) is None
