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


# ── _a2a_text_from_event ──────────────────────────────────────────────────────

class TestA2aTextFromEvent:
    @pytest.fixture(autouse=True)
    def _fn(self):
        m = _load_ui_server()
        self.fn = m._a2a_text_from_event

    def test_message_event_with_text_part(self):
        event = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {
                "role": "agent",
                "messageId": "m1",
                "parts": [{"kind": "text", "text": "## Differential Diagnosis\n\n"}],
            },
        }
        assert self.fn(event) == "## Differential Diagnosis\n\n"

    def test_message_event_returns_first_non_empty_part(self):
        event = {
            "result": {
                "parts": [
                    {"kind": "text", "text": ""},
                    {"kind": "text", "text": "hello"},
                    {"kind": "text", "text": "world"},
                ]
            }
        }
        assert self.fn(event) == "hello"

    def test_task_status_update_event_returns_none(self):
        """TaskStatusUpdateEvent has 'status', not top-level 'parts' — must be ignored."""
        event = {
            "result": {
                "id": "task-1",
                "status": {
                    "state": "working",
                    "message": {
                        "role": "agent",
                        "parts": [{"kind": "text", "text": "Step 1/4: Fetching FHIR data..."}],
                    },
                },
            }
        }
        assert self.fn(event) is None

    def test_task_object_event_returns_none(self):
        """Initial Task event has no 'parts' at top level."""
        event = {
            "result": {
                "id": "task-1",
                "contextId": "ctx-1",
                "status": {"state": "submitted"},
                "history": [],
            }
        }
        assert self.fn(event) is None

    def test_empty_parts_list_returns_none(self):
        event = {"result": {"parts": []}}
        assert self.fn(event) is None

    def test_parts_with_all_empty_text_returns_none(self):
        event = {"result": {"parts": [{"kind": "text", "text": ""}, {"kind": "text", "text": ""}]}}
        assert self.fn(event) is None

    def test_missing_result_key_returns_none(self):
        assert self.fn({}) is None
        assert self.fn({"jsonrpc": "2.0"}) is None

    def test_error_response_returns_none(self):
        event = {"jsonrpc": "2.0", "id": "1", "error": {"code": -32600, "message": "bad request"}}
        assert self.fn(event) is None
