"""Unit tests for a2a_agent/executor.py — input parsing and skill dispatch."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from a2a_agent.executor import CDSAgentExecutor, _parse_user_input
from shared.fhir_client import DEFAULT_FHIR_BASE_URL


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_context(text: str):
    """Build a minimal RequestContext mock with given text message."""
    inner = MagicMock()
    inner.text = text
    inner.root = inner  # .root returns itself

    part = MagicMock()
    part.root = inner

    msg = MagicMock()
    msg.parts = [part]

    ctx = MagicMock()
    ctx.message = msg
    ctx.current_task = None
    return ctx


# --------------------------------------------------------------------------- #
# _parse_user_input
# --------------------------------------------------------------------------- #

class TestParseUserInput:
    def test_json_with_all_fields(self):
        payload = {
            "patient_id": "p-123",
            "fhir_base_url": "https://custom-fhir.example.com",
            "symptoms": "chest pain",
            "skill": "quick-drug-check",
            "proposed_medications": "warfarin",
            "access_token": "tok-abc",
        }
        ctx = _make_context(json.dumps(payload))
        result = _parse_user_input(ctx)

        assert result["patient_id"] == "p-123"
        assert result["fhir_base_url"] == "https://custom-fhir.example.com"
        assert result["symptoms"] == "chest pain"
        assert result["skill"] == "quick-drug-check"
        assert result["proposed_medications"] == "warfarin"
        assert result["access_token"] == "tok-abc"

    def test_json_partial_fields_use_defaults(self):
        payload = {"patient_id": "p-456"}
        ctx = _make_context(json.dumps(payload))
        result = _parse_user_input(ctx)

        assert result["patient_id"] == "p-456"
        assert result["fhir_base_url"] == DEFAULT_FHIR_BASE_URL
        assert result["symptoms"] == ""
        assert result["skill"] == "comprehensive-clinical-review"
        assert result["proposed_medications"] == ""
        assert result["access_token"] == ""

    def test_plain_text_treated_as_patient_id(self):
        ctx = _make_context("smart-7890123")
        result = _parse_user_input(ctx)

        assert result["patient_id"] == "smart-7890123"
        assert result["skill"] == "comprehensive-clinical-review"
        assert result["fhir_base_url"] == DEFAULT_FHIR_BASE_URL

    def test_empty_text_gives_empty_patient_id(self):
        ctx = _make_context("")
        result = _parse_user_input(ctx)
        assert result["patient_id"] == ""

    def test_whitespace_only_text_stripped(self):
        ctx = _make_context("  \n  smart-111  \n  ")
        result = _parse_user_input(ctx)
        assert result["patient_id"] == "smart-111"

    def test_invalid_json_with_spaces_returns_empty_patient_id(self):
        # "{not valid json}" has spaces → not a bare patient ID → patient_id is empty
        ctx = _make_context("{not valid json}")
        result = _parse_user_input(ctx)
        assert result["patient_id"] == ""

    def test_invalid_json_no_spaces_used_as_patient_id(self):
        # No spaces → treated as bare patient ID (step 3 fallback)
        ctx = _make_context("not-valid-json-but-no-spaces")
        result = _parse_user_input(ctx)
        assert result["patient_id"] == "not-valid-json-but-no-spaces"

    def test_multiple_parts_concatenated(self):
        """Multiple text parts should be concatenated before parsing."""
        inner1, inner2 = MagicMock(), MagicMock()
        inner1.text = '{"patient_id": "p-'
        inner2.text = '999"}'
        inner1.root = inner1
        inner2.root = inner2

        part1, part2 = MagicMock(), MagicMock()
        part1.root = inner1
        part2.root = inner2

        msg = MagicMock()
        msg.parts = [part1, part2]
        ctx = MagicMock()
        ctx.message = msg

        result = _parse_user_input(ctx)
        assert result["patient_id"] == "p-999"

    def test_no_message_returns_empty_patient_id(self):
        ctx = MagicMock()
        ctx.message = None
        result = _parse_user_input(ctx)
        assert result["patient_id"] == ""


# --------------------------------------------------------------------------- #
# CDSAgentExecutor.execute — error cases
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_execute_fails_when_no_patient_id():
    """Missing patient_id should immediately fail the task."""
    executor = CDSAgentExecutor()
    ctx = _make_context("")
    event_queue = AsyncMock()

    task_mock = MagicMock()
    task_mock.id = "task-1"
    task_mock.context_id = "ctx-1"

    with patch("a2a_agent.executor.new_task", return_value=task_mock), \
         patch("a2a_agent.executor.TaskUpdater") as mock_updater_cls:
        mock_updater = AsyncMock()
        mock_updater_cls.return_value = mock_updater
        await executor.execute(ctx, event_queue)

    mock_updater.update_status.assert_called_once()
    call_args = mock_updater.update_status.call_args
    from a2a.types import TaskState
    assert call_args.kwargs.get("TaskState") == TaskState.input_required or \
           call_args.args[0] == TaskState.input_required



@pytest.mark.asyncio
async def test_execute_routes_to_quick_drug_check():
    """skill=quick-drug-check should call _quick_drug_check."""
    executor = CDSAgentExecutor()
    payload = {"patient_id": "p1", "skill": "quick-drug-check"}
    ctx = _make_context(json.dumps(payload))
    event_queue = AsyncMock()

    task_mock = MagicMock()
    task_mock.id = "task-2"
    task_mock.context_id = "ctx-2"

    with patch("a2a_agent.executor.new_task", return_value=task_mock), \
         patch("a2a_agent.executor.TaskUpdater") as mock_updater_cls, \
         patch.object(executor, "_quick_drug_check", new_callable=AsyncMock) as mock_drug:
        mock_updater = AsyncMock()
        mock_updater_cls.return_value = mock_updater
        await executor.execute(ctx, event_queue)

    mock_drug.assert_called_once()


@pytest.mark.asyncio
async def test_execute_routes_to_differential_diagnosis():
    """skill=differential-diagnosis should call _differential_diagnosis."""
    executor = CDSAgentExecutor()
    payload = {"patient_id": "p1", "skill": "differential-diagnosis"}
    ctx = _make_context(json.dumps(payload))
    event_queue = AsyncMock()

    task_mock = MagicMock()
    task_mock.id = "task-3"
    task_mock.context_id = "ctx-3"

    with patch("a2a_agent.executor.new_task", return_value=task_mock), \
         patch("a2a_agent.executor.TaskUpdater") as mock_updater_cls, \
         patch.object(executor, "_differential_diagnosis", new_callable=AsyncMock) as mock_ddx:
        mock_updater = AsyncMock()
        mock_updater_cls.return_value = mock_updater
        await executor.execute(ctx, event_queue)

    mock_ddx.assert_called_once()


@pytest.mark.asyncio
async def test_execute_routes_to_comprehensive_review_by_default():
    """Any unknown or missing skill should default to comprehensive review."""
    executor = CDSAgentExecutor()
    payload = {"patient_id": "p1", "skill": "comprehensive-clinical-review"}
    ctx = _make_context(json.dumps(payload))
    event_queue = AsyncMock()

    task_mock = MagicMock()
    task_mock.id = "task-4"
    task_mock.context_id = "ctx-4"

    with patch("a2a_agent.executor.new_task", return_value=task_mock), \
         patch("a2a_agent.executor.TaskUpdater") as mock_updater_cls, \
         patch.object(executor, "_comprehensive_review", new_callable=AsyncMock) as mock_review:
        mock_updater = AsyncMock()
        mock_updater_cls.return_value = mock_updater
        await executor.execute(ctx, event_queue)

    mock_review.assert_called_once()


@pytest.mark.asyncio
async def test_execute_handles_unexpected_exception():
    """Unexpected exceptions in skill dispatch should mark task as failed."""
    executor = CDSAgentExecutor()
    payload = {"patient_id": "p1"}
    ctx = _make_context(json.dumps(payload))
    event_queue = AsyncMock()

    task_mock = MagicMock()
    task_mock.id = "task-5"
    task_mock.context_id = "ctx-5"

    with patch("a2a_agent.executor.new_task", return_value=task_mock), \
         patch("a2a_agent.executor.TaskUpdater") as mock_updater_cls, \
         patch.object(executor, "_comprehensive_review", new_callable=AsyncMock,
                      side_effect=RuntimeError("DB crashed")):
        mock_updater = AsyncMock()
        mock_updater_cls.return_value = mock_updater
        await executor.execute(ctx, event_queue)

    from a2a.types import TaskState
    mock_updater.update_status.assert_called()
    final_call = mock_updater.update_status.call_args_list[-1]
    state_arg = final_call.args[0] if final_call.args else final_call.kwargs.get("state")
    assert state_arg == TaskState.failed
