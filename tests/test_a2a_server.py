"""Tests for a2a_agent/server.py — RobustRequestHandler correctness.

These tests are designed to catch deployment-breaking issues before they
reach Railway, particularly around async generator vs coroutine semantics.
"""

import inspect
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from a2a.server.request_handlers import DefaultRequestHandler
from a2a_agent.server import RobustRequestHandler


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_params(task_id=None):
    params = MagicMock()
    params.message = MagicMock()
    params.message.task_id = task_id
    return params


def _make_handler():
    """Instantiate RobustRequestHandler with mocked dependencies."""
    with patch.object(DefaultRequestHandler, "__init__", return_value=None):
        h = RobustRequestHandler.__new__(RobustRequestHandler)
    return h


# ── _ensure_task_id ───────────────────────────────────────────────────────────

class TestEnsureTaskId:
    def test_sets_uuid_when_task_id_is_none(self):
        h = _make_handler()
        params = _make_params(task_id=None)
        h._ensure_task_id(params)
        assert params.message.task_id is not None
        uuid.UUID(params.message.task_id)  # raises ValueError if not a valid UUID

    def test_sets_uuid_when_task_id_is_empty_string(self):
        h = _make_handler()
        params = _make_params(task_id="")
        h._ensure_task_id(params)
        assert params.message.task_id  # non-empty

    def test_preserves_existing_task_id(self):
        h = _make_handler()
        existing = str(uuid.uuid4())
        params = _make_params(task_id=existing)
        h._ensure_task_id(params)
        assert params.message.task_id == existing


# ── on_message_send_stream ────────────────────────────────────────────────────

class TestOnMessageSendStream:
    def test_is_async_generator_function(self):
        """on_message_send_stream MUST be an async generator function.

        The A2A SDK's jsonrpc_handler calls:
            async for event in handler.on_message_send_stream(params, ctx):
        This requires the method to return an async generator when called
        (i.e. the method body uses 'yield', not 'return').

        If it is a plain async def with 'return', calling it produces a
        coroutine, and 'async for' on a coroutine raises TypeError.
        """
        assert inspect.isasyncgenfunction(RobustRequestHandler.on_message_send_stream), (
            "on_message_send_stream must be an async generator function "
            "(body must use 'yield'). A plain 'async def ... return ...' "
            "returns a coroutine, which the A2A SDK cannot iterate."
        )

    def test_calling_returns_async_generator_not_coroutine(self):
        """Calling the method must return an async generator object."""
        h = _make_handler()
        params = _make_params()

        async def _fake_super_gen(p, c):
            yield MagicMock()

        with patch.object(DefaultRequestHandler, "on_message_send_stream", _fake_super_gen):
            result = h.on_message_send_stream(params, None)

        assert inspect.isasyncgen(result), (
            f"Expected async generator, got {type(result).__name__}. "
            "The SDK iterates this with 'async for' without awaiting first."
        )
        result.aclose()  # clean up

    @pytest.mark.asyncio
    async def test_does_not_inject_task_id_for_streaming(self):
        """on_message_send_stream must NOT call _ensure_task_id.

        Injecting a random task_id causes the SDK to look it up in
        InMemoryTaskStore, find nothing, and return a -32001 error.
        The SDK creates its own task for new streams when task_id is absent.
        """
        h = _make_handler()
        params = _make_params(task_id=None)
        sentinel = object()

        async def _fake_super_gen(self_, p, c):
            yield sentinel

        with patch.object(DefaultRequestHandler, "on_message_send_stream", _fake_super_gen):
            events = [e async for e in h.on_message_send_stream(params, None)]

        assert params.message.task_id is None, (
            "on_message_send_stream must not set task_id — SDK handles task creation"
        )
        assert events == [sentinel]

    @pytest.mark.asyncio
    async def test_forwards_all_events_from_super(self):
        """Every event yielded by the parent must be forwarded."""
        h = _make_handler()
        params = _make_params(task_id="existing-id")
        events_in = [MagicMock(), MagicMock(), MagicMock()]

        async def _fake_super_gen(self_, p, c):  # patch.object passes self as first arg
            for ev in events_in:
                yield ev

        with patch.object(DefaultRequestHandler, "on_message_send_stream", _fake_super_gen):
            events_out = [e async for e in h.on_message_send_stream(params, None)]

        assert events_out == events_in


# ── on_message_send ───────────────────────────────────────────────────────────

class TestOnMessageSend:
    @pytest.mark.asyncio
    async def test_patches_task_id_and_delegates(self):
        h = _make_handler()
        params = _make_params(task_id=None)
        sentinel = MagicMock()

        with patch.object(DefaultRequestHandler, "on_message_send", new=AsyncMock(return_value=sentinel)):
            result = await h.on_message_send(params, None)

        assert params.message.task_id is not None
        assert result is sentinel
