"""CDS UI Server — FastAPI + SSE streaming frontend for the A2A agent."""

import base64
import json
import logging
import os
import sys
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from httpx_sse import aconnect_sse
from a2a.types import (
    SendStreamingMessageResponse, SendStreamingMessageSuccessResponse,
    TaskArtifactUpdateEvent, TaskStatusUpdateEvent, Message as A2AMessage,
    TaskState, TextPart,
)
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
PORT = int(os.environ.get("PORT", os.environ.get("UI_PORT", 7000)))


def _resolve_a2a_url() -> str:
    url = os.environ.get("A2A_AGENT_URL")
    if url:
        return url
    host = os.environ.get("RAILWAY_SERVICE_A2A_AGENT_URL")
    if host:
        scheme = "https" if "railway.app" in host else "http"
        return f"{scheme}://{host}"
    return "http://localhost:9999"


A2A_AGENT_URL = _resolve_a2a_url()

app = FastAPI(title="CDS UI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _text_from_a2a_event(response: SendStreamingMessageResponse) -> str | None:
    """Extract streamed text from an A2A SendStreamingMessageResponse.

    Handles artifact-update (streaming chunks), message (final), and
    status-update (failed + working states) events.
    """
    root = response.root
    if not isinstance(root, SendStreamingMessageSuccessResponse):
        return None
    result = root.result
    if isinstance(result, TaskArtifactUpdateEvent):
        for part in (result.artifact.parts or []):
            inner = part.root if hasattr(part, "root") else part
            if isinstance(inner, TextPart) and inner.text:
                return inner.text
    elif isinstance(result, A2AMessage):
        for part in (result.parts or []):
            inner = part.root if hasattr(part, "root") else part
            if isinstance(inner, TextPart) and inner.text:
                return inner.text
    elif isinstance(result, TaskStatusUpdateEvent):
        status = result.status
        if status.state == TaskState.failed:
            msg = status.message
            if msg:
                for part in (msg.parts or []):
                    inner = part.root if hasattr(part, "root") else part
                    if isinstance(inner, TextPart) and inner.text:
                        return f"\n\n**Error:** {inner.text}"
        elif status.state == TaskState.working:
            msg = status.message
            if msg:
                for part in (msg.parts or []):
                    inner = part.root if hasattr(part, "root") else part
                    if isinstance(inner, TextPart) and inner.text:
                        return f"status:{inner.text}"
    return None


def _build_a2a_payload(params: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/stream",
        "params": {
            "message": {
                "role": "user",
                "messageId": str(uuid.uuid4()),
                "parts": [{"kind": "text", "text": json.dumps(params)}],
            }
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/stream")
async def stream(
    request: Request,
    patient_id: str = "",
    skill: str = "comprehensive-clinical-review",
    symptoms: str = "",
    proposed_medications: str = "",
):
    if not patient_id.strip():
        return JSONResponse({"error": "patient_id is required"}, status_code=400)

    valid_skills = {"comprehensive-clinical-review", "differential-diagnosis", "quick-drug-check"}
    if skill not in valid_skills:
        skill = "comprehensive-clinical-review"

    params = {
        "patient_id": patient_id.strip(),
        "skill": skill,
        "symptoms": symptoms,
        "proposed_medications": proposed_medications,
    }

    payload = _build_a2a_payload(params)

    async def event_gen():
        try:
            timeout = httpx.Timeout(connect=10, read=300, write=30, pool=30)
            async with httpx.AsyncClient(timeout=timeout) as http_client:
                async with aconnect_sse(http_client, "POST", A2A_AGENT_URL, json=payload) as event_source:
                    async for sse in event_source.aiter_sse():
                        if await request.is_disconnected():
                            return
                        if not sse.data:
                            continue
                        try:
                            resp = SendStreamingMessageResponse.model_validate_json(sse.data)
                        except Exception:
                            continue
                        text = _text_from_a2a_event(resp)
                        if text:
                            yield {"data": text}
        except httpx.ConnectError:
            yield {"data": f"\n\n**Error:** Cannot connect to A2A agent at {A2A_AGENT_URL}"}
        except Exception as e:
            logger.error(f"A2A stream error: {e}", exc_info=True)
            yield {"data": f"\n\n**Error:** {e}"}

    return EventSourceResponse(event_gen())


@app.get("/chat")
async def chat(
    request: Request,
    question: str = "",
    patient_id: str = "",
    context: str = "",
):
    """Follow-up chat: stream a Claude answer given prior analysis context."""
    if not question.strip():
        return JSONResponse({"error": "question is required"}, status_code=400)

    try:
        prior = base64.b64decode(context).decode("utf-8") if context else ""
    except Exception:
        prior = ""

    system = (
        "You are a clinical decision support assistant. "
        "A clinician has just reviewed an AI-generated analysis for a patient and has a follow-up question. "
        "Answer concisely and clinically. Always remind the user that AI outputs require review by a licensed provider."
    )
    user_msg = f"Patient ID: {patient_id}\n\nPrior analysis summary:\n{prior[:4000]}\n\nFollow-up question: {question}"

    async def event_gen():
        try:
            from shared.claude_client import get_async_client, CLAUDE_MODEL
            client = get_async_client()
            async with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            ) as stream:
                async for chunk in stream.text_stream:
                    if await request.is_disconnected():
                        return
                    yield {"data": chunk}
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            yield {"data": f"\n\n**Error:** {e}"}

    return EventSourceResponse(event_gen())


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting CDS UI Server on port {PORT} → A2A agent: {A2A_AGENT_URL}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
