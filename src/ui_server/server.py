"""CDS UI Server — FastAPI + SSE streaming frontend for the A2A agent."""

import json
import logging
import os
import sys
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
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
    # Railway injects the service hostname without a scheme
    host = os.environ.get("RAILWAY_SERVICE_A2A_AGENT_URL")
    if host:
        scheme = "https" if "railway.app" in host else "http"
        return f"{scheme}://{host}"
    return "http://localhost:9999"

A2A_AGENT_URL = _resolve_a2a_url()

app = FastAPI(title="CDS UI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _a2a_text_from_event(event_data: dict) -> str | None:
    """Extract streamed text from an A2A SSE event (Message events only)."""
    result = event_data.get("result", {})
    # Message event has "parts" at top level; TaskStatusUpdateEvent has "status"
    parts = result.get("parts")
    if not parts:
        return None
    for part in parts:
        text = part.get("text", "")
        if text:
            return text
    return None


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

    payload = {
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

    async def event_gen():
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream(
                    "POST",
                    A2A_AGENT_URL,
                    json=payload,
                    headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
                ) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        yield {"data": f"\n\n**Error:** A2A agent returned {resp.status_code}: {err[:200].decode(errors='replace')}"}
                        return

                    async for line in resp.aiter_lines():
                        if await request.is_disconnected():
                            return
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            event_data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        text = _a2a_text_from_event(event_data)
                        if text:
                            yield {"data": text}
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to A2A agent at {A2A_AGENT_URL}: {e}")
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

    import base64
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
