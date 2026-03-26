"""CDS UI Server — FastAPI + SSE streaming frontend for the A2A executor."""

import asyncio
import json
import logging
import os
import sys
import uuid
from asyncio import Queue
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a2a_agent.executor import CDSAgentExecutor
from shared.fhir_client import DEFAULT_FHIR_BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
PORT = int(os.environ.get("PORT", os.environ.get("UI_PORT", 7000)))

app = FastAPI(title="CDS UI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_executor = CDSAgentExecutor()


# ── A2A event text extraction ─────────────────────────────────────────────────

def _extract_text(event) -> str:
    """Extract plain text from any A2A event type."""
    try:
        from a2a.types import Message
        if isinstance(event, Message):
            for part in event.parts:
                inner = part.root if hasattr(part, "root") else part
                if hasattr(inner, "text"):
                    return inner.text or ""
    except Exception:
        pass
    return ""


# ── SSE queue bridge ──────────────────────────────────────────────────────────

class SSEQueue:
    """Bridges CDSAgentExecutor events into an asyncio Queue for SSE consumption."""

    def __init__(self):
        self._q: Queue = Queue()

    async def enqueue_event(self, event) -> None:
        text = _extract_text(event)
        if text:
            await self._q.put(text)

    async def drain(self):
        while True:
            chunk = await self._q.get()
            if chunk is None:
                return
            yield chunk

    async def close(self):
        await self._q.put(None)


# ── A2A RequestContext bridge ─────────────────────────────────────────────────

from a2a.types import Message, Part, TextPart


def _build_context(params: dict):
    """Build a minimal RequestContext using real A2A types so Pydantic validates."""
    msg = Message(
        role="user",
        messageId=str(uuid.uuid4()),
        parts=[Part(root=TextPart(text=json.dumps(params)))],
    )

    class _Context:
        def __init__(self, message: Message):
            self.message = message
            self.current_task = None

    return _Context(msg)


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
        "fhir_base_url": DEFAULT_FHIR_BASE_URL,
        "symptoms": symptoms,
        "skill": skill,
        "proposed_medications": proposed_medications,
        "access_token": "",
    }

    queue = SSEQueue()
    ctx = _build_context(params)

    async def run():
        try:
            await _executor.execute(ctx, queue)
        except Exception as e:
            logger.error(f"Executor error: {e}", exc_info=True)
            await queue._q.put(f"\n\n**Error:** {e}")
        finally:
            await queue.close()

    async def event_gen():
        task = asyncio.create_task(run())
        last_keepalive = asyncio.get_event_loop().time()
        try:
            async for chunk in queue.drain():
                if await request.is_disconnected():
                    logger.info("Client disconnected — cancelling executor task")
                    task.cancel()
                    return
                yield {"data": chunk}
                now = asyncio.get_event_loop().time()
                if now - last_keepalive > 15:
                    yield {"comment": "keepalive"}
                    last_keepalive = now
        finally:
            task.cancel()

    return EventSourceResponse(event_gen())


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting CDS UI Server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
