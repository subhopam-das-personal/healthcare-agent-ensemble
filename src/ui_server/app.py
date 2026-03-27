"""CDS UI — Streamlit frontend for the Healthcare Agent Ensemble."""

import base64
import json
import os
import sys
import uuid

import httpx
import streamlit as st
from dotenv import load_dotenv
from httpx_sse import connect_sse

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from a2a.types import (
    Message as A2AMessage,
    SendStreamingMessageResponse,
    SendStreamingMessageSuccessResponse,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)


# ── Config ────────────────────────────────────────────────────────────────────

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

SKILL_LABELS = {
    "comprehensive-clinical-review": "Comprehensive Clinical Review",
    "differential-diagnosis": "Differential Diagnosis",
    "quick-drug-check": "Quick Drug Check",
}

SKILL_STEPS = {
    "comprehensive-clinical-review": [
        "Fetching patient FHIR record",
        "Resolving medications via RxNav",
        "Differential Diagnosis (Claude)",
        "Drug Interaction Analysis (Claude)",
        "Integrated Clinical Assessment (Claude)",
    ],
    "differential-diagnosis": [
        "Fetching patient FHIR record",
        "Differential Diagnosis (Claude)",
    ],
    "quick-drug-check": [
        "Fetching patient FHIR record",
        "Resolving medications via RxNav",
        "Drug Interaction Analysis (Claude)",
    ],
}

SECTION_HEADERS = [
    "Differential Diagnosis",
    "Drug Interaction Analysis",
    "Integrated Clinical Assessment",
]


# ── SSE parsing ───────────────────────────────────────────────────────────────

def _text_from_a2a_event(response: SendStreamingMessageResponse) -> str | None:
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


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _render_ddx(parsed: dict) -> None:
    differentials = parsed.get("differentials", [])
    if differentials:
        rows = []
        for d in differentials:
            conf = d.get("confidence", "")
            badge = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf, "⚪")
            rows.append({
                "Rank": d.get("rank", ""),
                "Diagnosis": d.get("diagnosis", ""),
                "Confidence": f"{badge} {conf}",
                "Key Evidence": ", ".join(d.get("supporting_evidence", [])[:2]),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    red_flags = parsed.get("red_flags", [])
    if red_flags:
        st.error("⚠️ Red Flags: " + " · ".join(red_flags))

    reasoning = parsed.get("reasoning_summary")
    if reasoning:
        st.info(f"💭 {reasoning}")


def _render_drug(parsed: dict) -> None:
    interactions = parsed.get("interactions", [])
    overall = parsed.get("overall_risk_level", "")
    if overall:
        badge = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}.get(overall, "⚪")
        st.metric("Overall Risk", f"{badge} {overall}")

    for ix in interactions:
        sev = ix.get("severity", "Low")
        color = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}.get(sev, "⚪")
        drugs = " + ".join(ix.get("drug_pair", []))
        with st.expander(f"{color} {drugs} — {sev}"):
            st.write(ix.get("description", ""))
            rec = ix.get("recommendation")
            if rec:
                st.caption(f"**Recommendation:** {rec}")

    concerns = parsed.get("patient_specific_concerns", [])
    if concerns:
        st.warning("Patient-specific concerns:\n" + "\n".join(f"• {c}" for c in concerns))


def _render_synthesis(parsed: dict) -> None:
    key_findings = parsed.get("key_findings", [])
    if key_findings:
        st.subheader("Key Findings")
        for f in key_findings:
            st.write(f"• {f}")

    summary = parsed.get("assessment_summary")
    if summary:
        st.info(f"📋 {summary}")

    steps = parsed.get("recommended_next_steps", [])
    if steps:
        st.subheader("Recommended Next Steps")
        for s in steps:
            st.write(f"→ {s}")

    red_flags = parsed.get("red_flags", [])
    if red_flags:
        st.error("⚠️ Red Flags: " + " · ".join(red_flags))

    care_gaps = parsed.get("care_gaps", [])
    if care_gaps:
        st.warning("Care Gaps:\n" + "\n".join(f"• {g}" for g in care_gaps))


def _try_render_section(name: str, raw: str) -> None:
    raw = raw.strip()
    if not raw:
        return
    parsed = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        for marker in ("```json", "```"):
            if marker in raw:
                try:
                    parsed = json.loads(raw.split(marker)[1].split("```")[0].strip())
                    break
                except (json.JSONDecodeError, IndexError):
                    pass

    icon = {"Differential Diagnosis": "🔬", "Drug Interaction Analysis": "💊",
             "Integrated Clinical Assessment": "📋"}.get(name, "📄")
    with st.expander(f"{icon} {name}", expanded=True):
        if parsed and not parsed.get("raw_response") and not parsed.get("error"):
            if name == "Differential Diagnosis":
                _render_ddx(parsed)
            elif name == "Drug Interaction Analysis":
                _render_drug(parsed)
            elif name == "Integrated Clinical Assessment":
                _render_synthesis(parsed)
        else:
            st.markdown(raw)


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e0e3ea; }
.stExpander { border: 1px solid #e0e3ea !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 Clinical Decision Support")
    st.caption("AI-powered · Healthcare Agent Ensemble")
    st.divider()

    patient_id = st.text_input(
        "Patient ID (FHIR UUID)",
        value="fa064acf-b7f1-4279-83d3-7a94686da7ba",
        placeholder="Enter FHIR patient UUID…",
    )

    skill = st.radio(
        "Analysis Type",
        options=list(SKILL_LABELS.keys()),
        format_func=lambda s: SKILL_LABELS[s],
    )

    symptoms = st.text_area("Symptoms (optional)", height=72, placeholder="e.g. chest pain, shortness of breath")
    proposed_meds = st.text_input("Proposed Medications", placeholder="comma-separated")

    st.divider()
    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)
    st.caption("⚠️ AI outputs require review by a licensed provider.")


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("### 🤖 Clinical Decision Support Assistant")

if not run_btn and not st.session_state.analysis_done:
    st.info("Configure patient details in the sidebar and click **▶ Run Analysis** to begin.")
    st.stop()


# ── Run analysis ──────────────────────────────────────────────────────────────

if run_btn:
    if not patient_id.strip():
        st.error("Patient ID is required.")
        st.stop()

    st.session_state.messages = []
    st.session_state.analysis_context = ""
    st.session_state.analysis_done = False

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/stream",
        "params": {
            "message": {
                "role": "user",
                "messageId": str(uuid.uuid4()),
                "parts": [{"kind": "text", "text": json.dumps({
                    "patient_id": patient_id.strip(),
                    "skill": skill,
                    "symptoms": symptoms,
                    "proposed_medications": proposed_meds,
                })}],
            }
        },
    }

    sections: dict[str, str] = {}
    current_section: str | None = None
    error_text: str | None = None

    # Agent timeline
    with st.status("🤖 Agent running…", expanded=True) as status_box:
        steps = SKILL_STEPS.get(skill, [])
        step_placeholders = {s: st.empty() for s in steps}
        for s in steps:
            step_placeholders[s].markdown(f"⬜ {s}")

        narration_log: list[str] = []

        def _update_step_from_narration(msg: str) -> None:
            import re
            m = re.search(r"Step (\d+)/\d+:", msg)
            if m:
                idx = int(m.group(1)) - 1
                for i, s in enumerate(steps):
                    if i < idx:
                        step_placeholders[s].markdown(f"✅ {s}")
                    elif i == idx:
                        step_placeholders[s].markdown(f"⟳ **{s}**")
                    else:
                        step_placeholders[s].markdown(f"⬜ {s}")

        try:
            timeout = httpx.Timeout(connect=10, read=300, write=30, pool=30)
            with httpx.Client(timeout=timeout) as client:
                with connect_sse(client, "POST", A2A_AGENT_URL, json=payload) as event_source:
                    for sse in event_source.iter_sse():
                        if not sse.data:
                            continue
                        try:
                            resp = SendStreamingMessageResponse.model_validate_json(sse.data)
                        except Exception:
                            continue
                        text = _text_from_a2a_event(resp)
                        if not text:
                            continue

                        if text.startswith("status:"):
                            msg = text[7:].strip()
                            narration_log.append(msg)
                            _update_step_from_narration(msg)
                            continue

                        if "**Error:**" in text:
                            error_text = text.replace("\n\n**Error:** ", "").strip()
                            break

                        # Route content to section
                        for header in SECTION_HEADERS:
                            if f"## {header}" in text:
                                current_section = header
                                sections[header] = text.split(f"## {header}", 1)[1]
                                break
                        else:
                            if current_section:
                                sections[current_section] = sections.get(current_section, "") + text

            # Mark all steps done
            for s in steps:
                step_placeholders[s].markdown(f"✅ {s}")

            if error_text:
                status_box.update(label="❌ Analysis failed", state="error", expanded=True)
            else:
                status_box.update(label="✅ Analysis complete", state="complete", expanded=False)

        except httpx.ConnectError:
            status_box.update(label="❌ Cannot connect to agent", state="error")
            st.error(f"Cannot connect to A2A agent at {A2A_AGENT_URL}. Is it running?")
            st.stop()
        except Exception as e:
            status_box.update(label="❌ Error", state="error")
            st.error(f"Error: {e}")
            st.stop()

    if error_text:
        st.error(f"⚠️ {error_text}")
    else:
        # Render output sections
        for header in SECTION_HEADERS:
            if header in sections:
                _try_render_section(header, sections[header])

        # Build context for follow-up chat
        raw_context = "\n\n".join(f"## {k}\n{v}" for k, v in sections.items())
        try:
            st.session_state.analysis_context = base64.b64encode(
                raw_context[:6000].encode()
            ).decode()
        except Exception:
            pass

        st.session_state.analysis_done = True


# ── Follow-up chat ────────────────────────────────────────────────────────────

if st.session_state.analysis_done:
    st.divider()
    st.markdown("#### 💬 Follow-up Questions")

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a follow-up question about this patient…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            try:
                from shared.claude_client import CLAUDE_MODEL, get_client
                client = get_client()
                prior = base64.b64decode(st.session_state.analysis_context).decode("utf-8") \
                    if st.session_state.analysis_context else ""
                system = (
                    "You are a clinical decision support assistant. "
                    "A clinician has reviewed an AI-generated analysis and has a follow-up question. "
                    "Answer concisely and clinically. Always remind the user that AI outputs require review by a licensed provider."
                )
                user_msg = f"Patient ID: {patient_id}\n\nPrior analysis:\n{prior[:4000]}\n\nQuestion: {prompt}"

                response_text = ""
                with client.messages.stream(
                    model=CLAUDE_MODEL,
                    max_tokens=1024,
                    system=system,
                    messages=[{"role": "user", "content": user_msg}],
                ) as stream:
                    response_placeholder = st.empty()
                    for chunk in stream.text_stream:
                        response_text += chunk
                        response_placeholder.markdown(response_text + "▌")
                    response_placeholder.markdown(response_text)

                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                st.error(f"Chat error: {e}")
