"""CDS UI — Streamlit frontend for the Healthcare Agent Ensemble."""

import base64
import json
import os
import pathlib
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
        "Matching clinical trials",
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
    "Clinical Trials",
    "Differential Diagnosis",
    "Drug Interaction Analysis",
    "Integrated Clinical Assessment",
]

# ── Demo Mode ─────────────────────────────────────────────────────────────────

_DEMO_BUNDLE_PATH = pathlib.Path(__file__).parent.parent.parent / "demo" / "patient_serotonin_syndrome.json"
_DEMO_PATIENT_ID = "demo-serotonin-patient-001"

def _load_demo_patient_json() -> str:
    """Return the serotonin syndrome demo patient bundle as a JSON string."""
    try:
        return _DEMO_BUNDLE_PATH.read_text()
    except OSError:
        return ""

_STATIC_DRUG_CHECKER_MD = """
**Patient:** Margaret Alvarez, 55F, post-surgical

**Active Medications:**
- Sertraline 100 MG Oral Tablet
- Tramadol 50 MG Oral Tablet
- Linezolid 600 MG Oral Tablet

---

**Alerts:**

⚠️ **Low:** Monitor renal function
*(Tramadol — caution if CrCl < 30 mL/min)*

---

✅ **No critical drug–drug interactions detected.**

*Standard pairwise checker evaluates each drug pair independently.*
"""


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

def _render_trials(trials: list) -> None:
    if not trials:
        st.info("No recruiting trials found matching this patient's conditions.")
        return
    for trial in trials:
        phase = trial.get("phase", "N/A")
        status = trial.get("status", "")
        status_badge = "🟢" if status == "RECRUITING" else "⚪"
        nct_id = trial.get("nct_id", "")
        title = trial.get("title", nct_id)
        with st.expander(f"{status_badge} {nct_id} — {title}", expanded=False):
            cols = st.columns(3)
            cols[0].metric("Phase", phase)
            cols[1].metric("Status", status)
            cols[2].metric("Sponsor", trial.get("sponsor", "—")[:30])

            conditions = trial.get("conditions", [])
            if conditions:
                st.caption("**Conditions:** " + " · ".join(conditions[:5]))

            age_range = " – ".join(filter(None, [trial.get("min_age"), trial.get("max_age")]))
            gender = trial.get("gender", "ALL")
            if age_range or gender != "ALL":
                st.caption(f"**Eligibility:** {age_range or 'Any age'} · {gender}")

            locations = trial.get("locations", [])
            if locations:
                st.caption("**Sites:** " + " · ".join(locations))

            summary = trial.get("summary", "")
            if summary:
                st.write(summary)

            elig = trial.get("eligibility_summary", "")
            if elig:
                st.caption("**Eligibility criteria (excerpt):**")
                st.code(elig[:400], language=None)

            st.markdown(
                f"[View on ClinicalTrials.gov](https://clinicaltrials.gov/study/{nct_id})"
            )


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

    icon = {
        "Clinical Trials": "🧪",
        "Differential Diagnosis": "🔬",
        "Drug Interaction Analysis": "💊",
        "Integrated Clinical Assessment": "📋",
    }.get(name, "📄")
    with st.expander(f"{icon} {name}", expanded=True):
        if name == "Clinical Trials":
            if isinstance(parsed, list):
                _render_trials(parsed)
            elif isinstance(parsed, dict) and not parsed.get("error"):
                _render_trials(parsed.get("trials", []))
            else:
                st.markdown(raw)
        elif isinstance(parsed, dict) and not parsed.get("raw_response") and not parsed.get("error"):
            if name == "Differential Diagnosis":
                _render_ddx(parsed)
            elif name == "Drug Interaction Analysis":
                _render_drug(parsed)
            elif name == "Integrated Clinical Assessment":
                _render_synthesis(parsed)
        else:
            st.markdown(raw)


# ── Page setup ────────────────────────────────────────────────────────────────

MCP_SERVER_URL = os.environ.get(
    "MCP_SERVER_URL",
    (
        "https://" + os.environ["RAILWAY_SERVICE_HEALTHCARE_AGENT_ENSEMBLE_URL"]
        if "RAILWAY_SERVICE_HEALTHCARE_AGENT_ENSEMBLE_URL" in os.environ
        else "http://localhost:8000"
    ),
)
MCP_API_KEY = os.environ.get("MCP_API_KEY", "")


def _mcp_call(tool: str, arguments: dict) -> dict:
    """Synchronous JSON-RPC call to the MCP server."""
    import uuid as _uuid
    payload = {
        "jsonrpc": "2.0",
        "id": str(_uuid.uuid4()),
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    headers = {"Content-Type": "application/json"}
    if MCP_API_KEY:
        headers["X-API-Key"] = MCP_API_KEY
    try:
        resp = httpx.post(
            f"{MCP_SERVER_URL}/mcp",
            json=payload,
            headers=headers,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("result", {}).get("content", [])
        if content:
            return json.loads(content[0].get("text", "{}"))
        return data.get("result", {})
    except Exception as e:
        return {"error": str(e)}


st.set_page_config(
    page_title="Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
/* Use Streamlit's own CSS vars so colors follow the active theme automatically */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.2);
}
.stExpander {
    border: 1px solid rgba(128,128,128,0.25) !important;
    border-radius: 8px !important;
}
/* Ensure dataframe text is readable in both themes */
[data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ── Tabs ──────────────────────────────────────────────────────────────────────

_tab_review, _tab_query = st.tabs(["🏥 Clinical Review", "🔍 Find Patients"])


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 Clinical Decision Support")
    st.caption("AI-powered · Healthcare Agent Ensemble")
    st.divider()

    demo_mode = st.checkbox("🎬 Demo Mode", value=False, help="Loads the serotonin syndrome near-miss scenario")

    st.divider()

    if demo_mode:
        patient_id = _DEMO_PATIENT_ID
        skill = "comprehensive-clinical-review"
        symptoms = ""
        proposed_meds = ""
        st.info(
            "**Demo patient loaded**\n\n"
            "Margaret Alvarez · 55F\n\n"
            "Sertraline + Tramadol + Linezolid\n\n"
            "*Synthetic patient — constructed from published FDA adverse event patterns.*"
        )
    else:
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


# ── Clinical Review tab ───────────────────────────────────────────────────────

with _tab_review:
    if demo_mode:
        st.markdown("### 🎬 Near-Miss Reconstruction Demo")
        st.warning(
            "⚠️ **Serotonin syndrome is underrecognized:** 85% of general practitioners were unaware it "
            "existed as a diagnosis *(Mackay et al., Br J Gen Pract 1999)*. "
            "Annual incidence estimated at 14–16 cases/million per year, likely higher due to underreporting."
        )
        st.caption(
            "The patient below is a synthetic representation constructed from published FDA adverse event patterns — not a real individual's record."
        )
    else:
        st.markdown("### 🤖 Clinical Decision Support Assistant")

    if not run_btn and not st.session_state.analysis_done:
        if demo_mode:
            st.info("Click **▶ Run Analysis** in the sidebar to run the serotonin syndrome near-miss demo.")
        else:
            st.info("Configure patient details in the sidebar and click **▶ Run Analysis** to begin.")
        st.stop()

    # ── Run analysis ──────────────────────────────────────────────────────────

    if run_btn:
        if not patient_id.strip():
            st.error("Patient ID is required.")
            st.stop()

        st.session_state.messages = []
        st.session_state.analysis_context = ""
        st.session_state.analysis_done = False

        msg_data: dict = {
            "patient_id": patient_id.strip(),
            "skill": skill,
            "symptoms": symptoms,
            "proposed_medications": proposed_meds,
        }
        if demo_mode:
            demo_json = _load_demo_patient_json()
            if demo_json:
                msg_data["patient_json"] = demo_json
            else:
                st.warning("Demo patient file not found — running without pre-loaded patient data.")
                st.stop()

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": str(uuid.uuid4()),
                    "parts": [{"kind": "text", "text": json.dumps(msg_data)}],
                }
            },
        }

        sections: dict[str, str] = {}
        current_section: str | None = None
        error_text: str | None = None

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

                            for header in SECTION_HEADERS:
                                if f"## {header}" in text:
                                    current_section = header
                                    sections[header] = text.split(f"## {header}", 1)[1]
                                    break
                            else:
                                if current_section:
                                    sections[current_section] = sections.get(current_section, "") + text

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
        elif demo_mode:
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("#### Standard Pairwise Drug Checker *(simulated)*")
                st.markdown(_STATIC_DRUG_CHECKER_MD)
            with col_right:
                st.markdown("#### Healthcare Agent Ensemble")
                for header in SECTION_HEADERS:
                    if header in sections:
                        _try_render_section(header, sections[header])
        else:
            for header in SECTION_HEADERS:
                if header in sections:
                    _try_render_section(header, sections[header])

        raw_context = "\n\n".join(f"## {k}\n{v}" for k, v in sections.items())
        try:
            st.session_state.analysis_context = base64.b64encode(
                raw_context[:6000].encode()
            ).decode()
        except Exception:
            pass
        st.session_state.analysis_done = True

    # ── Follow-up chat ────────────────────────────────────────────────────────

    if st.session_state.analysis_done:
        st.divider()
        st.markdown("#### 💬 Follow-up Questions")

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


# ── Find Patients tab (NL Query Engine) ───────────────────────────────────────

with _tab_query:
    st.markdown("### 🔍 Find Patients")
    st.caption(
        "Search the indexed patient cohort in plain English. "
        "The engine extracts medical entities, expands via ontology, and runs a validated SQL query — "
        "falling back to text similarity if needed."
    )

    if "query_result" not in st.session_state:
        st.session_state.query_result = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    query_input = st.text_input(
        "Clinical question",
        placeholder="e.g. patients with heart failure on ACE inhibitors who have elevated creatinine",
        key="nl_query_input",
    )
    search_btn = st.button("🔍 Search", type="primary", key="nl_search_btn")

    # Example queries
    with st.expander("Example queries", expanded=False):
        examples = [
            "patients with diabetes on metformin",
            "patients with hypertension who have elevated creatinine",
            "patients taking anticoagulants with atrial fibrillation",
            "patients with heart failure on beta blockers",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}"):
                st.session_state.last_query = ex
                st.rerun()

    # Use last_query if set by example button
    effective_query = st.session_state.last_query or query_input
    if st.session_state.last_query:
        st.session_state.last_query = ""

    if search_btn and effective_query.strip():
        with st.spinner("Searching patient cohort…"):
            result = _mcp_call("nl_query_patients", {"question": effective_query.strip()})
        st.session_state.query_result = result

    if st.session_state.query_result:
        result = st.session_state.query_result

        if "error" in result:
            st.error(f"Query failed: {result['error']}")
        else:
            count = result.get("count", 0)
            mode = result.get("mode", "")
            mode_label = {"structured": "✅ SQL", "text_fallback": "🔤 Text search"}.get(mode, mode)
            patients = result.get("patients", [])
            expansion = result.get("expansion", {})

            col1, col2 = st.columns([3, 1])
            col1.metric("Patients found", count)
            col2.metric("Search mode", mode_label)

            # Ontological expansion panel
            if any(v for v in expansion.values() if isinstance(v, list) and v):
                with st.expander("🧬 Ontological expansion", expanded=True):
                    cols = st.columns(3)
                    if expansion.get("icd10_codes"):
                        cols[0].markdown("**ICD-10 codes matched**")
                        cols[0].write(", ".join(expansion["icd10_codes"][:10]))
                    if expansion.get("drug_classes"):
                        cols[1].markdown("**Drug classes matched**")
                        cols[1].write(", ".join(expansion["drug_classes"][:10]))
                    if expansion.get("loinc_codes"):
                        cols[2].markdown("**LOINC codes matched**")
                        cols[2].write(", ".join(expansion["loinc_codes"][:10]))

            # SQL transparency
            if result.get("sql"):
                with st.expander("🔎 Generated SQL", expanded=False):
                    st.code(result["sql"], language="sql")

            # Results table
            if patients:
                st.divider()
                rows = []
                for p in patients:
                    birth = p.get("birth_date", "")
                    age = ""
                    if birth:
                        try:
                            from datetime import date
                            b = date.fromisoformat(str(birth)[:10])
                            age = str((date.today() - b).days // 365)
                        except Exception:
                            pass
                    rows.append({
                        "Name": f"{p.get('given_name', '')} {p.get('family_name', '')}".strip(),
                        "Age": age,
                        "Gender": (p.get("gender") or "").capitalize(),
                        "Patient ID": p.get("id", ""),
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)

                st.caption(
                    "Click a Patient ID to use it in the **Clinical Review** tab for full AI analysis."
                )
            else:
                st.info("No patients found. Try broadening your query or run the indexer first.")

    elif not search_btn:
        st.info(
            "Enter a clinical question above and click **Search**. "
            "The patient cohort is indexed from the SMART Health IT FHIR sandbox — "
            "run `python -m src.ddm.indexer` to populate it."
        )
