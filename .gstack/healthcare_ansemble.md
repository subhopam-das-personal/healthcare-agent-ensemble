# Winning the Healthcare AI Hackathon: A Technical Blueprint

**The "Agents Assemble" hackathon ($25K in prizes, deadline May 11, 2026) rewards submissions that combine MCP tools + A2A agents on the Prompt Opinion platform to solve real clinical pain points using GenAI.** The three judging criteria — AI Factor, Potential Impact, and Feasibility — are equally weighted. Your highest-leverage move: build a focused Clinical Decision Support agent that uses Claude as a reasoning engine over FHIR patient data, exposed as MCP "Superpowers" and orchestrated by an A2A agent, all published to the Prompt Opinion Marketplace with SHARP-on-MCP context propagation. Below is everything you need to build it.

---

## 1. Building MCP servers in Python with FastMCP

The official Python MCP SDK (`pip install "mcp[cli]"`, v1.26.0, Python 3.10+) provides two API tiers. **FastMCP** is the high-level API you want for rapid hackathon development — it auto-generates JSON Schema `inputSchema` from type hints and docstrings.

### Core server pattern

```python
from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("ClinicalDecisionSupport")

@mcp.tool()
async def check_drug_interactions(
    medications: list[str],
    patient_id: str,
    ctx: Context
) -> str:
    """Check drug-drug interactions for a patient's medication list.

    Args:
        medications: List of RxNorm codes for current medications
        patient_id: FHIR Patient resource ID
    """
    ctx.info(f"Checking {len(medications)} medications for patient {patient_id}")
    # ctx is injected automatically, NOT exposed in the tool schema
    interactions = await lookup_interactions(medications)
    return json.dumps(interactions)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")  # Serves on http://localhost:8000/mcp
```

The decorator extracts the function name as `tool.name`, the first docstring line as `tool.description`, and type hints as JSON Schema properties. Parameters with defaults become optional. The `Context` parameter is injected by the framework and excluded from `inputSchema`.

### Transport options

| Transport | Arg | Endpoint | When to use |
|---|---|---|---|
| stdio | `"stdio"` | stdin/stdout | Claude Desktop, local dev |
| SSE | `"sse"` | `/sse` + `/messages/` | Legacy (deprecated) |
| **Streamable HTTP** | `"streamable-http"` | `/mcp` | **Production, Prompt Opinion** |

For the hackathon, use **streamable-http** since Prompt Opinion's hosted MCP endpoints use `/mcp` paths. You can mount into an existing ASGI app:

```python
from starlette.applications import Starlette
from starlette.routing import Mount
app = Starlette(routes=[Mount("/mcp", app=mcp.streamable_http_app())])
```

### JSON-RPC protocol: how tools are discovered and invoked

MCP uses JSON-RPC 2.0. Clients call `tools/list` to discover available tools, receiving an array of `{name, description, inputSchema}` objects. They invoke with `tools/call`, sending `{name, arguments}` and receiving `{content: [{type: "text", text: "..."}], isError: false}`. The full session lifecycle is: `initialize` → `initialized` notification → `tools/list` → `tools/call` (repeated). **Critical gotcha**: never use `print()` in stdio servers — it corrupts the JSON-RPC stream. Use `logging` (writes to stderr).

### Low-level API for custom schemas

When you need exact control over `inputSchema` (e.g., for SHARP context fields):

```python
from mcp.server.lowlevel import Server
import mcp.types as types

server = Server("cds-server")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [types.Tool(
        name="get_patient_summary",
        description="Generate AI clinical summary from FHIR data",
        inputSchema={
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "FHIR Patient ID"},
                "fhir_base_url": {"type": "string", "description": "FHIR server base URL"},
                "access_token": {"type": "string", "description": "SMART on FHIR bearer token"}
            },
            "required": ["patient_id", "fhir_base_url"]
        }
    )]
```

---

## 2. A2A protocol: building a compliant agent in Python

Google's Agent-to-Agent protocol uses JSON-RPC 2.0 over HTTP POST. The Python SDK is `pip install "a2a-sdk[http-server]"` (v0.3.22, implements spec v0.3). The three core concepts: **Agent Cards** for discovery, **Tasks** for lifecycle management, and **Messages/Parts** for communication.

### Agent Card (served at `/.well-known/agent.json`)

```json
{
  "name": "Clinical Decision Support Agent",
  "description": "AI-powered CDS agent that performs differential diagnosis, drug interaction checks, and patient summarization using FHIR data",
  "url": "http://localhost:9999/",
  "version": "1.0.0",
  "provider": {"organization": "YourTeam", "url": "https://yourteam.dev"},
  "capabilities": {"streaming": true, "pushNotifications": false},
  "defaultInputModes": ["text/plain", "application/json"],
  "defaultOutputModes": ["text/plain", "application/json"],
  "skills": [
    {
      "id": "differential-diagnosis",
      "name": "Differential Diagnosis Generator",
      "description": "Analyzes patient symptoms, vitals, and labs to produce ranked differential diagnoses with reasoning",
      "tags": ["diagnosis", "clinical", "fhir"],
      "examples": ["Generate differential diagnosis for a 58yo male with chest pain and elevated troponin"]
    },
    {
      "id": "drug-interaction-check",
      "name": "Intelligent Drug Interaction Checker",
      "description": "Checks medication interactions with AI-powered clinical significance interpretation",
      "tags": ["medications", "safety", "pharmacology"],
      "examples": ["Check interactions for warfarin, metformin, and lisinopril"]
    }
  ]
}
```

### Task state machine

Tasks transition through: `submitted` → `working` → `completed` | `failed` | `input-required`. The `input-required` state pauses execution and asks the client for more information — useful for multi-turn clinical workflows where the agent needs additional patient data. Terminal states are `completed`, `failed`, `canceled`, and `unknown`.

### Python A2A server implementation

```python
# executor.py
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message, new_task

class CDSAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)
        
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(TaskState.working, "Analyzing patient data...")
        
        user_input = context.get_user_input()
        # Call your CDS logic (Claude API, FHIR fetches, etc.)
        result = await run_clinical_reasoning(user_input)
        
        await event_queue.enqueue_event(new_agent_text_message(result))
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")

# server.py
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

agent_card = AgentCard(
    name="CDS Agent", url="http://localhost:9999/", version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    skills=[AgentSkill(id="ddx", name="Differential Diagnosis", 
                       description="AI-powered differential diagnosis")]
)

handler = DefaultRequestHandler(
    agent_executor=CDSAgentExecutor(), task_store=InMemoryTaskStore()
)
app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

if __name__ == "__main__":
    uvicorn.run(app.build(), host="0.0.0.0", port=9999)
```

The `DefaultRequestHandler` automatically manages JSON-RPC routing, task lifecycle, and SSE streaming. Clients interact via `message/send` (synchronous), `message/stream` (SSE), `tasks/get` (polling), and `tasks/cancel`. For streaming, the executor emits multiple events through the `EventQueue`, and the handler serializes them as SSE `data:` lines.

### Key protocol endpoints

- **`message/send`** — POST with `{role: "user", parts: [{kind: "text", text: "..."}]}`. Returns a Task or Message object.
- **`message/stream`** — Same request format, returns `text/event-stream` with `TaskStatusUpdateEvent` and `TaskArtifactUpdateEvent` objects.
- **`tasks/get`** — POST with `{id: "task-uuid"}`. Returns current task state, artifacts, and optional message history.

---

## 3. Prompt Opinion platform integration requirements

**Prompt Opinion** (by Darena Health) is the mandatory platform for this hackathon. It sits at the intersection of MCP, A2A, and FHIR, handling authentication, credential bridging, context routing, and governance. The platform URL is **chatwithpo.ai**, with the marketing site at promptopinion.ai.

### Two submission tracks

**Track 1 — "Superpower" (MCP Server):** Build an MCP server exposing healthcare tools. These become reusable capabilities any agent can invoke. Best if your value is in a specific tool (drug interaction checker, FHIR data extractor, guideline lookup).

**Track 2 — "Agent" (A2A Protocol):** Configure an intelligent agent for complex workflows. Can be built **no-code directly on the platform** or externally with A2A standards. Best if your value is in orchestrating multiple tools into a clinical workflow.

**Optimal strategy**: Build MCP tools (Track 1) AND show them orchestrated by an agent, demonstrating the full "Agents Assemble" vision.

### SHARP-on-MCP: the healthcare extension spec

**SHARP** (Standardized Healthcare Agent Remote Protocol) at **sharponmcp.com** layers healthcare context propagation on top of standard MCP. The core mechanism: the Prompt Opinion platform bridges EHR session credentials into SHARP context, propagating patient IDs and FHIR access tokens through multi-agent call chains. This means your MCP tools receive patient context automatically — you don't build bespoke token-handling.

The practical implication for your tools: accept `patient_id` and FHIR server parameters in your tool `inputSchema`, and the platform will populate them from the active EHR session context. Build on SMART on FHIR OAuth 2.0 patterns for patient-scoped access (`patient/*.read`).

### Community MCP repo and hosting

The reference implementations live at **github.com/prompt-opinion/po-community-mcp** in TypeScript and .NET. There is no Python reference, but the "Build Your Own MCP Server" path explicitly allows any language. Hosted test instances:

- .NET: `https://dotnet.fhir-mcp.promptopinion.ai/mcp`
- TypeScript: `https://ts.fhir-mcp.promptopinion.ai/mcp`

### Mandatory submission checklist

1. Create free Prompt Opinion account at chatwithpo.ai
2. Build MCP server or A2A agent with SHARP Extension Specs
3. Use FHIR server data (highly recommended — use SMART Health IT sandbox at `https://r4.smarthealthit.org/`)
4. Publish to the Prompt Opinion Marketplace
5. Record a **sub-3-minute demo video** showing the project functioning within the platform
6. Submit on Devpost by **May 11, 2026 at 11pm EDT**

Watch the getting-started video at `https://youtu.be/Qvs_QK4meHc` for platform-specific registration and publishing workflows. Join the Discord at `https://discord.gg/cCBxKpdS7j` for support.

---

## 4. Clinical decision support patterns that win hackathons

The predecessor hackathon by the same organizer ("Predictive AI in Healthcare with FHIR," $25K pool) produced revealing winners: **MyHeartRisk** (cardiac risk scores via CDS Hooks), **Orama** (patient data → actionable insights), **ClinicalConnect** (clinical trial matching), and **Anecdotal AI** (differential diagnosis from symptoms). Every winner used FHIR data, demonstrated working EHR integration, and applied AI for reasoning that rule-based systems cannot replicate.

### Highest-scoring use cases across all three criteria

**#1 — Polypharmacy / Drug Interaction Intelligence Agent.** AI Factor: Claude reasons over complex medication regimens, comorbidities, and guidelines simultaneously — combinatorial complexity that exceeds rule-based systems. Impact: adverse drug events cost **$3.5B+ annually**; elderly patients with 5+ medications face the highest risk. Feasibility: MedicationRequest, AllergyIntolerance, and Condition are mature FHIR resources; human-in-the-loop is natural since clinicians approve medication changes.

**#2 — Differential Diagnosis Assistant.** AI Factor: LLMs achieve **>80% accuracy on top-10 differential lists** (JMIR Med Inform 2023); Google's AMIE (Nature 2025) showed LLM-assisted clinicians significantly outperform those using search engines alone. Impact: diagnostic errors affect **12 million Americans annually**. Feasibility: outputs recommendations to a clinician, not autonomous decisions — straightforward regulatory posture.

**#3 — FHIR Patient Summary with AI Clinical Briefing.** AI Factor: synthesizes hundreds of FHIR resources into a 1-page briefing, identifying patterns humans miss (worsening lab trends, medication non-adherence signals). Impact: clinicians spend **49% of their time on documentation**, not patients. Feasibility: maps directly to the `patient-view` CDS Hook; low clinical risk since it's informational.

### Synthea for synthetic FHIR data

**Synthea** (synthetichealth.github.io/synthea/) generates realistic FHIR R4 patient bundles with Conditions (SNOMED), Observations (LOINC), MedicationRequests (RxNorm), AllergyIntolerances, Procedures, CarePlans, and more. Quick start:

```bash
git clone https://github.com/synthetichealth/synthea.git && cd synthea
./gradlew build check test
./run_synthea -p 50 Massachusetts  # 50 patients, FHIR R4 output in ./output/fhir/
./run_synthea -p 20 -m diabetes    # 20 diabetic patients specifically
```

Pre-built datasets (1,000+ patients) are downloadable from the Synthea site. The **SMART on FHIR sample bulk datasets** at `github.com/smart-on-fhir/sample-bulk-fhir-datasets` provide 100-patient ndjson exports. The live SyntheticMass FHIR API at `syntheticmass.mitre.org/fhir/metadata` is also available.

### Claude API as the CDS reasoning engine

Use **claude-sonnet-4-20250514** with tool_use for structured clinical reasoning. The pattern: define FHIR data retrieval and drug-lookup as tools, send a clinical vignette, then let Claude call tools iteratively before synthesizing its analysis.

```python
import anthropic, json

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a clinical decision support assistant. Analyze patient 
data from FHIR resources to generate: ranked differential diagnoses with rationale, 
drug interaction alerts with clinical significance, and care gap identification.
Structure reasoning as: Key Findings → Differential (ranked) → Recommended Next Steps → Red Flags.
DISCLAIMER: All outputs require review by a licensed provider."""

tools = [
    {
        "name": "fetch_patient_fhir",
        "description": "Retrieve FHIR resources (Condition, Observation, MedicationRequest, AllergyIntolerance) for a patient",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "resource_types": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "check_interactions",
        "description": "Check drug-drug interactions given RxNorm codes. Returns severity and descriptions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rxnorm_codes": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["rxnorm_codes"]
        }
    }
]

def run_cds_reasoning(patient_context: str) -> str:
    messages = [{"role": "user", "content": patient_context}]
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=4096,
        system=SYSTEM_PROMPT, tools=tools, messages=messages
    )
    
    # Agentic tool-use loop
    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result)
                })
        messages.extend([
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results}
        ])
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=4096,
            system=SYSTEM_PROMPT, tools=tools, messages=messages
        )
    
    return next(b.text for b in response.content if hasattr(b, "text"))
```

For drug interaction data, **DDInter** (ddinter.scbdd.com) provides 236,834 open-access interactions with severity levels for 1,833 drugs. **RxNorm/RxNav** (NLM) handles medication normalization. DrugBank offers a free research tier.

---

## 5. Optimizing for all three judging criteria simultaneously

The three criteria are equally weighted with no explicit percentage breakdowns. Based on predecessor hackathon winners, **platform integration is the #1 disqualifier** — submissions that don't demonstrate the app functioning within the Prompt Opinion platform will not be considered.

### The AI Factor: prove GenAI does what rules cannot

Show Claude performing **multi-source clinical reasoning** — synthesizing unstructured notes + structured FHIR data + clinical guidelines into a contextual assessment. The strongest signal: a tool that takes ambiguous, incomplete patient data and produces nuanced clinical reasoning with uncertainty quantification. The weakest signal: a chatbot wrapper or a lookup table behind an LLM.

Concrete demonstration pattern: show the same clinical scenario processed by (a) a rule-based system producing a generic alert and (b) your AI agent producing contextual, patient-specific reasoning with ranked differentials and personalized recommendations.

### Potential Impact: quantify the pain point

Choose problems with published statistics. **Diagnostic errors** affect 12 million Americans annually (BMJ Quality & Safety). **Adverse drug events** cost $3.5B+ (AHRQ). **Prior authorization** delays care for 93% of physicians (AMA survey). Frame impact using the Triple Aim: outcomes, cost, experience. In your demo video, state the problem with numbers first, then show your solution.

### Feasibility: respect healthcare's constraints

Five concrete signals judges look for:

- **FHIR integration** using standard resources (Patient, Condition, MedicationRequest, Observation, AllergyIntolerance)
- **Human-in-the-loop** at every clinical decision point — your agent recommends, clinicians decide
- **HIPAA awareness** — use synthetic data (Synthea), mention de-identification in architecture discussions
- **SHARP compliance** — use the SHARP extension specs for context propagation
- **Regulatory framing** — reference ONC HTI-1 DSI transparency requirements; position your tool as decision *support*, not autonomous diagnosis (avoiding FDA SaMD classification)

### The "perfect score" architecture

Build both an MCP Superpower and an A2A Agent to demonstrate multi-agent composition:

**MCP Server 1** — "Clinical Evidence Extractor": tools that pull patient data from FHIR, build clinical vignettes, and extract relevant conditions/labs/medications.

**MCP Server 2** — "AI Clinical Reasoner": tools that take a clinical vignette and return differential diagnoses, drug interaction analysis, or care gap identification using Claude.

**A2A Agent** — "CDS Orchestrator": coordinates the two MCP servers into a complete clinical workflow. Receives a patient context via SHARP, calls the extractor to gather data, passes it to the reasoner, and returns structured CDS cards.

This architecture maps precisely to the hackathon's "Agents Assemble" theme, uses both submission tracks simultaneously, and demonstrates the full MCP + A2A + FHIR + SHARP stack.

### Implementation timeline for remaining ~51 days

**Week 1**: Set up accounts (Prompt Opinion, Devpost), generate Synthea data, build basic MCP server with one tool (patient FHIR data extractor), test against hosted MCP endpoints. Watch the getting-started video and join Discord.

**Week 2**: Implement Claude reasoning engine with tool_use loop, build drug interaction checker tool, wire up the core CDS workflow (FHIR → vignette → Claude → structured output).

**Week 3**: Build A2A agent orchestration layer, implement SHARP context propagation, integrate with Prompt Opinion platform, publish to marketplace.

**Week 4-5**: Polish demo scenarios (pick 2-3 compelling patient cases from Synthea), add error handling and guardrails, record sub-3-minute demo video with clear problem statement → solution demo → impact quantification narrative.

**Week 6-7 (buffer)**: Edge cases, documentation, final testing on platform, submit.

---

## Conclusion: key technical decisions

The winning move is **not** building the most complex system — predecessor winners each solved **one problem well** rather than attempting comprehensive platforms. Build a polypharmacy drug interaction agent or a differential diagnosis assistant (both score highest across all three criteria), implement it as MCP tools orchestrated by an A2A agent, and demonstrate it running on Prompt Opinion with real FHIR data flowing through SHARP context.

The technical stack that aligns best: Python MCP server using FastMCP (`mcp.server.fastmcp`) with streamable-http transport, A2A orchestration using `a2a-sdk` with `DefaultRequestHandler`, Claude claude-sonnet-4-20250514 with tool_use for clinical reasoning, Synthea-generated FHIR R4 data loaded into the SMART Health IT sandbox, and SHARP-on-MCP for healthcare context propagation. This gives you standards compliance (MCP + A2A + FHIR), genuine AI differentiation (Claude reasoning over clinical data), and feasibility (synthetic data, human-in-the-loop, established FHIR resources) — all three judging criteria covered.