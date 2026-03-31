# Healthcare Agent Ensemble

> **Hackathon submission** — AI agent ensemble for clinical decision support, built on MCP + A2A + Claude.

**Live demo:** [ui-server-production.up.railway.app](https://ui-server-production.up.railway.app)

---

## The Problem

Standard drug interaction checkers work pairwise. They check Drug A vs Drug B, and Drug B vs Drug C — but they miss emergent interactions that only appear when three or more drugs are combined. Serotonin syndrome is the canonical example: sertraline + tramadol + linezolid individually each carry warnings, but the combined serotonergic load that causes a crisis only emerges from the triple combination.

**85% of general practitioners were unaware serotonin syndrome existed as a diagnosis** (Mackay et al., _Br J Gen Pract_ 1999). Annual incidence is estimated at 14–16 cases/million per year, likely higher due to underreporting.

This project shows how an AI agent ensemble catches what pairwise checkers miss.

---

## Demo

The fastest path to seeing the core value:

1. Open [ui-server-production.up.railway.app](https://ui-server-production.up.railway.app)
2. Check **🎬 Demo Mode** in the sidebar
3. Click **▶ Run Analysis**

The demo loads a synthetic patient (Margaret Alvarez, 55F, post-surgical) on Sertraline + Tramadol + Linezolid. The side-by-side result shows:

| Standard Pairwise Checker | Healthcare Agent Ensemble |
|--------------------------|--------------------------|
| "No critical drug-drug interactions detected." | Serotonin syndrome flagged in differential diagnosis, all three pair interactions flagged High severity |

The synthetic patient is constructed from published FDA adverse event patterns — not a real individual's record.

---

## Architecture

Three Railway services, one repo, dispatched via `start.sh`:

```
┌─────────────────────────────────────────────────────────┐
│  Browser / User                                          │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTPS
┌──────────────────────▼──────────────────────────────────┐
│  UI Server (Streamlit)                                   │
│  ui-server-production.up.railway.app                    │
│                                                          │
│  - Clinical Review tab: run agent analysis               │
│  - Find Patients tab: NL query engine (DDM)              │
└──────────┬───────────────────────┬──────────────────────┘
           │ A2A (SSE streaming)   │ JSON-RPC / MCP
           │                       │
┌──────────▼──────────┐  ┌────────▼────────────────────────┐
│  A2A Agent           │  │  MCP Server                     │
│  a2a-agent-          │  │  healthcare-agent-ensemble-     │
│  production-         │  │  production.up.railway.app      │
│  03a5.up.railway.app │  │                                 │
│                      │  │  Tools:                         │
│  Orchestrates the    │  │  - get_patient_summary          │
│  ensemble; calls MCP │  │  - check_drug_interactions      │
│  tools in sequence   │  │  - generate_differential_dx     │
│  and streams results │  │  - synthesize_clinical_assess.  │
│  back to UI          │  │  - find_matching_trials         │
│                      │  │  - nl_query_patients (DDM)      │
└──────────────────────┘  │  - index_fhir_source            │
                          └─────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
              ┌─────▼────┐   ┌──────▼───┐   ┌───────▼──────┐
              │  FHIR    │   │  RxNav   │   │  Claude AI   │
              │  Server  │   │  API     │   │  (Anthropic) │
              │ (SMART   │   │ (drug    │   │ reasoning    │
              │ sandbox  │   │  data)   │   │ + DDx        │
              │ or live) │   └──────────┘   └──────────────┘
              └──────────┘
```

### Services

| Service | URL | Start command |
|---------|-----|---------------|
| UI Server | [ui-server-production.up.railway.app](https://ui-server-production.up.railway.app) | `streamlit run src/ui_server/app.py` |
| A2A Agent | [a2a-agent-production-03a5.up.railway.app](https://a2a-agent-production-03a5.up.railway.app) | `python src/a2a_agent/server.py` |
| MCP Server | [healthcare-agent-ensemble-production.up.railway.app](https://healthcare-agent-ensemble-production.up.railway.app) | `python src/mcp_server/server.py` |

All three share the same Docker image and are dispatched by `RAILWAY_SERVICE_NAME` in `start.sh`.

---

## MCP Tools

| Tool | Description |
|------|-------------|
| `get_patient_summary` | Fetches patient data from a FHIR server (demographics, conditions, medications, observations) |
| `check_drug_interactions` | Cross-references active medications against RxNav for interactions; Claude synthesizes serotonergic load |
| `generate_differential_diagnosis` | Claude reasons over patient data to produce a ranked differential |
| `synthesize_clinical_assessment` | Integrates all agent outputs into a final clinical narrative |
| `find_matching_trials` | Queries ClinicalTrials.gov for relevant open trials by condition |
| `nl_query_patients` | NL Query Engine — translates clinical English to SQL against the DDM index |
| `index_fhir_source` | Indexes a FHIR server into the DDM local SQLite store |
| `add_fhir_source` | Registers a new FHIR server URL with the DDM |

---

## Domain Data Manager (DDM)

The DDM (`src/ddm/`) is a local FHIR index that enables natural-language patient search without querying the live FHIR server on every request.

**Flow:**

```
FHIR Server → indexer.py → SQLite (local)
                                ↓
NL query → enricher.py → query_engine.py → SQL → results
```

**Pipeline:**
1. `index_fhir_source` MCP tool crawls the FHIR server and stores patient snapshots in SQLite
2. `enricher.py` expands clinical terms via ontology (e.g. "heart failure" → ICD-10 codes)
3. `query_engine.py` translates the enriched NL query into a validated SQL query
4. Results are returned as a ranked patient table

**Try it:** On the **Find Patients** tab, enter any clinical question:
- `diabetic patients on insulin over 65`
- `ACE inhibitor patients who have elevated creatinine`
- `post-surgical patients with serotonin syndrome risk`

---

## Test Patients

### Demo (built-in, no FHIR server needed)

| Field | Value |
|-------|-------|
| Name | Margaret Alvarez |
| Age/Sex | 55F, post-surgical |
| Medications | Sertraline 100mg, Tramadol 50mg, Linezolid 600mg |
| Clinical scenario | Triple serotonergic drug combination — emergent SS risk |
| How to load | Enable **🎬 Demo Mode** checkbox in sidebar |

### Live FHIR Sandbox (SMART Health IT)

| Field | Value |
|-------|-------|
| Name | Jeffery Daniel |
| Patient ID | `fa064acf-b7f1-4279-83d3-7a94686da7ba` |
| FHIR base | `https://r4.smarthealthit.org` |
| Profile | 16 conditions, 9 medications |
| How to load | Paste UUID into **Patient ID** field, leave Demo Mode off |

---

## Running Locally

### Prerequisites

- Python 3.11+
- `ANTHROPIC_API_KEY` — get one at [console.anthropic.com](https://console.anthropic.com)
- PostgreSQL — required for the DDM / Find Patients tab. Set `DATABASE_URL=postgresql://user:pass@localhost:5432/ddm`. Demo mode and Clinical Review work without it.

### Setup

```bash
git clone https://github.com/subhopam-das-personal/healthcare-agent-ensemble
cd healthcare-agent-ensemble
pip install -r requirements.txt
```

### Start all three services

```bash
# Terminal 1 — MCP Server
python src/mcp_server/server.py

# Terminal 2 — A2A Agent
MCP_SERVER_URL=http://localhost:8000 python src/a2a_agent/server.py

# Terminal 3 — UI
MCP_SERVER_URL=http://localhost:8000 \
A2A_AGENT_URL=http://localhost:8001 \
streamlit run src/ui_server/app.py
```

Open [localhost:8501](http://localhost:8501).

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key |
| `MCP_SERVER_URL` | No | `http://localhost:8000` | MCP server base URL (no trailing `/mcp`) |
| `A2A_AGENT_URL` | No | `http://localhost:8001` | A2A agent URL |
| `MCP_API_KEY` | No | — | API key for MCP server auth |
| `DATABASE_URL` | No | — | PostgreSQL connection string for DDM index. Required for Find Patients tab. Railway sets this automatically when you add a Postgres plugin to the MCP Server service. Format: `postgresql://user:pass@host:5432/db` |
| `A2A_PUBLIC_URL` | No | — | Public URL for A2A agent (Railway sets this) |

---

## Deploying to Railway

The repo uses a single Docker image dispatched by `RAILWAY_SERVICE_NAME`. Each Railway service needs:

**MCP Server service** (`RAILWAY_SERVICE_NAME=healthcare-agent-ensemble`):
```
ANTHROPIC_API_KEY=<key>
MCP_API_KEY=clin-intel-2026
DATABASE_URL=<set automatically by Railway Postgres plugin>
```
Add a **Postgres** plugin to this service in the Railway dashboard — Railway injects `DATABASE_URL` automatically. The DDM runs migrations on startup (idempotent, safe to redeploy).

**A2A Agent service** (`RAILWAY_SERVICE_NAME=a2a-agent`):
```
ANTHROPIC_API_KEY=<key>
MCP_SERVER_URL=https://healthcare-agent-ensemble-production.up.railway.app
MCP_API_KEY=clin-intel-2026
A2A_PUBLIC_URL=https://<a2a-service-url>.up.railway.app
```

**UI Server service** (`RAILWAY_SERVICE_NAME=ui-server`):
```
MCP_SERVER_URL=https://healthcare-agent-ensemble-production.up.railway.app
MCP_API_KEY=clin-intel-2026
A2A_AGENT_URL=https://a2a-agent-production-03a5.up.railway.app
```

See `DEPLOY_A2A.md` for step-by-step Railway setup.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| Agent orchestration | Google A2A SDK (`a2a-sdk`) |
| Tool protocol | Model Context Protocol (`mcp`, FastMCP) |
| AI reasoning | Claude 3.5 Sonnet (Anthropic) |
| Drug data | RxNav API (NLM) |
| Patient data | FHIR R4 (SMART Health IT sandbox) |
| Clinical trials | ClinicalTrials.gov API v2 |
| NL query index | SQLite + custom DDM pipeline |
| Deploy | Railway (monorepo, 3 services) |

---

## Project Structure

```
src/
├── mcp_server/     # FastMCP server — all clinical tools
├── a2a_agent/      # A2A orchestrator — sequences MCP tool calls
├── ui_server/      # Streamlit frontend + DDM query UI
├── ddm/            # Domain Data Manager — FHIR indexer + NL query engine
└── shared/         # FHIR, RxNav, Claude, trials clients

demo/
└── patient_serotonin_syndrome.json   # Synthetic demo patient bundle

tests/              # pytest test suite
```

---

## Disclaimer

This system is a research prototype for demonstration purposes. AI outputs require review by a licensed healthcare provider before clinical use. Not intended for use in actual patient care.
