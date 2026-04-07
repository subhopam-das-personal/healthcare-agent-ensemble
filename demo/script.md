# Demo Script — Healthcare Agent Ensemble
### PPD Digital Data Capabilities | Data Domain Manager, Trial Data
### Target: 5 minutes | Rehearse 5× before the interview

---

## PRE-FLIGHT — Do this 10 minutes before the call

**No local setup needed. Everything runs on Railway.**

Open these 3 browser tabs and leave them ready:

| Tab | URL | What it shows |
|-----|-----|---------------|
| **Tab A — UI** | https://ui-server-production.up.railway.app | The main demo screen |
| **Tab B — Agent Card** | https://a2a-agent-production-03a5.up.railway.app/.well-known/agent.json | A2A open protocol discovery |
| **Tab C — MCP Tools** | https://healthcare-agent-ensemble-production.up.railway.app/mcp | MCP server / composable tools |

**Load Tab A first and confirm the UI is up before the call starts.**
**Start on Tab A. Do not touch anything until you begin speaking.**

---

## RUN 1 of 5 — Date: __________ Time: __________  Score: ___/5

---

## [0:00 – 0:45] HOOK — The $500K/day number

**Action: No screen interaction. Talk first.**

> "Clinical trial data has a $500,000-per-day problem. Every day a trial runs late —
> fragmented pipelines, siloed sources, slow data activation — that's the cost to the sponsor.
> The highest-leverage place to apply AI is the data layer between raw trial data and the
> reasoning system. That's what I built. Let me show you."

**Checkpoint:** Did you say "$500,000-per-day" and "data layer"? ☐

---

## [0:45 – 1:15] ARCHITECTURE — Open protocols (Tab B → Tab C)

**Action: Switch to Tab B** → https://a2a-agent-production-03a5.up.railway.app/.well-known/agent.json

> "This is the A2A agent card — the open protocol Google and the industry are standardizing
> on for agent interoperability. PPD's Preclarus suite doesn't expose an interface like this.
> Any orchestrating AI can discover and call this agent."

**Action: Switch to Tab C** → https://healthcare-agent-ensemble-production.up.railway.app/mcp

> "And this is the MCP server — four composable clinical tools: FHIR R4 patient retrieval,
> differential diagnosis, drug interaction analysis, care gap identification.
> The A2A agent orchestrates these into a complete clinical workflow.
> This is what Horizon 3 looks like — FHIR-structured data exposed through open protocols
> that AI agents can actually call."

**Checkpoint:** Did you say "Preclarus doesn't expose this" and "Horizon 3"? ☐

---

## [1:15 – 1:45] LOAD PATIENT — Demo Mode (Tab A)

**Action: Switch to Tab A** → https://ui-server-production.up.railway.app

> "The demo patient is Margaret Alvarez, 55-year-old female.
> Three active conditions: major depressive disorder, chronic pain, post-operative infection.
> Three medications: sertraline, tramadol, linezolid.
> Synthetic patient — constructed from published FDA adverse event patterns. Strictly HIPAA-compliant."

**Action:** Check **🎬 Demo Mode** checkbox in the sidebar.

Confirm sidebar shows:
```
Margaret Alvarez · 55F
Sertraline + Tramadol + Linezolid
Synthetic patient — constructed from published FDA adverse event patterns.
```

> "I'm going to run the full comprehensive clinical review. Watch what happens."

**Action:** Click **▶ Run Analysis**

**Checkpoint:** Demo Mode checked ☐  |  Run Analysis clicked ☐

---

## [1:45 – 3:15] THE FINDING — Serotonin syndrome

*Narrate steps as they appear on screen. Do not go silent.*

> "Step 1: FHIR R4 bundle ingestion — same resource model Clario's eCOA streams produce.
> Patient, Condition, MedicationRequest, Observation, AllergyIntolerance.
> Step 3: Clinical trials matched against ClinicalTrials.gov in real time.
> Now — DDx agent and drug safety agent running in parallel."

**Action:** When **Drug Interaction Analysis** section appears — expand it.

> "Standard pairwise drug checkers — RxNav, clinical databases — return this combination clean.
> Sertraline + tramadol: fine. Tramadol + linezolid: fine.
> But the dangerous signal is the triad.
>
> Our system passed all three medications to Claude simultaneously.
> It reasoned over the full context.
> It flagged serotonin syndrome — a potentially fatal triad that 85% of GPs
> don't recognize as a diagnosis.
>
> This is not a rule lookup. This is AI clinical reasoning over structured FHIR data.
> That is the distinction between what Preclarus does today and what the next
> generation of clinical AI needs to do."

**Action:** Expand **Integrated Clinical Assessment** section.

> "The synthesis agent integrates the DDx findings and the drug safety findings
> into a single clinical assessment — severity, consequence, recommendation.
> This is what a data domain product looks like. Not a pipeline that moves data.
> A capability that produces a decision."

**Checkpoint:** Did you say "not a rule lookup" and "data domain product"? ☐

---

## [3:15 – 4:00] CLINICAL TRIALS — Enrollment intelligence

**Action:** Expand the **🧪 Clinical Trials** section.

> "Step 3 queried ClinicalTrials.gov in real time. These are recruiting interventional trials
> matched to Margaret's conditions — filtered by age and gender.
>
> Over 80% of clinical trials miss enrollment timelines. The bottleneck is that eligible
> patients are never identified. This system does that identification automatically,
> at the point of care.
>
> For PPD, this is the Datavant linkage story — connecting internal trial data to
> real-world patient populations through a standards-based layer.
> Same pattern. Production-grade."

**Checkpoint:** Did you say "80% miss enrollment timelines" and "Datavant linkage"? ☐

---

## [4:00 – 4:45] POPULATION INTELLIGENCE — Data domain product view

**Action:** Click the **📊 Population Intelligence** tab.

> "This is the data domain product view. Not a single patient — a population.
> Condition distribution, medication distribution, comorbidity clusters.
>
> This is what a sponsor needs before they design a Phase II trial:
> does the patient population I need actually exist?
> What comorbidities will complicate my eligibility criteria?
> How many patients at this site qualify?
>
> I treat data domains the way a product manager treats a product —
> with measurable outputs, not just pipelines.
> This tab is the difference between a data engineering project and a data product."

**Checkpoint:** Did you say "data product, not a pipeline"? ☐

---

## [4:45 – 5:00] CLOSE — The competitive clock

**Action: No new screen interaction. Look at camera.**

> "Three services, open protocols, production-ready.
> FHIR R4 in. Clinical decisions out.
> This is Horizon 3 — built today so I understand exactly what Horizons 1 and 2
> at PPD need to deliver to make it real.
>
> IQVIA launched 150 production AI agents last month.
> The Data Domain Manager who builds AI-ready, protocol-compliant trial data today
> determines whether PPD's OpenAI investment compresses trial timelines —
> or stays an internal efficiency tool.
>
> That's the role I want."

**Checkpoint:** Did you say "150 production AI agents" and "that's the role I want"? ☐

---

## BACKUP — If something breaks

| What broke | What to say | What to do |
|---|---|---|
| A2A agent not responding | "The Railway service is restarting — this happens occasionally on the free tier. The architecture is what matters." | Hard-refresh Tab A, click ▶ Run Analysis again |
| Clinical trials section empty | "ClinicalTrials.gov API is live — occasionally rate-limited. The matching logic queries the v2 REST API by condition, age, and gender." | Move on, don't dwell |
| Analysis takes >2 min | Narrate steps appearing. "You're watching a multi-agent workflow — DDx and drug safety agents running in parallel, synthesis integrating both." | Keep narrating |
| Demo Mode doesn't load patient | Type `demo-serotonin-patient-001` manually in Patient ID field. Run without Demo Mode checkbox. | Silent recovery |
| Railway UI is down | Switch to Tab B (agent card JSON) and Tab C (MCP endpoint) — explain the architecture from those. | Pivot to architecture story |

---

## THE ONE SENTENCE — If they ask "what's the business case?"

> "Tufts CSDD 2025: integrated CRO/CDMO services reduce timelines by 34 months
> and generate $63M net financial benefit per program.
> The data unification layer is what makes that possible.
> This project proves it can be built."

---

## REHEARSAL LOG

| Run | Date | Time taken | What went wrong | Score (1–5) |
|-----|------|------------|-----------------|-------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

**Target:** Run 5 clean. No silence gaps. No reading from screen.
By run 3 you should know the checkpoints without looking at this file.
By run 5 you should be able to do it with the services down and recover live.

---

## KEY NUMBERS — Memorize these

| Number | Source | When to use |
|--------|--------|-------------|
| $500,000/day | Tufts CSDD 2024 | Opening hook |
| 80% | Applied Clinical Trials 2022 | Clinical trials section |
| 85% of GPs | Mackay et al., Br J Gen Pract 1999 | Serotonin syndrome moment |
| 150 AI agents | IQVIA.ai, Mar 2026 | Closing |
| 34 months / $63M | Tufts CSDD 2025 | Business case answer |
| $8.875B | Thermo Fisher IR, Oct 2025 | If Clario comes up |
| 6.7% | Norstella/Citeline 2023 | If Phase I approval rates come up |
