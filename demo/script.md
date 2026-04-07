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

**Load Tab A first. Confirm the UI is up. Navigate to the 🔍 Find Patients tab and leave it there.**
**Do not touch anything until you begin speaking.**

---

## RUN 1 of 5 — Date: __________ Time: __________  Score: ___/5

---

## [0:00 – 0:45] HOOK — The recruitment problem

**Action: No screen interaction. Talk first.**

> "Over 80% of clinical trials miss their enrollment timelines. The bottleneck is
> not the science — it's finding the right patients. A site coordinator today pulls
> charts one by one, cross-references inclusion and exclusion criteria manually.
> 45 to 90 minutes per patient. Most eligible patients are never identified.
>
> I built a system that does this in seconds. Let me show you what that looks like
> for a Phase II COVID-19 human trial."

**Checkpoint:** Did you say "80% miss enrollment timelines" and "Phase II COVID-19 trial"? ☐

---

## [0:45 – 1:15] ARCHITECTURE — Open protocols (Tab B → Tab C)

**Action: Switch to Tab B** → https://a2a-agent-production-03a5.up.railway.app/.well-known/agent.json

> "This is the A2A agent card — the open protocol Google and the industry are standardizing
> on for agent interoperability. PPD's Preclarus suite doesn't expose an interface like this.
> Any orchestrating AI can discover and call this agent."

**Action: Switch to Tab C** → https://healthcare-agent-ensemble-production.up.railway.app/mcp

> "And this is the MCP server — four composable clinical tools built on FHIR R4.
> Patient retrieval, differential diagnosis, drug interaction analysis, care gap identification.
> FHIR-structured data exposed through open protocols that AI agents can actually call.
> This is what Horizon 3 looks like."

**Checkpoint:** Did you say "Preclarus doesn't expose this" and "Horizon 3"? ☐

---

## [1:15 – 2:15] COHORT DISCOVERY — Find trial candidates (Tab A, Find Patients tab)

**Action: Switch to Tab A** → https://ui-server-production.up.railway.app

Make sure you are on the **🔍 Find Patients** tab.

> "I'm a trial sponsor running a Phase II COVID-19 study. I need to find patients
> in this hospital cohort who match my enrollment criteria. Watch what happens
> when I describe what I'm looking for in plain English."

**Action:** Type this query in the search box:

```
patients with respiratory infections or COVID-19
```

> "No FHIR query syntax. No ICD-10 lookup. Plain English."

**Action:** Click **🔍 Search**

*While it loads, say:*

> "The engine is extracting medical entities from that sentence, expanding them
> through a clinical ontology — ICD-10 codes, LOINC codes, drug classes —
> then running a validated SQL query against the indexed patient cohort."

**Action:** When results appear — point to the **🧬 Ontological expansion** panel.

> "Look at what it understood. It mapped 'respiratory infections' to ICD-10 codes
> automatically — the same coding standard clinical trials use for eligibility criteria.
> And here is my candidate cohort."

**Action:** Point to the **Generated SQL** expander (expand it briefly).

> "Full transparency — the exact query it ran, auditable, reproducible.
> This is not a black box. This is AI-readiness engineering."

**Checkpoint:** Ontological expansion visible ☐  |  SQL shown ☐  |  Patient list visible ☐

---

## [2:15 – 2:45] SELECT A CANDIDATE — One click to clinical review

**Action:** Click any patient row in the results table.

*The success banner appears: "Patient [Name] selected. Switch to Clinical Review tab."*

> "I've identified a candidate. One click. Now I want to understand whether
> this patient is actually safe to enroll — what medications are they on,
> what interactions could affect trial outcomes, what's their full clinical picture.
> Let me run the full pre-screening."

**Action:** Click the **🏥 Clinical Review** tab.

*Confirm the Patient ID field is pre-filled from the selection.*

**Action:** Click **▶ Run Analysis** in the sidebar.

**Checkpoint:** Patient ID pre-filled from Find Patients ☐  |  Run Analysis clicked ☐

---

## [2:45 – 4:00] SAFETY PRE-SCREENING — Drug interactions + clinical trials

*Narrate steps as they appear. Do not go silent.*

> "Step 1: FHIR R4 bundle — full patient record. Same resource model Clario's
> eCOA streams produce. Step 3: matching against ClinicalTrials.gov in real time."

**Action:** When **Drug Interaction Analysis** appears — expand it.

> "This is the safety pre-screening layer. Before you enroll a patient in a trial,
> you need to know what their current medications do in combination.
> Standard pairwise drug checkers miss multi-drug signals.
>
> Our system passes the full medication list to Claude simultaneously.
> It reasons over the entire context — not rule lookups, not pairwise checks.
> AI clinical reasoning over structured FHIR data.
>
> An adverse event in Phase II costs you months and millions.
> Catching a drug interaction before enrollment costs you nothing."

**Action:** Expand **Integrated Clinical Assessment**.

> "The synthesis agent produces a single integrated assessment — severity,
> consequence, recommendation. This is what a data domain product looks like.
> Not a pipeline that moves data. A capability that produces a decision."

**Action:** Expand the **🧪 Clinical Trials** section.

> "And step 3 has already matched this patient against recruiting trials on
> ClinicalTrials.gov — filtered by their conditions, age, and gender.
> The coordinator gets a pre-screened candidate and a shortlist of trials
> they qualify for, in the same workflow. Zero extra work."

**Checkpoint:** Drug interactions expanded ☐  |  Clinical trials visible ☐

---

## [4:00 – 4:45] POPULATION INTELLIGENCE — Cohort-level view

**Action:** Click the **📊 Population Intelligence** tab.

> "That was one patient. This is the population view.
> Condition distribution, medication distribution, comorbidity clusters
> across the entire indexed cohort.
>
> Before a sponsor commits to a Phase II design, they need to know:
> does the patient population I need actually exist at this site?
> What comorbidities will complicate my eligibility criteria?
> How many patients survive the inclusion filters?
>
> This is the feasibility intelligence layer PPD can offer sponsors
> before a trial even starts. No other CRO is doing this today.
>
> I treat data domains the way a product manager treats a product —
> with measurable outputs, not just pipelines."

**Checkpoint:** Did you say "feasibility intelligence" and "data domain product"? ☐

---

## [4:45 – 5:00] CLOSE — The competitive clock

**Action: No new screen interaction. Look at camera.**

> "Three services, open protocols, running in production right now.
> Plain English in. Trial candidates, safety pre-screening, and enrollment
> intelligence out.
>
> IQVIA launched 150 production AI agents last month.
> Parexel cut data engineering costs 85% with an AI-native platform.
> The competitive clock is running.
>
> The Data Domain Manager who builds AI-ready, protocol-compliant trial data
> today determines whether PPD's OpenAI investment compresses trial timelines —
> or stays an internal efficiency tool.
>
> That's the role I want."

**Checkpoint:** Did you say "150 production AI agents" and "that's the role I want"? ☐

---

## BACKUP — If something breaks

| What broke | What to say | What to do |
|---|---|---|
| Find Patients returns zero results | "The cohort is Synthea synthetic data — let me try a broader query." | Try: `patients with chronic respiratory disease` or `patients with hypertension` |
| A2A agent not responding | "The Railway service is restarting. The architecture is what matters." | Hard-refresh Tab A, click ▶ Run Analysis again |
| Clinical trials section empty | "ClinicalTrials.gov API is occasionally rate-limited. The matching logic queries by condition, age, and gender in real time." | Move on, don't dwell |
| Analysis takes >2 min | Narrate steps appearing. "DDx and drug safety agents running in parallel — synthesis integrating both." | Keep narrating |
| Patient ID not pre-filled after row click | Type any Patient ID manually. The cohort discovery story already landed. | Silent recovery |
| Railway UI is down | Go to Tab B + Tab C. Walk through the architecture from the JSON. | Pivot to architecture story |

**Backup queries if COVID-19 returns no results:**
- `patients with viral respiratory infections`
- `patients with pneumonia or bronchitis`
- `patients with diabetes and hypertension` (always returns results in Synthea)

---

## THE ONE SENTENCE — If they ask "what's the business case?"

> "Tufts CSDD 2025: integrated CRO/CDMO services reduce timelines by 34 months
> and generate $63M net financial benefit per program.
> Cohort discovery and safety pre-screening at the data layer is how you get there.
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
By run 5 you should be able to do it cold — services up or down.

---

## KEY NUMBERS — Memorize these

| Number | Source | When to use |
|--------|--------|-------------|
| 80% | Applied Clinical Trials 2022 | Opening hook |
| 45–90 min/patient | Industry standard | Manual chart review cost |
| $500,000/day | Tufts CSDD 2024 | If they ask about trial delay cost |
| 150 AI agents | IQVIA.ai, Mar 2026 | Closing |
| 85% cost reduction | Parexel/Palantir 2024 | Closing |
| 34 months / $63M | Tufts CSDD 2025 | Business case answer |
| $8.875B | Thermo Fisher IR, Oct 2025 | If Clario comes up |
| 6.7% | Norstella/Citeline 2023 | If Phase I approval rates come up |

---

## DEMO FLOW AT A GLANCE

```
Tab B (Agent Card) → Tab C (MCP Tools) → Tab A (Find Patients)
        ↓
  Type query: "patients with respiratory infections or COVID-19"
        ↓
  Show: Ontological expansion (ICD-10) + Generated SQL + Patient list
        ↓
  Click a patient row → Switch to Clinical Review tab
        ↓
  Run Analysis → Drug Interaction Analysis + Clinical Trials
        ↓
  Population Intelligence tab → Cohort view
        ↓
  Close
```
