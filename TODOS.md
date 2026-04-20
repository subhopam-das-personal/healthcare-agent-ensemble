# TODOS

## P2 — Multi-Patient Panel / Ward View
**What:** Build a ward view showing 3-4 patients simultaneously with AI priority-ranking.
**Why:** The 10x vision from the hackathon demo planning. A clinical judge asking "what happens with other patients?" after the demo is a signal this is the right next step.
**Pros:** Turns the demo into a real product vision. The executor already supports parallel patient analysis.
**Cons:** New UI paradigm — need 3 distinct clinical stories, not just one. Post-demo work.
**Context:** Identified during /plan-ceo-review (2026-03-27). Core infrastructure (executor.py) already supports it. Pure UI build.
**Effort:** L (human) / M (CC+gstack)
**Priority:** P2 — post-demo
**Depends on:** Near-miss reconstruction demo shipped first

## P3 — NL Query Engine Integration with Demo
**What:** Allow judges to type of clinical scenario in English to find matching FHIR patients, instead of entering a UUID.
**Why:** Removes the "where do I get a patient UUID?" friction entirely. The NL Query Engine design exists (approved 2026-03-27).
**Pros:** Makes the demo self-serve. Judges can try their own scenarios.
**Cons:** Scope expansion on top of existing demo work. The NL Query Engine itself is a separate large build.
**Context:** Identified during /plan-ceo-review (2026-03-27). Design doc at ~/.gstack/projects/.../subhopam-master-design-fhir-nl-query-engine-20260327-163132.md
**Effort:** XL (human) / L (CC+gstack)
**Priority:** P3 — longer-term
**Depends on:** NL Query Engine design fully implemented

## Completed
### P1 — Verify 85% Serotonin Syndrome Miss Rate Stat
**Resolved:** Stat reframed. Source is Mackay et al., Br J Gen Pract 1999 (85.4% of UK GPs unaware SS existed as a diagnosis) — not Boyer & Shannon NEJM 2005. Banner copy updated in design doc to: "85% of general practitioners were unaware it existed as a diagnosis (Mackay et al., Br J Gen Pract 1999)." Citable and defensible.
**Completed:** 2026-04-27

### P2 — Multi-Patient Panel / Ward View
**Completed:** 2026-04-27

### P3 — NL Query Engine Integration with Demo
**Completed:** 2026-04-27
