# TODOS

## P1 — Verify 85% Serotonin Syndrome Miss Rate Stat
**What:** Confirm the "misdiagnosed or missed in ~85% of cases at presentation" claim is supported by a citable source before presenting to judges.
**Why:** A clinician judge who challenges an uncited clinical stat will derail the demo faster than any technical failure. The stat is the emotional hook of the demo — it must be defensible.
**Pros:** Credibility with medical professionals; easy to verify or replace.
**Cons:** Stat may not exist as stated; fallback language ("frequently underrecognized at initial presentation") is less punchy but safe.
**Context:** Flagged during /plan-eng-review (2026-03-27). Design doc cites Boyer & Shannon, NEJM 2005, but the exact 85% figure needs confirmation. Sternbach 2003 is another candidate. If unsupported, update both the UI banner and demo script to use defensible language.
**Effort:** S (human) / XS (CC+gstack — just a citation lookup)
**Priority:** P1 — must do before demo
**Depends on:** Nothing

## P2 — Multi-Patient Panel / Ward View
**What:** Build a ward view showing 3-4 patients simultaneously with AI priority-ranking.
**Why:** The 10x vision from the hackathon demo planning. A clinical judge asking "what happens with other patients?" after the demo is the signal this is the right next step.
**Pros:** Turns the demo into a real product vision. The executor already supports parallel patient analysis.
**Cons:** New UI paradigm — need 3 distinct clinical stories, not just one. Post-demo work.
**Context:** Identified during /plan-ceo-review (2026-03-27). Core infrastructure (executor.py) already supports it. Pure UI build.
**Effort:** L (human) / M (CC+gstack)
**Priority:** P2 — post-demo
**Depends on:** Near-miss reconstruction demo shipped first

## P3 — NL Query Engine Integration with Demo
**What:** Allow judges to type the clinical scenario in English to find matching FHIR patients, instead of entering a UUID.
**Why:** Removes the "where do I get a patient UUID?" friction entirely. The NL Query Engine design exists (approved 2026-03-27).
**Pros:** Makes the demo self-serve. Judges can try their own scenarios.
**Cons:** Scope expansion on top of existing demo work. The NL Query Engine itself is a separate large build.
**Context:** Identified during /plan-ceo-review (2026-03-27). Design doc at ~/.gstack/projects/.../subhopam-master-design-fhir-nl-query-engine-20260327-163132.md
**Effort:** XL (human) / L (CC+gstack)
**Priority:** P3 — longer-term
**Depends on:** NL Query Engine design fully implemented
