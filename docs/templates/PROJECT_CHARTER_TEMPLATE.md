# Project Charter Template

> This template pairs with [`CLAUDE_TEMPLATE.md`](CLAUDE_TEMPLATE.md): the charter defines
> *what and what-not* (intent, boundaries, invariants); the CLAUDE.md defines *how*
> (gates, feedback loops, permissions). This repository's instantiation lives at
> [`../PROJECT_CHARTER.md`](../PROJECT_CHARTER.md).

---

# Project Charter — {PROJECT_NAME} (stable long-term plan)

> **Status: STABLE / SLOW-CHANGING.** This document is the north-star that humans and AI
> agents (and sub-agents) read first to stay on track. It changes rarely and only by
> deliberate decision — not per task. Day-to-day, changeable work lives in
> `NEXT_STEPS.md`; architecture detail in `C4.md` / `ARCHITECTURE.md`. If a task seems to
> require changing this charter, **stop and raise it explicitly** rather than editing it
> in passing.
>
> **Charter integrity is governed by §7 (Amendment Protocol).** Amendments are budgeted.
> When a budget or trigger fires, the amendment path closes and a full charter review is
> mandatory. Procedural tidiness is not a substitute for scope discipline.

---

## 0. Executive Summary

*(3–6 sentences, written last, updated on every ratified amendment. A reader who stops
here should know: what the system is, who it serves, the one or two hardest boundaries
it will not cross, and the current amendment posture.)*

> {PROJECT_NAME} is a {one-line system description} for {primary user/operator}. It does
> {core capability} and deliberately does **not** do {hardest non-goal}. The charter has
> {N} active carve-outs against a budget of {B}; the last full charter review was
> {DATE}. Execution detail lives in NEXT_STEPS.md; this document defines intent,
> boundaries, and invariants only.

---

## 1. Vision

*(One paragraph. The end-state as experienced by the user/operator, not the
implementation. Should survive a full rewrite of the codebase.)*

{VISION}

## 2. Mission (what we are building)

*(3–6 bullets. Concrete capabilities the system delivers. Each bullet should be
verifiable — you can point at a demo or artifact and say "that's this bullet.")*

- {CAPABILITY_1}
- {CAPABILITY_2}
- {CAPABILITY_3}

## 3. Scope

**In scope:** {components, packages, layers, and deliverables this project owns}.

**Out of scope (non-goals):** Each non-goal carries a stable ID. IDs are never reused or
renumbered; a retired non-goal keeps its ID with status `RETIRED`. Amendments (§7) are
counted against these IDs.

| ID | Non-goal | Status | Active carve-outs |
|------|----------|--------|-------------------|
| NG-1 | {e.g., "The system does not X"} | ACTIVE | 0 / 2 |
| NG-2 | {e.g., "The system does not Y"} | ACTIVE | 0 / 2 |
| NG-3 | {e.g., "The system does not Z"} | ACTIVE | 0 / 2 |

*(Rationale note per non-goal is encouraged: one sentence on **why** the boundary
exists — safety, liability, focus, dependency risk. A non-goal without a rationale is
the first one to erode.)*

## 4. Invariants (must not drift — enforce in review)

*(Rules that hold across every scope change, including ratified carve-outs. Written so a
reviewer — human or agent — can mechanically check compliance. Carve-outs may expand
scope; they may never relax an invariant.)*

1. **{Extensibility rule}** — new {plugins/transports/backends} are added through
   {registry/extension mechanism}; core {router/engine/models} are not edited per
   addition.
2. **{Compatibility rule}** — {versioning scheme}; breaking changes follow the full bump
   ritual (version + tests + docs + CHANGELOG).
3. **{Testability rule}** — dependencies are injected via {Protocol/interface}; unit
   tests use fakes and require no {hardware/network/live services}.
4. **{Purity/layering rule}** — {stateful concerns} live in {layer A}, not {layer B}.
5. **{Configuration rule}** — every operational value is a config field with an explicit
   default; no magic numbers.
6. **{Quality-gate rule}** — {linters, type checks, test suite} at {coverage floor} stay
   green; {narrow exception, e.g., hardware glue} is the only permitted exclusion.
7. **{Security rule}** — no secrets or machine fingerprints in the repo; credentials
   live in runtime config.

## 5. Long-term roadmap (themes, not dates)

> Detailed milestones live in `ROADMAP.md`; keep the two aligned — milestone detail
> changes there, not here.

- **M1 — {MVP theme}:** {one line} *(status)*
- **M2 — {Hardening theme}:** {one line}
- **M3 — {Enrichment theme}:** {one line}
- **M4 — {Scale/resilience theme}:** {one line}
- **M5 — {Packaging/release theme}:** {one line}

## 6. How agents and contributors use this document

- Read this **before** planning a task; keep changes consistent with §3 scope and §4
  invariants.
- Put concrete, changeable to-dos in `NEXT_STEPS.md`, not here.
- When a change would violate an invariant or expand scope, **surface it for a human
  decision** — do not implement first and ratify later.
- Agents must treat the §7 budgets as hard constraints: if a proposed task requires a
  carve-out and the relevant budget is exhausted or a trigger has fired, the correct
  output is a *charter-review request*, not an amendment draft.

## 7. Amendment Protocol (carve-out budget & review triggers)

Non-goals are amended only through **carve-outs**: dated, ratified, bounded exceptions
appended as blockquotes under §3. The default stance always remains the non-goal; the
carve-out is the exception.

### 7.1 Required carve-out format

Every carve-out MUST contain all of the following, or it is invalid and unenforceable:

> **Carve-out CO-{n} (deliberate amendment — ratified by {ROLE} on {DATE} per §7,
> against NG-{id}): {short title}.**
> {One paragraph: exactly what is now permitted, and for which component only.}
> Constraints (all required):
> - **{Bounding constraint}** — what the exception explicitly does NOT permit.
> - **{Mechanism constraint}** — how it is added (via the §4 extension seams, never by
>   editing core).
> - **{Safety/audit constraint}** — gates, confirmations, logging, default-off flags.
> - **{Sequencing constraint}** — what must land before this ships, if anything.
>
> This remains a bounded exception; every other §3 non-goal still stands.

### 7.2 Budgets (hard limits, not guidelines)

- **Per-non-goal budget:** at most **2 active carve-outs** per non-goal ID. A **third
  proposed amendment to the same non-goal automatically closes the amendment path** for
  that non-goal and forces a full charter review (§7.4). The review must choose one of:
  (a) rewrite the non-goal to honestly reflect the new scope, (b) reject the proposal
  and reaffirm the boundary, or (c) split the capability into a separate project with
  its own charter. "Ratify a third carve-out" is not an available outcome.
- **Global density trigger:** **3 or more ratified carve-outs within any rolling 90-day
  window — across all non-goals — forces a full charter review.** This catches the
  ratchet that per-ID counting misses: many small exceptions to *different* boundaries
  that together transform what the system is.
- **Cumulative trigger:** when total active carve-outs reach **50% of the number of
  non-goals**, the next amendment proposal forces a review regardless of the other two
  triggers.

### 7.3 Anti-gaming rules

- **No splitting.** A proposal may not be divided into multiple smaller carve-outs to
  stay under budget; reviewers should count intent, not paperwork.
- **No non-goal inflation.** Adding new narrowly-worded non-goals solely to reset
  denominators for the cumulative trigger is itself a charter change requiring review.
- **Carve-outs expire on rewrite.** After a charter review rewrites a non-goal, its
  prior carve-outs are folded into the new text and retired; counters reset only
  through review, never administratively.

### 7.4 Full charter review (what a trigger forces)

A full charter review is a deliberate, human-led session (not an agent task) that:

1. Re-reads §1–§3 against what the system has actually become.
2. Rewrites Vision/Mission/Scope so the *default text* — not the exception list —
   describes reality.
3. Re-ratifies or retires each invariant.
4. Updates the Executive Summary and the Carve-out Ledger (§8).
5. Records the review date in §0. Until the review completes, **no new carve-outs may
   be ratified** and agents must treat pending amendment proposals as blocked.

### 7.5 Ratification authority

- Amendments are ratified by {ROLE — e.g., project maintainer}; charter reviews require
  {ROLE(S) — e.g., maintainer + one independent reviewer}. An amendment ratified by the
  same person who proposed it, on the same day, should be treated as a smell even when
  procedurally valid — the protocol works only if ratification is a genuine checkpoint,
  not a formality.

## 8. Carve-out Ledger (append-only)

| CO | Date | NG | Title | Ratified by | Status |
|----|------|----|-------|-------------|--------|
| — | — | — | *(none yet)* | — | — |

**Budget state:** {N} active carve-outs / {non-goal count} non-goals · last 90-day
window: {n} ratifications · last full review: {DATE or "never"}.
