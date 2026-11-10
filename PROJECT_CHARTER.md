# Project Charter — Civ6-AlphaZero (stable long-term plan)

> **Status: STABLE / SLOW-CHANGING.** This document is the north-star that humans and AI
> agents (and sub-agents) read first to stay on track. It changes rarely and only by
> deliberate decision — not per task. Day-to-day, changeable work lives in
> [`ROADMAP.md`](ROADMAP.md) (Now / Next / Later) and the open change under
> [`openspec/changes/`](openspec/changes/); architecture detail in
> [`docs/architecture/C4_ARCHITECTURE.md`](docs/architecture/C4_ARCHITECTURE.md). If a task seems to require changing
> this charter, **stop and raise it explicitly** rather than editing it in passing.
>
> **Charter integrity is governed by §7 (Amendment Protocol).** Amendments are budgeted.
> When a budget or trigger fires, the amendment path closes and a full charter review is
> mandatory. Procedural tidiness is not a substitute for scope discipline.

---

## 0. Executive Summary

Civ6-AlphaZero is an AlphaZero-style reinforcement-learning system that trains strategic
agents by self-play in a simplified Civilization environment (MicroCiv) using Monte Carlo
Tree Search and deep neural networks, scales that training on GCP Vertex AI, and deploys
trained agents to play a live, unmodified Civilization VI through a Lua-mod/FireTuner
bridge. It serves a single maintainer-researcher. Its two hardest boundaries: it **never
trains against the real Civ6 game** (MicroCiv is the sole training environment, NG-1),
and the **core learning loop never depends on an LLM** (all LLM guidance is opt-in,
default-off, and fail-safe, NG-2). The charter has **0 active carve-outs** against a
budget of 2 per non-goal across 7 non-goals; the last full charter review was
**2026-07-13** (initial ratification). Execution detail lives in `ROADMAP.md` (Now / Next / Later); this
document defines intent, boundaries, and invariants only.

---

## 1. Vision

A single researcher can train a strategy agent from nothing — on a CPU laptop or a
budget cloud fleet — and then watch that same agent autonomously play a real, unmodified
copy of Civilization VI. The project's defining split is *train cheap, deploy real*:
learning happens in MicroCiv, a fast simplified 4X environment, while the payoff is
delivered through a bridge into the actual game. Everything is reproducible without a
GPU, scalable to distributed self-play when budget allows, and engineered like a product
rather than a research script: typed, tested, gated, and observable. The end state is an
agent whose skill demonstrably compounds across training generations and holds up in
live Civ6 play — proof that AlphaZero-class self-play can master an open-ended 4X
strategy game, not just a board game.

## 2. Mission (what we are building)

- **Learn from self-play.** Train MicroCiv agents whose loss decreases and which earn
  promotion over incumbent champions. **Validation status (2026-11-10):** not yet
  demonstrated — every committed run predates the 2026-09-04 search-correctness fixes
  (`docs/evidence/README.md`); the first post-fix run is specified in
  `docs/EVALUATION.md` §7.
- **Search efficiently.** Pluggable MCTS behind one factory — `create_mcts(...)` with
  GumbelMCTS as the default and batched self-play (`BatchGumbelMCTS` +
  `VectorizedMicroCivEnv`); throughput gains are targets until a committed benchmark
  artifact measures them.
- **Train at scale.** A 3-phase distributed pipeline (self-play → train → eval) on
  Vertex AI — `infra/vertex_launcher.py`, containerized workers, completed 50-iteration
  production runs.
- **Play the real game.** Deliver agent decisions into a running Civilization VI
  instance headlessly — `alpha_zero/bridge/tuner_socket_bridge.py` (FireTuner socket,
  loopback only), live autonomous turns verified 2026-07-12 (v1.8.0).
- **Support N players.** 2–6 player MicroCiv with per-player value heads, DUCT
  simultaneous actions, and an N-player training pipeline —
  `alpha_zero/multiplayer/`, `tests/unit/test_nplayer_training_pipeline.py`.
- **Observe and evaluate everything.** MLflow experiment tracking, ELO tournaments, and
  an offline LLM-advisor contract gate — `alpha_zero/monitoring/`,
  `alpha_zero/tournament/`, `make eval-llm`.

## 3. Scope

**In scope:** the `alpha_zero/` package (environments, MCTS variants, networks,
distributed/cloud training, storage backends, monitoring, multiplayer, the Civ6 bridge
and its Lua mod, orchestrator, RAG/LLM advisory layer), `tests/`, `configs/`, `scripts/`,
`infra/` + `dockerfiles/` (Vertex AI deployment), and project documentation.

**Out of scope (non-goals):** Each non-goal carries a stable ID. IDs are never reused or
renumbered; a retired non-goal keeps its ID with status `RETIRED`. Amendments (§7) are
counted against these IDs.

| ID | Non-goal | Status | Active carve-outs |
|------|----------|--------|-------------------|
| NG-1 | The system does not train against the real Civ6 game — the bridge is inference/evaluation only | ACTIVE | 0 / 2 |
| NG-2 | The core learning loop does not depend on an LLM — all LLM guidance is opt-in, default-off, fail-safe | ACTIVE | 0 / 2 |
| NG-3 | The system does not modify Civ6 game rules or content | ACTIVE | 0 / 2 |
| NG-4 | No workflow requires a GPU | ACTIVE | 0 / 2 |
| NG-5 | No new Weights & Biases instrumentation | ACTIVE | 0 / 2 |
| NG-6 | This is not a general-purpose game-AI framework | ACTIVE | 0 / 2 |
| NG-7 | No unattended large-scale or paid runs | ACTIVE | 0 / 2 |

Rationale per non-goal:

- **NG-1** — Real Civ6 is non-headless, non-resettable, and orders of magnitude too slow
  for self-play; MicroCiv is the sole training environment, and blurring this line would
  sink the project's train-cheap/deploy-real economics.
- **NG-2** — Reproducibility and cost: self-play training and the default GumbelMCTS
  path must run with zero LLM calls. Existing LLM hooks (`Civ6MCTS`'s optional
  `llm_prior_fn` blend, reflection memory, `StrategyAdvisor`) are permitted **only** as
  opt-in, default-off enhancements that degrade gracefully when the LLM fails — never as
  required dependencies of training or evaluation.
- **NG-3** — The Lua mod observes state and injects commands strictly through official
  surfaces (`ExposedMembers`, `GameEvents`, the FireTuner protocol); altering game rules
  would invalidate evaluation against the real game.
- **NG-4** — CPU-must-run is a hard constraint (CLAUDE.md §2.1); GPUs are accelerators,
  never dependencies, so the project stays reproducible on any contributor machine.
- **NG-5** — MLflow (`MLflowConfig`, single-source) is the experiment-tracking system of
  record; legacy wandb calls remain in `cloud/`, `pbt/`, `evaluation/`, and
  `optimizations/` pending the migration sweep, but new code must not add direct wandb
  instrumentation — two tracking systems accumulating independently is drift by design.
- **NG-6** — The Unciv, chess/LM-Studio, and Strategic-Planning-SDK directions are
  experiments, not product scope; the product is MicroCiv training plus real-Civ6
  deployment, and generalization stays in the long-term-vision column until deliberately
  chartered (this is the focus boundary most likely to erode).
- **NG-7** — Runs above 1000 games and any cloud spend require explicit human
  confirmation (hook-enforced); the v1.2.0 production run that wasted ~$28.55 producing
  zero promotions is the standing justification.

## 4. Invariants (must not drift — enforce in review)

*(Rules that hold across every scope change, including ratified carve-outs. Written so a
reviewer — human or agent — can mechanically check compliance. Carve-outs may expand
scope; they may never relax an invariant.)*

1. **Extensibility** — new MCTS variants, encoders, and storage backends are added
   through the existing factory/registry seams (the `create_mcts(...)` variant
   registry in `alpha_zero/core/mcts_factory.py`, the encoder registry,
   `alpha_zero/storage/factory.py` against the abstract `StorageBackend`); callers are
   not edited per addition and do not branch on concrete component types.
2. **Compatibility** — every code path runs on CPU-only Python ≥ 3.11 (the floor
   `pyproject.toml` enforces; raised from 3.10 on 2026-11-10 to match it) with Linux CI
   green; the documented Windows deselect list is the only permitted platform
   divergence.
3. **Testability** — the full suite stays green with line coverage ≥ 85%, enforced by
   `fail_under` in `pyproject.toml` and the CI gate; new modules ship with tests, and
   the documented `[tool.coverage]` omit list is the only permitted exclusion.
4. **Layering** — `alpha_zero/envs/` and the MCTS action-selection path perform no
   network, cloud, LLM, or experiment-tracking calls; those integrations live in
   `orchestrator/`, `cloud/`, and `monitoring/`. (Scoped to the action path:
   `core/reflection_memory.py`'s vector-store import is a memory store behind an
   optional, default-off hook, not action selection.)
5. **Configuration** — every operational value is a (frozen where practical) dataclass
   config field with an explicit default, overridable via the `CIV6AZ_*` environment
   convention loaded through `alpha_zero/utils/env_loader.py`; no hardcoded paths,
   ports, or magic numbers in new code.
6. **Quality gate** — no direct commits to `main` and no >1000-game runs without
   confirmation, mechanically enforced by the PreToolUse hook
   (`.claude/hooks/pretooluse_guard.sh` via `.claude/settings.json`); `ruff` (100-char)
   and `mypy` stay green.
7. **Security** — no secrets or machine fingerprints in the repo; credentials live only
   in the gitignored `.env` / environment, and the Civ6 bridge binds loopback only
   (`127.0.0.1:4318`) and touches only official game API surfaces.

## 5. Long-term roadmap (themes, not dates)

> Detailed milestones live in [`ROADMAP.md`](ROADMAP.md); keep the two aligned —
> milestone detail changes there, not here.

- **M1 — Learning Core:** AlphaZero on MicroCiv, built end-to-end. *(delivered; learning not yet validated on the corrected search — see `ROADMAP.md`'s validation column)*
- **M2 — Scale & Observability:** Vertex AI 3-phase pipeline, GCS storage, MLflow monitoring. *(delivered)*
- **M3 — Live Game Bridge:** agent plays real Civ6 headlessly via the FireTuner socket. *(delivered)*
- **M4 — Multiplayer & Throughput:** N-player training runs, batch self-play in cloud, test-debt paydown, coverage re-tighten. *(active)*
- **M5 — Full Civ6 Mastery:** fog of war, full-scale encoder/action space, sustained skilled live play. *(future)*

## 6. How agents and contributors use this document

- Read this **before** planning a task; keep changes consistent with §3 scope and §4
  invariants.
- Authority order when documents disagree: this charter (what and what-not) →
  `CLAUDE.md` (how: gates, permissions, KB workflow) → `ROADMAP.md` Now / Next / Later
  and the open OpenSpec change (what now and what next) →
  `docs/architecture/C4_ARCHITECTURE.md`, `docs/MODULE_REFERENCE.md` (how it is built).
- Put concrete, changeable to-dos in `ROADMAP.md`'s Now / Next / Later or a GitHub
  issue, not here.
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
  non-goals** (currently 4 of 7, rounding up), the next amendment proposal forces a
  review regardless of the other two triggers.

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

This repository has a single maintainer — **ianshank (Mango-Metrics-NLM)** — who is the
sole ratification authority for amendments and the convener of charter reviews.
Ratification is expressed as a maintainer-merged pull request touching this file, never
a direct commit to `main` (Invariant 6). AI agents and contributors may **propose**
amendments and carve-outs via PR but can never self-ratify; an agent-authored change to
this charter that has not been merged by the maintainer has no force. Two honest caveats
of solo governance: (a) an amendment proposed and ratified by the same person on the
same day is a smell even when procedurally valid — sleep on it; (b) the anti-gaming
rules mechanically bind agents, but for the maintainer they are self-discipline made
visible — the ledger (§8) exists precisely so the maintainer's own drift leaves a paper
trail.

## 8. Carve-out Ledger (append-only)

| CO | Date | NG | Title | Ratified by | Status |
|----|------|----|-------|-------------|--------|
| — | — | — | *(none yet)* | — | — |

**Budget state:** 0 active carve-outs / 7 non-goals · last 90-day window: 0
ratifications · last full review: 2026-07-13 (initial ratification).

---

*Instantiated from [`docs/templates/PROJECT_CHARTER_TEMPLATE.md`](docs/templates/PROJECT_CHARTER_TEMPLATE.md)
at v1.8.0, 2026-07-13.*
