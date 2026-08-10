# Spec: Project Charter Instantiation (2026-07-13)

**Source:** user-supplied charter template + user-approved in-session plan
(session_01AAR7bMpmkZtman9kFaeqft, branch `claude/project-charter-template-b4n9fu`).

## Deliverables

1. `PROJECT_CHARTER.md` (root) — template instantiated for Civ6-AlphaZero: executive
   summary, vision, 6 verifiable mission bullets, 7 non-goals (NG-1..NG-7, stable IDs,
   0/2 carve-out budgets), 7 invariants naming their real enforcement mechanisms,
   M1–M5 roadmap themes, amendment protocol with budgets/triggers, empty carve-out
   ledger. Ratifier: sole maintainer (ianshank / Mango-Metrics-NLM).
2. `templates/PROJECT_CHARTER_TEMPLATE.md` — generic template, placeholders intact.
3. `ROADMAP.md` (root) — themes-not-dates M1–M5, distilled from README phase table,
   CLAUDE.md §13, NEXT_STEPS.md, docs/NEXT_STEPS_ANALYSIS_2026-07.md.
4. Pointer edits only: CLAUDE.md §6.1 step 1 + §13 header note; README.md governance
   line. CLAUDE.md v1.7.0 footer drift explicitly out of scope (noted for follow-up).

## Key content decisions

- NG-2 reworded during implementation after first-hand verification: `Civ6MCTS` has an
  opt-in `llm_prior_fn` blend + reflection-memory hooks (both default-off; default
  GumbelMCTS path has neither), so the boundary is "core loop never *depends* on an
  LLM" rather than "no LLM in action selection".
- Mission bullet cites `scripts/validate_convergence.py` (not the deprecated
  `validate_v130_training.py` shim).
- Invariant 4 (layering) scoped to envs/ + the MCTS action-selection path so
  `core/reflection_memory.py`'s optional vector-store import doesn't falsify it.

## Acceptance

- All relative links in new/edited markdown resolve.
- v1.8.0 / 2026-07-13 / 7 non-goals / 0 carve-outs consistent across §0, §3, §8.
- No code touched; `ruff check alpha_zero/` unchanged from HEAD.
