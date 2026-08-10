# DEV-SQE → Pre-PR Handoff

**Date:** 2026-07-13
**Spec:** kb/specs/20260713-project-charter.md
**Files Changed:**
- `PROJECT_CHARTER.md` (new — instantiated charter: 7 non-goals, 7 invariants, amendment protocol, carve-out ledger)
- `templates/PROJECT_CHARTER_TEMPLATE.md` (new — generic template, placeholders intact)
- `ROADMAP.md` (new — M1–M5 themes, not dates)
- `CLAUDE.md` (two pointer insertions: §6.1 step 1, §13 header note)
- `README.md` (one governance link line above Project Status)
- `kb/specs/20260713-project-charter.md` (new)

**Tests Added:** none (docs-only change; no code or tests touched)
**Test Results:** not run — pytest/torch are not installable in this restricted session
(known issue, CLAUDE.md §5.3); **CI must be the gate**. No Python files were modified,
so suite results are expected identical to HEAD.
**Lint Status:** `ruff check alpha_zero/` → 1 pre-existing C420
(`alpha_zero/multiplayer/mcts.py:310`), verified identical on clean HEAD (stash test);
not introduced by this change.
**Type Check Status:** not run (mypy unavailable in session); no Python files changed.
**Smoke Tests:** n/a (docs-only). Manual checks performed: all relative markdown links
in the 3 new + 2 edited files resolve; version/date/count consistency across charter
§0/§3/§8 (v1.8.0, 2026-07-13, 7 non-goals, 0 carve-outs).
**Known Issues:**
- CLAUDE.md footer still says v1.7.0/2026-06-20 while CHANGELOG/README are at v1.8.0 —
  deliberately left out of scope (footer-only bump would misrepresent body content);
  flagged for follow-up.
- NG-2 was reworded from the planned "no LLM-in-the-loop action selection" to "core
  loop does not *depend* on an LLM" after verifying `Civ6MCTS` ships an opt-in,
  default-off `llm_prior_fn` blend and reflection-memory hooks.
