# DEV-SQE → Pre-PR Handoff

**Date:** 2026-11-19
**Spec:** kb/specs/20261119-game-mechanics-plan.md (slice GM-K2; OpenSpec `openspec/changes/add-game-rules-kernel`, tasks 2.1 to 2.4 and 3.1)
**PR:** draft #147, branch `claude/gm-k2-rules-tooling`, base the default branch; contains GM-K1 (PR #146) until that merges
**Files Changed:**
- `alpha_zero/rules/pacing.py`
- `alpha_zero/rules/gameinfo.py`
- `alpha_zero/rules/__main__.py`
- `alpha_zero/rules/Agent.md`
- `docs/MODULE_REFERENCE.md`
- `docs/GAME_RULES.md`
- `CHANGELOG.md`
- `CLAUDE.md`
- `openspec/changes/add-game-rules-kernel/tasks.md`

**Tests Added:**
- `tests/unit/test_rules/test_pacing.py`
- `tests/unit/test_rules/test_gameinfo.py`
- `tests/unit/test_rules/test_no_gameinfo_in_repo.py`
- `tests/fixtures/gameinfo/synthetic_export.json`

**Test Results:** 315 pass, 0 fail in `tests/unit/test_rules` and `tests/regression/test_game_rules_log_regression.py`; coverage of `alpha_zero/rules` 100% (branch coverage on). 841 pass across the debt ratchet, suite wiring, public-readiness, play-path closure and `tests/governance`.
**Lint Status:** `ruff check .` clean; `ruff format --check .` clean (1005 files).
**Type Check Status:** each module of `alpha_zero/rules` clean in a single-file mypy run. `mypy alpha_zero/` does not run in this sandbox (it stops on the installed numpy's stub file), so CI's Lint and Type Check job is the gate.
**Smoke Tests:** `python -m alpha_zero.rules pace science-v1-draft` prints 780 and 580 and exits 0; an import to a path outside the checkout exits 0 and validates; a path under `docs/` exits 2 and writes nothing. Training smoke tests not run; nothing they import changed.
**Verification not done:** `make verify` past its `typecheck` step (see above), so the suites it would run after that step were not run through it; `make test-all`.
**Play path:** the diff against the default branch over `scripts/release_play_path.txt` is empty.
**Known Issues:**
- At 60 turns the first-pass capstone needs 14 to 16 science per turn; the draft costs are too high for that game length. Nothing is tuned here; calibration settles it.
- `Civics` and `CivicPrereqs` are counted, not imported; the design note can be read either way.
- A reference to a technology skipped for a non-positive cost is an error; real exports may need a rule for it (GM-B4 will show).
- `band_science_rates` does not list rates above the cost, all of which take one turn.
- PRs #143, #146 and #147 each append to `kb/ledger/ledger.jsonl` and `kb/summaries/dev-sqe.md`; later merges need the default branch merged in first.
