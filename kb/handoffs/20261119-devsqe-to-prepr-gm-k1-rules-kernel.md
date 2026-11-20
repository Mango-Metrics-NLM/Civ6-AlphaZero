# DEV-SQE → Pre-PR Handoff

**Date:** 2026-11-19
**Spec:** kb/specs/20261119-game-mechanics-plan.md (slice GM-K1; OpenSpec `openspec/changes/add-game-rules-kernel`, tasks 1.1 to 1.9)
**PR:** draft #144, branch `claude/gm-k1-rules-kernel`, stacked on #142's branch
**Files Changed:**
- `alpha_zero/rules/__init__.py`
- `alpha_zero/rules/__main__.py`
- `alpha_zero/rules/spec.py`
- `alpha_zero/rules/loader.py`
- `alpha_zero/rules/validate.py`
- `alpha_zero/rules/queries.py`
- `alpha_zero/rules/registry.py`
- `alpha_zero/rules/builtin.py`
- `alpha_zero/rules/data/science_v1_draft.yaml`
- `alpha_zero/rules/Agent.md`
- `pyproject.toml`
- `docs/GAME_RULES.md`
- `docs/MODULE_REFERENCE.md`
- `CHANGELOG.md`
- `openspec/changes/add-game-rules-kernel/tasks.md`

**Tests Added:** `tests/unit/test_rules/` (six modules and a fixtures helper), `tests/regression/test_game_rules_log_regression.py`
**Test Results:** 166 pass, 0 fail; coverage of `alpha_zero/rules` 100% (branch coverage on). Debt ratchet, suite wiring, play-path closure and `tests/governance` pass.
**Lint Status:** `ruff check .` clean; new files formatted.
**Type Check Status:** `alpha_zero/rules/loader.py` clean in a single-file mypy run after `a897315`; the whole-package run does not work in this sandbox (numpy stubs), so CI's Lint and Type Check job is the gate and was green on `a897315`.
**Smoke Tests:** `python -m alpha_zero.rules validate science-v1-draft` exits 0; `hash` prints `0c336c21e209a76a63fd9b70d2c2e8dff18fb0d326562e8b362989e159d18c3c`. The training smoke tests were not run; nothing they import changed.
**Verification not done:** `make verify` and `make test-all` in full; loading the rule set from an installed wheel.
**Play path:** the diff against the default branch over `scripts/release_play_path.txt` is empty.
**Known Issues:**
- Unit and building costs in `science_v1_draft.yaml` are first-pass values chosen in this slice; the plan gives tech costs only.
- `hash` on a file does not validate it; `build_registry` does not validate its input.
- The rule `unreachable-capstone` can only fire alone when the capstone names no tech (resolved, acyclic prerequisites make every declared tech reachable).
- `CLAUDE.md` is not updated here: its command rows are OpenSpec task 3.1, after GM-K2; it still says 30 submodules where `docs/MODULE_REFERENCE.md` says 31.
- PRs #142, #143 and #144 each append to `kb/ledger/ledger.jsonl` and `kb/summaries/dev-sqe.md`; later merges will conflict there until the default branch is merged in.
