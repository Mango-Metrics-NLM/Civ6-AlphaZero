# DEV-SQE → Pre-PR Handoff

**Date:** 2026-11-20
**Spec:** kb/specs/20261119-game-mechanics-plan.md (slice GM-B0; OpenSpec `openspec/changes/add-bridge-research-parity`, tasks 1.1 to 1.3)
**Branch:** `claude/gm-b0-lua-bundle`, from the default branch (not stacked)
**Files Changed:**
- `scripts/build_lua_bundle.py`
- `mod/AlphaZeroAI/Scripts/Config.lua`
- `mod/AlphaZeroAI/Scripts/ActionExecutor.lua`
- `mod/AlphaZeroAI/Scripts/Main.lua`
- `mod/AlphaZeroAI/AlphaZeroAI.modinfo`
- `openspec/changes/add-bridge-research-parity/bundle-drift.md`
- `openspec/changes/add-bridge-research-parity/tasks.md`
- `docs/LIVE_CIV6.md`
- `CLAUDE.md`
- `CHANGELOG.md`

**Tests Added:** `tests/scripts/test_lua_bundle_sync.py` (61 tests)
**Test Results:** 61 pass, 0 fail in the new file. Suite wiring, debt ratchet, public-readiness, `tests/governance` and the existing tests that read the mod's Lua files pass. In `tests/scripts` as a whole, 7 tests fail for environment reasons that predate this change (`test_deployment_infra.py` needs a `.env` file; `test_gcp_diagnose.py` needs Google credentials). Coverage: not applicable, nothing under `alpha_zero/` changed.
**Lint Status:** `ruff check .` clean; the new files are formatted.
**Type Check Status:** `scripts/build_lua_bundle.py` clean in a single-file mypy run; CI's job is the gate.
**Smoke Tests:** `python scripts/build_lua_bundle.py --check` exits 0. `python -m alpha_zero.civ6_local --mode civ6 --dry-run` not run; nothing it imports changed.
**Verification not done:** `make verify`; **nothing was run against a real game**. The claim that live behaviour is unchanged rests on the bundle being byte-identical (sha256 `e33177f9c33c11e1fcf6e2986bec3d97097868ef51fafce39f0caa7ba80622f6` before and after), not on a live session.
**Play path:** the diff against the default branch over `scripts/release_play_path.txt` is empty.
**Known Issues:**
- Reconciling the module files to the bundle removed two things from the module files that never ran in the game: diplomacy error reporting in `ActionExecutor.lua` and a configurable heartbeat interval in `Main.lua`. They are recorded in `bundle-drift.md`; bringing them back changes live behaviour and belongs to GM-B2.
- The modinfo is read with patterns, not an XML parser (the standard parsers are flagged by the lint rules); any entry the patterns cannot read is refused, never skipped.
- Line endings are normalised in the generator; no `.gitattributes` rule was added.
- The plan said 27 differing lines; the count of non-blank lines is 24.
- This branch appends to `kb/ledger/ledger.jsonl` and `kb/summaries/dev-sqe.md`, as the other open branches do; later merges need the default branch merged in first.
