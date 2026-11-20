# DEV-SQE → Pre-PR Handoff

**Date:** 2026-11-20
**Spec:** kb/specs/20261119-game-mechanics-plan.md (slice GM-B1; OpenSpec `openspec/changes/add-bridge-research-parity`, tasks 2.1 to 2.3)
**Branch:** `claude/gm-b1-lua-harness`, from the default branch (not stacked; independent of the GM-B0 branch)
**Files Changed:**
- `tests/fixtures/lua/civ6_stub.lua`
- `tests/fixtures/lua_harness.py`
- `.github/workflows/ci.yml`
- `pyproject.toml`
- `docs/evidence/tech_debt_baseline/baseline.json`
- `CLAUDE.md`
- `CHANGELOG.md`
- `openspec/changes/add-bridge-research-parity/tasks.md`

**Tests Added:** `tests/unit/test_lua_harness.py` (71 tests)
**Test Results:** 71 pass, 0 fail. Suite wiring, the optional-import gate, debt ratchet, public-readiness and `tests/governance` pass. Coverage: not applicable, nothing under `alpha_zero/` changed.
**Lint Status:** `ruff check .` clean; new files formatted.
**Type Check Status:** not applicable to the package; the harness is test code.
**Smoke Tests:** the mod's bundle loads under Lua 5.1 with and without the stand-in; the exporter's JSON parses with the Python protocol reader with no warnings.
**Verification not done:** `make verify`; nothing was run against a real game; whether the game's sandbox lacks every name the lockdown removes is not verified (only `io` is documented).
**Play path:** the diff against the default branch over `scripts/release_play_path.txt` is empty; `mod/` is untouched.
**Debt baseline:** `pytest_skip_sites` 148 -> 149, deliberate: the harness skips when `lupa` is absent (and fails under `CIV6AZ_REQUIRE_LUA=1`). PR #147 moves the same number for another reason, so the second of the two to merge must set it to 150.
**Known Issues (the mod's Lua, pinned not fixed; inputs for GM-B2):**
- Move targets include occupied tiles: a unit is offered a move onto a tile holding an enemy unit or a friendly unit.
- A `produce` action is offered with nothing to produce (empty `items`) when a city's queue is empty.
- One missing game method empties the whole player block to `{"id": 0}`, not only the field it feeds.
- `Game.GetCurrentGameTurn` is the one call in the exporter not guarded by `pcall`.
- `players` lists only the local player.
- `UI.RequestAction` raising gives `EXECUTION_FAILED`; only a missing `UI` gives "End turn requested".
- Known gaps pinned as today's behaviour: researched techs not exported; current research a raw index; no buildings or districts per city; no research in legal actions; no `schema_version`; END_TURN answers "Turn ended" unconditionally.
- The stand-in's units and map do not change when a command runs, so move, attack and found-city commands cannot be tested end to end yet.
