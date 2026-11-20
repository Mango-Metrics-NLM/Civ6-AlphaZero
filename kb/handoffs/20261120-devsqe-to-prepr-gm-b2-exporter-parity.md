# DEV-SQE → Pre-PR Handoff

**Date:** 2026-11-20
**Spec:** kb/specs/20261119-game-mechanics-plan.md (slice GM-B2; OpenSpec `openspec/changes/add-bridge-research-parity`, tasks 3.1 to 3.4 and 3.4a)
**Branch:** `claude/gm-b2-exporter-parity`, from the default branch with `claude/gm-b0-lua-bundle` (#148) and `claude/gm-b1-lua-harness` (#149) merged in; its PR shows their commits until they merge
**Files Changed:**
- `mod/AlphaZeroAI/Scripts/StateExporter.lua`
- `mod/AlphaZeroAI/Scripts/ActionExecutor.lua`
- `mod/AlphaZeroAI/Scripts/Main.lua`
- `mod/AlphaZeroAI/Scripts/AlphaZeroMain.lua`
- `alpha_zero/bridge/protocol.py`
- `alpha_zero/bridge/agent_loop.py`
- `tests/fixtures/lua/civ6_stub.lua`
- `tests/unit/test_lua_harness.py`
- `docs/LIVE_CIV6.md`
- `CHANGELOG.md`
- `CLAUDE.md`
- `openspec/changes/add-bridge-research-parity/tasks.md`
- `openspec/changes/add-bridge-research-parity/specs/civ6-bridge/spec.md`
- `openspec/changes/add-bridge-research-parity/bundle-drift.md`

**Tests Added:** `tests/unit/test_lua_exporter_parity.py` (158), `tests/unit/test_bridge_research_parity.py` (29); `tests/unit/test_lua_harness.py` updated (64)
**Test Results:** 251 pass in those three files; `tests/scripts/test_lua_bundle_sync.py` 62 pass; `tests/unit` + `tests/integration` (`-m "not slow"`): 8,426 passed, 1 failed (`test_pinned_env.py::test_git_archive_of_head_is_extracted_and_read_only`, "DID NOT RAISE PermissionError": the sandbox runs as root; unrelated, and it ran before the review fixes). Debt ratchet, suite wiring, public-readiness, play-path closure, `tests/governance` pass. Coverage not measured locally (the sandbox's numpy breaks the tracer); CI is the gate.
**Lint Status:** `ruff check .` clean; `ruff format --check` clean.
**Type Check Status:** `python -m mypy alpha_zero/bridge/protocol.py alpha_zero/bridge/agent_loop.py --follow-imports=silent`: no issues. `mypy alpha_zero/` cannot finish in this sandbox; CI is the gate.
**Smoke Tests:** the regenerated bundle loads under Lua 5.1 and its export parses with the Python reader with the new fields (`TestRealExport`). `python scripts/build_lua_bundle.py --check` exits 0. Bundle sha256 `082a62014115f863acec92e9a09e0748eb1e685afeeebc92a471235d21b80297`.
**Verification not done:** `make verify`; **nothing was run against a real game**.
**Play path:** the diff against the default branch over `scripts/release_play_path.txt` is empty.

## What changed
- **Exporter (state schema 2):** `schema_version`; researched techs and civics by type name; `current_research` as a type name (was a raw index); `research_status` / `civic_status` (cost, progress, turns left, boost); own cities' `buildings` and finished `districts`; `research_tech` / `research_civic` legal actions with `items` (indexes) and `item_names`; `end_turn_blocking` and its message.
- **Executor:** `end_turn` answers `END_TURN_BLOCKED` and sends nothing while the game reports a blocker, unless `UI.CanEndTurn()` is true; with no `UI.RequestAction` it fails. Research commands take an index or a type name, refuse what is already researched or closed, request through `UI.RequestPlayerOperation` with the row's hash and an exclusive insert mode, and fall back to the GameCore setter.
- **Python:** additive protocol fields; the agent loop counts `blocked_end_turns` and accepts the socket transport's dict ack.
- **Drift pieces** (maintainer decision: bring both back): diplomacy error reporting; config-read heartbeat interval (plus a fallback for a non-numeric value).

## Review of record
- Tests by `sdlc-qa-sdet` in a separate context, against a written wire contract, before the Lua existed: 106 failed on the old bundle; after the review the contract changed and 46 failed on the then-current bundle.
- Review by `sdlc-architect` (read-only). Findings and what was done:
  1. Research request sent the row index; the game's choosers send the hash. **Fixed**, after reading the game's own UI Lua in CQUI Community Edition. Negative control: with the index put back the stub applies 0 requests and the state-follows test fails.
  2. A choice the game cannot accept still succeeded and the blocker was dismissed. **Fixed**: refusals, and no dismissal at all.
  3. On the socket transport the loop read the ack dict as an object: every acknowledged command was followed by a fallback `end_turn`, and the new handling could not run. **Fixed** in `agent_loop.py` (pre-existing defect).
  4. A refused `end_turn` repeats until the mod's 30 s command timeout; `end_turn` is still listed as legal while blocked; the agent cannot clear blockers it has no action for. **Not fixed, documented** (`docs/LIVE_CIV6.md`). The blocker is now read with the game's own query.
  5. `alpha_zero/envs/real_civ6_env.py` penalises a blocked `end_turn` as an invalid action and leaves the re-exported state unread. **Not fixed**: the file is on the release play path. Known issue for after v2.1.0.
  6. Spec wording. **Fixed**: one legal action per kind with aligned items; `current_research` named as the one retyped field.
  7. Districts under construction were listed. **Fixed** (`IsComplete`).
  8. A wrong comment in `protocol.py`. **Fixed**.

## Known issues and open questions
- **Unverified against a game:** every game call. Sources: the game's UI Lua as redistributed in CQUI Community Edition (a 2018 snapshot) and civ6-mcp. `docs/LIVE_CIV6.md` has a five-step live checklist.
- **One assumption carries the blocked-turn flow:** that the game clears its own "choose research" notification when a request takes effect. The stub encodes it and says so. If it is false, a turn stays `END_TURN_BLOCKED` after a successful choice; the fix would be a state-verified dismissal at `end_turn`.
- A research command answers "requested"; the state exported in the same frame may still show the old research.
- Nothing on the Python side uses the new fields yet (encoder slots, action mask: GM-B3, a four-module change that needs the maintainer's confirmation).
- Pinned, not changed: move targets include occupied tiles; an empty `produce` action; `players` lists only the local player; game-over is never reported; one missing method in the old part of the player block still empties the block.
- **Merge order:** #148 and #149 first. `pytest_skip_sites` stays 149 here (inherited from #149); when #147 is also in, the baseline must be 150. `CLAUDE.md`'s footer date is 2026-11-20 here and 2026-11-19 on #147: keep the later one.
- `docs/LIVE_CIV6.md` on #148 still carries a stale "bundle has drifted (GM-B0: planned)" limitation; it is removed here.
