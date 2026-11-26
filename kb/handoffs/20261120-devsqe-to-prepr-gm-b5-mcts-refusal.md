# DEV-SQE → Pre-PR Handoff

**Date:** 2026-11-20
**Spec:** kb/specs/20261119-game-mechanics-plan.md (slice GM-B5; OpenSpec `openspec/changes/add-bridge-research-parity`, tasks 5.1, 5.2, 5.3)
**Branch:** `claude/gm-b5-mcts-refusal`, built on `claude/gm-b4-gameinfo-snapshot` (PR #152, open). Its PR targets the default branch and contains GM-B4 until #152 merges.
**Files Changed:**
- `alpha_zero/bridge/microciv_proxy.py`
- `alpha_zero/bridge/mcts_policy.py`
- `alpha_zero/bridge/state_converter.py`
- `alpha_zero/civ6_local.py`
- `alpha_zero/bridge/Agent.md`
- `tests/fixtures/bridge_search_states.py`
- `tests/unit/test_bridge_mcts_policy.py`
- `README.md`
- `ROADMAP.md`
- `docs/LIVE_CIV6.md`
- `docs/MODULE_REFERENCE.md`
- `CHANGELOG.md`
- `CLAUDE.md`
- `openspec/changes/add-bridge-research-parity/tasks.md`

**Tests Added:** `tests/unit/test_bridge_search_mode_wiring.py` (9), `tests/unit/test_bridge_microciv_proxy.py` (268), `tests/unit/test_bridge_ego_player_index.py` (20), `tests/unit/test_civ6_local_search_refusal.py` (29), `tests/unit/test_bridge_search_board_size.py` (32)
**Test Results:** 358 pass in those five files. `tests/unit` + `tests/integration` (`-m "not slow"`, one root-user sandbox test deselected) with `tests/regression`, `tests/governance` and the play-path closure test, on the final code head `5416de1`: 14,046 passed, 71 skipped, 7 xfailed, none failed. Coverage not measured locally; CI is the gate.
**Lint Status:** `ruff check .` clean; `ruff format --check` clean on `alpha_zero/` and `tests/`.
**Type Check Status:** `python -m mypy` on the four changed modules with `--follow-imports=silent`: no issues. `mypy alpha_zero/` cannot finish in this sandbox; CI is the gate.
**Smoke Tests:** `python -m alpha_zero.civ6_local --mode civ6 --mcts` exits 1 naming GM-B5 and GM-B6; `python -m alpha_zero.civ6_local --mode civ6 --dry-run` exits 0.
**Verification not done:** `make verify`; **nothing was run against a real game**.
**Play path:** the diff against the default branch over `scripts/release_play_path.txt` is empty; nothing under `alpha_zero/envs/` changed.

## What changed
- **The live search mode is refused.** `civ6_local.search_mode_refusal(config)`: `--mode civ6` with `--mcts` (or `CIV6AZ_LOCAL_USE_MCTS=1`, or a config file) exits 1 before the bridge directory, the network or a connection is made, also under `--dry-run`. Checked in `main` ahead of the dry run and again at the top of `run_civ6_bridge`. `microciv` and `web` modes are not refused.
- **Why:** the search plays in the MicroCiv simulator. The live mode gave it the full-game network (84 input channels against the simulator's 13), so it raised on every turn and the agent loop sent `end_turn` each time. With a fitting network its answer is one of the simulator's 376 actions, and the bridge reads that number in its own 88,562-action space (only `end_turn` is the same number in both). `tests/unit/test_bridge_search_mode_wiring.py` records this with nothing mocked and was committed before any fix.
- **`bridge/microciv_proxy.py` (new, standard library only):** converter unit categories onto MicroCiv's three unit types (melee, cavalry: warrior; ranged, siege: archer; `UNIT_SETTLER` by the game's own name: settler; everything else left out and counted), tiles onto its three terrain types (mountains, water, ice: the impassable one). `BridgeMCTSEnv` seeds through it, reports `proxy_stats`, and `clone()` copies instead of seeding again.
- **Two older defects fixed on the way.** (1) `StateConverter` reported `current_player` as the game's player id while numbering the agent's own units 0; with the agent as game player 1 the encoder swapped own and enemy channels and no unit or city action was legal, on the greedy path too. (2) The simulator was built at the game's map size while every position is scaled onto the converter's board; the converted state now carries `_board_size`.

## Review of record
- Tests by `sdlc-qa-sdet` against a written contract before the code. On the unchanged tree: the 9 characterisation tests pass (by design), all 268 proxy tests fail or error, 12 of 20 player-index tests fail, 16 of 25 refusal tests fail. It corrected the contract once: END_TURN is index 0 in both action spaces, so the "same number, other action" claim holds for moves, not for ending the turn.
- Negative controls on the real implementation: 12 planted defects (one at a time), each failed between 1 and 22 tests; 5 more for the board-size fix, each failed between 5 and 13.
- Review by `sdlc-architect` (read-only): nothing serious. It found the player-index fix complete (every reader of `current_player` compares it with a unit index; nothing needs the raw id through it) and the refusal complete for `civ6_local`. Findings and what was done:
  1. The simulator's board size came from `_map_size` (older defect). **Fixed** (14 of 32 new tests failed first).
  2. A unit type the converter's table does not know becomes `melee`, so an unlisted ship is a warrior on water. **Not changed**; pinned as a known limit and documented.
  3. Two doc claims were stronger than the code ("end_turn alone"; "every simulator move"). **Corrected**; the commit message of `1eddee1` still says "end_turn alone" (history not rewritten).
  4. No test of the refusal through a YAML config. **Added** (passes).

## Known issues and open questions
- The refusal covers `civ6_local`. Code that builds `BridgeAgentLoop` with a `Civ6MctsPolicy` by hand still gets a failed search turned into `end_turn` (changing that needs `agent_loop.py`).
- Limits of the stand-in, all documented in `docs/LIVE_CIV6.md`: unlisted unit types; on a scaled map a board tile is impassable when any game tile it covers is; units left out shift the simulator's unit indexes against the live ones, so a translator (GM-B6) has to map by unit id.
- GM-B6 (the translator, a network and checkpoint that fit the simulator, an un-mocked end-to-end test) is not scheduled. The refusal stays until then.
- This closes the Phase 1 slices of the plan. Open for the maintainer: the live checklist, PR #152, and whether the converters' research counters go into the agent loop's statistics.
