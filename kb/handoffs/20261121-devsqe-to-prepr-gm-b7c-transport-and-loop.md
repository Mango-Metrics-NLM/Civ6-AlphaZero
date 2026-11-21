# DEV-SQE to Pre-PR: GM-B7c, the transport keeps every state and the loop keeps a ledger

- **Date:** 2026-11-21
- **Session:** session_01CXusP9Ct9R7eg25YLcY7R7
- **Branches:** `claude/gm-b7c1-transport-states` (step 1, from the default branch, no mod change)
  and `claude/gm-b7c2-turn-ledger` (step 2; contains GM-B7a #158, GM-B7b #159 and step 1). No PR
  for either yet: both slots are taken by #158 and #159.
- **Spec:** `kb/specs/20261120-live-command-layer-plan.md`; OpenSpec `fix-live-state-delivery`
  (step 1) and `fix-live-command-layer` tasks 3.1 to 3.3 (step 2)

## What was built

Step 1, GM-B7c-1:
- `alpha_zero/bridge/log_tailer.py`: states carry a sequence number and are handed out by it.
  Before, `wait_for_state` cleared its event and then waited, so a state read in the same batch
  of log lines as the ack before it was discarded and the loop waited out its timeout.
- The state already in the log at start is released by a heartbeat that reports its own turn.
- A state sent in pieces keeps a character or a space cut at a piece boundary.
- `alpha_zero/bridge/state_diff.py`; `scripts/live_command_check.py` (one command, what changed).

Step 2, GM-B7c-2:
- `alpha_zero/bridge/turn_ledger.py` (pure) and `offer_filter.py`: within a game turn a refused
  command is not sent again, research and each city's production are chosen once, a command with
  no visible effect is tried once more, a blocked `end_turn` is held back while something else is
  offered, past the cap only `end_turn` (or what can unblock), and a stall is declared (exit 3).
- `agent_loop.py` uses it; settings moved to `agent_loop_config.py`;
  `CIV6AZ_AGENT_MAX_GAME_TURNS` counts game turns.
- `tests/fixtures/lua_bridge.py`: the stand-in game as a `GameBridge`, so the real loop runs
  against the real mod Lua.

Also on the open PRs, from review: #158, the mod's export wait counts frames as well as event
batches (an idle game); #159, a production row offered as text.

## How it was checked

- Tests first by `sdlc-qa-sdet` from written contracts; an independent review by `sdlc-architect`
  for each step, both of which found real defects (step 1: a state from an earlier load released
  as current; step 2: three ways a playable turn ended in a declared stall), fixed with tests.
- Planted defects: step 1, fourteen, each caught; step 2, twenty, nineteen caught and one a
  redundant guard. The step 2 review tests were run against the pre-fix code and fail there.
- On the step 2 branch: bridge, Lua and script suites 4,475 passed, 26 skipped; regression 471
  passed; governance passed; `ruff check .`, single-file type checks, `build_lua_bundle.py
  --check`, doc-sync, OpenSpec and kb validators, debt ratchet clean; play-path diff empty.
  `make verify`, the full type check and coverage were not run; CI is the gate.

## Not verified

Nothing ran in a real game. That the lost state is the cause of the live turn stall (LF-H1) is a
hypothesis. That the mod's context receives frame updates is read from the game's UI code.
`docs/LIVE_CIV6.md` has the items: a command's next state arrives without a 30-second wait; ten
turns end by themselves with `CIV6AZ_AGENT_MAX_TURNS=0 CIV6AZ_AGENT_MAX_GAME_TURNS=10`.

## Different from the approved plan

- The game-turn limit is a new setting; `CIV6AZ_AGENT_MAX_TURNS` still counts loop steps.
- The module is `turn_ledger.py`, not `command_policy.py`; sending stays in the loop.

## Open

- A state sent in pieces has no export id; a restarted game whose log has grown past the old
  read position is read from the middle (both registered, the first needs a mod change).
- The effect check reads the export: `unit_skip` and `unit_fortify` show nothing and are tried
  twice a turn. A reload of the same save at the same turn keeps the ledger's refusals.
- Past the cap, skips of other units count as unblocking and can use up the allowance.
- Order of PRs: step 1 when #158 merges; step 2 when #159 merges. Then the live sitting, then
  GM-B6a/b.
