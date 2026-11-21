# Live command layer (GM-B7) and search translator (GM-B6): task plan

Companion to [`20261119-game-mechanics-plan.md`](20261119-game-mechanics-plan.md) and its
[gap register](20261119-game-mechanics-gap-register.md). Approved by the maintainer on
2026-11-20 (answers: GM-B7 first and then GM-B6; the `--mcts` refusal stays; review findings in
one follow-up PR first). Owner: architect. OpenSpec: `openspec/changes/fix-live-command-layer`
(GM-B7); GM-B6 gets its own change when it starts.

## Why

The mod's commands were written from memory of the game's API and have never been run by a test:
the stand-in game used by the Lua tests has no unit or city operations. Read against the game's
own UI scripts, most of them cannot work. The end-of-turn and research side (GM-B2) was taken from
the same source and holds. This is the strongest lead on the live turn-cycle stall (LF-H1); it
stays a lead until the maintainer sees it live.

## Evidence

"Game UI" is the game's own UI Lua as shipped in the public CQUI mod
(`Azurency/CQUI_Community-Edition`, commit `1e6c208`, read 2026-11-20; nothing is copied from
it). Mod line numbers are at commit `976a2af`.

| # | The mod today | The game's own UI | Status |
|---|---|---|---|
| E1 | Move sends `UnitOperationTypes.MOVETO` with x and y (`ActionExecutor.lua:180`) | `UnitOperationTypes.MOVE_TO` with x, y and `PARAM_MODIFIERS` (`civ6common.lua:105-166`). `MOVETO` appears nowhere | Confirmed. No unit move can have worked; whether the game raises or ignores the nil type is unverified |
| E2 | Skip, fortify and sleep call `UnitManager.RequestCommand` with `UnitCommandTypes.WAIT`, `.FORTIFY`, `.SLEEP` (`ActionExecutor.lua:258, 281, 304`) | The unit panel lists actions from `GameInfo.UnitOperations` and `GameInfo.UnitCommands` by row hash, each gated by `CanStartOperation` / `CanStartCommand` (`unitpanel.lua:462-604, 2249-2290`). `UNITOPERATION_FORTIFY` is named there; `WAIT` is not | Plausible, not proven: the names for skip and sleep are in the game database, not in the UI code |
| E3 | Production puts the raw item id into the request (`ActionExecutor.lua:395-399`) | The row's `Hash` plus `PARAM_INSERT_MODE = VALUE_EXCLUSIVE` (`productionpanel.lua:216-220`); districts and placed buildings need a plot | Confirmed |
| E4 | A city's choices: `CanProduce(row.Index, true)` (`StateExporter.lua:273`) | `CanProduce(row.Hash, ...)`, strict form for "can start now" (`productionpanel.lua:2284-2285`) | Confirmed |
| E5 | Neighbouring tiles from two hand tables (`StateExporter.lua:30-31, 217`) | `Map.GetAdjacentPlot(x, y, DirectionTypes.*)` (`unitflagmanager.lua:1027-1038`), which also handles the map's wrap | Confirmed wrong for both row parities: one neighbour missing, one tile two steps away offered |
| E6 | Offered actions come from hand rules; attacks are offered on any other player's unit (`StateExporter.lua:833-867`) | Strict `CanStartOperation`; `CombatManager.IsAttackChangeWarState` before any move (`civ6common.lua:118-130`) | Confirmed; a peace-time attack would be offered |
| E7 | The next state is exported in the same frame as the request (`Main.lua:337-339`) | Requests are asynchronous | Confirmed: the next state can look as if nothing happened |
| E8 | The command window is 30 s from turn start, then the mod stops listening without ending the turn (`Main.lua:261-269`) | n/a | Confirmed |
| E9 | `--max-turns` counts loop steps, one per command batch, not game turns (`agent_loop.py:383, 490`) | n/a | Confirmed |
| E10 | Greedy production maps an item name to a slot with Python's per-process string hash (`action_converter.py`) | n/a | Confirmed: the slot differs between processes |

`README.md` and `docs/LIVE_CIV6.md` said "command delivery verified". Only `END_TURN` deliveries
were ever observed (four, 2026-07-12); the wording is corrected with this plan.

## Non-goal NG-3

NG-3 says the mod "observes state and injects commands strictly through official surfaces" and
does not alter game rules. `UnitManager.RequestOperation`, `CityManager.RequestOperation` and
`UI.RequestPlayerOperation` are the calls the game's own UI makes for a human player's click, so
GM-B7 is inside NG-3: it changes which official call carries a command, not what the game allows.
The executor asks the game's own strict check first and never starts a war.

## Slices (each its own PR, tests first by `sdlc-qa-sdet`, independent review, no stacked branches)

**GM-B7a. The executor.** `ActionExecutor.lua`: one helper that resolves an operation by type
name from `GameInfo.UnitOperations` (then `GameInfo.UnitCommands`), takes its hash, asks the strict
check, refuses anything that would start a war, requests, and reports a refusal with its reason.
Move (no attack modifier, so a move cannot become an attack), melee and ranged attack, skip,
fortify, sleep and found city go through it. Production sends the row's hash with the exclusive
mode, for items that need no placement. `Main.lua`: the next state is exported on a later tick;
the command window runs from the last command, with a hard cap per turn. `protocol.py`: a
production command may name its item, with no net growth of that file. The stand-in gains unit,
city and combat managers that apply requests on a later tick. Tests assert on the resulting game
state, never on a recorded call.

**GM-B7b. The exporter, and greedy production.** Neighbours from the game; every offered unit
action is one the strict check accepts; attacks only where no war would start; production choices
by hash, strict form, no placed items. Greedy production by name in a new small module.

**GM-B7c. The loop.** The send-and-account tail of `agent_loop.py` moves into
`bridge/command_policy.py` with a per-turn ledger (state digest, refused commands, cap) and
game-turn counting. `scripts/live_command_check.py` sends one named command and prints what
changed.

**The maintainer's live sitting** (about an hour, before T0b). The six items in
`docs/LIVE_CIV6.md`, plus: a unit moves; a skipped unit clears "unit needs orders"; a city starts
a build from a name and replaces rather than queues; how many ticks until the state shows a
request; the greedy agent ends ten turns by itself. No GM-B6 code merges before it.

**GM-B6a. The translator.** `bridge/search_translator.py`: simulator action to live command, by
id and never by list position; search root restricted to actions that translate; a root with no
visible enemy is not a finished game; clock clamped under the checkpoint's horizon; every decision
seeded and recorded.

**GM-B6b. The seam and the end-to-end test.** A second policy protocol that returns a command;
`bridge/policy_factory.py`; an un-mocked end-to-end test through the stand-in; the `--mcts`
refusal stays. The comparison against a random policy (600 games, three arms) is registered there
and run after A1.

**Parked, registered, not scheduled:** lifting the `--mcts` refusal.

## Acceptance (M = only the maintainer can observe it)

- GM-B7a: `build_lua_bundle.py --check` exits 0; reverting any of E1, E2, E3, E7, E8 fails a named
  test; a refused operation answers with its reason. (M) a unit moves; skip clears the blocker; a
  build starts from a name.
- GM-B7b: every offered move target is one step away for both row parities, against a Python
  check that shares no code with the stand-in; a production command carries the row's hash, the
  same in two processes. (M) the greedy mode clears a production blocker.
- GM-B7c: a refused command is not sent twice in one turn; `--max-turns 3` plays three game turns
  with several commands per turn. (M) ten turns end by themselves.
- GM-B6a, GM-B6b: as in the gap register rows.

## Limits

The stand-in is hand-written from the game's UI code. Until the live sitting, GM-B7 proves the
mod does what that code does, not that the game accepts it. The operation names for skip and
sleep are the least certain part; a wrong name answers with a refusal, not silence. Rollback is
reinstalling the previous mod; the commands it replaces did not work.
