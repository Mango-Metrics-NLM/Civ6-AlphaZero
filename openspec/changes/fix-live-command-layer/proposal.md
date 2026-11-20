# Make the mod's commands ones the game can carry out

## Why

Read against the game's own UI scripts, most commands the mod sends cannot work: a unit move
names an operation type that does not exist, skip, fortify and sleep are sent as unit commands
under names the UI does not use, and a production order carries a raw id where the game expects
the row's hash. The exporter offers moves to tiles that are not neighbours and attacks that would
start a war. No test ever ran these, because the stand-in game has no unit or city operations.
Evidence with file and line: `kb/specs/20261120-live-command-layer-plan.md` (E1 to E10). This is
the strongest lead on the live turn-cycle stall (LF-H1), unconfirmed until seen live.

## What Changes

- **GM-B7a, the executor** (`mod/AlphaZeroAI/Scripts/ActionExecutor.lua`, `Main.lua`): unit
  actions resolve their operation from the game's database by type name, ask the game's strict
  check, refuse anything that would start a war, and report a refusal with its reason. Production
  sends the row's hash with the exclusive insert mode. The next state is exported on a later
  tick; the command window runs from the last command, with a hard cap per turn.
  `alpha_zero/bridge/protocol.py`: a production command may name its item.
- **GM-B7b, the exporter** (`StateExporter.lua`): neighbours from the game; offered actions are
  ones the strict check accepts; no attack that would start a war; production choices by hash.
  Greedy production by name, in a new small module under `alpha_zero/bridge/`.
- **GM-B7c, the loop** (new `alpha_zero/bridge/command_policy.py`): a refused command is not
  offered again in the same turn, a hard cap per turn, `--max-turns` counts game turns;
  `scripts/live_command_check.py` for the maintainer's live sitting.
- The stand-in (`tests/fixtures/lua/civ6_stub.lua`) gains unit, city and combat managers that
  apply a request on a later tick and drop one the game would drop.

## Impact

- Live behaviour of the mod changes for every mode (confirm-first; approved by the maintainer on
  2026-11-20). Rollback is reinstalling the previous mod; the commands replaced did not work.
- Capability `civ6-bridge`: requirements added, none modified.
- Nothing on the release play path changes. Non-goal NG-3 holds: the calls are the ones the
  game's own UI makes for a player's click.
- Not verified in a real game. The maintainer's live sitting is a task of this change; it also
  carries the open live checklist of `add-bridge-research-parity`.
