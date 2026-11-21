# Playing a live Civilization VI game (demo-grade)

> **Status: demo-grade.** The bridge can reach a running, unmodified
> Civilization VI and read the game state back. What has actually been verified
> against a real game (2026-07-12) is **four autonomous turns, all `END_TURN`,
> driven by a randomly initialised network**. No unit or city command has been
> seen to work. Eight of them were rewritten from the game's own UI scripts in
> GM-B7a (`kb/specs/20261120-live-command-layer-plan.md`) and have not been run
> in a real game; the rest are still written from memory. Nothing here is
> evidence that the agent plays Civilization VI well. The open issues are listed
> under *Known limitations*.

This page is a how-to for an outside reader with a Windows machine and a Steam
copy of the game. The engineering history (why the file transport was replaced,
the wire-protocol notes, the per-module design) lives in
[`CHANGELOG.md`](../CHANGELOG.md) under `[1.8.0]` and in
`alpha_zero/bridge/Agent.md`; the original plan is `kb/specs/20260402-full-civ6-local-gameplay.md`.

## What exists

| Piece | Where | What it does |
|-------|-------|--------------|
| Lua mod `AlphaZeroAI` | `mod/AlphaZeroAI/` | Runs inside the game. Writes the game state out through `Lua.log`, polls `ExposedMembers.AlphaZeroCommand` for the next command, and acknowledges with `AZ_ACK`. |
| Socket transport | `alpha_zero/bridge/tuner_socket_bridge.py` | Speaks the game's own FireTuner debug protocol over TCP (`127.0.0.1:4318`), finds the `GameCore_Tuner` Lua state and sets `ExposedMembers.AlphaZeroCommand` directly. No GUI, no relay panel, no file hop. |
| File transport (legacy) | `alpha_zero/bridge/tuner_bridge.py` | Writes `commands.json` for a FireTuner GUI relay panel. The Lua sandbox has no `io` library, so the game cannot read the file itself; this path needs the FireTuner application open. Kept as the default for compatibility; the socket path is the one that was verified headless. |
| Agent loop | `alpha_zero/bridge/agent_loop.py` | Decodes state, runs the policy (the greedy network; the search over a snapshot-seeded `MicroCivEnv` exists but is refused on a live game, see "Known limitations"), sends one command per turn. |
| Entry point | `python -m alpha_zero.civ6_local --mode civ6` | Wires the above together; `--dry-run` validates the configuration without a game. |
| Mod installer | `scripts/install_civ6_mod.py` | Copies the mod into the game's `Mods` directory (OneDrive-safe). |
| Bundle generator | `scripts/build_lua_bundle.py` | Builds `Scripts/AlphaZeroMain.lua`, the one Lua file the game loads, from the module files beside it. |

## Prerequisites

- **Windows** with Civilization VI installed through Steam. The default paths
  the tooling resolves (`%USERPROFILE%\Documents\My Games\Sid Meier's Civilization VI\...`,
  `%LOCALAPPDATA%\Firaxis Games\Sid Meier's Civilization VI\Logs\Lua.log`) are
  Windows paths; other platforms have not been exercised.
- **FireTuner enabled in the game.** The socket transport talks to the
  game's debug listener, which is off by default. Enable it in the game's
  `AppOptions.txt` (`EnableTuner 1` under `[Debug]`; check your install for the
  exact key) and restart the game. The listener binds to `127.0.0.1:4318`.
- **Python 3.11+** with this repository installed (`pip install -e .`), on the
  same machine as the game.
- **No trained checkpoint is required** — and none at Civ6 scale exists. Without
  `--checkpoint`, the bridge builds a randomly initialised network sized to the
  demo action space. That is what every live verification so far used.

## Changing the mod's Lua

The game loads one file, `mod/AlphaZeroAI/Scripts/AlphaZeroMain.lua`. It is generated:
the module files beside it (`Config.lua`, `Logger.lua`, `JsonEncoder.lua`, `FileIO.lua`,
`StateExporter.lua`, `GameInfoExporter.lua`, `ActionExecutor.lua`, `Main.lua`), in the order the `.modinfo` lists
them, each between a `BEGIN` and an `END` marker.

- Edit the module files, never the bundle.
- Regenerate and commit both:

  ```bash
  python scripts/build_lua_bundle.py
  ```

- A test fails when the committed bundle is out of date (`python scripts/build_lua_bundle.py --check`
  exits 1 and names the first differing line).
- Reinstall the mod afterwards (`scripts/install_civ6_mod.py`); the game reads its own copy.

Before 2026-11-19 the bundle was edited by hand and the module files had drifted from it,
so an edit to a module file changed nothing in the game. The module files were made equal
to the bundle (the bundle is what ran); what they lost is recorded in
`openspec/changes/add-bridge-research-parity/bundle-drift.md`, and both lost pieces were
brought back on 2026-11-20 (GM-B2).

## What the mod exports (state schema 2)

The state the mod writes carries `schema_version: 2` since GM-B2; a state without the key is
version 1. Every addition is optional on the Python side (`alpha_zero/bridge/protocol.py`),
so a version 1 state still parses.

| Where | Field | Meaning |
|---|---|---|
| top level | `schema_version` | `2` |
| top level | `end_turn_blocking`, `end_turn_blocking_message` | Name of the `EndTurnBlockingTypes` value of the first notification that blocks ending the turn, and its message. Absent when nothing blocks. |
| `player` | `techs_researched`, `civics_researched` | Type names the player has (`HasTech`, `HasCivic`). |
| `player` | `current_research`, `current_civic` | Type name in progress (the tech was a raw row index in version 1). Absent when none. |
| `player` | `research_status`, `civic_status` | `{type, index, cost, progress, turns_left, boost_triggered}` for the one in progress. |
| each own city | `buildings`, `districts` | Type names; districts only once finished (`IsComplete`). Other players' cities stay empty. |
| `legal_actions` | `research_tech`, `research_civic` | `items` are row indexes and `item_names` the type names of what can be chosen now, without the one already in progress. Techs come from `CanResearch` and civics from `CanProgress`, as in the game's own choosers; where `CanProgress` is missing, civics come from the prerequisite table. |

Commands:

- `research_tech` and `research_civic` take `item_id` as a row index or a type name. They are
  refused (`INVALID_TARGET`) for something already researched or something the game says
  cannot be chosen now. Otherwise the mod sends the request the game's own chooser sends:
  `UI.RequestPlayerOperation` with the row's **hash** and an exclusive insert mode. The
  request is fire-and-forget, so the answer is "requested", never "done", and the mod touches
  no notification: the game is taken to clear its own "choose research" blocker when the
  choice takes effect. Only when the UI request is unavailable does the mod call the GameCore
  setter (with the row index).
- `end_turn` asks the game what blocks the turn the way its action panel does
  (`NotificationManager.GetFirstEndTurnBlocking`; a scan of the notification list where that
  call is missing) and answers `END_TURN_BLOCKED` while something does, unless
  `UI.CanEndTurn()` says the turn can end.

Each game call added for this is guarded, so a call that does not exist in the game's Lua
context costs one field, not the export.

**Where the calls come from.** None was run against a game from this repository. They were
read from two sources: the game's own UI Lua as redistributed in a public UI mod
([CQUI Community Edition](https://github.com/Azurency/CQUI_Community-Edition), a 2018
snapshot, so older than the current game), and [civ6-mcp](https://github.com/lmwilki/civ6-mcp)
(MIT), which plays the real game over the same debug protocol. Where they disagreed the game's
own code won: civ6-mcp sends the row index in the research request and then falls back to the
GameCore setter when the request has no effect, while the game's choosers send the hash.

### Research slots (GM-B3)

The encoder has a position for each technology and civic, and the action space a research
action for each. Which number a technology gets must be the one the network was trained
with, so it comes from a game rule set: a technology's slot is its position in the rule set
(`alpha_zero.rules`), and the same number is its research action. The rules kernel has no
civic tree yet, so civic slots follow the `Civics` table of a GameInfo export, in row order.

| | With `CIV6AZ_BRIDGE_RULESET` | Without (the default) |
|---|---|---|
| Encoder | Researched techs (and civics, with `CIV6AZ_BRIDGE_GAMEINFO`) set their slots. | Technology and civic positions stay zero, as before. Counted, and warned about once. |
| Research actions | A choice is offered when the rule set has a slot for it. The command names it by the game's type name, and the mod finds its own row. | A choice's action is the game's own row index, so research can be chosen with no setup. |

**Turning the registry on can make the live agent worse at clearing a blocked turn.** A
choice the rule set has no slot for cannot be offered, and the small built-in rule set knows
few of the real game's technologies and no civics at all: with it, a turn blocked on a civic
choice cannot be cleared, where the default mode would have offered every civic by row
index. Use a registry when a network trained with that rule set is loaded; for a
random-weights session the default is the better choice. The rule set that covers the real
tree is the one `python -m alpha_zero.rules import-gameinfo` writes from a GameInfo export;
"The game's own rule tables" below says how to capture that export from a running game. No
network has been trained with any of these slots.

What to look for in the log:

- At start-up each converter says which mode it is in: the rule set, the start of its
  fingerprint and the number of technology and civic slots, or "no research registry". A
  registry with no civic slots gets a warning.
- The two converters must use the same registry. The state carries the registry's
  fingerprint, and the action converter refuses a state numbered by another one.
- What is lost is counted, not hidden, and warned about once, naming the first item
  (`StateConverter.research_stats`, `ActionConverter.research_stats`): researched or offered
  items the rule set does not know, and slots beyond what the encoder (100 each) or the
  action space (73 technologies, 56 civics) has room for. A state converted twice in a row
  is counted once. The counters are not yet part of the agent loop's own statistics.

The current game has more technologies and civics than the action space has research
actions; those cannot be chosen until the action space is sized from the rule set, which is
a change on the release play path and waits for v2.1.0.

### The game's own rule tables (GM-B4)

The real technology and civic trees are the game's data, not this repository's. The mod
copies a few of the game's `GameInfo` tables once per game load: technologies, civics,
buildings and units with their costs, eras and prerequisites. Identifiers and whole numbers
only, no text. With file access it writes `gameinfo.json` beside its other files; without
(the usual case) it prints the snapshot to `Lua.log` as `AZ_GAMEINFO` lines, cut into pieces
like a state. Each attempt starts with an `AZ_GAMEINFO_BEGIN` line (or, with file access, by
overwriting the previous `gameinfo.json` with that line, which the capture refuses as a
snapshot the mod began and did not finish; a read-only file is removed instead), and the closing line of a snapshot in pieces
carries a checksum of the whole text. A table with a row or a value the mod cannot export
(anything but an identifier or a whole number) is left out whole, and the capture then
refuses the snapshot for lacking it.

`ENABLE_GAMEINFO_EXPORT = false` turns the export off, but only where the mod can read its
override file (`alphazero_config.json`), which needs file access. Without file access (the
usual case) there is no switch at run time: the export runs at game load, before any
outside channel exists. To turn it off there, change the default in
`mod/AlphaZeroAI/Scripts/Config.lua`, run `python scripts/build_lua_bundle.py` and reinstall
the mod. The export only prints to `Lua.log`; nothing leaves the machine.

To use it:

```bash
python -m alpha_zero.bridge.gameinfo_capture --from-log "PATH/TO/Lua.log"
```

```bash
python -m alpha_zero.rules import-gameinfo SNAPSHOT.json
```

The first command checks that the rules importer accepts the snapshot and files it as
`<CIV6AZ_DATA_ROOT>/gameinfo/gameinfo-<time>.json`, and it prints the second command. It
refuses, with a one-line reason:

- to write inside the repository (symbolic links are followed first), with no data root
  set, or over an existing file;
- a snapshot that lacks one of the six tables: the mod leaves a table out when it could
  not read it, and a rule set built from the rest would silently have no units, say;
- a snapshot that is not strict JSON (a key written twice, a number too long to read), or
  whose civics could not number research slots (a row with no name, the same civic twice);
- when the newest snapshot in the log is unfinished: still being written, cut short, with
  a line missing (the checksum then does not match, so pieces of two snapshots are never
  joined), or an attempt that could read no table. A `Lua.log` covers a whole launch and
  holds one snapshot per game loaded, whose tables can differ; `--allow-earlier` files the
  last complete one anyway and says so. A game load with the export turned off leaves no
  line at all, so the capture cannot know about it. The second writes a rule-set file, also under the data root. Then
`CIV6AZ_BRIDGE_RULESET=<rule-set file>` and `CIV6AZ_BRIDGE_GAMEINFO=<snapshot>` give the
bridge slots for the real trees ("Research slots" above).

The snapshot is the user's own game data: it is never committed, and a test fails if a
file shaped like one is tracked. Like the rest of this page, the export has run only
against the hand-written stand-in for the game. In particular, whether the game's tables
have exactly these column names in the current version is unverified; a column the game
does not have is left out of the rows, and the capture command then says what the
importer misses.

### Live checklist for the maintainer (GM-B2)

Nothing below can be checked without the game. After `python scripts/install_civ6_mod.py`:

1. Start a game and read one exported state: `schema_version` is 2, `techs_researched` and the
   capital's `districts` are not empty, `research_status` has a cost and a progress.
2. With no research chosen, send `end_turn`: the answer is `END_TURN_BLOCKED` naming
   `ENDTURN_BLOCKING_RESEARCH`, and the next state carries `end_turn_blocking`.
3. Send `research_tech` with an `item_id` from the `research_tech` legal action: the next
   state (or the one after: the request is applied after the command returns) shows it as
   `current_research`, and `end_turn_blocking` is gone. If the research changed but the
   blocker stayed, the game does not clear its own notification and the mod has to dismiss
   it; if the research did not change, the request's parameters are wrong for this game
   version. Either is worth an issue.
4. Send `end_turn` again: the turn ends.
5. Note any field that stays empty in the real game and any call the log reports as failing.
6. GM-B4: after loading a game, run the capture command above on the game's `Lua.log`. It
   should report non-zero row counts for all six tables and "importable". If it reports a
   missing column or table, that is the column name to correct in `GameInfoExporter.lua`.

### Unit and city commands (GM-B7a)

GM-B7a rewrote eight commands from the game's own UI code: `unit_move`, `unit_attack`,
`unit_ranged_attack`, `unit_skip`, `unit_fortify`, `unit_sleep`, `unit_found_city` and
`city_produce`. Before it, a move named an operation type the game does not have and a
production order carried a raw id where the game expects a hash (both certain); skip, fortify
and sleep were sent as unit commands under names the game's UI does not use (likely wrong, not
proven). **The other commands are unchanged and still written from memory**: `unit_build`,
`unit_promote`, `city_buy` and `city_citizen` should be expected not to work.

The eight now go the way the game's own UI sends them:

- A unit action is looked up by type name in the game's own tables (`GameInfo.UnitOperations`,
  then `GameInfo.UnitCommands`), requested by that row's hash, and only after the game's strict
  check says the unit can start it now. A refusal comes back as `OPERATION_REFUSED`, with the
  game's reason where the game gives one; a name this version of the game does not have comes
  back as a failure that names it. The names used: `UNITOPERATION_MOVE_TO`, `UNITOPERATION_RANGE_ATTACK`,
  `UNITOPERATION_SKIP_TURN`, `UNITOPERATION_FORTIFY`, `UNITOPERATION_SLEEP`,
  `UNITOPERATION_FOUND_CITY`. **The names for skip and sleep are the least certain**: they
  live in the game's database, not in the UI code this was written from.
- A move carries no attack modifier and is meant never to become an attack; an attack is its
  own command. That rests on the game's strict check refusing a move onto another player's
  unit, which the game's UI never asks and is therefore unverified (checklist item 12).
  Neither is sent when the game says it would start a war (`WOULD_START_WAR`), or when the
  game cannot be asked. A move into a tile the player cannot see is not asked about, as in
  the game's own move routine.
- `city_produce` takes a type name (`"UNIT_WARRIOR"`) or a row index, and sends the row's hash
  with the exclusive insert mode: the item replaces what the city was building. Districts,
  and buildings whose row says they need placement (wonders), need a plot and are refused
  (`NOT_IMPLEMENTED`). The greedy policy can still choose them; each such choice is a refused
  command.
- A success answer means the game was *asked*. The game carries a request out after the script
  returns, so the mod exports the next state `STATE_EXPORT_DELAY_TICKS` frames later (default
  2), not in the same frame, and reads no command in between.
- The mod keeps listening for `COMMAND_TIMEOUT` seconds after the *last* command, not after the
  turn's start. After `MAX_COMMANDS_PER_TURN` commands (default 200) in a turn it accepts only
  `end_turn` and answers anything else with `COMMAND_CAP_REACHED`. One exception, so the cap
  cannot trap a turn: while the game refuses to end the turn, commands that can answer the
  open choice (research, production, skip, fortify, sleep) are still carried out, up to twice
  the cap. When the window does run out, the mod stops listening and the turn stays open
  until someone ends it by hand, as before.

**What the mod offers (GM-B7b).** The `legal_actions` of an exported state are now exactly
what the executor's own check accepts at that moment
(`AlphaZeroActionExecutor.Check`, the executor run without sending anything), so an offered
action is one the mod will not refuse:

- a unit's move targets are the game's own adjacent plots (`Map.GetAdjacentPlot`), minus
  what the game's check refuses (occupied tiles, impassable ones). The table of offsets the
  exporter used to carry was wrong for both row parities and is gone; where the game's
  adjacency call is missing no move is offered at all;
- an attack is offered only where it would not start a war; a ranged attack only to a unit
  the game lets start one;
- a city's choices are asked by the row's hash, in the strict form, and exclude anything that
  needs a plot. They are listed by type name.

On the Python side a production item's action slot is its position among the offered items
of its kind (`alpha_zero/bridge/production_slots.py`), the same in every process, and the
command carries the item's name. Before, a name was mapped with Python's salted string hash
into 200 buckets: the slot differed from one process to the next, and most items landed
outside the action space's 20 slots and could never be chosen. Limit: a slot means "the
n-th offered item", not a fixed item, so it is enough for the greedy policy and not for a
policy that is to learn what to build.

All of this has run only against the hand-written stand-in for the game
(`tests/fixtures/lua/civ6_stub_commands.lua`), which was written from the same reading of the
game's UI code. It shows the mod does what that code does, not that the game accepts it.

Live checklist, continuing the one above:

7. Send `unit_move` for a unit to a neighbouring free tile: a later state shows it there. If
   the answer names an unknown operation, that is the type name to correct.
8. With "a unit needs orders" blocking the end of turn, send `unit_skip` for that unit: the
   blocker is gone in a later state. Same for `unit_fortify` and `unit_sleep`.
9. Send `city_produce` with `item_id` `"UNIT_WARRIOR"` to a city that is building something
   else: it now builds the warrior and nothing is queued behind it.
10. Count the states (or frames) between a command and the first state that shows its effect.
    If two frames is too few, raise `STATE_EXPORT_DELAY_TICKS`.
11. Send `unit_attack` on a unit of a player you are at peace with: the answer is
    `WOULD_START_WAR` and no war starts.
12. Send `unit_move` onto a tile holding a unit you are at war with: it is refused and there
    is no combat. Then `unit_attack` on it: combat happens. (The stand-in invents both.)
13. Send `unit_move` to a tile three or more steps away, and one into the fog: both are
    accepted, or the refusal's text says why.
14. `unit_ranged_attack` with a ranged unit, `unit_found_city` with a settler, and
    `city_produce` by row index: each takes effect.
15. Send `city_produce` naming a wonder: `NOT_IMPLEMENTED`, and nothing is queued.
16. Note the text of one refusal: whether it carries the game's reason, and in what form.
17. Read one exported state: every unit with moves has `move` targets and each is one step
    away; a city that is building nothing lists items by name. If a unit has no `move`
    action at all, `Map.GetAdjacentPlot` or `DirectionTypes` is missing in the mod's context.
18. Let the greedy agent run (`python -m alpha_zero.civ6_local --mode civ6`): a city with no
    production starts building something, and no command comes back `OPERATION_REFUSED`.

What the stand-in cannot tell us, because it was written from the same reading as the mod:
whether skip, sleep and fortify are operations under those names; whether the strict check
accepts a reachable tile and refuses an occupied one; whether the attack modifier alone
carries out a melee attack; whether `unit:GetAttacksRemaining()` exists in the mod's Lua
context (without it every attack fails with an exception message); and how many published
frames the game takes to carry a request out.

## Steps

1. **Install the mod** into the game's `Mods` directory, then enable it in the
   game's *Additional Content* menu.

   ```bash
   python scripts/install_civ6_mod.py                 # default Steam Mods dir
   python scripts/install_civ6_mod.py --mod-dir PATH  # custom Mods dir
   ```

2. **Check the bridge configuration** without a game running. This resolves
   the communication directory, the log path and the transport, and exits.

   ```bash
   python -m alpha_zero.civ6_local --mode civ6 --dry-run
   ```

3. **Start a new single-player game** with the mod enabled and let it reach
   the first turn.

4. **Run the agent** against the live game over the socket transport. It
   plays the greedy network policy. `--mcts` (or `CIV6AZ_LOCAL_USE_MCTS=1`) is
   refused in this mode: the command exits 1 before it connects to anything
   (see "Known limitations").

   ```bash
   # Windows PowerShell
   $env:CIV6_BRIDGE_TRANSPORT = "socket"
   python -m alpha_zero.civ6_local --mode civ6
   ```

   Watch `Lua.log`: a delivered command appears as
   `Command received via ExposedMembers` → `Executing: END_TURN` → `AZ_ACK`.
   Add `--eval-harness` to record per-turn telemetry to MLflow.

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CIV6_BRIDGE_TRANSPORT` | `file` | `socket` for the headless FireTuner path (recommended); `file` needs the FireTuner GUI relay. |
| `CIV6_BRIDGE_TUNER_HOST` / `CIV6_BRIDGE_TUNER_PORT` | `127.0.0.1` / `4318` | Where the game's FireTuner listener is. |
| `CIV6_BRIDGE_KICK_START` | unset | Set to `1` to send one `END_TURN` at start-up to un-stall an idle game (socket transport only). |
| `CIV6AZ_LOCAL_LOG_PATH` | auto | Path to the game's `Lua.log` if it is not in the default location. |
| `CIV6AZ_LOCAL_USE_EVAL_HARNESS` | `0` | Same as `--eval-harness`. |
| `CIV6AZ_BRIDGE_RULESET` | unset | The game rule set whose technology order gives the encoder slots and research actions: a built-in id (`science-v1-draft`) or the path of a rule-set file. Unset: see "Research slots" below. A value that cannot be loaded stops the start-up. |
| `CIV6AZ_BRIDGE_GAMEINFO` | unset | Optional, with the rule set: the path of a GameInfo export whose `Civics` table gives the civic slots. |

## Known limitations

- **Turn-cycle stall.** After a handful of automated end-turns the game's own
  turn cycle stalls: the local player's `TURN_ACTIVE` flag reads `false` with
  no popup blocking it, and no headless way to force-advance the turn from
  that state has been found. Commands are still delivered and acknowledged
  while this happens, so it is a game-side automation limbo rather than a
  transport fault. The likely fix is mod-side ("unit needs orders" handling);
  it has not been attempted. **Open.**
- **Only `END_TURN` has been observed live.** The channel bug that forced the
  agent to fall back to `END_TURN` every turn (an 84-channel observation
  collapsed to one channel) was fixed in v1.8.0, so the agent now runs its
  actual policy over the demo action space — but a session in which it issues
  and executes other actions has not been demonstrated against a real game.
- **Random weights.** No checkpoint trained on the full-game encoder exists.
  The MicroCiv checkpoints are not usable here, and the project's own training
  evidence is not yet a learning signal (see the Validation column in
  [`../ROADMAP.md`](../ROADMAP.md)). A live demo with a trained policy is
  sequenced after that.
- **Demo action space.** The bridge builds a small action space
  (`max_units=20`, `max_cities=5`, `num_production_items=20` at the configured
  board size), not the full 84-channel / hierarchical space described for the
  real game.
- **Research is exported, offered and chosen; buildings are exported and not
  used (GM-B2 and GM-B3 done 2026-11-20, tested on a stand-in for the game
  only).** The exporter writes researched techs and civics by type name, the
  current research and civic with cost, progress and boost, each of the
  player's cities' buildings and districts, and research choices as legal
  actions; research is requested the way the game's own chooser requests it
  (see "What the mod exports" below). On the Python side the agent can now
  choose research (it is in the action mask), and with a rule set configured
  the encoder's technology and civic positions are filled (see "Research
  slots" below). Still unused: city buildings and districts, the research
  status (cost, progress, boost), and the end-turn blocker as a network
  input. **None of this has run in a real game:** the tests execute the mod's
  Lua against `tests/fixtures/lua/civ6_stub.lua`, which is hand-written.
  Plan: `kb/specs/20261119-game-mechanics-plan.md`.
- **`--mcts` search mode is refused (GM-B5 done 2026-11-20).**
  `python -m alpha_zero.civ6_local --mode civ6 --mcts` exits 1 at start-up,
  before the bridge directory, the network or any connection is made, and so
  does `--dry-run` with it. The search does not search the real game: it
  plays in the MicroCiv simulator, seeded from the live state. As wired, the
  live mode gave it the full-game network (84 input channels) where the
  simulator's observation has 13, so the search raised on every turn and the
  agent loop turned each failure into `END_TURN`: the mode ended turns and
  did nothing else. A test that mocks nothing now shows this
  (`tests/unit/test_bridge_search_mode_wiring.py`). A network that fits would
  not be enough: the search answers with one of the simulator's 376 actions,
  and the bridge would read that number as one of its own (88,562 with the
  demo settings). Only `END_TURN` is the same number in both; a simulator
  move, which names a unit and a direction, is read by the bridge as a move
  to some other tile. What is missing is a
  translator from a simulator action to a live command (**GM-B6**, not
  scheduled); the refusal stays until that exists, the search is given a
  network and checkpoint that fit the simulator, and an un-mocked end-to-end
  test passes.
- **What the search's simulator sees of a live state (GM-B5).** MicroCiv has
  three unit types and three terrain types, so a live state is stood in for
  (`alpha_zero/bridge/microciv_proxy.py`): melee and cavalry units are
  warriors, ranged and siege units are archers, a settler is a settler;
  mountains, water and ice are the simulator's mountain, the one terrain it
  refuses to enter; hills are hills and everything else is plains. Units
  with no stand-in are left out and counted: builders, traders, scouts,
  ships, aircraft, support, religious and great-person units. No simulator
  rule was added, so there is no water, no embarking and no ranged attack in
  the search. Before this, a settler reached the simulator as a "civilian",
  which could not found a city, and a mountain as "grassland", which could
  be walked over. Three limits that remain: a unit the converter's own table
  does not know is counted as a melee unit, so an unlisted ship becomes a
  warrior standing on water; on a map larger than the board several game
  tiles share one board tile, and that tile is impassable when any of them
  is; and because units are left out, a unit's position in the simulator's
  list is not its position in the live list.
- **A game where the agent is not player 0 (fixed 2026-11-20, not seen
  live).** With two players the converter numbers the agent's own units 0
  whatever the agent's player id in the game is, but it reported the acting
  player as that game id. When the two differed, the encoder put the agent's
  units in the enemy's channels and no unit or city action was legal (ending
  the turn, and research the game offered, still were). The acting player is
  now reported by the same numbering. A single-player game normally makes the human seat player 0,
  which is why this did not show in the sessions run so far.
- **A blocked `END_TURN` is now reported, and the agent cannot clear every
  blocker (GM-B2 done 2026-11-20, not yet seen live).** When a notification
  blocks ending the turn (an unanswered research or production choice, for
  example), `END_TURN` answers `END_TURN_BLOCKED` with the blocking type, sends
  nothing to the game, and the mod exports the state again with
  `end_turn_blocking` set and keeps listening. Before, it answered "Turn ended"
  whatever the game did and stopped listening. The agent loop counts these
  (`blocked_end_turns`) instead of treating them as invalid actions. The
  agent can choose research since GM-B3, which is what clears a research or
  civic blocker; whether it does depends on the policy, and with random
  weights nothing steers it there. Limits: a blocker that is not cleared
  repeats until the mod's command timeout (30 s from the start of the turn,
  after which the mod goes quiet, as it did before); blockers with no action
  in the action space (a policy slot, a governor, a pantheon) cannot be
  cleared by the agent at all; and `end_turn` is still listed as a legal
  action while the turn is blocked (`end_turn_blocking` is the signal).
  `alpha_zero/envs/real_civ6_env.py` treats any failed command as an invalid
  action, so there a blocked `end_turn` costs the invalid-action penalty; that
  file is on the release play path and is not changed before v2.1.0. Whether a
  pending choice explains the turn-cycle stall above is still a hypothesis to
  check in a live session, not a finding.
- **Every acknowledged command was followed by a second `end_turn` on the
  socket transport (fixed 2026-11-20, GM-B2).** `LiveBridge` returns the
  acknowledgement as a dict and the agent loop read it as an object; the error
  was caught by the step's catch-all, which sends a fallback `end_turn`. Found
  by review and shown by a test, not in a live session. It may be part of why
  only `END_TURN` was ever observed live and of the turn-cycle stall; that is
  a lead to check, not a finding.
- **Windows-only paths and a single machine.** The game, the mod and the agent
  must share a filesystem and a loopback interface.

## What would move this past demo-grade

In order, because each depends on the previous:

1. A sustained run (20+ turns) on a fresh single-player game in which the
   agent issues non-`END_TURN` actions and state decoding stays correct.
2. Mod-side end-turn robustness so the turn cycle self-sustains.
3. A checkpoint trained with the post-2026-09-04 search, loaded with
   `--checkpoint`, once the training loop has a demonstrated learning signal.
4. Retiring the file transport once the socket path has been repeated across
   several fresh games.

These items are tracked in [`../ROADMAP.md`](../ROADMAP.md) (M3 validation, M5).
