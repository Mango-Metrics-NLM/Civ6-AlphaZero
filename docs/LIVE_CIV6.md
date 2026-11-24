# Playing a live Civilization VI game (demo-grade)

> **Status: demo-grade.** The bridge reaches a running, unmodified
> Civilization VI over FireTuner (`127.0.0.1:4318`) and reads schema-2 game states back.
> Live verification on 2026-11-21 verified FireTuner `LSQ:`/`CMD:` connectivity,
> 6-table `AZ_GAMEINFO` extraction + rules import (`import-gameinfo`), schema-2 exports,
> `research_tech`, `unit_move`, `unit_skip`, `unit_fortify`, `unit_found_city`, and
> `city_produce` (`UNIT_WARRIOR`) acknowledgements against a live Turn 1–7 session, and
> fixed five live Lua engine discrepancies (LLP64 JSON integer formatting, sole
> `NotificationManager`/`GetEndTurnBlocker` end-turn guard, two-step `CanProduce` unlock
> check, `GetCurrentProductionTypeHash()` fallback, and short `YieldTypes` keys). Four
> autonomous `END_TURN`s ran on 2026-07-12; a fresh Turn 1 multi-turn soak on the rebuilt
> bundle remains open, and combat (`unit_attack`, `unit_ranged_attack`), builder/citizen/buy,
> and trained-policy play are not yet live-verified. Nothing here is evidence that the agent
> plays Civilization VI well. Open issues are listed under *Known limitations*.

This page is a how-to for an outside reader with a Windows machine and a Steam
copy of the game. The engineering history (why the file transport was replaced,
the wire-protocol notes, the per-module design) lives in
[`CHANGELOG.md`](../CHANGELOG.md) under `[1.8.0]` and in
`alpha_zero/bridge/Agent.md`; the original plan is `kb/specs/20260402-full-civ6-local-gameplay.md`.

## Status at a glance

This table is the one place that says what state each part is in. `README.md`,
`ROADMAP.md` and `CLAUDE.md` link here and do not restate it.

- **Built**: the code is on the default branch.
- **Stand-in**: tests run the mod's real Lua against a hand-written stand-in for the game
  (`tests/fixtures/lua/`). That shows the mod and the Python side agree with each other and
  with the game's UI code as we read it. It does not show that the game accepts it.
- **Seen live**: observed against a running game, with the date. "Answer seen" means the mod
  acknowledged the command in the session of 2026-11-21 (turns 1 to 7); the effect in the
  following state was not recorded in this repository, and that session has no committed
  artifact under `docs/evidence/`.

| Capability | Built | Stand-in | Seen live |
|---|---|---|---|
| Socket connection, schema-2 state export | yes | yes | yes, 2026-11-21 |
| GameInfo capture (six tables) and rules import | yes | yes | yes, 2026-11-21 |
| `research_tech` | yes | yes | answer seen, 2026-11-21 |
| `research_civic` | yes | yes | no |
| `unit_move`, `unit_skip`, `unit_fortify`, `unit_found_city` | yes | yes | answer seen, 2026-11-21 |
| `city_produce` | yes | yes | answer seen for `UNIT_WARRIOR`, 2026-11-21 |
| `unit_sleep` | yes | yes | no |
| `unit_attack`, `unit_ranged_attack`, the war check, the declare-war gate (GM-B9) | yes | yes | no |
| The make-peace and denounce gates (`ALLOW_MAKE_PEACE`, `ALLOW_DENOUNCE`) | yes, off by default | yes | no |
| `unit_build`, `unit_promote`, `city_buy`, `city_citizen` | written from memory | no | no; expect them not to work |
| `end_turn` refused while the game reports a blocker, or cannot be asked (GM-B9) | yes | yes | the cause was found live on 2026-11-21 and fixed; the fix has not had a multi-turn run. "Cannot be asked": no |
| The five fixes from the live session (below) | yes | yes; since GM-B10 the stand-in models the live behaviour by default | found live; the rebuilt mod has not been run since |
| A state reaches the agent together with the answer before it (GM-B7c-1) | yes | yes | no |
| Offered actions are what the mod will carry out (GM-B7b) | yes | yes | no |
| The greedy agent playing several turns by itself (GM-B7c-2) | yes | yes | no. The four `END_TURN`s of 2026-07-12 predate all of the above |
| Other players in the export (GM-B8a) | yes, off by default | yes | no |
| Search in the simulator answering with live commands (GM-B6, GM-B8b) | yes, refused at start-up | yes | no |
| A trained policy | no | n/a | no |

**Never measured, live or otherwise:** how well any agent plays the real game; whether a
decision made in the simulator is a good decision in the real game; how long one export
takes in the middle of a game.

**One measurement needs only your log, and no game played for it.** After a session of at
least 40 turns that reaches turn 40 or later, run
`python scripts/census_search_expressivity.py --log PATH/Lua.log --out DIR`
with `--mod-version VERSION --played-by WHO`. Both are required for the result to count: they
are your statement that the log is a real game's. It reads the first complete state of
every turn and counts how much of what the game offered the search in the
simulator could have said at all. The answer decides one thing, fixed beforehand in
`kb/specs/20261123-expressivity-census-preregistration.md`: if at board 8 the search can
express under 30% of the offered commands, or can use under half of the units that had
something to do, work on search in the simulator stops until the simulator grows or the
network is no longer tied to its board. A result above both lines is not a go. It says
only that this does not rule the work out. The script refuses to write anything unless it
could measure the state of at least 40 turns, 10 of them after turn 30. A state that lost a
piece in the log, that the parser had to repair by dropping part of it, or that cannot be
measured, does not count toward those numbers. Two games
are better than one: give `--log` twice.

### Live checklist status

The checklist items are written where their subject is explained (the numbers are stable;
items 28 and 29 were numbered 7 and 8 a second time before 2026-11-23). "Answer seen" is all
the record of 2026-11-21 supports; the maintainer ticks an item when its effect was seen.

| Items | About | Status |
|---|---|---|
| 1 | a schema-2 state with research and districts filled | export seen 2026-11-21; the named fields not recorded |
| 2, 4 | `end_turn` blocked, then ending the turn | open (the blocker check changed after the session) |
| 3 | `research_tech` takes effect and clears its blocker | answer seen 2026-11-21 |
| 5 | fields that stay empty, calls that fail | five recorded and fixed (PR #166); open for the rest |
| 6 | GameInfo capture | seen 2026-11-21 |
| 7 | `unit_move` to a free neighbour | answer seen 2026-11-21 |
| 8 | skip, fortify, sleep clear "unit needs orders" | answer seen for skip and fortify; sleep open |
| 9 | `city_produce` replaces the current build | answer seen for a warrior; "replaces, not queues" open |
| 10 | frames between a request and the state that shows it | open |
| 11, 12 | no war is started; a move never becomes an attack; an attack fights | **open, and first in the next session**: a wrong answer here acts in the game |
| 13 to 20 | far moves, ranged attack, wonders, refusal text, offers, greedy run, export time | open (`unit_found_city` in 14: answer seen) |
| 21 to 27 | other players (`OBSERVE_ALL_PLAYERS`) | open |
| 28 | a state arrives within a second or two of a command | open |
| 29 | ten turns end by themselves | open |
| 30 | the bridge plays the right player | open |
| 31, 32 | making peace and denouncing with their settings on | open |

## What exists

| Piece | Where | What it does |
|-------|-------|--------------|
| Lua mod `AlphaZeroAI` | `mod/AlphaZeroAI/` | Runs inside the game. Writes the game state out through `Lua.log`, polls `ExposedMembers.AlphaZeroCommand` for the next command, and acknowledges with `AZ_ACK`. |
| Socket transport | `alpha_zero/bridge/tuner_socket_bridge.py` | Speaks the game's own FireTuner debug protocol over TCP (`127.0.0.1:4318`), finds the `GameCore_Tuner` Lua state and sets `ExposedMembers.AlphaZeroCommand` directly. No GUI, no relay panel, no file hop. |
| File transport (legacy) | `alpha_zero/bridge/tuner_bridge.py` | Writes `commands.json` for a FireTuner GUI relay panel. The Lua sandbox has no `io` library, so the game cannot read the file itself; this path needs the FireTuner application open. Kept as the default for compatibility; the socket path is the one that was verified headless. |
| Agent loop | `alpha_zero/bridge/agent_loop.py` | Decodes state, runs the policy (the greedy network; the search over a snapshot-seeded `MicroCivEnv` exists but is refused on a live game, see "Known limitations"), sends commands until the turn ends, with a per-turn ledger of what was refused or had no effect. |
| Entry point | `python -m alpha_zero.civ6_local --mode civ6` | Wires the above together; `--dry-run` validates the configuration without a game. |
| Mod installer | `scripts/install_civ6_mod.py` | Copies the mod into the game's `Mods` directory (OneDrive-safe in-place overwrite) and verifies SHA-256 install parity (`--check`). |
| Live command probe | `scripts/live_command_check.py` | Torch-free CLI sending one command and comparing pre/post states via `state_diff.py` (`--verbose` prints endpoint and timing diagnostics). |
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
`openspec/changes/archive/2026-11-20-add-bridge-research-parity/bundle-drift.md`, and both lost pieces were
brought back on 2026-11-20 (GM-B2).

## What the mod exports (state schema 2)

The state the mod writes carries `schema_version: 2` since GM-B2; a state without the key is
version 1. Every addition is optional on the Python side (`alpha_zero/bridge/protocol.py`),
so a version 1 state still parses.

| Where | Field | Meaning |
|---|---|---|
| top level | `schema_version` | `2` |
| top level | `end_turn_blocking`, `end_turn_blocking_message` | Name of the `EndTurnBlockingTypes` value of the first notification that blocks ending the turn, and its message. Absent when nothing blocks. `ENDTURN_BLOCKING_NOT_ASKABLE` when the game could not be asked (GM-B9). |
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
  call is missing) and answers `END_TURN_BLOCKED` while something does. `UI.CanEndTurn()` is
  not allowed to override a listed blocker because the live UI returns `true` for
  `UI.CanEndTurn()` even when active blockers exist. When the game cannot be asked at all
  (no `NotificationManager`, or neither way of asking works) the answer is also
  `END_TURN_BLOCKED`, naming `ENDTURN_BLOCKING_NOT_ASKABLE`, and the state carries that blocker
  (GM-B9): an end of turn sent while a blocker is active deactivates the turn, so "could not
  ask" is not "free to end". If you see this in a real game, the mod cannot end turns in that
  Lua context and the agent stops with a declared stall; that is worth an issue.
- Successful `AZ_ACK` replies omit `error_code` and `error_message` cleanly instead of leaking
  errors from unrelated prior/skipped fields, and `JsonEncoder.lua` encodes integers with
  `%.0f` (and `-0.0` as `"0"`) to prevent 32-bit C `long` truncation on Windows Win64 builds.

Each game call added for this is guarded, so a call that does not exist in the game's Lua
context costs one field, not the export.

**Where the calls come from.** They were written before any ran against a game, from two
sources (what has since been seen live is in "Status at a glance"): the game's own UI Lua as redistributed in a public UI mod
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
file shaped like one is tracked. On 2026-11-21 a live Steam session exported all six
`GameInfo` tables cleanly and imported them into a calibrated rule-set YAML via
`alpha_zero.rules import-gameinfo`.

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
  Neither is sent when the game says it would start a war (`WOULD_START_WAR`), when the
  game cannot be asked, or when its answer is not in a form the mod understands (GM-B9).
  The separate declare-war command is refused the same way unless `ALLOW_DECLARE_WAR = true`
  (off by default). Without file access, the usual case, that means changing the default in
  `Config.lua`, rebuilding the bundle and reinstalling the mod; with file access the override
  file can set it too. Making peace and denouncing have their own settings, `ALLOW_MAKE_PEACE`
  and `ALLOW_DENOUNCE`, off by default and changed the same way; off, the command answers
  `DIPLOMACY_TURNED_OFF` and sends nothing. On, the target must be a major civilization
  (`IsMajor`; not barbarians, city-states or free cities), peace is sent only while the game
  reports the pair at war, a denouncement only to a player the game reports as met and not at
  war (`OPERATION_REFUSED` otherwise), and where the game cannot be asked the answer is
  `EXECUTION_FAILED` ("Cannot ask the game ..."). Any diplomacy command aimed at the acting
  player, declaring war and the deferred sub-actions included, is `INVALID_TARGET`, checked
  before anything else. Neither is offered to the agent, and neither has run in a game.
  `end_turn` is also refused when the bridge's player is not the game's local player, because
  the game's end-of-turn request always ends the local player's turn. A move into a tile the player cannot see is not asked about, as in
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
  `end_turn` and answers anything else with `COMMAND_CAP_REACHED`; a message that would carry
  the count past the cap is refused whole. One exception, so the cap
  cannot trap a turn: while the game refuses to end the turn, or cannot be asked what blocks it, commands that can answer the
  open choice (research, production, skip, fortify, sleep) are still carried out, up to twice
  the cap. When the window does run out, the mod stops listening and the turn stays open
  until someone ends it by hand, as before.

**What the mod offers (GM-B7b, GM-B9).** The `legal_actions` of an exported state are
what the executor's own check accepts at that moment
(`AlphaZeroActionExecutor.Check`, the executor run without sending anything), so an offered
action is one the mod will not refuse. One kind of accepted action is not offered: attacks
while other players are not exported (third item below).

- a unit's move targets are the game's own adjacent plots (`Map.GetAdjacentPlot`), minus
  what the game's check refuses (occupied tiles, impassable ones). The table of offsets the
  exporter used to carry was wrong for both row parities and is gone; where the game's
  adjacency call is missing no move is offered at all;
- an attack is offered only where it would not start a war; a ranged attack only to a unit
  the game lets start one. Since GM-B9 neither is offered at all while other players are
  not exported (`OBSERVE_ALL_PLAYERS` off, the default): the agent is not shown what it would
  attack. An attack you send by hand is still carried out, so checklist items 11, 12 and 14
  work with the default settings;
- a city's choices are asked by the row's hash, in the strict form, and exclude anything that
  needs a plot or can only be bought. They are listed by type name. When the game cannot be
  asked, nothing is listed.

On the Python side a production item's action slot is its position among the offered items
of its kind (`alpha_zero/bridge/production_slots.py`), the same in every process, and the
command carries the item's name. Before, a name was mapped with Python's salted string hash
into 200 buckets: the slot differed from one process to the next, names could collide, and in
`civ6_local`'s 20-slot action space most items landed outside the slots and could never be
chosen. Items beyond the slots are still dropped (the alphabetically last of their kind);
they are now counted (`ActionConverter.production_stats`). Limit: a slot means "the
n-th offered item", not a fixed item, so it is enough for the greedy policy and not for a
policy that is to learn what to build.

All of this was built against the hand-written stand-in for the game
(`tests/fixtures/lua/civ6_stub_commands.lua`), which was written from the same reading of the
game's UI code: that shows the mod does what that code does, not that the game accepts it.
The session of 2026-11-21 then saw the game acknowledge move, skip, fortify, found city and a
warrior build; attacks and the war check have not run in a game ("Status at a glance").

Live checklist, continuing the one above:

7. Send `unit_move` for a unit to a neighbouring free tile: a later state shows it there. If
   the answer names an unknown operation, that is the type name to correct.
8. With "a unit needs orders" blocking the end of turn, send `unit_skip` for that unit: the
   blocker is gone in a later state. Same for `unit_fortify` and `unit_sleep`.
9. Send `city_produce` with `item_id` `"UNIT_WARRIOR"` to a city that is building something
   else: it now builds the warrior and nothing is queued behind it.
10. Count the states between a command and the first state that shows its effect. The wait is
    `STATE_EXPORT_DELAY_TICKS` (2) batches of game events, or `STATE_EXPORT_MAX_FRAMES` (30) frames
    when the game raises none; if the first state after a command still shows the old game, raise
    the first. Also watch for the opposite: in an idle game the state should still arrive within
    about half a second, not never (the mod used to wait for event batches only).
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
17. Read one exported state: a unit that has moves and a free neighbouring tile has `move`
    targets, each one step away; a city that is building nothing lists items by name. If no
    unit is offered a move, run one check in FireTuner and read its message:
    `AlphaZeroActionExecutor.Check(0, { action_type = "unit_move", unit_id = <id>, target = { x = .., y = .. } })`.
    The cause can be a missing adjacency call, a missing `CombatManager` (the mod refuses
    every move it cannot ask about), a missing `UnitManager`, no `UNITOPERATION_MOVE_TO` row,
    or a war-check answer the mod cannot read. For the last one `Lua.log` has one line
    starting `[AlphaZero] WARN: IsAttackChangeWarState answered`: send it to us, because it
    means every move and attack is being refused.
18. Let the greedy agent run (`python -m alpha_zero.civ6_local --mode civ6`): a city with no
    production starts building something, and no command comes back `OPERATION_REFUSED`.
    On a map larger than the agent's board a chosen move target is scaled down and back and
    often comes back as a tile that was not offered; the agent then ends the turn instead.
    So units may hardly move even when nothing is refused: that is a known limit of the
    scaling, not of the commands.
19. Time one export in the middle of a game (dozens of units, several cities). It asks the
    game once per unit, neighbour and action and once per unit and building row per city.
20. Note whether a refused check writes anything to `Lua.log`, whether a `MustPurchase` item
    is ever offered, and what a unit at the map's wrap seam is offered.

Two more for making peace and denouncing (2026-11-24), each with its setting turned on in
`Config.lua` and the mod rebuilt and reinstalled; turn it off again afterwards:

31. At war with a major civilization, send `diplomacy` with `sub_action` 1 and that player as
    `target_player_id`. A success answer means the peace was *requested*, not that it was
    made: confirm it in the game's own diplomacy screen, or with `OBSERVE_ALL_PLAYERS = true`
    and a visible unit of the target (no longer offered as an attack). The game may refuse
    peace on its own terms, for example the AI declines, or the minimum number of turns after
    a declaration of war has not passed. Sent to a player you are at peace with, or to
    barbarians, a city-state or a free city: `OPERATION_REFUSED`.
32. Send `diplomacy` with `sub_action` 5 to a major civilization you have met and are at peace
    with. Success again means *requested*: confirm the denouncement in the game's diplomacy
    screen. To a player you have not met, or to a city-state: `OPERATION_REFUSED`.

For both: the request call itself (`GetDiplomacy():DoAction` with a `DIPLOACTION_*` name, then
`UI.RequestDiplomacyMove`) is unverified, written from memory and not from the game's UI code.
An answer `Diplomacy action ... failed` or `No diplomacy API available` points there, not at the
gate; an answer `Cannot ask the game ...` points at the gate's questions. Send us the message.

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
   python scripts/install_civ6_mod.py --check         # verify installed SHA-256 parity
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

### One command at a time (GM-B7c-1)

`scripts/live_command_check.py` sends a single command to the running game and prints the mod's
answer and what the next state shows, so each checklist item is one command and a few lines:

```bash
python scripts/live_command_check.py --action end_turn
python scripts/live_command_check.py --action research_tech --item TECH_MINING
python scripts/live_command_check.py --action unit_move --unit-id 65536 --x 10 --y 7
python scripts/live_command_check.py --action end_turn --verbose   # debug logs to stderr
```

It connects as the agent does (the same environment variables, below). Exit 0: accepted and the
effect is visible; 1: refused, or accepted with nothing to show; 2: no state or no answer. If it
reports "no state from the game", the mod is not waiting for a command (it stops 30 seconds after
the last one); `--use-backlog` acts on the last state already in `Lua.log`. An accepted `end_turn`
with no new state inside the timeout is reported as "accepted, NOT confirmed" and exits 1: the
other players may still be moving, so try a longer `--timeout`.

Started in the middle of a turn, the agent and this tool use the state already in `Lua.log` once
a heartbeat for the same game turn arrives. One case cannot be told apart from here: the same save
reloaded at the same turn. If you reload, start the agent after the turn has begun again.

What it compares between the two states: the turn number, the end-turn blocker, current research
and civic, each own city's current production, and each own unit's tile and remaining moves. A
change the mod does not export (a unit's fortified state) reads as "nothing", which is a limit of
the export, not proof the game ignored the command.

One more item for the checklist, the reason for this tool's transport fix:

28. Send any command that does not end the turn. The line `changed: ...` (or `changed: nothing`)
   should appear within a second or two, not after 30. Before GM-B7c-1 the state that follows a
   command was discarded whenever it reached the log together with the command's answer, and the
   agent then waited out its timeout while the mod gave the turn up.

### What the agent remembers inside a game turn (GM-B7c-2)

The agent sends several commands per game turn and gets a new state after each. Within one game
turn it now keeps a ledger (`alpha_zero/bridge/turn_ledger.py`):

- a command the mod refused is not sent again that turn;
- research (technology, civic) and each city's production are chosen once per turn. The game
  never offers the item already in progress, so without this the agent changed its mind after
  every choice until the cap;
- a command that was accepted but whose expected effect is not in the next state is tried once
  more and then left alone. This reads what the mod exports: `unit_skip` and `unit_fortify` show
  no change there, so they are tried twice per turn. It is a suspicion, not a verdict, until the
  number of frames a request takes has been measured live;
- after the game blocks `end_turn`, the agent does not choose `end_turn` again while anything
  else is offered and the state has not changed;
- a choice the converter rejects is chosen again, not answered by ending the turn;
- what the ledger rules out is removed from the offers before the policy is asked;
- past the cap, commands that can answer a blocker (research, production, skip, fortify, sleep)
  still go out while the game blocks `end_turn`, up to twice the cap;
- research or a city's production may be chosen again in the same turn when the game asks again
  (a boost finished the research; the state shows nothing in progress);
- three blocked `end_turn` answers in a row with no change in between is a stall: see
  `CIV6AZ_AGENT_ON_STALL` below.

The session statistics name each: `commands_sent`, `refused_commands`, `no_effect_commands`,
`reselections`, `blocked_end_turns`, `stalled_turns`, `game_turns_played`.

For the checklist: 30. After the game has loaded, `Lua.log` has a line `AI player <id>: ...`
saying which player the bridge plays and how it found it (the game's local player, normally).
If the id is not yours, every check asks about the wrong player: stop and tell us.

For the checklist: 29. With `CIV6AZ_AGENT_MAX_TURNS=0 CIV6AZ_AGENT_MAX_GAME_TURNS=10`, start the
greedy agent: it should end ten turns by itself and stop with exit code 0. If it stops with 3,
the log line "Turn N is stalled" names the blocker it could not answer.

### Other players (GM-B8a)

By default the mod exports only your own units and cities. With `OBSERVE_ALL_PLAYERS = true`
(in `mod/AlphaZeroAI/Scripts/Config.lua`, then rebuild the bundle and reinstall the mod) it also
exports what you currently see of everyone else:

- a unit only when the game reports both its tile and the unit itself as visible to you, with
  its type, position, damage and strengths, and without its moves, attacks or promotions;
- a city only while its tile is visible now, with name, position and population;
- `other_players`: for each owner that appears, `kind` (`major`, `minor`, `barbarian`, `free`),
  `at_war` and `met`. `at_war` is left out when the game could not be asked.

The state always says which mode produced it (`observes_others`). On the Python side only
hostile owners (at war, barbarians, free cities) are put on the enemy side; a neutral unit is
left out and counted (`_others` in the converted state).

The search (still refused on a live game) plays in a simulator that holds only what you can
see. Its rules call a game won when one side has nothing left, which there would mean "nothing
in sight is left", so the bridge never lets the acting player win that way, nor at the
simulator's turn limit unless an enemy city is in sight (GM-B8b). The search can therefore
lose in the simulator but not win there: its judgement is the network's estimate of a board
that shows only what you see. A network trained where the number of cities decides a game
will read "no enemy city in sight" as nearly won; only training changes that. With an
untrained network all of this proves wiring, not judgement. Two hostile players are one side in the simulator, and a scout
(a unit type the simulator has no stand-in for) is not in it at all.

Whatever the setting, a tile you do not see now is exported without its owner, improvement,
resource and yields, and an attack is only offered on something you can see.

**It is off by default because this part has not run in a real game:** the live session of
2026-11-21 covered commands and the default export, not other players. The calls were taken from
the game's own UI code; the items below are what only a live game can show.

For the checklist:

21. Turn `OBSERVE_ALL_PLAYERS` on and read one state with a barbarian scout in sight: the scout
    is in `units` with `owner` 63, and `other_players` has an entry with `kind` `barbarian`.
    Move away until it is in the fog: it is gone from the next state.
22. With a city-state and another major in sight: each has an entry with the right `kind`, and
    `at_war` matches the diplomacy screen (false for someone you are at peace with).
23. Read `Lua.log` for a line about `Game.GetPlayers`: if the call does not exist, the mod falls
    back to walking the player ids, and says so.
24. A unit you know to be hidden from you (a submarine out of sight range, a spy) on a visible
    tile is not exported; note where a spy inside a city shows up, if at all.
25. A tile you explored and left: it is still in the map with its terrain, and has no `owner_id`.
26. Time one export with the setting on, in the middle of a game.
27. With a unit you cannot see standing next to one of yours (hard to arrange; a submarine
    beside a land unit on the coast will do): is a `move` onto its tile offered? The mod asks
    the game's own check, which knows the unit is there, so a missing offer gives it away. If
    you send the move anyway, the answer must say only that it was refused.

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CIV6_BRIDGE_TRANSPORT` | `file` | `socket` for the headless FireTuner path (recommended); `file` needs the FireTuner GUI relay. |
| `CIV6_BRIDGE_TUNER_HOST` / `CIV6_BRIDGE_TUNER_PORT` | `127.0.0.1` / `4318` | Where the game's FireTuner listener is. |
| `CIV6_BRIDGE_KICK_START` | unset | Set to `1` to send one `END_TURN` at start-up to un-stall an idle game (socket transport only). **It bypasses the mod's blocker check**: sent while a research, production or unit-orders choice is open, it is the very request that deactivates the turn (LF-H1). Use it only when nothing blocks the turn. |
| `CIV6AZ_AGENT_MAX_GAME_TURNS` | `0` | Stop after this many finished game turns (0: no limit). `CIV6AZ_AGENT_MAX_TURNS` (default 500) counts loop steps, one per command, not game turns; set it to `0` when using this. There is no command-line flag for either. |
| `CIV6AZ_AGENT_MAX_COMMANDS_PER_TURN` | `100` | Past this many commands in one game turn the agent sends only `end_turn` (the mod's own cap is 200). |
| `CIV6AZ_AGENT_MAX_RESELECTIONS` | `8` | How often per step the policy is asked again after a choice that cannot be sent. |
| `CIV6AZ_AGENT_ON_STALL` | `exit` | When the game keeps blocking the end of turn and nothing the agent sends changes that: `exit` stops with exit code 3; `wait` sends nothing more in that turn and waits (up to `CIV6AZ_AGENT_STALL_WAIT_S`, default 300) for you to end it by hand. The agent never forces the turn. |
| `CIV6AZ_LOCAL_LOG_PATH` | auto | Path to the game's `Lua.log` if it is not in the default location. |
| `CIV6AZ_LOCAL_USE_EVAL_HARNESS` | `0` | Same as `--eval-harness`. |
| `CIV6AZ_BRIDGE_RULESET` | unset | The game rule set whose technology order gives the encoder slots and research actions: a built-in id (`science-v1-draft`) or the path of a rule-set file. Unset: see "Research slots" below. A value that cannot be loaded stops the start-up. |
| `CIV6AZ_BRIDGE_GAMEINFO` | unset | Optional, with the rule set: the path of a GameInfo export whose `Civics` table gives the civic slots. |

## Known limitations

- **The search mode is still refused, and would see nobody else.** GM-B6a added the translation from
  the search's simulator action to an offered live command (`alpha_zero/bridge/search_translator.py`),
  and GM-B6b lets the agent loop run a policy that answers with a command
  (`alpha_zero/bridge/command_policy.py`), proven end to end against the stand-in game only.
  `--mode civ6 --mcts` is refused all the same: no trained checkpoint exists, the search path
  has not run in a real game, and the registered comparison against a random policy has not
  been run. A research or production choice, which the search's simulator cannot
  express, is answered by a fixed rule (the first offered command that may still be sent), not by
  the search. Separately, with the mod's default settings
  (`OBSERVE_ALL_PLAYERS` off) only your own units and cities are exported, so the search's
  simulator never contains another player (GM-B8, a mod change).
- **Turn-cycle stall (LF-H1: leading cause seen once, 2026-11-21; the fix is unconfirmed).** Calling `ACTION_ENDTURN`
  while a turn-blocking notification is active deactivates the local player seat (`TURN_ACTIVE`
  becomes `false`) without advancing the turn counter. Because the live game's `UI.CanEndTurn()`
  returns `true` even when blockers exist, the old `ActionExecutor.lua` override allowed bad
  `end_turn` calls through. That override was removed so `NotificationManager` /
  `GetEndTurnBlocker` is the sole authority; a sustained fresh Turn 1 run on the rebuilt bundle
  is still needed to confirm no other blocker path stalls the cycle.
- **Partial live command coverage.** Live 2026-11-21 verification exercised schema-2 export,
  6-table `GameInfo` capture, `research_tech`, `unit_move`, `unit_skip`, `unit_fortify`,
  `unit_found_city`, and `city_produce` (`UNIT_WARRIOR`) with live fixes for two-step
  `CanProduce`, `GetCurrentProductionTypeHash()` fallback, and short `YieldTypes` keys.
  `unit_attack`, `unit_ranged_attack`, `unit_build`, `unit_promote`, `city_buy`, `city_citizen`,
  and a fresh-game `BridgeAgentLoop` multi-turn soak remain open.
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
  input. The tests execute the mod's Lua against
  `tests/fixtures/lua/civ6_stub.lua`, which is hand-written; what has been seen
  in a real game since is in "Status at a glance".
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
  to some other tile. That translator
  now exists (GM-B6a), as do a factory that builds a network fitting the
  simulator from a checkpoint and an un-mocked end-to-end test against the
  stand-in (GM-B6b). The refusal stays for the three reasons in the first
  item of this list.
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
  blocker (GM-B2 done 2026-11-20; changed after the live session of 2026-11-21).** When a notification
  blocks ending the turn (an unanswered research or production choice, for
  example), `END_TURN` answers `END_TURN_BLOCKED` with the blocking type, sends
  nothing to the game, and the mod exports the state again with
  `end_turn_blocking` set and keeps listening. Before, it answered "Turn ended"
  whatever the game did and stopped listening. The agent loop counts these
  (`blocked_end_turns`) instead of treating them as invalid actions. The
  agent can choose research since GM-B3, which is what clears a research or
  civic blocker; whether it does depends on the policy, and with random
  weights nothing steers it there. Limits: a blocker that is not cleared
  repeats until the agent declares a stall (three blocked answers with no change,
  GM-B7c-2) or the mod's command window closes (30 s after the last command); blockers with no action
  in the action space (a policy slot, a governor, a pantheon) cannot be
  cleared by the agent at all; and `end_turn` is still listed as a legal
  action while the turn is blocked (`end_turn_blocking` is the signal).
  `alpha_zero/envs/real_civ6_env.py` treats any failed command as an invalid
  action, so there a blocked `end_turn` costs the invalid-action penalty; that
  file is on the release play path and is not changed before v2.1.0. The live
  session of 2026-11-21 found that an `end_turn` sent while a choice was
  pending is what deactivated the turn (the stall item above).
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
