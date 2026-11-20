# Playing a live Civilization VI game (demo-grade)

> **Status: demo-grade.** The bridge can deliver commands to a running,
> unmodified Civilization VI and read the game state back. What has actually
> been verified against a real game (2026-07-12) is **four autonomous turns,
> all `END_TURN`, driven by a randomly initialised network**. Nothing here is
> evidence that the agent plays Civilization VI well; it is evidence that the
> plumbing works. The open issues are listed under *Known limitations*.

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
| Agent loop | `alpha_zero/bridge/agent_loop.py` | Decodes state, runs the policy (greedy network or Gumbel MCTS over a snapshot-seeded `MicroCivEnv`), sends one command per turn. |
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
`StateExporter.lua`, `ActionExecutor.lua`, `Main.lua`), in the order the `.modinfo` lists
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

4. **Run the agent** against the live game over the socket transport. `--mcts`
   runs Gumbel MCTS; `--no-mcts` uses the greedy network policy.

   ```bash
   # Windows PowerShell
   $env:CIV6_BRIDGE_TRANSPORT = "socket"
   python -m alpha_zero.civ6_local --mode civ6 --mcts
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
- **Research and buildings are exported and offered by the mod, but the agent
  does not use them yet (GM-B2 done 2026-11-20, tested on a stand-in for the
  game only; GM-B3 planned).** The exporter now writes researched techs and
  civics by type name, the current research and civic with cost, progress and
  boost, each of the player's cities' buildings and districts, and research
  choices as legal actions; research is requested the way the game's own
  chooser requests it (`UI.RequestPlayerOperation` with the row's hash), with
  the GameCore setter as the fallback (see "What the mod exports" below). **None of this has run in a real game:** the
  tests execute the mod's Lua against `tests/fixtures/lua/civ6_stub.lua`, which
  is hand-written. On the Python side the new fields are parsed and nothing
  else: the encoder's tech and civic slots are still zero and the action mask
  ignores research actions until GM-B3. Plan:
  `kb/specs/20261119-game-mechanics-plan.md`.
- **`--mcts` search mode is not usable as wired (GM-B5: planned, not done).** It
  pairs the 84-channel network with the 13-channel MicroCiv simulator and
  decodes a simulator action index as a hierarchical action; an exception in
  the policy becomes `END_TURN`. This is the reading of the code recorded in
  the plan, not yet shown by a test; GM-B5 starts by writing that test, then
  makes the mode refuse at start-up, citing its gap id, until an action
  translator exists. It does not refuse today.
- **A blocked `END_TURN` is now reported, and the agent cannot clear every
  blocker (GM-B2 done 2026-11-20, not yet seen live).** When a notification
  blocks ending the turn (an unanswered research or production choice, for
  example), `END_TURN` answers `END_TURN_BLOCKED` with the blocking type, sends
  nothing to the game, and the mod exports the state again with
  `end_turn_blocking` set and keeps listening. Before, it answered "Turn ended"
  whatever the game did and stopped listening. The agent loop counts these
  (`blocked_end_turns`) instead of treating them as invalid actions. Limits:
  the agent does not yet choose research (GM-B3), so a research blocker
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
