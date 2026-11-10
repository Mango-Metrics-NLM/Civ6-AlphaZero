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
