---
name: live-civ6-verify
description: Verify the AlphaZeroAI Lua mod bundle, offline Lua 5.1 parity harness, Steam mod SHA-256 install parity, FireTuner socket diagnostics, and GameInfo table capture end to end. Use before or after live Civilization VI sessions or mod/bridge changes.
allowed-tools: Bash, Read
---

# Live Civ6 Verification Skill

Standardizes the repeatable offline-to-live verification loop for `mod/AlphaZeroAI/`
and `alpha_zero/bridge/`, keeping bundle generation, Steam installation, Lua 5.1 parity,
FireTuner socket probes, and `GameInfo` capture deterministic and free of stale copies.

## Why this exists

The game loads only the bundled `mod/AlphaZeroAI/Scripts/AlphaZeroMain.lua` copied into
the user's Civ6 `Mods` directory. Editing a source `.lua` module without regenerating the
bundle and re-installing into `Mods` tests stale code in the live game. Running every
offline parity check first ensures Lua 5.1, JSON encoding, production checks, and end-turn
blocker logic agree before touching a running game.

## Prerequisites

- Python 3.11+ with the `[dev,lua]` extras installed (`lupa` for Lua 5.1 execution).
- For live socket steps only: Windows with Civilization VI running single-player on an
  active human turn and `EnableTuner 1` set under `[Debug]` in `AppOptions.txt`.
- Set `CIV6_BRIDGE_TRANSPORT=socket` when probing a running game over TCP `127.0.0.1:4318`.

## Steps

### 1. Rebuild and verify the Lua bundle and offline parity suite

```bash
python scripts/build_lua_bundle.py --check
make test-lua
```

`make check-lua-bundle` verifies that `mod/AlphaZeroAI/Scripts/AlphaZeroMain.lua`
matches the `.modinfo` load order bit-for-bit. `make test-lua` sets `CIV6AZ_REQUIRE_LUA=1`
so any missing `lupa` installation fails instead of silently skipping.

### 2. Install the mod and verify SHA-256 parity in Civ6 Mods

```bash
python scripts/install_civ6_mod.py
python scripts/install_civ6_mod.py --check --verbose
```

Uses OneDrive-safe in-place overwrite with SHA-256 verification and stale-file cleanup.
You can also run `make check-mod-install` for the read-only parity gate.

### 3. Validate bridge configuration without touching a live game

```bash
python -m alpha_zero.civ6_local --mode civ6 --dry-run
```

Confirm resolved `bridge_ipc_dir`, `civ6_log_path`, and clean `--mcts` refusal checks
before connecting.

### 4. Probe a single command against a live turn

```bash
python scripts/live_command_check.py --action end_turn --verbose
```

Read stdout (`state:`, `sending:`, `answer:`, `changed:`, `expected effect:`) and inspect
stderr debug diagnostics when diagnosing `END_TURN_BLOCKED`, `OPERATION_REFUSED`, or
timeout behavior.

### 5. Capture and import live GameInfo tables (optional, per game load)

```bash
python -m alpha_zero.bridge.gameinfo_capture --from-log <path-to-Lua.log>
python -m alpha_zero.rules import-gameinfo <snapshot.json> --out <output-ruleset.yaml>
```

Keep captured snapshots and imported rule sets under `$CIV6AZ_DATA_ROOT`, never inside
the repository.

## Honesty and Evidence Rules

1. Distinguish offline `lupa` + `civ6_stub.lua` parity from live Civilization VI execution.
2. Never claim a command works in-game until a live session observes both `AZ_ACK` and
   the expected post-command state delta.
3. If an active session enters turn-cycle stall (`TURN_ACTIVE=false`), start a fresh Turn 1
   game rather than trusting state mutations from the wedged session.

## Related

- `verify` — full CI parity gate (includes `CIV6AZ_REQUIRE_LUA=1`).
- [`docs/LIVE_CIV6.md`](../../../docs/LIVE_CIV6.md) — full live-bridge checklist and limitations.
