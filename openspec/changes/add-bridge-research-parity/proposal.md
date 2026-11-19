# Bring the Civ6 bridge to research parity

## Why

The live bridge cannot observe or choose what a science game needs. The Lua
exporter never fills `techs_researched`, city buildings or districts. Research is
executable but never offered as a legal action. The 84-channel encoder's tech and
civic slots are always zero. `--mcts` hands an 84-channel network to a 13-channel
simulator and decodes its index in the wrong action space. The loaded
`AlphaZeroMain.lua` has drifted 27 lines from its module sources. This change
covers slices GM-B0..GM-B5 of `kb/specs/20261119-game-mechanics-plan.md` (local-first
register LF-H1 and LF-H5 context).

## What Changes

- GM-B0: `scripts/build_lua_bundle.py` generates `AlphaZeroMain.lua` from the
  module files. The first generation is byte-identical to the committed bundle,
  and `--check` is a sync gate.
- GM-B1: a `lupa` test extra (Lua 5.1 runtime) and `tests/fixtures/lua/civ6_stub.lua`,
  so the exporter runs in tests.
- GM-B2: `StateExporter.lua` and `ActionExecutor.lua` export researched techs and
  civics, current research, per-city buildings and districts, and a
  `schema_version`. They offer research as legal actions. END_TURN reports a blocked turn.
- GM-B3: additive `protocol.py` fields, new `bridge/tech_registry.py`, and wiring in
  `state_converter.py` and `action_converter.py`.
- GM-B4: a Lua `export_gameinfo` and `bridge/gameinfo_capture.py`, writing only
  under `CIV6AZ_DATA_ROOT`.
- GM-B5: `bridge/microciv_proxy.py`. `--mcts` refuses at start-up with its gap
  id until an action translator exists.

## Impact

- Affected specs: `civ6-bridge` (new capability).
- Release play path: untouched. All edits are under `alpha_zero/bridge/`,
  `alpha_zero/civ6_local.py`, `mod/`, `scripts/build_lua_bundle.py`, `tests/`,
  `pyproject.toml`, CI and docs, none of which `scripts/release_play_path.txt`
  matches. No file under `alpha_zero/envs/` changes, and the `default` preset is untouched.
- Confirm-first items covered by the maintainer's approval of the plan
  (2026-11-19):
  - generating the loaded bundle (byte-identical first);
  - the GM-B2 live mod behaviour changes;
  - additive wire fields;
  - `--mcts` refusing (a documented command changes behaviour);
  - `lupa` as a test-only extra.
- Confirm-first items the plan does not name, to be confirmed by the maintainer
  at their PR: END_TURN may now report a blocked turn (a protocol behaviour change,
  shipped with its Python handling); GM-B3 touches four bridge modules.
- Not in this change: an action translator, a working live search mode, committed
  real GameInfo tables, the full-scale encoder or action space.
