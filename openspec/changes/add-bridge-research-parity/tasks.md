# Tasks

## 1. GM-B0: generated Lua bundle
- [ ] 1.1 Pin line endings for `mod/**/*.lua` (`.gitattributes`) or normalise in the generator
- [ ] 1.2 Reconcile module sources so that `scripts/build_lua_bundle.py` reproduces the committed `AlphaZeroMain.lua` byte for byte; list the 27 differing source lines in the PR as GM-B2 candidates
- [ ] 1.3 `tests/scripts/test_lua_bundle_sync.py` runs `--check`; negative control: an edit to one module file without regeneration fails it

## 2. GM-B1: Lua harness
- [ ] 2.1 `lupa` in its own test extra; tests use the Lua 5.1 runtime (`lupa.lua51` or LuaJIT) and `importorskip`
- [ ] 2.2 The CI job that installs the extra sets a flag that turns a missing `lupa` into a failure; the suite-wiring gate names the job
- [ ] 2.3 `tests/fixtures/lua/civ6_stub.lua` stubs `Players` and `GameInfo`; the current exporter runs against it

## 3. GM-B2 and GM-B3: exporter parity and Python wiring
- [ ] 3.1 Tests first, against the stub: researched techs and civics, current research, buildings, districts, `schema_version`, research legal actions, a blocked END_TURN
- [ ] 3.2 `StateExporter.lua`, `ActionExecutor.lua` (every new call `pcall`-guarded; `UI.RequestPlayerOperation` with setter fallback); regenerate the bundle
- [ ] 3.3 `protocol.py` additive fields; a legacy payload without them still parses
- [ ] 3.4 `agent_loop` handles a blocked END_TURN without raising, in the same PR as 3.2's END_TURN change
- [ ] 3.5 `tech_registry.py` through `alpha_zero.rules.registry`; encoder tech and civic slots non-zero on a fixture; a missing registry warns once and counts
- [ ] 3.6 `docs/MODULE_REFERENCE.md` `bridge` count updated

## 4. GM-B4: GameInfo snapshot
- [ ] 4.1 Lua `export_gameinfo` in the kernel's input format; `gameinfo_capture.py` writes only under `CIV6AZ_DATA_ROOT`
- [ ] 4.2 Hygiene test: no tracked file is a GameInfo snapshot; a path inside the checkout is refused

## 5. GM-B5: search proxy and `--mcts`
- [ ] 5.1 An un-mocked test that drives `Civ6MctsPolicy` as `civ6_local` wires it and records the failure
- [ ] 5.2 `microciv_proxy.py` maps unit classes and impassable tiles using existing MicroCiv rules only
- [ ] 5.3 `--mcts` refuses at start-up naming the gap id; CLAUDE.md section 5.1 rows for `--mcts` and `--eval-harness` updated

## 6. Close-out
- [ ] 6.1 Empty play-path diff on every PR; `make verify`; maintainer live checklist (reinstall the mod, one research choice, one reported blocker, one snapshot)
