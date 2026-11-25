# Tasks

## 1. Tests first
- [x] 1.1 Lua tests through the bundle on the stand-in game (`tests/unit/test_lua_rearm.py`), running the exact `REARM_LUA` text
- [x] 1.2 Python tests: the socket frame, the deprecated alias, `LiveBridge.kick_turn`, `civ6_local` (`tests/unit/test_kick_rearm_python.py`); the old kick test in `tests/unit/test_tuner_socket_bridge.py` changed
- [x] 1.3 Regression: the re-arm Lua can never end a turn (`tests/regression/test_kick_never_ends_turn_regression.py`)

## 2. Implementation
- [x] 2.1 `Main.lua`: re-arm check before the early return; mid-turn start in `Initialize`; flag cleared on `Reset`; bundle regenerated
- [x] 2.2 `tuner_socket_bridge.py`: `REARM_LUA`, `request_rearm`, `kick_end_turn` deprecated
- [x] 2.3 `live_bridge.py` and `civ6_local.py`
- [x] 2.4 Docs: `docs/LIVE_CIV6.md`, `alpha_zero/bridge/Agent.md`, CHANGELOG, CLAUDE.md, gap register

## 3. Verification
- [x] 3.1 Planted defects; Lua and bridge suites with `CIV6AZ_REQUIRE_LUA=1`; empty play-path diff
- [x] 3.2 Independent review; its findings (stale flag across turns, window open, queued command, turn ending, reasons, machine line) fixed tests first
- [ ] 3.3 Maintainer, on the real game: checklist items 33 and 34 (`docs/LIVE_CIV6.md`)
