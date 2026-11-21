# DEV-SQE to Pre-PR: GM-B7a, the mod's command executor

- **Date:** 2026-11-21
- **Session:** session_01CXusP9Ct9R7eg25YLcY7R7
- **Branch:** `claude/gm-b7a-command-executor` (from the default branch; the default branch merged in after #154 and #156)
- **Spec:** `kb/specs/20261120-live-command-layer-plan.md`; OpenSpec `openspec/changes/fix-live-command-layer` (tasks 1.1 to 1.7)

## What was built

Eight commands of the mod are requested the way the game's own UI requests them: move, attack,
ranged attack, skip, fortify, sleep, found city and city production
(`mod/AlphaZeroAI/Scripts/ActionExecutor.lua`). The main loop exports the next state two frames
after a command, keeps its command window open from the last command and caps commands per turn
(`Main.lua`, `Config.lua`). A production command may name its item (`alpha_zero/bridge/protocol.py`).
The stand-in game gained unit, city and combat requests that are carried out on a later frame
(`tests/fixtures/lua/civ6_stub_commands.lua`). The JSON encoder's control range is written out
(`JsonEncoder.lua`), a Windows defect the maintainer found.

## How it was checked

- Tests first, from a written contract, by `sdlc-qa-sdet`: 84 failed on the old bundle.
- Independent review by `sdlc-architect`: one high finding (the cap could trap a blocked turn)
  and several smaller ones, all fixed with tests (10 failed before).
- Twelve planted defects each fail the suite.
- Regression, governance, Lua, bridge and `civ6_local` suites on the merged branch: 3,512 passed,
  13 skipped. `ruff check .` clean, single-file type check clean, `build_lua_bundle.py --check`
  up to date, play-path diff empty. `make verify`, the full type check and coverage were not run
  (they do not run in the sandbox); CI is the gate.

## Not verified

Nothing ran in a real game. The stand-in was written from the same reading of the game's UI code
as the mod, so it cannot show that the game accepts these calls. The live checklist in
`docs/LIVE_CIV6.md` has 16 items; items 7 to 16 are this slice's.

## Open

- `unit_build`, `unit_promote`, `city_buy`, `city_citizen` are unchanged and still written from memory.
- A move onto an own unit is refused; the game's UI would swap the two units.
- The greedy policy can still choose districts and wonders; each is now a refused command (GM-B7b).
- Next slices: GM-B7b (exporter, greedy production by name), GM-B7c (the agent loop).
