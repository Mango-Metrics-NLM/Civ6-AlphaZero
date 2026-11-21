# DEV-SQE to Pre-PR: GM-B7b, the exporter offers what the executor carries out

- **Date:** 2026-11-21
- **Session:** session_01CXusP9Ct9R7eg25YLcY7R7
- **Branch:** `claude/gm-b7b-exporter-offers` (contains GM-B7a, PR #158, until that merges)
- **Spec:** `kb/specs/20261120-live-command-layer-plan.md`; OpenSpec `openspec/changes/fix-live-command-layer` (tasks 2.1 to 2.4)

## What was built

- `AlphaZeroActionExecutor.Check(playerId, command)`: the executor run without requesting.
- `StateExporter.lua`: every offered unit action and production item is one `Check` accepts;
  neighbours from `Map.GetAdjacentPlot`; no hand table of offsets.
- `alpha_zero/bridge/production_slots.py` and its use in `action_converter.py`: slots by position
  among the offered items of a kind; the command carries the item's name.
- Stand-in: the game's adjacency call; a city answers production only by hash.

## How it was checked

- Tests first by `sdlc-qa-sdet` from a written contract: 105 failed before.
- Independent review by `sdlc-architect`: no blocking defect; two medium and five low findings
  fixed with tests (8 failed before).
- Ten planted defects each fail the suite.
- Lua, bridge, action-converter and regression suites on the merged branch: 3,568 passed,
  13 skipped; integration 1,097 passed. `ruff check .`, single-file type check,
  `build_lua_bundle.py --check`, doc-sync, debt ratchet and play-path diff clean.
  `make verify`, the full type check and coverage were not run; CI is the gate.

## Not verified

Nothing ran in a real game. `docs/LIVE_CIV6.md` items 17 to 20 are this slice's: read one
exported state, let the greedy agent run, time one export, and three things only the game shows.

## Open

- A production slot means "the n-th offered item"; a learning policy needs a production registry.
- On a scaled map a chosen move target often is not an offered tile after scaling back.
- `sleep` is still never offered. Next slice: GM-B7c (the agent loop).
