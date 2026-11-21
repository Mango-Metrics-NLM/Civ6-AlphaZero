# Tasks

## 1. GM-B7a: the executor
- [x] 1.1 Plan of record: task plan `kb/specs/20261120-live-command-layer-plan.md`, gap rows, this change; "command delivery verified" wording corrected
- [x] 1.2 Stand-in: unit, city and combat managers, operation rows with hashes, deferred application, strict and loose checks
- [x] 1.3 Tests first (`sdlc-qa-sdet`), each failing on the old bundle: move, attacks, skip, fortify, sleep, found city, production by name and by index, refusals with reasons, no war
- [x] 1.4 `ActionExecutor.lua` helper and the commands that go through it; bundle regenerated
- [x] 1.5 `Main.lua`: state exported on a later tick; command window from the last command; cap per turn
- [x] 1.6 `protocol.py`: a production command may name its item (no net growth)
- [x] 1.7 Negative controls: reverting each of E1, E2, E3, E7, E8 fails a named test

## 2. GM-B7b: the exporter and greedy production
- [x] 2.1 Neighbours from `Map.GetAdjacentPlot`; an independent Python adjacency check for both row parities
- [x] 2.2 Offered unit actions from the strict check; no attack that would start a war
- [x] 2.3 Production choices by hash, strict form, no placed items
- [x] 2.4 Greedy production by name in a new module; the same command in two processes

## 3. GM-B7c: the loop
- [ ] 3.1 `bridge/command_policy.py`: per-turn ledger, refused commands not resent, cap, game-turn counting
- [ ] 3.2 `scripts/live_command_check.py`

## 4. Verification
- [ ] 4.1 Empty play-path diff on every PR; debt ratchet; `build_lua_bundle.py --check`; Lua suites with `CIV6AZ_REQUIRE_LUA=1`
- [ ] 4.2 Maintainer's live sitting: the checklist of `add-bridge-research-parity` (reinstall the mod, one research choice, one reported blocker, one snapshot), plus a unit moves, skip clears "unit needs orders", a build starts from a name and replaces, ticks until a request shows, ten turns end by themselves
