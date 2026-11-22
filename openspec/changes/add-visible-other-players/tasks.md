# Tasks

## 1. GM-B8a: an honest export
- [x] 1.1 Evidence for every game call from the game's UI code; plan reviewed by four roles
- [x] 1.2 Stand-in game: per-player visibility of tiles and units, a city-state, barbarians at a high id, diplomacy (`sdlc-qa-sdet`)
- [x] 1.3 Tests first from a written contract (`sdlc-qa-sdet`)
- [x] 1.4 `StateExporter.lua`: players, unit and city visibility, `other_players`, tiles
- [x] 1.5 `ActionExecutor.lua`: attack targets the player sees
- [x] 1.6 `protocol.py`, `other_players.py`, `StateConverter`
- [x] 1.7 Independent review (`sdlc-architect`); four findings fixed with tests (`tests/unit/test_lua_other_players_review.py`)
- [x] 1.8 Docs: `docs/LIVE_CIV6.md`, `CHANGELOG.md`, `CLAUDE.md`, gap register

## 2. Not in this change
- [ ] 2.1 GM-B8b: the simulator with an opponent (winning by elimination of what is visible is not a win); decision record
- [ ] 2.2 Turning `OBSERVE_ALL_PLAYERS` on by default (after the maintainer's live sitting)
- [ ] 2.3 A last-seen memory of cities now in fog; resources the player's technology does not reveal
