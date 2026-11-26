# Tasks

## 1. GM-B10: the stand-in's defaults
- [x] 1.1 `UI.CanEndTurn()`, `CurrentlyBuilding()`, `YieldTypes` and a locked item as the live game has them; old readings as named overrides
- [x] 1.2 A test file that pins the defaults

## 2. GM-B9: refuse where the game cannot be asked
- [x] 2.1 Tests first (`sdlc-qa-sdet`), on the stand-in's game state through the mod's Lua
- [x] 2.2 End of turn: unknown blocker
- [x] 2.3 War check: unreadable answer
- [x] 2.4 `ALLOW_DECLARE_WAR`
- [x] 2.5 No attack offered when other players are not exported
- [x] 2.6 `findAIPlayer`
- [x] 2.7 Bundle regenerated; refusal message; docs; gap register
- [x] 2.8 `ALLOW_MAKE_PEACE` / `ALLOW_DENOUNCE`: off by default; with one on, a major-civilization target and the game's war and met state decide, asked fail-closed; a self-targeted diplomacy command is refused (tests first, planted defects)

## 3. Verification
- [x] 3.1 Independent review; planted defects; Lua and bridge suites with `CIV6AZ_REQUIRE_LUA=1`; empty play-path diff
- [ ] 3.2 Maintainer, on the real game: checklist items 11 and 12 first (`docs/LIVE_CIV6.md`) *(Partial 2026-11-26: Item 11 `unit_attack` vs peaceful `p10` -> `REFUSED [WOULD_START_WAR]` & `WAR_WITH_10=false`, and Item 12a full-movement occupied-peaceful `unit_move` -> `REFUSED [WOULD_START_WAR]`, passed live on Japan Turn 36/37; hostile melee combat vs an active war target remains open.)*
