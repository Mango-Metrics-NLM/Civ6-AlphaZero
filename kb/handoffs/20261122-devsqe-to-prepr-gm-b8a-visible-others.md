# DEV-SQE to Pre-PR: GM-B8a, what the player can see of other players

- **Date:** 2026-11-22
- **Session:** session_01CXusP9Ct9R7eg25YLcY7R7
- **Branch:** `claude/gm-b8a-visible-others`, from the default branch, independent of the GM-B6
  branches. **No PR:** #162 and #163 hold both slots and may not merge before the live sitting.
- **Spec:** `/root/.claude/plans` section "GM-B8" as approved 2026-11-22; OpenSpec
  `add-visible-other-players`; gap register rows B8a to B8c.

## What was built

- `mod/AlphaZeroAI/Scripts/StateExporter.lua`: other players found as the game's UI finds them;
  a foreign unit only when the game reports its tile and the unit visible, with only the facts
  the game shows; a foreign city only while its tile is visible now; `other_players`
  (`id`, `kind`, `at_war`, `met`) and `observes_others`. Tiles not visible now carry no owner,
  improvement, resource or yields; with no visibility table nothing is visible.
- `mod/AlphaZeroAI/Scripts/ActionExecutor.lua`: an attack needs a target the player sees; a
  refused move does not describe an occupant the player cannot see.
- `OBSERVE_ALL_PLAYERS` is still off by default.
- `alpha_zero/bridge/other_players.py` (new), `protocol.py` (two fields, no growth),
  `state_converter.py` (only hostile owners on the enemy side; `_others` counts).
- Stand-in: `tests/fixtures/lua/civ6_stub_others.lua` (visibility of tiles and units, a
  city-state, barbarians at 63, `Game.GetPlayers`, diplomacy on the combat check's one record).

## How it was checked

- Evidence for every game call from the game's own UI code before the plan was written; the
  plan was reviewed by four roles and rewritten.
- Tests first by `sdlc-qa-sdet` (157 tests in five files; 80 of the 105 Lua tests fail on the
  old mod, the rest are controls).
- Independent review by `sdlc-architect`: four findings, all fixed with tests
  (`tests/unit/test_lua_other_players_review.py`, 8 of 13 fail without the fixes).
- Twenty-three planted defects each fail the suite.
- Lua, bridge, script and regression suites pass apart from seven sandbox-only failures;
  governance passes; bundle in sync; `ruff check .`; single-file type checks; doc-sync, kb and
  OpenSpec validators; play-path diff empty. `make verify` not run; no CI (no PR).
- Two existing tests were corrected: an attack into fog now has no target (it used to answer
  `WOULD_START_WAR`, which was the leak), and the golden default state gained the two new keys.

## Not verified

Nothing ran in a real game. Seven items were added to the live checklist (21 to 27).

## Open

- GM-B8b: the search's simulator must not read killing the last visible enemy as a win. It
  needs this branch and the GM-B6b branch.
- B8c (registered): move offers still give away a hidden unit; no last-seen memory of cities;
  resources not gated by technology; a parsed foreign unit defaults `can_attack` to true.
