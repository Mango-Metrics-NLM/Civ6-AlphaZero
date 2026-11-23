# DEV-SQE to Pre-PR: GM-B9 and GM-B10, the mod refuses where it cannot ask

- **Date:** 2026-11-23
- **Session:** session_01CXusP9Ct9R7eg25YLcY7R7
- **Branch:** `claude/gm-b9-fail-closed` (contains the close-out branch of #167 until that merges)
- **Spec:** OpenSpec `fix-bridge-fail-closed`; gap register rows B9, B10

## What was built

Mod (`ActionExecutor.lua`, `StateExporter.lua`, `Main.lua`, `Config.lua`; bundle regenerated):
- the end-of-turn check has a third answer, "could not ask"; `end_turn` then answers
  `END_TURN_BLOCKED` naming `ENDTURN_BLOCKING_NOT_ASKABLE`, sends nothing, and the state
  carries that blocker; a blocker query that answers anything but a number falls through to
  the list scan; `end_turn` is refused when the bridge's player is not the local player;
- a war-check answer the mod cannot read refuses the move or attack and is reported once;
- `ALLOW_DECLARE_WAR`, off by default;
- no attack is offered while other players are not exported;
- the bridge's player comes from `Game.GetLocalPlayer()`, then the game's own player list.

Stand-in (`tests/fixtures/lua/`): the defaults are what the live session of 2026-11-21 showed;
an end-of-turn request sent while blocked records that the turn was deactivated.

Python: only the wording of the `--mcts` refusal and one docstring.

## How it was checked

- Tests first from a written contract by `sdlc-qa-sdet`: 110 failed before the mod change.
- Independent review by `sdlc-architect`. One blocking finding (a `nil` answer from the
  blocker query still read as "nothing blocks", and an older test pinned it) and nine others,
  all fixed with tests: the local-player mismatch, the name collision, the silent loss of
  every move on an unreadable war answer, seven older assertions that had become vacuous, a
  stand-in that could not tell a wrong request from a correct refusal, two untested branches.
- Planted defects: 15 in `plant_b9`, all caught, plus seven by the test role for the review
  fixes.
- Lua and bridge suites with `CIV6AZ_REQUIRE_LUA=1`, regression and governance suites, bundle
  check, `ruff`, validators. Play-path diff empty.

## Not done, and what only a live game can show

- Nothing ran in a real game. The ranked list of game calls still unseen live is in the PR
  description; the war check and "a move never becomes an attack" come first.
- `CIV6_BRIDGE_KICK_START` still sends the end-of-turn request with no check (documented).
- Making peace and denouncing are not gated.
- If the live game's blocker query or notification manager behaves differently, the agent now
  stalls where it used to send a bad end of turn. That is the intended direction.
