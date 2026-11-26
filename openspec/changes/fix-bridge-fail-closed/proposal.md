# The mod refuses where it cannot ask

## Why

A five-role review of the bridge track (2026-11-23) found four places where the mod goes
ahead when it should refuse, all reachable with the default settings:

1. The end-of-turn check answers "nothing blocks" when the game cannot be asked. The live
   session of 2026-11-21 showed that an end of turn sent while a blocker is active
   deactivates the turn (LF-H1), so "could not ask" must not read as "free to end".
2. The war check reads an answer it does not understand as "no war".
3. The declare-war command has no gate, while every move and attack is refused when it
   would start a war.
4. Attacks on visible barbarians are offered to an agent whose exported state, by default,
   contains no foreign unit.

The same review found that the stand-in game used by the Lua tests still modelled, by
default, four behaviours the live session disproved; only the tests written for that
session switched the live behaviour on.

Gap register rows GM-B9 and GM-B10.

## What Changes

- `GetEndTurnBlocker` reports "could not ask" as a third value; `end_turn` then answers
  `END_TURN_BLOCKED` naming `ENDTURN_BLOCKING_NOT_ASKABLE`, sends nothing, and the state carries
  that blocker.
- A war-check answer that is neither nothing nor a table refuses the move or attack; a
  non-empty table in any form is a war.
- New setting `ALLOW_DECLARE_WAR`, off by default; with it off the declare-war command
  answers `WOULD_START_WAR` and nothing reaches the game.
- With `OBSERVE_ALL_PLAYERS` off the exporter offers no attack and no ranged attack. The
  executor still carries out one that is sent by hand.
- The bridge finds its player from `Game.GetLocalPlayer()`, then from the game's own player
  list; `PlayerManager.GetWasEverAliveCount` is no longer called.
- The stand-in's defaults are the live behaviour; the old readings are named overrides.
- The `--mcts` refusal message no longer says the translator is missing.

## Impact

- Live behaviour of the mod changes for the greedy mode (confirm-first; approved by the
  maintainer with the close-out plan on 2026-11-22). A Lua context without
  `NotificationManager` can no longer end a turn through the mod: the agent declares a stall.
- **Not run in a real game.** The stand-in is written from the game's UI code.
- Nothing on the release play path changes.
