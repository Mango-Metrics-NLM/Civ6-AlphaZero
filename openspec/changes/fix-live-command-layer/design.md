# Design notes -- fix-live-command-layer

## The source of truth is the game's own UI code

GM-B2's first version copied a working third-party project and was wrong; the game's own UI
scripts were right. Every call here is taken from those scripts (read in the public CQUI mod;
nothing copied), and the stand-in is written from the same reading. The stand-in therefore
proves "the mod does what the game's UI does", not "the game accepts it".

## One helper, resolved by name

The game's unit panel does not hard-code operations: it walks `GameInfo.UnitOperations` and
`GameInfo.UnitCommands` and uses each row's hash. The executor does the same through one helper:
type name -> row (operations first, then commands) -> hash -> strict check -> request. A name the
game does not have answers with a refusal that names it. This is why the least certain names
(skip, sleep) are safe to ship unverified: wrong is loud.

## A move cannot become an attack, and nothing starts a war

The game's own move routine adds an attack modifier, so a human's click on an enemy attacks. The
agent's move sends no attack modifier; attacking is its own command. Before any move or attack
the executor asks `CombatManager.IsAttackChangeWarState` and refuses when the answer is not
empty. Declaring war stays the diplomacy command's business.

## Requests are asynchronous

A request is applied by the game after the script returns. Exporting the state in the same frame
shows the old state, which the loop's spin guard (GM-B7c) would read as "the command did
nothing". The mod exports on a later tick, and the stand-in applies requests on a later tick so
that a test of the old behaviour fails.

## The command window

Thirty seconds from turn start strands a turn of many commands. The window runs from the last
command handled; a hard cap on commands per turn bounds a spin.

## Stand-in rules that make the tests mean something

- Operation and command type tables hold exactly the database rows' hashes; `MOVETO` and `WAIT`
  do not exist.
- The strict and the loose check differ (a unit with no moves passes the loose one only).
- A build is accepted only with a row hash and the exclusive insert mode; anything else is
  dropped silently.
- A unit that still needs orders blocks the end of turn.
- Tests assert on the game state after the tick, never on a recorded call.
