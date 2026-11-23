# civ6-bridge delta

## MODIFIED Requirements

### Requirement: A blocked end of turn is reported, not hidden

END_TURN SHALL report a blocked turn with the blocking type instead of returning
success unconditionally, and MUST NOT send the request to the game while the turn
is blocked. The game's list of blockers is the only authority: the mod MUST NOT let
`UI.CanEndTurn()` overrule a listed blocker, because the live game answers true while
one is active (seen 2026-11-21). When the game cannot be asked what blocks the turn,
the turn MUST be treated as blocked: END_TURN reports blocked naming a blocker that
says so, sends nothing, and the exported state carries that blocker. END_TURN MUST also be refused when the
bridge's player is not the game's local player, whose turn the request would end. The
Python agent loop MUST handle a blocked report without raising and count it.

#### Scenario: Unanswered research choice

- **WHEN** the stub blocks end-turn on a research choice and END_TURN is executed
- **THEN** the acknowledgement reports blocked with the research blocking type, and the agent loop increments its blocked counter without an exception

#### Scenario: The game says the turn can end while a blocker is listed

- **WHEN** `UI.CanEndTurn()` answers true and the blocker list names a blocker
- **THEN** END_TURN reports blocked and sends nothing

#### Scenario: The game cannot be asked

- **WHEN** there is no notification manager, or neither way of asking it works
- **THEN** END_TURN reports blocked naming `ENDTURN_BLOCKING_NOT_ASKABLE`, nothing is requested, and the state carries that blocker

### Requirement: A move or an attack does not start a war

Before an attack, and before a move onto a tile the player can see, the mod SHALL ask the game
whether it would change the war state, and
MUST refuse the command when the game says it would, when the game cannot be asked, or when
the game's answer is in a form the mod does not understand; the first such answer MUST be
reported in the game's log. The declare-war command MUST be
refused unless the setting `ALLOW_DECLARE_WAR` is on, and that setting MUST be off by default.

#### Scenario: An attack on a player at peace

- **WHEN** an `attack` command targets a unit of a player the agent is at peace with
- **THEN** the answer is a failure saying it would start a war, and nothing is requested

#### Scenario: The game cannot be asked

- **WHEN** the game's war-state check is missing or raises
- **THEN** the move or attack is refused and nothing is requested

#### Scenario: An answer that cannot be read

- **WHEN** the game's war-state check answers with something that is neither nothing nor a table
- **THEN** the move or attack is refused, nothing is requested, and the action is not offered

#### Scenario: Declaring war with the default settings

- **WHEN** the declare-war command is sent and `ALLOW_DECLARE_WAR` is off
- **THEN** the answer is a failure saying it would start a war, and the game's war state is unchanged

### Requirement: Offered actions are ones the game accepts

Every unit action the exporter offers SHALL be one the game's strict check accepts at export
time; move targets MUST be the game's own adjacent plots; an attack MUST NOT be offered where it
would start a war; a city's production choices SHALL be asked by row hash. An attack or a
ranged attack MUST be offered only on a unit or city the same exported state contains, so
none is offered while other players are not exported.

#### Scenario: Move targets are neighbours

- **WHEN** a unit stands on an even or an odd row
- **THEN** every offered move target is exactly one step away

#### Scenario: A visible enemy while other players are not exported

- **WHEN** `OBSERVE_ALL_PLAYERS` is off and a unit of a player at war stands next to an own unit
- **THEN** no attack is offered, and an attack sent by hand is still carried out
