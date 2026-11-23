# civ6-bridge -- delta

## ADDED Requirements

### Requirement: Unit commands are requested the way the game's own UI requests them

The mod SHALL resolve a unit action's operation by type name from the game's database, use that
row's hash, and request it only after the game's strict check accepts it. A move MUST carry no
attack modifier. A name the game's database does not have, or an operation the strict check
refuses, MUST be answered as a failure that names the reason, and nothing is requested.

#### Scenario: A unit moves

- **WHEN** a `move` command names an own unit and a tile the game accepts
- **THEN** the unit is on that tile after the game applies the request, and the answer is success

#### Scenario: The game refuses

- **WHEN** the strict check refuses the operation (no moves left, an impassable tile)
- **THEN** the answer is a failure with the reason, and the game state does not change

#### Scenario: An operation name the game does not have

- **WHEN** the type name is not a row of the game's operation or command tables
- **THEN** the answer is a failure naming it, not a success

### Requirement: A move or an attack does not start a war

Before a move or an attack the mod SHALL ask the game whether it would change the war state, and
MUST refuse the command when the game says it would or when the game cannot be asked. This
requirement covers moves and attacks only. Two gaps are registered and not closed by it: an
answer from the game in a form other than the expected list is read as "no change", and the
separate declare-war command has no gate.

#### Scenario: An attack on a player at peace

- **WHEN** an `attack` command targets a unit of a player the agent is at peace with
- **THEN** the answer is a failure saying it would start a war, and nothing is requested

#### Scenario: The game cannot be asked

- **WHEN** the game's war-state check is missing or raises
- **THEN** the move or attack is refused and nothing is requested

### Requirement: A production order names the item by the game's hash

A production command SHALL be sent with the database row's hash and the exclusive insert mode,
whether the command names the item by type name or by row index. An item that needs a plot MUST
be refused.

#### Scenario: Build from a name

- **WHEN** a production command names `UNIT_WARRIOR` for a city that can build it
- **THEN** the city is building it after the request is applied, replacing what it was building

#### Scenario: An item that needs placement

- **WHEN** a production command names a district
- **THEN** the answer is a failure saying placement is not supported

### Requirement: The state after a command reflects the command

After a command that did not end the turn, the mod SHALL export the next state on a later tick
than the request, and SHALL keep listening for commands for a window measured from the last
command handled, up to a fixed number of commands per turn.

#### Scenario: A move is visible in the next state

- **WHEN** a move is accepted
- **THEN** the next exported state shows the unit on its new tile

#### Scenario: A long turn

- **WHEN** commands arrive for longer than the window, each within the window of the one before
- **THEN** the mod is still listening

### Requirement: Offered actions are ones the game accepts

Every unit action the exporter offers SHALL be one the game's strict check accepts at export
time; move targets MUST be the game's own adjacent plots; an attack MUST NOT be offered where it
would start a war; a city's production choices SHALL be asked by row hash.

#### Scenario: Move targets are neighbours

- **WHEN** a unit stands on an even or an odd row
- **THEN** every offered move target is exactly one step away

### Requirement: The loop does not resend what the game refused

Within one game turn the agent loop SHALL NOT send a command the mod refused in that turn again,
SHALL choose research and each city's production at most once unless the choice had no effect,
SHALL send only `end_turn` after a fixed number of commands, and SHALL be able to stop after a
configured number of game turns (`CIV6AZ_AGENT_MAX_GAME_TURNS`). A choice that cannot be sent
MUST be re-selected, not answered by ending the turn. When the game keeps blocking the end of
turn and nothing sent changes that, the loop SHALL declare a stall and stop sending, not spin.

#### Scenario: A refused command

- **WHEN** the mod answers a command with a failure
- **THEN** the same command is not sent again before the turn ends

#### Scenario: A turn that cannot be ended

- **WHEN** the game answers `end_turn` with a blocker three times running and the state does not change
- **THEN** the loop reports a stalled turn and exits with code 3 (or, with `on_stall=wait`, sends nothing more in that turn)

#### Scenario: A limit in game turns

- **WHEN** `max_game_turns` is 3 and each turn takes several commands
- **THEN** commands are sent in exactly three game turns

### Requirement: The mod follows what the live game was seen to do

Where the live session of 2026-11-21 showed the game behaving differently from the reading
the mod was written from, the mod SHALL follow the game: a city's item counts as buildable
only when both the loose and the strict form of the game's check accept it; a city's current
build is read from its type hash when the game does not return the item itself; tile yields
are read under the yield names the live game uses; and whole numbers are written to JSON
without loss above 2^31. Each MUST have a test that fails on the earlier behaviour.

#### Scenario: A building that is not unlocked yet

- **WHEN** the game's loose check accepts a building and its strict check does not
- **THEN** the building is not offered and a `city_produce` for it is refused

#### Scenario: A large whole number

- **WHEN** a value above 2^31 is exported
- **THEN** the JSON carries every digit

## MODIFIED Requirements

### Requirement: A blocked end of turn is reported, not hidden

END_TURN SHALL report a blocked turn with the blocking type instead of returning
success unconditionally, and MUST NOT send the request to the game while the turn
is blocked. The game's list of blockers is the only authority: the mod MUST NOT let
`UI.CanEndTurn()` overrule a listed blocker, because the live game answers true while
one is active (seen 2026-11-21). The Python agent loop MUST handle a blocked report
without raising and count it.

#### Scenario: Unanswered research choice

- **WHEN** the stub blocks end-turn on a research choice and END_TURN is executed
- **THEN** the acknowledgement reports blocked with the research blocking type, and the agent loop increments its blocked counter without an exception

#### Scenario: The game says the turn can end while a blocker is listed

- **WHEN** `UI.CanEndTurn()` answers true and the blocker list names a blocker
- **THEN** END_TURN reports blocked and sends nothing
