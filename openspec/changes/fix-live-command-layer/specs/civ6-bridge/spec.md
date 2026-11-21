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

### Requirement: No command starts a war

Before a move or an attack the mod SHALL ask the game whether it would change the war state, and
MUST refuse the command when it would.

#### Scenario: An attack on a player at peace

- **WHEN** an `attack` command targets a unit of a player the agent is at peace with
- **THEN** the answer is a failure saying it would start a war, and nothing is requested

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
SHALL stop sending after a fixed number of commands, and SHALL count `--max-turns` in game turns.

#### Scenario: A refused command

- **WHEN** the mod answers a command with a failure
- **THEN** the same command is not sent again before the turn ends
