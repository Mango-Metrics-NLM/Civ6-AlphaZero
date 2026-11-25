# civ6-bridge delta

## ADDED Requirements

### Requirement: The kick re-arms the mod and never ends the turn

The start-up kick (`CIV6_BRIDGE_KICK_START`, socket transport) SHALL only ask the mod to re-arm
its command window, by setting `ExposedMembers.AlphaZeroRearm` to a fresh nonce through the same
Lua context as commands. The Lua it sends MUST NOT name `RequestAction`, `ACTION_ENDTURN` or any
other end-of-turn request. The mod SHALL read the flag before its poll's early return and MUST
accept it only when the game's local player is the bridge's player, the game says that player's
turn is active, the command window is closed, no command is queued or awaits its state export,
and no end of turn has been carried out this turn; on accepting it SHALL reopen the command
window from now and export the state once. Where the game cannot be asked whether the turn is
active, the request MUST be ignored. The mod MUST clear the flag whether it accepts or ignores
it, and MUST drop a pending flag at a turn start, a reset and initialisation. Each answer SHALL be
printed as `AZ_REARM:<nonce>:accepted` or `AZ_REARM:<nonce>:ignored:<reason>`. The deprecated `kick_end_turn` MUST behave as the re-arm. A failing kick MUST
be logged as a warning and MUST NOT stop the agent loop from starting.

#### Scenario: A lapsed window on our active turn

- **WHEN** the command window has lapsed on the bridge player's active turn and the re-arm flag is set
- **THEN** on the next poll the state is exported once, the mod waits for a command again, the flag is cleared, `Re-arm accepted <nonce>` is logged, and no request to the game is made

#### Scenario: Not our turn, turn not active, or the game cannot be asked

- **WHEN** the flag is set while the local player is another player, the turn is not active, or `IsTurnActive` cannot be called
- **THEN** nothing is exported, the window stays closed, the flag is cleared and `Re-arm ignored` is logged

#### Scenario: A command in flight

- **WHEN** the flag is set while a command's state export is pending, or a command is queued and unread
- **THEN** the re-arm is ignored with `command in flight` and only the command's own export follows

#### Scenario: The window is open, or the turn is ending

- **WHEN** the flag is set while the mod already waits for a command, or after an end of turn was carried out
- **THEN** it is ignored with `window open` or `turn ending` and nothing is exported

#### Scenario: A flag left pending into the next turn

- **WHEN** the flag is still set when the bridge player's turn starts
- **THEN** it is dropped (`Re-arm dropped <nonce>`) and the turn start's export is the only one

#### Scenario: A malformed flag

- **WHEN** the flag is an empty string or not a string
- **THEN** it is ignored and cleared

#### Scenario: The kick fails at start-up

- **WHEN** the kick raises (a tuner error or any other exception)
- **THEN** a warning carrying its message is logged and the agent loop still runs

### Requirement: A mod loaded during its own active turn starts the turn

When the mod initializes while the game's local player is the bridge's player and the game says
that player's turn is active, it SHALL start the turn as the turn-start event would: export the
state and wait for a command. Otherwise, including when the game cannot be asked, it MUST NOT.

#### Scenario: Loaded mid-turn

- **WHEN** the mod loads during the bridge player's active turn
- **THEN** one state is exported and the mod waits for a command

#### Scenario: Loaded on another player's turn

- **WHEN** the mod loads while another player's turn is active, or no turn is active
- **THEN** no state is exported and the mod does not wait for a command
