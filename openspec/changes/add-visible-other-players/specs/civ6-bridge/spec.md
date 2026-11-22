# civ6-bridge -- deltas

## ADDED Requirements

### Requirement: Other players are exported only as far as the player sees them

With `OBSERVE_ALL_PLAYERS` on, the mod SHALL export another player's unit only when
the game reports both its tile and the unit itself as visible to the local player,
and another player's city only while its tile is visible now. A foreign unit MUST
NOT carry its remaining moves, remaining attacks or promotions, and a foreign city
MUST NOT carry its production. When the game cannot be asked about visibility,
nothing of another player SHALL be exported. With the setting off, no other
player's unit or city SHALL be exported.

#### Scenario: A stealth unit on a visible tile

- **WHEN** another player's unit stands on a tile the player sees but the game reports the unit as not visible
- **THEN** the unit is not in the exported state

#### Scenario: Barbarians and city-states

- **WHEN** a barbarian unit (player id 63) and a city-state unit are in sight
- **THEN** both are exported, whatever the number of major players

#### Scenario: A city seen earlier

- **WHEN** another player's city stands on a tile that was revealed but is not visible now
- **THEN** the city is not in the exported state

### Requirement: The state says what each other owner is to the player

The state SHALL carry `observes_others` and, for every owner of an exported foreign
unit or city, an `other_players` entry with `kind` (major, minor, barbarian, free)
and `at_war`. `at_war` MUST be absent, not false, when the game cannot be asked.
The Python converter SHALL put only hostile owners (at war, barbarian, free city)
on the enemy side and SHALL count the units and cities it leaves out. A state that
carries no `observes_others` SHALL be converted as before.

#### Scenario: A city-state at war

- **WHEN** a city-state the player is at war with has a unit in sight
- **THEN** its entry has `kind` minor and `at_war` true, and the unit is on the enemy side

#### Scenario: A neutral scout

- **WHEN** a major the player is at peace with has a unit in sight
- **THEN** the unit is not on the enemy side and is counted as left out

### Requirement: Nothing the player cannot see now is given away

A tile that is not visible now MUST NOT carry its owner, improvement, resource or
yields. When the game cannot be asked about visibility, no tile SHALL be reported
as visible. An attack MUST NOT be offered or carried out on a plot whose only
foreign occupant the player cannot see.

#### Scenario: A tile in fog

- **WHEN** a tile was revealed earlier and is not visible now
- **THEN** it is exported with its terrain and features and without owner, improvement, resource or yields

#### Scenario: A hidden unit next to an archer

- **WHEN** the only foreign unit on a plot in range is one the player cannot see
- **THEN** no attack on that plot is offered, and one sent anyway is refused
