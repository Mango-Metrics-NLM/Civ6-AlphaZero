## ADDED Requirements

### Requirement: The default environment's seeded trajectory is pinned by a golden fingerprint

A committed fixture MUST pin a digest of seeded legal play on the default `MicroCivEnv`
at board 8, with 30 and 60 turns and 2 and 3 players. The digest MUST be computed from
integer game state only: units, cities, turn, current player, winner and the action
sequence. It MUST NOT read float tensors or observations. The fixture MUST record the
`ENV_VERSION` and the numpy version it was generated with. A digest that differs from
the fixture while `ENV_VERSION` is unchanged MUST fail. An `ENV_VERSION` that differs
from the fixture's MUST also fail until the fixture is regenerated, so the default game
cannot change without an `ENV_VERSION` bump. The failure message SHALL say which of
`ENV_VERSION` and the numpy version moved, or that neither did.

#### Scenario: A refactor leaves the game unchanged
- **WHEN** the env code is refactored and the seeded games produce the same integer state and action sequences
- **THEN** every configuration's digest equals the fixture's and the test passes

#### Scenario: The game changes without an ENV_VERSION bump
- **WHEN** a change to the env alters any configuration's digest while `ENV_VERSION` and the numpy version equal the fixture's
- **THEN** the test fails, names the configuration, and says that neither `ENV_VERSION` nor the numpy version moved

#### Scenario: ENV_VERSION is bumped without regenerating the fixture
- **WHEN** `ENV_VERSION` differs from the value recorded in the fixture
- **THEN** the test fails and says that `ENV_VERSION` moved and the fixture must be regenerated in the same change

#### Scenario: numpy is upgraded
- **WHEN** a digest differs and the installed numpy version differs from the one recorded in the fixture while `ENV_VERSION` is unchanged
- **THEN** the test fails and says that the numpy version moved, so the difference is not reported as an env change

#### Scenario: The negative control
- **WHEN** the action sequence of one seeded game is perturbed, or a game rule is changed in a scratch copy of the env
- **THEN** the test fails, which shows that the fingerprint can see a change to the game
