# Training Reproducibility

## Purpose

Makes a seeded training or evaluation run exactly reproducible from its seed: the
seed reaches every source of randomness (environments, every search variant,
opponents, replay sampling, batch order, network initialisation), searches own
their generators rather than drawing from global state, simulation never perturbs
the game being simulated, and an evaluation probe of a fixed checkpoint at a fixed
seed returns the same result every time.

## Requirements

### Requirement: A seed determines the whole run

A training or evaluation run given a seed SHALL be exactly reproducible from that
seed alone. The seed MUST reach every source of randomness the run depends on —
environments, every search variant, the random opponent, replay sampling, batch
order, and network initialisation — because a run that is only partly seeded
looks reproducible and is not, which is more dangerous than one that is plainly
unseeded.

Omitting the seed SHALL leave the run nondeterministic. A default seed would
silently change the behaviour of every existing caller.

#### Scenario: The same seed is given twice

- **WHEN** two training runs are started with the same seed and otherwise identical configuration
- **THEN** they produce identical self-play trajectories and identical final network weights

#### Scenario: Global RNG state is disturbed between two seeded runs

- **WHEN** the process-global numpy and torch generators are advanced between two runs at the same seed
- **THEN** the two runs still produce identical results, because the seed does not rely on global state

#### Scenario: Different seeds are given

- **WHEN** two runs are started with different seeds
- **THEN** they diverge, so that a passing equality check cannot be satisfied by a search that ignores its seed

#### Scenario: No seed is given

- **WHEN** a run is started with no seed
- **THEN** it executes normally and two such runs differ

### Requirement: No search draws from the global RNG

Every MCTS variant SHALL draw its randomness from a generator it owns, seeded
from its configuration. No module under the search packages may call a global
`np.random` sampling function, because such a draw is unreachable by any seed and
therefore cannot be made reproducible by a caller.

#### Scenario: A global draw is reintroduced in a variant no test exercises

- **WHEN** a module under `alpha_zero/core/mcts/` or `alpha_zero/core/batch_mcts.py` calls a global `np.random` sampling function
- **THEN** the determinism gate fails, whether or not that variant lies on a path any behavioural test executes

### Requirement: Simulation must not alter the game being simulated

Cloning an environment for search SHALL carry the environment's RNG state, and
SHALL NOT advance the RNG of the environment it was cloned from.

#### Scenario: A search clones the environment to simulate

- **WHEN** an environment is cloned during search and the clone consumes randomness
- **THEN** the clone's stream continues deterministically from the parent's state, and the parent's own stream is unchanged

### Requirement: An evaluation probe is repeatable

An evaluation of a fixed checkpoint at a fixed seed SHALL return the same result
on every run, so that a reported score is a measurement rather than one draw from
an unstated distribution.

#### Scenario: The same checkpoint is probed twice at one seed

- **WHEN** the same checkpoint is evaluated twice against the same baseline at the same seed
- **THEN** both probes report identical win, draw and loss counts

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
- **WHEN** a change to the env alters any configuration's digest while `ENV_VERSION` and numpy's Generator stream (the recorded stream canary) equal the fixture's
- **THEN** the test fails, names the configuration, and says that neither `ENV_VERSION` nor numpy's stream moved

#### Scenario: ENV_VERSION is bumped without regenerating the fixture
- **WHEN** `ENV_VERSION` differs from the value recorded in the fixture
- **THEN** the test fails and says that `ENV_VERSION` moved and the fixture must be regenerated in the same change

#### Scenario: numpy's Generator stream moved
- **WHEN** a digest differs and the recomputed stream canary differs from the one recorded in the fixture while `ENV_VERSION` is unchanged
- **THEN** the test fails and says that numpy's stream moved, so the difference is not reported as an env change

#### Scenario: numpy is upgraded and its stream is unchanged
- **WHEN** a digest differs, the installed numpy version differs from the one recorded in the fixture, and the stream canary equals the fixture's
- **THEN** the test fails and reports an env change: a version string alone does not explain a different game

#### Scenario: The negative control
- **WHEN** the action sequence of one seeded game is perturbed, or a game rule is changed in a scratch copy of the env
- **THEN** the test fails, which shows that the fingerprint can see a change to the game
