# training-reproducibility — deltas

## ADDED Requirements

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
