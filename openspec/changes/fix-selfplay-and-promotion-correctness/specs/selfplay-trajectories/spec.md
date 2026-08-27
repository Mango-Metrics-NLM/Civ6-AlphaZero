# Self-Play Trajectories — deltas for fix-selfplay-and-promotion-correctness

## ADDED Requirements

### Requirement: Every self-play path records action masks

Every self-play path that persists trajectories — sequential, MuZero and batch —
SHALL record the per-position legal-action mask alongside the observation and
search policy. A trajectory MUST NOT reach storage carrying observations without
their masks, because the trainer substitutes an all-ones mask for a missing key
and thereby disables masked policy loss without reporting a fault.

#### Scenario: Batch self-play stores informative masks

- **WHEN** batch self-play generates trajectories
- **THEN** each trajectory carries one action mask per position, each mask is the
  width of the action space, its entries are binary with at least one legal
  action, and the masks are not uniformly all-ones

#### Scenario: Batch and sequential trajectories are schema-identical

- **WHEN** batch and sequential self-play each generate a trajectory
- **THEN** the two trajectories have exactly the same set of fields

#### Scenario: Dry runs match the real trajectory shape

- **WHEN** any self-play path runs with `--dry_run`
- **THEN** the synthetic trajectories carry action masks, and mask and policy
  widths both equal the configured action-space size

### Requirement: Per-step fields advance together

The per-step fields of a trajectory SHALL always have equal length. A search
failure part-way through a game MUST NOT leave a trajectory in which some
per-step fields were recorded and others were not, because the trainer iterates
one field and indexes the rest.

#### Scenario: A mid-game search failure leaves no ragged trajectory

- **WHEN** the search raises after at least one position has already been
  recorded
- **THEN** every per-step field of every returned trajectory has the same length,
  and preparing a training dataset from them does not raise

### Requirement: Trajectories written before masks were recorded remain trainable

The training pipeline SHALL accept trajectories that predate mask recording, and
SHALL treat an absent or short mask list as "no mask available" rather than
failing. A rolling training window spans iterations written before and after this
change.

#### Scenario: Legacy and current trajectories train together

- **WHEN** a dataset is prepared from a trajectory carrying masks and one written
  without them
- **THEN** both contribute training rows, the legacy rows receive an all-ones
  mask, and the current rows keep their recorded mask

#### Scenario: An empty mask list falls back instead of raising

- **WHEN** a trajectory carries an `action_masks` field that is an empty list
- **THEN** the affected positions receive the all-ones fallback and no
  index error is raised

### Requirement: Requested self-play parallelism is honoured or reported

Vectorized self-play requires both more than one parallel game and the batch MCTS
variant. When parallelism is requested in a configuration that cannot use it, the
system SHALL warn rather than discard the request silently, and the shipped
defaults SHALL describe what the run will actually do.

#### Scenario: An unusable parallelism request warns

- **WHEN** self-play is configured with more than one parallel game and a
  non-batch MCTS variant
- **THEN** a warning names the required variant and states that the run will be
  sequential

#### Scenario: Defaults are self-consistent

- **WHEN** the orchestrator is invoked with no arguments
- **THEN** the resulting configuration does not request parallelism it will not
  use

#### Scenario: Warning about self-play does not change other phases

- **WHEN** the orchestrator warns about an unusable parallelism request
- **THEN** the configured MCTS variant is left unchanged, so the evaluation,
  arena and tournament phases are unaffected
