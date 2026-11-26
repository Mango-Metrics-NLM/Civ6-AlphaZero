# selfplay-trajectories -- deltas

## ADDED Requirements

### Requirement: Checkpoints record their provenance

Every checkpoint written by the trainer SHALL carry primitive-only metadata
(`board_size`, `max_turns`, `num_players`, `mcts`, `gumbel_k`, `gumbel_sims`,
`c_scale`, `policy_target`, `bootstrap`, `seed`, `git_sha`, `env_version`) that
loads under `weights_only=True`. A reader that supplies a horizon MUST refuse a
checkpoint recorded at a different `max_turns`; a checkpoint with no metadata
MUST load with a warning.

#### Scenario: Horizon mismatch on resume

- **WHEN** a checkpoint trained at `max_turns` 30 is resumed at `max_turns` 60
- **THEN** `HorizonMismatch` is raised before any weight is loaded

#### Scenario: Legacy file

- **WHEN** a checkpoint without metadata is loaded with a horizon supplied
- **THEN** a WARNING is logged and the load succeeds

### Requirement: Iteration 0 is switchable and its starting network is pinned

`--bootstrap simple_mcts` (default) SHALL play iteration 0 with SimpleMCTS;
`--bootstrap none` SHALL play it with the configured search and construct no
SimpleMCTS. A fresh run SHALL write `checkpoint_0.pt` before any training, and a
resumed run MUST NOT rewrite it.

#### Scenario: Resuming from checkpoint_0

- **WHEN** a run resumes from `checkpoint_0.pt`
- **THEN** `checkpoint_0.pt` is byte-identical afterwards
