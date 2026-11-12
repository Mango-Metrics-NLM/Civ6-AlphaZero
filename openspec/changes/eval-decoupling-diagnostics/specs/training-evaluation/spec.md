# training-evaluation -- deltas

## ADDED Requirements

### Requirement: In-loop evaluation has its own cadence

The local trainer SHALL schedule in-loop evaluation on `--eval_interval`
(defaulting to `--save_interval`) with `--eval_games` games, independently of
checkpointing, and SHALL run no in-loop evaluation when `--eval_games` is 0. The
game-budget model MUST price exactly the evaluations the loop plays.

#### Scenario: Evaluation and checkpoints on different cadences

- **WHEN** a 6-iteration run uses `--save_interval 3 --eval_interval 2`
- **THEN** evaluation runs after iterations 2, 4 and 6, and checkpoints are written after 3 and 6

#### Scenario: Evaluation turned off

- **WHEN** a run passes `--eval_games 0`, with or without `--dry_run`
- **THEN** no evaluation game is played and none is priced

### Requirement: Evaluation reports the full score summary

`evaluate_against_random` SHALL return a `ScoreSummary`: the counts, the
draw-aware score rate with its interval, and the draw rate. Each in-loop
evaluation SHALL write `eval_<iter>.json` holding those fields, the legacy win
rate beside them, and an evidence block. The legacy float MUST be available only
through a shim that warns of deprecation.

#### Scenario: Draws are visible in the result

- **WHEN** an evaluation of four games ends in one win, two draws and one loss
- **THEN** the summary reports those counts and a score rate of 0.5
- **AND** the shim returns 0.25 with a `DeprecationWarning`

### Requirement: Prequential value diagnostics are recorded

After self-play and before the fresh examples enter the replay buffer, the
trainer SHALL record:

- R2_oos of the generating network's value predictions, against the buffer's
  prior mean target (0 when the buffer is empty);
- sign accuracy on decisive positions, beside the best constant sign;
- target and policy entropy;
- the action-type histogram;
- the invalid-action count.

Only steps actually played SHALL be recorded; search rollouts MUST NOT be. The
diagnostics MUST NOT change the trained weights.

#### Scenario: Rollouts are excluded

- **WHEN** a self-play game is played with a step sink, by a search that steps cloned environments
- **THEN** the sink holds exactly one record per step played on the real environment

#### Scenario: Iteration-zero baseline

- **WHEN** the replay buffer is empty
- **THEN** the R2_oos baseline is 0
