# Decouple in-loop evaluation from checkpointing, report the full score summary, add prequential diagnostics

## Why

The local trainer (`alpha_zero/training_civ6.py`) evaluated against a random
opponent only when it saved a checkpoint (`(iteration + 1) % save_interval == 0`),
always with 20 games, and returned `wins / num_games`, a figure that scores every
draw as a loss. The registered A1 learning-evidence run needs a checkpoint every
5 iterations and no in-loop evaluation, because the benchmark evaluates it
separately. The coupling made that impossible without paying for four extra
20-game evaluations. A run also had no record of whether the value head predicts
the outcomes of games it has not yet trained on. These are LF-A14, LF-A3 and
LF-A13 of the local-first plan, and they must land before any learning-evidence run.

## What Changes

- **LF-A14.**
  - `--eval_interval N` defaults to `--save_interval`'s value, so a run that sets
    neither flag behaves and prices exactly as before.
  - `--eval_games N` defaults to 20. `0` turns in-loop evaluation off, including
    under `--dry_run`.
  - Checkpoints keep the `--save_interval` cadence.
  - The game-budget model prices `eval_games x evaluations` on the evaluation
    cadence, so the A1 command with `--save_interval 5 --eval_games 0` prices at 480.
- **LF-A3 (CONFIRM-FIRST, BREAKING).**
  - `evaluate_against_random` returns the full `ScoreSummary` from
    `alpha_zero/evaluation/scoring.py`: counts, draw-aware score rate,
    ternary-variance interval and draw rate.
  - The old float is available from `evaluate_win_rate_against_random`, which
    emits a `DeprecationWarning`.
  - Each in-loop evaluation writes `eval_<iter>.json` (via `write_eval_json`) into
    the checkpoint directory, with an evidence block shaped like
    `docs/evidence/SCHEMA.md`. It is a run artifact, not committed evidence.
- **LF-A13.**
  - `self_play_game` and `run_self_play` take an optional
    `step_sink: list[StepRecord] | None`. Their return types are unchanged.
  - After self-play and before the fresh examples enter the replay buffer, the
    trainer computes `prequential_value_metrics`, which is new in
    `alpha_zero/training/diagnostics.py`. `TrainingDiagnostics` is unchanged.
  - The metrics:
    - R2_oos against the buffer's prior mean (0 at iteration 0);
    - sign accuracy on `|v| = 1`, beside the best constant sign;
    - target and policy entropy;
    - the action-type histogram;
    - the invalid-action count.
  - They are logged and written to `<checkpoint_dir>/iterations.jsonl`. They are
    recorded, never gated.

## Approval

LF-A3 changes a public return type, which CLAUDE.md §3.2 makes CONFIRM-FIRST. The
maintainer's approval of the local-first plan
(`kb/specs/20261111-local-first-and-branching-plan.md`, task T10) covers it,
because the plan specifies the float shim and its deprecation warning.

## Impact

- **Behaviour:**
  - Training is unchanged. The diagnostics forward pass runs in eval mode under
    `no_grad` and restores the training mode afterwards; a weights-equality test
    pins this.
  - Each run writes two new kinds of file: `eval_<iter>.json` and `iterations.jsonl`.
- **API:**
  - The return type of `evaluate_against_random` changes (breaking; a shim is
    provided).
  - New: `TrainingConfig.eval_interval`, `TrainingConfig.eval_games` and
    `ReplayBuffer.mean_value`.
- **Affected specs:** `training-evaluation` (a new capability).
- **Code:** `alpha_zero/training_civ6.py`, `alpha_zero/training/diagnostics.py`
  and `alpha_zero/utils/game_budget.py`.
- **Tests:** `tests/unit/test_eval_summary_and_diagnostics.py` and
  `tests/unit/test_game_budget.py`.
