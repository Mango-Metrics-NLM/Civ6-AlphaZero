# Design notes -- eval-decoupling-diagnostics

## The default coupling is preserved

`TrainingConfig.eval_interval` is `None` by default and resolves to
`save_interval`. A run without the new flags evaluates on exactly the iterations
it did before, with the same 20 games. The budget model falls back to
`save_interval` when `eval_interval` is absent, so the defaults still price at
10,200 and `--dry_run` at 8.

The budget model counts evaluations exactly: the iterations `i` in `[S, I)` with
`(i + 1) % eval_interval == 0`. It does not use the plan's
`ceil((I - S) / eval_interval)`, which is an approximation. The two agree at the
defaults and on the A1 command, and the exact count is what the loop plays.

## Why the float moved to a shim

`wins / num_games` cannot exceed `1 - draw_rate`, and it reads a run of draws as
a run of losses. While it stays the primary return value, it stays the number
people quote. The shim keeps old callers working, with a warning. The legacy
figure is still logged, and it is written as `win_rate` in `eval_<iter>.json`,
beside the score rate and never instead of it.

## Prequential, not in-sample

The metrics are computed on the iteration's fresh positions:

- by the network as it stood before training on them;
- before those positions enter the buffer;
- before the network trains on them.

That network also played the games, with one exception: under the default
`--bootstrap simple_mcts`, iteration 0 is played by random-rollout SimpleMCTS and
scored by the random-init network. Every prequential block records its
`generator` (`simple_mcts` or `network`) so the two cases are never confused.

This is predict-then-update. The R2 baseline is a constant the network could have
known beforehand: the buffer's mean value target before insertion. At iteration 0
the buffer is empty, so the baseline is 0. Using the fresh games' own mean there
would leak the targets into the baseline.

Sign accuracy reuses `TrainingDiagnostics.compute_value_accuracy` on the
positions with `|z| = 1` (a prediction of exactly 0 counts as wrong there). It is
rounded to six places, because that helper works in float32. Two baselines sit
beside it:

- `sign_accuracy_baseline`: the best constant sign for these very targets (the
  majority class). It is an in-sample oracle: the most any predictor blind to the
  position could score, and so a conservative bar.
- `sign_accuracy_baseline_prequential`: the constant `sign(c)`, with c the
  buffer mean known before the games, matching the R2 baseline. When c is 0 the
  constant predicts 0, which abstains on every position, so the baseline is
  `None` rather than a misleading 0.

## Sample size: games, not positions

All positions of one game share its outcome (up to sign), so R2_oos and sign
accuracy over a few hundred positions from 24 games rest on roughly 24
independent samples. The block records `n_games` (distinct games in the step
records) beside `n_positions`. These numbers are trend indicators across
iterations, not estimates to quote; a per-game bootstrap interval is the
right next step if anyone wants to quote one.

## Only played steps are recorded

`self_play_game` appends to the sink right after the real `env.step`. Every
search variant simulates on `env.clone()`, which is a different object, so
rollouts cannot reach the sink.

`test_histogram_counts_only_played_steps` counts steps separately on the real env
and on every clone. It requires the sink to match the real env's count, and it
requires the clone count to be strictly larger, so the test cannot pass vacuously.

## Failure mode

If a diagnostic raises, the trainer records it as `{"error": ...}` in the
iteration record and does not re-raise: a diagnostic must not end a multi-hour
run. The first failure is logged at ERROR with its traceback, later ones at
WARNING, and the run ends with a `prequential failures: k/N` line, so a
systematic failure cannot finish looking green. Both the fail-soft path and the
presence of real metrics are tested.

## The run record (`iterations.jsonl`)

- **Identity.** Every line carries `run_id`, `seed`, `git_sha`, `started_at`,
  `resumed_from` and `start_iteration`, so lines from different processes in
  one directory can always be partitioned.
- **Fresh runs rotate.** A run that starts from the top (no `--resume_from`,
  loop index 0) first moves an existing `iterations.jsonl` and every
  `eval_*.json` into a new `previous_<UTC stamp>/`. Nothing is deleted, and a
  shorter re-run cannot leave an earlier run's higher-numbered evaluations
  beside its own.
- **Resumes mark.** A resume appends; a line for an iteration the file already
  holds gets `supersedes_earlier_record: true`. Readers wanting one line per
  iteration take the last.
- **Crash safety.** Each line is a single `write`, then `flush` and `os.fsync`.
  A crash can still tear the line being written, so readers
  (`read_iteration_records`) skip any line that does not parse, and an
  appender terminates a torn tail before writing so a new line is never glued
  onto it.

## Which network an evaluation scored

Evaluation and checkpoint cadences are independent, so the weights an
evaluation scores may never be saved. `eval_<iter>.json` names its `checkpoint`
(relative to the checkpoint directory) or `null`, and its claim says "the network
after iteration N", not "this checkpoint".

## `git_sha` in run artifacts

`read_git_sha` resolves, in order: `CIV6AZ_GIT_SHA` (accepted only as 40 hex;
anything else is ignored with a warning), then the checkout at the package's
own source root, following a worktree's `commondir` to loose and packed refs.
It never reads the working directory or an ancestor: a pinned run executes from
a `git archive` export under the data root, and an unrelated repository above
that must not lend it a SHA. The run-card driver sets `CIV6AZ_GIT_SHA` for every
child, so pinned artifacts carry the pinned commit.
