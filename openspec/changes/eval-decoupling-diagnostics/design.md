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

- by the network that generated them;
- before those positions enter the buffer;
- before the network trains on them.

This is predict-then-update. The R2 baseline is a constant the network could have
known beforehand: the buffer's mean value target before insertion. At iteration 0
the buffer is empty, so the baseline is 0. Using the fresh games' own mean there
would leak the targets into the baseline.

Sign accuracy reuses `TrainingDiagnostics.compute_value_accuracy` on the
positions with `|z| = 1`. Its baseline is the best constant sign (the majority
class), which is the most any predictor blind to the position could score.

## Only played steps are recorded

`self_play_game` appends to the sink right after the real `env.step`. Every
search variant simulates on `env.clone()`, which is a different object, so
rollouts cannot reach the sink.

`test_histogram_counts_only_played_steps` counts steps separately on the real env
and on every clone. It requires the sink to match the real env's count, and it
requires the clone count to be strictly larger, so the test cannot pass vacuously.

## Failure mode

If a diagnostic raises, the trainer logs the exception with its traceback and
records it as `{"error": ...}` in the iteration record. It does not re-raise: a
diagnostic must not end a multi-hour run. The per-iteration test asserts that
the metrics are actually present, so a silent failure would be caught there.
