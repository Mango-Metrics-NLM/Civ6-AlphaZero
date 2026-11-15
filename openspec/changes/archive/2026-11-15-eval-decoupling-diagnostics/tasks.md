# Tasks

## 1. Test first
- [x] 1.1 Write `tests/unit/test_eval_summary_and_diagnostics.py`, RED before implementation. It covers:
  - the evaluation cadence and `--eval_games 0`;
  - pricing;
  - the ScoreSummary return and the float shim;
  - `eval_<iter>.json`;
  - the R2_oos fixture and the iteration-0 baseline;
  - the played-steps-only histogram.
- [x] 1.2 `tests/unit/test_game_budget.py`: eval-interval pricing cases

## 2. Implement
- [x] 2.1 `--eval_interval`/`--eval_games`, decoupled from `--save_interval`; budget model updated
- [x] 2.2 `evaluate_against_random -> ScoreSummary`, deprecated float shim, `write_eval_json`
- [x] 2.3 `StepRecord`, `step_sink`, `prequential_value_metrics`, `iterations.jsonl`
- [x] 2.4 Negative controls:
  - recording rollout steps fails the histogram test;
  - re-coupling evaluation to `save_interval` fails the interval test.

## 3. Documentation
- [x] 3.1 CLAUDE.md §5.1 row for the new flags and §5.2 decision row

## 4. Review fix round 1
- [x] 4.1 I1: `CIV6AZ_GIT_SHA` first (set by `run_card`), worktree `commondir`, package-anchored lookup
- [x] 4.2 I2: run identity on every `iterations.jsonl` line; fresh runs rotate into `previous_<UTC>/`; resumes mark re-run iterations
- [x] 4.3 M1-M8: fail-soft test and failure count, prequential sign baseline, `generator`, `n_games`, eval `checkpoint`, reproducible `command`, record consistency, fsync per line
