# Tasks

## 1. Test first
- [x] 1.1 `tests/unit/test_search_budget.py`: schedule unit tests and `SearchStats` equality for all three searches; RED on the old floors
- [x] 1.2 `k = 1` with several legal moves spends `n` in all three searches; RED on the first schedule
- [x] 1.3 Adversarial `(16, 8)` one-hot test that no unvisited move is returned; RED (5 of 50 seeds)
- [x] 1.4 CLI round trip, Gumbel-only flags refused without `--mcts gumbel`, dry-run Gumbel bound, pinned default config

## 2. Fix
- [x] 2.1 `halving_schedule`, `planned_halving_sims` and `choose_among_most_visited` in `gumbel_ops`; all three searches use them
- [x] 2.2 `SearchStats` measured at the call sites
- [x] 2.3 `training_civ6` search flags through one builder; `--mcts gumbel` honours the budget
- [x] 2.4 Negative control: restoring the old floor in `batch_mcts` fails the equality tests

## 3. Documentation
- [x] 3.1 CLAUDE.md §5.2 decision row, §5.4 bug row and §5.1 command row
- [x] 3.2 CLAUDE.md batch-path claims corrected; §5.3 known-issue row for `BatchGumbelMCTS`
