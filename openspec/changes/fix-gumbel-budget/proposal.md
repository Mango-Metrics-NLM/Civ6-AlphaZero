# Spend exactly the configured Gumbel search budget

## Why

Sequential halving lived in three hand-written copies: `GumbelMCTS`
(`core/mcts/gumbel.py`), `BatchGumbelMCTS` (`core/batch_mcts.py`) and
`MuZeroMCTS` (`core/mcts/muzero.py`). Each used a per-round floor of
`max(1, (n // rounds) // survivors)` and stopped at one survivor. As a result
the realised number of simulations was not the configured `n`, and the copies
disagreed with each other.

This is defect LF-A11 of the local-first plan. Two related defects ship with it:

- A6: the search flags were missing.
- A7: `--mcts gumbel` ignored `--num_simulations`.

The fix must land before any learning-evidence run, because a run's search
budget is part of what its evidence claims.

Measured before the fix, with deterministic fakes and a never-terminating
environment:

| (k, n, num_legal) | realised (all three copies) |
|---|---|
| (8, 24, 30), (16, 64, 30) | 24, 64 (already exact) |
| (16, 64, 5) / (16, 64, 3) | 40 / 30 |
| (16, 32, 30) / (16, 16, 30) | 40 / 32 (overspend) |
| (3, 10, 30) | 3 |
| (16, 8, 30) | 30 |

The copies also diverged in two ways:

- Batch derived its round count from the largest candidate set in the batch.
  One position spent 30 simulations when searched alone and 15 when searched
  beside a position with more legal moves.
- MuZero hard-coded the survivor floor, which its siblings read from config.

## What Changes

- **One schedule.** `gumbel_ops.halving_schedule` and
  `planned_halving_sims(k, n, num_legal)` are the single schedule that all three
  searches iterate. It follows Danihelka et al., as implemented in `mctx`.
  - It spends exactly `n` whenever `num_legal >= 2`. This includes `k = 1`,
    where all `n` simulations go to the lone candidate.
  - It spends 0 for a forced move (`num_legal <= 1`).
- **Searched moves only.** The returned move is chosen only among the
  most-visited candidates (`choose_among_most_visited`, after mctx's
  `score_considered`).
- **Measurement.** Each search measures itself into
  `last_search_stats: SearchStats(root_evals, sim_evals)`. Return types are
  unchanged.
- **Trainer flags.** `training_civ6` gains `--gumbel_k`, `--gumbel_sims`,
  `--c_scale`, `--policy_target` and `--greedy_threshold`. One builder threads
  them into both self-play and evaluation.
  - `--mcts gumbel` honours `--num_simulations` unless `--gumbel_sims` is given.
  - Passing a Gumbel-only flag without `--mcts gumbel` is a parser error.
  - `--dry_run` bounds the Gumbel budget when the run left it at its default.

## Impact

- **The shipped Gumbel defaults change more than seeded trajectories (m-6).**
  - At the default `k = 16, n = 16`, self-play now gives every candidate exactly
    one visit and does no halving at all. With `R = 4` rounds,
    `16 // (4 * 16) = 0`, raised to the floor of 1 visit per survivor, which
    uses the whole budget in round 0. Before the fix it spent 32 simulations
    over 4 rounds.
  - Evaluation searches at half the budget, `(16, 8)`. It now visits only the
    top 8 of the 16 candidates; before the fix it spent 30 simulations.
- **The default is a degenerate shape.** `k = 16, n = 16` cannot halve: the
  schedule needs roughly `n >= k * R` before any halving happens. The registered
  budget for evidence runs is `k = 8` with `n >= 24`
  (`--gumbel_k 8 --gumbel_sims 24`), where the schedule is three full rounds of
  8 simulations each.
- **Trajectories.** Seeded Gumbel trajectories differ from earlier runs at
  every point that was not already exact.
- **Config semantics.** `min_actions_after_halving` now only reshapes the
  rounds; values below 2 act as 2.
- **Not changed.** `BatchGumbelMCTS` still has three known defects
  (CLAUDE.md §5.3):
  - It still emits a one-hot target.
  - It still ignores `policy_target` and `greedy_threshold`.
  - It still negates `child.value` unconditionally when ranking survivors.
    The sign rule is tracked with T18.
- **Affected specs:** `search-budget` (new capability).
- **Code:** `alpha_zero/core/mcts/{gumbel_ops,gumbel,muzero}.py`,
  `alpha_zero/core/batch_mcts.py` and `alpha_zero/training_civ6.py`. Tests are in
  `tests/unit/test_search_budget.py`.
