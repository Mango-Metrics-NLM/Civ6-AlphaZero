# Validation Iteration Zero: first post-fix training run

Generated 2026-08-28T02:44:55.233985+00:00 from commit `0c394f98d7f7ffbf4accfa65e139a471abc99f59`.

## Per-iteration results

| Iter | Self-play games | Examples | Self-play (s) | Train (s) | Final loss | Win rate vs random |
|-----:|----------------:|---------:|--------------:|----------:|-----------:|-------------------:|
| 1 | 20 | 2568 | 127.6 | 46.1 | 1.8438 | 5.0% |
| 2 | 20 | 2555 | 127.9 | 98.9 | 1.8210 | 10.0% |
| 3 | 20 | 2432 | 128.4 | 145.9 | 1.7847 | 0.0% |
| 4 | 20 | 2501 | 119.5 | 208.4 | 1.7432 | 10.0% |
| 5 | 20 | 2389 | 122.8 | 257.3 | 1.7208 | 0.0% |

## Run

| Setting | Value |
|---------|-------|
| `OMP_NUM_THREADS` | 2 |
| `batch_size` | 128 |
| `board_size` | 8 |
| `learning_rate` | 0.001 |
| `max_turns` | 20 |
| `mcts` | civ6 |
| `num_epochs` | 10 |
| `num_games` | 20 |
| `num_iterations` | 5 |
| `num_simulations` | 12 |
| `save_interval` | 1 |
| `seed` | none |
| device | `cpu` |
| python_version | `3.11.15` |
| torch_version | `2.13.0+cu130` |
| git_revision | `0c394f98d7f7ffbf4accfa65e139a471abc99f59` |
| network_parameters | 570367 |
| action_space_size | 376 |
| spatial_channels | 13 |
| num_players | 2 |
| wall clock | 1478.5s |
| exit code | 0 |

## Scope and limitations

**This run proves the loop executes end to end. It does not show learning.**

What it establishes:

* The training loop completes 5 iterations and exits 0, on CPU, in 24m39s. Before
  the fixes in this branch the documented smoke test ran 3h33m without finishing.
* Training loss falls monotonically across all five iterations (1.8438 -> 1.7208),
  so gradients flow and the network fits its targets.
* Self-play cost is now flat across iterations (127.6s, 127.9s, 128.4s, 119.5s,
  122.8s) even though iteration 1 uses SimpleMCTS and iterations 2-5 use the
  network. Before the thread fix those two arms differed by ~175x per game
  (1.8 s/game vs 321 s/game), which is what made network self-play look broken.

What it does **not** establish, and must not be cited for:

* **No learning.** Win rate vs random over the five iterations was 5%, 10%, 0%,
  10%, 0% -- no trend. Five iterations and 100 total self-play games is far below
  any budget at which AlphaZero-style improvement would be visible. Falling loss
  means the network fits its own targets, not that those targets are getting better.
* **The final model plays worse than random.** Five independent 20-game probes of
  the final checkpoint (`final_checkpoint_vs_random.json`) give draw-aware score
  rates of 0.25, 0.35, 0.375, 0.375 and 0.40 -- every one below 0.5. This is a real
  result at this budget, not a measurement artifact, and it is why no capability
  claim is made here.
* **The reported win rate is draw-blind.** `evaluate_against_random` returns
  `wins / num_games`, so the "0.0%" logged for iterations 3 and 5 conceals a
  50-75% draw rate. This is the same defect as the deployed promotion gate
  (`cloud/gcp/arena.py`, see `docs/runbooks/promotion-gate.md`) -- a third site of
  the same pattern, found by this run. The breakdown is now logged alongside.
* **Not reproducible.** `training_civ6.py` exposes no `--seed` flag and
  `MicroCivEnv.reset()` is unseeded in the eval path, so neither the run nor the
  probes can be replayed exactly. The five probes above are the direct evidence:
  same checkpoint, same opponent seed, score rates spanning 0.25 to 0.40. Any
  single 20-game number from this setup should be treated as one draw from a wide
  distribution -- which is also why the run's own per-iteration win rates cannot
  support a trend claim either way.
* Reduced scale throughout: 8x8 MicroCiv, 20 turns, 12 simulations, CPU only, one
  run, no baseline arm to compare against.

`OMP_NUM_THREADS=2` was set for this run. It is recorded because it materially
changed the wall clock (the training phase is ~17x slower at torch's default on
this host) and therefore belongs in the provenance, not because it affects the
results.
