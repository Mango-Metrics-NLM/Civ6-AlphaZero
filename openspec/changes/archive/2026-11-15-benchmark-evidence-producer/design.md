# Design notes -- benchmark-evidence-producer

## The pair interval is the plain pentanomial interval

Each pair score is the mean of its two seat-swapped game scores, so it lies in
`{0, 0.25, 0.5, 0.75, 1}`. With `P` pair scores `s_p` and `z` the two-sided
normal quantile for the confidence level:

    m  = mean(s_p)
    v  = sum (s_p - m)^2 / (P - 1)          (sample variance, ddof 1)
    CI = m +/- z * sqrt(v / P), clipped to [0, 1]

There are no pseudo-counts. For `P < 2` the interval is `(0, 1)`. With zero
variance it has zero width at the mean. That is safe because the verdict reads
the hull, and the hull's iid member (`score_interval`, which keeps its
Agresti-Coull pseudo-counts) always has positive width.

### Why not pseudo-counts (fix round 1, review C1)

The first version applied `score_interval`'s adjustment per pair: `z^2/2`
pseudo-pairs at 1 and at 0, each of maximal variance. A pair is two games, so
that is twice the iid estimator's shrinkage weight. At the symmetry arm's size
(20 pairs, 99%) the pseudo-pairs were 25% of the sample. Because VOID needs both
intervals to exclude 0.5, the over-wide pair interval vetoed the iid interval's
detections. Measured by the seeded operating-characteristic tests in
`tests/unit/test_scoring_pair_interval.py`, which run the benchmark's own
`interval_summary`:

| Verdict condition | pseudo-count pair | plain pentanomial (now) | reviewer, 40k reps (plain / iid alone) |
|---|---:|---:|---:|
| Symmetry, 20 pairs, 99%: P(VOID) at a 0.20 asymmetry (mu 0.70, d 0.47) | 0.395 | 0.832 | 0.83 / 0.84 |
| Symmetry: P(VOID) for a fair arena (mu 0.5, d 0.47) | 0.0001 | 0.0042 | 0.0047 / 0.0049 |
| Positive control, 40 games, 95%: P(L <= 0.5) at mu 0.7375, d 0.3 | 0.062 | 0.029 (iid alone 0.027) | 0.029 / 0.026 |

The pseudo-count estimator fails two of the three tests; that is the negative
control.

## Both confidence levels are always reported

A1 judges its arms at 95% and the symmetry control at 99%. The registered
commands carry no confidence flag, so every row reports both levels. The legacy
`ci_low` / `ci_high` / `elo_ci_*` fields stay at 95% iid, unchanged.

## `hull_ratio` and the widened MDE

`mde_alpha025_80pct` is the INCONCLUSIVE report's MDE:
`minimum_detectable_effect(n, d, alpha=0.025)`.

`hull_ratio` (`benchmark_evidence.facing_hull_ratio`) is how much further the
95% hull reaches toward 0.5 than the 95% iid interval does, measured from the
point estimate `c` (the score rate):

- `c > 0.5`: `(c - hull_low) / (c - iid_low)`;
- `c < 0.5`: `(hull_high - c) / (iid_high - c)`;
- `c = 0.5`: the larger of the two.

A degenerate denominator gives 1. The result is always at least 1. The verdict
and the MDE concern the bound that faces 0.5, so a hull wider only on the far
side does not widen the MDE. The first version used the ratio of full widths.
`mde_alpha025_80pct_hull` is `mde_alpha025_80pct * hull_ratio`.

## Shards: the pair is the unit, the seeds are global

`--shard i/N` (1-based) takes the `i`-th of `N` contiguous, near-equal blocks of
pair indices. The arena plays games `2p` and `2p + 1` for each pair `p` in the
block, through `run_arena(game_indices=...)`.

Every seed is derived from the global index:

- the map comes from `protocol.env_seeds(seed)[p]`, i.e. `derive_seed(seed, "bench.env", p)`;
- each side's search comes from `derive_seed(seed, ARENA_SEARCH, g, side)`.

A shard therefore plays exactly the games the unsharded run plays at those indices.

`--merge` validates every shard and refuses a set that is not exactly `1..N`,
or whose shards disagree on protocol, candidate, seed, commit or opponents.
It then rebuilds each row from the union of the per-game records, so every
statistic is recomputed rather than averaged. The merged evidence keeps the
shards' `git_sha`, because that commit played the games, and lists each shard's
command, time and hardware.

`validate_envelope` requires each row's games to be exactly the shard's pairs,
from both seats. A split pair therefore cannot be written or merged.

## Evidence block

`search_epoch` is `post-2026-09-04`, one of the two values the contract allows.
The previous free-form label survives as `search_semantics`.

`command` is the `shlex`-joined command that actually ran:

- `python scripts/benchmark_suite.py ...` (as typed) when the wrapper ran;
- `python -m alpha_zero.evaluation.benchmark ...` otherwise, including
  `run_card` children.

`git_sha` comes from `utils.checkpoint.read_git_sha`, which honours
`CIV6AZ_GIT_SHA`. The block also records:

- `device`;
- `torch` and `numpy` versions;
- `search`: `mcts_type`, `num_simulations`, `gumbel_k` and `search_semantics`.

`--merge` refuses shards that disagree on any of these, and the merged block
carries the shards' values rather than the merging process's.

## A result is never overwritten (fix round 1, review I1)

The default `--out` is `./benchmark_out`, and the seven registered A1 arena
commands carry no `--out`, so each arm would have replaced the previous arm's
`result.json`. The CLI now refuses, with exit 2, before playing any game when
`<out>/result.json` exists, and it writes the file in exclusive-create mode.
Giving each arm its own `--out` in section 7.1 is T13's job.

A shard's `non_claims` begin with "this is shard i/N; merge every shard
before quoting it".

## What was measured

The A1 card's per-invocation prices, through `alpha_zero.local.run_card.price_card`:

| Invocation | Games |
|---|---:|
| training | 480 |
| primary | 200 |
| symmetry | 40 |
| positive control | 40 |
| secondary 5, 10, 15 | 60 each |
| final vs random | 40 |
| **Total** | **980** |
