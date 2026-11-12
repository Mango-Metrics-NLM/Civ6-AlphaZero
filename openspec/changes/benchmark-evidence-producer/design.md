# Design notes -- benchmark-evidence-producer

## The pair interval is the existing estimator, applied to pairs

Each pair score is the mean of its two seat-swapped game scores, so it lies in
`{0, 0.25, 0.5, 0.75, 1}`. With `P` pair scores `s_p`, `z` the two-sided normal
quantile for the confidence level, and `k = z^2 / 2` pseudo-pairs at 1 and `k` at 0:

    n' = P + 2k
    m  = (sum s_p + k) / n'
    v  = (sum s_p^2 + k) / n' - m^2
    CI = m +/- z * sqrt(v / n'), clipped to [0, 1]

This is `score_interval`'s ternary-variance interval with its Agresti-Coull
style adjustment, with the pair as the unit instead of the game. A test pins the
reduction: units of 0, 0.5 and 1 give exactly `score_interval(w, d, l)`. No new
estimator is introduced.

The pseudo-counts are per unit, so the pair interval shrinks toward 0.5 a little
more than the iid one at the same number of games. Its centre is therefore not
exactly the score rate. That is one reason the decision rule takes the hull,
`[min(lows), max(highs)]`, instead of choosing one of the two intervals.

## Why both confidence levels are always reported

A1 judges its arms at 95% and the symmetry control at 99%. The registered
commands carry no confidence flag, so every row reports both levels. The legacy
`ci_low` / `ci_high` / `elo_ci_*` fields stay at 95% iid, unchanged.

`mde_alpha025_80pct` is the INCONCLUSIVE report's MDE:
`minimum_detectable_effect(n, d, alpha=0.025)`. `hull_ratio` is the 95% hull
width over the 95% iid width, and `mde_alpha025_80pct_hull` is their product.

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
The previous free-form label survives as `search_semantics`. `command` is
`shlex.join(["python", "-m", "alpha_zero.evaluation.benchmark", *argv])`; the
two entry points take the same CLI. `git_sha` comes from
`utils.checkpoint.read_git_sha`, which honours `CIV6AZ_GIT_SHA`.

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
