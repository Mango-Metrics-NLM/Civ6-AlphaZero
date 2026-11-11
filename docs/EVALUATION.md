# Evaluation protocol

This page defines how strength is measured in this repository, which numbers may
be quoted as evidence, and why the shipped promotion gate cannot promote at the
measured draw rate. It is the single reference for the vocabulary used in
`alpha_zero/evaluation/scoring.py`, the arena (`alpha_zero/cloud/gcp/arena.py`),
the benchmark protocol (`alpha_zero/evaluation/benchmark.py`) and every artifact
under [`docs/evidence/`](evidence/README.md).

## 1. Vocabulary

| Term | Definition | Use |
|---|---|---|
| `win_rate` | wins / games. **Draw-blind.** At a 73.5% draw rate its ceiling is 0.265. | Legacy. Logged for dashboard continuity only; never gate on it, never quote it as strength. |
| `score_rate` | (wins + 0.5 · draws) / games | **The statistic for every claim.** 0.5 is parity. |
| `draw_rate` | draws / games | Always reported next to `score_rate`; it decides how informative a game is. |
| `decisive_win_rate` | wins / (wins + losses) | Only meaningful with the draw rate beside it. |
| interval | `scoring.score_interval(wins, draws, losses)`: a normal-approximation interval with Agresti-Coull-style pseudo-counts on the **ternary** 0 / 0.5 / 1 variance. A binomial (Wilson) interval on fractional successes is too wide by `draw_rate / 4` in variance and is not used. | Quote `[lo, hi]` at 95% with every score rate. |
| MDE | `scoring.minimum_detectable_effect(n, draw_rate)`: the smallest score difference detectable at α = 0.05 (one-sided) with 80% power | Decides the sample size *before* a run. |
| Elo (pairwise) | `-400 · log10(1/p − 1)` at `p = score_rate`; the interval maps through the same monotone transform (`scoring.elo_from_score_rate`, `scoring.elo_interval`) | A readable re-expression of one score rate, nothing more. |
| Elo (ladder) | `alpha_zero/tournament/elo.py`: sequential, K-decaying ratings | A **ladder rating**, order-dependent, with no interval. Not an estimator of strength; not quoted in evidence. |

Every logged `*win_rate` must be accompanied by `score_rate` and `draw_rate`
(`tests/regression/test_draw_aware_metrics_regression.py`). New draw-blind
computations are refused outside the reasoned allowlist in
`tests/regression/draw_aware_allowlist.toml`.

## 2. Sample size: the power table

Rows are games played; columns are the draw rate you expect; each cell is the
smallest score-rate difference from parity the run can detect (80% power,
one-sided α = 0.05), with the Elo that difference implies. Computed from
`scoring.minimum_detectable_effect`; `tests/scripts/test_evaluation_power_table.py`
recomputes every cell.

| games | draw rate 0.30 | draw rate 0.50 | draw rate 0.735 |
|---:|---|---|---|
| 20 | 0.233 (+175 Elo) | 0.197 (+144 Elo) | 0.143 (+102 Elo) |
| 50 | 0.147 (+105 Elo) | 0.124 (+88 Elo) | 0.091 (+64 Elo) |
| 100 | 0.104 (+73 Elo) | 0.088 (+62 Elo) | 0.064 (+45 Elo) |
| 200 | 0.074 (+51 Elo) | 0.062 (+43 Elo) | 0.045 (+32 Elo) |
| 400 | 0.052 (+36 Elo) | 0.044 (+31 Elo) | 0.032 (+22 Elo) |
| 800 | 0.037 (+26 Elo) | 0.031 (+22 Elo) | 0.023 (+16 Elo) |
| 1500 | 0.027 (+19 Elo) | 0.023 (+16 Elo) | 0.017 (+11 Elo) |

Two consequences that decide runs here:

- **The deployed arena plays 20 games.** At the measured draw rate it can only
  detect a difference of about 0.14, roughly +100 Elo in one iteration. Smaller
  real improvements return INCONCLUSIVE by construction.
- **Detecting a true +14 Elo (score 0.52) needs about 1,160 games** at 70% draws
  (one-sided), or about 1,473 under the gate's two-sided lower-bound rule.

Draws make a game *more* informative at parity (the variance of a 0/0.5/1
outcome is `(1 − d)/4`), which is why the right-hand column needs fewer games;
but a 0.5 is still worth less than a decisive result to a learner, which is why
the draw rate is also an environment problem (`docs/runbooks/promotion-gate.md`).

## 3. The promotion gate

Policies are registered in `alpha_zero/evaluation/promotion.py`
(`CIV6AZ_PROMOTION_POLICY`, `CIV6AZ_PROMOTION_THRESHOLD`, `CIV6AZ_PROMOTION_CONFIDENCE`,
`CIV6AZ_PROMOTION_MIN_GAMES`):

| Policy | Promotes when | At 73.5% draws and 20 games |
|---|---|---|
| `legacy_win_rate` (**default**) | `win_rate ≥ 0.55` | impossible: the ceiling is 0.265 |
| `score_rate` | `score_rate ≥ threshold` | a point estimate, no significance |
| `score_rate_lower_bound` | interval lower bound ≥ threshold | at 0.52 needs an observed score near 0.65 (≈ +107 Elo) |

The default is deliberately unchanged in code: changing deployed behaviour is a
recorded maintainer decision (CLAUDE.md §3.2). What the code does instead is
report **feasibility** at evaluation start — `PromotionConfig.feasibility(n_games,
expected_draw_rate)` logs the minimum observed score the configured policy can
promote and the Elo it implies, and warns above `CIV6AZ_PROMOTION_MAX_IMPLIED_ELO`
(default 50). A gate that cannot be reached should say so, not stay silent.

The runbook (`runbooks/promotion-gate.md`) separates two questions that were
conflated: *is the candidate better than parity* (a superiority test at 0.5 with
`n` from the table) and *is it better by at least X Elo* (a bar that needs far
more games than the pipeline plays).

## 4. Benchmark protocol

`BenchmarkProtocol` (`alpha_zero/evaluation/benchmark.py`) is a schema and a
procedure, executed by the same `run_arena` function the promotion judge uses,
so the benchmark and production cannot disagree:

- board 8 × 8, `max_turns` 60, 25 simulations, Gumbel top-k 16;
- **paired seats**: map `i // 2` is played from both seats, so map variance
  cancels; `n` is even;
- environment seeds from `derive_seed(base, "bench.env", pair)` and a seeded
  search, so two runs at one seed are identical;
- opponents `random`, `heuristic`, `simple_mcts` (random-rollout search) and
  `checkpoint:<sha256>` (a previous model);
- output `result.json` = `{benchmark_schema_version, protocol, evidence, results[]}`
  where `evidence` is the block from [`evidence/SCHEMA.md`](evidence/SCHEMA.md)
  and each result embeds the versioned `EvaluationResult` plus `draw_rate`,
  `decisive_win_rate`, the interval and the MDE at that `n`.

Run it with `python scripts/benchmark_suite.py --help`. The CI smoke runs the
protocol at tiny settings and asserts something that can fail: the greedy
heuristic must beat the random agent with its interval above 0.5 (measured
while building it: 27 W / 4 D / 9 L, score 0.725, interval [0.578, 0.833]). A
benchmark that can only pass is not a benchmark. The random-rollout search was
the first candidate for that check and **lost to random** at board 6 / 20 turns
(score 0.388, interval [0.295, 0.500]): at temperature 0 its near-tied visit
counts break to index 0, which is END_TURN, in 87% of moves. That is recorded
here because it is exactly the kind of fact the protocol exists to surface.

Helpers: `scoring.elo_from_score_rate`, `scoring.elo_interval` and
`scoring.games_for_score_rate` (inverts the MDE for a target score).

## 5. Seeding contract

Every random stream is derived by name from one base seed
(`alpha_zero/utils/seeding.py`, `derive_seed(base, stream, *coords)`): self-play
varies with the training iteration, evaluation uses a fixed held-out stream, and
the benchmark uses its own. Streams cannot collide unless spelled the same. Two
runs at one seed are byte-identical
(`tests/regression/test_training_determinism_regression.py`). The deployed
self-play worker and orchestrator are **not yet seeded** (tracked; the change
touches distributed coordination and is confirm-first).

## 6. Evidence rules

- An artifact may be quoted as evidence of strength only if its `evidence` block
  says `search_epoch: post-2026-09-04` and it reports a score-rate interval.
- Pre-fix artifacts carry a superseded banner and stay in the tree as records of
  mechanism and wall clock.
- **There is no post-fix artifact yet.** The honest statement today is:
  evaluation infrastructure validated; learning not yet demonstrated.

## 7. First post-fix run (pre-registered protocol, milestone v2.1)

Written before the run so the result cannot move the goalposts:

- Local, CPU, seeded; board 8, 60 turns, 10 iterations × 20 games (well under the
  1,000-game charter cap), with `--mcts` naming the search path used.
- **Primary endpoint:** final checkpoint vs a **random-initialised network** under
  the same search and simulation budget, 100 paired games (50 maps × both
  seats), ternary 95% interval. The claim "learns at this budget" is allowed only
  if the lower bound exceeds 0.5. The MDE at n = 100 and d ≈ 0.5 is 0.088
  (≈ +61 Elo): a null result does not rule out smaller gains, and the artifact
  must say so.
- **Secondary:** score rate vs the random agent for both networks.
- **Negative control:** random-init vs a differently seeded random-init must give
  an interval containing 0.5 (checks seat and harness bias).
- Reported whichever way it comes out, with the evidence block and an index row.

## 8. The LLM advisor gate is not an evaluation

`make eval-llm` is an **advisor output-contract gate**: it runs the
`StrategyAdvisor` against a deterministic stub LLM and checks the shape of its
JSON. It does not measure advice quality, in offline or online mode. The advisor
is experimental and default-off (charter NG-2).
