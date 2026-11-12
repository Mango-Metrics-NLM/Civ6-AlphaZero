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
- output `result.json` = `{benchmark_schema_version: 2, protocol, candidate, shard,
  evidence, results[]}` where `evidence` is the block from
  [`evidence/SCHEMA.md`](evidence/SCHEMA.md) (`search_epoch: post-2026-09-04`,
  `claims`, `non_claims`, a `shlex`-joined command) and each result embeds the
  versioned `EvaluationResult` plus `draw_rate`, `decisive_win_rate`, the MDE at
  that `n`, the per-game records `{pair, seat, result}`, and an `intervals` block:
  at 95% and 99%, the iid ternary interval, the pair-clustered interval
  (`scoring.pair_score_interval`: the plain pentanomial interval, the
  seat-swapped pair as the unit, no pseudo-counts) and their hull
  `[min(lows), max(highs)]`, beside `mde_alpha025_80pct` and the same widened by
  `hull_ratio` (how far the hull reaches toward 0.5 past the iid bound);
- an existing `<out>/result.json` is never overwritten, so every arm needs its
  own `--out`;
- `--candidate heuristic|random` benchmarks a baseline as the candidate (a
  positive control); `--shard i/N` plays the `i`-th block of whole pairs with
  the unsharded run's seeds, and `--merge` recombines shards and recomputes every
  statistic, so the merged file equals the unsharded one.

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

> **Superseded (2026-11-11) by [section 7.1](#71-a1-registered-protocol-lf-a1).**
> The draft below was never run. Section 7.1 replaces it with the A1 protocol
> of `kb/specs/20261111-local-first-and-branching-plan.md` §5.2: 20 iterations,
> a pinned `checkpoint_0` as the comparator, a symmetry control, a positive
> control and a hull-based verdict. Nothing below is in force; it is kept as the
> record of what was first proposed.

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

<!-- A1-PROTOCOL-BEGIN: the registered text of section 7.1 is every line between this marker and the end marker; see "Hashing section 7.1" after the end marker. -->
### 7.1 A1: registered protocol (LF-A1)

Registered 2026-11-11 (task T13), before any A1-family run, exploratory runs
included. Source: `kb/specs/20261111-local-first-and-branching-plan.md` §5.2.
The A1 artifact records the sha256 of this text, and a run is judged only
under the registration whose hash its artifact records. Every registered
digest is logged after this section ("Registration log").

**Amendments.** An **A1-family run** is any execution of a command
registered below (the training command or an arena arm) at a registered seed
(1001, 1002, 1003, 20261111 or 20261112); an exploratory run is an A1-family
run at 1001–1003. T0b's wall-clock calibration and its positive-control
reference runs are **not** A1-family runs, although they use seeds 1001–1003,
so they lock nothing. Before any A1-family run starts, this section may be
re-registered, with a new log row. **Once any exploratory run has started**,
the only amendments allowed are the slots marked "pending T0b": the measured
positive-control reference with its P values and P(40) interval, the 40/60
branch in force, and the cards' `sha`. Any other change (an arm, a threshold, a verdict row, the
search settings, the card form) is a **new registration with a new
confirmatory seed pair**; 20261111 and 20261112 are then retired unrun.

**Code binding.** The card's pinned `sha` must contain this section with the
digest the A1 artifact records; T14 checks `protocol_sha256` of
`docs/EVALUATION.md` at `run.json`'s `sha`. The settings below are the ones
that commit's code resolves.

**Question.** Does the local training loop, run on the fixed search at this
budget, produce a network that beats its own random initialisation under the
same search? A1 answers it once at seed 20261111. A1-R repeats it at seed
20261112, and **A1-R must also PASS before any public "learns" claim.**

#### Seeds

- **Exploratory:** 1001, 1002, 1003. Exploratory runs use the registered
  commands with every `--seed 20261111` replaced. They are never quoted as the
  verdict.
- **Confirmatory:** **20261111** (A1) and **20261112** (A1-R). Neither has been
  run before this registration.
- **Disjointness** (LF-A17, `kb/specs/20261111-seed-overlap-memo.md`). Every
  map, search and sampling stream is `derive_seed(base, <stream name>,
  *coords)`; network initialisation is `torch.manual_seed(base)`. At A1's sizes, the 480
  self-play maps and the 100 benchmark maps share no seed and no map, and this
  holds for each of the five seeds. A1's and A1-R's streams share nothing, and
  the exploratory and confirmatory derived seeds are all distinct.
  `tests/regression/test_a1_seed_disjointness_regression.py` gates this, and
  its negative control is the arithmetic `seed + pair` rule, which overlaps on
  99 of 100 maps. To reproduce for one seed (seconds). The block is Python,
  fenced as `text` so that no formatter rewrites registered text:

  ```text
  from alpha_zero.envs.microciv import MicroCivEnv
  from alpha_zero.utils.seeding import SeedStream, derive_seed
  import hashlib
  def fp(s):
      obs, _ = MicroCivEnv(board_size=8, max_turns=60).reset(seed=s)
      return hashlib.sha256(obs["spatial"].tobytes() + obs["vector"].tobytes()).hexdigest()
  S = 20261111
  train = {fp(derive_seed(S, SeedStream.SELF_PLAY_ENV, it, g)) for it in range(20) for g in range(24)}
  bench = {fp(derive_seed(S, SeedStream.BENCH_ENV, p)) for p in range(100)}
  assert not train & bench  # holds for 1001, 1002, 1003, 20261111, 20261112
  ```

- **The arms of one card share maps.** Each arena arm runs at the same
  `--seed`, so pair `p` is the same map in every arm. Common random numbers make
  checkpoints comparable map for map, but **the arms are not independent
  samples**: a map-specific pathology shows in several arms at once.
- T0b measures the positive-control reference at seeds 1001–1003, which are on
  `bench.env` maps disjoint from the confirmatory arm's. The reference is
  therefore out of sample for A1.

#### Search settings (LF-A5: Option A, current settings registered)

No maintainer decision is recorded in
`kb/specs/20261111-completed-q-alignment-memo.md` §7. The plan's rule is to
register the current settings in that case. These are the values
`training_civ6` resolves for A1's flags (`gumbel_search_config`), for
self-play:

- `num_simulations` (`--gumbel_sims`): 24
- `num_sampled_actions` (`--gumbel_k`): 8
- `c_visit`: 50.0
- `c_scale`: 1.0
- `policy_target`: `completed_q`
- `greedy_threshold`: 0.99
- `use_sequential_halving`: True
- `temperature`: 1.0
- `temperature_threshold`: 30
- Q normalisation before σ: none. The target applies σ to raw Q in [−1, 1],
  i.e. `softmax(logits + (c_visit + max N) · c_scale · completedQ)` over the
  visited actions.
- σ in the sequential-halving score: **not applied**. Survivors are ranked by
  `g + log π + completedQ`, with coefficient 1 on Q.

These settings diverge from Danihelka et al. (2022) **deliberately**, in items
3–8 of the memo's §3: the Q normalisation, the completion, the root value in
the target, the target's support, the halving score, and interior selection.
The memo is the reference.

- `MCTS_C_VISIT`, `MCTS_GREEDY_THRESHOLD` and every other `MCTS_*` variable
  override these defaults from the environment (`core/mcts_constants.py`).
  `run.json` records environment variable **names** only, so they must be
  **unset**. A run whose `run.json` `env_var_names` contains any `MCTS_` name is
  VOID.
- **If the maintainer records a different A5 decision (B, C or D), A1 must be
  re-registered before any A1-family run.** That means a new section 7.1 and a
  new hash.

**Arena search.** Every searching side of every arena arm is `GumbelMCTS`,
built by `create_mcts` inside `run_arena` (`alpha_zero/cloud/gcp/arena.py`):

- `num_simulations` 24 and `num_sampled_actions` 8, from `--sims` and
  `--gumbel-k`;
- a seed per game and side, `derive_seed(seed, "arena.search", game, side)`;
- no root noise; every other `GumbelMCTSConfig` field at its default
  (`c_visit` 50.0, `c_scale` 1.0, `policy_target` `completed_q`, sequential
  halving on);
- the move played is `select_action(target, temperature=0)`, the argmax of the
  target;
- board 8, `max_turns` 60, gameplay preset `default`.

Each `result.json` records `protocol` and `evidence.search` (`mcts_type`,
`num_simulations`, `gumbel_k`, `search_semantics`); T14 compares them with this
paragraph.

#### Registered commands

`<D>` is `E:\civ6az-data\runs\<card>`, where `<card>` is the run card's
`card_id`. A card therefore runs with `--data_root E:\civ6az-data` (or
`CIV6AZ_DATA_ROOT` set to it); under any other root, `run.json` and the run's
outputs land in different trees, and the run is VOID for want of a
`COMPLETED` `run.json` at `<D>`. The commands run in this order. Each is executed through
`scripts/run_card.py` from the card's pinned SHA (T1b); the cards are
`configs/run_cards/a1.yaml` (A1) and `configs/run_cards/a1r.yaml` (A1-R).
Every arena arm has its own `--out`, because `result.json` is never overwritten
(T11 review I1).

The training command is copied verbatim from the plan. Each arena command is the
plan's common prefix, then the arm's flags, then `--out`. Two places expand the
plan's table:

- the plan's `…<N>.pt` abbreviation is written out as `<D>\ckpt\checkpoint_<N>.pt`;
- the "final vs random agent" row names its candidate, `--checkpoint
  <D>\ckpt\checkpoint_20.pt`. Without it, the benchmark would score an untrained
  network.

<!-- a1-commands:base -->
```text
# train
python -m alpha_zero.training_civ6 --seed 20261111 --board_size 8 --max_turns 60 --num_iterations 20 --num_games 24 --mcts gumbel --gumbel_k 8 --gumbel_sims 24 --device cpu --bootstrap none --save_interval 5 --eval_games 0 --checkpoint_dir <D>\ckpt
# primary
python scripts/benchmark_suite.py --opponents= --seed 20261111 --board-size 8 --max-turns 60 --sims 24 --gumbel-k 8 --mcts-type gumbel --device cpu --checkpoint <D>\ckpt\checkpoint_20.pt --opponent-checkpoint <D>\ckpt\checkpoint_0.pt --n-games 200 --out <D>\arena\primary
# symmetry
python scripts/benchmark_suite.py --opponents= --seed 20261111 --board-size 8 --max-turns 60 --sims 24 --gumbel-k 8 --mcts-type gumbel --device cpu --checkpoint <D>\ckpt\checkpoint_0.pt --opponent-checkpoint <D>\ckpt\checkpoint_0.pt --n-games 40 --out <D>\arena\symmetry
# positive-control
python scripts/benchmark_suite.py --opponents= --seed 20261111 --board-size 8 --max-turns 60 --sims 24 --gumbel-k 8 --mcts-type gumbel --device cpu --candidate heuristic --opponents=random --n-games 40 --out <D>\arena\positive_control
# secondary-5
python scripts/benchmark_suite.py --opponents= --seed 20261111 --board-size 8 --max-turns 60 --sims 24 --gumbel-k 8 --mcts-type gumbel --device cpu --checkpoint <D>\ckpt\checkpoint_5.pt --opponent-checkpoint <D>\ckpt\checkpoint_0.pt --n-games 60 --out <D>\arena\secondary_5
# secondary-10
python scripts/benchmark_suite.py --opponents= --seed 20261111 --board-size 8 --max-turns 60 --sims 24 --gumbel-k 8 --mcts-type gumbel --device cpu --checkpoint <D>\ckpt\checkpoint_10.pt --opponent-checkpoint <D>\ckpt\checkpoint_0.pt --n-games 60 --out <D>\arena\secondary_10
# secondary-15
python scripts/benchmark_suite.py --opponents= --seed 20261111 --board-size 8 --max-turns 60 --sims 24 --gumbel-k 8 --mcts-type gumbel --device cpu --checkpoint <D>\ckpt\checkpoint_15.pt --opponent-checkpoint <D>\ckpt\checkpoint_0.pt --n-games 60 --out <D>\arena\secondary_15
# final-vs-random
python scripts/benchmark_suite.py --opponents= --seed 20261111 --board-size 8 --max-turns 60 --sims 24 --gumbel-k 8 --mcts-type gumbel --device cpu --checkpoint <D>\ckpt\checkpoint_20.pt --opponents=random --n-games 40 --out <D>\arena\final_vs_random
```

| Invocation | Games |
|---|---:|
| train: 20 iterations × 24 self-play games, no in-loop evaluation | 480 |
| primary: `checkpoint_20` vs `checkpoint_0` | 200 |
| symmetry: `checkpoint_0` vs `checkpoint_0` | 40 |
| positive control: heuristic vs random agent | 40 |
| secondary: `checkpoint_5`, `_10`, `_15` vs `checkpoint_0`, 60 each | 180 |
| final vs random agent: `checkpoint_20` vs random | 40 |
| **card total** (arms 500) | **980** |

The total is priced across every invocation, each by its runner's own parser
(`run_card --price_only`). No invocation exceeds 1,000 games.
`tests/scripts/test_a1_registration.py` holds these commands, the cards and the
price together. Under `run_card`, a benchmark child runs as
`python -m alpha_zero.evaluation.benchmark` with the same arguments (the same
CLI as `scripts/benchmark_suite.py`), and its `result.json` records that entry
point; compare the arguments, not the leading tokens.

- **The 60-game branch** (see the positive control below) replaces two
  invocations. The total stays 980:

<!-- a1-commands:branch-60 -->
```text
# positive-control
python scripts/benchmark_suite.py --opponents= --seed 20261111 --board-size 8 --max-turns 60 --sims 24 --gumbel-k 8 --mcts-type gumbel --device cpu --candidate heuristic --opponents=random --n-games 60 --out <D>\arena\positive_control
# final-vs-random
python scripts/benchmark_suite.py --opponents= --seed 20261111 --board-size 8 --max-turns 60 --sims 24 --gumbel-k 8 --mcts-type gumbel --device cpu --checkpoint <D>\ckpt\checkpoint_20.pt --opponents=random --n-games 20 --out <D>\arena\final_vs_random
```

- **Branch in force: pending T0b.** T0b replaces "pending T0b" with "40" or
  "60" and, for "60", puts the two branch commands into both cards.
- **A1-R** is the same card with every `--seed 20261111` replaced by
  `--seed 20261112`, under its own `<card>`.
- **Only the one-card form is registered.** If T0b's wall-clock ETA for the
  card is over 12 h, A1 may instead run as two cards (training, then a sharded
  arena), but that form is **not registered here**. Before it is used it needs a
  re-registration of its exact commands, both cards' `<D>` directories, the
  `--shard i/N` and `--merge` commands with their `--out` directories, and the
  rule that every card's `run.json` must say `COMPLETED`. That is a change of
  card form, so it must be registered **before any exploratory run starts**
  (see Amendments). Wall-clock is not estimated in advance; T0b measures it.

#### Intervals

Every arm is read from its `result.json` row, which `benchmark_suite.py` writes
(`alpha_zero/evaluation/benchmark.py`, `benchmark_evidence.py`). For each arm:

- (a) the iid ternary interval, `scoring.score_interval(w, d, l)`;
- (b) the plain pentanomial interval, `scoring.pair_score_interval`, with the
  seat-swapped pair as the unit and no pseudo-counts.

The decision bounds are their **hull**: L = min of the two lows, U = max of the
two highs (`intervals[<level>]["hull"]`).

- The primary arm and the positive control are read at **95%**.
- The symmetry control is read at **99%**. On a fair arena it VOIDs with
  probability ≤ 0.8% (measured 0.42–0.48%). It detects only a gross asymmetry,
  about 0.20 in score (detection about 83% at 20 pairs).
- `final vs random` and the secondary arms are reported, not judged.

#### Verdict

The first matching row wins.

| Order | Verdict | Condition |
|---|---|---|
| 1 | VOID | the run was interrupted or resumed (below); OR the symmetry arm's 99% hull excludes 0.5 (L > 0.5 or U < 0.5); OR the positive control's 95% L ≤ 0.5 |
| 2 | PASS | the primary arm's 95% L > 0.5 |
| 3 | FAIL | the primary arm's 95% U < 0.5 |
| 4 | INCONCLUSIVE | otherwise |

An INCONCLUSIVE verdict reports two numbers:

- **MDE** = `minimum_detectable_effect(200, d, alpha=0.025)`, where `d` is the
  primary arm's observed draw rate. That is about 0.072 at d = 0.47, i.e. about
  +50 Elo.
- **The MDE widened by `hull_ratio`.** `hull_ratio` is
  `benchmark_evidence.facing_hull_ratio(score, iid95, hull95)`: "How much further
  the hull reaches toward 0.5 than the iid interval does."
  - It is `(score − hull_low) / (score − iid_low)` when score > 0.5,
    `(hull_high − score) / (iid_high − score)` when score < 0.5, and the larger
    of the two at exactly 0.5.
  - It is always ≥ 1, and 1 for a degenerate denominator. A hull that is wider
    only on the side away from 0.5 does not widen the MDE.
  - The row records `mde_alpha025_80pct`, `hull_ratio` and their product
    `mde_alpha025_80pct_hull`.

#### Interruptions

**An interrupted or resumed confirmatory run is VOID.** `scripts/run_card.py`
(`alpha_zero/local/run_card.py`) writes `<D>\run.json` with status
`RUNNING`, then:

- an interruption (Ctrl+C, Ctrl+Break, SIGTERM) sets `VOID`;
- a second attempt that finds a `RUNNING` record sets it to `VOID` ("a card
  cannot resume");
- a lingering `RUNNING` record (a hard kill) is read as `VOID`.

The verdict is VOID unless `run.json` says `COMPLETED`. The same applies if any
`iterations.jsonl` line carries a non-null `resumed_from` or
`"supersedes_earlier_record": true`, or if `run.json` names an `MCTS_`
variable.

**After a VOID.** A VOID records no learning result, so a release cannot rest
on it: T14's verifier exits 1 when a role has no non-VOID attempt.

1. Fix the cause. An interruption needs nothing fixed. A symmetry or
   positive-control VOID needs the arena or harness fixed; that is a play-path
   change, so it lands as a new rc.
2. Run the same card again under a **new `card_id`** (for example
   `a1-20261111-r2`, so `<D>` is new) at the **same seed**. This is the one
   exception to "neither has been run before".
   - After an **interruption**, nothing changed: confirmatory runs are
     deterministic on CPU, so the re-run reproduces the same games and cannot be
     used to fish for another result.
   - After a **symmetry or positive-control VOID**, the code changed, so the
     re-run plays different games: it is a **second draw**, made after the VOID
     attempt's arms (its primary arm included, if it ran) were seen. Listing the
     VOID attempt in the bundle is what discloses that.
3. Keep the VOID attempt in the release bundle beside the re-run, with whatever
   components it left (an invocation or arm that never ran is omitted). Each
   role has exactly one non-VOID attempt, and that attempt is the one judged.

#### Positive control

- **Rule** (T11 review I2): the calibration is computed on the hull, as
  **P(hull L ≤ 0.5)**, not P(`ci_low` ≤ 0.5).
- **How it is computed: pair bootstrap of T0b's measured pairs** (T13 review
  I1). `pairs` is the concatenation, in seed order 1001, 1002, 1003, of
  `a1_protocol.pairs_from_games(row["games"])` for the heuristic-vs-random row
  of each of T0b's three `result.json` files (about 300 seat-swapped pairs).
  P(n) is `alpha_zero.evaluation.a1_protocol.positive_control_void_probability(pairs,
  n)` with its defaults (20,000 replicates, seed 1111, 95%): each replicate
  draws n / 2 whole pairs with replacement and reads L from the benchmark's own
  `interval_summary`. Resampling pairs keeps the map clustering the hull exists
  for.
- **Its uncertainty** (T13 rereview N3): one reference of about 300 pairs makes
  the point estimate noisy, so T0b also records a 95% percentile interval for
  P(40), `a1_protocol.positive_control_void_probability_interval(pairs, 40)`
  with its defaults (200 outer replicates, 2,000 inner, seed 1111, level 0.95,
  confidence 95%). Each outer replicate resamples `len(pairs)` pairs with
  replacement and runs the pair bootstrap on that resampled reference.
- `iid_positive_control_void_probability(p_win, p_draw, 40)` at T0b's pooled
  proportions is reported beside it as an **informational lower bound**: it
  draws games independently, so it cannot see clustering. The branch rule never
  reads it.
- **Reference:** T0b measures it on post-B1 code at board 8 / 60 turns,
  heuristic vs random, seeds 1001–1003 × 200 games. This replaces the board-6
  figure of 0.725.
  - **Measured reference: pending T0b.** T0b records here, before any
    exploratory or confirmatory run: W/D/L, the score and its hull, the number
    of pooled pairs, P(40) with its 95% interval and P(60) by pair bootstrap,
    and the iid P(40).
  - Pre-B1 reference, superseded by T0b: 0.7375, CI [0.711, 0.761].
- **Branch:** if the **upper bound of P(40)'s 95% interval** exceeds 0.05,
  the positive control takes 60 games and "final vs random" drops to 20,
  keeping the card at 980. (Erring to 60 costs only 20 reported, unjudged
  "final vs random" games; erring to 40 risks a VOID 980-game run.) If P(60) also
  exceeds 0.05, the branch is still 60 (the most the 980-game card allows), and
  T0b records P(60) as the VOID risk the run accepts.
- **Either branch is plausible.** At μ = 0.7375 the iid lower bound crosses
  5% near d ≈ 0.23 (5.9% at d = 0.20, 4.6% at d = 0.24), and clustering raises
  it. The pre-B1 reference cannot settle it, because its game count is not
  recorded. Figures (informational, μ = 0.7375):

  | Model | P(40) | P(60) |
  |---|---|---|
  | iid, d ≈ 0 | about 14% (T11 review) | about 3% |
  | iid, d = 0.3 | about 3% | about 0.3% |
  | per-map clustering (κ = 2), d = 0.3 | about 7% (T11 and T13 reviews) | about 1.4% |

#### Trend (LF-A2)

`scripts/check_learning_trend.py` is **descriptive only**. It reads the run's
`iterations.jsonl` and `eval_*.json` and the arms' `result.json`. It reports
score-rate series against iteration, with least-squares slopes, beside the
per-iteration diagnostics. It decides nothing: the verdict is the table above.

- Exit 0: a trend was computed (a score-rate series with ≥ 3 iterations).
- Exit 1: malformed input.
- Exit 2: fewer than 3 points.

#### Target-sharpness trigger for the Option E ablation (LF-A5)

Registered from the memo's §6, with the T12 review's corrections. All
quantities are the A13 fields in `iterations.jsonl`, computed over
**non-forced positions only**.

- **`sᵢ`** is `prequential.target_share_above_greedy_threshold` at loop index
  `i` (the record with `"iteration": i + 1`), and **`mᵢ`** is
  `prequential.target_max_p_mean`. A record that carries an `error`, or whose
  value is `null` (no non-forced positions), contributes nothing.
- **`s₀`** is `s` from A1's loop-index-0 record, written as
  `"iteration": 1`. It is used **only if** that record has
  `generator == "network"` (A1 runs `--bootstrap none`) and carries no `error`.
  **Otherwise, or if that share is `null`, `s₀ := 0`**, and the 0.25 floor
  applies.

After A1 finishes, whatever its verdict, check:

- **T1:** `sᵢ > max(0.25, 5 · s₀)` at any loop index `i ≥ 1` (any record with
  `"iteration"` ≥ 2); or
- **T2:** `mᵢ > 0.80` in any record, the first included.

For reference, the non-forced starting values in the review's A1-style
self-play were about 0–1.5% for the share (0.4% measured through the code) and
**0.32** for the mean max-p. (The memo's "about 0.36" was a round-1 estimate;
T12 review M2.) So the 0.25 floor binds, and `5 · s₀` matters only if `s₀`
exceeds 5%.

If either fires, the Option E ablation is the next experiment. Run B (σ in
halving) first, because it is the smaller change; run C only if the maintainer
has chosen its normalisation. The ablation is pre-registered on its own
(≤1,000 games), under the same seed protocol, with a flag-gated variant. If
neither fires, the ablation is not motivated by target sharpness, and a FAIL or
INCONCLUSIVE A1 is diagnosed through A1-P first. The trigger is descriptive: it
changes nothing during A1 and stops nothing.
`scripts/check_learning_trend.py` reports it.

#### Follow-ups

- **A1-R** (seed 20261112) must PASS before any public "learns" claim.
- **On FAIL**, run A1-P: the same card at the same seed with the PUCT search
  (`--mcts civ6` in training, `--mcts-type civ6` in the arena). It is
  registered on its own before it runs. It tells a search fault from a loop
  fault.
- The result is reported whichever way it comes out, with the evidence block
  (`search_epoch: post-2026-09-04`) and an index row.
<!-- A1-PROTOCOL-END -->

**Hashing section 7.1.** The registered text is every line strictly between the
`A1-PROTOCOL-BEGIN` and `A1-PROTOCOL-END` marker lines. CRLF is normalised to LF,
each line ends in LF, and the text is encoded as UTF-8. Its sha256 is what the
A1 artifact records. Compute it with:

- `python scripts/check_learning_trend.py --print-protocol-sha`, or
- `alpha_zero.evaluation.a1_protocol.protocol_sha256(text)`.

The hash is deliberately not written inside the region.

**Registration log.** One row per registered version of section 7.1, newest
last. `tests/regression/test_a1_registration_digest_regression.py` fails unless
the last row's digest is the digest of the text as it stands, so an edit to the
region, deliberate or a formatter's, needs a new row here in the same change.
When T0b fills in the "pending T0b" slots it adds a row ("T0b slot fill-in:
measured reference, branch N"), and the A1 artifact records that row's digest.
The cards' `sha` is the rc, which already carries this section, so it is recorded
in `run.json` and the release bundle, not here.

| Date | sha256 of section 7.1 | Change |
|---|---|---|
| 2026-11-11 | `990989eea43b90e777a67a40da5bc319f62c2e49c82174b74530d4d1d6f2e377` | Initial registration (T13), with fix round 1: pair-bootstrap calibration, amendment limits, code binding, arena search, one-card form only, `text` fence |
| 2026-11-12 | `8aad6a84171aa157aa606a74c225b4e3743f7f91842109adbcde9feebf9a753e` | Fix round 2 (before any A1-family run): "A1-family run" defined (T0b's calibration runs excluded); the 40/60 branch reads the upper bound of a bootstrap 95% interval for P(40) |
| 2026-11-12 | `9853c719b3549822cfa3f6215615ff0b12d637a1194847141b359417ba1e002e` | T14 review I-1 (before any A1-family run): "After a VOID" -- a VOID records no learning result; fix the cause and re-run under a new `card_id` at the same seed; the VOID attempt stays in the bundle and exactly one non-VOID attempt per role is judged |
| 2026-11-12 | `79a4ba546d4b21c6cbfcab570c2327bb9901eaacc2111300becc75777e238a4b` | T14 rereview N-1/N-2 (before any A1-family run): "After a VOID" says the same-seed re-run is a reproduction only after an interruption; after a symmetry or positive-control VOID it is a second draw, disclosed by the listed VOID attempt, which keeps whatever components it left |

## 8. The LLM advisor gate is not an evaluation

`make eval-llm` is an **advisor output-contract gate**: it runs the
`StrategyAdvisor` against a deterministic stub LLM and checks the shape of its
JSON. It does not measure advice quality, in offline or online mode. The advisor
is experimental and default-off (charter NG-2).
