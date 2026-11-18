# Evaluation Protocol

## Purpose

Defines how playing strength is measured and quoted: the score rate with its
ternary interval as the statistic for every claim, the provenance and epoch label
every evidence artifact carries, a benchmark executed through the promotion
judge's own arena code, and the promotion-feasibility report that says when a
configured gate cannot be reached. The vocabulary lives in ``docs/EVALUATION.md``.

## Requirements

### Requirement: Score rate is the statistic for claims

Any statement about agent strength in documentation or evidence SHALL use the
score rate (wins plus half the draws, over games) with its ternary confidence
interval from ``alpha_zero/evaluation/scoring.py``. A draw-blind win rate MAY be
logged for compatibility but MUST be accompanied by ``score_rate`` and
``draw_rate`` wherever it is logged, and new draw-blind computations MUST NOT be
added outside the allowlist in ``tests/regression/draw_aware_allowlist.toml``.

#### Scenario: A new draw-blind win rate is introduced

- **WHEN** a module under ``alpha_zero/`` or ``scripts/`` divides a ``wins`` quantity without pairing it with a score rate
- **THEN** ``pytest tests/regression/test_draw_aware_metrics_regression.py`` fails naming the file and function

#### Scenario: A win rate is logged alone

- **WHEN** a metrics dictionary passed to a logger contains a ``win_rate`` key without ``score_rate`` and ``draw_rate``
- **THEN** the draw-aware regression test fails naming the call site

### Requirement: Evidence artifacts carry provenance and an epoch label

Every directory under ``docs/evidence/`` that reports a measurement SHALL contain
one ``report.md`` and at least one JSON file whose ``evidence`` block records the
git SHA, generation time, seed (or a reason it is null), command, hardware,
library versions, game count and a ``search_epoch`` of ``pre-2026-09-04`` or
``post-2026-09-04``, and the directory MUST be listed in ``docs/evidence/README.md``.

#### Scenario: An artifact without provenance is committed

- **WHEN** an evidence directory lacks a JSON ``evidence`` block with a ``git_sha``
- **THEN** ``pytest tests/regression/test_evidence_contract_regression.py`` fails naming the directory

#### Scenario: A pre-fix artifact is quoted as current

- **WHEN** a report in a ``pre-2026-09-04`` directory lacks the superseded banner
- **THEN** the evidence-contract regression test fails naming the report

### Requirement: Benchmarks run through the promotion judge's code path

The benchmark protocol SHALL be executed by the same arena function the
promotion gate uses, with a seeded search and paired seats, and its output
SHALL embed the versioned evaluation record plus ``draw_rate``,
``decisive_win_rate`` and the minimum detectable effect at the game count used.

#### Scenario: The protocol is reproducible

- **WHEN** the benchmark is run twice with the same seed and protocol
- **THEN** the two result files are identical except for the generation timestamp

#### Scenario: The protocol can detect a known difference

- **WHEN** a random-rollout search with 64 simulations plays a random agent for 40 paired games
- **THEN** the lower bound of the score-rate interval exceeds 0.5

### Requirement: Promotion feasibility is reported

Before an evaluation that feeds a promotion decision, the system SHALL compute
and log the minimum observed score the configured policy can promote at the
configured game count and expected draw rate, and the Elo that score implies,
and it MUST warn when that implied Elo exceeds a configurable ceiling.

#### Scenario: The deployed sample size cannot support the threshold

- **WHEN** the policy is evaluated at 20 games with an expected draw rate of 0.7
- **THEN** the feasibility report states an implied Elo above 90 and a warning is logged

### Requirement: Benchmark results carry per-game records and pair-aware intervals

Each benchmark result row SHALL keep one `{pair, seat, result}` record per game.
At 95% and 99%, it SHALL report the iid ternary score interval, the
pair-clustered interval (`scoring.pair_score_interval`: the plain pentanomial
interval, with the seat-swapped pair as the unit and no pseudo-counts) and their
hull, `[min(lows), max(highs)]`. It SHALL also
report the one-sided 2.5% minimum detectable effect. Every count and interval
MUST be recomputable from the per-game records.

#### Scenario: Correlated pairs widen the hull

- **WHEN** six pairs are won from both seats and four are lost from both seats
- **THEN** the 95% hull equals the pair-clustered interval, [0.2799, 0.9201], which is wider than the iid interval on both sides

#### Scenario: Each side of the hull may come from a different interval

- **WHEN** five pairs split W/L, two are won twice, and one is lost and drawn, at 99%
- **THEN** the hull's low comes from the iid interval and its high from the pair interval

#### Scenario: The symmetry control keeps its power

- **WHEN** 20-pair arms at 99% are simulated with a 0.20 score asymmetry (mu 0.70, d 0.47)
- **THEN** the hull excludes 0.5 in at least 80% of them, and in at most 0.8% of fair (mu 0.5) arms

### Requirement: A benchmark result is never overwritten

The CLI SHALL refuse, before playing any game, to write into an output
directory that already holds `result.json`.

#### Scenario: Two arms share an output directory

- **WHEN** a second run is pointed at an `--out` that already holds `result.json`
- **THEN** it exits non-zero and the existing file is unchanged

### Requirement: Benchmark runs can be sharded and merged exactly

`--shard i/N` SHALL play the `i`-th of `N` contiguous blocks of whole
seat-swapped pairs, with the map and search seeds the unsharded run gives those
games. `--merge` SHALL rebuild the unsharded envelope from shards `1..N`,
recomputing every statistic. A pair MUST NOT be split across shards. Shards that
disagree on protocol, candidate, seed, commit or opponents MUST be refused.

#### Scenario: The union of shards is the unsharded run

- **WHEN** a run on a tiny board is played unsharded and as three shards, and the shards are merged
- **THEN** the merged result rows equal the unsharded rows, game for game

#### Scenario: A split pair is refused

- **WHEN** a shard's games hold one seat of a map without the other
- **THEN** the envelope fails validation and cannot be written or merged

### Requirement: Benchmark evidence follows the evidence contract

The benchmark's `result.json` SHALL carry an `evidence` block that
`tests/regression/test_evidence_contract_regression.py` accepts. The block SHALL
hold `schema_version`, `search_epoch: post-2026-09-04`, `claims`, `non_claims`,
the `shlex`-joined command, the seed and the git SHA. A shard's `non_claims`
SHALL state that it is partial.

#### Scenario: A benchmark result is checked against the contract

- **WHEN** the CLI writes `result.json` and `report.md` into a directory
- **THEN** the contract's predicates find exactly one report and an evidence block with no problems

### Requirement: Baselines can be benchmarked as the candidate

`--candidate heuristic` and `--candidate random` SHALL field the corresponding
baseline as the candidate, with no network. Combining either with
`--checkpoint` SHALL be refused.

#### Scenario: The positive control runs

- **WHEN** the benchmark runs with `--candidate heuristic --opponents=random`
- **THEN** it writes a valid envelope whose candidate is `heuristic`
