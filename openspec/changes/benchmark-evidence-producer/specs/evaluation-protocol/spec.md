# evaluation-protocol -- deltas

## ADDED Requirements

### Requirement: Benchmark results carry per-game records and pair-aware intervals

Each benchmark result row SHALL keep one `{pair, seat, result}` record per game.
At 95% and 99%, it SHALL report the iid ternary score interval, the
pair-clustered interval (`scoring.pair_score_interval`, with the seat-swapped
pair as the unit) and their hull, `[min(lows), max(highs)]`. It SHALL also
report the one-sided 2.5% minimum detectable effect. Every count and interval
MUST be recomputable from the per-game records.

#### Scenario: Correlated pairs widen the hull

- **WHEN** six pairs are won from both seats and four are lost from both seats
- **THEN** the 95% hull equals the pair-clustered interval, [0.3116, 0.8329], which is wider than the iid interval on both sides

#### Scenario: Each side of the hull may come from a different interval

- **WHEN** five pairs split W/L, two are won twice, and one is lost and drawn
- **THEN** the hull's low comes from the pair interval and its high from the iid interval

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
