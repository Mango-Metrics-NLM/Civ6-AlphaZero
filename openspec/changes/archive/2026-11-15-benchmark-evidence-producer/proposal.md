# Make the benchmark the A1 evidence producer

## Why

The pre-registered A1 learning protocol (plan §5.2, `docs/EVALUATION.md` §7.1)
reaches its verdict from `scripts/benchmark_suite.py`. Before this change the
benchmark could not produce that verdict:

- it reported one iid ternary interval per opponent, and the decision rule reads
  the hull of that interval and a pair-clustered one;
- its `evidence` block failed `docs/evidence/SCHEMA.md` (no `schema_version`, no
  `claims`/`non_claims`, and a `search_epoch` outside the two allowed values);
- it kept no per-game records, so nothing could be recomputed or audited;
- it could not run a baseline as the candidate, which the positive-control arm needs;
- it could not be split across processes (`--shard`), which a long arena needs.

This is LF-D6 of the local-first plan (task T11).

## What Changes

- **Statistics (`evaluation/scoring.py`).**
  - `pair_score_interval(pair_scores, confidence)`: the plain pentanomial
    interval, with the seat-swapped pair as the unit and no pseudo-counts.
    Fix round 1 (review C1) removed per-pair pseudo-counts that halved the
    symmetry control's power.
  - `interval_hull(*intervals)`.
  - `minimum_detectable_effect` accepts `alpha=0.025`.
- **Per-game records (`cloud/gcp/arena.py`, additive).**
  - `ArenaResult.games`: `[{pair, seat, result}]`, defaulting to `[]`.
    `to_dict()` keys are unchanged.
  - `run_arena(..., game_indices=...)` plays a subset of a run. Each game keeps
    the map and seeds the full run gives it.
- **Benchmark (`evaluation/benchmark.py`, new `evaluation/benchmark_evidence.py`).**
  - `benchmark_schema_version` 2: top-level `candidate` and `shard`; each row
    gains `games`, `pairs`, `intervals` (iid, pair and hull at 95% and 99%),
    `mde_alpha025_80pct`, `hull_ratio` and `mde_alpha025_80pct_hull`.
  - The `evidence` block follows `docs/evidence/SCHEMA.md`: `schema_version` 1,
    `search_epoch: post-2026-09-04` (the finer label moves to `search_semantics`),
    `claims`, `non_claims`, a `shlex`-joined command and `read_git_sha`.
  - New flags: `--candidate {checkpoint,heuristic,random}`, `--shard i/N` and
    `--merge SHARD...`. `--opponents=`, `--opponent-checkpoint`, `--gumbel-k`,
    `--max_games` and `--approval_ref` already existed and are unchanged.
  - Pricing: a shard prices its own games and `--merge` prices 0, so the A1
    card prices at 480 (training) + 500 (arena arms) = 980.
  - An existing `<out>/result.json` is never overwritten (review I1).
  - Evidence records `device` and the `search` config, and `--merge` refuses
    shards that disagree on them or on torch/numpy. `command` names the entry
    point that ran.
  - An impossible `--shard` is a parser error.
- `scripts/benchmark_suite.py` puts its own checkout first on `sys.path`, as
  `scripts/run_card.py` does.

## Impact

- **Behaviour:** games are played exactly as before; a run with the same seed
  plays the same games. The envelope's shape changes (schema 2), and
  `validate_envelope` rejects schema-1 files. No schema-1 benchmark artifact is
  committed.
- **API:** additive only. `ArenaResult(...)` positional construction and
  `run_arena(...)` calls without `game_indices` behave as before.
- **Debt baseline:** `coverage_omitted_lines` rises from 66,843 to 66,850. The
  plan assigns `ArenaResult.games` to this task, and `alpha_zero/cloud/*` is
  omitted from the global coverage gate. The new arena lines are measured by
  CI's `cloud-tests` job and covered by `tests/cloud/unit/test_arena_game_records.py`.
- **Affected specs:** `evaluation-protocol`.
- **Tests:** `tests/unit/test_benchmark_producer.py`,
  `tests/unit/test_scoring_pair_interval.py` and
  `tests/cloud/unit/test_arena_game_records.py`.
