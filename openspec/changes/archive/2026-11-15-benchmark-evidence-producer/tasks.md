# Tasks

## 1. Test first
- [x] 1.1 Write `tests/unit/test_scoring_pair_interval.py` and record RED. It covers:
  - the pair interval on two fixtures, by hand;
  - the hull;
  - the MDE at alpha 0.025.
- [x] 1.2 Write `tests/cloud/unit/test_arena_game_records.py` and record RED. It covers:
  - per-game records;
  - a subset replaying a run;
  - unchanged serialisation.
- [x] 1.3 Write `tests/unit/test_benchmark_producer.py` and record RED. It covers:
  - the evidence contract;
  - `--candidate heuristic`;
  - the shard union equalling an unsharded run;
  - merge refusals;
  - the A1 card pricing at 980;
  - the horizon check on both checkpoint flags.

## 2. Implement
- [x] 2.1 `pair_score_interval`, `interval_hull` and `alpha=0.025` in `scoring.py`
- [x] 2.2 `ArenaResult.games` and `run_arena(game_indices=...)`
- [x] 2.3 Benchmark schema 2: evidence block, intervals, `--candidate`, `--shard`, `--merge`, pricing
- [x] 2.4 Negative controls, each reverted:
  - splitting a pair across shards fails the union test;
  - taking the hull from the iid interval alone fails the hull test;
  - an arena that ignores global game indices fails the subset and union tests.

## 3. Documentation
- [x] 3.1 CLAUDE.md §5.1 rows and §5.2 decision row
- [x] 3.2 `docs/EVALUATION.md` §4: the envelope, shards and intervals
- [x] 3.3 `docs/MODULE_REFERENCE.md`: evaluation file count

## 4. Review fix round 1
- [x] 4.1 C1: plain pentanomial `pair_score_interval` (no pseudo-counts; `P < 2` gives (0, 1); zero variance gives zero width); reviewer fixtures; seeded OC tests; negative control (the pseudo-count estimator fails detection and positive control)
- [x] 4.2 I1: refuse to overwrite `<out>/result.json`
- [x] 4.3 Minors: `--shard` parser error; `device`/`search` in evidence, merge agreement on device/torch/numpy/search; real entry point in `command`; `hull_ratio` on the side facing 0.5
