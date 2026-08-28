# Tasks — fix-selfplay-and-promotion-correctness

## 1. Self-play trajectory integrity

- [x] 1.1 (M) Add `new_trajectory` / `append_step` / `dummy_trajectory` to
      `cloud/gcp/selfplay_worker.py`; adopt in the sequential, MuZero and batch
      paths and in both dry-run stubs, so the per-step shape has one definition
- [x] 1.2 (S) Batch path: buffer pre-search observations and commit every
      per-step field atomically after a successful search, so a failed search
      cannot leave a ragged trajectory
- [x] 1.3 (S) Take dry-run dimensions from `NetworkArchitectureConfig` instead of
      the literals `13`/`64`/`100` (the sequential stub emitted a 100-wide policy
      against a 376-wide action space)
- [x] 1.4 (S) `cloud/gcp/training_worker.py`: guard the mask fallback on length
      rather than key presence, so an empty `action_masks` list falls back
      instead of raising `IndexError`
- [x] 1.5 (M) Tests: de-vacuum the batch/sequential parity test (assert
      non-emptiness first, compare full key sets); assert mask density is
      strictly between 0 and 1; assert alignment under an induced search
      failure; assert masks survive the pickle round trip into `prepare_dataset`;
      cover mixed legacy/current trajectories and the empty-mask case
- [x] 1.6 (S) Replace the source-text grep in
      `tests/cloud/unit/test_selfplay_worker.py` with behavioral assertions

## 2. Honest self-play parallelism

- [x] 2.1 (S) Add `BATCH_MCTS_TYPE` to `cloud/gcp/config.py`; use it in both the
      orchestrator and the worker's dispatch gate
- [x] 2.2 (S) Default `--num_parallel_games` to 1 and correct the help text
- [x] 2.3 (S) Add `warn_on_ignored_parallelism` to the orchestrator, called at
      the normalize-once point before the tracker config is built; add the
      matching defensive warning in the worker
- [x] 2.4 (S) Extract `should_use_batch` from `main()` so the dispatch decision
      is testable rather than restated in tests
- [x] 2.5 (S) Document `num_parallel_games` / `max_batch_size` in
      `configs/cloud_training_v1.yaml`
- [x] 2.6 (M) Rewrite `TestBatchModeDetection` and `TestOrchestratorArgs` to call
      production functions; assert the warning does not mutate `mcts_type`

## 3. Promotion decision consolidation

- [x] 3.1 (S) Correct `evaluation/promotion.py`'s docstring, which listed
      `cloud/evaluator.py` as a peer of the deployed path; add LEGACY banners to
      `cloud/evaluator.py` and `cloud/selfplay_worker.py` naming the live path
- [x] 3.2 (M) Route `cloud/evaluator.py` through the registry via one
      `_promotion_verdict` seam shared by the early-stop check and the final
      gate; pass the evaluator's own threshold explicitly so it is not orphaned
- [x] 3.3 (S) Persist `score_rate` / `ci_low` / `ci_high` on `EvaluationResult`
      and expose the tri-state decision in the run result
- [x] 3.4 (S) Validate `confidence` against `scoring.SUPPORTED_CONFIDENCE_LEVELS`
      so an unsupported level fails at configuration time, not mid-evaluation
- [x] 3.5 (M) Tests: default-policy byte-compatibility, a policy flip that
      asserts an actual promotion, threshold precedence, early-stop consistency

## 4. Documentation sync

- [ ] 4.1 (S) `ROADMAP.md`: record that the champion-load fix landed via
      `--resume_from=latest`; fix the broken analysis-doc link
- [ ] 4.2 (S) Archive `docs/NEXT_STEPS_ANALYSIS.md` and update the referring link
      in `docs/README.md` in the same commit (the link gate resolves it)
- [ ] 4.3 (S) `CLAUDE.md`: refresh the stale skip-count with its counting rule;
      note the fog/N-player dependency; date-correct the value-sign entries
- [ ] 4.4 (S) Finish the peer-review edits to `docs/NEXT_STEPS_CASE_2026-08.md`,
      including correcting its own claim about the second promotion site
