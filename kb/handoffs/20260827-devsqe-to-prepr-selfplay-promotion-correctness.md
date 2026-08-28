# DEV-SQE → Pre-PR Handoff

**Date:** 2026-08-27
**Spec:** `openspec/changes/fix-selfplay-and-promotion-correctness/proposal.md`
**Branch / PR:** `claude/next-steps-codebase-review-ir7b0e` → PR #113 (draft)

## What changed and why

Three silent defects in the cloud self-play and promotion paths. Each let the
pipeline report success while producing degraded data or an untrustworthy
decision.

1. **Batch self-play trained with masking disabled.** `run_batch_selfplay` never
   stored `action_masks`; `prepare_dataset` substitutes an all-ones mask when the
   key is absent, so every batch-generated sample trained the network to place
   probability mass on illegal actions. Root cause was duplication — three
   self-play paths each built the trajectory inline, and the batch path split its
   appends across two loops and dropped the mask. Now built by shared
   `new_trajectory`/`append_step`/`dummy_trajectory` helpers.
2. **The batch loop could emit ragged trajectories.** Observations were appended
   before the search that may raise; on failure the loop breaks, leaving
   `spatial` longer than `policies`/`players`, which the trainer indexes past.
   Observations are now buffered and committed atomically.
3. **Requested parallelism was silently ignored.** `--num_parallel_games`
   defaulted to 4 against a default `--mcts_type=gumbel`, which fails the batch
   dispatch gate. Default is now 1 (what those runs actually did) and both the
   orchestrator and worker warn on an unusable combination.

Plus: `cloud/evaluator.py` routed through the promotion-policy registry with
draw-aware shadow statistics, and `PromotionConfig` now validates `confidence`
against the z-table `score_interval` actually supports.

## Files changed

`alpha_zero/cloud/gcp/{selfplay_worker,training_worker,vertex_orchestrator,config}.py`,
`alpha_zero/cloud/{evaluator,selfplay_worker}.py`,
`alpha_zero/evaluation/{promotion,scoring}.py`, `configs/cloud_training_v1.yaml`,
`openspec/changes/fix-selfplay-and-promotion-correctness/` (new),
`CLAUDE.md`, `ROADMAP.md`, `PROJECT_CHARTER.md`, `docs/README.md`,
`docs/NEXT_STEPS_CASE_2026-08.md`, `docs/archive/NEXT_STEPS_ANALYSIS.md` (moved).

## Tests added

`tests/cloud/unit/test_evaluator_promotion.py` (new, 14 tests);
`TestBatchActionMasks` (mask informativeness + ragged-trajectory guard) in
`tests/integration/test_batch_selfplay_integration.py`;
`TestBatchTrajectoryReachesTrainer` (GCS pickle round-trip, mixed legacy/current,
empty-mask fallback) in `tests/cloud/unit/test_training_worker.py`; behavioral
replacements for a source-text grep in `tests/cloud/unit/test_selfplay_worker.py`;
rewrites of `TestBatchModeDetection` / `TestOrchestratorArgs`.

## Test results

- `pytest tests/unit/ -m "not slow"` — **10,548 passed**, 58 skipped
- `pytest tests/integration/ -m "not slow and not requires_env"` — **601 passed**, 12 skipped, 1 xfailed
- `pytest tests/sanity/ tests/cloud/unit/ tests/security/` — **1,035 passed**, 4 skipped
- Governance — **50 passed**
- **Lint:** clean (`ruff check .`; 920 files already formatted)
- **Type check:** clean (`mypy alpha_zero/`, 359 source files)
- **Governance gates:** `make validate` green (openspec, kb, skills, doc-sync)

**Coverage: not measurable for this change, and deliberately not claimed.** All
of `alpha_zero/cloud/` is in the coverage omit list, so these fixes cannot move
the number. That omit-list gap is finding F6 of the case document; it is not
something this change closes. CI remains the gate.

## Smoke tests

- `python -m alpha_zero.orchestrator --preset fast --dry_run` — **exit 0**
- Both self-play dry-run stubs verified directly: masks present, mask and policy
  widths aligned at 376 (previously 100 vs 376 on the sequential stub)
- `training_civ6 --dry_run` and `distributed.coordinator --dry_run` are CPU-bound
  and exceeded practical local timeouts. Neither imports any module changed here
  (verified by AST import check), and both were left running at handoff.

## Verification note

The new and repaired tests were run against the **original** code as a negative
control: 5 fail there and pass here. This mattered because the tests being
replaced could not fail — the parity test wrapped every assertion in
`if batch_trajs and seq_trajs:` and compared key sets with `issubset`, and the
batch-mode tests re-implemented the production booleans in their own bodies.

## Known issues / caveats for Pre-PR

- **`cloud/evaluator.py` is not a deployed path.** `cloudbuild.yaml` builds only
  `Dockerfile.vertex`. The registry wiring there is divergence-prevention and
  documentation honesty, not a production fix — the commit message says so, and
  the case document's own F3 severity was corrected downward as a result.
- **One intended behavior change:** `--num_parallel_games` default 4 → 1. No run
  changes engine; the old default only ever produced a discarded flag.
- **One intended data change:** batch trajectories gain one mask row per step,
  matching what the sequential path has always written. Pre-change trajectories
  still load via the fallback, which now guards on length rather than key
  presence.
- **`run_batch_selfplay` accepts no seed** while `run_selfplay` does, so
  batch/sequential parity can be asserted on schema but never on content. Stated
  rather than papered over.
- **Fog of war is deliberately out of scope.** Its fogged mask is
  *over-permissive* for `found_city` next to a hidden enemy city — a behavior
  question, not a defect fix — so it needs its own OpenSpec change and PR.
