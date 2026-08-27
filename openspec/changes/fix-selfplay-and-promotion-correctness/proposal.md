# Fix silent data and decision defects in the cloud self-play and promotion paths

## Why

Three defects were measured on 2026-08-27 while reviewing the M4 roadmap. Each is
silent: the pipeline reports success while producing degraded data or an
untrustworthy decision.

1. **Batch self-play produced training data with masking disabled.**
   `run_batch_selfplay` initialised its trajectory dict without `action_masks`
   (`cloud/gcp/selfplay_worker.py`), and `prepare_dataset`
   (`cloud/gcp/training_worker.py`) substitutes an all-ones mask when the key is
   absent. Every batch-generated sample therefore trained with masked policy
   loss effectively disabled: the network was taught to place probability mass
   on illegal actions. The test that should have caught this compared key sets
   with `issubset` (which cannot detect an omitted field) and wrapped every
   assertion in `if batch_trajs and seq_trajs:`, so it also passed on empty
   output.

2. **The batch loop could emit ragged trajectories.** Observations were appended
   before the batch search that may raise; on failure the loop breaks, leaving
   `spatial` longer than `policies`/`players`. `prepare_dataset` iterates
   `spatial` and indexes the others, so such a shard raises `IndexError` during
   training. Adding a fourth pre-search field would have widened the window.

3. **Self-play parallelism was requested and silently ignored.**
   `--num_parallel_games` defaulted to 4 while `--mcts_type` defaulted to
   `gumbel`; the worker dispatches to the batch path only when parallelism > 1
   **and** the variant is `batch_gumbel`. Every unflagged run forwarded a
   parallelism flag that was then discarded, with no warning on either side, so
   the configuration advertised vectorized self-play and the run was sequential.

A fourth item is corrected as documentation rather than behavior: `cloud/evaluator.py`
decides promotion independently of the deployed path, and
`evaluation/promotion.py` described it as a peer decision site. It is not:
`cloudbuild.yaml` builds only `dockerfiles/Dockerfile.vertex`, whose entrypoint
routes through `cloud/gcp/entrypoint.py`. It is nonetheless a divergence risk if
revived, so it is routed through the promotion-policy registry and labelled.

## What Changes

- Self-play trajectory construction gains one definition
  (`new_trajectory`/`append_step`/`dummy_trajectory`) shared by the sequential,
  MuZero and batch paths, so a path cannot omit a per-step field. The stored
  object remains a plain `dict`: shards are pickled and read across a rolling
  window of iterations, so a dataclass would be a cross-version hazard.
- The batch path buffers observations and commits every per-step field
  atomically after a successful search.
- Both dry-run stubs emit masks, with dimensions taken from
  `NetworkArchitectureConfig` rather than literals (the sequential stub emitted a
  100-wide policy against a 376-wide action space).
- The trainer's mask fallback guards on length rather than key presence, so an
  empty `action_masks` list falls back instead of indexing out of range.
- `--num_parallel_games` defaults to 1 — what every historical default run
  actually did — and both the orchestrator and the worker warn when parallelism
  is requested with a variant that cannot use it. The orchestrator deliberately
  warns rather than rewriting `mcts_type`, which is shared with the evaluation,
  arena and tournament phases.
- `cloud/evaluator.py` decides promotion through the registry via a single seam
  shared by the early-stop check and the final gate, records `score_rate`/CI
  bounds as shadow statistics, and exposes the tri-state decision.
- `PromotionConfig` validates `confidence` against the z-table `score_interval`
  actually supports, instead of accepting any value in (0, 1) and raising
  mid-evaluation.

## Impact

- **Behavior change (intended):** the default `--num_parallel_games` moves from
  4 to 1. This matches what default runs already did; no run changes engine.
- **Data change (intended):** batch trajectories are one mask row per step
  larger, matching what the sequential path has always written.
- **No promotion behavior change by default:** the `legacy_win_rate` policy
  reproduces the historical `wins / total` arithmetic exactly. Draw-aware
  scoring remains opt-in via `CIV6AZ_PROMOTION_POLICY`.
- **Backwards compatible reads:** trajectories written before this change (no
  `action_masks`) still load and train via the documented fallback.
- Fog-of-war's related action-mask leak is deliberately **out of scope** and
  tracked separately: its fogged mask is over-permissive rather than merely less
  informative, which is a behavior question needing its own change.
