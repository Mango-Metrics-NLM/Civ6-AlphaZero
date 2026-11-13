# Make training and evaluation runs reproducible

## Why

The project could not replay its own runs. This is not a hypothesis — it is the
recorded finding of its only committed training run
(`docs/evidence/validation_iteration_zero/validation_iteration_zero_report.md`):

> **Not reproducible.** `training_civ6.py` exposes no `--seed` flag and
> `MicroCivEnv.reset()` is unseeded in the eval path, so neither the run nor the
> probes can be replayed exactly. The five probes above are the direct evidence:
> same checkpoint, same opponent seed, score rates spanning **0.25 to 0.40**.

Five 20-game probes of a single checkpoint scored 0.25 / 0.35 / 0.375 / 0.375 /
0.40. Every judgement downstream inherits that spread: the promotion threshold
(`docs/runbooks/promotion-gate.md`), the encoder A/B comparison, and any claim
about a per-iteration trend. The report says so itself — a single 20-game number
"should be treated as one draw from a wide distribution".

PR #115 met the same defect from the other side. A CI failure that presented as
a flake was a half-seeded test: `run_batch_selfplay(seed=...)` seeded the
environments only, which the worker's own docstring stated. Fixing the test did
not touch the cause.

## What Changes

- **Every search owns its RNG.** Eight global `np.random.*` draws across
  `core/mcts/` and `core/batch_mcts.py` are replaced by a per-instance
  generator seeded from the variant's config. `gumbel_sample_actions` takes an
  `rng` parameter, threaded from all three of its callers.
- **`MicroCivEnv.clone()` carries its RNG.** It previously constructed a fresh
  environment with no seed, so every MCTS rollout — cloning is how the search
  simulates, and `step` itself draws — used a new unseeded generator no caller
  could reach. Deep-copied, so a simulation cannot advance the real game's
  stream.
- **`training_civ6.py` gains `--seed`,** threaded to self-play environments and
  search, evaluation environments, search and random opponent, the replay
  buffer, the DataLoader shuffle, and network initialisation. Offsets keep the
  streams disjoint.
- **The evaluation entry points gain `--seed`.** The environments and baseline
  opponents were already seeded; `NeuralNetworkPlayer` — the *evaluated agent* —
  sampled from the global RNG while its siblings in the same module each owned a
  seeded generator.
- **A determinism gate** (`tests/regression/test_training_determinism_regression.py`)
  asserting the property behaviourally against the real `train()` entry point,
  plus a static check that no search module draws globally.

Every new parameter defaults to `None`, meaning unseeded, so existing callers
are unaffected.

## Impact

- **Behaviour:** seeded runs draw from a `numpy.random.Generator` rather than the
  legacy global functions, so action sequences differ from an unseeded run today.
  Nothing depended on the previous sequences — there was no reproducibility to
  regress.
- **Not addressed:** the ~73.5% draw rate that `docs/runbooks/promotion-gate.md`
  identifies as the upstream problem. Reproducibility narrows the *variance* of a
  measurement; it does not move the measurement. This change must not be read as
  fixing the promotion gate.
