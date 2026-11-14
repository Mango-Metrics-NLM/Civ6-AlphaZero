# Run the registered cards on GPUs, and make a GPU run mean what it says

## Why

The A1 protocol (`docs/EVALUATION.md` section 7.1) and its run cards were pinned
to `--device cpu`, and the maintainer has decided the registered runs are
GPU-first: training on `cuda:0`, the arena on `cuda:1`. Changing the device
token alone would have produced runs that look like GPU runs and are not:

- the pinned environment installs whatever torch PyPI serves, which on Windows
  is CPU-only, and is cached by commit alone;
- every runner falls back to CPU with a warning when CUDA is missing, so a
  `cuda:1` card would train on CPU and exit 0;
- seeding does not make CUDA kernels deterministic, and section 7.1's
  "After a VOID" rule depends on a same-seed re-run reproducing the same games;
- evidence named GPU 0 whatever device ran.

No A1-family run has started, so section 7.1 may still be re-registered without
retiring the confirmatory seeds.

## What Changes

- Run cards gain `torch_spec` and `torch_index_url`. The pinned environment
  installs that exact torch first and is keyed by the pin as well as the commit.
- `run_card` refuses a card whose `--device` names a CUDA device the pinned
  environment cannot provide, or whose installed torch is not the pinned one.
  `--probe` rehearses the pin, preflight and device check without claiming the
  card's single-use directory. Children get `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
  `CUBLAS_WORKSPACE_CONFIG`; `run.json` records the device plan.
- `utils/seeding.enforce_deterministic_backend`: a seeded run on a CUDA device
  enforces deterministic algorithms. Called by the trainer, the benchmark and
  the throughput calibration. A no-op on any other device.
- `calibrate_throughput` refuses an explicit GPU it cannot run and records the
  device it used. Evidence `gpu_model` names the device used.
- Section 7.1 is re-registered: `cuda:0` for training, `cuda:1` for every arena
  arm; the torch build and the GPU at each ordinal named in a machine-readable
  `a1-devices` block; GPU determinism made a precondition with a stated
  consequence; the re-run after an interruption made a reproduction check that a
  differing re-run cannot pass. Two registration-log rows and digests. The A1,
  A1-R and T0b cards follow; T0b runs the reproduction check first, also times
  `cuda:1` and `cpu`, and prices at 660 games.
- `alpha_zero/evaluation/reproduction.py` compares two run directories;
  `scripts/check_determinism.py` (runner `determinism_check`) runs an A1-shaped
  job twice in separate processes and compares them.
- The device check reads each invocation's device from its runner's own parser,
  checks the build's architecture list and one real CUDA operation per device,
  and holds each ordinal to the GPU the card names. The release verifier voids a
  run whose recorded devices depart from the registration.

## Impact

- Behaviour: CPU runs are unchanged. A seeded GPU run now raises on an operation
  with no deterministic implementation. A card that names a missing GPU fails
  before launching anything.
- Not verified in CI: CI is CPU-only. The CUDA install, the determinism settings
  and all timings are first exercised on the maintainer's GPU machine by
  `run_card --probe`, `tests/regression/test_cuda_determinism_regression.py`
  (skipped without CUDA) and the T0b card.
- Affected specs: `training-reproducibility`, `validation-gates`.
