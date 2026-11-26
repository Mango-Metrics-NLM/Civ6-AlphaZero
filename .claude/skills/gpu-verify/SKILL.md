---
name: gpu-verify
description: Run deterministic single-GPU and dual-GPU CUDA verification, multi-device training/arena checks, and run-card dry probes under the registered environment contract. Use after CUDA kernels, seeding, evaluation, or model changes.
allowed-tools: Bash, Read
---

# GPU Determinism Verification Skill

Executes deterministic multi-GPU regression checks and run-card device preflights
under the exact environment contract required for reproducible CUDA execution.

## Why this exists

PyTorch CUDA kernels and cuBLAS workspace reductions are non-deterministic unless
pinned device ordering and workspace configuration are active before process start.
Running GPU checks under mismatched `MCTS_*` overrides or unpinned device ordering
masks determinism bugs and produces invalid timing/reproduction comparisons.

## Environment Contract

Export these variables before running any GPU check:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
unset MCTS_NUM_SIMULATIONS MCTS_C_PUCT MCTS_DIRICHLET_ALPHA MCTS_DIRICHLET_EPSILON
```

On Windows PowerShell, set `$env:CUDA_DEVICE_ORDER="PCI_BUS_ID"`,
`$env:CUDA_VISIBLE_DEVICES="0,1"`, `$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"`,
`$env:OMP_NUM_THREADS="2"`, and clear any `Env:MCTS_*` variables.

## Steps

### 1. Run developer-side GPU determinism and E2E AQA suites

```bash
CIV6AZ_REQUIRE_GPU=2 pytest tests/regression/test_cuda_determinism_regression.py tests/regression/test_gm_bridge_and_gpu_e2e_aqa.py -v
```

Setting `CIV6AZ_REQUIRE_GPU=N` (or running `make test-gpu` across all 21 GPU-gated modules) enables `tests/fixtures/gpu_skip_gate.py`, which fails at startup if fewer than `N` CUDA devices are visible and converts any CUDA/GPU/device skip (`CIV6AZ_GPU_SKIP_PATTERN`) into a test failure. Leave `CIV6AZ_REQUIRE_GPU` unset on CPU-only hosts.

Runs **in-process** developer-side checks on every visible GPU (repeating seeded
training/arena within the same Python process, cross-GPU checkpoint save/load parity,
finite gradients, and legal-action mask zeroing). These developer checks catch
kernel and plumbing regressions quickly, but they do not establish bit-exact
reproduction across independent process starts.

### 2. Run cross-process determinism verification and run-card preflights

```bash
python -m scripts.check_determinism --train-device cuda:0 --arena-device cuda:1 --out determinism.json
python scripts/run_card.py configs/run_cards/smoke.yaml --probe
python scripts/run_card.py configs/run_cards/t0b.yaml --price_only
```

- `scripts/check_determinism.py` runs each repeat in its own subprocess under the
  inherited CUDA/cuBLAS environment (the registered cross-process gate for A1 is
  T0b's `determinism` step inside the card's pinned environment).
- `--probe` rehearses pin, preflight, and GPU device verification in a throwaway
  `probe-<card_id>-<stamp>` directory without claiming the single-use `runs/<card_id>`
  directory (use `--probe` on `configs/run_cards/t0b.yaml` once its `sha` is filled with a
  merged commit).
- `--price_only` inspects the game budget and exits immediately after pricing without
  building the pinned environment or probing any GPU device.

## Guardrails

1. Never execute a single-use production/evidence run card (`t0b`, `a1`, `a1r`) without
   explicit maintainer confirmation (`make verify-registration` holds the digest gate).
2. Keep total priced games <= 1,000 unless explicitly running an approved card with
   `--approval_ref`.
3. Never treat a skipped CUDA test on a CPU host as a GPU pass; set `CIV6AZ_REQUIRE_GPU=<n>`
   on GPU hosts so CUDA/GPU skips fail at pytest level, and report the observed
   `torch.cuda.is_available()` and GPU names explicitly.

## Related

- `verify` — full CI parity gate (CPU suite + coverage 85% gate).
- `a1-reregister` — procedure when changing A1 devices or commands.
