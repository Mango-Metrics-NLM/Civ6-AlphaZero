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

### 1. Run GPU determinism and E2E AQA suites

```bash
pytest tests/regression/test_cuda_determinism_regression.py tests/regression/test_gm_bridge_and_gpu_e2e_aqa.py -v
```

Verifies two-process seeded training/arena reproduction on every visible GPU,
cross-GPU checkpoint save/load parity, finite gradients, and legal-action mask zeroing.

### 2. Check cross-process determinism script and run-card pricing

```bash
python scripts/check_determinism.py --help
python scripts/run_card.py configs/run_cards/smoke.yaml --price_only
```

Use `--price_only` or `--probe` to verify device preflights and the 1,000-game cap
without claiming a single-use `runs/<card_id>` evidence directory.

## Guardrails

1. Never execute a single-use production/evidence run card (`t0b`, `a1`, `a1r`) without
   explicit maintainer confirmation (`make verify-registration` holds the digest gate).
2. Keep total priced games <= 1,000 unless explicitly running an approved card with
   `--approval_ref`.
3. Never treat a skipped CUDA test on a CPU host as a GPU pass; report the observed
   `torch.cuda.is_available()` and GPU names explicitly.

## Related

- `verify` — full CI parity gate (CPU suite + coverage 85% gate).
- `a1-reregister` — procedure when changing A1 devices or commands.
