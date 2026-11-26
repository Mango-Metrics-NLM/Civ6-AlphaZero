---
name: run-card-evidence
description: Prepare, probe, calibrate, compare VOID runs, and build/verify release evidence bundles for GPU-first run cards (t0b, a1, a1r) without invalidating the release play path.
allowed-tools: Bash, Read
---

# Run Card & Release Evidence Skill

Standardizes the operator and release-manager workflow around
`configs/run_cards/t0b.yaml`, `configs/run_cards/a1.yaml`, and
`configs/run_cards/a1r.yaml` so budget caps, dual-GPU determinism checks,
`T0b` slot calibration, "After a VOID" reproduction checks, and release evidence
bundles stay aligned with `docs/runbooks/branching-and-release.md`.

## Why this exists

Run cards use single-use output directories under `E:\civ6az-data\runs\<card_id>`,
require a clean play-path diff between the evidence commit, `-rc.N` tag, and release
HEAD, and obey strict rules when recovering from an interrupted `VOID` attempt.
Checking prices, probes, calibration outputs, and bundle invariants in a consistent
sequence prevents accidental invalidation of confirmatory GPU evidence.

## Prerequisites

- Python 3.11+ with the `[dev]` extra installed (`torch==2.11.0+cu128` for GPU cards).
- Dual-GPU host pinned to `cuda:0` / `cuda:1` (`CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUDA_VISIBLE_DEVICES=0,1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=2`),
  zero `MCTS_*` environment variables set, Windows Update paused, and WSL/`llama-server`
  stopped before running real cards.
- Existing `E:\civ6az-data\runs\` directories must never be overwritten or deleted.

## Steps

### 1. Verify run-card game budget and device probe preflight

```bash
python scripts/run_card.py configs/run_cards/t0b.yaml --price_only
python scripts/run_card.py configs/run_cards/a1.yaml --price_only
python scripts/run_card.py configs/run_cards/a1r.yaml --price_only
python scripts/run_card.py configs/run_cards/smoke.yaml --probe
python -m scripts.check_determinism --train-device cuda:0 --arena-device cuda:1 --board_size 8 --max_turns 60 --k 8 --n 24 --arena-games 20 --out determinism.json
```

`--price_only` verifies every invocation fits within the 1,000-game per-runner cap without
touching disk. `--probe` exercises a throwaway `probe-<card_id>-<stamp>` directory and
leaves `runs/<card_id>` unclaimed.

### 2. Calibrate `T0b` outputs and verify §7.1 registration digest

```bash
python -m alpha_zero.evaluation.a1_protocol calibrate <seed_1001_dir> <seed_1002_dir> <seed_1003_dir> --throughput <cuda0_throughput.json> --arena-throughput <cuda1_throughput.json>
make verify-registration
```

Commit the `T0b` `determinism.json`, `run.json`, three throughput files, and §7.1 slot fill-in
to `dev` before promoting `dev -> qa` and tagging the release candidate.

### 3. Compare interrupted `VOID` attempts ("After a VOID")

```bash
python -m alpha_zero.evaluation.reproduction compare <void_run_dir> <rerun_dir> --out compare.json
```

Run only when re-running the identical `sha`, pinned torch build, GPU ordinals, and NVIDIA
driver (`0` = `REPRODUCED`, `1` = `DIFFERS`, `2` = `NOTHING TO COMPARE`). Never run
`reproduction compare` across a code change (a symmetry or positive-control `VOID` requires
a new `-rc.N` tag and a second draw).

### 4. Build and verify the release evidence bundle & release readiness

```bash
python scripts/build_release_bundle.py --release <X.Y.Z> --rc <vX.Y.Z-rc.N> --run A1=<a1_dir> --run A1-R=<a1r_dir>
python scripts/verify_release_evidence.py --repo Mango-Metrics-NLM/Civ6-AlphaZero
make check-release-readiness
make validate
```

Verify every entry in `scripts/release_play_path.txt` has zero diff between each run commit,
the highest `-rc.N` tag, and the release head.

## Guardrails

1. Maintainer-only scripts (`scripts/maintainer/tag_release.py`,
   `scripts/maintainer/merge_backmerge.sh`) must never be executed with `--execute` by an
   agent, and agents must never create or push `v*` tags.
2. Never edit files in `scripts/release_play_path.txt` on an open release-candidate line.
3. Never delete or overwrite `E:\civ6az-data\runs\` or run `t0b`/`a1`/`a1r` cards without
   explicit maintainer authorization and `--probe` passing first.

## Related

- `gpu-verify` — dual-GPU CUDA determinism developer verification.
- `a1-reregister` — procedure for changing §7.1 devices or A1 command registration.
- `public-release-audit` — pre-release readiness, secret hygiene, and baseline ratchets.
