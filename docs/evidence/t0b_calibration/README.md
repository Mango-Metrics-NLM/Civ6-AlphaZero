# T0b Dual-GPU Determinism, Throughput & Positive-Control Calibration (2026-11-26)

> **Scope & non-claims:** T0b (`t0b-20261112`, `660` games on `dev` commit `5138af23eb906f8bf697584351cf9bcf85573576`) is a **calibration and hardware-reproducibility precondition** for Section 7.1 of [`../../EVALUATION.md`](../../EVALUATION.md), not an A1-family learning run. Heuristic vs. random measures the fixed positive-control baseline only.

## Hardware & Environment (`run.json`, `determinism.json`)

- **Host:** Windows (`AMD64`, `24` logical cores), Python `3.11.9`, NumPy `1.26.4`, PyTorch `2.11.0+cu128` (`CUDA 12.8`, cuDNN `91900`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0,1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`).
- **Devices:**
  - `cuda:0`: `NVIDIA GeForce RTX 5060 Ti` (`sm_120`, training & self-play)
  - `cuda:1`: `NVIDIA GeForce RTX 5060` (`sm_120`, arena evaluation)

## 1. Dual-Process GPU Determinism Precondition (`determinism.json`)

- **Command / budget:** `scripts/check_determinism.py --train-device cuda:0 --arena-device cuda:1 --board_size 8 --max_turns 60 --k 8 --n 24 --arena-games 20` (`48` games priced across `2` independent repeats × `2` iterations × `2` self-play games + `20` arena games, seed `20261114`).
- **Result:** **`status: "REPRODUCED"`** (`differences: []`, `coverage_gaps: []`; `3` checkpoints, `2` iterations, and `1` arena comparison matched tensor-for-tensor and game-for-game across separate processes).

## 2. Wall-Clock Throughput Calibration (`board=8`, `max_turns=60`, Gumbel `k=8`, `n=24`)

| Device | Role | `games` | `s_per_game` | `ms_per_eval` | `total_seconds` | Source |
|---|---|---:|---:|---:|---:|---|
| `cuda:0` (`RTX 5060 Ti`) | Self-play / training | `4` | `23.9615` | `2.9741` | `95.8462` | [`throughput.json`](throughput.json) |
| `cuda:1` (`RTX 5060`) | Arena evaluation | `4` | `22.7267` | `2.8208` | `90.9067` | [`throughput_cuda1.json`](throughput_cuda1.json) |
| `cpu` | Informational only | `4` | `18.9866` | `2.3361` | `75.9464` | [`throughput_cpu.json`](throughput_cpu.json) |

## 3. Positive-Control Calibration & Section 7.1 Branch Decision (`calibration_summary.json`)

Run via `python -m alpha_zero.evaluation.a1_protocol calibrate` across seeds `1001`, `1002`, and `1003` (`200` games each, `600` games / `300` pooled seat-swapped pairs):

- **Positive control (`heuristic` vs `random`):** `W/D/L 303/255/42`, score **`0.7175`**, 95% hull **`[0.6912, 0.7421]`**.
- **Pair-bootstrap void risk:** **`P(40) = 0.0166`** (95% interval **`[0.0030, 0.0493]`**), **`P(60) = 0.0010`**, informational `iid P(40) = 0.0215`.
- **Branch in force:** **`40`** (`P(40)` interval upper bound `0.0493 <= 0.05`).
- **One-card A1 ETA:** `480` self-play games × `24.0 s` (`cuda:0`) + `460` arena games × `22.7 s` (`cuda:1`) = **`6.1 h`** (excluding training epochs; `<= 12 h`, so the registered one-card form applies).
