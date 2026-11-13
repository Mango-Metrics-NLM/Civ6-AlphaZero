# Windows baseline and calibration harness smoke

**Date:** 2026-11-11 · **Search epoch:** post-2026-09-04 · **Raw data:** [`machine_snapshot.json`](machine_snapshot.json), [`harness_smoke.json`](harness_smoke.json)

This directory records (1) the machine the local-first program will run on and
(2) one tiny run of `scripts/calibrate_throughput.py` showing the harness works.

**It contains no throughput estimate and no ETA.** The harness smoke is board 4,
5 turns, 4 simulations, 1 game; its timings say nothing about the board 8 /
60-turn / 24-simulation configuration. The real calibration is plan task T0b.

## Machine snapshot

| Item | Value |
|---|---|
| OS | Windows 11 Home, build 26200 |
| CPU | AMD Ryzen 9 9900X, 12 cores / 24 logical |
| RAM | 61.4 GiB total, 16.5 GiB free at capture |
| GPUs | NVIDIA GeForce RTX 5060 Ti (16311 MiB), NVIDIA GeForce RTX 5060 (8151 MiB) |
| Windows Update active hours | 07:00 to 01:00 (`ActiveHoursStart=7`, `ActiveHoursEnd=1`), read without admin |
| Windows Update restart policy | Group-policy key `...\WindowsUpdate\AU` returned no values; no other restart policy is readable without admin, so automatic restarts between 01:00 and 07:00 remain possible and must be assumed to interrupt long runs |

GPUs came from `nvidia-smi --query-gpu=name,memory.total --format=csv`. The
harness smoke ran on CPU (`--device cpu`, `OMP_NUM_THREADS=2`).

## Harness smoke

```
python scripts/calibrate_throughput.py --board_size 4 --max_turns 5 --mcts gumbel --k 4 --n 4 --device cpu --games 1 --seed 20261111 --out harness_smoke.json
```

It plays one complete seeded self-play game through `run_self_play` (the
trainer's own path) and reports `ms_per_eval`, `actions_per_game` and
`s_per_game`; all three came back positive. Values are in `harness_smoke.json`
and are deliberately not quoted here.

## Non-claims

- Not a throughput or wall-clock estimate for any training run.
- Not evidence of playing strength.
- `git_sha` is the base commit; the harness script was not yet committed when this ran.
