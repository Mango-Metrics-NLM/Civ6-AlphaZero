# Probe reproducibility: seeded vs unseeded

> **Pre-fix artifact (superseded).** Generated before the 2026-09-04 search fixes (`e9d0572`: sign-correct value backup, completed-Q policy target, per-iteration seeds). The numbers below are a record of the mechanism and the wall clock, not evidence of strength. See [`docs/evidence/README.md`](../README.md) for the epoch convention.

**Date:** 2026-08-30 · **Raw data:** [`probe_reproducibility.json`](probe_reproducibility.json)

## What this measures

`docs/evidence/validation_iteration_zero/` recorded that a single evaluation
number could not be trusted:

> **Not reproducible.** … The five probes above are the direct evidence: same
> checkpoint, same opponent seed, score rates spanning **0.25 to 0.40**. Any
> single 20-game number from this setup should be treated as one draw from a
> wide distribution.

This is the same experiment repeated after the reproducibility work, as a
controlled comparison: **one checkpoint, one code path, ten probes** — five with
a seed and five without. The seed is the only variable.

## Result

| | Score rates over 5 probes | Distinct values | Range |
|---|---|---:|---:|
| **Seeded** (`seed=4242`) | 0.325, 0.325, 0.325, 0.325, 0.325 | **1** | **0.000** |
| **Unseeded** (`seed=None`) | 0.45, 0.40, 0.30, 0.45, 0.40 | 3 | 0.150 |

The seeded probes are identical — not merely close. The unseeded probes
reproduce the original phenomenon at a comparable magnitude (0.30–0.45 here
against 0.25–0.40 in the earlier run), which is what makes this a control rather
than a demonstration: the variance is still there when the seed is withheld, so
the collapse is attributable to the seed and not to some other change.

Score rate is draw-aware, `(wins + 0.5 · draws) / games` — the statistic
`evaluation/scoring.py` owns, not the draw-blind `wins / games` that
`docs/runbooks/promotion-gate.md` documents as capping the deployed gate.

## Setup

One checkpoint trained at `seed=2026`: 8×8 board, 20 turns, 12 MCTS simulations,
4 games, 1 iteration, 2 epochs, 2 residual blocks, 16 filters, CPU. Each probe is
20 games against the random baseline. Training 39 s; the ten probes 241 s.
`OMP_NUM_THREADS=2` (CLAUDE.md §5.3 — it changes the wall clock materially on
this host, not the results).

## What this does and does not show

**Does:** a probe of a fixed checkpoint at a fixed seed now returns one answer,
so a reported evaluation number is a measurement that can be re-taken and
compared.

**Does not:** say anything about whether the agent is any good. The checkpoint
here is deliberately tiny — 4 self-play games — and 0.325 is well below the 0.5
a coin-flip opponent would score. No capability claim is made or implied.

**Does not:** address the draw rate. 12 of 20 games drew in the probes above.
`docs/runbooks/promotion-gate.md` identifies the draw rate as the upstream
problem for promotion, and reproducibility narrows the variance of a measurement
without moving the measurement.

## Reproducing

```bash
OMP_NUM_THREADS=2 python -m alpha_zero.training_civ6 --dry_run --seed 42   # twice; diff
pytest tests/regression/test_training_determinism_regression.py -q         # the gate, ~7s
```
