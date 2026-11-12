# Gumbel policy-target sharpness probe (LF-A5 / LF-A12)

**Date:** 2026-11-12 · **Search epoch:** post-2026-09-04 · **Games played:** 0 · **Raw data:** [`probe_target_sharpness.json`](probe_target_sharpness.json)

For each of 10 seeds (0, 1, 2, 3, 4, 5, 6, 7, 42, 1234), a randomly initialised `MicroCivNetwork` (`torch.manual_seed(seed)`, 0 games of training) searches 200 positions from seeded uniform-random play at board 8 / max_turns 60, with k=8, n=24 (A1's budget). `greedy_threshold` = 0.99: above it `select_action` ignores temperature, so on that move the self-play temperature schedule (temperature 1.0 before turn 30) is inert.

```
python scripts/probe_target_sharpness.py --positions 200 --seeds 0 1 2 3 4 5 6 7 42 1234 --out_dir docs/evidence/target_sharpness_probe
```

## Pooled over every seed (2000 positions, 999 in the temperature phase, 11 forced)

| Variant | Search (halving score) | Target | Entropy mean / median (nats) | max-p mean / median | Share max-p > threshold | Same in temperature phase | Support (median) | Argmax = `current` argmax |
|---|---|---|---|---|---|---|---|---|
| `current` | current | repo | 1.378 / 1.518 | 0.410 / 0.289 | 0.019 | 0.011 | 8 | 1.000 |
| `current_search_paper_target` | current | paper | 0.775 / 0.692 | 0.634 / 0.563 | 0.413 | 0.418 | 15 | 0.614 |
| `sigma_halving` | sigma_halving | repo | 1.310 / 1.469 | 0.430 / 0.329 | 0.018 | 0.009 | 8 | 0.933 |
| `paper` | paper | paper | 0.769 / 0.693 | 0.627 / 0.530 | 0.383 | 0.392 | 15 | 0.534 |
| `paper_mctx_scale` | paper_mctx_scale | paper | 1.722 / 1.816 | 0.425 / 0.296 | 0.005 | 0.005 | 15 | 0.602 |

## Per seed: share of positions with max-p above the threshold (mean max-p)

| Seed | abs root value, median | Visited-Q spread, median | `current` | `current_search_paper_target` | `sigma_halving` | `paper` | `paper_mctx_scale` | Halving winner agrees with `current` (sigma_halving / paper) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.014 | 0.026 | 0.015 (0.211) | 0.035 (0.417) | 0.015 (0.211) | 0.015 (0.395) | 0.000 (0.207) | 0.93 / 0.67 |
| 1 | 0.046 | 0.070 | 0.025 (0.667) | 0.780 (0.952) | 0.030 (0.686) | 0.650 (0.910) | 0.010 (0.713) | 0.57 / 0.17 |
| 2 | 0.038 | 0.056 | 0.020 (0.569) | 0.655 (0.842) | 0.020 (0.590) | 0.650 (0.833) | 0.005 (0.711) | 0.64 / 0.39 |
| 3 | 0.057 | 0.055 | 0.030 (0.263) | 0.400 (0.619) | 0.020 (0.272) | 0.385 (0.615) | 0.000 (0.311) | 0.86 / 0.41 |
| 4 | 0.072 | 0.141 | 0.005 (0.255) | 0.295 (0.525) | 0.000 (0.270) | 0.275 (0.522) | 0.000 (0.252) | 0.86 / 0.42 |
| 5 | 0.029 | 0.058 | 0.030 (0.586) | 0.905 (0.975) | 0.030 (0.618) | 0.860 (0.964) | 0.015 (0.707) | 0.56 / 0.14 |
| 6 | 0.097 | 0.172 | 0.020 (0.291) | 0.170 (0.462) | 0.020 (0.312) | 0.145 (0.455) | 0.010 (0.233) | 0.71 / 0.42 |
| 7 | 0.154 | 0.297 | 0.005 (0.301) | 0.030 (0.251) | 0.005 (0.318) | 0.025 (0.259) | 0.005 (0.188) | 0.82 / 0.77 |
| 42 | 0.164 | 0.224 | 0.010 (0.309) | 0.010 (0.329) | 0.010 (0.336) | 0.010 (0.350) | 0.005 (0.241) | 0.77 / 0.61 |
| 1234 | 0.045 | 0.078 | 0.030 (0.644) | 0.850 (0.964) | 0.035 (0.683) | 0.815 (0.965) | 0.005 (0.688) | 0.58 / 0.16 |

The current target's sharpness scales with the *absolute* spread of the visited root-child Q values (raw Q, no normalisation); the paper's is min-max normalised, so any spread at all is stretched to the full `(c_visit + max N) * c_scale` range.

## Variants

- `current` — A1's settings: c_visit 50.0, c_scale 1.0, target `completed_q`; halving ranks by g + log pi + completedQ_raw (no sigma); target restricted to visited actions; completion `q + ((maxN-N)/maxN)·v_root`, raw Q.
- `current_search_paper_target` — the same tree, scored with the paper's target.
- `sigma_halving` — LF-A12 alone: halving ranks by `g + log pi + (c_visit + max N)·c_scale·completedQ` on this repository's completion; repository target.
- `paper` — Danihelka et al. (2022), Go/chess: c_visit 50, c_scale 1.0, `v_mix` completion, min-max normalised Q, sigma in the halving score and in a softmax target over all legal actions (as `mctx`'s `qtransform_completed_by_mix_value`).
- `paper_mctx_scale` — `paper` with c_scale 0.1 (`mctx`'s default; the paper's Atari setting).

Not varied: interior selection (this repository's PUCT, c_puct 1.0) and the halving schedule (already `mctx`'s).

## Non-claims

- not evidence of playing strength or of learning
- untrained networks only; a trained network's targets may be sharper or flatter
- positions come from random play, not from self-play
- interior (non-root) selection is this repository's PUCT in every variant
- the seed field is the first of several; every seed is listed under seeds

Interpretation and the decision this feeds: `kb/specs/20261111-completed-q-alignment-memo.md`.
