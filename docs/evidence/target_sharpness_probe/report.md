# Gumbel policy-target sharpness probe (LF-A5 / LF-A12)

**Date:** 2026-11-12 · **Search epoch:** post-2026-09-04 · **Games played:** 0 · **Raw data:** [`probe_target_sharpness.json`](probe_target_sharpness.json)

For each of 10 seeds (0, 1, 2, 3, 4, 5, 6, 7, 42, 1234), a randomly initialised `MicroCivNetwork` (`torch.manual_seed(seed)`, 0 games of training) searches 200 positions from seeded uniform-random play at board 8 / max_turns 60, with k=8, n=24 (A1's budget). `greedy_threshold` = 0.99: above it `select_action` ignores temperature, so on that move the self-play temperature schedule (temperature 1.0 before turn 30) is inert.

```
python scripts/probe_target_sharpness.py --positions 200 --seeds 0 1 2 3 4 5 6 7 42 1234 --out_dir docs/evidence/target_sharpness_probe
```

## Pooled over every seed (2000 positions, 999 in the temperature phase, 11 forced)

| Variant | Search (halving score) | Target | Entropy mean / median (nats) | max-p mean / median | Share max-p > threshold | Same in temperature phase | Support (median) | Argmax = `current` argmax | Share across seeds: SE [min, max] |
|---|---|---|---|---|---|---|---|---|---|
| `current` | current | repo | 1.378 / 1.518 | 0.410 / 0.289 | 0.019 | 0.011 | 8 | 1.000 | 0.003 [0.005, 0.030] |
| `sigma_halving` | sigma_halving | repo | 1.310 / 1.469 | 0.430 / 0.329 | 0.018 | 0.009 | 8 | 0.933 | 0.004 [0.000, 0.035] |
| `paper_affine` | paper_affine | affine | 2.401 / 2.571 | 0.201 / 0.127 | 0.013 | 0.009 | 15 | 0.680 | 0.003 [0.000, 0.030] |
| `current_search_paper_affine_target` | current | affine | 2.411 / 2.564 | 0.200 / 0.127 | 0.014 | 0.009 | 15 | 0.695 | 0.003 [0.000, 0.030] |
| `mctx_minmax_c1` | mctx_minmax_c1 | minmax | 0.769 / 0.693 | 0.627 / 0.530 | 0.383 | 0.392 | 15 | 0.534 | 0.107 [0.010, 0.860] |
| `current_search_mctx_minmax_c1_target` | current | minmax | 0.775 / 0.692 | 0.634 / 0.563 | 0.413 | 0.418 | 15 | 0.614 | 0.113 [0.010, 0.905] |
| `mctx_default` | mctx_default | minmax | 1.722 / 1.816 | 0.425 / 0.296 | 0.005 | 0.005 | 15 | 0.602 | 0.002 [0.000, 0.015] |

## Per seed: share of positions with max-p above the threshold (mean max-p)

| Seed | abs root value, median | Visited-Q spread, median | `current` | `sigma_halving` | `paper_affine` | `current_search_paper_affine_target` | `mctx_minmax_c1` | `current_search_mctx_minmax_c1_target` | `mctx_default` | Halving winner agrees with `current` (sigma_halving / paper_affine / mctx_minmax_c1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.014 | 0.026 | 0.015 (0.211) | 0.015 (0.211) | 0.015 (0.110) | 0.015 (0.111) | 0.015 (0.395) | 0.035 (0.417) | 0.000 (0.207) | 0.93 / 0.97 / 0.67 |
| 1 | 0.046 | 0.070 | 0.025 (0.667) | 0.030 (0.686) | 0.020 (0.276) | 0.020 (0.293) | 0.650 (0.910) | 0.780 (0.952) | 0.010 (0.713) | 0.57 / 0.74 / 0.17 |
| 2 | 0.038 | 0.056 | 0.020 (0.569) | 0.020 (0.590) | 0.015 (0.239) | 0.015 (0.255) | 0.650 (0.833) | 0.655 (0.842) | 0.005 (0.711) | 0.64 / 0.82 / 0.39 |
| 3 | 0.057 | 0.055 | 0.030 (0.263) | 0.020 (0.272) | 0.020 (0.119) | 0.020 (0.118) | 0.385 (0.615) | 0.400 (0.619) | 0.000 (0.311) | 0.86 / 0.86 / 0.41 |
| 4 | 0.072 | 0.141 | 0.005 (0.255) | 0.000 (0.270) | 0.000 (0.111) | 0.000 (0.107) | 0.275 (0.522) | 0.295 (0.525) | 0.000 (0.252) | 0.86 / 0.89 / 0.42 |
| 5 | 0.029 | 0.058 | 0.030 (0.586) | 0.030 (0.618) | 0.030 (0.260) | 0.030 (0.261) | 0.860 (0.964) | 0.905 (0.975) | 0.015 (0.707) | 0.56 / 0.70 / 0.14 |
| 6 | 0.097 | 0.172 | 0.020 (0.291) | 0.020 (0.312) | 0.015 (0.159) | 0.015 (0.152) | 0.145 (0.455) | 0.170 (0.462) | 0.010 (0.233) | 0.71 / 0.75 / 0.42 |
| 7 | 0.154 | 0.297 | 0.005 (0.301) | 0.005 (0.318) | 0.005 (0.180) | 0.005 (0.171) | 0.025 (0.259) | 0.030 (0.251) | 0.005 (0.188) | 0.82 / 0.85 / 0.77 |
| 42 | 0.164 | 0.224 | 0.010 (0.309) | 0.010 (0.336) | 0.010 (0.273) | 0.010 (0.243) | 0.010 (0.350) | 0.010 (0.329) | 0.005 (0.241) | 0.77 / 0.80 / 0.61 |
| 1234 | 0.045 | 0.078 | 0.030 (0.644) | 0.035 (0.683) | 0.005 (0.286) | 0.010 (0.288) | 0.815 (0.965) | 0.850 (0.964) | 0.005 (0.688) | 0.58 / 0.71 / 0.16 |

The current target's sharpness scales with the *absolute* spread of the visited root-child Q values: raw Q in [-1, 1] times `(c_visit + max N) * c_scale`, i.e. 57 per unit of Q at n=24. Under the affine reading the paper's coefficient per unit of raw Q is half that (28.5). Under per-node min-max any spread at all is stretched to the full 57-logit range.

## Variants

- `current` — A1's settings: c_visit 50.0, c_scale 1.0, target `completed_q`; halving ranks by g + log pi + completedQ_raw (no sigma); target restricted to visited actions; completion `q + ((maxN-N)/maxN)·v_root`, raw Q.
- `sigma_halving` — LF-A12 alone: halving ranks by `g + log pi + (c_visit + max N)·c_scale·completedQ` on this repository's completion; repository target.
- `paper_affine` — Danihelka et al. (2022) Go/chess constants (c_visit 50, c_scale 1.0, Appendix F) with Q mapped to [0, 1] by `(q + 1) / 2`; `v_mix` completion; sigma in the halving score and in a softmax target over all legal actions. The paper says only that Go/chess Q values are normalised to [0, 1]; this is the fixed-bounds reading.
- `mctx_minmax_c1` — the same, but with `mctx`'s per-node min-max rescale (`qtransform_completed_by_mix_value`) at c_scale 1.0. No published configuration pairs the two: `mctx` uses min-max with value_scale 0.1, and the paper shows a min-max form only for Atari (section 7).
- `mctx_default` — `mctx`'s defaults: per-node min-max, c_scale 0.1, c_visit 50.
- `current_search_*_target` — the `current` tree, scored with that target formula.

Not varied: interior selection (this repository's PUCT, c_puct 1.0) and the halving schedule (already `mctx`'s).

## Non-claims

- not evidence of playing strength or of learning
- untrained networks only; a trained network's targets may be sharper or flatter
- positions come from random play, not from self-play
- interior (non-root) selection is this repository's PUCT in every variant
- the paper does not state how Go/chess Q is normalised; paper_affine is one reading of it, mctx_minmax_c1 is mctx's rescale at the paper's c_scale
- the seed field is the first of several; every seed is listed under seeds
- per-position rows are omitted; they are regenerated from the seeds

Interpretation and the decision this feeds: `kb/specs/20261111-completed-q-alignment-memo.md`.
