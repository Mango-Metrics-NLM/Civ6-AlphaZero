# Encoder A/B evidence: patch_embed vs hex_crystalline

> **Pre-fix artifact (superseded).** Generated before the 2026-09-04 search fixes (`e9d0572`: sign-correct value backup, completed-Q policy target, per-iteration seeds). The numbers below are a record of the mechanism and the wall clock, not evidence of strength. See [`docs/evidence/README.md`](../README.md) for the epoch convention.

Generated 2026-08-10T11:44:12.220492+00:00 from commit `b14534a5ad84050f8c2583f8f3246643a25d4380`.

## Head-to-head result

`hex_crystalline` versus `patch_embed`, played from `hex_crystalline`'s perspective.

| Metric | Value |
|--------|-------|
| Games | 800 |
| Win / Draw / Loss | 142 / 588 / 70 |
| Score rate (draws = 0.5) | 0.545 |
| Wilson 95% CI | [0.510, 0.579] |
| CI half-width | ±0.034 |
| Elo difference | +31.4 |
| Significant at 95% | yes |

## Training arms

| Arm | Parameters | Games | Examples | Final loss | Wall clock |
|-----|-----------:|------:|---------:|-----------:|-----------:|
| `patch_embed` | 494,137 | 280 | 57283 | 1.9320 | 109.8 min |
| `hex_crystalline` | 553,663 | 280 | 56925 | 2.0837 | 136.7 min |

## Budget

Both arms were trained from the same code, seed, and budget; they differ only in encoder type. (Verified by comparing the two arms' sidecars.)

| Setting | Value |
|---------|-------|
| `iterations` | 20 |
| `games_per_iter` | 14 |
| `epochs_per_iter` | 2 |
| `batch_size` | 32 |
| `learning_rate` | 0.001 |
| `mcts_sims` | 25 |
| `mcts_type` | gumbel |
| `board_size` | 8 |
| `max_turns` | 30 |
| `seed` | 42 |
| device | cuda |
| torch | 2.11.0+cu128 |
| training commit | `b14534a5ad84050f8c2583f8f3246643a25d4380` |

## Scope and limitations

This is a reduced-scale MicroCiv comparison (13-channel encoder, 8x8 board), not a full-Civ6 result. Each arm was trained **once**, so training-run variance is not captured: the interval above reflects game sampling only, and a different seed could plausibly reorder the arms. Treat this as evidence that both encoders train and play, plus a bound on how large any difference between them could be at this budget.

**Evaluation status: superseded.** An 800-game head-to-head (2026-08-10, commit `b14534a`, CUDA) scored hex_crystalline 142 W / 588 D / 70 L against patch_embed (score rate 0.545; ternary 95% CI 0.527–0.562, game-sampling noise only). This result is **not evidence that either encoder is better**: both arms were trained once each, with a search implementation later found to back up values with the wrong sign on same-player moves and to train on one-hot policy targets (fixed 2026-09-04), and 280 of the 800 evaluation games reused training seeds. A re-run on disjoint seeds with the corrected search is LF-A9 (milestone v2.3, not v2.1). It waits until `scripts/run_encoder_ab.py` draws its seeds from named streams: re-run as it stands at seed 42, 560 of its 800 arena games would reuse a training map (`kb/specs/20261111-seed-overlap-memo.md` §1.3). v2.1's learning run (A1, `docs/EVALUATION.md` section 7.1) does not compare encoders. The encoder's architecture description is unaffected.
