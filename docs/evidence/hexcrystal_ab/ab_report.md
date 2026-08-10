# Encoder A/B evidence: patch_embed vs hex_crystalline

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
