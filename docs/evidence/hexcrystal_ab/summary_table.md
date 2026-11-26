| hex_crystalline vs patch_embed | Value |
|---|---|
| Games | 800 |
| W/D/L | 142/588/70 |
| Score rate | 0.545 |
| Wilson 95% CI | [0.510, 0.579] |
| Elo difference | +31.4 |

**Evaluation status: superseded.** An 800-game head-to-head (2026-08-10, commit `b14534a`, CUDA) scored hex_crystalline 142 W / 588 D / 70 L against patch_embed (score rate 0.545; ternary 95% CI 0.527–0.562, game-sampling noise only). This result is **not evidence that either encoder is better**: both arms were trained once each, with a search implementation later found to back up values with the wrong sign on same-player moves and to train on one-hot policy targets (fixed 2026-09-04), and 280 of the 800 evaluation games reused training seeds. A re-run on disjoint seeds with the corrected search is LF-A9 (milestone v2.3, not v2.1). It waits until `scripts/run_encoder_ab.py` draws its seeds from named streams: re-run as it stands at seed 42, 560 of its 800 arena games would reuse a training map (`kb/specs/20261111-seed-overlap-memo.md` §1.3). v2.1's learning run (A1, `docs/EVALUATION.md` section 7.1) does not compare encoders. The encoder's architecture description is unaffected.
