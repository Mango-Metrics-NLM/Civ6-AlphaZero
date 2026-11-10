| hex_crystalline vs patch_embed | Value |
|---|---|
| Games | 800 |
| W/D/L | 142/588/70 |
| Score rate | 0.545 |
| Wilson 95% CI | [0.510, 0.579] |
| Elo difference | +31.4 |

**Evaluation status: superseded.** An 800-game head-to-head (2026-08-10, commit `b14534a`, CUDA) scored hex_crystalline 142 W / 588 D / 70 L against patch_embed (score rate 0.545; ternary 95% CI 0.527–0.562, game-sampling noise only). This result is **not evidence that either encoder is better**: both arms were trained once each, with a search implementation later found to back up values with the wrong sign on same-player moves and to train on one-hot policy targets (fixed 2026-09-04), and 280 of the 800 evaluation games reused training seeds. A re-run on disjoint seeds with the corrected search is tracked in the public backlog (milestone v2.1). The encoder's architecture description is unaffected.
