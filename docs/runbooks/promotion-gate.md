# Runbook — Configuring the promotion gate

**Status:** the shipped defaults cannot promote, and that is deliberately
unchanged. This runbook explains why, separates the two questions a gate has to
answer — *is the candidate better than the champion at all?* and *by how much?*
— and gives the sample sizes each one needs. Nothing here changes a code
default: flipping `DEFAULT_POLICY` or `DEFAULT_PROMOTION_THRESHOLD` is a
CONFIRM-FIRST decision (CLAUDE.md §3.2) with a recorded rationale, not a
runbook edit.

## 1. The problem, in one table

Measured evidence from `docs/evidence/hexcrystal_ab/arena_result.json` — 800
games at the shipped configuration (`board_size=8`, `max_turns=30`):

| statistic | value | vs the default threshold 0.55 |
|---|---|---|
| draws | 588 / 800 = **73.5%** | — |
| `legacy_win_rate` (the default policy) | 0.1775 | **impossible** — its ceiling is `1 − draw_rate` = 0.265 |
| `score_rate` | 0.5450 | fails |
| `score_rate` 95% CI | [0.5271, 0.5624] | lower bound fails |
| implied Elo (95% CI) | +31 [+19, +44] | — |

Under the default policy the gate is not merely strict, it is *arithmetically
unreachable*: draws are scored as losses, so with a 73.5% draw rate no
candidate can exceed 0.265 however strong it is. `evaluate_promotion` now logs
the arithmetic beside every verdict (`promotion feasibility [...]`) and warns
when the gate is under-powered for the record it was given.

## 2. Two different questions

A single threshold conflates them. Keep them apart.

**(a) Superiority at parity.** "Is the candidate better than the champion?" is
a one-sided test of `score_rate > 0.5`. It has a *sample-size* answer: with `n`
games at draw rate `d`, the smallest true edge detectable at 80% power
(one-sided α = 0.05) is

```
MDE(n, d) = (z_0.05 + z_0.80) · sqrt((1 − d) / 4) / sqrt(n)
          = 2.486 · sqrt((1 − d) / 4) / sqrt(n)
```

(`scoring.minimum_detectable_effect`; the variance is the *ternary* one, since
a per-game score is in {0, ½, 1}, not Bernoulli). Read the table in §3: pick
`n` from the edge you need to see, not the other way round.

**(b) The Elo bar.** "Is it better by *enough* to be worth promoting?" is a
threshold on the score rate, converted through the standard relation

```
Elo = −400 · log10(1 / score_rate − 1)        (scoring.elo_from_score_rate)
```

| score_rate threshold | implied bar |
|---|---|
| 0.55 | **+34.9 Elo** |
| 0.54 | +27.9 Elo |
| 0.53 | +20.9 Elo |
| 0.52 | +13.9 Elo |

So the shipped 0.55, read as a score rate, demands roughly **+35 Elo from a
single iteration**. That number was never chosen; it is a win-rate constant
inherited from AlphaGo Zero (where Go with komi is draw-free) applied to a
different statistic.

A threshold above 0.5 is not a superiority test: a gate has zero power *at* its
own threshold by definition, so a bar of 0.52 needs the candidate to be well
above +14 Elo before it is seen to clear it. Decide the bar from what one
iteration is expected to deliver, then size `n` from (a) for that edge.

## 3. Sample sizes

`MDE` at 80% power (one-sided α = 0.05) and the implied Elo of `0.5 + MDE`, by
game budget `n` and draw rate `d`. Produced by
`scoring.minimum_detectable_effect` and `scoring.elo_from_score_rate`; the
0.735 column is the measured draw rate at the shipped defaults.

| n | d = 0.30 | d = 0.50 | d = 0.735 |
|---:|---|---|---|
| 20 | 0.233 (+175 Elo) | 0.197 (+144 Elo) | 0.143 (+102 Elo) |
| 50 | 0.147 (+105) | 0.124 (+88) | 0.091 (+64) |
| 100 | 0.104 (+73) | 0.088 (+62) | 0.064 (+45) |
| 200 | 0.074 (+51) | 0.062 (+43) | 0.045 (+32) |
| 400 | 0.052 (+36) | 0.044 (+31) | 0.032 (+22) |
| 800 | 0.037 (+26) | 0.031 (+22) | 0.023 (+16) |
| 1500 | 0.027 (+19) | 0.023 (+16) | 0.017 (+11) |

Games needed to detect a true **+14 Elo** (`score_rate` 0.52) at 80% power
(`scoring.games_for_score_rate(0.52, d)`):

| d = 0.30 | d = 0.50 | d = 0.735 |
|---:|---:|---:|
| 2,705 | 1,933 | 1,024 |

Two things the table says that intuition does not:

* **Draws make the test *cheaper*, not dearer.** On the score scale a draw sits
  at the mean, so it lowers the per-game variance; a 73.5% draw rate needs
  fewer games for the same MDE than a 30% one. (It still caps `win_rate`,
  which is why `win_rate` cannot be the statistic.)
* **The deployed evaluation budget cannot answer either question.** At 20 games
  (`--evaluation_games` default) the smallest visible edge is about +100 Elo;
  at 400 games it is about +22. A REJECT at `n = 20` is not evidence of no
  improvement — it is the absence of a measurement, which is what the
  `inconclusive` state exists to say.

`PromotionConfig.feasibility(n_games, expected_draw_rate)` computes the row for
any configuration, and `evaluate_promotion` logs it at INFO on every call,
WARNING when the implied Elo exceeds `CIV6AZ_PROMOTION_MAX_IMPLIED_ELO`
(default 50).

## 4. Recommended configuration

```bash
export CIV6AZ_PROMOTION_POLICY=score_rate_lower_bound
export CIV6AZ_PROMOTION_THRESHOLD=0.52
```

Both are read by `PromotionConfig` (`alpha_zero/evaluation/promotion.py`), so
they apply to the deployed orchestrator path without a code change.

The reasoning, so the numbers are defensible rather than fitted:

1. **Pick the Elo bar first.** 0.52 corresponds to **≈ +14 Elo** — a plausible
   gain for one iteration, and low enough that a real improvement is not
   discarded.
2. **Require significance, not just a point estimate.** `score_rate_lower_bound`
   promotes only when the *lower* bound of the confidence interval clears the
   bar, so a candidate that merely got lucky is held back. It can also return
   `inconclusive` — meaningfully different from a rejection, and reported
   separately.
3. **Size the budget from §3, not from habit.** For the bar to be clearable the
   evaluation needs on the order of 1,000 games at the measured draw rate;
   `feasibility()` prints the exact `min_observed_score` the policy needs for
   the budget you actually have.

If you prefer the simpler point-estimate policy, use `score_rate` with the same
threshold; it is less conservative but easier to reason about.

A note on what this runbook no longer says: an earlier version "sanity-checked"
the 0.52 threshold against the +31 Elo A/B arm (`ci_low = 0.5271` clears 0.52).
That check is circular — the arm is the only measurement there is, so a
threshold tuned until it admits the arm has been fitted to it, not validated by
it — and it has been dropped. The threshold stands on the Elo intent in step 1
and the power arithmetic in §3 alone.

## 5. Why the code default is unchanged

`DEFAULT_POLICY = "legacy_win_rate"` and `DEFAULT_PROMOTION_THRESHOLD = 0.55`
reproduce the deployed gate exactly, defect included. They stay because:

* changing them changes *which models get promoted* in a running pipeline,
  which CLAUDE.md §3.2 lists as CONFIRM-FIRST (distributed training
  coordination), and the GCS champion lineage (v0–v268) was selected under the
  old rule;
* the right replacement depends on a decision not yet recorded — the Elo bar
  per iteration and the game budget that can see it — and `WS-C` in
  `kb/specs/20260902-validated-learning-plan.md` is the run meant to produce the
  evidence for that decision;
* an operator can opt in today with two environment variables, so nothing is
  blocked on the default.

The `legacy_win_rate` policy warns loudly whenever a record makes promotion
arithmetically impossible, so the defect cannot pass unnoticed while the
default stands.

## 6. The upstream problem is the draw rate

A 73.5% draw rate compresses every statistic toward 0.5 and is the real
pathology. `docs/evidence/draw_rate_sweep/` measured it against the turn limit
(0.68 / 0.47 / 0.37 draws at 30 / 60 / 100 turns on board 8) and found a
structural ceiling: cities cannot be captured, so once both players found, a
1–1 tie is the modal result. Treat the settings above as the fix for the gate
being *unreachable*, and the draw rate as the thing to actually solve. Note
also that the 2026-09-04 search-correctness fixes (`core/mcts/backup.py`,
completed-Q policy target) change how a given checkpoint plays, so the draw
rate should be re-measured on post-fix code before any threshold is retuned
against it.

## 7. Verifying a change

```bash
# What the gate would decide for a given record, under the configured policy,
# and what that budget could have detected.
CIV6AZ_PROMOTION_POLICY=score_rate_lower_bound CIV6AZ_PROMOTION_THRESHOLD=0.52 \
python3 -c "
from alpha_zero.evaluation.promotion import PromotionConfig, evaluate_promotion
cfg = PromotionConfig()
v = evaluate_promotion(wins=142, draws=588, losses=70, config=cfg)
print(v.decision.value, round(v.statistic, 4), v.reason)
print(cfg.feasibility(800, 0.735).describe())
"
```

A benchmark under a fixed protocol (paired seats, named map seeds, draw-aware
statistics, provenance envelope) is available as
`python scripts/benchmark_suite.py --checkpoint PATH --seed N --out DIR`
(`alpha_zero/evaluation/benchmark.py`); its `report.md` carries the MDE for
the budget played, so a number is never quoted without its resolution.
