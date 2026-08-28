# Runbook — Configuring the promotion gate

**Status:** the shipped defaults cannot promote. This runbook explains why, and
what to set instead. Nothing here changes a code default; the recommendation is a
configuration change, made deliberately and reversibly.

## The problem, in one table

Measured evidence from `docs/evidence/hexcrystal_ab/arena_result.json` — 800 games
at exactly the shipped configuration (`board_size=8`, `max_turns=30`), for an arm
independently measured at **+31 Elo**, i.e. a genuine improvement:

| statistic | value | vs the default threshold 0.55 |
|---|---|---|
| draws | 588 / 800 = **73.5%** | — |
| `legacy_win_rate` (the default policy) | 0.1775 | **impossible** — its ceiling is `1 − draw_rate` = 0.265 |
| `score_rate` | 0.5450 | fails |
| `score_rate` 95% CI | [0.5271, 0.5624] | lower bound fails |

Under the default policy the gate is not merely strict, it is *arithmetically
unreachable*: draws are scored as losses, so with a 73.5% draw rate no candidate can
exceed 0.265 however strong it is. Switching to a draw-aware policy fixes that, and
still rejects this model.

## Why 0.55 is the wrong number for a draw-aware policy

0.55 is a sensible **win-rate** threshold, and that is where it came from. Applied to
a **score rate** it silently means something far stricter, because score rate is
compressed toward 0.5 by draws. Converting through the standard Elo relation
`Elo = −400·log₁₀(1/score_rate − 1)`:

| score_rate threshold | implied bar |
|---|---|
| 0.55 | **+34.9 Elo** |
| 0.54 | +27.9 Elo |
| 0.53 | +20.9 Elo |
| 0.52 | +13.9 Elo |

So the shipped 0.55, read as a score rate, demands roughly **+35 Elo from a single
iteration**. That is a bar most individual training iterations will not clear, and
it is not what anyone chose — it is an artifact of reusing a win-rate number for a
different statistic.

## Recommended configuration

```bash
export CIV6AZ_PROMOTION_POLICY=score_rate_lower_bound
export CIV6AZ_PROMOTION_THRESHOLD=0.52
```

Both are read by `PromotionConfig` (`alpha_zero/evaluation/promotion.py`), so they
apply to the deployed orchestrator path without a code change.

The reasoning, so the numbers are defensible rather than fitted:

1. **Pick the Elo bar first.** 0.52 corresponds to **≈ +14 Elo** — a plausible gain
   for one iteration, and low enough that a real improvement is not discarded.
2. **Require significance, not just a point estimate.** `score_rate_lower_bound`
   promotes only when the *lower* bound of the confidence interval clears the bar,
   so a candidate that merely got lucky is held back. It can also return
   `inconclusive` — meaningfully different from a rejection, and reported
   separately.
3. **Sanity-check against the known winner.** The +31 Elo arm has `ci_low = 0.5271`,
   which clears 0.52. The threshold was derived from the Elo intent in step 1 and
   then *checked* against this observation — not reverse-engineered from it.

If you prefer the simpler point-estimate policy, use `score_rate` with the same
threshold; it is less conservative but easier to reason about.

## The upstream problem is the draw rate

A 73.5% draw rate is the real pathology. It compresses every statistic toward 0.5,
which is what makes the threshold so sensitive in the first place. Before spending
much effort tuning the threshold, consider whether `max_turns=30` is simply too
short for MicroCiv games to reach a decisive result — lengthening games would widen
the signal and make promotion decisions more robust than any threshold choice can.

Treat the settings above as the fix for the gate being *unreachable*, and the draw
rate as the thing to actually solve.

## Verifying a change

```bash
# What the gate would decide for a given record, under the configured policy.
CIV6AZ_PROMOTION_POLICY=score_rate_lower_bound CIV6AZ_PROMOTION_THRESHOLD=0.52 \
python3 -c "
from alpha_zero.evaluation.promotion import PromotionConfig, evaluate_promotion
v = evaluate_promotion(wins=142, draws=588, losses=70, config=PromotionConfig())
print(v.decision.value, v.statistic, v.reason)
"
```

The default policy logs a loud warning whenever a record makes promotion
arithmetically impossible (`LegacyWinRatePolicy`, `evaluation/promotion.py`). If you
see that warning in an evaluation log, this runbook is the reason.
