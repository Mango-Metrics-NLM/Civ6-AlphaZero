# Memo: completed-Q alignment (LF-A5) and σ in sequential halving (LF-A12)

**Status:** memo complete; **decision pending (maintainer)** · **Date:** 2026-11-11 ·
**Gap rows:** LF-A5 "Completed-Q alignment memo + decision (CFP S3)", CONFIRM-FIRST; LF-A12 "σ in
sequential halving", memo in 2.1, implementation per the A5 decision
(`kb/specs/20261111-local-first-gap-register.md`) · **Plan:**
`kb/specs/20261111-local-first-and-branching-plan.md` §5.1, §5.5 ("Over-sharp target → the A5
probe"), task T12 · **Evidence:** [`docs/evidence/target_sharpness_probe/`](../../docs/evidence/target_sharpness_probe/report.md)
· **Owner:** ml-scientist · **Feeds:** T13 (A1 registration, `docs/EVALUATION.md` section 7.1).

---

## 1. The questions, from their sources

**A5 (CFP S3).** `kb/specs/20260904-correctness-first-plan.md` §4, row S3:

> Return the visit/completed-Q distribution; remove or gate `GREEDY_THRESHOLD`; align
> `compute_completed_q` with Danihelka et al. (the monotonic σ transform, Q normalisation) — or
> record deliberately that it diverges. **CONFIRM-FIRST** — changes training targets.

The first two clauses landed on 2026-09-04 (`9cd074c`: completed-Q target, `greedy_threshold`
as a config field). What remains open, and is A5, is the third: **align with the paper, or record
deliberately that the code diverges.**

**A12.** Plan §5.1: "σ missing from sequential halving. Memo first; implementation per the A5
decision." The question: does the halving *score* omit σ, where Danihelka et al. rank survivors
by `g + logits + σ(q̂)`?

**Risk to A1** (plan §5.5): an over-sharp target. A target whose max-p exceeds
`greedy_threshold` makes `select_action` ignore temperature, so the self-play temperature
schedule is dead on that move, and the policy head is trained towards a near-delta.

## 2. The reference

Verified on 2026-11-11 against the paper text and the `mctx` source.

- **Danihelka, Guez, Schrittwieser, Silver (2022), "Policy improvement by planning with
  Gumbel", ICLR** (openreview.net/forum?id=bERaNdoegnO):
  - Eq. 8, §3.4: `σ(q̂(a)) = (c_visit + max_b N(b)) · c_scale · q̂(a)`.
  - **Go and chess: `c_visit = 50`, `c_scale = 1.0`** (§3.4; Appendix F: "In all Go and chess
    experiments, Gumbel MuZero scales the Q-values by c_visit = 50 and c_scale = 1.0"). Atari
    (§7): `c_scale = 0.1`.
  - Q is **normalised to [0, 1]** before σ (Appendix F; §7 gives the min-max form over `v̂π` and
    the Q values).
  - §3.4 and Algorithm 2: sequential halving identifies the action with the highest
    **`g(a) + logits(a) + σ(q̂(a))`**.
  - §4, Eq. 10: `completedQ(a) = q(a)` if `N(a) > 0`, else `v_π` (Appendix D: the `v_mix`
    estimator, which interpolates `v̂π` and the prior-weighted mean of the visited Q values).
  - §4, Eq. 11–12: target `π' = softmax(logits + σ(completedQ))` over **all** actions, loss
    `KL(π', π)` — "this loss trains all actions, not only the action A_{n+1}".
- **`mctx`** (github.com/google-deepmind/mctx, `main`):
  - `qtransform_completed_by_mix_value(value_scale=0.1, maxvisit_init=50.0,
    rescale_values=True, use_mixed_value=True, epsilon=1e-8)`: completes unvisited actions with
    `v_mix`, min-max rescales over the completed vector, multiplies by
    `(maxvisit_init + max N) · value_scale`. **Its default `value_scale` is 0.1, the paper's
    Atari value, not the board-game 1.0.**
  - `gumbel_muzero_root_action_selection` passes that transformed vector to
    `seq_halving.score_considered`, so **the halving score contains σ**.
  - `gumbel_muzero_policy`: `action_weights = softmax(prior_logits + completed_qvalues)` over
    every valid action. Interior nodes use `argmax(π' − N / (1 + ΣN))`.

## 3. Code facts (what A1 runs today)

A1's search is `GumbelMCTS` built by `training_civ6.gumbel_search_config`
(`alpha_zero/training_civ6.py:674-705`) with `--gumbel_k 8 --gumbel_sims 24`, every other field
at its default.

| # | Item | This repository | Paper (Go/chess) | Same? |
|---|---|---|---|---|
| 1 | `c_visit` | 50 (`core/mcts_constants.py:32`; `training_civ6.py:131`) | 50 | yes |
| 2 | `c_scale` | 1.0 (`utils/constants.py:203`; `training_civ6.py:132`) | 1.0 (0.1 Atari / `mctx` default) | **yes** |
| 3 | Q normalisation before σ | none: raw Q in [−1, 1] (`core/mcts/gumbel_ops.py:425-427`) | min-max to [0, 1] | no |
| 4 | Completion | every action: `q + ((maxN − N)/maxN) · v_root` (`gumbel_ops.py:342-345`) | visited: own `q`; unvisited: `v_mix` | no |
| 5 | Root value in the target | search mean `root.value` (`core/mcts/gumbel.py:413`) | `v̂π` inside `v_mix` | no |
| 6 | Target support | visited actions only (`gumbel_ops.py:432`) | every legal action | no |
| 7 | **Halving score (A12)** | `completedQ + (log π(a)/T + g(a))`, coefficient **1** on Q (`gumbel.py:509-529`; `gumbel_ops.py:290`; T = 1.0 in A1, `training_civ6.py:879`) | `g + logits + σ(q̂)`, coefficient `(50 + max N)` on q̂ ∈ [0, 1] | **no** |
| 8 | Interior selection | PUCT, `c_puct = c_visit / 50 = 1.0` (`gumbel.py:626`) | `argmax(π' − N/(1 + ΣN))` | no (not probed) |
| 9 | Halving schedule | `gumbel_ops.halving_schedule` (T8) | `mctx` | yes, except a forced move spends 0 sims |
| 10 | Greedy short-circuit | `select_action` greedy when max-p > 0.99 (`gumbel.py:739`; `mcts_constants.py:35`) | none: the move is the halving winner `A_{n+1}` | n/a |

**A12, precisely.** All three Gumbel searches rank survivors by `completed + scores[a]`, with no
σ multiplier: `gumbel.py:528`, `muzero.py:597`, `batch_mcts.py:521`. In A1's search the score is
`g(a) + log π(a) + [q(a) + ((maxN − N(a))/maxN) · v̂]`, where `v̂` is the network's root value
(`gumbel.py:358, 499, 502`). At (k, n) = (8, 24) the schedule is 8×1, 4×2, 2×4 visits, so
`max N = 7` at the end and the paper's coefficient on Q would be **57** on a [0, 1] scale. The
Q term in today's score is therefore ~57× too small relative to `g + log π`: the survivors are
chosen almost entirely by the Gumbel-perturbed prior. **The answer to A12 is yes, σ is missing,
in all three variants.**

Two things limit how much that matters on A1's path:

- **The halving winner is not the move played.** Self-play samples from the *target*
  (`training_civ6.py:754-755`), and the arena plays `argmax(target)` (`cloud/gcp/arena.py:79-80`).
  The halving decides which candidates get the deeper visits, and so what the target sees; it
  does not pick the move.
- **The target does carry σ.** `build_completed_q_policy` multiplies the completed Q by
  `(c_visit + max N) · c_scale` (`gumbel_ops.py:425`). So the current target is the paper's
  formula minus normalisation, with this repository's completion and support.

**Recorded correction.** The `build_completed_q_policy` docstring said "the paper uses 0.1"
for `c_scale`. That is the Atari value and `mctx`'s default; the board-game value is 1.0, which
is what this repository uses. Corrected in this change (docstring only, no behaviour change).

**Recorded, outside A1's path.**
- `BatchGumbelMCTS` declares `policy_target = completed_q` (`core/batch_mcts.py:119`) but
  always returns a one-hot (`batch_mcts.py:336-341`). `git log -S build_completed_q_policy --
  alpha_zero/core/batch_mcts.py` is empty: the batch path **never** used the builder, although
  the `9cd074c` commit message and CLAUDE.md §5.2 (2026-09-04, "Visit counts are AlphaZero's
  improvement operator") say both paths share it. Not on A1's path (`training_civ6` uses
  `GumbelMCTS`); it belongs with T18 / LF-C4.
- `MuZeroMCTS` returns visit counts (`core/mcts/muzero.py:480, 865-887`), the target CFP S3
  calls wrong for sequential halving.

## 4. Probe (`scripts/probe_target_sharpness.py`, 0 games)

Ten random-initialisation seeds (0–7, 42, 1234) × 200 positions from seeded random play at
board 8 / 60 turns; k = 8, n = 24; 2,000 positions, 999 of them before turn 30 (the temperature
phase). Raw data and the per-seed table: `docs/evidence/target_sharpness_probe/`. The seeds
deliberately exclude the registered ones (1001–1003, 20261111, 20261112).

| Variant | Entropy, mean (nats) | max-p mean / median | max-p > 0.99 | … in temperature phase | argmax = `current` |
|---|---|---|---|---|---|
| prior (network, legal actions) | — | 0.09 mean | — | — | — |
| **`current`** (A1 as registered) | 1.378 | 0.410 / 0.289 | **1.9%** | 1.1% | 100% |
| `sigma_halving` (A12 alone) | 1.310 | 0.430 / 0.329 | 1.8% | 0.9% | 93.3% |
| `current_search_paper_target` | 0.775 | 0.634 / 0.563 | 41.3% | 41.8% | 61.4% |
| **`paper`** (Go/chess, c_scale 1.0) | 0.769 | 0.627 / 0.530 | **38.3%** | 39.2% | 53.4% |
| `paper_mctx_scale` (c_scale 0.1) | 1.722 | 0.425 / 0.296 | 0.5% | 0.5% | 60.2% |

Per seed, the paper target's share above the threshold ranges from **1% to 86%**; the current
target's from **0.5% to 3%**. The 11 forced positions (one legal action) are counted in every
row and account for about 0.6 points of each share.

What the numbers say:

1. **At A1's iteration 0 the current target is not over-sharp.** With `--bootstrap none` the
   iteration-0 self-play network is exactly this kind of untrained network. Under 2% of
   positions trip the greedy short-circuit, so the temperature schedule is live.
2. **The paper's formula, at this budget and with an untrained value head, is over-sharp.**
   Min-max normalisation stretches whatever spread the noise in an untrained value head
   produces to the full `(50 + 7) · 1.0 = 57`-logit range, so 38% of targets are near-deltas on
   uninformative values. How often depends on the initialisation, not on any information in
   the values: 1–86% by seed, with no monotone relation to the visited-Q spread (seeds 0, 7 and
   42 — the smallest and the two largest spreads — are all at 1–3%). The paper reports these
   settings with networks that learn; this probe cannot say whether the over-sharpness persists
   once the value head is informative, only that it is present where A1 starts.
3. **σ in halving alone barely moves the target**: entropy 1.38 → 1.31, argmax agreement 93%. It
   changes the halving winner more (agreement 57–93% by seed), but the winner is not the move
   played (§3).
4. **The current target's sharpness is not invariant to training.** It scales with the
   *absolute* spread of the searched Q values (median 0.03–0.30 at initialisation). With
   coefficient 57, a single leader ahead of `m − 1` equal-prior competitors by Δ reaches
   max-p 0.99 at `57Δ > ln(99(m − 1))`, i.e. **Δ ≳ 0.08 (m = 2) to 0.12 (m = 8)**. A trained
   value head that separates moves by a tenth of a game result will push the current target to
   the threshold. The probe cannot measure that; A13's per-iteration target entropy
   (`compute_policy_entropy`, recorded by T10) is the in-run detector.

## 5. Options

| Option | What changes | Code before T13 | Effect at iteration 0 (probe) | Risk to A1 |
|---|---|---|---|---|
| **A. Register current settings unchanged (default)** | nothing; the divergences in §3 are recorded as deliberate (this closes CFP S3's "or record deliberately") | none (the docstring correction is already in) | max-p > 0.99 on 1.9% | over-sharpening later in the run is possible (finding 4) and is watched by A13, not prevented |
| B. A12 only: σ in the halving score | `gumbel`, `muzero`, `batch_mcts` (+ a shared helper in `gumbel_ops`) — CONFIRM-FIRST | yes, with the forced-win oracle and `SearchStats` equality re-run; A16 provenance needs a new field so B-checkpoints are distinguishable | 1.8%; target argmax differs on 6.7% | small benefit at n = 24; delays T13 by one PR + review |
| C. Full paper alignment, c_scale 1.0 | B + `v_mix` completion, min-max normalisation, all-legal support — changes training targets (CONFIRM-FIRST, CFP S3) | yes, larger: new negative controls for the target, new provenance field, the oracle re-validated | **38% near-delta targets**; argmax differs on 47% | trains on over-sharp targets from uninformative values from iteration 0; temperature schedule inert on ~40% of early moves |
| D. Paper alignment with c_scale 0.1 (`mctx` default) | C with 0.1 | as C | 0.5%; argmax differs on 40% | a different search from today's on 40% of moves with no evidence it is better; Atari's setting, not the board-game one |
| E. A, plus a pre-registered ablation after A1 | A for A1; then one ≤1,000-game run (v2.2) on a flag-gated B or C, same seed protocol | none for A1; flags + provenance in v2.2 | — | none to A1 |

## 6. Recommendation

**Option A for A1, with E as its follow-up.** Register the current settings unchanged and record
in section 7.1 that the Gumbel search diverges from Danihelka et al. in items 3–8 of §3,
deliberately, with this memo as the reference. Reasons:

- the probe finds no over-sharpness in the current target at A1's starting point (1.9%), and
  the paper's formula is the over-sharp one at this budget (38%);
- A12 alone changes the target's argmax on only 6.7% of positions, for a CONFIRM-FIRST change
  across three modules that would have to land, be reviewed and re-gated before T13;
- A1 already has a search-vs-loop discriminator (A1-P, PUCT at the same seed), so a FAIL is
  attributable without changing the search first.

Two cheap additions, both autonomous: (1) section 7.1 records the expected iteration-0 share
(about 2%) next to A13's target entropy, so a drift towards the threshold during A1 is visible
against a baseline; (2) A13 records the **share of targets with max-p > `greedy_threshold`** per
iteration, beside the entropy it already records (descriptive, not gated).

If A1 FAILs or is INCONCLUSIVE and A13 shows the target collapsing towards deltas, Option E's
ablation is the next experiment — B first (smaller), C only with a normalisation floor, since
min-max on an untrained value head is what makes C over-sharp.

## 7. Decision

**For the maintainer.** Record the choice here (one line: option, date, initials or issue).
CONFIRM-FIRST applies to B, C and D (CFP §9: they change training targets or what every deployed
search computes). **If no decision is recorded when T13 starts, T13 uses Option A — "register
current settings unchanged".**

| Option | Consequence for T13 registration |
|---|---|
| **A — register current settings unchanged (DEFAULT)** | T13 proceeds immediately. Section 7.1 registers the commands of plan §5.2 verbatim, states the divergences of §3 items 3–8 as deliberate with a link to this memo, and records the probe's iteration-0 baseline (1.9% of targets above `greedy_threshold`). No code change; no new gate. |
| B — σ in halving | T13 waits for a CONFIRM-FIRST PR: shared σ-ranking helper used by `gumbel`, `muzero`, `batch_mcts`; forced-win oracle and `SearchStats` equality re-run with negative controls; A16 provenance gains a halving-score field. Section 7.1 records the new search. The probe is re-run on the merged code. |
| C — full paper alignment (c_scale 1.0) | As B, plus the target change (CFP S3's "changes training targets"): `v_mix` completion, min-max normalisation, all-legal support, a provenance field for the target variant, target-width/negative-control tests. Section 7.1 must state that iteration-0 targets are near-deltas on ~38% of positions and that the temperature schedule is inert there. Not recommended. |
| D — paper alignment, c_scale 0.1 | As C with the scale changed; section 7.1 records that the scale is the paper's Atari value, not its board-game value. |
| E — A now, ablation later | T13 as Option A. A v2.2 pre-registration (≤1,000 games, same seed protocol, flag-gated variant) is added to the plan's §5.4 list. |

Decision: _(none recorded)_
