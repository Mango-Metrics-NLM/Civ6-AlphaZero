# Memo: completed-Q alignment (LF-A5) and σ in sequential halving (LF-A12)

**Status:** memo complete, revised after review (fix round 1); **decision pending (maintainer)** ·
**Date:** 2026-11-11 (probe evidence generated 2026-11-12 UTC) · **Gap rows:** LF-A5
"Completed-Q alignment memo + decision (CFP S3)", CONFIRM-FIRST; LF-A12 "σ in sequential
halving", memo in 2.1, implementation per the A5 decision
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
  - **Normalisation.** For Go/chess the paper says only that "inside the tree search, the
    Q-values are normalized to be from the [0, 1] interval" (Appendix F). **It does not say
    how.** The only min-max formula in the paper is in §7 (Atari): divide by
    `max(v̂π, max q) − min(v̂π, min q)` over values found inside the tree search. For a board
    game whose values are known to lie in [−1, 1], the conventional [0, 1] map is the fixed
    affine `(q + 1) / 2` (MuZero's pseudocode uses known bounds (−1, 1) for board games). This
    memo therefore reports the paper's setting under **both readings**: *affine* and *per-node
    min-max*.
  - §3.4 and Algorithm 2: sequential halving identifies the action with the highest
    **`g(a) + logits(a) + σ(q̂(a))`**.
  - §4, Eq. 10: `completedQ(a) = q(a)` if `N(a) > 0`, else `v_π` (Appendix D: the `v_mix`
    estimator, which interpolates `v̂π` and the prior-weighted mean of the visited Q values).
  - §4, Eq. 11–12: target `π' = softmax(logits + σ(completedQ))` over **all** actions, loss
    `KL(π', π)`: "this loss trains all actions, not only the action A_{n+1}".
- **`mctx`** (github.com/google-deepmind/mctx, `main`, `mctx/_src/qtransforms.py`):
  - `qtransform_completed_by_mix_value(value_scale=0.1, maxvisit_init=50.0,
    rescale_values=True, use_mixed_value=True, epsilon=1e-8)`: completes unvisited actions with
    `v_mix` (`_compute_mixed_value`), **min-max rescales per node** over the completed vector
    (`_rescale_qvalues`), and multiplies by `(maxvisit_init + max N) · value_scale`. **`mctx`
    pairs min-max with `value_scale = 0.1`, the paper's Atari value.** No published configuration
    pairs per-node min-max with c_scale 1.0.
  - `gumbel_muzero_root_action_selection` passes that transformed vector to
    `seq_halving.score_considered`, so **the halving score contains σ**.
  - `gumbel_muzero_policy`: `action_weights = softmax(prior_logits + completed_qvalues)` over
    every valid action. Interior nodes use `argmax(π' − N / (1 + ΣN))`.

## 3. Code facts (what A1 runs today)

A1's search is `GumbelMCTS` built by `training_civ6.gumbel_search_config`
(`alpha_zero/training_civ6.py:674-705`) with `--gumbel_k 8 --gumbel_sims 24`, every other field
at its default. (The review checked that A1's resolved config and the probe's `current` config
are identical, field for field.)

| # | Item | This repository | Paper (Go/chess) | Same? |
|---|---|---|---|---|
| 1 | `c_visit` | 50 (`core/mcts_constants.py:32`; `training_civ6.py:131`) | 50 | yes |
| 2 | `c_scale` | 1.0 (`utils/constants.py:203`; `training_civ6.py:132`) | 1.0 (0.1 Atari / `mctx` default) | **constant yes; effective scale no** (row 3) |
| 3 | Q normalisation before σ | none: raw Q in [−1, 1] (`core/mcts/gumbel_ops.py`, `build_completed_q_policy`) | to [0, 1], method unstated (affine or per-node min-max) | no |
| 4 | Completion | every action: `q + ((maxN − N)/maxN) · v_root` (`compute_completed_q`) | visited: own `q`; unvisited: `v_mix` | no |
| 5 | Root value in the target | search mean `root.value` (`core/mcts/gumbel.py:413`) | `v̂π` inside `v_mix` | no |
| 6 | Target support | visited actions only (`build_completed_q_policy`) | every legal action | no |
| 7 | **Halving score (A12)** | `g(a) + log π(a)/T + completedQ`, coefficient **1** on Q (`gumbel.py:509-529`; `gumbel_ops.py:290`; T = 1.0 in A1, `training_civ6.py:879`) | `g + logits + σ(q̂)` | **no** |
| 8 | Interior selection | PUCT, `c_puct = c_visit / 50 = 1.0` (`gumbel.py:626`) | `argmax(π' − N/(1 + ΣN))` | no (not probed) |
| 9 | Halving schedule | `gumbel_ops.halving_schedule` (T8) | `mctx` | yes, except a forced move spends 0 sims |
| 10 | Greedy short-circuit | `select_action` greedy when max-p > 0.99 (`gumbel.py:739`; `mcts_constants.py:35`) | none: the move is the halving winner `A_{n+1}` | n/a |

**The σ coefficient per unit of raw Q.** At (k, n) = (8, 24) the schedule is 8×1, 4×2, 2×4
visits, so the final survivors have `max N = 7` and `c_visit + max N = 57`.

| Formula | Coefficient on raw Q (q ∈ [−1, 1]) |
|---|---|
| current target (`build_completed_q_policy`, raw Q) | **57** |
| paper, affine reading: `57 · (q + 1)/2` | **28.5** |
| paper/mctx, per-node min-max, c_scale 1.0 | `57 / range(completedQ)`: unbounded as the spread shrinks |
| `mctx` default (min-max, 0.1) | `5.7 / range(completedQ)` |

So the current target is **twice as sharp per unit of Q spread** as the paper's board-game
target under the affine reading. Under min-max, sharpness depends only on the relative order of
the searched values.

**A12, precisely.** All three Gumbel searches rank survivors by `completed + scores[a]`, with no
σ multiplier: `gumbel.py:528`, `muzero.py:597`, `batch_mcts.py:521`. In A1's search the score is
`g(a) + log π(a) + q(a) + ((maxN_s − N(a))/maxN_s) · v̂`. Here `maxN_s` is the maximum over the
survivors being ranked (`gumbel.py:517-518`), and `v̂` is the network's root value, not the
search mean (`gumbel.py:358, 499, 502`). At each re-rank the survivors have equal visits, so the
completion term is 0 and the ranking reduces to `g + log π + q`. Compared with the paper, the Q
term carries a coefficient of 1 instead of 28.5 per unit of raw Q (affine reading), or instead
of `57 / range` (min-max). The survivors are therefore chosen almost entirely by the
Gumbel-perturbed prior. **The answer to A12 is yes, σ is missing, in all three variants.**

Two things limit how much that matters on A1's path:

- **The halving winner is not the move played.** Self-play samples from the *target*
  (`training_civ6.py:754-755`), and the arena plays `argmax(target)` (`cloud/gcp/arena.py:79-80`).
  The halving decides which candidates get the deeper visits, and so what the target sees; it
  does not pick the move.
- **The target does carry σ.** `build_completed_q_policy` multiplies the completed Q by
  `(c_visit + max N) · c_scale`. So the current target is the paper's formula without
  normalisation, with this repository's completion and support.

**Recorded corrections (docstrings only, no behaviour change).**
- The `build_completed_q_policy` docstring said "the paper uses 0.1" for `c_scale`. That is the
  Atari value and `mctx`'s default; the board-game constant is 1.0, which this repository uses.
  The docstring now also says that the *effective* scale differs, because Q is not normalised.
- The `compute_completed_q` docstring said the paper min-max normalises "over the tree". It now
  says the paper does not state the Go/chess method, documents min-max only for Atari, and that
  `mctx` uses per-node min-max.

**Recorded, outside A1's path (routed to T18 / LF-C4).**
- `BatchGumbelMCTS` declares `policy_target = completed_q` (`core/batch_mcts.py:119`) but
  always returns a one-hot (`batch_mcts.py:336-341`). `git log -S build_completed_q_policy --
  alpha_zero/core/batch_mcts.py` is empty, so the batch path **never** used the builder. The
  `9cd074c` commit message says it does. CLAUDE.md §5.2 (2026-09-04 row, corrected 2026-11-11)
  and §5.3 already record the one-hot and route it to T18.
- `MuZeroMCTS` returns visit counts (`core/mcts/muzero.py:480, 865-887`), the target CFP S3
  calls wrong for sequential halving. Its `_rank_by_completed_q` also reads `child.value` with
  no perspective sign (`muzero.py:594`), consistent with its latent-model alternation fallback.
  Both are worth a line in T18.

## 4. Probe (`scripts/probe_target_sharpness.py`, 0 games)

The probe uses ten random-initialisation seeds (0–7, 42, 1234) × 200 positions from seeded
random play at board 8 / 60 turns, with k = 8, n = 24. That is 2,000 positions, 999 of them
before turn 30 (the temperature phase). The seeds deliberately exclude the registered ones
(1001–1003, 20261111, 20261112).

Two tests pin the probe:
- the probe's `current` target equals `GumbelMCTS.search`'s output, with a negative control;
- the σ transform matches a hand-computed `mctx` `qtransform_completed_by_mix_value` example.

The raw data and the per-seed table are in `docs/evidence/target_sharpness_probe/`.

| Variant | Entropy, mean (nats) | max-p mean / median | max-p > 0.99, pooled | across seeds: SE [min, max] | argmax = `current` |
|---|---|---|---|---|---|
| prior (network, legal actions) | — | 0.09 mean | — | — | — |
| **`current`** (A1 as registered) | 1.378 | 0.410 / 0.289 | **1.9%** | 0.3 pt [0.5%, 3.0%] | 100% |
| `sigma_halving` (A12 alone) | 1.310 | 0.430 / 0.329 | 1.8% | 0.4 pt [0.0%, 3.5%] | 93.3% |
| **`paper_affine`** (Go/chess constants, `(q+1)/2`) | 2.401 | 0.201 / 0.127 | **1.3%** | 0.3 pt [0.0%, 3.0%] | 68.0% |
| `current_search_paper_affine_target` | 2.411 | 0.200 / 0.127 | 1.4% | 0.3 pt [0.0%, 3.0%] | 69.5% |
| **`mctx_minmax_c1`** (per-node min-max, c_scale 1.0) | 0.769 | 0.627 / 0.530 | **38.3%** | 10.7 pt [1.0%, 86.0%] | 53.4% |
| `current_search_mctx_minmax_c1_target` | 0.775 | 0.634 / 0.563 | 41.3% | 11.3 pt [1.0%, 90.5%] | 61.4% |
| `mctx_default` (min-max, c_scale 0.1) | 1.722 | 0.425 / 0.296 | 0.5% | 0.2 pt [0.0%, 1.5%] | 60.2% |

The probe's shares are over all positions. It counts 11 positions with at most one legal
action (0.55% of 2,000), so at most 0.55 points of any share come from them. That is small in
random play but not in self-play: the review's A1-style self-play run found **8.4% forced
positions**, and every above-threshold target there was a forced move (0 of 727 non-forced
positions). A forced move's target is a one-hot whatever the search does, and temperature
cannot matter on it. A13 therefore measures sharpness over **non-forced positions only** (see
finding 4 and §6), and the probe's shares are not directly comparable to A13's.
The across-seed spread of the min-max rows is driven by the network, not by the position set.
The review held positions fixed and varied the network (shares 2–83%), and held the network
fixed and varied positions (63–77%).

What the numbers say:

1. **At A1's iteration 0 the current target is not over-sharp.** With `--bootstrap none` the
   iteration-0 self-play network is exactly this kind of untrained network. 1.9% of positions
   (0.5–3% by seed) trip the greedy short-circuit, so the temperature schedule is live. These
   are random-play positions, so the number is context, not A1's baseline (see §6).
2. **The paper's setting is softer or sharper than the current one, depending on a choice the
   paper does not state.**
   - Under the affine reading, full alignment gives 1.3% above the threshold and entropy
     2.40 nats. That is *softer* than the current target: lower mean max-p at every seed, and
     a share never above the current one. The main reasons are that the support grows from the
     ~8 visited actions to all ~15 legal ones, and that the coefficient per unit of Q halves.
   - Under per-node min-max at c_scale 1.0, the same alignment gives 38.3% near-deltas
     (1–86% by network). Min-max stretches any spread produced by an untrained value head to the
     full 57-logit range.
   - Under `mctx`'s actual default (min-max at 0.1), the figure is 0.5%.
   - Whichever reading applies, aligning with the paper changes the target's argmax on 32–47%
     of positions.
3. **σ in halving alone barely moves the target.** Entropy goes from 1.38 to 1.31 and argmax
   agreement is 93%. σ changes the halving winner more (agreement 56–93% by seed), but the
   winner is not the move played (§3).
4. **Of the candidates, the current target sharpens fastest as the value head learns.** Its
   sharpness scales with the *absolute* spread of the searched Q values (median 0.03–0.30 at
   initialisation), with coefficient 57, twice the affine paper's 28.5. Take a single leader
   ahead of `m − 1` equal-prior competitors by Δ. Max-p reaches 0.99 when
   `57Δ > ln(99(m − 1))`, i.e. **Δ ≳ 0.08 (m = 2) to 0.12 (m = 8)**. A value head that
   separates moves by about a tenth of a game result will push the current target to the
   threshold. The probe cannot measure that.
   - The in-run detector is A13's per-iteration record.
   - It includes `target_share_above_greedy_threshold` and `target_max_p_mean`
     (`alpha_zero/training/diagnostics.py`, `prequential_value_metrics`), beside the target
     entropy. Both are computed over **non-forced positions** (more than one legal action;
     `n_unforced` of them), and the forced share is recorded on its own as
     `forced_move_share`.

## 5. Options

| Option | What changes | Code before T13 | Effect at iteration 0 (probe) | Risk to A1 |
|---|---|---|---|---|
| **A. Register current settings unchanged (default)** | nothing; the divergences in §3 are recorded as deliberate (this closes CFP S3's "or record deliberately") | none | max-p > 0.99 on 1.9% | the current target is the one that sharpens fastest per unit of Q (finding 4); watched by A13's target-sharpness record (non-forced positions) and the §6 trigger, not prevented |
| B. A12 only: σ in the halving score | `gumbel`, `muzero`, `batch_mcts` (+ a shared helper in `gumbel_ops`), CONFIRM-FIRST | yes, with the forced-win oracle and `SearchStats` equality re-run; A16 provenance needs a new field so B-checkpoints are distinguishable | 1.8%; target argmax differs on 6.7% | small effect at n = 24; delays T13 by one PR + review |
| C. Full paper alignment (c_scale 1.0) | B + `v_mix` completion, a [0, 1] normalisation, all-legal support. Changes training targets (CONFIRM-FIRST, CFP S3). **The maintainer must also choose the normalisation, which the paper does not specify** | yes, larger: new negative controls for the target, new provenance fields (target variant, normalisation), the oracle re-validated | **affine: 1.3%**, entropy 2.40, argmax differs on 32%. **Per-node min-max: 38%**, argmax differs on 47% | affine: softer targets and a different search on about a third of moves, with no evidence that it is better at this budget. Min-max: near-delta targets from uninformative values from iteration 0 |
| D. `mctx` defaults (min-max, c_scale 0.1) | as C, with `mctx`'s normalisation and scale | as C | 0.5%; argmax differs on 40% | a different search on 40% of moves; Atari's scale, not the board-game one |
| E. A, plus a pre-registered ablation after A1 | A for A1; then one ≤1,000-game run (v2.2) on a flag-gated B or C, under the same seed protocol, if the §6 trigger fires or A1 does not PASS | none for A1; flags + provenance in v2.2 | — | none to A1 |

## 6. Recommendation

**Option A for A1, with E as its follow-up.** Register the current settings unchanged. Record
in section 7.1 that the Gumbel search diverges from Danihelka et al. in items 3–8 of §3,
deliberately, with this memo as the reference. Reasons:

- The current target is not over-sharp at A1's starting point (1.9% above the threshold, with an
  across-seed SE of 0.3 points), so A1 is not compromised at iteration 0.
- Every alternative is CONFIRM-FIRST across three modules (four for C/D). Each would have to
  land, be reviewed and be re-gated before T13. No alternative has evidence that it is better at
  this budget: B barely moves the target (argmax agreement 93%), and C's effect depends on a
  normalisation the paper does not specify (1.3% or 38%).
- A1 already has a search-vs-loop discriminator (A1-P, PUCT at the same seed), so a FAIL is
  attributable without changing the search first.

The live risk is the one finding 4 names: the current target sharpening mid-run. It is now
measured.
- A13 records `target_share_above_greedy_threshold` and `target_max_p_mean` per iteration, over
  non-forced positions, with `forced_move_share` and `n_unforced` beside them. These are
  recorded, not gated. Excluding forced moves matters: counted in, they would make the share
  track how often moves are forced (about 8% in self-play, and drifting as the policy changes)
  rather than how sharp the targets are.
- T13 should take the **baseline from A1's own iteration 0** (A13 measures self-play positions,
  which differ from the probe's random-play positions). It should quote the probe's 1.9% only as
  context.

**Pre-declared trigger for Option E**, for T13 to register in section 7.1 before A1 runs. All
quantities are the A13 fields in `iterations.jsonl`, which are computed over **non-forced
positions only**.

- **`sᵢ`** is `prequential.target_share_above_greedy_threshold` at iteration `i`, and **`mᵢ`** is
  `prequential.target_max_p_mean`. An iteration whose record carries an `error`, or whose
  value is `null` (no non-forced positions), contributes nothing.
- **`s₀`** is `s` from A1's iteration-0 record (loop index 0) **only if** that record has
  `generator == "network"` (A1 runs `--bootstrap none`) and carries no `error`. **Otherwise
  `s₀ := 0`**, so the 0.25 floor applies.

After A1 finishes, whatever its verdict, check:

- **T1:** `sᵢ > max(0.25, 5 · s₀)` at any iteration `i ≥ 1`; or
- **T2:** `mᵢ > 0.80` at any iteration.

For reference, the non-forced starting values in the review's A1-style self-play were about
0–1.5% (share) and about 0.36 (mean max-p). So the 0.25 floor binds, and `5 · s₀` matters only
if `s₀` exceeds 5%.

If either fires, the Option E ablation is the next experiment. Run B first, because it is the
smaller change; run C only if the maintainer has chosen its normalisation. It is pre-registered
on its own (≤1,000 games), with the same seed protocol and a flag-gated variant. If neither
fires, the ablation is not motivated by target sharpness. A FAIL or INCONCLUSIVE A1 is then
diagnosed through A1-P first.

The trigger is descriptive. It changes nothing during A1 and stops nothing.

**Other items for T13 to record.**
- `MCTS_GREEDY_THRESHOLD` and `MCTS_C_VISIT` override `greedy_threshold` and `c_visit` from the
  environment (`core/mcts_constants.py:32-35`). The run card must require that both are unset,
  or must record their values.

## 7. Decision

**For the maintainer.** Record the choice here (one line: option, date, initials or issue).
CONFIRM-FIRST applies to B, C and D (CFP §9: they change training targets or what every deployed
search computes). **If no decision is recorded when T13 starts, T13 uses Option A — "register
current settings unchanged".**

| Option | Consequence for T13 registration |
|---|---|
| **A — register current settings unchanged (DEFAULT)** | T13 proceeds immediately. Section 7.1 registers the commands of plan §5.2 verbatim and states the divergences of §3 items 3–8 as deliberate, with a link to this memo. It registers the §6 trigger (T1/T2, on the non-forced A13 fields), the definition of `s₀` (A1's own iteration-0 record, `generator == "network"` and no `error`, else `s₀ := 0`), and the requirement that `MCTS_GREEDY_THRESHOLD` / `MCTS_C_VISIT` are unset or recorded. No code change; no new gate. |
| B — σ in halving | T13 waits for a CONFIRM-FIRST PR. The PR adds a shared σ-ranking helper used by `gumbel`, `muzero` and `batch_mcts`. The forced-win oracle and the `SearchStats` equality are re-run with negative controls. A16 provenance gains a halving-score field. Section 7.1 records the new search and the §6 trigger. The probe is re-run on the merged code. |
| C — full paper alignment (c_scale 1.0) | As B, plus the target change (CFP S3's "changes training targets"): `v_mix` completion, a [0, 1] normalisation **named in the decision** (affine or per-node min-max), all-legal support, provenance fields for the target variant and normalisation, and target-width/negative-control tests. Section 7.1 states the iteration-0 consequence under the chosen normalisation: about 1.3% near-deltas with affine, about 38% with min-max. |
| D — `mctx` defaults (min-max, c_scale 0.1) | As C with `mctx`'s normalisation and scale. Section 7.1 records that the scale is the paper's Atari value, not its board-game value. |
| E — A now, ablation later | T13 as Option A. Additionally, a v2.2 pre-registration (≤1,000 games, same seed protocol, flag-gated variant) is added to the plan's §5.4 list, to run if the §6 trigger fires. |

Decision: _(none recorded)_
