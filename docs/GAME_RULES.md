# Game rule sets

This page is the home of the game rule sets the training environment can be built
with, the log that pins each one, and the protocol a new rule set must pass before
anyone trains on it. How strength is measured is defined in
[`EVALUATION.md`](EVALUATION.md); evidence artifacts follow
[`evidence/SCHEMA.md`](evidence/SCHEMA.md).

> **Status (2026-11-20): the rules kernel exists; no training environment uses it yet.** The program is
> [`kb/specs/20261119-game-mechanics-plan.md`](../kb/specs/20261119-game-mechanics-plan.md).
> Slices GM-K1/K2 added `alpha_zero/rules/` (rule sets as validated, hashed data) with
> `python -m alpha_zero.rules validate`, `hash`, `pace` and `import-gameinfo`, and one calibration draft,
> `science-v1-draft`. The live Civ6 bridge (`alpha_zero/bridge/tech_registry.py`, GM-B3, off the play path)
> consumes it for opt-in research slot indexing, while no training environment reads it yet and the log
> below stays empty until a frozen preset arrives with GM-S0.

## 1. What a game rule set is

A game rule set is a named, versioned set of MicroCiv rules, selected by one
setting, `gameplay_preset` (the term `docs/EVALUATION.md` section 7.1 registers).
"Ruleset" alone means GitHub branch rulesets in this repository, so this page says
"game rule set".

- **`default` is the only registered rule set today.** The A1 protocol fixes it,
  and it must stay byte-identical.
- **No other rule set exists yet.** `growth-v1` (GM-S1), `science-v1` (GM-S2) and
  `warfare-v1` (GM-S3) are planned in the spec. Each one needs its own OpenSpec
  change, confirmed when it is proposed.
- **Identity.** A rule set is `(id, sha256 of its canonical parsed form)`. The
  form is canonical so that line endings and key order cannot change the hash.
- **A rule change to a frozen rule set is a new id** (for example `science-v2`),
  never an edit under an old one.
- **Drafts.** A calibration draft is a built-in whose id ends in `-draft` (for
  example `science-v1-draft`, the first-pass tree of the plan). Its hash is
  pinned and checked on load, but it may be re-pinned, it is never logged, and
  nothing confirmatory runs on it. Freezing creates a new entry under the
  target id.

## 2. Rule-set log

Append-only, newest last. A row is added when a rule set is frozen for
confirmation (section 3, step 3), whatever the confirmation then shows. The log
has no row while only drafts exist.

| Id | Content sha256 | Date | What changed | OpenSpec change |
|---|---|---|---|---|

Rules for appending:

1. A rule change is a new id and a new row; an existing row is never edited or
   removed, and a pinned hash is never edited.
2. The id and hash in a row equal the pinned values of a frozen entry in the
   kernel's builtin table, and every frozen entry has a row. Once the kernel
   lands (GM-K1), a test holds this log equal to those entries, as
   `tests/regression/test_a1_registration_digest_regression.py` does for section 7.1.
3. Calibration drafts (ids ending in `-draft`) are not logged. A rule set enters
   the log when it is frozen for confirmation (section 3, step 3), under an id
   without the suffix. If its confirmation fails, the row stays, and the probe
   verdict lives in its evidence artifact.

## 3. Probe protocol (pre-registration template)

Copy this section into the rule set's own pre-registration (a dated file named in
its OpenSpec change), fill in every bracket, and commit it **before** step 4.

**Steps.** (1) Zero-game pacing model (`python -m alpha_zero.rules pace`, once it
exists). (2) Exploratory calibration sweeps on calibration seeds only, tuning
costs one factor at a time, loading each candidate rule set by path and recording
its content sha256 in the sweep's artifact; the scripted agent may change here
too. (3) Freeze: add a frozen built-in entry under the target id (for example
`science-v1`) with its pinned hash, freeze the scripted agent's code, append the
log row, commit this template filled in. (4) One confirmation run on held-out seeds. Nothing is tuned
after step 3.

**Hypothesis.** Rule set `[id]` at `[hash]` (board `[8]`, `max_turns` `[60]`) is
calibrated: a scripted strategy that uses the new mechanic wins decisively and on
schedule, random play rarely finishes it and rarely draws, and the seats are
balanced.

**Metric.** The draw-aware score rate of `docs/EVALUATION.md` section 1, with the
iid ternary interval, the pair-clustered interval and their hull, as the benchmark
writes them. Draw and capstone rates are proportions with a pair-bootstrap
interval (the seat-swapped pair as the unit). The seat-0 score is derived from
the per-game `{pair, seat, result}` records. No tool computes it today, and the
rung that first runs a probe must add one, because the paired design cancels
seat bias out of the headline score.

**Sample sizes.** Fixed in advance, each card priced with
`scripts/run_card.py CARD --price_only` and under the 1,000-game cap:
random against random, 600 games (300 pairs); scripted agent against random,
3 seeds x 200 games. The total games of the step-2 sweeps are reported.

**Seeds.** Calibration seeds and confirmation seeds are disjoint and both listed
here before step 2: calibration `[3001-3099]`, confirmation `[one seed per card
from 5001-5999]`. None may be 1001-1003, 20261111 or 20261112 (section 7.1). A
confirmation seed that has been run is spent.

**Bands** (all that apply must hold; reading the intervals at the stated levels, in pairs).
B1 and B4 apply only to a rule set that declares a capstone. A rule set without one (for
example `growth-v1`) must pre-register its own pacing criterion in their place, in the same
file and before step 4; it may not simply drop them:

| Band | Statistic | Passes when |
|---|---|---|
| B1 pacing | scripted agent against random: median turn it completes the capstone | its 95% pair-bootstrap interval lies inside [0.60, 0.70] x `max_turns`; a game where the capstone is not reached counts as `max_turns` |
| B2 positive control | scripted agent against random, score rate | 95% hull lower bound >= 0.90 |
| B3 draws | random against random, draw rate | 95% upper bound <= 0.10 |
| B4 too easy | random against random, share of games that complete the capstone | 95% upper bound <= 0.05 |
| B5 seat balance | random against random, seat-0 score rate | equivalence: its 90% pair-clustered interval lies inside [0.44, 0.56] |

What these sizes can decide (iid figures; clustering widens them): at 600 games a
true draw rate near 0.065, or a true capstone rate of 0.025, passes with about
85% probability; B2 needs a true score near 0.93; B5 has about 85% power at true
balance. A band that sits nearer its threshold than that is undecidable here; a
wider band or a larger sample (with an approval reference above 1,000 games) is a
new pre-registration.

**Stopping rule.** No interim looks and no early stop. An interrupted run, a
determinism failure or a single executor rejection makes the confirmation VOID.
After a VOID, fix the cause and run again on the same seeds, and report the VOID.
Each frozen hash gets one confirmation.

**Outcomes.**

- **PASS**: every band holds, and the rule set is calibrated.
- **NOT CALIBRATED**: any band's bound lands on the wrong side. Name the bands.
  - B2 means the rules do not reward the scripted strategy, or the script is
    wrong.
  - B3 or B4 means random play draws too often or wins too easily.
  - B5 means the seats are not balanced.
  - Re-tuning returns to step 2 with fresh confirmation seeds, and any rule
    change is a new id.
- **VOID**: as in the stopping rule.

**Refutation.** The hypothesis is refuted by NOT CALIBRATED. It is not supported
by a PASS on seeds that were used in calibration.

**Exploratory versus confirmatory.** Steps 1 and 2 are exploratory, are never
quoted as the verdict, and may run on a GPU. Step 4 is confirmatory: seeded,
deterministic, and run on CPU unless the pre-registration names devices whose
determinism check has passed.

**Reporting.** Report the result whichever way it comes out, as an artifact under
`docs/evidence/` with the `evidence` block (`search_epoch: post-2026-09-04`)
and a row in [`evidence/README.md`](evidence/README.md). The block has no
rule-set field yet, so `claims` must name the preset id and hash.

## 4. Relation to the A1 protocol

- Probes and their pre-registrations are never written between the A1 markers of
  `docs/EVALUATION.md`, and never reuse its seeds.
- No training card for a rule set other than `default` runs before A1 and A1-R
  are recorded. Such a card is confirmatory only if both PASS. Otherwise it is
  exploratory, and it says so.
- After an A1 FAIL, the registered follow-ups are A1-P and the target-sharpness
  trigger (section 7.1, "Follow-ups"). Running a different rule set (for example
  `growth-v1`) as a remedy is a separate question with its own pre-registration,
  made before A1's verdict is seen. Its result cannot change A1's verdict.

## 5. Claims and non-claims

A passed probe licenses only this sentence: "rule set `[id]` at `[hash]` is
calibrated at board `[b]` / `[t]` turns: a scripted strategy beats random play at
score `[s]` `[hull]`, random play draws `[d]`, and the seats are balanced within
±0.06".

It does **not** license saying:

- that any agent learns the new mechanic. A scripted control shows the rules
  reward a strategy, not that self-play finds it within 1,000 games;
- anything about learning at all. That needs A1 and A1-R to PASS first;
- anything about Civilization VI. A micro rule set is an original abstraction,
  not the real game;
- anything about other board sizes, turn limits or rule-set ids.
