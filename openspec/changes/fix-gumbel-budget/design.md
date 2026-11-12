# Design notes -- fix-gumbel-budget

## The schedule

`halving_schedule(k, n, num_legal, *, min_survivors=2)` returns the rounds as a
sequence of `HalvingRound(survivors, simulations)`. Let `k_eff = min(k, num_legal)`.

1. **No search.** If `num_legal <= 1` (a forced move) or `n == 0`, the schedule
   is empty.
2. **Lone candidate.** If `k_eff == 1` and `num_legal >= 2`, the schedule is one
   round of `n` simulations on that candidate, as in `mctx`. This is not a
   forced move: the candidate's searched value feeds the completed-Q target
   over the whole action set.
3. **Halving rounds.** Otherwise there are `R = ceil(log2 k_eff)` rounds.
   - Round `r` visits `m_r` survivors, where `m_0 = k_eff` and
     `m_{r+1} = max(floor, m_r // 2)`.
   - `floor = min(k_eff, max(2, min_survivors))`.
   - Each survivor gets `max(1, n // (R * m_r))` visits in the round.
4. **Budget ceiling.** `n` is a hard ceiling. A round that would overrun it is
   cut short, and any later rounds are dropped. Budget that the `R` rounds
   leave unspent goes to the final round.
5. **Dealing order.** A round is `full_passes = simulations // survivors`
   complete passes over its survivors, dealt in the ranking the round started
   with, followed by `remainder = simulations % survivors` odd visits. Before
   the odd visits the survivors are **re-ranked** by completed Q plus Gumbel
   score, and the top `remainder` get one more visit. Within a full pass the
   order changes no visit count, so only this boundary needs a re-rank.

A reviewer re-implemented `mctx`'s `get_sequence_of_considered_visits` and
compared it for k in 1..32 and n in 1..129. There were 0 mismatches in total
spend, and 0 in the per-rank allocation **assuming a stable ranking**.

That assumption is exactly where the first version of this change diverged
from `mctx`. `mctx` re-scores every candidate before each simulation, so the
odd visits go to the candidates that lead *at that moment*. The first version
dealt them by the ranking frozen at the start of the round, so they went to
whichever candidates led on the previous round's evidence. Together with the
most-visited restriction below, that let a stale leader become the returned
move. With a `one_hot` target at (3, 10), and a root action that looks good on
its first evaluation and bad afterwards, this happened in 47 of 150 runs for
gumbel, 23 for batch and 44 for muzero. Re-ranking before the odd visits
removes it: 0 of 150 in each search, in a test where the odd visit cannot itself
move a Q value. When the odd visit does lower the leader it lands on, the final
ranking can still flip after it; that happened in 3 of 150 runs for gumbel and 1
for muzero. `mctx` behaves the same way, since its final choice is also
restricted to the most-visited candidates.

Rounds with no remainder are unchanged: (8, 24), (16, 64), (8, 30) and the
default evaluation shape (16, 8). For (16, 8) the re-rank happens before any
visit, so it is the Gumbel order itself. On those shapes the search outputs
(policy target and root value) are bit-identical to the version without
re-ranking, for 20 seeds, in all three searches.

## Choosing the move

After the last round, the survivors are ranked by completed Q plus Gumbel
score. The move returned is the best-ranked survivor among those with the
**maximum visit count** (`choose_among_most_visited`).

The restriction matters when the first round is truncated, as it is at
`(16, 8)`, the default evaluation shape. The restriction is safe only because
the odd visits follow current evidence (see "Dealing order"). Without it:

- Unvisited candidates stay in the ranking, with a completed Q equal to the
  root value.
- An adversarial fake (root value +0.9, every leaf -0.9) then returned an
  unvisited move for 5 of 50 seeds.

## Measuring

`SearchStats(root_evals, sim_evals)` is counted at the call sites:

- A root evaluation is counted after the root network call.
- A simulation is counted when it ends in a leaf network evaluation or a
  terminal outcome.

Tests compare these counts with the schedule and with an independent count of
the fake network's calls: the root is the first call's rows, and the
simulations are the rest.

## Forced move

This is the one deviation from `mctx`. With a single legal action no simulation
is spent, so `search` returns `root.value == 0.0`. `training_civ6` discards the
returned value. The MuZero cloud path stores it under a key that nothing reads.
