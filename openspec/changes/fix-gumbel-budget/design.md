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
5. **Dealing order.** Visits are dealt pass-major in rank order, so a partial
   pass favours the higher-ranked survivors.

A reviewer re-implemented `mctx`'s `get_sequence_of_considered_visits` and
compared it for k in 2..32 and n in 1..129. There were 0 mismatches, both in
total spend and in the per-rank allocation.

## Choosing the move

After the last round, the survivors are ranked by completed Q plus Gumbel
score. The move returned is the best-ranked survivor among those with the
**maximum visit count** (`choose_among_most_visited`).

The restriction matters when the first round is truncated, as it is at
`(16, 8)`, the default evaluation shape. Without it:

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
