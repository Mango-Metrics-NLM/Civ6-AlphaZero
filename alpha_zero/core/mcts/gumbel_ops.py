"""Gumbel-Top-k sampling primitives shared by the Gumbel and MuZero searches.

These live in their own module because :class:`~alpha_zero.core.mcts.muzero.MuZeroMCTS`
uses them as well; keeping them in ``gumbel`` would make the MuZero variant
depend on the Gumbel variant.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from alpha_zero.core.mcts_constants import EPSILON

#: Sequential halving never narrows below two survivors: the last round is a
#: comparison, and a single survivor has nothing to be compared with. Matches
#: ``mctx``'s ``max(2, num_considered // 2)``.
MIN_HALVING_SURVIVORS = 2


@dataclass(frozen=True, slots=True)
class SearchStats:
    """What one ``search()`` call actually spent, measured by the search itself.

    Every Gumbel search exposes the stats of its most recent call as
    ``last_search_stats``; the return types of ``search`` are unchanged.

    Attributes:
        root_evals: Network evaluations of the root position (1 per search).
        sim_evals: Simulations that ended in a leaf evaluation -- a network
            evaluation of a newly reached leaf, or the game's own outcome when
            the simulation reached a terminal state. This is the realised
            simulation budget, and for a sequential-halving search it must
            equal :func:`planned_halving_sims`.
    """

    root_evals: int
    sim_evals: int


@dataclass(frozen=True, slots=True)
class HalvingRound:
    """One round of sequential halving.

    Attributes:
        survivors: How many of the top-ranked candidates this round visits.
        simulations: Simulations spent in this round. They are dealt pass-major
            over the survivors in rank order -- survivor 0, 1, ..., m-1, then
            again -- so when ``simulations`` is not a multiple of ``survivors``
            the higher-ranked survivors receive the extra visit.
    """

    survivors: int
    simulations: int

    def visits(self) -> list[int]:
        """Visits each survivor receives this round, in rank order."""
        base, extra = divmod(self.simulations, self.survivors)
        return [base + (1 if rank < extra else 0) for rank in range(self.survivors)]

    def order(self) -> Iterator[int]:
        """Survivor ranks in execution order (pass-major)."""
        for sim in range(self.simulations):
            yield sim % self.survivors


def halving_schedule(
    k: int,
    n: int,
    num_legal: int,
    *,
    min_survivors: int = MIN_HALVING_SURVIVORS,
) -> tuple[HalvingRound, ...]:
    """The sequential-halving schedule every Gumbel search follows.

    This is the single definition of the schedule. ``GumbelMCTS``,
    ``BatchGumbelMCTS`` and ``MuZeroMCTS`` each iterate it rather than computing
    their own per-round allocation, so their realised budgets are identical by
    construction (and :class:`SearchStats` checks that they are).

    The rule (Danihelka et al., 2022, as implemented by ``mctx``'s
    ``get_sequence_of_considered_visits``):

    * ``k_eff = min(k, num_legal)`` candidates enter.
    * **Forced move.** With ``num_legal <= 1`` the schedule is empty: **no
      simulations are spent**. This is the one deliberate deviation from
      ``mctx``, which spends the whole budget on a lone legal action; searching
      a forced move cannot change the move or its policy target.
    * **Lone candidate.** With ``k_eff == 1`` but two or more legal moves
      (``k = 1``) the move is *not* forced, so -- as in ``mctx`` -- all ``n``
      simulations go to the one candidate, in a single round. Its searched
      value then feeds the completed-Q target over the whole action set.
    * There are ``R = ceil(log2(k_eff))`` halving rounds. Round ``r`` visits
      ``m_r`` survivors, ``m_0 = k_eff`` and
      ``m_{r+1} = max(floor, m_r // 2)`` with ``floor = min(k_eff,
      max(2, min_survivors))``.
    * Each survivor gets ``max(1, n // (R * m_r))`` visits in round ``r``.
    * **Truncation.** The budget is a hard ceiling: a round that would overrun
      ``n`` is cut to what remains (dealt pass-major, so top-ranked survivors
      are visited first) and any later round is dropped.
    * **Leftover.** If the ``R`` rounds spend less than ``n``, the remainder is
      added to the final round, i.e. spent on the final survivors -- exactly
      what ``mctx`` does by continuing to cycle over its last two candidates.

    Consequently the realised total is exactly ``n`` whenever ``num_legal >= 2``,
    and 0 for a forced move; ``min_survivors`` changes the shape of the schedule
    but never its total.

    The move a search *returns* is chosen by :func:`choose_among_most_visited`,
    not by the ranking alone: a truncated round leaves some candidates unvisited.

    Args:
        k: Configured number of Gumbel-sampled candidates.
        n: Configured simulation budget.
        num_legal: Legal actions at the root.
        min_survivors: Lower bound on survivors per round (values below 2 act
            as 2). Searches pass their ``min_actions_after_halving`` here.

    Returns:
        The rounds, in order. Empty when no simulation is to be spent.

    Raises:
        ValueError: If ``k < 1``, ``n < 0`` or ``num_legal < 0``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}")
    if num_legal < 0:
        raise ValueError(f"num_legal must be >= 0; got {num_legal}")

    k_eff = min(k, num_legal)
    if num_legal <= 1 or n == 0:
        return ()
    if k_eff == 1:
        return (HalvingRound(survivors=1, simulations=n),)

    num_rounds = math.ceil(math.log2(k_eff))
    floor = min(k_eff, max(MIN_HALVING_SURVIVORS, min_survivors))
    remaining = n
    survivors = k_eff
    sims: list[int] = []
    sizes: list[int] = []
    for _ in range(num_rounds):
        if remaining == 0:
            break
        per_survivor = max(1, n // (num_rounds * survivors))
        spent = min(remaining, per_survivor * survivors)
        sizes.append(survivors)
        sims.append(spent)
        remaining -= spent
        survivors = max(floor, survivors // 2)
    if remaining:
        sims[-1] += remaining
    return tuple(HalvingRound(m, s) for m, s in zip(sizes, sims, strict=True))


def planned_halving_sims(k: int, n: int, num_legal: int) -> int:
    """Simulations a sequential-halving search spends for ``(k, n, num_legal)``.

    The total of :func:`halving_schedule`: ``n`` when ``num_legal >= 2``, else 0
    (a forced move). It does not depend on a search's ``min_actions_after_halving``,
    which only reshapes the rounds. Every search's
    ``last_search_stats.sim_evals`` must equal this.

    Raises:
        ValueError: As :func:`halving_schedule`.
    """
    return sum(r.simulations for r in halving_schedule(k, n, num_legal))


def choose_among_most_visited(ranked_actions: Sequence[int], visit_counts: Sequence[int]) -> int:
    """The best-ranked action among those with the maximum visit count.

    ``mctx``'s ``score_considered`` restricts the final choice to the actions
    holding the maximum visit count. Without that restriction a truncated round
    -- at ``(k=16, n=8)``, half the candidates are never visited -- leaves
    unvisited candidates in the ranking with a completed Q equal to the root
    value, and one of them can be returned: a move the search never looked at.

    Args:
        ranked_actions: Candidates, best first by completed Q plus Gumbel score.
        visit_counts: Root visit count of each action, aligned with
            ``ranked_actions``.

    Returns:
        The first action in ``ranked_actions`` whose visit count is maximal.

    Raises:
        ValueError: If ``ranked_actions`` is empty or the lengths differ.
    """
    if not ranked_actions:
        raise ValueError("ranked_actions must not be empty")
    if len(ranked_actions) != len(visit_counts):
        raise ValueError(
            f"visit_counts has {len(visit_counts)} entries for {len(ranked_actions)} actions"
        )
    most = max(visit_counts)
    for action, visits in zip(ranked_actions, visit_counts, strict=True):
        if visits == most:
            return int(action)
    raise AssertionError("unreachable: max() is attained")  # pragma: no cover


def gumbel_sample_actions(
    policy: np.ndarray,
    action_mask: np.ndarray,
    num_actions: int,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample actions using Gumbel-Top-k trick.

    The Gumbel-Top-k trick provides a way to sample k elements from a categorical
    distribution without replacement, while maintaining the policy improvement
    guarantee of Gumbel MuZero.

    Args:
        policy: Policy probabilities for all actions (A,).
        action_mask: Mask of valid actions (A,).
        num_actions: Number of actions to sample (k).
        temperature: Temperature for sampling (higher = more exploration).
        rng: Generator for the Gumbel noise. ``None`` draws from a fresh
            unseeded generator, matching the historical behaviour of drawing
            from the global ``np.random`` state.

            This is a *parameter* rather than module state because all three
            searches that call this function -- ``GumbelMCTS``, ``MuZeroMCTS``
            and ``BatchGumbelMCTS`` -- own their own generator, and a caller
            that seeds its search must have that seed reach the noise drawn
            here. It previously did not: ``run_batch_selfplay(seed=...)`` seeded
            the environments while this line drew from the global RNG, so a
            "seeded" self-play run was only half determined.

    Returns:
        Tuple of (selected_actions, gumbel_scores).
        - selected_actions: Indices of selected actions (k,).
        - gumbel_scores: Gumbel scores for selected actions (k,).
    """
    if rng is None:
        rng = np.random.default_rng()
    valid_actions = np.where(action_mask > 0)[0]

    if len(valid_actions) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # Get log probabilities for valid actions
    valid_policy = policy[valid_actions]
    # Ensure numerical stability
    valid_policy = np.clip(valid_policy, EPSILON, 1.0)
    log_probs = np.log(valid_policy)

    # Sample Gumbel noise: g ~ Gumbel(0, 1)
    # Gumbel(0,1) = -log(-log(U)) where U ~ Uniform(0,1)
    uniform = rng.uniform(size=len(valid_actions))
    gumbel_noise = -np.log(-np.log(np.clip(uniform, EPSILON, 1 - EPSILON)))

    # Gumbel-Top-k: score = log(pi) / temperature + g
    scores = log_probs / temperature + gumbel_noise

    # Select top-k by score
    k = min(num_actions, len(valid_actions))
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    selected_actions = valid_actions[top_indices]
    selected_scores = scores[top_indices]

    return selected_actions, selected_scores


def compute_completed_q(
    q_value: float,
    visit_count: int,
    max_visit_count: int,
    root_value: float,
) -> float:
    """Compute completed Q-value with sigma penalty for low visits.

    In Gumbel MuZero, we "complete" Q-values by hallucinating that missing
    visits would have returned the root value. This provides a principled
    way to compare actions with different visit counts.

    Known deviations from Danihelka et al. (2022), recorded, not changed
    --------------------------------------------------------------------
    * **Completion rule.** The paper completes an *unvisited* action's Q with
      the root's value mixture and leaves a *visited* action's Q as its mean
      (eq. 9: ``completedQ(a) = q(a)`` if ``N(a) > 0`` else ``v_pi``). This
      function instead blends every action: ``q + ((maxN - N) / maxN) *
      root_value``, a non-convex combination that leaks root value into
      visited actions in proportion to how far they sit below the best-visited
      one. At equal visit counts (sequential halving's survivors) the two
      agree; elsewhere they do not.
    * **No Q normalisation.** The paper's ``sigma`` is applied to Q values
      min-max normalised over the tree (their Appendix); here raw Q in
      ``[-1, 1]`` is used, so ``c_scale`` acts on a different scale.

    Args:
        q_value: Current Q-value estimate for the action.
        visit_count: Number of visits to this action.
        max_visit_count: Maximum visits among all candidate actions.
        root_value: Value estimate at the root node.

    Returns:
        Completed Q-value.
    """
    if max_visit_count == 0:
        return q_value

    # sigma = fraction of "missing" visits
    sigma = (max_visit_count - visit_count) / max_visit_count

    # Completed Q assumes missing visits would return root value
    completed_q = q_value + sigma * root_value

    return completed_q


def build_completed_q_policy(
    *,
    priors: np.ndarray,
    visit_counts: np.ndarray,
    q_values: np.ndarray,
    root_value: float,
    c_visit: float,
    c_scale: float,
) -> np.ndarray:
    """Return Gumbel MuZero's improved policy ``softmax(logits + sigma(completedQ))``.

    Why not visit counts
    --------------------
    Normalised visit counts are AlphaZero's policy-improvement operator, and they
    are the wrong one here. Sequential halving allocates the budget *equally*
    among the survivors of each round, so the root visit distribution is close to
    uniform over the candidate set by construction and the final survivors are
    typically tied. Measured on a synthetic forced-win position, the visit
    distribution was exactly ``[1/3, 1/3, 1/3]`` while the search had in fact
    identified the winning move: the answer lives in the completed Q values and
    the Gumbel scores, and a visit-count target throws it away.

    Danihelka et al. (2022) define the improved policy over the *completed* Q
    values instead, which is what this builds.

    Known deviations from the paper, recorded here so a reader comparing the
    target against the reference does not have to rediscover them (behaviour
    is deliberately unchanged; each is a separate, measurable decision):

    * ``c_scale`` defaults to **1.0** (``utils/constants.py:GUMBEL_C_SCALE``)
      where the paper uses **0.1** with ``c_visit = 50``. Combined with the
      missing normalisation below, ``sigma`` is ~10x the reference, so the
      searched values dominate the prior sooner than the paper's targets do.
    * **No min-max Q normalisation** before ``sigma`` (see
      :func:`compute_completed_q`).
    * **Non-convex completion** ``q + ((maxN - N) / maxN) * root_value`` for
      visited actions, where the paper keeps a visited action's own Q.
    * **Zero mass on unvisited actions.** The paper's improved policy is a
      softmax over the *whole* action set with unvisited actions completed by
      the root value; this target restricts the softmax to the considered
      set, so an action the search never expanded gets exactly 0, which is a
      harder target for the policy head than the paper's.

    Args:
        priors: Network prior over the whole action space.
        visit_counts: Root child visit counts, aligned with ``priors``. Actions
            the search never considered must be zero.
        q_values: Root child Q values from the acting player's perspective,
            aligned with ``priors``.
        root_value: The root's own value estimate, used to complete the Q value
            of an action the search did not visit.
        c_visit: Visit constant of the monotonic transform.
        c_scale: Scale constant of the monotonic transform. Previously declared
            on the config and read nowhere; this is its intended use.

    Returns:
        A probability vector over the action space. Mass is placed only on
        actions the search considered; if it considered none, an empty (all
        zero) vector is returned and the caller decides the fallback.
    """
    considered = visit_counts > 0
    if not considered.any():
        return np.zeros_like(priors, dtype=np.float32)

    max_visits = float(visit_counts.max())
    completed = np.array(
        [
            compute_completed_q(float(q), int(n), int(max_visits), root_value)
            for q, n in zip(q_values, visit_counts, strict=False)
        ],
        dtype=np.float64,
    )

    # The paper's monotonic transform: larger budgets let the searched values
    # outweigh the prior, which is exactly the behaviour a policy target wants.
    sigma = (c_visit + max_visits) * c_scale
    logits = np.log(np.maximum(priors.astype(np.float64), EPSILON))
    scores = logits + sigma * completed

    # Restrict to the considered set before the softmax: an action the search
    # never looked at has no completed Q to speak of, and including it would let
    # the raw prior leak into a target that is supposed to be search-improved.
    scores = np.where(considered, scores, -np.inf)
    scores -= scores[considered].max()  # shift for numerical stability
    exp = np.where(considered, np.exp(scores), 0.0)
    total = exp.sum()
    if total <= 0.0:
        return np.zeros_like(priors, dtype=np.float32)
    normalised: np.ndarray = (exp / total).astype(np.float32)
    return normalised
