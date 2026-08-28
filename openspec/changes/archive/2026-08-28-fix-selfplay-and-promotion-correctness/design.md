# Design — fix-selfplay-and-promotion-correctness

## Trajectory shape stays a plain dict

A dataclass would give the trajectory a typed identity and make a missing field
impossible at construction. It is still the wrong choice here: shards are
pickled (`cloud/gcp/storage.py`) and read back across a rolling window of past
iterations, so the stored payload must remain interpretable by trainers running
older and newer code at once. Putting a class reference into those pickles makes
the format version-coupled. The helpers give the shape one definition without
changing what reaches disk.

`alpha_zero/distributed/shared_storage.py` already defines a `TrainingExample`
with a first-class `action_mask`, and it is tempting to reuse. It does not fit:
it has no `player` field (the cloud path needs one to sign outcomes per
position), it holds arrays rather than JSON-friendly lists, and adopting it would
change the pickled shape for the same reason.

## Buffering observations rather than appending eagerly

The batch loop read observations, appended them, then searched — so a search
failure left `spatial` one element longer than `policies`. The alternative fix,
truncating all fields to the shortest at finalisation, repairs the symptom each
time it occurs. Buffering and committing atomically makes the ragged state
unrepresentable, and is what the sequential and MuZero paths already did
implicitly by appending only after their search returned.

## Warning rather than promoting `gumbel` to `batch_gumbel`

Honouring an explicit parallelism request by switching the variant is defensible:
`batch_gumbel` is the batched implementation of the same algorithm, and there is
precedent for orchestrator-side rewriting in the MuZero override. It was
rejected because `args.mcts_type` is one namespace shared with the evaluation,
arena and tournament phases. A rewrite intended for self-play would silently
change the search algorithm those phases run — a much larger and less visible
change than the one being fixed.

Correcting the default to 1 does the substantive work: the previous default of 4
was the only reason the ignored-parallelism path was reached by unflagged runs,
and 1 is what those runs already did. Parallelism is now genuinely opt-in, and
asking for it in an unusable combination is loud rather than silent.

## Early stopping derives its statistic from the configured policy

Early stopping previously compared `wins / total` against the promotion
threshold regardless of policy. Under a draw-aware policy that mixes two
statistics: a draw-heavy evaluation could be abandoned as a "decisive loss"
measured one way while the final gate, measuring another, would have promoted
it. Deriving both from the same verdict removes the regime mismatch. Under the
default `legacy_win_rate` policy the statistic is `wins / total`, so the
behavior is unchanged.

## Fixing a path that is not deployed

`cloud/evaluator.py` is not built by the deployment pipeline. The honest
response is not to silently improve it, nor to leave a second promotion rule
lying in the tree: it is to say plainly which path ships, and to make the
non-shipping one consistent with the shipping one so a future revival does not
resurrect a divergent gate. Hence the banners, the docstring correction, and the
registry adoption together — and an explicit statement in the commit message
that this changes nothing in production.

The tri-state decision is preserved rather than collapsed because
`score_rate_lower_bound` can return "inconclusive" — not enough evidence — which
is a different statement from "the candidate lost". Recording only a boolean
would discard exactly the distinction the scoring module exists to express.

## Fog of war is deliberately excluded

The related action-mask leak in `envs/microciv.py` shares this change's theme —
a mask that does not match its information set — but not its risk profile.
Deriving the mask from the fogged state makes `found_city` *over*-permissive
next to a hidden enemy city: the mask offers an action the executor then
refuses. That is a behavior question (what should a player be allowed to attempt
against terrain they cannot see?) rather than a defect fix, it adds cost to the
MCTS hot path, and fog is off by default and unexposed from the training entry
point. It is tracked as its own change so it can be reviewed on those terms.
