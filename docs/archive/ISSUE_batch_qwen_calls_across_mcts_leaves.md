# `ChessAdvisor` issues N sequential Qwen calls per MCTS move when it could issue 1

## Summary

`ChessAdvisor.prior_for_board` is called once per MCTS leaf expansion via
`Civ6MCTS._blend_llm_prior(...)`. With `num_simulations=64` and a typical
opening branching factor, MCTS explores ~30–60 unique FENs per agent
move. The advisor's per-FEN LRU cache only catches *revisits within the
search tree*, not first-time encounters, so the agent ends up making
30–60 sequential 1–3-second Qwen requests per move — wall-clock 30–180
seconds per agent move just for LLM I/O. End-to-end this means a single
chess game can take **2+ hours on CPU** even though the MCTS itself
takes 13 CPU-seconds (verified empirically: 7,187 wall seconds vs 13.4
CPU seconds during a real run).

The fix is to batch Qwen requests across all leaf expansions of a single
`mcts.search(env)` call so 64 simulations issue 1 batched request
instead of up to 64 sequential ones. LM Studio (and the OpenAI Chat
Completions API generally) support concurrent requests, and Qwen-14B
inference latency is not strictly proportional to batch size when
batched on the model server, so this is roughly a 30× wall-clock
speedup for the opening phase (the only place the advisor is consulted,
per `max_ply_for_prior=24`).

## Evidence / measurement

A 2-hour real run with `num_simulations=64`, `max_ply_for_prior=24`:

```
Process: scripts/run_chess_training.py
StartTime: 17:40:54
Wall clock at snapshot: ~19:58:00 (2h 17m elapsed)
Total CPU: 13.4 seconds
Active TCP connection to LM Studio at port 1234: 1 Established + 6 TimeWait
PGN files written: 0
```

CPU÷wall = 0.16% busy. The agent process is blocked on I/O essentially
the entire time. Each TimeWait socket represents a recently completed
Qwen request, suggesting steady (sequential) request issuance.

## Where the fix lives

Two cooperating changes:

1. **`ChessAdvisor.priors_for_boards(boards: list[chess.Board]) -> list[np.ndarray | None]`** —
   batched API that issues one request per unique uncached FEN, in
   parallel via `asyncio.gather` over `LLMClient.generate(...)`, and
   returns priors in the same order as the input.
2. **`Civ6MCTS.search(env)`** — accumulate leaf-expansion candidates
   inside the simulation loop, then prior-blend them in a single
   `priors_for_boards([...])` call at the end of the simulation budget.
   This requires restructuring the inner loop so prior application is
   deferred until all simulations have selected their leaves; today
   it's eager.

The deferred-prior restructure is the bigger of the two changes. A
simpler intermediate is **batched at the level of one `mcts.search`
call only** — every leaf expansion *within a single search* gets
batched together, even if the order of leaf evaluation changes
slightly. This trades a tiny bit of MCTS exploration shape for a
massive latency win and is the right first step.

## Proposed API

```python
# alpha_zero/orchestrator/chess_advisor.py


class ChessAdvisor:
    async def priors_for_boards_async(self, boards: list[chess.Board]) -> list[np.ndarray | None]:
        """Return priors for many boards. Cached entries return synchronously;
        uncached entries are fetched concurrently via asyncio.gather."""
        ...

    def priors_for_boards(self, boards: list[chess.Board]) -> list[np.ndarray | None]:
        """Synchronous wrapper for callers in non-async contexts (MCTS)."""
        return asyncio.run(self.priors_for_boards_async(boards))
```

```python
# alpha_zero/core/civ6_mcts.py — Civ6MCTS.search restructure


def search(self, env, add_noise=True):
    root = MCTSNode()
    leaf_envs: list = []  # accumulator for batched prior fetch
    leaf_nodes: list[MCTSNode] = []
    leaf_policies: list = []
    leaf_masks: list = []

    # ... existing root expansion (uses single-board prior, fine to keep) ...

    for _ in range(self.config.num_simulations):
        # ... selection / leaf reach as before, but DON'T expand yet
        # if it's a non-terminal leaf: defer expansion
        if not (terminated or truncated):
            obs = sim_env.get_observation()
            policy, value = self._evaluate(obs["spatial"], obs["vector"], obs["action_mask"])
            leaf_envs.append(sim_env)
            leaf_nodes.append(node)
            leaf_policies.append(policy)
            leaf_masks.append(obs["action_mask"])
            # backprop a placeholder value here OR defer backprop too
        # else: backprop terminal value as before

    # Single batched prior fetch for ALL leaves of THIS search
    if self._llm_prior_fn is not None and leaf_envs:
        priors = self._llm_prior_fn_batch(leaf_envs, leaf_masks)  # NEW
        for node, base_policy, mask, prior in zip(leaf_nodes, leaf_policies, leaf_masks, priors):
            blended = self._blend_with_prior(base_policy, mask, prior)
            node.expand(blended, mask)
    else:
        for node, policy, mask in zip(leaf_nodes, leaf_policies, leaf_masks):
            node.expand(policy, mask)

    # ... existing visit-count → action_probs ...
```

The `_llm_prior_fn_batch` is either a new attribute on `Civ6MCTS`
constructor (preferred — explicit API), or sniffed from the existing
`_llm_prior_fn` callable (worse — relies on duck typing).

## Expected impact

Conservative estimate, with a 14B model running on a single GPU
(estimated 200 ms per call when batch=1, ~400 ms for batch=32 due to
model-server concurrency overhead):

| Setting | Sequential | Batched |
|---|---|---|
| 64 sims × 30 unique FENs × 1.5 s ea (CPU LLM) | ~45 s/move | ~3 s/move |
| 64 sims × 30 unique FENs × 200 ms ea (GPU LLM) | ~6 s/move | ~0.5 s/move |

Times 30 moves per game → game time goes from ~22 min to ~1.5 min on
CPU, or from ~3 min to ~15 s on GPU. The relative speedup is the same
either way; the absolute numbers depend on whether LM Studio is GPU- or
CPU-backed.

## Acceptance criteria

- [ ] `ChessAdvisor.priors_for_boards(...)` exists and passes a unit test
      that verifies parallel issuance (mock `LLMClient.generate` with a
      tracked sleep, assert wall-clock is close to one call's duration,
      not N's).
- [ ] `Civ6MCTS` accepts an optional `llm_prior_fn_batch` constructor
      argument; when provided, leaf expansions in a single `search()`
      call are deferred and prior-blended in one batch.
- [ ] Backwards compatibility: `Civ6MCTS` with only `llm_prior_fn`
      (the current scalar callback) still works exactly as before.
- [ ] An end-to-end test in `tests/integration/test_chess_qwen_prior.py`
      asserts that running a single move with 64 simulations issues at
      most M HTTP requests where M ≪ 64 (e.g. M ≤ 5 with mocked LLM
      that records call count).
- [ ] Documented in the script's docstring and in CLAUDE.md.

## Why this is the right next fix

Without this change the agent is unusable in the configured 64-sim/
24-ply regime — even on a fast GPU, 30 sequential 200-ms calls per
move (~6 s) is too slow to iterate on, and any reduction in
`num_simulations` weakens the agent. With this change, the same
configuration becomes interactive and the next bottleneck moves to
either MCTS itself (cheap) or LM Studio's model loading (one-time).

## References

- Empirical observation that prompted this issue:
  `scripts/run_chess_training.py` PID 29128, 2h 18m wall, 13.4 s CPU.
- Sister issue: `ISSUE_chess_advisor_placeholder_game_loop.md` (wires
  the advisor *into* MCTS at all — prerequisite for this).
- Sister issue: `ISSUE_editable_install_submodule_discovery.md`
  (separate, install-time concern).
- Code locations:
  - `alpha_zero/orchestrator/chess_advisor.py::ChessAdvisor.prior_for_board`
  - `alpha_zero/core/civ6_mcts.py::Civ6MCTS.search` and `_blend_llm_prior`
  - `alpha_zero/utils/llm_client.py::LLMClient.generate` (already async)
