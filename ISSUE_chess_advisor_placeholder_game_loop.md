# `scripts/run_chess_training.py` constructs `ChessAdvisor` but never queries it during games

## Summary

PR #75 wired the LM Studio (Qwen) opening-prior advisor end-to-end — config
loading, `LLMClient` construction, FEN cache, JSON parsing, and a
`mcts_callback()` shaped for `Civ6MCTS(llm_prior_fn=...)` are all implemented
and verifiably correct. However, the entry-point script
`scripts/run_chess_training.py` plays games with a placeholder action selector
that picks **random** legal moves. It never invokes MCTS, so it never invokes
the advisor, so Qwen is never actually consulted during games launched
through this script.

The advisor is correctly *constructed* (you can see the
`INFO __main__: ChessAdvisor: enabled, weight=0.3` log line) but never *called*.

This is the gap between "LM Studio is integrated" (true) and "LM Studio shapes
play" (not yet). The script's docstring acknowledges the limitation but does
not flag it loudly enough that a casual reader would notice the advisor is
inert at runtime.

## Evidence

The placeholder in `scripts/run_chess_training.py` (around line 154):

```python
def _play_one_game(env, opponent, max_plies, rng=None):
    """Play a game with the agent picking random legal moves (placeholder).
    ...
    """
    if rng is None:
        rng = np.random.default_rng(0)
    obs, _ = env.reset()
    agent_player = env.current_player
    for _ply in range(max_plies):
        mask = obs["action_mask"]
        legal = np.flatnonzero(mask > 0)
        if env.current_player == agent_player:
            action = int(rng.choice(legal))   # ← random move, no MCTS, no advisor
        else:
            action = opponent.get_action(env, mask)
        obs, _, terminated, truncated, info = env.step(action)
        ...
```

The advisor is built but its callback is discarded:

```python
advisor = _build_advisor(advisor_cfg, llm_cfg)   # constructed
...
for it in range(iterations):
    for _ in range(games_per_iter):
        reward = _play_one_game(env, opponent, max_plies)   # advisor not passed in
```

End-to-end proof (from a smoke test outside the script) that the advisor *can*
return a usable prior:

```python
import chess
from alpha_zero.orchestrator.chess_advisor import ChessAdvisor, ChessAdvisorConfig
from alpha_zero.utils.llm_client import LLMClient, LLMConfig, LLMProvider

cfg = ChessAdvisorConfig(enabled=True, prior_weight=0.3, max_ply_for_prior=24)
llm = LLMClient(LLMConfig(
    provider=LLMProvider.OPENAI,
    openai_api_key="lm-studio",
    openai_base_url="http://localhost:1234/v1",
    openai_model="qwen2.5-14b-instruct-1m",
))
adv = ChessAdvisor(cfg, llm_client=llm)
prior = adv.prior_for_board(chess.Board())
# → prior is a numpy array of shape (4672,), 15 non-zero indices,
#   normalized to a probability distribution. Qwen was hit. Cache works.
```

So the advisor → LM Studio path is healthy. The script just doesn't use it.

## Proposed fix

Replace the placeholder action selector with a `Civ6MCTS` (or `GumbelMCTS`)
instance that takes the advisor's callback as `llm_prior_fn`. The
plumbing on the MCTS side already exists (`alpha_zero/core/civ6_mcts.py`
exposes `llm_prior_fn` and `llm_prior_weight` constructor args).

Sketch (replacement for the random-move branch in `_play_one_game`):

```python
from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig

def _build_mcts(network, advisor, training_cfg):
    mcts_cfg = MCTSConfig(
        num_simulations=int(training_cfg.get("num_simulations", 64)),
        c_puct=float(training_cfg.get("c_puct", 1.5)),
    )
    return Civ6MCTS(
        network=network,
        config=mcts_cfg,
        llm_prior_fn=advisor.mcts_callback() if advisor is not None else None,
        llm_prior_weight=float(advisor.config.prior_weight) if advisor is not None else 0.5,
    )

def _play_one_game(env, opponent, max_plies, mcts, rng=None):
    obs, _ = env.reset()
    agent_player = env.current_player
    for _ply in range(max_plies):
        mask = obs["action_mask"]
        if env.current_player == agent_player:
            policy, _value = mcts.search(env, mask)         # uses Qwen prior
            action = int(np.argmax(policy * mask))           # or sample for exploration
        else:
            action = opponent.get_action(env, mask)
        obs, _, terminated, truncated, info = env.step(action)
        ...
```

Caller threads the MCTS through:

```python
network = _build_network(config.get("network", {}))   # placeholder; reuse existing factory
mcts = _build_mcts(network, advisor, config.get("training", {}))
for it in range(iterations):
    for _ in range(games_per_iter):
        reward = _play_one_game(env, opponent, max_plies, mcts)
```

The network requirement is the only real friction — `Civ6MCTS` needs a
network to call. Three options for the network:

1. **Untrained `MicroCivNetwork`** (simplest, just to validate the path):
   the network's policy/value will be near-uniform and Qwen's prior will
   dominate the early-game decision through the blend weight. Demonstrates
   the prior is shaping play but isn't a serious chess agent.
2. **A small `Civ6Network` adapted for chess** (medium effort): proper
   chess-shape input/output heads. Useful for actually running training.
3. **A pretrained chess network** (largest scope; out of this issue's scope):
   load Lc0 weights or similar. Properly addresses the "self-play loop"
   follow-up the script's docstring mentions.

For this issue I'd recommend **option 1** — wire the placeholder network just
to prove Qwen's prior reaches the action selector. Option 2/3 are tracked
separately as part of the broader self-play orchestrator wiring.

## Why this matters

Anyone running the script today and watching the log sees:

```
INFO ... OpenAI client initialized
INFO ... ChessAdvisor: enabled, weight=0.3
INFO ... Iteration 0: win_rate=0.000, curriculum=0
```

…and reasonably concludes the LM Studio prior is being used. They'd be wrong:
games complete in milliseconds because no MCTS runs and Qwen is never called.
The misleading nature of "advisor enabled" + "iteration completed" is the
core bug from a UX standpoint.

## Acceptance criteria

- [ ] `python scripts/run_chess_training.py --config configs/chess_config.yaml --iterations 1 --games-per-iter 1 --skill 0` makes at least one POST request to `http://localhost:1234/v1/chat/completions` (verifiable in LM Studio's Developer panel)
- [ ] When `advisor.enabled: false`, the script still runs (no Qwen calls, but games still complete)
- [ ] When LM Studio is unreachable, the script logs a single warning and continues with network-only priors (graceful-degradation contract from `ChessAdvisor._log_once`)
- [ ] The placeholder docstring is updated to either describe the new MCTS-driven loop or explicitly state that this script remains a wiring smoke test and the orchestrator is the production entry point
- [ ] An integration test extends `tests/integration/test_chess_qwen_prior.py` to assert that running `_play_one_game` with `advisor.enabled: true` results in at least one call into `LLMClient.generate_sync` (mock the LLM)

## References

- Triggered by validating PR #75's LM Studio integration end-to-end on Windows.
- Related: the script's own docstring already flags the placeholder ("does not implement the full self-play/learner loop; that lives in the orchestrator and will be wired in a follow-up").
- Sister issue: see `ISSUE_editable_install_submodule_discovery.md` (separate problem, separate fix).
- Code locations:
  - Placeholder: `scripts/run_chess_training.py` lines ~125–180
  - Advisor callback: `alpha_zero/orchestrator/chess_advisor.py::ChessAdvisor.mcts_callback`
  - MCTS hook: `alpha_zero/core/civ6_mcts.py::Civ6MCTS.__init__` (`llm_prior_fn`, `llm_prior_weight`)
