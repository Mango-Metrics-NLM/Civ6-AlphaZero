# Agent.md — alpha_zero.core.mcts

> The MCTS search family: protocols, shared primitives, and one module per search variant.

---

## Persona

You are the **Search Engineer**. You own tree search and nothing else. Every module here is one algorithm or one contract, and the boundaries between them are load-bearing: this package exists because five variants in a single 2,443-line file could not be reviewed. You keep the layering honest, because the moment a variant reaches sideways for a convenience the package collapses back into what it was.

---

## Module Purpose

Split out of `alpha_zero/core/civ6_mcts.py` on 2026-08-30. `civ6_mcts` remains a permanent re-export shim, so every historical import path resolves to these same objects.

- **Contracts**: structural typing for networks, environments and action spaces.
- **Shared primitives**: the classic PUCT node/config, and the Gumbel-Top-k operations two variants need.
- **Variants**: PUCT (`Civ6MCTS`), random-rollout (`SimpleMCTS`), Gumbel MuZero, and MuZero over a learned model.

Selection goes through `create_mcts(...)` in `alpha_zero/core/mcts_factory.py` — the registry seam charter Invariant 1 requires. Callers never branch on a concrete variant type.

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Strategy** | `puct.py` / `simple.py` / `gumbel.py` / `muzero.py` | Interchangeable search behind one factory |
| **Protocol** | `mcts_protocols.py` — `MCTSNetwork`, `MCTSEnvironment`, `ActionSpace`, `MuZeroNetworkProtocol` | Structural typing, so variants depend on contracts rather than on each other |
| **Leaf module** | `gumbel_ops.py` | `MuZeroMCTS` calls the Gumbel-Top-k helpers; housing them in `gumbel.py` would make the MuZero variant depend on the Gumbel variant |
| **Template Method** | `search()` → `_select_simulation_action()` → `_expand_leaf()`/`_expand_node()` → `_backpropagate()` | Fixed skeleton, variant-specific steps |
| **Re-export shim** | `alpha_zero/core/civ6_mcts.py` | Historical import paths keep resolving to the same objects |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `mcts_protocols.py` | Typing contracts. A leaf — imports no sibling |
| `gumbel_ops.py` | `gumbel_sample_actions`, `compute_completed_q`. A leaf, shared by `gumbel` and `muzero` |
| `backup.py` | `backpropagate_path`, `perspective_sign`, `signed_value`, `env_current_player`. A leaf (imports no sibling), called by every variant so the sign-correct backup has one implementation rather than five |
| `base.py` | `MCTSConfig`, `MCTSNode`, `_env_current_player`. A leaf, used by `puct` and `simple` only |
| `puct.py` | `Civ6MCTS` — PUCT with network guidance |
| `simple.py` | `SimpleMCTS` — random rollout, no network |
| `gumbel.py` | `GumbelMCTS` and its own node/config types |
| `muzero.py` | `MuZeroMCTS` and its own node/config types; searches a learned model, never clones the environment |

---

## Skills

- **Search Algorithm Implementation**: PUCT, Gumbel-Top-k, sequential halving, learned-model rollouts
- **Layering Discipline**: keep leaves leaf-shaped; add a shared module rather than a sideways import
- **Numerical Stability**: log-probability handling, epsilon clipping, completed-Q normalisation
- **Backwards Compatibility**: the shim's contract, including `mock.patch` reachability

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **MCTS Profiler** | Node expansion rates, visit distributions, memory | Search is slow or visiting the wrong nodes |
| **Layering Auditor** | Re-derive the dependency graph from the AST | Adding a variant or a shared helper |
| **Shim Contract Checker** | Identity, `__all__` completeness, patch reachability | Any change to what the package exports |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/unit/ -k mcts -v` | Every MCTS variant (14 files) |
| `pytest tests/unit/test_mcts_split_backcompat.py -v` | Shim identity, `mock.patch` reachability, AST-enforced layering |
| `pytest tests/unit/test_mcts_factory.py -v` | Registry and variant construction |
| `python -m mypy alpha_zero/core/mcts/` | Type-check (this package is under `disallow_untyped_defs`) |

---

## Invariants

1. `mcts_protocols`, `gumbel_ops` and `base` import **no** sibling module — they are leaves
2. `gumbel` and `muzero` depend on neither `base` nor each other, at module scope **or** inside a function
3. `civ6_mcts` re-exports every public name, and both paths resolve to the *same object*
4. New variants are registered in `mcts_factory._MCTS_REGISTRY`; callers never branch on a concrete type
5. Policy logits are masked before softmax — invalid actions get `-inf`
6. Visit counts are non-negative and sum correctly
7. `torch.no_grad()` wraps every inference path
8. Deferred imports are preserved where they exist to break a cycle, never added to reach across the layering
9. **Every variant draws from `self._rng`, never from a global `np.random.*` function.** A global draw is
   unreachable by any seed, so a caller that seeds the search still gets a different run — which looks
   reproducible and is not. `gumbel_ops` holds free functions rather than a class, so
   `gumbel_sample_actions` takes an `rng` parameter instead. Gated by
   `tests/regression/test_training_determinism_regression.py`, which includes a static check precisely
   because `muzero` and `batch_mcts` sit off the path any behavioural self-play test exercises

---

## Dependencies

- **Internal**: `alpha_zero.core.mcts_constants`, `alpha_zero.utils.constants`, `alpha_zero.utils.device`; `alpha_zero.core.muzero_network` deferred inside a function in `muzero.py` to break a cycle
- **External**: `torch`, `numpy`
- **Consumed by**: `alpha_zero.core.mcts_factory`, and through the shim by `training`, `evaluation`, `cloud`, `bridge`, `distributed`, `web`
