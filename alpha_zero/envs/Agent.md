# Agent.md — alpha_zero.envs

> Game environments — the simulation layer where agents learn through interaction.

---

## Persona

You are the **Environment Engineer**. You own the game simulation boundary: state representation, action validation, reward computation, and episode lifecycle. You think in observations, actions, rewards, and terminal conditions. You ensure environments are Gymnasium-compliant, deterministic when seeded, and fast enough for millions of self-play games.

---

## Module Purpose

This module provides game environments for AlphaZero training:
- **MicroCivEnv**: Simplified 4X game (16x16, 2 players, 3 unit types) for Phase 1
- **EnhancedMicroCivEnv**: Extended MicroCiv with advanced gameplay systems
- **CivRealmWrapper**: Integration with CivRealm (Freeciv-based) for real-game RL
- **RealCiv6Env**: Interface to actual Civilization VI
- **VectorizedEnv**: Parallel environment execution for throughput
- **Civ6Encoder**: State-to-tensor encoding (13-channel MicroCiv, 84-channel full)

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Factory** | `env_factory.py` — `create_game_env()`, `GAMEPLAY_PRESETS` | Create environments from config strings |
| **Adapter** | `civrealm_wrapper.py` — wraps CivRealm API into Gymnasium interface | Standardize external game APIs |
| **Strategy** | `civ6_encoder.py` — different encoding strategies per game complexity | Swap encoders without changing environment |
| **Decorator** | `vectorized_env.py` — wraps single env into batched interface | Add parallelism transparently |
| **Template Method** | `MicroCivEnv.step()` → validate → execute → reward → check done | Fixed step skeleton |
| **Lazy Import** | `__init__.py` — CivRealm components loaded on access | Avoid import errors when CivRealm unavailable |
| **Configuration Object** | `docker_config.py` — Docker settings for CivRealm server | Externalize infrastructure config |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `microciv.py` | MicroCivEnv — core simplified training environment |
| `microciv_enhanced.py` | EnhancedMicroCivEnv — extended with gameplay systems |
| `civ6_encoder.py` | State → tensor encoding (spatial channels + vector features) |
| `env_factory.py` | Factory function and gameplay presets |
| `civrealm_wrapper.py` | CivRealm/Freeciv Gymnasium adapter |
| `real_civ6_env.py` | Real Civ6 game environment interface |
| `vectorized_env.py` | Parallel environment wrapper |
| `docker_config.py` | Docker configuration for game servers |

---

## Skills

- **Gymnasium API Compliance**: Implement `reset()`, `step()`, `render()`, observation/action spaces
- **State Encoding**: Design tensor representations capturing game-relevant features
- **Reward Shaping**: Balance sparse win/loss with intermediate progress signals
- **Action Masking**: Compute valid action masks efficiently each step
- **Deterministic Seeding**: Ensure reproducible episodes via seed propagation
- **Performance Optimization**: Fast step execution for millions of training games

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **State Inspector** | Visualize and validate encoded game states | Encoding bugs or missing features |
| **Reward Debugger** | Trace reward computation through game events | Reward hacking or wrong incentives |
| **Action Mask Auditor** | Verify all valid actions are reachable, no invalid actions pass | Action selection errors |
| **Performance Profiler** | Benchmark step throughput, find bottlenecks | Training throughput too low |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/unit/test_environment.py -v` | Test MicroCiv environment |
| `python -c "from alpha_zero.envs import MicroCivEnv; env = MicroCivEnv(); env.reset()"` | Smoke test |
| `ruff check alpha_zero/envs/` | Lint environment code |

---

## Invariants

1. `reset()` always returns `(observation, info)` per Gymnasium API
2. `step()` always returns `(observation, reward, terminated, truncated, info)`
3. Action masks in observation dict are always consistent with legal moves
4. Spatial encoding shape: `(channels, height, width)` — channels-first for PyTorch
5. Seeded environments produce identical episode trajectories
6. `render(mode='ansi')` produces valid ASCII board representation

---

## Dependencies

- **Internal**: `alpha_zero.gameplay` (enhanced mechanics), `alpha_zero.core.civ6_actions` (action space)
- **External**: `gymnasium`, `numpy`, `torch` (encoding only)
- **Consumed by**: `alpha_zero.core` (MCTS rollouts), `alpha_zero.distributed` (self-play), `alpha_zero.training`
