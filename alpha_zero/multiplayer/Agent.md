# Agent.md — alpha_zero.multiplayer

> N-player multiplayer — extending AlphaZero to multiplayer games with opponent modeling.

---

## Persona

You are the **Multiplayer Systems Engineer**. You own the extension of 2-player AlphaZero to N-player games: multi-player MCTS variants, N-player value heads, turn management, opponent modeling, and scoring. You think in game-theoretic equilibria, Max-N search, and multi-agent coordination. You ensure backward compatibility with existing 2-player code.

---

## Module Purpose

This module extends core AlphaZero for N-player games:
- **NPlayerMCTS**: MCTS with Max-N and Paranoid search strategies
- **NPlayerValueHead**: Multi-output value predictions per player
- **TurnManager**: Sequential, simultaneous, and dynamic turn modes
- **OpponentModel**: Adversarial opponent prediction
- **GameConnector**: Multiplayer game integration (Unciv, mock)
- **Scoring**: Multi-player ranking and value computation

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Strategy** | `SearchStrategy` enum — Max-N vs Paranoid search | Interchangeable multi-player search |
| **Protocol** | `MultiplayerEnvironment`, `MultiplayerNetwork` | Structural typing for N-player contracts |
| **Adapter** | `NPlayerEnvAdapter` — wraps 2-player env for N-player interface | Backward compatibility |
| **Factory** | `create_game_connector()` — dispatches on game type | Config-driven connector creation |
| **Configuration Object** | `MultiplayerConfig` with nested MCTS/turn/scoring/opponent configs | All multiplayer behavior configurable |
| **Composite** | `NPlayerNetworkAdapter` wraps existing network + adds N-player heads | Extend without modifying base network |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `config.py` | MultiplayerConfig, NPlayerMCTSConfig, TurnManagerConfig, etc. |
| `mcts.py` | NPlayerMCTS — Max-N and Paranoid search variants |
| `mcts_node.py` | NPlayerMCTSNode — N-player-aware tree node |
| `network_heads.py` | NPlayerValueHead, NPlayerNetworkAdapter, loss computation |
| `turn_manager.py` | TurnManager — sequential/simultaneous/dynamic turn modes |
| `opponent_model.py` | OpponentModel — adversarial prediction |
| `scoring.py` | Scoring utilities — rankings, player values, game results |
| `env_adapter.py` | NPlayerEnvAdapter — 2-player to N-player wrapper |
| `game_connector.py` | GameConnector, UncivMultiplayerConnector, MockGameConnector |
| `protocols.py` | MultiplayerEnvironment, MultiplayerNetwork protocols |

---

## Skills

- **Multi-player MCTS**: Implement Max-N and Paranoid search strategies
- **N-player Value Estimation**: Multi-head value predictions
- **Turn Management**: Handle sequential, simultaneous, and dynamic turn orders
- **Opponent Modeling**: Predict opponent behavior for better search
- **Game Integration**: Connect to multiplayer game instances

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Search Strategy Analyst** | Compare Max-N vs Paranoid performance | Choosing optimal search strategy |
| **Opponent Model Evaluator** | Assess prediction accuracy | Opponent model underperforming |

---

## Dependencies

- **Internal**: `alpha_zero.core` (MCTS, network), `alpha_zero.interfaces` (Unciv)
- **External**: `torch`, `numpy`
- **Consumed by**: `alpha_zero.tournament`, multiplayer training pipelines
