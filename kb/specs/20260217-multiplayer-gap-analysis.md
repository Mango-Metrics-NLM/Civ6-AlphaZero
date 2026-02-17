# Civ6 Multiplayer Mode — Gap Analysis

**Date:** 2026-02-17
**Status:** Implementation Started
**Phase:** Multiplayer Infrastructure (New)

---

## Executive Summary

This document analyzes the gaps between the current Civ6-AlphaZero system (designed for 2-player self-play) and full Civilization 6 multiplayer support (2-8 players with simultaneous turns, fog of war, and diplomacy). A new `alpha_zero/multiplayer/` module has been implemented to bridge the algorithmic gaps, with remaining gaps documented in the roadmap below.

---

## 1. Current System State

### What Exists (2-Player Foundation)

| Component | Status | Location |
|-----------|--------|----------|
| MicroCiv Environment | 2-player, turn-based | `alpha_zero/envs/microciv.py` |
| State Encoder | 13-channel, own/enemy split | `alpha_zero/envs/civ6_encoder.py` |
| MCTS (4 variants) | Negamax backprop (2-player) | `alpha_zero/core/civ6_mcts.py` |
| Neural Network | Scalar value head | `alpha_zero/core/civ6_network.py` |
| Self-Play Actor | Binary win/loss values | `alpha_zero/distributed/actor.py` |
| Replay Buffer | Single value per example | `alpha_zero/distributed/shared_storage.py` |
| Unciv Interface | JSON save file I/O | `alpha_zero/interfaces/unciv_interface.py` |
| Civ6 Bridge | File-based IPC with Lua mod | `alpha_zero/bridge/` |
| Web Interface | Single-player vs AI | `alpha_zero/web/` |

### What's New (Multiplayer Module)

| Component | Status | Location |
|-----------|--------|----------|
| N-Player MCTS Node | **Implemented** | `alpha_zero/multiplayer/mcts_node.py` |
| N-Player MCTS (Max-N, Paranoid, Best-Reply) | **Implemented** | `alpha_zero/multiplayer/mcts.py` |
| N-Player Value Head | **Implemented** | `alpha_zero/multiplayer/network_heads.py` |
| Environment Adapter | **Implemented** | `alpha_zero/multiplayer/env_adapter.py` |
| Turn Manager (Sequential/Simultaneous/Dynamic) | **Implemented** | `alpha_zero/multiplayer/turn_manager.py` |
| Opponent Modeling | **Implemented** | `alpha_zero/multiplayer/opponent_model.py` |
| Scoring/Ranking System | **Implemented** | `alpha_zero/multiplayer/scoring.py` |
| Game Connectors (Unciv, Mock) | **Implemented** | `alpha_zero/multiplayer/game_connector.py` |
| Configuration System | **Implemented** | `alpha_zero/multiplayer/config.py` |
| Protocols | **Implemented** | `alpha_zero/multiplayer/protocols.py` |
| Test Suite (175 tests) | **All Passing** | `tests/unit/test_multiplayer/` |

---

## 2. Gap Categories

### 2.1 CLOSED Gaps (Addressed by Implementation)

#### G1: Negamax-Only Backpropagation -> N-Player Value Vectors
- **Was:** `value = -value` at each ply (only works for 2 players)
- **Now:** `NPlayerMCTSNode` stores per-player value dicts; `backpropagate()` accumulates per-player values without flipping
- **Strategies:** Max-N (each player maximizes own value), Paranoid (opponents collude), Best-Reply (strongest opponent only)

#### G2: Scalar Value Head -> Per-Player Value Output
- **Was:** `ValueHead` outputs `(B, 1)` scalar via `tanh`
- **Now:** `NPlayerValueHead` outputs `(B, num_players)` with `tanh` per player
- **Adapter:** `NPlayerNetworkAdapter` wraps existing 2-player networks, reusing spatial/vector encoders

#### G3: Binary Win/Loss Scoring -> Ranked Outcome Values
- **Was:** Winner = +1, loser = -1, draw = 0
- **Now:** Three scoring methods:
  - `linear_ranking`: 1st = +1, last = -1, interpolated between
  - `winner_take_all`: Winner = +1, all others = -1
  - `proportional`: Values scaled by in-game scores
- **Rankings computed from:** winner ID, player scores, eliminated set

#### G4: No Turn Mode Support -> Sequential/Simultaneous/Dynamic
- **Was:** Alternating 2-player turns only
- **Now:** `TurnManager` handles:
  - Sequential: One player per step, rotating
  - Simultaneous: All players act concurrently
  - Dynamic: Simultaneous during peace, sequential during war (Civ6 recommended MP mode)

#### G5: No Opponent Modeling -> Frequency-Based Prediction
- **Was:** No tracking of opponent behavior
- **Now:** `OpponentModel` with exponential-decay frequency counting, Bayesian blending with uniform prior, per-player independent tracking

#### G6: No Multiplayer Game Connection -> Unciv/Mock Connectors
- **Was:** No way to connect to multiplayer game instances
- **Now:** Abstract `GameConnector` with `UncivMultiplayerConnector` (JSON save file monitoring) and `MockGameConnector` (testing)

---

### 2.2 OPEN Gaps (Remaining Work)

#### G7: MicroCiv Environment N-Player Support
**Priority:** HIGH | **Effort:** Medium

The core MicroCiv environment is hardcoded for 2 players:
- `_end_turn()` uses `1 - self.current_player` (only swaps 0 and 1)
- `_check_victory()` hardcodes P0/P1 elimination checks
- `_place_starting_units()` places units for exactly 2 players
- Turn increment logic assumes 2-player alternation

**Required changes:**
```python
# Current
self.current_player = 1 - self.current_player
# Needed
self.current_player = (self.current_player + 1) % self.num_players
```

**Files:** `alpha_zero/envs/microciv.py` (~50 lines changed)

#### G8: State Encoder N-Player Channels
**Priority:** HIGH | **Effort:** Medium

The 13-channel MicroCivEncoder assumes binary own/enemy split:
- Channels 3-5: Own units (warrior, settler, archer)
- Channels 6-8: Enemy units (single opponent)
- Channels 9-10: Own/enemy cities

**Required:** Dynamic channel allocation per player:
- Formula: `base_terrain(3) + num_players * (unit_channels(3) + city_channels(1)) + extra(2)`
- 2 players: 13 channels (backward compatible)
- 4 players: 21 channels
- 8 players: 37 channels

**Config:** `MultiplayerConfig.total_spatial_channels` already computes this.

**Files:** `alpha_zero/envs/civ6_encoder.py` (~100 lines changed)

#### G9: Training Pipeline N-Player Value Integration
**Priority:** HIGH | **Effort:** Medium

The training pipeline stores single scalar values:
- `TrainingExample.value: float` (+/-1)
- `GameTrajectory.winner: int | None`
- Loss: `F.mse_loss(predicted_value, target_value)` on scalars

**Required:**
- `TrainingExample.value` -> `values: dict[int, float]` or `np.ndarray`
- `GameTrajectory` -> add `rankings`, `player_scores`, `num_players`
- Loss -> `compute_nplayer_value_loss()` (already implemented)
- Actor value assignment -> use `compute_player_values()` (already implemented)

**Files:** `alpha_zero/distributed/shared_storage.py`, `actor.py`, `learner.py`

#### G10: Fog of War / Partial Observability
**Priority:** MEDIUM | **Effort:** High

Civ6 multiplayer has fog of war -- players cannot see the full map. Current system assumes full observability.

**Approaches (from literature):**
1. **Information Set MCTS (IS-MCTS):** Group nodes by information set
2. **Determinization:** Sample multiple consistent game states, run MCTS on each
3. **Perfect Information Monte Carlo (PIMC):** AlphaZe* approach

**Recommendation:** Start with determinization (simplest). Each MCTS simulation samples a "world" consistent with what the player has observed, then runs standard search.

**Effort:** New module, ~500 lines + tests

#### G11: Simultaneous Action Resolution
**Priority:** MEDIUM | **Effort:** Medium

When players act simultaneously (Civ6 online multiplayer), action ordering matters. Current MCTS assumes sequential play.

**Approaches (from literature):**
1. **Decoupled UCT (DUCT):** Treat each player's action independently (simple but weak)
2. **Regret Matching (RM):** Minimize cumulative regret (strong empirical results)
3. **Online Outcome Sampling (OOS):** Approaches Nash equilibrium

**Recommendation:** Start with DUCT for simplicity; the `TurnManager` already supports simultaneous mode. The MCTS needs adaptation to sample joint actions.

#### G12: Real Civ6 Game Integration
**Priority:** LOW | **Effort:** Very High

No project has successfully connected external ML to actual Civ6. Available paths:

| Path | Reliability | Speed | Effort |
|------|------------|-------|--------|
| Lua mod file IPC | Medium | ~1 action/sec | High |
| AutoplayManager | Low | ~0.5 action/sec | Medium |
| Community Extension DLL | High | Fast | Very High (Windows only) |
| Save file manipulation | High | ~1 turn/min | Medium |
| **Unciv JSON (current)** | **High** | **~10 turns/sec** | **Low** |

**Recommendation:** Continue with Unciv for training; add Civ6 Lua mod bridge for evaluation only.

#### G13: Diplomatic Decision Making
**Priority:** LOW | **Effort:** High

Civ6 multiplayer involves complex diplomacy: alliances, trade deals, war declarations, peace treaties. The action space for diplomacy is open-ended.

**Current state:**
- `alpha_zero/mechanics/diplomacy.py` has basic diplomacy system
- Action space includes `DIPLOMACY` category with `num_diplomacy_actions`
- `TurnManager.declare_war()`/`make_peace()` handle mode transitions

**Missing:**
- LLM-based diplomatic reasoning (CivAgent approach)
- Coalition formation modeling
- Communication channel between players

#### G14: Population-Based Training for Multiplayer
**Priority:** LOW | **Effort:** Medium

For N>2 players, population-based training (PBT) becomes important to avoid convergence to a single strategy.

**Current state:** `alpha_zero/pbt/` module exists for hyperparameter optimization.

**Missing:** Extending PBT to maintain diverse policy populations where different policies represent different playstyles competing against each other.

---

## 3. Architecture Overview

```
+--------------------------------------------------+
|              Multiplayer Module                    |
|  alpha_zero/multiplayer/                          |
|                                                   |
|  +--------------+  +--------------+               |
|  | NPlayerMCTS  |  | NPlayerValue |               |
|  | (Max-N,      |  | Head         |               |
|  |  Paranoid,   |  | (B,N) output |               |
|  |  Best-Reply) |  |              |               |
|  +------+-------+  +------+-------+               |
|         |                  |                       |
|  +------+------------------+------+               |
|  |       NPlayerMCTSNode         |               |
|  |  per-player value storage     |               |
|  |  Max-N selection              |               |
|  +-------------------------------+               |
|                                                   |
|  +--------------+  +--------------+               |
|  | TurnManager  |  | OpponentModel|               |
|  | Seq/Sim/Dyn  |  | Freq-based   |               |
|  +--------------+  +--------------+               |
|                                                   |
|  +--------------+  +--------------+               |
|  | Scoring      |  | EnvAdapter   |               |
|  | Linear/WTA/  |  | Wraps 2-play |               |
|  | Proportional |  | for N-player |               |
|  +--------------+  +--------------+               |
|                                                   |
|  +--------------+  +--------------+               |
|  | GameConnector|  | Protocols    |               |
|  | Unciv/Mock   |  | MP Env/Net   |               |
|  +--------------+  +--------------+               |
+--------------------------------------------------+
                        |
                        v
+--------------------------------------------------+
|              Existing Core                        |
|  alpha_zero/core/                                 |
|  - civ6_mcts.py (2-player, unchanged)            |
|  - civ6_network.py (scalar value, unchanged)     |
|  - civ6_actions.py (action space, unchanged)     |
+--------------------------------------------------+
```

---

## 4. Civ6 Multiplayer Mechanics Reference

### Turn Modes
| Mode | Description | Civ6 Usage |
|------|-------------|------------|
| Sequential | One player per step | Hotseat, war phases |
| Simultaneous | All players act together | Online peacetime |
| Dynamic | Simultaneous then Sequential on war | Recommended competitive mode |

### Multiplayer Variants
| Variant | Players | Network | Save File |
|---------|---------|---------|-----------|
| Online | 2-8 | P2P UDP | No |
| LAN | 2-8 | Local P2P | No |
| Hotseat | 2-8 | None | Local |
| Play by Cloud | 2-8 | 2K Cloud | Cloud save |
| PYDT (async) | 2-8 | AWS/email | Save files |

### Known AI Integration Projects
| Project | Game | Approach | Status |
|---------|------|----------|--------|
| **This Project** | MicroCiv/Unciv | AlphaZero + MCTS | Active |
| CivAgent | Unciv | LLM + RAG + Simulator | Active (2025) |
| CivRealm | Freeciv | RL (Gymnasium) + LLM | Active (ICLR 2024) |
| ML_CIV6 | Simplified Civ6 | Basic ML scaffold | Early stage |

---

## 5. Implementation Roadmap

### Phase A: Core N-Player Training (Next)
1. Extend MicroCiv for N players (G7)
2. Implement N-player state encoder (G8)
3. Integrate scoring into training pipeline (G9)
4. Train 3-4 player MicroCiv agents

### Phase B: Multiplayer Self-Play
5. N-player self-play actor using rotation
6. Multi-policy population training
7. Tournament evaluation for N-player

### Phase C: Partial Observability
8. Information set sampling for fog of war (G10)
9. Determinized MCTS implementation
10. Observation model training

### Phase D: Civ6 Integration
11. Unciv multiplayer connector (fully functional)
12. Civ6 Lua mod bridge (evaluation only)
13. Diplomatic action space expansion

---

## 6. Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| N-player MCTS too slow | HIGH | Medium | Paranoid search reduces to 2-player depth; batch inference |
| Fog of war quality | HIGH | High | Start with determinization; iterate |
| Training instability (N>2) | MEDIUM | Medium | Population-based training; curriculum |
| Civ6 mod compatibility | LOW | High | Primary path via Unciv; Civ6 optional |
| Simultaneous turn conflicts | MEDIUM | Medium | Dynamic mode falls back to sequential |

---

## 7. References

1. Petosa, N. "Multiplayer AlphaZero" (NeurIPS 2019 Workshop) - arXiv:1910.13012
2. Nijssen, J.P.A.M. "Monte-Carlo Tree Search for Multi-Player Games" (PhD Thesis, Maastricht)
3. Lanctot et al. "MCTS Variants for Simultaneous Move Games" (CIG 2014)
4. CivAgent - "Digital Player: LLM-based Agent in Games" (arXiv:2502.20807, 2025)
5. CivRealm - "A Learning and Reasoning Odyssey" (ICLR 2024)
6. Lisy et al. "Convergence of MCTS in Simultaneous Move Games" (NeurIPS)

---

*Generated: 2026-02-17*
*Module: alpha_zero/multiplayer/ (8 source files, 175 tests)*
