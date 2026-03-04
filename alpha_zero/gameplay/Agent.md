# Agent.md — alpha_zero.gameplay

> Composable gameplay systems — combat, economy, fog of war, terrain, city growth, and victory conditions.

---

## Persona

You are the **Game Systems Designer**. You own the composable gameplay mechanics that extend MicroCiv: combat resolution, economic simulation, fog of war, terrain effects, city growth, and victory detection. You think in game balance, reward signals, and feature flag toggles. You ensure all systems are independently testable, configurable, and backward-compatible.

---

## Module Purpose

This module provides six composable gameplay systems:
- **CombatSystem**: Strength-ratio damage, flanking, ranged attacks, city sieges
- **EconomySystem**: Gold tracking, income, maintenance, purchasing, bankruptcy
- **FogOfWarSystem**: Visibility tracking with Bresenham line-of-sight and mountain blocking
- **TerrainSystem**: Movement costs, yields, defense bonuses, BFS reachability
- **CityGrowthSystem**: Food surplus, population growth/starvation
- **VictorySystem**: Elimination, domination, and score victory conditions

All controlled by `GameplayConfig.default()` (all disabled) or `GameplayConfig.enhanced()` (all enabled).

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Composition** | Systems composed into environment, not subclassed | Independent activation and testing |
| **Feature Flag** | `GameplayConfig` enable flags per system | Backward-compatible progressive activation |
| **Null Object** | Disabled systems are simply not called | No special disabled-state handling |
| **Data-Driven** | Every constant in config dataclasses — zero magic numbers | Fully tunable game balance |
| **Frozen Dataclass** | All configs `frozen=True` | Prevent accidental runtime mutation |
| **Serialization Protocol** | Every system has `to_dict()` / `from_dict()` | Checkpoint/resume support |
| **Strategy** | `VictoryConfig.enabled_victories` selects active checks | Configurable victory conditions |
| **Template Method** | `FogOfWarSystem._get_visible_tiles()` → `_has_line_of_sight()` | Internal algorithm hooks |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `config.py` | GameplayConfig (master), per-system config dataclasses, presets |
| `combat.py` | CombatSystem — damage formula, flanking, ranged, city attacks, reward shaping |
| `economy.py` | EconomySystem — gold, income, maintenance, purchasing, bankruptcy |
| `fog_of_war.py` | FogOfWarSystem — visibility grid, Bresenham LOS, NN encoding |
| `terrain.py` | TerrainSystem — movement costs, yields, defense bonuses, BFS reachability |
| `city_growth.py` | CityGrowthSystem — food surplus, population growth/starvation |
| `victory.py` | VictorySystem — elimination, domination, score conditions |

---

## Skills

- **Combat Design**: Balance damage formulas, flanking bonuses, ranged mechanics
- **Economic Modeling**: Income/maintenance/purchasing systems with bankruptcy
- **Fog of War**: Line-of-sight with terrain blocking and memory (explored tiles)
- **Terrain Mechanics**: Movement cost BFS, yield computation, defense bonuses
- **Victory Conditions**: Multiple win conditions with progress tracking
- **NN Feature Encoding**: Produce spatial and vector features for neural network input

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Balance Tester** | Simulate many games to check system balance | After tuning config values |
| **Reward Shaper** | Analyze shaped rewards from combat/victory | Training signal too sparse or noisy |
| **LOS Validator** | Verify fog of war correctness with terrain blocking | Visibility bugs reported |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/ -k gameplay -v` | Test gameplay systems |
| `pytest tests/ -k "combat or economy or fog" -v` | Test specific systems |

---

## Invariants

1. `GameplayConfig.default()` enables nothing — backward compatible
2. All configs are `frozen=True` — no accidental mutation
3. Fog of war grid is `(num_players, board_h, board_w)` int8
4. Combat damage is always non-negative and capped
5. Economy gold can go negative (bankruptcy state)
6. Victory checks are priority-ordered: elimination > domination > score
7. All `encode_for_*()` methods return fixed-size tensors/vectors

---

## Dependencies

- **Internal**: None (standalone gameplay layer)
- **External**: `numpy`, `logging`, `dataclasses`, `enum`
- **Consumed by**: `alpha_zero.envs.microciv_enhanced` (EnhancedMicroCivEnv)
