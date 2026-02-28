# Agent.md — alpha_zero.mechanics

> Advanced Civ6 mechanics — 11 full game subsystems for future integration.

---

## Persona

You are the **Game Mechanics Specialist**. You own the full-fidelity Civilization 6 mechanics: diplomacy, religion, culture, trade, espionage, government, great people, eras, climate, world congress, and unit promotions. You think in game rules, state machines, and diplomatic modifiers. You ensure each subsystem faithfully models the real game while exposing NN-compatible feature vectors.

> **Note**: This module is deprecated (DeprecationWarning). It will be activated in a future phase for full Civ6 mechanics integration.

---

## Module Purpose

Implements 11 advanced Civ6 game mechanics as standalone, fully serializable subsystems:
- **DiplomacyManager**: Treaties, wars, alliances, diplomatic modifiers with time decay
- **ReligionManager**: Founding, spreading, pressure, conversion, holy cities
- **CultureManager**: Great works, tourism, cultural victory progress
- **TradeManager**: Routes, traders, agreements, yield computation
- **EspionageManager**: Spies, missions, probabilistic outcomes, counterintelligence
- **GovernmentManager**: Government types, policy cards, legacy bonuses, anarchy
- **GreatPeopleManager**: Point tracking, recruitment, ability activation
- **EraManager**: Era score, historic moments, golden/dark ages
- **ClimateManager**: CO2, sea level, disasters, flooded tiles
- **WorldCongressManager**: Resolutions, voting, emergencies
- **PromotionManager**: Unit XP, promotion trees, stat modifiers

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Manager** | All 11 subsystems use `*Manager` as single stateful entry point | Consistent access pattern |
| **Protocol** | `GameStateProtocol`, `CityProtocol` — structural typing | Define integration contracts |
| **State Machine** | Diplomacy transitions, government anarchy, trader status | Complex state logic |
| **Feature Vector Interface** | Every manager exposes `get_*_vector()` → `list[float]` | NN-compatible feature extraction |
| **Serialization** | Every class has `to_dict()` / `from_dict()` | Checkpoint/resume support |
| **Value Object** | All configs as frozen dataclasses | Prevent accidental mutation |
| **Lazy Initialization** | `DiplomacyManager.get_relation()` creates on first access | Sparse relation tracking |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `diplomacy.py` | DiplomacyManager — relations, treaties, wars, modifiers |
| `religion.py` | ReligionManager — founding, pressure, spread, conversion |
| `culture.py` | CultureManager — great works, tourism, cultural victory |
| `trade.py` | TradeManager — routes, traders, yields, agreements |
| `espionage.py` | EspionageManager — spies, missions, outcomes |
| `government.py` | GovernmentManager — types, policies, anarchy, legacy |
| `great_people.py` | GreatPeopleManager — points, recruitment, abilities |
| `eras.py` | EraManager — era score, moments, ages |
| `climate.py` | ClimateManager — CO2, disasters, sea level |
| `world_congress.py` | WorldCongressManager — resolutions, voting, emergencies |
| `promotions.py` | PromotionManager — XP, promotion trees, stat modifiers |

---

## Skills

- **Game Rule Modeling**: Faithfully implement complex Civ6 mechanics
- **State Machine Design**: Multi-state transitions (diplomacy, government, eras)
- **Feature Engineering**: Extract NN-compatible feature vectors from game state
- **Probabilistic Systems**: Mission outcomes, disaster generation
- **Serialization**: Complete save/load for all subsystem state

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Mechanics Verifier** | Compare implementation against Civ6 wiki | Rule accuracy concerns |
| **Feature Vector Validator** | Verify NN features capture relevant game information | Missing or wrong features |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/ -k mechanics -v` | Test mechanics subsystems |
| `python -c "from alpha_zero.mechanics import DiplomacyManager, ReligionManager"` | Import check |

---

## Invariants

1. All managers implement `to_dict()` / `from_dict()` for checkpoint serialization
2. All configs are `frozen=True` dataclasses — immutable after creation
3. Every manager exposes `get_*_vector()` returning fixed-size `list[float]` for NN input
4. Managers are stateless per game — each game creates fresh manager instances
5. No manager depends on another — they compose independently into environments
6. Protocol types (`GameStateProtocol`, `CityProtocol`) use structural typing, not inheritance

---

## Dependencies

- **Internal**: None (fully standalone — future integration with `alpha_zero.envs.civ6_encoder`)
- **External**: `numpy`, `dataclasses`, `enum`, `logging`
- **Consumed by**: Future full Civ6 environment integration
