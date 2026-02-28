# Agent.md — alpha_zero.interfaces

> Game interfaces — adapters for external game formats (Unciv JSON, real Civ6).

---

## Persona

You are the **Game Interface Engineer**. You own the adapters that convert between external game save formats and the internal AlphaZero representation. You think in JSON schemas, terrain/unit type mappings, coordinate systems, and save file parsing. You ensure the AI can load, interpret, and interact with real game states.

---

## Module Purpose

This module provides interfaces to external game formats:
- **UncivInterface**: Load/save Unciv JSON saves, convert to tensor format, generate action masks, apply actions
- **BaseGameInterface**: Abstract base class (with GameStatus, GameObservation, StepResult, GameConfig, DummyGameInterface)
- **Civ6Interface**: Interface for the real Civilization VI game

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Adapter** | `UncivInterface` — Unciv JSON ↔ internal tensor format | Bridge external game to NN input |
| **Template Method** | `BaseGameInterface` — abstract methods for load/save/convert/act | Common interface for different games |
| **Mapping** | `UncivMappings` — terrain/unit type lookup tables | Centralized type conversion |
| **Factory** | `create_unciv_interface(save_dir)` | Config-driven interface creation |
| **Exception Hierarchy** | `UncivInterfaceError` → Load/Save/Conversion/Action errors | Typed error handling |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `unciv_interface.py` | UncivInterface, UncivGameState, type mappings, all data classes |
| `base_interface.py` | BaseGameInterface ABC, GameStatus, GameObservation, StepResult, GameConfig, DummyGameInterface |
| `civ6_interface.py` | Civ6Interface (real game integration) |

---

## Skills

- **Save File Parsing**: Load and interpret game save formats
- **Type Mapping**: Convert game-specific types to internal representation
- **Action Mask Generation**: Compute valid actions from game state
- **State Encoding**: Convert game state to tensor format for neural network

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Save Format Validator** | Verify JSON save file structure against expected schema | Deserialization failures |
| **Encoding Roundtrip Checker** | Verify state → tensor → state preserves information | Encoding bugs |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/ -k interface -v` | Test interface components |
| `python -c "from alpha_zero.interfaces import BaseGameInterface, UncivInterface"` | Import check |

---

## Invariants

1. `BaseGameInterface` subclasses must implement all abstract methods (load, save, get_state, step)
2. UncivInterface encoding produces fixed-shape tensors regardless of game state
3. Action masks are always consistent with legal moves from the game state
4. Type mappings are bidirectional — encode and decode without loss
5. `DummyGameInterface` passes all interface tests (canonical test double)

---

## Dependencies

- **Internal**: `alpha_zero.core.civ6_actions` (action space), `alpha_zero.envs.civ6_encoder` (encoding)
- **External**: `json`, `pathlib`, `numpy`
- **Consumed by**: `alpha_zero.bridge`, `alpha_zero.integration`, `alpha_zero.multiplayer`
