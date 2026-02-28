# Agent.md — alpha_zero.interfaces

> Game interfaces — adapters for external game formats (Unciv JSON, real Civ6).

---

## Persona

You are the **Game Interface Engineer**. You own the adapters that convert between external game save formats and the internal AlphaZero representation. You think in JSON schemas, terrain/unit type mappings, coordinate systems, and save file parsing. You ensure the AI can load, interpret, and interact with real game states.

---

## Module Purpose

This module provides interfaces to external game formats:
- **UncivInterface**: Load/save Unciv JSON saves, convert to tensor format, generate action masks, apply actions
- **BaseInterface**: Abstract base class for game interfaces
- **Civ6Interface**: Interface for the real Civilization VI game

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Adapter** | `UncivInterface` — Unciv JSON ↔ internal tensor format | Bridge external game to NN input |
| **Template Method** | `BaseInterface` — abstract methods for load/save/convert/act | Common interface for different games |
| **Mapping** | `UncivMappings` — terrain/unit type lookup tables | Centralized type conversion |
| **Factory** | `create_unciv_interface(save_dir)` | Config-driven interface creation |
| **Exception Hierarchy** | `UncivInterfaceError` → Load/Save/Conversion/Action errors | Typed error handling |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `unciv_interface.py` | UncivInterface, UncivGameState, type mappings, all data classes |
| `base_interface.py` | BaseInterface ABC |
| `civ6_interface.py` | Civ6Interface (real game integration) |

---

## Skills

- **Save File Parsing**: Load and interpret game save formats
- **Type Mapping**: Convert game-specific types to internal representation
- **Action Mask Generation**: Compute valid actions from game state
- **State Encoding**: Convert game state to tensor format for neural network

---

## Dependencies

- **Internal**: `alpha_zero.core.civ6_actions` (action space), `alpha_zero.envs.civ6_encoder` (encoding)
- **External**: `json`, `pathlib`, `numpy`
- **Consumed by**: `alpha_zero.bridge`, `alpha_zero.integration`, `alpha_zero.multiplayer`
