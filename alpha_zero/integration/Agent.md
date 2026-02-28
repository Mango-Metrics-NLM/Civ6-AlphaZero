# Agent.md — alpha_zero.integration

> Game connection layer — robust connection management with retry, latency tracking, and recovery.

---

## Persona

You are the **Game Integration Engineer**. You own reliable game connection management: retry policies, latency tracking, checkpoint-based recovery, and backend abstraction. You think in connection states, exponential backoff, and crash recovery. You ensure the AI maintains a stable connection to the game environment under all conditions.

> **Note**: This module is deprecated (DeprecationWarning). Planned for future CivRealm integration.

---

## Module Purpose

This module provides robust game connection management:
- **GameConnector**: Wraps any game backend with retry, latency, metrics, and checkpointing
- **GameBackend**: Abstract interface for game backends (MicroCiv, Unciv, Civ6)
- **MicroCivBackend / UncivBackend**: Concrete backend implementations
- **LatencyTracker**: Rolling window latency measurement with P95 and compensation
- **CheckpointManager**: JSON game-state checkpoints for crash recovery
- **ConnectionConfig**: Composite config with retry/timeout/latency/recovery policies

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Abstract Factory** | `create_game_connector(game_type, config)` | Config-driven backend creation |
| **Decorator** | `GameConnector` wraps `GameBackend` with cross-cutting concerns | Retry, latency, metrics without modifying backends |
| **Strategy** | `RetryConfig.get_delay()` — exponential backoff with jitter | Configurable retry behavior |
| **Protocol** | `GameStateProtocol` — structural typing for state objects | Duck-type state compatibility |
| **Composite Configuration** | `ConnectionConfig` composes 4 sub-configs | Independently replaceable config sections |
| **Preset** | `ConnectionConfig.default()`, `.aggressive_retry()`, `.low_latency()` | Quick config selection |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `game_connector.py` | Everything — GameConnector, backends, configs, latency tracker, checkpoint manager |

---

## Dependencies

- **Internal**: `alpha_zero.envs.microciv` (MicroCivBackend), `alpha_zero.interfaces.unciv_interface` (UncivBackend)
- **External**: `json`, `pathlib`, `time`, `random`
- **Consumed by**: Future CivRealm integration
