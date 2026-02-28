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

## Skills

- **Connection Resilience**: Retry with exponential backoff and jitter
- **Latency Tracking**: Rolling window P95/P99 measurement with compensation
- **Crash Recovery**: JSON checkpoint-based game state recovery
- **Backend Abstraction**: Wrap game APIs behind a common interface

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Connection Debugger** | Diagnose retry failures and latency spikes | Connection instability |
| **Recovery Tester** | Verify checkpoint recovery roundtrips | Crash recovery failures |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/ -k integration -v` | Test integration components |
| `python -c "from alpha_zero.integration import GameConnector"` | Import check |

---

## Invariants

1. `GameConnector` retries are bounded — never exceeds `max_retries` from config
2. `LatencyTracker` window has fixed capacity — old measurements roll off
3. Checkpoints are atomic — partial writes never corrupt recovery state
4. `GameBackend` interface is symmetric: `get_state()` / `apply_action()` always paired
5. All config presets produce valid, runnable configurations

---

## Dependencies

- **Internal**: `alpha_zero.envs.microciv` (MicroCivBackend), `alpha_zero.interfaces.unciv_interface` (UncivBackend)
- **External**: `json`, `pathlib`, `time`, `random`
- **Consumed by**: Future CivRealm integration
