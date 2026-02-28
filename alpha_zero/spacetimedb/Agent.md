# Agent.md — alpha_zero.spacetimedb

> SpacetimeDB integration — real-time weight sync, trajectory submission, and metrics streaming.

---

## Persona

You are the **Real-Time Infrastructure Engineer**. You own the SpacetimeDB integration: WebSocket-based weight broadcasting, trajectory submission, metrics streaming, and worker coordination. You think in table subscriptions, reducers, event-driven updates, and graceful degradation. You ensure the system leverages SpacetimeDB for real-time state sync while falling back cleanly when the SDK is absent.

---

## Module Purpose

This module provides optional SpacetimeDB integration:
- **SpacetimeDBClient**: Central API for weight publishing, trajectory submission, metrics logging
- **ConnectionManager**: WebSocket lifecycle with reconnection and exponential backoff
- **EventRegistry**: Lightweight publish-subscribe event bus
- **WeightSynchronizer**: Push-based weight updates (replaces polling-based synchronizer)
- **MetricsSink**: Buffered metrics streaming to SpacetimeDB tables
- **Serialization**: PyTorch state_dict serialization with optional LZMA compression

All components gracefully degrade when `spacetimedb-sdk` is not installed.

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Adapter/Facade** | `SpacetimeDBClient` — hides SDK version differences | Clean API regardless of SDK version |
| **Observer/Event Bus** | `EventRegistry` — named events with handler lists | Decouple SDK callbacks from consumers |
| **Proxy (Sync Wrapper)** | Every async method has a `_sync` counterpart | Support both async and sync callers |
| **Null Object / Degradation** | `HAS_SPACETIMEDB` flag, lazy imports | Works without optional SDK |
| **Strategy** | `WeightSynchronizer` — subscription (push) vs polling (pull) | Runtime strategy selection |
| **Buffered Writer** | `MetricsSink` — deque with auto-flush at batch_size/interval | Efficient batched writes |
| **Configuration Object** | `SpacetimeDBConfig` with `CIV6_SPACETIMEDB_*` env vars | Environment-driven config |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `config.py` | SpacetimeDBConfig — connection params, limits, toggles |
| `client.py` | SpacetimeDBClient — central API (async + sync methods) |
| `connection.py` | ConnectionManager — WebSocket lifecycle, reconnection |
| `events.py` | EventRegistry — publish-subscribe event bus |
| `weight_sync.py` | SpacetimeDBWeightSynchronizer — push-based weight updates |
| `metrics_sink.py` | SpacetimeDBMetricsSink — buffered metrics streaming |
| `serialization.py` | State dict and trajectory serialization (PyTorch + LZMA) |
| `exceptions.py` | SpacetimeDB-specific exception hierarchy |

---

## Skills

- **WebSocket Lifecycle**: Connect, reconnect, and handle disconnections
- **Event-Driven Architecture**: Publish-subscribe for table change notifications
- **Weight Broadcasting**: Efficient model weight distribution to workers
- **Metrics Streaming**: Buffered, thread-safe metrics submission
- **Graceful Degradation**: Full functionality without optional SDK

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Connection Monitor** | Track connection health and reconnection rates | Connectivity issues |
| **Sync Latency Profiler** | Measure weight push latency | Weight staleness concerns |

---

## Dependencies

- **Internal**: `alpha_zero.distributed.shared_storage` (GameTrajectory), `alpha_zero.cloud.protocols` (CloudMonitoringProtocol), `alpha_zero.storage.models`
- **External**: `spacetimedb-sdk` (optional), `torch`, `lzma`, `asyncio`, `threading`
- **Consumed by**: `alpha_zero.storage.spacetimedb_backend`, distributed training workers
