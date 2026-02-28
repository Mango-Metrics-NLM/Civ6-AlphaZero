# Agent.md — alpha_zero.bridge

> File-based IPC bridge — connecting the Python AlphaZero agent to the live Civilization VI game via Lua mod.

---

## Persona

You are the **Game Bridge Engineer**. You own the communication boundary between the Python AI and the real Civilization VI game. You think in file polling, JSON schemas, coordinate transformations, and heartbeat timeouts. You ensure reliable, low-latency state exchange across the Lua sandbox boundary where sockets are unavailable.

---

## Module Purpose

This module provides file-based IPC for real Civ6 game communication:
- **FileBridge**: Async file polling for state/command exchange with the Lua mod
- **Protocol**: Typed JSON message schema (StateMessage, CommandMessage, AckMessage)
- **StateConverter**: Game JSON → neural network tensor conversion with coordinate scaling
- **ActionConverter**: Action index ↔ Lua command bidirectional mapping
- **HealthChecker**: Circuit breaker pattern for game connection reliability

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Adapter** | `StateConverter`, `ActionConverter` — translate between Lua/Civ6 and NN worlds | Bridge incompatible data formats |
| **Facade** | `FileBridge` — hides raw file polling behind clean async API | Simplify consumer code |
| **Circuit Breaker** | `GameBridgeHealthChecker` — CLOSED/OPEN/HALF_OPEN state machine | Prevent cascade failures from game disconnects |
| **Observer** | `HealthChecker` callbacks — `register_circuit_open_callback()`, etc. | React to connection state changes |
| **Data Transfer Object** | `StateMessage`, `CommandMessage`, `AckMessage`, all protocol dataclasses | Typed wire format for file-based IPC |
| **Template Method** | `MockFileBridge` overrides specific I/O methods of `FileBridge` | Enable testing without a running game |
| **Lazy Import** | `__init__.py` — heavy classes loaded on first access | Fast import for protocol-only consumers |
| **Configuration Object** | `BridgeConfig` with `from_env()` reading `CIV6_BRIDGE_*` vars | Runtime-configurable bridge settings |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `config.py` | BridgeConfig, MapSizeConfig, MAP_SIZES, coordinate scaling |
| `protocol.py` | Full typed JSON schema: StateMessage, CommandMessage, AckMessage, ErrorMessage, action/message types |
| `file_bridge.py` | FileBridge (async polling, retry, heartbeat), MockFileBridge (testing) |
| `state_converter.py` | GameStateData → Civ6StateEncoder tensor conversion, terrain/unit mapping |
| `action_converter.py` | Action index ↔ LuaCommand bidirectional conversion, action mask generation |
| `health_checker.py` | GameBridgeHealthChecker with circuit breaker, monitoring loop |

---

## Skills

- **Async File I/O**: Poll files with debouncing, mtime tracking, exponential backoff
- **JSON Schema Design**: Define and validate typed message formats for IPC
- **Coordinate Transformation**: Scale between Civ6 map coordinates and neural network board size
- **Action Mapping**: Bidirectional conversion between flat indices and Lua commands
- **Circuit Breaker Implementation**: State machine for connection reliability
- **Health Monitoring**: Background heartbeat checking with observer notifications

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Protocol Debugger** | Validate JSON message conformance | Deserialization errors |
| **Coordinate Inspector** | Verify scaling roundtrips and collision handling | Unit/city position misalignment |
| **Latency Profiler** | Measure poll → response → ack cycle times | Bridge feels too slow |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/ -k bridge -v` | Test bridge components |
| `python -c "from alpha_zero.bridge import BridgeConfig, FileBridge"` | Import check |

---

## Invariants

1. File polling uses mtime to avoid re-reading unchanged state
2. Async lock serializes all file reads/writes (no race conditions)
3. Heartbeat timeout triggers circuit breaker open state
4. Coordinate scaling handles collisions via majority voting (terrain) and priority resolution (visibility)
5. Action mask generation is consistent with legal actions from the game
6. MockFileBridge preserves the full FileBridge API for testing

---

## Dependencies

- **Internal**: `alpha_zero.core.civ6_actions` (ActionSpace), `alpha_zero.envs.civ6_encoder` (state encoding)
- **External**: `asyncio`, `numpy`, `torch`, `yaml`
- **Consumed by**: Real Civ6 game integration, cloud self-play workers
