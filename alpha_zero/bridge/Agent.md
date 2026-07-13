# Agent.md — alpha_zero.bridge

> IPC bridge — connecting the Python AlphaZero agent to the live Civilization VI game via Lua mod, over either file polling or a headless FireTuner socket.

---

## Persona

You are the **Game Bridge Engineer**. You own the communication boundary between the Python AI and the real Civilization VI game. You think in file polling, JSON schemas, coordinate transformations, heartbeat timeouts, and — now — raw TCP wire protocols. You ensure reliable, low-latency state exchange across the Lua sandbox boundary, where sockets are unavailable *inside* Lua but very much available from the Python side via the game's own FireTuner debug listener.

---

## Module Purpose

This module provides IPC for real Civ6 game communication over two selectable command transports (state always flows out via `Lua.log`; only the *command* direction differs):
- **FileBridge**: The original all-in-one file bridge (state read *and* command write via JSON files); still used directly by tests and simpler consumers.
- **LiveBridge**: The composed live-game bridge = `LogTailer` (state IN via `Lua.log`) + a command transport that is either **TunerBridge** or **SocketTunerBridge**.
- **TunerBridge** (`tuner_bridge.py`): The `command_transport="file"` command writer used by `LiveBridge` — writes `commands.json` for a FireTuner GUI + `.ltp` relay panel to deliver, since the Civ6 Lua sandbox has no `io` library.
- **SocketTunerBridge** / **FireTunerClient** (`tuner_socket_bridge.py`): Headless, GUI-free command delivery (`command_transport="socket"`). Speaks the reverse-engineered Firaxis FireTuner TCP protocol directly to the game's debug listener on `127.0.0.1:4318` and sets `ExposedMembers.AlphaZeroCommand = <json>` in the `GameCore_Tuner` Lua context — no `commands.json` hop, no `.ltp` panel. `SocketTunerBridge` is an async drop-in for **TunerBridge** (`start`/`stop`/`send_command`/`stats`), selected by `LiveBridge` via `BridgeConfig.command_transport`.
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
| **Configuration Object** | `BridgeConfig` with `from_env()` reading `CIV6_BRIDGE_*` vars, including `command_transport` | Runtime-configurable bridge settings |
| **Strategy** | `LiveBridge` selects `TunerBridge` (file) vs `SocketTunerBridge` (socket) at construction time via `BridgeConfig.command_transport` | Swap the command-delivery mechanism without touching state-read or Lua-mod code |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `config.py` | BridgeConfig, MapSizeConfig, MAP_SIZES, coordinate scaling |
| `protocol.py` | Full typed JSON schema: StateMessage, CommandMessage, AckMessage, ErrorMessage, action/message types |
| `file_bridge.py` | FileBridge (async polling, retry, heartbeat), MockFileBridge (testing) |
| `live_bridge.py` | LiveBridge — composes LogTailer (state) + a command transport (TunerBridge or SocketTunerBridge) selected by `command_transport` |
| `log_tailer.py` | LogTailer — reads state/heartbeat/ack from the Civ6 `Lua.log` (state IN) |
| `tuner_bridge.py` | TunerBridge — `command_transport="file"` command writer (`commands.json` for the FireTuner `.ltp` panel relay) |
| `tuner_socket_bridge.py` | FireTunerClient (raw FireTuner TCP wire protocol), SocketTunerBridge (headless `command_transport="socket"` drop-in for TunerBridge), `lua_longstring()` helper |
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

## Transports

Command delivery is pluggable; state always flows out via `Lua.log` regardless of which transport is active.

| Transport | `command_transport` | How commands reach the mod |
|-----------|---------------------|------------------------------|
| **file** (default) | `"file"` | `TunerBridge` writes `commands.json`; requires the FireTuner GUI + `.ltp` relay panel polling that file and setting `ExposedMembers.AlphaZeroCommand` — the Lua sandbox has no `io` library and cannot read the file itself |
| **socket** | `"socket"` | `SocketTunerBridge`/`FireTunerClient` connects directly to the game's FireTuner TCP listener and sets `ExposedMembers.AlphaZeroCommand` in the `GameCore_Tuner` Lua context — fully headless, no GUI, no file hop |

Select the transport via `BridgeConfig.command_transport` or the env vars consumed by `BridgeConfig.from_env()`:
- `CIV6_BRIDGE_TRANSPORT` — `"file"` or `"socket"` (case-insensitive)
- `CIV6_BRIDGE_TUNER_HOST` / `CIV6_BRIDGE_TUNER_PORT` — override the FireTuner endpoint (default `127.0.0.1:4318`)
- `CIV6_BRIDGE_KICK_START` — forces one `END_TURN` on startup to un-stall a game that's already idling on a human/AI turn

### FireTuner wire protocol (reverse-engineered, verified live)

- Framing: 8-byte header `struct.pack("<Ii", payload_len_incl_nul, tag)` (uint32 LE length including the trailing NUL, then int32 LE tag) followed by a NUL-terminated UTF-8 payload.
- `tag=3` (`TAG_COMMAND`): execute code, formatted as `CMD:<state_index>:<lua>`; an empty (NUL-only) `tag=3` frame signals command completion.
- `tag=4` (`TAG_QUERY`): handshake queries — `APP:` for identity, `LSQ:` to discover Lua state indices.
- `LSQ:` replies with NUL-separated `(index, name)` pairs; the state index used in `CMD:` is the *explicit* number from the pair, not its position in the list.
- `tag=-1` (`TAG_OUTPUT`): server→client `print()`/output frames, format `O\0<ctx>: <text>`.
- `FireTunerClient` connects, sends `LSQ:` to find the `GameCore_Tuner` state, then issues `ExposedMembers.AlphaZeroCommand = <lua_longstring(command_json)>` as a `tag=3` command against that state index.

---

## Invariants

1. File polling uses mtime to avoid re-reading unchanged state
2. Async lock serializes all file reads/writes (no race conditions)
3. Heartbeat timeout triggers circuit breaker open state
4. Coordinate scaling handles collisions via majority voting (terrain) and priority resolution (visibility)
5. Action mask generation is consistent with legal actions from the game
6. MockFileBridge preserves the full FileBridge API for testing
7. `SocketTunerBridge` preserves the same async `start`/`stop`/`send_command`/`stats` surface as `TunerBridge`, so `LiveBridge` can swap transports via config alone
8. `BridgeConfig.command_transport` only accepts `"file"` or `"socket"`; construction raises `ValueError` otherwise
9. `FireTunerClient` always resolves the `GameCore_Tuner` state index via `LSQ:` before sending `CMD:` — the index is not assumed to be stable across game sessions

---

## Dependencies

- **Internal**: `alpha_zero.core.civ6_actions` (ActionSpace), `alpha_zero.envs.civ6_encoder` (state encoding)
- **External**: `asyncio`, `numpy`, `torch`, `yaml`
- **Consumed by**: Real Civ6 game integration, cloud self-play workers
