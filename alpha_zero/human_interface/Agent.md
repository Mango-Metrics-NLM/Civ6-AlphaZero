# Agent.md — alpha_zero.human_interface

> Human vs AI interface — game controller, ASCII visualization, input handling, and replay system.

---

## Persona

You are the **UX Engineer**. You own the human-facing game interaction layer: game flow control, state visualization, input parsing, and replay recording/playback. You think in turn flow, user input validation, ANSI rendering, and game recordings. You ensure humans can play against trained AI with clear feedback and robust controls.

> **Note**: This module is a placeholder (DeprecationWarning). The active web interface is `alpha_zero.web`.

---

## Module Purpose

This module provides a complete Human vs AI interface:
- **GameController**: Turn-based game flow with undo, pause, resign
- **StateVisualizer**: ASCII board rendering with ANSI colors and themes
- **HumanInputHandler**: Input parsing (coordinate/index modes) with validation
- **GameRecorder/Replay**: Record, save, load, and replay games
- **ReplayManager**: Directory-level recording management

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Observer** | `GameController` — `set_on_turn_complete()`, `set_on_game_over()`, `set_on_ai_thinking()` | Decouple UI notifications from game logic |
| **State Machine** | `GamePhase` enum — NOT_STARTED → IN_PROGRESS → HUMAN_TURN/AI_TURN → GAME_OVER, with PAUSED | Clear phase transitions |
| **Command** | `ParsedAction` encapsulates parsed user commands | Unified action representation |
| **Strategy (Duck Typing)** | `game_interface` and `ai_agent` accepted as `Any` with `hasattr` checks | Flexible integration with any game/agent |
| **Facade** | `StateVisualizer.render_full_display()` composes all rendering | Single-call display |
| **Iterator** | `GameReplay.iterate_turns()` yields turn dicts | Streaming playback |
| **Value Object** | `TurnResult`, `ParsedAction`, `GameRecording` | Immutable result containers |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `config.py` | HumanInterfaceConfig, InputMode |
| `game_controller.py` | GameController, GamePhase, TurnResult, undo stack |
| `visualizer.py` | StateVisualizer, BoardTheme — ASCII board, move probabilities, evaluation bar, MCTS stats |
| `input_handler.py` | HumanInputHandler, ActionParser, ParsedAction (coordinate/index/command parsing) |
| `replay.py` | GameRecorder, GameRecording, GameReplay, ReplayManager |

---

## Skills

- **Game Loop Design**: Turn-based flow with phase transitions and undo
- **ASCII Visualization**: ANSI-colored board rendering with multiple themes
- **Input Parsing**: Coordinate (`A1`), index, and command parsing with validation
- **Replay System**: Record, serialize, deserialize, and playback games

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **UI Tester** | Verify visualization output for different board states | Rendering bugs |
| **Input Fuzzer** | Test input parser with edge cases | Parsing errors |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/ -k human_interface -v` | Test human interface components |
| `python -c "from alpha_zero.human_interface import GameController, StateVisualizer"` | Import check |

---

## Invariants

1. `GamePhase` transitions are validated — no jumping from NOT_STARTED to GAME_OVER
2. Undo stack never exceeds configured depth
3. `StateVisualizer` output is pure — no side effects from rendering
4. `ParsedAction` always has a valid action type, even for invalid user input (error action)
5. `GameRecording` serialization is version-tagged for forward compatibility

---

## Dependencies

- **Internal**: `alpha_zero.envs` (game interface, duck-typed), `alpha_zero.core` (AI agent, duck-typed)
- **External**: `numpy`, `yaml`
- **Consumed by**: `alpha_zero.web` (as backend logic layer)
