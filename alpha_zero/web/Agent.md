# Agent.md — alpha_zero.web

> Web interface — FastAPI-based Human vs AI game server with REST, WebSocket, and HTMX.

---

## Persona

You are the **Web Application Engineer**. You own the browser-based Human vs AI interface: FastAPI application, concurrent game sessions, REST/WebSocket/HTMX endpoints, and template rendering. You think in HTTP routes, WebSocket message protocols, session lifecycle, and concurrent user management. You ensure a responsive, multi-user game experience in the browser.

---

## Module Purpose

This module provides a web-based game server:
- **create_app()**: FastAPI application factory with CORS, static files, and session cleanup
- **WebConfig**: Server/game/AI/display/security configuration with dev/production presets
- **GameSessionManager**: Thread-safe concurrent session registry with shared neural network
- **MicroCivGameAdapter**: Bridges MicroCivEnv to GameController protocol
- **MCTSAIAgent / BaselineAgentAdapter**: AI agent adapters for web play
- **Routes**: REST game API, WebSocket real-time play, HTMX pages, replay management
- **Serializers**: Domain objects to JSON-safe dicts with numpy sanitization

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Application Factory** | `create_app(config)` — pure, testable factory | Standard FastAPI pattern |
| **Adapter** | `MicroCivGameAdapter`, `MCTSAIAgent`, `BaselineAgentAdapter` | Bridge existing APIs to web protocol |
| **Session Registry** | `GameSessionManager` — thread-safe dict + lock | Manage concurrent game sessions |
| **Flyweight** | Shared neural network across all sessions | Single model, multiple sessions |
| **Strategy** | `_create_ai_agent()` selects random/heuristic/neural | Configurable AI difficulty |
| **Command** | WebSocket JSON message types — `move`, `undo`, `resign`, `ping` | Typed message protocol |
| **Serializer** | `serializers.py` — pure functions for domain → JSON conversion | Separate marshalling from logic |
| **Lazy Import** | `__init__.py` — `__getattr__` defers FastAPI imports | Fast import for config-only access |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `app.py` | `create_app()` — FastAPI factory, routers, lifespan, middleware |
| `config.py` | WebConfig — server/game/AI/display/security, dev/production presets |
| `session.py` | GameSession, GameSessionManager — concurrent session management |
| `adapter.py` | MicroCivGameAdapter, MCTSAIAgent, BaselineAgentAdapter |
| `serializers.py` | Game state/turn/board/confidence serialization to JSON |
| `routes/game.py` | REST API — new game, move, undo, resign, pause, resume, valid actions |
| `routes/websocket.py` | WebSocket — real-time game play with JSON protocol |
| `routes/pages.py` | HTMX — Jinja2 template-driven HTML pages |
| `routes/replay.py` | Replay management endpoints |
| `__main__.py` | CLI entry point — `python -m alpha_zero.web` |

---

## Skills

- **FastAPI Development**: Routes, middleware, dependency injection, lifespan
- **WebSocket Protocol**: JSON command/response protocol with async handlers
- **Session Management**: Thread-safe concurrent user session tracking
- **Template Rendering**: Jinja2 templates with HTMX for dynamic updates
- **AI Integration**: Bridge trained models to web-playable agents

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **API Tester** | Test REST and WebSocket endpoints | Endpoint bugs or protocol issues |
| **Session Monitor** | Track session lifecycle and expiry | Session leaks or memory growth |
| **Performance Profiler** | Measure AI response time and page load speed | Web interface feels slow |

---

## Tools

| Tool | Purpose |
|------|---------|
| `python -m alpha_zero.web --preset dev` | Start dev server |
| `pytest tests/ -k web -v` | Test web components |

---

## Invariants

1. `create_app()` is idempotent and testable
2. Session expiry is enforced by background cleanup task
3. AI moves run in `asyncio.to_thread()` — never block the event loop
4. WebSocket protocol always sends `error` message on failure, never silently drops
5. All numpy types are sanitized before JSON serialization
6. Max concurrent sessions enforced by `GameSessionManager`

---

## Dependencies

- **Internal**: `alpha_zero.human_interface` (GameController, ReplayManager), `alpha_zero.envs.microciv`, `alpha_zero.core` (network, MCTS, actions), `alpha_zero.evaluation.baselines`
- **External**: `fastapi`, `uvicorn`, `jinja2`, `pydantic`, `numpy`, `threading`, `asyncio`
- **Consumed by**: End users via browser
