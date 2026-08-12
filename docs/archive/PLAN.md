# Phase 19: Human vs AI Web Interface — Implementation Plan

> **Goal**: Web-based UI for playing MicroCiv against the trained AlphaZero agent.
> **Branch**: `claude/plan-web-ui-agent-9u36i`
> **Status**: Planning

---

## 1. Architecture Overview

The web interface is a **thin wrapper** around the existing `human_interface/` module (GameController, ReplaySystem) and `evaluation/` baselines. No game logic is reimplemented — the web layer only handles HTTP/WebSocket transport and HTML rendering.

### Key Reuse

| Existing Component | Reuse Strategy |
|---|---|
| `GameController` | **Direct use** — instantiated per session; manages full game loop |
| `GameRecorder` / `ReplayManager` | **Direct use** — attached to sessions via callbacks |
| `HumanInterfaceConfig` | **Mapped from** `WebConfig` for controller initialization |
| `MicroCivActionSpace.decode_action()` | **Direct use** — human-readable action display |
| `BaselineFactory` / `RandomPlayer` / `HeuristicPlayer` | **Direct use** via adapter |
| `create_environment()` factory | **Direct use** — creates `MicroCivEnv` |
| `safe_load_checkpoint()` | **Direct use** — loads trained models |
| `get_logger()` / structured logging | **Direct use** — all web components use project logger |
| `MicroCivEnv._get_game_state()` | **Used via adapter** — structured board data for HTML |

### What's New

Two adapter classes bridge existing interfaces to `GameController`'s duck-typed protocol:

1. **`MicroCivGameAdapter`** — wraps `MicroCivEnv` so `reset()` returns obs (not tuple) and `get_valid_actions()` returns `list[int]` (not `np.ndarray` mask)
2. **`MCTSAIAgent`** — wraps network + MCTS so `select_action()` returns `{"action", "value", "policy"}` dict
3. **`BaselineAgentAdapter`** — wraps `RandomPlayer`/`HeuristicPlayer` to same dict protocol

These adapters are the only "glue" needed — everything else is direct reuse.

---

## 2. New Module Structure

```
alpha_zero/web/
├── __init__.py                    # Public exports (create_app, WebConfig)
├── config.py                      # WebConfig dataclass
├── app.py                         # FastAPI application factory
├── adapter.py                     # MicroCivGameAdapter, MCTSAIAgent, BaselineAgentAdapter
├── session.py                     # GameSession, GameSessionManager
├── serializers.py                 # State → JSON/dict serialization helpers
├── routes/
│   ├── __init__.py                # Router aggregation
│   ├── game.py                    # Game REST + HTMX routes
│   ├── websocket.py               # WebSocket endpoint
│   ├── replay.py                  # Replay REST routes
│   └── pages.py                   # HTML page routes (index, game)
├── templates/
│   ├── base.html                  # Base HTMX layout
│   ├── index.html                 # Lobby / new game page
│   ├── game.html                  # Main game page (composes partials)
│   └── partials/
│       ├── board.html             # Board grid partial (HTMX swappable)
│       ├── game_info.html         # Turn/player info
│       ├── move_history.html      # Move history panel
│       ├── confidence.html        # AI evaluation + move probabilities
│       ├── valid_actions.html     # Clickable valid actions
│       └── game_over.html         # Game over screen
├── static/
│   ├── css/
│   │   └── style.css              # Board themes, layout, responsive design
│   └── js/
│       └── game.js                # WebSocket client + HTMX config (minimal)
└── __main__.py                    # CLI: python -m alpha_zero.web

configs/
├── web.yaml                       # Default web config
└── web_dev.yaml                   # Dev preset (small board, fast AI)

tests/
├── unit/
│   ├── test_web_config.py         # WebConfig tests
│   ├── test_web_adapter.py        # Adapter tests
│   ├── test_web_session.py        # Session manager tests
│   └── test_web_routes.py         # Route tests (TestClient)
└── integration/
    └── test_web_e2e.py            # Full game over HTTP + WebSocket
```

---

## 3. Backend Components — Detailed Design

### 3.1 `alpha_zero/web/config.py` — WebConfig

```python
@dataclass
class WebConfig:
    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    log_level: str = "info"
    workers: int = 1

    # Game defaults
    board_size: int = 16
    max_turns: int = 200
    default_thinking_time: float = 5.0
    max_concurrent_sessions: int = 10
    session_timeout_seconds: float = 3600.0

    # AI defaults
    checkpoint_path: str | None = None
    use_mcts: bool = True
    mcts_simulations: int = 50
    ai_temperature: float = 0.0
    device: str = "cpu"

    # Display defaults
    board_theme: str = "classic"
    show_ai_evaluation: bool = True
    show_move_probabilities: bool = True
    show_mcts_stats: bool = True

    # Replay/save
    enable_replay: bool = True
    replay_dir: str = "replays"
    save_dir: str = "saves/web_games"

    # Security
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_yaml(cls, path) -> WebConfig: ...
    @classmethod
    def from_env(cls, base_config=None) -> WebConfig: ...  # CIV6AZ_WEB_ prefix
    @classmethod
    def from_preset(cls, preset: str) -> WebConfig: ...  # "dev", "production"
    def to_dict(self) -> dict: ...
    def to_human_interface_config(self) -> HumanInterfaceConfig: ...
```

Follows exact pattern from `HumanInterfaceConfig` — `from_yaml`, `from_env` with `CIV6AZ_WEB_` prefix, `__post_init__` validation, `to_dict`.

### 3.2 `alpha_zero/web/adapter.py` — Bridge Adapters

**MicroCivGameAdapter**: Bridges `MicroCivEnv` to `GameController`'s `game_interface` protocol.

Key adaptations:
- `reset(seed)` → calls `env.reset(seed=seed)`, returns `obs` dict (not tuple)
- `get_valid_actions()` → calls `env.get_valid_actions()`, returns `list[int]` from mask
- `step(action)` → passes through `env.step(action)` directly
- `get_board_data()` → calls `env._get_game_state()` for structured HTML rendering
- `action_space_size` property → `env.action_space.n`
- `clone()` → proxies to `env.clone()` for MCTS search

**MCTSAIAgent**: Bridges network + MCTS to `GameController`'s `ai_agent` protocol.

Key behavior:
- `select_action(obs, valid_actions=)` → returns `{"action": int, "value": float, "policy": ndarray}`
- Uses `Civ6MCTS.search()` when `use_mcts=True`, else `network.predict_single()`
- Holds reference to current env (set via `set_env()`) for MCTS cloning
- All inference under `torch.no_grad()`

**BaselineAgentAdapter**: Wraps `RandomPlayer`/`HeuristicPlayer` to dict protocol.

Key behavior:
- Delegates to `baseline.select_action(env, observation)` → returns `{"action": int, "value": None, "policy": None}`
- Holds env reference for baseline evaluation

### 3.3 `alpha_zero/web/session.py` — Session Manager

```python
@dataclass
class GameSession:
    session_id: str
    controller: GameController
    adapter: MicroCivGameAdapter
    ai_agent: MCTSAIAgent | BaselineAgentAdapter | None
    recorder: GameRecorder | None
    created_at: datetime
    last_activity: datetime
    config: WebConfig
    websockets: set[WebSocket]  # Connected WebSocket clients


class GameSessionManager:
    """Thread-safe manager for concurrent game sessions."""

    def __init__(self, config: WebConfig) -> None: ...

    async def create_session(
        self,
        human_player: int = 0,
        seed: int | None = None,
        opponent_type: str = "random",  # random, heuristic, neural
        board_size: int | None = None,  # override default
    ) -> GameSession: ...

    def get_session(self, session_id: str) -> GameSession | None: ...
    def remove_session(self, session_id: str) -> None: ...
    async def cleanup_expired(self) -> int: ...
    def active_session_count(self) -> int: ...
```

`create_session` flow:
1. Check `max_concurrent_sessions` limit
2. Create `MicroCivEnv` via `create_environment()` factory
3. Wrap in `MicroCivGameAdapter`
4. Create AI agent based on `opponent_type`:
   - `"random"` → `BaselineAgentAdapter(RandomPlayer())`
   - `"heuristic"` → `BaselineAgentAdapter(HeuristicPlayer())`
   - `"neural"` → `MCTSAIAgent` from checkpoint (via `safe_load_checkpoint()`)
5. Create `GameController(config, adapter, agent)`
6. Optionally attach `GameRecorder` via `controller.set_on_turn_complete()`
7. Call `controller.start_game(human_player=human_player, seed=seed)`
8. Return `GameSession`

### 3.4 `alpha_zero/web/serializers.py` — State Serialization

Helper functions to convert game state into JSON-safe dicts for API responses:

```python
def serialize_game_state(session: GameSession) -> dict:
    ...
    # Returns: session_id, phase, turn_number, current_player, human_player,
    #          board data (terrain grid, units, cities), valid_actions,
    #          game_over, winner


def serialize_turn_result(result: TurnResult, action_space: MicroCivActionSpace) -> dict:
    ...
    # Returns: action (decoded to human-readable), player, is_human,
    #          reward, ai_evaluation, top move probabilities, thinking_time


def serialize_board_data(adapter: MicroCivGameAdapter) -> dict:
    ...
    # Returns: terrain grid, unit positions/types/health, city positions/names


def serialize_move_history(history: list[TurnResult], action_space) -> list[dict]:
    ...
    # Returns: list of decoded turn results


def serialize_confidence(result: TurnResult, action_space) -> dict:
    ...
    # Returns: win_probability bar data, top-5 moves with probabilities
```

### 3.5 `alpha_zero/web/app.py` — FastAPI Application Factory

```python
def create_app(config: WebConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
```

Responsibilities:
- Initialize `GameSessionManager` as app state
- Add CORS middleware with configurable origins
- Include API routers (game, replay, websocket, pages)
- Mount static files and configure Jinja2Templates
- Set up background task for session cleanup (periodic)
- Configure structured logging
- Health check endpoint at `/api/health`

### 3.6 `alpha_zero/web/routes/game.py` — Game REST API

| Method | Path | Body | Response |
|--------|------|------|----------|
| `POST` | `/api/game/new` | `{human_player, seed, opponent_type, board_size}` | `{session_id, state}` |
| `GET` | `/api/game/{id}` | — | `{state}` (full game state) |
| `POST` | `/api/game/{id}/move` | `{action: int}` | `{turn_result, state}` |
| `POST` | `/api/game/{id}/undo` | — | `{success, state}` |
| `POST` | `/api/game/{id}/resign` | — | `{turn_result, state}` |
| `POST` | `/api/game/{id}/pause` | — | `{state}` |
| `POST` | `/api/game/{id}/resume` | — | `{state}` |
| `GET` | `/api/game/{id}/valid-actions` | — | `{actions: [{index, decoded}]}` |
| `GET` | `/api/game/{id}/summary` | — | `{summary}` |

Error responses: 404 for invalid session_id, 400 for invalid action, 409 for wrong turn phase.

**AI move execution**: When a human move is submitted and it becomes the AI's turn, the API runs the AI move in a thread pool (`asyncio.to_thread(controller.play_ai_move)`) and includes both results in the response. This avoids blocking the event loop during MCTS search.

### 3.7 `alpha_zero/web/routes/websocket.py` — WebSocket

```
WS /ws/game/{session_id}
```

Messages (client → server):
- `{"type": "move", "action": int}` — Submit human move
- `{"type": "undo"}` — Undo last move
- `{"type": "resign"}` — Resign game
- `{"type": "ping"}` — Keepalive

Messages (server → client):
- `{"type": "state", "data": {...}}` — Full state update
- `{"type": "move_result", "data": {...}}` — Human move result
- `{"type": "ai_thinking"}` — AI is computing
- `{"type": "ai_move", "data": {...}}` — AI move result
- `{"type": "game_over", "data": {...}}` — Game ended
- `{"type": "error", "message": str}` — Error
- `{"type": "pong"}` — Keepalive response

The WebSocket enables real-time "AI thinking..." indicator followed by instant AI move display, which is smoother than polling.

### 3.8 `alpha_zero/web/routes/replay.py` — Replay REST

| Method | Path | Response |
|--------|------|----------|
| `GET` | `/api/replay/list` | `{replays: [{game_id, date, result, turns}]}` |
| `GET` | `/api/replay/{game_id}` | `{recording}` (full game data) |
| `GET` | `/api/replay/{game_id}/turn/{n}` | `{turn_data}` |
| `DELETE` | `/api/replay/{game_id}` | `{deleted: true}` |

Delegates to existing `ReplayManager`.

### 3.9 `alpha_zero/web/routes/pages.py` — HTMX Page Routes

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/` | `index.html` — Lobby with "New Game" form |
| `GET` | `/game/{session_id}` | `game.html` — Main game page |
| `POST` | `/htmx/game/{id}/move` | HTML partials (board + info + history + confidence) |
| `POST` | `/htmx/game/{id}/undo` | HTML partials |
| `POST` | `/htmx/game/{id}/resign` | `game_over.html` partial |
| `GET` | `/htmx/game/{id}/board` | `board.html` partial (for polling during AI turn) |

The HTMX routes return rendered Jinja2 template partials. The page routes return full HTML pages.

### 3.10 `alpha_zero/web/__main__.py` — CLI Entry Point

```bash
python -m alpha_zero.web                          # Start with defaults
python -m alpha_zero.web --host 127.0.0.1         # Bind to localhost
python -m alpha_zero.web --port 3000              # Custom port
python -m alpha_zero.web --config configs/web.yaml # Load from YAML
python -m alpha_zero.web --preset dev             # Dev preset
python -m alpha_zero.web --checkpoint model.pt    # Use trained model
python -m alpha_zero.web --dry-run                # Validate config only
```

Uses `argparse` following the pattern from `alpha_zero/orchestrator/__main__.py`.

---

## 4. Frontend Design (HTMX + Jinja2)

### 4.1 Why HTMX over React

- No Node.js build toolchain — fits Python-centric project
- Server-side rendering — Jinja2 templates are already used in the ecosystem
- HTMX handles partial page updates via `hx-post`, `hx-swap`, `hx-trigger`
- WebSocket for real-time AI turn updates (HTMX has native `hx-ws` support)
- Minimal JS — only `game.js` (~100 lines) for WebSocket management

### 4.2 Board Visualization (`partials/board.html`)

The 16×16 grid renders as a CSS Grid:
- Each cell is a `<div>` with CSS classes for terrain, units, cities
- Valid moves are highlighted and clickable (`hx-post` with action index)
- Last move is highlighted (different border/glow)
- Player units/cities color-coded (blue = human, red = AI)
- Icons: text symbols (W=Warrior, S=Settler, A=Archer, C=City) with CSS styling

Board data comes from `adapter.get_board_data()` which calls `env._get_game_state()`.

### 4.3 Board Themes (CSS)

Maps to existing `BoardTheme` enum:
- `.theme-classic`: Green terrain, blue/red units, light background
- `.theme-dark`: Dark background, muted terrain, bright units
- `.theme-colorblind`: Distinguishable by shape + pattern, not just color
- `.theme-minimal`: Simple borders, no backgrounds, high contrast text

Theme is a CSS class on the board container, switchable via dropdown.

### 4.4 Move History Panel (`partials/move_history.html`)

Scrollable list showing each turn:
- Turn number
- Player icon (human/AI)
- Decoded action (e.g., "Move Warrior NE", "Attack Archer S", "Found City")
- AI evaluation bar (if AI move)
- Thinking time (for AI moves)

Uses `MicroCivActionSpace.decode_action()` for human-readable descriptions.

### 4.5 Confidence Display (`partials/confidence.html`)

- **Win probability**: Horizontal bar from -1.0 (AI winning) to +1.0 (human winning)
- **Top moves**: Sorted list of top-5 AI-considered moves with probability bars
- **MCTS stats**: Simulations count, search depth, thinking time
- Only shown when `show_ai_evaluation` / `show_move_probabilities` are enabled

### 4.6 Game Flow (HTMX + WebSocket)

```
1. User visits / → index.html (New Game form)
2. User submits form → POST /api/game/new → redirect to /game/{id}
3. Game page loads → game.html with initial board
4. WebSocket connects to /ws/game/{id}
5. Human clicks valid move cell → WS sends {"type": "move", "action": N}
6. Server processes human move → WS sends {"type": "move_result", ...}
7. If AI turn → WS sends {"type": "ai_thinking"}
8. AI computes in thread pool → WS sends {"type": "ai_move", ...}
9. Client updates board, history, confidence panels via HTMX oob-swap
10. Repeat 5-9 until game over
11. WS sends {"type": "game_over", ...} → show game_over partial
```

Fallback: If WebSocket is unavailable, HTMX polling on `/htmx/game/{id}/board` every 1s during AI turn.

---

## 5. Configuration Files

### 5.1 `configs/web.yaml`

```yaml
web:
  host: "0.0.0.0"
  port: 8080
  debug: false
  log_level: "info"
  workers: 1

  game:
    board_size: 16
    max_turns: 200
    default_thinking_time: 5.0
    max_concurrent_sessions: 10
    session_timeout_seconds: 3600

  ai:
    checkpoint_path: null
    use_mcts: true
    mcts_simulations: 50
    ai_temperature: 0.0
    device: "cpu"

  display:
    board_theme: "classic"
    show_ai_evaluation: true
    show_move_probabilities: true
    show_mcts_stats: true

  replay:
    enable_replay: true
    replay_dir: "replays"
    save_dir: "saves/web_games"
```

### 5.2 `configs/web_dev.yaml`

```yaml
web:
  host: "127.0.0.1"
  port: 8080
  debug: true
  log_level: "debug"

  game:
    board_size: 8
    max_turns: 50
    default_thinking_time: 1.0
    max_concurrent_sessions: 3
    session_timeout_seconds: 600

  ai:
    use_mcts: false
    mcts_simulations: 10
```

---

## 6. Dependencies

### New packages:

| Package | Purpose | Version |
|---------|---------|---------|
| `fastapi` | Web framework | `>=0.104.0` |
| `uvicorn[standard]` | ASGI server | `>=0.24.0` |
| `websockets` | WebSocket support | `>=12.0` |
| `jinja2` | Template rendering | `>=3.1.0` |
| `python-multipart` | Form parsing | `>=0.0.6` |
| `httpx` | Test client (dev only) | `>=0.25.0` |

### pyproject.toml changes:

```toml
# New optional dependency group
web = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "websockets>=12.0",
    "jinja2>=3.1.0",
    "python-multipart>=0.0.6",
]

# Update 'all' to include web
all = [
    "alpha_zero[dev]",
    "alpha_zero[gcp]",
    "alpha_zero[distributed]",
    "alpha_zero[llm]",
    "alpha_zero[web]",
]

# Add httpx to dev deps
dev = [
    ...,
    "httpx>=0.25.0",
]

# Add mypy override for web module
[[tool.mypy.overrides]]
module = "alpha_zero.web.*"
disallow_untyped_defs = true
disallow_incomplete_defs = true

# Add ruff ignore for templates
[tool.ruff.lint.per-file-ignores]
"alpha_zero/web/templates/*" = ["ALL"]

# Add pytest marker
markers = [
    ...,
    "web: marks web interface tests",
]
```

---

## 7. File-by-File Implementation Order

### Phase A: Foundation (config + adapters)

| # | File | Description | LOC est. |
|---|------|-------------|----------|
| 1 | `alpha_zero/web/__init__.py` | Public exports, lazy imports | ~30 |
| 2 | `alpha_zero/web/config.py` | `WebConfig` dataclass | ~200 |
| 3 | `configs/web.yaml` | Default config YAML | ~30 |
| 4 | `configs/web_dev.yaml` | Dev preset YAML | ~20 |
| 5 | `alpha_zero/web/adapter.py` | `MicroCivGameAdapter`, `MCTSAIAgent`, `BaselineAgentAdapter` | ~250 |
| 6 | `tests/unit/test_web_config.py` | Config tests | ~150 |
| 7 | `tests/unit/test_web_adapter.py` | Adapter tests | ~200 |

### Phase B: Session management + serialization

| # | File | Description | LOC est. |
|---|------|-------------|----------|
| 8 | `alpha_zero/web/serializers.py` | State/turn/board serialization | ~150 |
| 9 | `alpha_zero/web/session.py` | `GameSession`, `GameSessionManager` | ~200 |
| 10 | `tests/unit/test_web_session.py` | Session manager tests | ~200 |

### Phase C: API routes

| # | File | Description | LOC est. |
|---|------|-------------|----------|
| 11 | `alpha_zero/web/routes/__init__.py` | Router aggregation | ~20 |
| 12 | `alpha_zero/web/routes/game.py` | Game REST endpoints | ~200 |
| 13 | `alpha_zero/web/routes/websocket.py` | WebSocket endpoint | ~150 |
| 14 | `alpha_zero/web/routes/replay.py` | Replay REST endpoints | ~80 |
| 15 | `alpha_zero/web/routes/pages.py` | HTML page routes | ~80 |
| 16 | `tests/unit/test_web_routes.py` | Route unit tests | ~300 |

### Phase D: Frontend templates + static assets

| # | File | Description | LOC est. |
|---|------|-------------|----------|
| 17 | `alpha_zero/web/static/css/style.css` | Board themes, layout, responsive | ~300 |
| 18 | `alpha_zero/web/static/js/game.js` | WebSocket client, HTMX config | ~120 |
| 19 | `alpha_zero/web/templates/base.html` | Base layout (HTMX CDN) | ~50 |
| 20 | `alpha_zero/web/templates/index.html` | Lobby / new game form | ~80 |
| 21 | `alpha_zero/web/templates/game.html` | Main game page | ~100 |
| 22 | `alpha_zero/web/templates/partials/board.html` | Board grid | ~80 |
| 23 | `alpha_zero/web/templates/partials/game_info.html` | Turn/player info | ~30 |
| 24 | `alpha_zero/web/templates/partials/move_history.html` | Move history | ~50 |
| 25 | `alpha_zero/web/templates/partials/confidence.html` | AI confidence | ~60 |
| 26 | `alpha_zero/web/templates/partials/valid_actions.html` | Action list | ~40 |
| 27 | `alpha_zero/web/templates/partials/game_over.html` | Game over | ~40 |

### Phase E: App factory + CLI + integration

| # | File | Description | LOC est. |
|---|------|-------------|----------|
| 28 | `alpha_zero/web/app.py` | FastAPI app factory | ~120 |
| 29 | `alpha_zero/web/__main__.py` | CLI entry point | ~80 |
| 30 | `tests/integration/test_web_e2e.py` | E2E integration tests | ~250 |

### Phase F: Project integration

| # | File | Description | LOC est. |
|---|------|-------------|----------|
| 31 | `pyproject.toml` | Add web deps, mypy overrides, markers | ~20 lines changed |
| 32 | `requirements.txt` | Add FastAPI, uvicorn, etc. | ~6 lines added |
| 33 | `alpha_zero/human_interface/__init__.py` | Remove deprecation warning | ~2 lines changed |
| 34 | `CLAUDE.md` | Add web commands, architecture decision | ~40 lines added |

**Total: ~34 files, ~3,100 LOC estimated**

---

## 8. Testing Strategy

### 8.1 Unit Tests (4 files)

**`test_web_config.py`**:
- Default initialization with all defaults
- Custom initialization with overrides
- Validation (negative port, invalid theme, etc.)
- `from_yaml()` loading
- `from_env()` with `CIV6AZ_WEB_` prefix
- `from_preset("dev")` and `from_preset("production")`
- `to_dict()` round-trip
- `to_human_interface_config()` mapping

**`test_web_adapter.py`**:
- `MicroCivGameAdapter.reset()` returns dict (not tuple)
- `MicroCivGameAdapter.get_valid_actions()` returns `list[int]`
- `MicroCivGameAdapter.step()` returns 5-tuple
- `MicroCivGameAdapter.get_board_data()` returns structured dict
- `MicroCivGameAdapter.action_space_size` returns int
- `MCTSAIAgent.select_action()` returns dict with required keys
- `MCTSAIAgent` works without MCTS (policy-only mode)
- `BaselineAgentAdapter` wraps RandomPlayer correctly
- `BaselineAgentAdapter` wraps HeuristicPlayer correctly

**`test_web_session.py`**:
- Create session with random opponent
- Create session with heuristic opponent
- Session timeout / expiration
- Max concurrent sessions enforcement
- Get/remove session
- Cleanup expired sessions
- Session isolation (multiple sessions don't interfere)

**`test_web_routes.py`** (uses `httpx.AsyncClient` + FastAPI `TestClient`):
- `POST /api/game/new` → 200 with session_id
- `GET /api/game/{id}` → 200 with state
- `GET /api/game/invalid` → 404
- `POST /api/game/{id}/move` with valid action → 200
- `POST /api/game/{id}/move` with invalid action → 400
- `POST /api/game/{id}/move` on AI's turn → 409
- `POST /api/game/{id}/undo` → 200
- `POST /api/game/{id}/resign` → 200 with game_over
- `GET /api/game/{id}/valid-actions` → list of decoded actions
- `GET /api/game/{id}/summary` → game stats
- `GET /api/replay/list` → list
- `GET /api/health` → 200

### 8.2 Integration Tests (1 file)

**`test_web_e2e.py`**:
- Play complete game via REST API (create → move → ... → game_over)
- WebSocket connection and move exchange
- WebSocket AI thinking + move response
- Replay: play game → list replays → load replay → step through turns
- Concurrent sessions: create 3, play in interleaved order
- Session cleanup after timeout

### 8.3 Test Fixtures

```python
@pytest.fixture
def web_config():
    return WebConfig(
        board_size=8, max_turns=50, mcts_simulations=2, default_thinking_time=0.1, use_mcts=False
    )


@pytest.fixture
def test_app(web_config):
    return create_app(web_config)


@pytest.fixture
async def client(test_app):
    async with httpx.AsyncClient(app=test_app, base_url="http://test") as c:
        yield c


@pytest.fixture
def game_adapter():
    env = MicroCivEnv(board_size=8, max_turns=50)
    return MicroCivGameAdapter(env)
```

### 8.4 Test Markers

```python
pytest.mark.web  # All web tests
pytest.mark.asyncio  # Async tests (auto mode enabled in pyproject.toml)
```

---

## 9. Logging & Debugging

All web components use the project's structured logger:

```python
from alpha_zero.utils.logging_config import get_logger

logger = get_logger(__name__)
```

Logging points:
- `app.py`: Server startup/shutdown, config loaded
- `session.py`: Session created/removed/expired, concurrent session count
- `routes/game.py`: Move submitted, invalid action, game over, AI thinking time
- `routes/websocket.py`: Connection opened/closed, message received/sent, errors
- `adapter.py`: Model loaded, MCTS search started/completed, inference time
- `config.py`: Config loaded from YAML/env, validation warnings

Debug mode (`config.debug=True`) enables:
- `DEBUG` log level
- FastAPI debug mode (auto-reload, detailed errors)
- Request/response logging middleware
- MCTS search tree depth logging

---

## 10. Ruff / MyPy / Numpy Considerations

### Ruff

- Line length ≤ 100 chars (existing rule)
- All web module files follow existing `select` rules: `["E", "F", "W", "I", "B", "C4", "UP", "N", "ANN", "S", "RUF"]`
- Templates excluded from linting: `"alpha_zero/web/templates/*" = ["ALL"]`
- Type annotations on all public functions (ANN rule)
- No `assert` outside tests (S101 rule)
- Security: no hardcoded secrets (S105), validated inputs

### MyPy

- `disallow_untyped_defs = true` for `alpha_zero.web.*`
- `ignore_missing_imports = true` for fastapi/uvicorn (optional deps)
- All public interfaces fully typed
- Use `from __future__ import annotations` for forward references
- Careful with numpy array types: `np.ndarray` not `np.ndarray[Any, Any]`

### Numpy

- Use `np.ndarray` in type hints (not `numpy.typing.NDArray` to match codebase)
- Convert numpy arrays to Python lists in serializers before JSON encoding (`arr.tolist()`)
- Use `np.where(mask > 0)[0].tolist()` pattern for action mask conversion
- Avoid numpy scalar JSON serialization issues: cast to `float()` / `int()` explicitly

---

## 11. Redundant Code Cleanup

During implementation, address:

1. **`human_interface/__init__.py`**: Remove deprecation warning (module is now activated)
2. **`human_interface/visualizer.py`**: Keep as-is (still useful for terminal mode), but the web templates replace its HTML-equivalent rendering
3. **`human_interface/input_handler.py`**: Keep as-is (terminal input), web uses API endpoints instead
4. **Duplicate config patterns**: `WebConfig` reuses `HumanInterfaceConfig` internally via `to_human_interface_config()`, avoiding duplication
5. **Any dead imports** found during implementation in touched files

---

## 12. Key Architectural Decisions

| Decision | Rationale |
|---|---|
| Adapter pattern for env/agent | `GameController` uses duck typing; adapters bridge interface gaps cleanly |
| HTMX over React | No build toolchain, server-side rendering, fits Python project |
| WebSocket for AI turns only | AI thinking is async; HTMX handles everything else |
| Session-based state | Each game gets own controller + env; simple memory model |
| Optional dependency group | `pip install -e ".[web]"` — training users unaffected |
| Thread pool for AI moves | MCTS is CPU-bound; `asyncio.to_thread()` avoids blocking event loop |
| Shared network across sessions | Single model loaded once, shared read-only across sessions |
| JSON API + HTMX dual paths | REST for programmatic access, HTMX for browser |

---

## 13. Verification Checklist

```bash
# 1. Install
pip install -e ".[web,dev]"

# 2. Lint
ruff check alpha_zero/web/

# 3. Type check
mypy alpha_zero/web/

# 4. Unit tests
pytest tests/unit/test_web_config.py tests/unit/test_web_adapter.py \
       tests/unit/test_web_session.py tests/unit/test_web_routes.py -v

# 5. Integration tests
pytest tests/integration/test_web_e2e.py -v

# 6. Coverage
pytest tests/unit/test_web_*.py tests/integration/test_web_e2e.py \
       --cov=alpha_zero.web --cov-report=term-missing

# 7. Import smoke test
python -c "from alpha_zero.web import create_app, WebConfig"

# 8. Dry run
python -m alpha_zero.web --dry-run

# 9. Full test suite (no regressions)
pytest tests/ -v --tb=short

# 10. Manual: start server and play a game
python -m alpha_zero.web --preset dev
# Open http://127.0.0.1:8080
```

---

## 14. Risk Mitigation

| Risk | Mitigation |
|---|---|
| MCTS blocks event loop | Run in `asyncio.to_thread()` |
| Memory from concurrent sessions | `max_concurrent_sessions` config + timeout cleanup |
| WebSocket disconnect during AI turn | Graceful cleanup; result discarded or session paused |
| No trained model available | Fall back to random/heuristic opponents |
| FastAPI not installed | Lazy imports with clear error message |
| State inconsistency on undo | `GameController.undo_move()` already handles this |
| Cross-browser WebSocket issues | HTMX polling fallback for AI turns |

---

*Plan created: 2026-02-14*
*Estimated files: 34*
*Estimated LOC: ~3,100*
