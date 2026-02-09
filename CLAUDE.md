# CLAUDE.md - Civ6 AlphaZero Agentic Coding System

> **Design Principle:** This document treats prompting as constraint programming, not instruction writing. It defines the feasible region, objective function, and search parameters for autonomous agent implementation.

---

## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

```
I am building: An AlphaZero-style reinforcement learning AI for Civilization 6 using self-play,
Monte Carlo Tree Search, and deep neural networks.
```

### 1.2 Success Criteria (Mechanically Verifiable)

```
This succeeds when:
- [ ] All tests pass: `pytest tests/ -v` exits with code 0
- [ ] Coverage >= 80%: `pytest --cov=alpha_zero --cov-report=term-missing` shows >= 80%
- [ ] Type checking passes: `mypy alpha_zero/` exits with code 0
- [ ] Linting passes: `ruff check alpha_zero/` exits with code 0
- [ ] Training smoke test: `python -m alpha_zero.training_civ6 --dry_run` completes
- [ ] Distributed smoke test: `python -m alpha_zero.distributed.coordinator --dry_run` completes
- [ ] Orchestrator smoke test: `python -m alpha_zero.orchestrator --preset fast --dry_run` completes
- [ ] LLM integration test: `python -c "from alpha_zero.utils import create_llm_client"` imports
- [ ] RAG integration test: `python -c "from alpha_zero.orchestrator import RAGIntegration"` imports
```

### 1.3 Problem Description

**Paragraph 1 - Core Problem:**
This system implements AlphaZero for Civilization 6, a complex 4X strategy game with astronomical
state and action spaces (~10^200 states, ~10^6 actions per turn). The AI learns through self-play,
using a neural network to evaluate game states and guide Monte Carlo Tree Search. The coordination
logic involves: game simulation (MicroCiv/Unciv), neural network training, MCTS search, and
distributed self-play across multiple actors.

**Paragraph 2 - Data Flows:**
Data flows from game environments (MicroCiv for training, Unciv for production) through state
encoders (13-channel MicroCiv, 84-channel full Civ6) to the neural network. The network outputs
policy distributions and value estimates. MCTS uses these to build search trees. Training samples
(state, policy, value) accumulate in replay buffers. The learner samples batches for gradient
updates. State synchronized across components: network weights (SharedStorage), replay buffer
(DistributedReplayBuffer), training metrics (MetricsTracker).

**Paragraph 3 - Failure Modes:**
Critical invariants: (1) replay buffer never exceeds memory limits, (2) network weights are
consistent across all actors, (3) MCTS visits are properly backpropagated, (4) action masking
prevents invalid actions, (5) distributed training maintains ordering guarantees. Failure modes
include: OOM from buffer overflow, stale weights causing training divergence, MCTS selection errors
from numerical instability, and race conditions in distributed components.

---

## SECTION 2: FEASIBLE REGION (Constraints)

### 2.1 Hard Constraints (Violations = Failure)

```
- Language/Runtime: Python 3.10+ (type hints required)
- Required Dependencies:
  - torch >= 1.13.0 (neural networks)
  - numpy >= 1.23.0 (numerical operations)
  - gymnasium >= 0.29.0 (RL environments)
  - pyyaml >= 6.0 (configuration)
  - pytest >= 7.0.0 (testing)
- Security:
  - NO hardcoded secrets (use .env or environment variables)
  - API keys loaded from environment only
  - All user inputs validated before processing
- Compatibility:
  - Must run on CPU (GPU optional)
  - Cross-platform (Windows, Linux, macOS)
  - Python multiprocessing for distributed training
```

### 2.2 Soft Constraints (Preferences)

```
- Style: Ruff linting, 100 character line limit
- Architecture:
  - Prefer composition over inheritance
  - Dataclasses for configuration objects
  - Async I/O for LLM calls
  - Type hints on all public interfaces
- Performance:
  - Batch operations where possible
  - Minimize memory allocations in hot paths
  - Use torch.no_grad() for inference
- Testing:
  - pytest with fixtures
  - Unit tests for all public methods
  - Integration tests for cross-module flows
  - Coverage >= 80% on core modules
- Logging:
  - Use alpha_zero.utils.logging module
  - Structured logging with MetricsTracker
  - Debug-level tracing for development
```

### 2.3 Anti-Constraints (Explicit Freedoms)

```
You ARE permitted to:
- Restructure existing file organization for better modularity
- Add dependencies from PyPI (update requirements.txt)
- Refactor adjacent code for consistency
- Choose implementation patterns not specified
- Create new test files and fixtures
- Add new configuration parameters to configs/civ6_config.yaml
- Implement helper functions and utilities
- Add logging statements for debugging
- Create new submodules within alpha_zero/
```

---

## SECTION 3: PERMISSION ARCHITECTURE

### 3.1 Scope (What You Can Touch)

```
IN SCOPE:
- All files in alpha_zero/
- All files in tests/
- Configuration files in configs/
- Root-level Python files (setup.py, pyproject.toml)
- Documentation files (CLAUDE.md, README.md)
- Environment templates (.env.template)
- Requirements files (requirements.txt)

OUT OF SCOPE:
- .git/ directory
- __pycache__/ directories
- .env file (contains secrets - read only)
- checkpoints/ (trained model files)
- External dependencies in site-packages
```

### 3.2 Autonomy Level

```
AUTONOMOUS (proceed without asking):
- File creation/deletion within scope
- Dependency installation (pip install)
- Running tests (pytest)
- Code refactoring within existing patterns
- Adding logging and debug statements
- Creating new test files
- Updating configuration defaults
- Documentation updates

CONFIRM FIRST (ask before proceeding):
- Architectural changes affecting >3 modules
- Breaking API changes to public interfaces
- Deletions of >100 lines of code
- Changes to distributed training coordination
- Modifications to neural network architecture
- New external service integrations

PROHIBITED (do not attempt):
- Committing directly to main branch
- Modifying .env with real credentials
- External API calls with side effects (except LLM)
- Deleting checkpoint files
- Running training with >1000 games without confirmation
```

### 3.3 Resource Budget

```
- Max iterations before requesting guidance: 5
- Max files to modify in single pass: 20
- Max lines in single file edit: 500
- Time-boxed exploration: ≤10 min research before asking
- Test execution timeout: 300 seconds
```

---

## SECTION 4: FEEDBACK LOOP SPECIFICATION

### 4.1 Verification Commands

```bash
# After writing code, run in this order:

# 1. Lint check
ruff check alpha_zero/

# 2. Type check
mypy alpha_zero/ --ignore-missing-imports

# 3. Unit tests (fast feedback)
pytest tests/unit/ -v --tb=short

# 4. Integration tests
pytest tests/integration/ -v --tb=short

# 5. Full test suite with coverage
pytest tests/ -v --cov=alpha_zero --cov-report=term-missing

# 6. Smoke tests
python -m alpha_zero.training_civ6 --dry_run
python -m alpha_zero.distributed.coordinator --dry_run
```

### 4.2 Error Handling Protocol

```
ON LINT FAILURE:
→ Read error output
→ Fix automatically using ruff guidance
→ Re-run: ruff check alpha_zero/

ON TYPE ERROR:
→ Analyze mypy error message
→ Add/fix type annotations
→ Consider if Optional or Union needed
→ Re-run: mypy alpha_zero/

ON TEST FAILURE:
→ Read full traceback
→ Identify root cause (implementation vs test bug)
→ If implementation bug: fix source code
→ If test bug: fix test (with justification)
→ Re-run specific test: pytest tests/path/to/test.py::test_name -v

ON IMPORT ERROR:
→ Check if dependency is in requirements.txt
→ If missing: add and run pip install -r requirements.txt
→ If version conflict: resolve in requirements.txt

ON REPEATED FAILURE (same error 3x):
→ Stop attempting fix
→ Document what was tried
→ Report analysis with hypothesis
→ Request human guidance
```

### 4.3 Success Verification

```
Before reporting completion:
1. All verification commands pass (lint, type, tests)
2. Coverage check: pytest --cov=alpha_zero shows >= 80% on changed modules
3. Smoke test: python -m alpha_zero.training_civ6 --dry_run
4. Import check: python -c "from alpha_zero import <new_module>"
5. Generate brief summary of changes made
6. Update CLAUDE.md if new commands/patterns discovered
```

---

## SECTION 5: CONTEXT PERSISTENCE

### 5.1 Session Memory Updates

When making changes, update the relevant sections below:

#### Build Commands

| Command | Purpose |
|---------|---------|
| `pytest tests/unit/ -v` | Run unit tests |
| `pytest tests/integration/ -v` | Run integration tests |
| `pytest tests/ -v --tb=short` | Run all tests |
| `pytest tests/ -v --cov=alpha_zero` | Run tests with coverage |
| `ruff check alpha_zero/` | Lint Python code |
| `mypy alpha_zero/` | Type check |
| `python -m alpha_zero.training_civ6 --dry_run` | Training smoke test |
| `python -m alpha_zero.distributed.coordinator --dry_run` | Distributed smoke test |
| `python -m alpha_zero.orchestrator --help` | Show orchestrator options |
| `python -m alpha_zero.orchestrator --preset fast --dry_run` | Fast preset smoke test |
| `python -m alpha_zero.orchestrator --config configs/orchestrator.yaml --dry_run` | Config file smoke test |

#### Cloud Deployment Commands

| Command | Purpose |
|---------|---------|
| `make build-all` | Build all Docker images |
| `make push` | Push images to Artifact Registry |
| `make local-test` | Run local docker-compose test |
| `make local-smoke` | Run smoke test (1 game + 1 step) |
| `docker-compose up -d` | Start local testing environment |
| `python -m infra.vertex_launcher --mode=selfplay --iteration=0 --dry-run` | Self-play job dry run |
| `python -m infra.vertex_launcher --mode=train --iteration=0 --dry-run` | Training job dry run |
| `python -m infra.vertex_launcher --mode=eval --iteration=0 --dry-run` | Evaluation job dry run |
| `python -m alpha_zero.cloud.selfplay_worker --help` | Self-play worker options |
| `python -m alpha_zero.cloud.trainer --help` | Trainer options |
| `python -m alpha_zero.cloud.evaluator --help` | Evaluator options |

#### Test Commands

| Command | Purpose |
|---------|---------|
| `pytest tests/unit/test_environment.py -v` | Test MicroCiv environment |
| `pytest tests/unit/test_network.py -v` | Test neural networks |
| `pytest tests/unit/test_mcts.py -v` | Test MCTS algorithm |
| `pytest tests/unit/test_config.py -v` | Test configuration system |
| `pytest tests/unit/test_logging.py -v` | Test logging infrastructure |
| `pytest tests/unit/test_orchestrator_config.py -v` | Test orchestrator configuration |
| `pytest tests/unit/test_scheduler.py -v` | Test hyperparameter scheduler |
| `pytest tests/unit/test_checkpointer.py -v` | Test checkpoint manager |
| `pytest tests/unit/test_factories.py -v` | Test component factories |
| `pytest tests/unit/test_rag_integration.py -v` | Test RAG integration |
| `pytest tests/unit/test_strategy_advisor.py -v` | Test strategy advisor |
| `pytest tests/unit/test_storage_backend.py -v` | Test storage backends |
| `pytest tests/unit/test_replay_buffer.py -v` | Test replay buffers |
| `pytest -k "test_name" -v` | Run specific test by name |

### 5.2 Architecture Decisions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-10 | 13-channel MicroCiv encoder | Simplified encoding for faster Phase 1 training |
| 2026-01-10 | Hierarchical action space | Reduces effective branching factor from ~10^6 to ~10^3 |
| 2026-01-10 | SimpleMCTS + Civ6MCTS | SimpleMCTS for testing, Civ6MCTS for training |
| 2026-01-10 | ResNet backbone | Proven architecture from AlphaZero, good spatial reasoning |
| 2026-01-11 | Multi-head attention in network | Better spatial reasoning for complex game states |
| 2026-01-11 | Process-based distributed training | Avoid GIL, maximize CPU utilization for self-play |
| 2026-01-11 | Unciv JSON interface | Open-source Civ V clone with accessible save format |
| 2026-01-11 | YAML config with env overrides | Flexible configuration for different environments |
| 2026-01-19 | LLM client with provider fallback | Unified API for OpenAI/Anthropic with auto-failover |
| 2026-01-19 | Test runner utility | Programmatic test execution with coverage tracking |
| 2026-01-19 | Agent tools module | Reusable utilities for Claude Code sub-agents |
| 2026-01-20 | Unified Training Orchestrator | Single entry point for all training modes with dynamic config |
| 2026-01-20 | RAG-LLM integration layer | ChromaDB knowledge base with strategy advisor for training insights |
| 2026-01-20 | Hyperparameter scheduling | Multiple schedule types (warmup_cosine, step, exponential) |
| 2026-01-20 | Factory pattern for components | Testable, flexible component creation |
| 2026-02-01 | Abstract StorageBackend interface | Unified storage API for local and GCS backends |
| 2026-02-01 | GCS-backed replay buffer | Streaming IterableDataset for memory-efficient cloud training |
| 2026-02-01 | Cloud-native worker components | Self-play, trainer, evaluator containerized for Vertex AI |
| 2026-02-01 | Preemption handling | Graceful shutdown on Spot VM preemption with trajectory flush |
| 2026-02-01 | Budget machine profiles | Pre-configured profiles for T4/L4/A100 GPUs with cost estimates |

### 5.3 Known Issues

| Issue | Workaround |
|-------|------------|
| No GPU in dev environment | Training works on CPU (slower) |
| SimpleMCTS random rollout is weak | Use neural network MCTS after initial training |
| Windows multiprocessing spawn | Use `if __name__ == "__main__"` guard |
| pytest-xdist not always installed | Tests run without parallel flag if missing |

### 5.4 Bug Fixes (2026-01-20)

| Bug | Fix |
|-----|-----|
| Action mask indices mismatch | Changed to player-relative indices in `civ6_actions.py` |
| Evaluation `_env` attribute error | Always create fresh wrapped environment per game |
| Value tensor shape mismatch | Added `value.unsqueeze(1)` in training loss |
| Curriculum `record_game` API | Changed `win=` to `won=` parameter |

### 5.5 Information to Preserve

- Build/test commands that work
- Non-obvious environment setup steps
- Architectural decisions and their rationale
- Gotchas discovered during implementation
- LLM API keys are in .env (never commit)

---

## SECTION 6: EXECUTION PROTOCOL

### 6.1 Initial Actions (Always Do First)

```bash
# 1. Read this file (CLAUDE.md)

# 2. Check environment
python --version  # Should be 3.10+

# 3. Install dependencies if needed
pip install -r requirements.txt
pip install -e .

# 4. Scan project structure
find . -type f -name "*.py" | grep -v __pycache__ | head -50

# 5. Run existing tests to establish baseline
pytest tests/ -v --tb=line | head -50

# 6. Check for any failing tests
pytest tests/ -v --tb=short 2>&1 | grep -E "(PASSED|FAILED|ERROR)"
```

### 6.2 Implementation Order

```
1. UNDERSTAND: Read existing code before modifying
   - Identify patterns used in similar code
   - Check for existing utilities that can be reused
   - Review related tests for expected behavior

2. IMPLEMENT: Write smallest working version first
   - Start with function signature and docstring
   - Implement core logic without edge cases
   - Add type hints

3. TEST: Write tests alongside implementation
   - Unit test for each public method
   - Test edge cases and error conditions
   - Mock external dependencies

4. VERIFY: Run verification loop
   - ruff check (lint)
   - mypy (types)
   - pytest (tests)

5. REFACTOR: Clean up only if needed
   - Only refactor if tests pass
   - Keep changes minimal
   - Preserve existing APIs

6. DOCUMENT: Update this file
   - Add new commands to build table
   - Document architectural decisions
   - Note any gotchas discovered
```

### 6.3 Completion Checklist

```
□ All success criteria from Section 1.2 met
□ All verification commands from Section 4.1 pass
□ CLAUDE.md updated with new commands/decisions
□ Summary of changes provided to user
□ Known limitations documented
□ No hardcoded values (check for API keys, paths, magic numbers)
□ Tests added for new functionality
□ Type hints on all new public interfaces
```

---

## SECTION 7: PROJECT STRUCTURE

```
Civ6-AlphaZero/
├── alpha_zero/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── microciv.py          # Simplified training environment
│   │   └── civ6_encoder.py      # State → Tensor encoding
│   ├── core/
│   │   ├── __init__.py
│   │   ├── civ6_network.py      # Neural networks
│   │   ├── civ6_mcts.py         # MCTS implementation
│   │   └── civ6_actions.py      # Action space definition
│   ├── distributed/             # Distributed training
│   │   ├── __init__.py
│   │   ├── actor.py             # Self-play actors
│   │   ├── learner.py           # Central learner
│   │   ├── coordinator.py       # Training orchestration
│   │   └── shared_storage.py    # Weight & buffer sharing
│   ├── storage/                 # Storage backends (NEW)
│   │   ├── __init__.py
│   │   ├── backend.py           # Abstract StorageBackend interface
│   │   ├── config.py            # StorageConfig with GCS settings
│   │   ├── local_backend.py     # Local filesystem implementation
│   │   ├── gcs_backend.py       # GCS implementation
│   │   ├── models.py            # Metadata dataclasses
│   │   ├── replay_buffer.py     # Streaming replay buffer
│   │   └── factory.py           # create_storage_backend
│   ├── cloud/                   # Cloud-native components (NEW)
│   │   ├── __init__.py
│   │   ├── config.py            # CloudConfig, MachineProfile
│   │   ├── preemption.py        # PreemptionHandler for Spot VMs
│   │   ├── selfplay_worker.py   # Self-play worker entry point
│   │   ├── trainer.py           # Trainer entry point
│   │   └── evaluator.py         # Evaluator entry point
│   ├── orchestrator/            # Unified Training Orchestrator
│   │   ├── __init__.py
│   │   ├── config.py            # OrchestratorConfig with schedules
│   │   ├── orchestrator.py      # TrainingOrchestrator main class
│   │   ├── scheduler.py         # HyperparameterScheduler
│   │   ├── checkpointer.py      # CheckpointManager
│   │   ├── rag_integration.py   # RAG-LLM integration layer
│   │   ├── strategy_advisor.py  # LLM-based strategy advisor
│   │   └── __main__.py          # CLI entry point
│   ├── interfaces/              # Game interfaces
│   │   ├── __init__.py
│   │   └── unciv_interface.py   # Unciv JSON interface
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── logging.py           # Logging infrastructure
│   │   ├── config.py            # Configuration management
│   │   ├── llm_client.py        # LLM API integration
│   │   ├── factories.py         # Component factory functions
│   │   ├── test_runner.py       # Test execution utilities
│   │   └── agent_tools.py       # Claude Code agent tools
│   └── training_civ6.py         # Training entry point
├── dockerfiles/                 # Container images (NEW)
│   ├── Dockerfile.base          # Base image with PyTorch
│   ├── Dockerfile.selfplay      # Self-play worker image
│   ├── Dockerfile.trainer       # Trainer image
│   └── Dockerfile.evaluator     # Evaluator image
├── infra/                       # Infrastructure (NEW)
│   ├── __init__.py
│   └── vertex_launcher.py       # Vertex AI job launcher
├── tests/
│   ├── __init__.py
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── configs/
│   ├── civ6_config.yaml         # Original configuration file
│   ├── orchestrator.yaml        # Orchestrator master config
│   ├── orchestrator_fast.yaml   # Fast testing preset
│   └── orchestrator_production.yaml  # Production preset
├── data/
│   └── knowledge/               # RAG knowledge base documents
│       ├── civ6_strategies.md
│       ├── unit_tactics.md
│       └── build_orders.md
├── .env                         # Environment variables (DO NOT COMMIT)
├── .env.template                # Environment template
├── requirements.txt             # Dependencies
├── pyproject.toml               # Package configuration
├── Makefile                     # Build and deployment targets (NEW)
├── docker-compose.yml           # Local testing environment (NEW)
└── CLAUDE.md                    # This file
```

---

## SECTION 8: QUICK START

```bash
# Setup
cd Civ6-AlphaZero
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Configure environment
cp .env.template .env
# Edit .env with your API keys

# Run smoke tests
pytest tests/unit/test_environment.py -v
python -m alpha_zero.training_civ6 --dry_run

# Full test suite
pytest tests/ -v --cov=alpha_zero

# Train MicroCiv
python -m alpha_zero.training_civ6 --num_games=100 --num_iterations=100
```

---

## SECTION 9: COMPONENT DOCUMENTATION

### 9.1 MicroCiv Environment

A simplified 4X game for Phase 1 training:
- **Board**: 16x16 grid (configurable)
- **Players**: 2 players, turn-based
- **Units**: Warrior, Settler, Archer
- **Actions**: Move, Attack, Skip, Produce, End Turn
- **Victory**: Eliminate enemy or most cities at turn limit

State encoding (13 channels):
| Channels | Feature |
|----------|---------|
| 0-2 | Terrain (plains, hills, mountain) |
| 3-5 | Own units (warrior, settler, archer) |
| 6-8 | Enemy units |
| 9 | Own cities |
| 10 | Enemy cities |
| 11 | Visibility |
| 12 | Constant (bias term) |

### 9.2 Neural Network Architecture

```
Spatial Input (13, H, W)
    │
    ▼
Conv2d (13 → 64) + BN + ReLU
    │
    ▼
ResBlock × 6 (64 filters)
    │
    ├─────────────────────┐
    │                     │
    ▼                     ▼
Policy Head          Value Head
Conv2d (64→2)        Conv2d (64→1)
FC (combined)        FC (combined)
    │                     │
    ▼                     ▼
Policy Logits (A)    Value [-1, 1]
```

### 9.3 LLM Integration

```python
from alpha_zero.utils import create_llm_client

# Create client (reads from .env)
client = create_llm_client()

# Generate response
response = await client.generate(
    prompt="Analyze this game state...",
    system_prompt="You are a Civ6 strategy expert."
)

# Synchronous version
response = client.generate_sync("Hello!")
```

### 9.4 Test Runner

```python
from alpha_zero.utils import TestRunner, run_tests

# Quick test
result = run_tests(with_coverage=True)
print(result.summary())

# Specific tests
runner = TestRunner.from_env()
result = runner.run_unit_tests()
result = runner.run_by_keyword("mcts")
```

### 9.5 Configuration System

```python
from alpha_zero.utils import ConfigManager, load_config

# Load from file
config = load_config("configs/civ6_config.yaml")

# With environment overrides
# Set CIV6AZ_TRAINING_BATCH_SIZE=512 in environment
manager = ConfigManager("configs/civ6_config.yaml")
training_config = manager.get_training_config()
```

### 9.6 Training Orchestrator

The unified training orchestrator provides a single entry point for all training modes.

```python
from alpha_zero.orchestrator import TrainingOrchestrator, OrchestratorConfig

# Load from preset
config = OrchestratorConfig.from_preset("fast")

# Or from YAML file
config = OrchestratorConfig.from_yaml("configs/orchestrator.yaml")

# Apply environment overrides (CIV6AZ_TRAINING_BATCH_SIZE=512)
config = config.apply_env_overrides()

# Create and run orchestrator
orchestrator = TrainingOrchestrator(config)
orchestrator.run()  # Full training loop
```

CLI usage:

```bash
# Show help
python -m alpha_zero.orchestrator --help

# Run with preset
python -m alpha_zero.orchestrator --preset fast

# Run with config file
python -m alpha_zero.orchestrator --config configs/orchestrator.yaml

# Dry run (validate configuration)
python -m alpha_zero.orchestrator --preset fast --dry_run

# Override settings
python -m alpha_zero.orchestrator --preset balanced \
    --iterations 10000 \
    --games-per-iteration 50 \
    --no-wandb --no-rag
```

### 9.7 Hyperparameter Scheduling

```python
from alpha_zero.orchestrator import HyperparameterScheduler
from alpha_zero.orchestrator.config import ScheduleConfig

# Create schedule configurations
lr_schedule = ScheduleConfig(
    schedule_type="warmup_cosine",
    initial_value=0.001,
    final_value=0.00001,
    decay_steps=100000,
    warmup_steps=5000,
)

# Create scheduler
scheduler = HyperparameterScheduler(
    lr_schedule=lr_schedule,
    batch_size_schedule=ScheduleConfig(initial_value=64),
)

# Get values at any step
lr = scheduler.get_learning_rate(step=1000)
batch_size = scheduler.get_batch_size(step=1000)
all_values = scheduler.get_all_values(step=1000)
```

### 9.8 RAG Integration

```python
from alpha_zero.orchestrator import RAGIntegration
from alpha_zero.orchestrator.config import RAGConfig

# Create RAG config
rag_config = RAGConfig(
    enabled=True,
    persist_directory="data/knowledge_db",
    knowledge_dir="data/knowledge",
    store_insights=True,
)

# Initialize RAG
rag = RAGIntegration(rag_config)
rag.initialize()  # Ingests knowledge base

# Query strategies
results = rag.query_strategy("early game expansion strategy")

# Store training insights
rag.store_training_insight("Found that aggressive expansion wins more games")

rag.close()
```

---

## SECTION 10: DEBUG COMMANDS

```bash
# Check tensor shapes
python -c "
from alpha_zero.envs.microciv import MicroCivEnv
env = MicroCivEnv()
obs, _ = env.reset()
print(f'Spatial: {obs[\"spatial\"].shape}')
print(f'Vector: {obs[\"vector\"].shape}')
print(f'Action mask: {obs[\"action_mask\"].shape}')
"

# Test network forward pass
python -c "
import torch
from alpha_zero.core.civ6_network import MicroCivNetwork
net = MicroCivNetwork(board_size=16, action_space_size=356)
spatial = torch.randn(1, 13, 16, 16)
vector = torch.randn(1, 64)
policy, value = net(spatial, vector)
print(f'Policy shape: {policy.shape}')
print(f'Value: {value.item():.3f}')
"

# Check LLM configuration
python -c "
from alpha_zero.utils.llm_client import LLMConfig
config = LLMConfig.from_env()
print(f'Provider: {config.provider}')
print(f'OpenAI key set: {bool(config.openai_api_key)}')
print(f'Anthropic key set: {bool(config.anthropic_api_key)}')
"

# Run single game with rendering
python -c "
from alpha_zero.envs.microciv import MicroCivEnv
import numpy as np
env = MicroCivEnv(board_size=8, render_mode='ansi', seed=42)
env.reset()
print(env.render())
for _ in range(10):
    mask = env.get_valid_actions()
    action = np.random.choice(np.where(mask > 0)[0])
    env.step(action)
print(env.render())
"
```

---

## SECTION 11: RAG SYSTEM

### 11.1 RAG Overview
The RAG (Retrieval-Augmented Generation) system enhances agent workflows with context from a vector database.

### 11.2 RAG Architecture
```
alpha_zero/rag/
├── config/         # Configuration management (Pydantic models)
├── stores/         # Vector store backends (Memory, Chroma, FAISS)
├── processors/     # Document processing and embeddings
├── agents/         # Agent workflow integration
├── utils/          # Logging, metrics, caching
└── pipeline.py     # Unified RAG pipeline
```

### 11.3 RAG Commands

| Command | Purpose |
|---------|---------|
| `pytest tests/unit/test_rag/ -v` | Run RAG unit tests |
| `pytest tests/integration/test_rag/ -v` | Run RAG integration tests |
| `pip install -e ".[rag]"` | Install with RAG dependencies |
| `pip install -e ".[all]"` | Install all dependencies |

### 11.4 RAG Usage Example

```python
from alpha_zero.rag import RAGPipeline, RAGConfig

# Create pipeline
config = RAGConfig.from_env()
pipeline = await RAGPipeline.create(config)

# Ingest documents
await pipeline.ingest_text("Strategy guide content...", source="guide.txt")

# Query with context
result = await pipeline.query("How to win early game?", top_k=5)
print(result.context)

await pipeline.close()
```

### 11.5 RAG Environment Variables
- `RAG_VECTOR_STORE_TYPE` - Backend: memory, chroma, faiss
- `RAG_EMBEDDING_MODEL` - Embedding model name
- `RAG_EMBEDDING_API_KEY` - API key for OpenAI embeddings
- `RAG_LOG_LEVEL` - Logging level

---

## SECTION 12: VERTEX AI CLOUD DEPLOYMENT

### 12.1 Overview
The system supports deployment to Google Vertex AI for distributed training at scale.

### 12.2 Storage Backend
```python
from alpha_zero.storage import create_storage_backend, StorageConfig

# Local storage (default)
storage = create_storage_backend(StorageConfig.for_local("./data"))

# GCS storage (for cloud)
storage = create_storage_backend(StorageConfig.for_gcs("my-bucket", "civ6-alphazero"))

# Save/load models
storage.save_model(state_dict, iteration=100)
state_dict, metadata = storage.load_champion()

# Save trajectories
storage.save_trajectory(trajectory, iteration=1, worker_id=0, game_id="game_001")
```

### 12.3 Machine Profiles
| Profile | GPU | VRAM | Spot $/hr | On-Demand $/hr | Use For |
|---------|-----|------|-----------|----------------|---------|
| `selfplay_t4_spot` | 1× T4 | 16 GB | $0.11 | $0.35 | Self-play (budget) |
| `selfplay_l4_spot` | 1× L4 | 24 GB | $0.24 | $0.80 | Self-play |
| `train_l4_single` | 1× L4 | 24 GB | — | $0.80 | Training (budget) |
| `train_l4_ondemand` | 2× L4 | 48 GB | — | $2.68 | Training |
| `train_a100_burst` | 1× A100 | 40 GB | — | $3.67 | Burst training |
| `eval_l4_spot` | 1× L4 | 24 GB | $0.24 | $0.80 | Evaluation |

### 12.4 Cloud Workers
```bash
# Self-play worker
python -m alpha_zero.cloud.selfplay_worker \
    --num-games=100 \
    --iteration=5 \
    --worker-id=0 \
    --num-simulations=800

# Trainer
python -m alpha_zero.cloud.trainer \
    --training-steps=1000 \
    --iteration=5 \
    --batch-size=256

# Evaluator
python -m alpha_zero.cloud.evaluator \
    --num-games=400 \
    --iteration=5 \
    --promotion-threshold=0.55
```

### 12.5 Environment Variables
| Variable | Purpose |
|----------|---------|
| `CIV6_STORAGE_BACKEND` | Backend type: "local" or "gcs" |
| `CIV6_GCS_BUCKET` | GCS bucket name |
| `CIV6_GCS_PREFIX` | Path prefix in bucket |
| `CIV6_ITERATION` | Current training iteration |
| `CIV6_WORKER_ID` | Self-play worker identifier |
| `CIV6_DEVICE` | PyTorch device (cuda/cpu) |
| `AIP_CHECKPOINT_DIR` | Vertex AI checkpoint directory (auto) |
| `AIP_MODEL_DIR` | Vertex AI model directory (auto) |

### 12.6 Docker Build
```bash
# Build all images
make build-all

# Push to Artifact Registry
make setup-artifact-registry
make push

# Local testing
docker-compose up -d
docker-compose logs -f trainer
```

### 12.7 Vertex AI Jobs
```bash
# Submit self-play fleet
python -m infra.vertex_launcher \
    --mode=selfplay \
    --iteration=5 \
    --num-workers=3 \
    --games-per-worker=100

# Submit training job
python -m infra.vertex_launcher \
    --mode=train \
    --iteration=5 \
    --training-steps=1000

# Submit evaluation job
python -m infra.vertex_launcher \
    --mode=eval \
    --iteration=5 \
    --num-games=400

# Run full iteration
python -m infra.vertex_launcher \
    --mode=iteration \
    --iteration=5 \
    --wait
```

### 12.8 Estimated Costs (Budget Tier)
| Component | Hours/mo | Rate | Monthly |
|-----------|----------|------|---------|
| Self-play (3× T4 Spot) | 270 | $0.11/hr | ~$30 |
| Training (L4 on-demand) | 90 | $0.80/hr | ~$72 |
| Evaluation (L4 Spot) | 45 | $0.24/hr | ~$11 |
| GCS Storage (500 GB) | — | $0.02/GB | ~$10 |
| **Total** | | | **~$145/mo** |

---

## SECTION 13: NEXT STEPS

### Phase 3+ (Planned)
1. **Multi-GPU Training**: Extend distributed training to multiple GPUs
2. **Population-Based Training**: Adaptive hyperparameter tuning
3. **Full Civ 6 Integration**: Direct game interface (mod or API)
4. **Tournament System**: ELO-based opponent selection
5. **Human vs AI**: Interface for human players to challenge trained models
6. **RAG Enhancement**: LLM-powered strategy recommendations using RAG context

---

## SECTION 14: KNOWLEDGE BASE PROTOCOL

This project uses a structured knowledge base in `kb/` for session continuity
across a three-role workflow pipeline: Planning → DEV-SQE → Pre-PR.

### 14.1 Directory Structure
- `kb/summaries/` — Rolling summaries per workflow role
- `kb/sessions/` — Session state with resume pointers (one mutable file per role)
- `kb/ledger/` — Append-only JSONL event log (shared across all roles)
- `kb/handoffs/` — Immutable cross-role handoff documents

### 14.2 Workflow Roles

| Role | Purpose | Upstream | Downstream |
|------|---------|----------|------------|
| **Planning** | Task breakdown, arch decisions, acceptance criteria | — | DEV-SQE |
| **DEV-SQE** | Implementation, testing, quality engineering | Planning | Pre-PR |
| **Pre-PR** | Validation, PR preparation, documentation | DEV-SQE | (PR ready) |

### 14.3 Handoff Flow
- Planning → DEV-SQE: `PLANNING-HANDOFF` event + handoff doc
- DEV-SQE → Pre-PR: `DEV-SQE-HANDOFF` event + handoff doc
- DEV-SQE → Planning: `DESIGN-GAP` event (feedback loop)
- Pre-PR → DEV-SQE: `QA-GAP` event (feedback loop)
- Pre-PR → Planning: `SCOPE-CHANGE` event (feedback loop)

### 14.4 Handoff Skills
- Planning: `.claude/skills/planning-handoff/SKILL.md`
- DEV-SQE: `.claude/skills/dev-sqe-handoff/SKILL.md`
- Pre-PR: `.claude/skills/pre-pr-handoff/SKILL.md`

### 14.5 Rules
1. NEVER overwrite or truncate `kb/ledger/ledger.jsonl` — only append
2. Handoff files in `kb/handoffs/` are immutable once created
3. Session state files (`kb/sessions/*.state.json`) are the ONLY mutable state per role
4. Summary files are append-at-top with a cap of 20 entries
5. All JSONL entries must be valid JSON — validate after writing
6. In Agent Teams, designate ONE teammate (the team lead) as the sole writer
   to `kb/ledger/ledger.jsonl` to avoid race conditions
7. When spawning Agent Teams, each teammate should read relevant KB summaries
   before starting work
8. Every session that does meaningful work MUST run its role's handoff skill
   before stopping
9. Cross-role handoffs create BOTH a ledger event AND a handoff doc

---

*Last updated: 2026-02-08*
*Phase: 5 - Google Vertex AI Cloud Deployment*
*Current Focus: Budget-tier cloud deployment with GCS storage and Spot VMs*
