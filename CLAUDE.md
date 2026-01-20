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
- [ ] LLM integration test: `python -c "from alpha_zero.utils import create_llm_client"` imports
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

#### Test Commands

| Command | Purpose |
|---------|---------|
| `pytest tests/unit/test_environment.py -v` | Test MicroCiv environment |
| `pytest tests/unit/test_network.py -v` | Test neural networks |
| `pytest tests/unit/test_mcts.py -v` | Test MCTS algorithm |
| `pytest tests/unit/test_config.py -v` | Test configuration system |
| `pytest tests/unit/test_logging.py -v` | Test logging infrastructure |
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

### 5.3 Known Issues

| Issue | Workaround |
|-------|------------|
| No GPU in dev environment | Training works on CPU (slower) |
| SimpleMCTS random rollout is weak | Use neural network MCTS after initial training |
| Windows multiprocessing spawn | Use `if __name__ == "__main__"` guard |
| pytest-xdist not always installed | Tests run without parallel flag if missing |

### 5.4 Information to Preserve

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
│   ├── interfaces/              # Game interfaces
│   │   ├── __init__.py
│   │   └── unciv_interface.py   # Unciv JSON interface
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── logging.py           # Logging infrastructure
│   │   ├── config.py            # Configuration management
│   │   ├── llm_client.py        # LLM API integration
│   │   ├── test_runner.py       # Test execution utilities
│   │   └── agent_tools.py       # Claude Code agent tools
│   └── training_civ6.py         # Training entry point
├── tests/
│   ├── __init__.py
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── configs/
│   └── civ6_config.yaml         # Configuration file
├── .env                         # Environment variables (DO NOT COMMIT)
├── .env.template                # Environment template
├── requirements.txt             # Dependencies
├── pyproject.toml               # Package configuration
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

## SECTION 12: NEXT STEPS

### Phase 3+ (Planned)
1. **Multi-GPU Training**: Extend distributed training to multiple GPUs
2. **Population-Based Training**: Adaptive hyperparameter tuning
3. **Full Civ 6 Integration**: Direct game interface (mod or API)
4. **Tournament System**: ELO-based opponent selection
5. **Human vs AI**: Interface for human players to challenge trained models
6. **RAG Enhancement**: LLM-powered strategy recommendations using RAG context

---

*Last updated: 2026-01-19*
*Phase: 2 - Distributed Training & Game Integration (Complete)*
*Current Focus: RAG DB Integration & Agent Workflow Enhancement*
