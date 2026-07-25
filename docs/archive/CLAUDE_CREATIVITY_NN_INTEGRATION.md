# Civ6-AlphaZero × Creativity NN Integration — Claude Code Implementation Plan

> **Scope**: Phase 1 ONLY — Crystalline Hex Spatial Encoder swap  
> **Repository**: `github.com/Mango-Metrics-NLM/Civ6-AlphaZero`  
> **Source Extraction**: `github.com/ianshank/Creativity-Neural-Network` → `src/creativity_nn/`  
> **Estimated Effort**: ~2 weeks  
> **Risk Level**: Low (drop-in replacement, rest of pipeline unchanged)

---

## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

```
I am replacing the Conv2d-based spatial encoder in the Civ6-AlphaZero neural network
with a crystalline hex convolution encoder extracted and adapted from the Creativity
Neural Network repository. The goal is to preserve hex grid topology in the spatial
encoding path, eliminating information loss from offset coordinate rectangular masking,
while maintaining full backwards compatibility with the existing residual tower,
policy head, value head, MCTS search, and training pipeline.
```

### 1.2 Success Criteria (Mechanically Verifiable)

```
GATE 1 — Layer Extraction (Days 1-2):
- [ ] CrystallineLatticeLayer extracted from creativity_nn/ and adapted to game state channels
- [ ] HexConvolution module compiles and produces correct output shapes
- [ ] All extracted modules pass isolated unit tests (shape, gradient flow, symmetry)
- [ ] Zero hardcoded values — all dimensions, kernel sizes, symmetry types from config

GATE 2 — Encoder Integration (Days 3-5):
- [ ] HexSpatialEncoder replaces Conv2d(84, 256, 3) in networks/encoders.py
- [ ] Encoder satisfies StateEncoder protocol (same input/output contract)
- [ ] Existing residual tower, policy head, value head receive identical tensor shapes
- [ ] Forward pass produces valid outputs for all mini-game environments
- [ ] Backward pass produces finite gradients through hex conv layers

GATE 3 — Configuration & Registry (Days 6-7):
- [ ] Encoder selectable via config: encoder.spatial.type: "hex_crystalline" | "conv2d"
- [ ] All hex conv hyperparameters (symmetry_type, lattice_order, kernel_topology) in YAML
- [ ] Config validation rejects invalid combinations (Pydantic)
- [ ] Old checkpoints load with fallback to conv2d encoder (backwards compatibility)
- [ ] New checkpoints include encoder type metadata for migration

GATE 4 — A/B Comparison Infrastructure (Days 8-10):
- [ ] A/B training harness runs both encoders on Tier 0 mini-game simultaneously
- [ ] Shared replay buffer, separate networks, identical MCTS config
- [ ] Metrics logged: spatial encoding loss, policy entropy, value prediction MSE, Elo
- [ ] Convergence comparison at 10K, 25K, 50K self-play games
- [ ] Automated report generation comparing convergence curves

GATE 5 — Full Test Suite (Days 11-14):
- [ ] Unit tests: 100% coverage on all new modules
- [ ] Integration tests: full forward/backward through augmented network
- [ ] Property-based tests: hex symmetry invariance (Hypothesis)
- [ ] Regression tests: conv2d encoder produces identical results to pre-integration
- [ ] Performance benchmarks: inference latency delta measured and acceptable (< 40%)
- [ ] All existing tests pass unchanged (zero regressions)
```

### 1.3 Problem Description

```
PARAGRAPH 1 — CONTEXT:
The Civ6-AlphaZero spatial encoder uses standard Conv2d(84, 256, kernel_size=3)
to process an 84-channel game state tensor of shape (B, 84, H, W). This tensor
encodes terrain, features, resources, units, cities, borders, fog of war, movement
ranges, and tile yields on a hex grid that has been converted to rectangular
offset coordinates. The conversion introduces two artifacts: (1) rectangular
convolution kernels connect tiles that are not hex-adjacent and fail to connect
tiles that ARE hex-adjacent, creating a topology mismatch; (2) some positions in
the rectangular grid correspond to no hex tile and must be masked, wasting
representational capacity. The Creativity Neural Network contains a
CrystallineLatticeLayer with hexagonal (6-fold), cubic (8-fold), and tetrahedral
(4-fold) symmetry modes that can be adapted as a hex-native convolution layer.

PARAGRAPH 2 — DATA FLOWS:
Data flows from CivRealm/MicroCiv environment → Civ6StateEncoder (produces
84-channel spatial tensor + 512-dim vector) → SpatialEncoder (THIS IS THE 
REPLACEMENT TARGET) → concat with VectorEncoder output → ResidualTower →
PolicyHead + ValueHead + AuxHeads → MCTS → ActionDecoder → Environment.
The SpatialEncoder's contract is: input (B, 84, H, W) → output (B, 256, H, W).
This contract MUST NOT change. The replacement only modifies the internal
convolution operations, not the input/output interface.

PARAGRAPH 3 — FAILURE MODES AND INVARIANTS:
Failure modes: hex conv kernel initialization collapse (mitigate with He init
adapted for 6-neighbor topology), training instability from novel gradient
paths (mitigate with gradient clipping warmup), shape mismatch at encoder
boundary (caught by protocol-based type checking), silent degradation if hex
topology is wrong (caught by symmetry invariance property tests). Invariants:
(1) output shape is always (B, 256, H, W) regardless of encoder type,
(2) encoder is stateless between calls (no hidden state),
(3) all parameters flow from config (zero hardcoded values),
(4) existing conv2d path remains functional and selectable,
(5) checkpoint migration is lossless for conv2d models.
```

---

## SECTION 2: FEASIBLE REGION (Constraints)

### 2.1 Hard Constraints (Violations = Build Failure)

```yaml
LANGUAGE_AND_RUNTIME:
  python: ">=3.11"                    # Required for TaskGroup, ExceptionGroup
  pytorch: ">=2.1"                    # Required for torch.compile
  cuda: "12.x with CPU fallback"

REQUIRED_DEPENDENCIES:
  - pydantic >= 2.0                   # Config validation, serialization
  - numpy >= 1.24                     # Tensor operations
  - torch >= 2.1                      # Neural networks, torch.compile
  - structlog >= 23.0                 # Structured logging
  - hydra-core >= 1.3                 # Hierarchical configuration
  - pytest >= 7.4                     # Testing framework
  - hypothesis >= 6.80                # Property-based testing

ZERO_HARDCODED_VALUES:
  - No magic numbers in any module
  - All dimensions, channel counts, kernel sizes from config
  - All paths from config or environment variables
  - All thresholds, learning rates, temperatures from config
  - Use named constants via enum or config dataclass, never bare literals

BACKWARDS_COMPATIBILITY:
  - Existing conv2d encoder path must remain functional
  - Existing checkpoints must load without error (with conv2d encoder)
  - All existing tests must pass without modification
  - StateEncoder protocol contract unchanged
  - No changes to residual tower, policy head, value head, MCTS interfaces

TESTING:
  - Every public function has at least one unit test
  - Every module has at least one integration test
  - All new layers have property-based symmetry tests
  - Test coverage >= 95% on new code (measured by pytest-cov)
  - No test uses sleep(), time-based assertions, or network calls
```

### 2.2 Soft Constraints (Preferences)

```yaml
CODE_STYLE:
  - Type hints on all function signatures
  - Google-style docstrings on all public functions
  - Dataclasses over raw dicts for structured data
  - Protocol classes for interfaces (not ABC unless necessary)
  - Single responsibility per module (< 300 lines per file)

LOGGING:
  - structlog for all runtime logging
  - Correlation IDs propagated through training pipeline
  - DEBUG: tensor shapes, gradient norms, config resolution
  - INFO: training progress, encoder selection, checkpoint events
  - WARNING: fallback to conv2d, shape mismatches corrected
  - ERROR: initialization failures, NaN detection, shape violations

PERFORMANCE:
  - Inference latency increase < 40% over baseline conv2d
  - Memory increase < 25% over baseline
  - torch.compile compatible (no dynamic control flow in forward pass)
```

### 2.3 Anti-Constraints (Explicit Permissions)

```yaml
YOU_ARE_PERMITTED_TO:
  - Add any PyPI dependencies (document in pyproject.toml)
  - Create new modules, packages, and directories within src/
  - Create new test files and fixtures within tests/
  - Create new config YAML files within configs/
  - Choose hex convolution kernel implementation strategy
  - Choose weight initialization scheme for hex kernels
  - Decide internal architecture of HexSpatialEncoder (num layers, activation)
  - Add utility functions to existing utils.py modules
  - Refactor existing encoder code IF doing so improves modularity
  - Add new Hydra config groups for encoder selection
  - Add new metrics to training logger
  - Create visualization utilities for hex conv kernels
  - Add pre-commit hooks or CI config changes
```

---

## SECTION 3: PERMISSION ARCHITECTURE

### 3.1 Scope (What You Can Touch)

```yaml
IN_SCOPE:
  # New files to create
  - src/core/networks/hex_conv.py              # HexConvolution layer
  - src/core/networks/crystalline_layers.py    # Extracted crystalline lattice layers
  - src/core/networks/hex_encoder.py           # HexSpatialEncoder (full encoder)
  - src/core/networks/encoder_registry.py      # Registry for encoder selection
  - src/core/networks/encoder_factory.py       # Factory from config → encoder
  - tests/unit/test_hex_conv.py                # Hex conv unit tests
  - tests/unit/test_crystalline_layers.py      # Crystalline layer unit tests
  - tests/unit/test_hex_encoder.py             # Hex encoder unit tests
  - tests/unit/test_encoder_registry.py        # Registry unit tests
  - tests/integration/test_hex_encoder_pipeline.py  # Full pipeline integration
  - tests/property/test_hex_symmetry.py        # Hypothesis symmetry tests
  - tests/benchmarks/test_encoder_performance.py    # Latency/memory benchmarks
  - configs/encoders/hex_crystalline.yaml      # Hex encoder config
  - configs/encoders/conv2d_baseline.yaml      # Baseline encoder config (extract current)
  - configs/experiments/ab_encoder_comparison.yaml  # A/B experiment config
  - scripts/compare_encoders.py                # A/B comparison runner

  # Existing files to modify (minimal, additive only)
  - src/core/networks/encoders.py              # Add import, register conv2d in registry
  - src/core/networks/model.py                 # Use encoder factory instead of direct Conv2d
  - src/core/config.py                         # Add EncoderConfig to main config
  - src/core/protocols.py                      # Verify StateEncoder protocol is sufficient
  - configs/networks/default.yaml              # Add encoder section with type field

OUT_OF_SCOPE:
  - src/core/mcts/                             # No MCTS changes in Phase 1
  - src/core/networks/heads.py                 # No policy/value head changes
  - src/core/networks/backbones.py             # No residual tower changes
  - src/core/training/                         # No training loop changes
  - src/environments/                          # No environment changes
  - deploy/, k8s/, terraform/                  # No infrastructure changes
  - .git/, vendor/                             # Never touch
```

### 3.2 Autonomy Level

```yaml
AUTONOMOUS:
  - File creation/deletion within IN_SCOPE
  - Dependency installation via pip/uv (document in pyproject.toml)
  - Running tests (pytest, hypothesis, benchmarks)
  - Running linters (ruff, mypy)
  - Creating git commits with descriptive messages
  - Choosing internal implementation details of hex conv layers
  - Adding structured logging statements
  - Creating test fixtures and factory functions
  - Adding type aliases and constants

CONFIRM_FIRST:
  - Modifying any existing file in src/core/networks/
  - Modifying src/core/config.py or src/core/protocols.py
  - Changing any config schema (new required fields)
  - Adding dependencies not in REQUIRED_DEPENDENCIES list
  - Changing test infrastructure (conftest.py, pytest.ini)

PROHIBITED:
  - Modifying MCTS, training loop, or environment code
  - Changing the StateEncoder protocol signature
  - Removing the conv2d encoder path
  - Modifying existing test expectations
  - Hardcoding dimensions, paths, or thresholds
  - Using `# type: ignore` without explanation
  - Suppressing warnings without logging them
```

---

## SECTION 4: AGENT ARCHITECTURE

### 4.1 Primary Agent: Integration Orchestrator

```yaml
name: integration_orchestrator
role: >
  Coordinates the full Phase 1 integration. Sequences sub-agent tasks,
  validates gates, manages dependencies between work streams.
  
tools:
  - bash (git, pytest, ruff, mypy, pip)
  - file_read / file_write / file_edit
  - grep / find (codebase search)
  - sequential_thinking (planning, dependency resolution)

responsibilities:
  - Parse existing codebase to understand current encoder interface
  - Create execution plan with dependency ordering
  - Dispatch to sub-agents in correct sequence
  - Validate each gate before proceeding to next
  - Produce final integration report

sequential_thinking_protocol:
  step_1: "Read src/core/protocols.py → extract StateEncoder protocol signature"
  step_2: "Read src/core/networks/encoders.py → map current Conv2d encoder implementation"
  step_3: "Read src/core/networks/model.py → understand how encoder is instantiated"
  step_4: "Read src/core/config.py → understand config schema for networks"
  step_5: "Read configs/networks/default.yaml → understand current encoder config"
  step_6: "Dispatch to extraction_agent for creativity_nn layer extraction"
  step_7: "Dispatch to encoder_agent for hex encoder construction"
  step_8: "Dispatch to integration_agent for registry + factory + config"
  step_9: "Dispatch to test_agent for full test suite"
  step_10: "Dispatch to benchmark_agent for performance validation"
  step_11: "Run all gates, produce report"
```

### 4.2 Sub-Agent: Layer Extraction Agent

```yaml
name: extraction_agent
role: >
  Extracts crystalline lattice layer components from Creativity-Neural-Network
  repo, adapts them for game state tensor input, and validates in isolation.

tools:
  - bash (git clone, pip install)
  - file_read / file_write
  - pytest (isolated layer tests)

input_context: |
  The Creativity NN has these key source modules in src/creativity_nn/:
  - Crystalline lattice layers: hexagonal (6-fold), cubic (8-fold), tetrahedral (4-fold)
  - Each lattice type applies rotational symmetry transformations to conv features
  - Input: multi-channel tensors (originally 3ch image, we need 84ch game state)
  - The HexConv must connect exactly the 6 hex-adjacent neighbors, not the 4/8 rect neighbors

sequential_thinking_protocol:
  step_1: >
    Clone Creativity-Neural-Network repo to temp directory.
    Read src/creativity_nn/ module structure.
    Identify: CrystallineLatticeLayer, HexConv components, config dataclasses.
  step_2: >
    Extract ONLY the hexagonal lattice layer code.
    Identify all dependencies (imports, utility functions, base classes).
    Map parameter flow: what is configurable vs hardcoded.
  step_3: >
    Create src/core/networks/crystalline_layers.py:
    - Port CrystallineLatticeLayer with hexagonal symmetry mode
    - Replace all hardcoded values with config parameters
    - Adapt input channels: remove image-specific assumptions (3ch, 224x224)
    - Make input_channels, output_channels, symmetry_type fully configurable
    - Add type hints, docstrings, structured logging
  step_4: >
    Create src/core/networks/hex_conv.py:
    - Implement HexConvolution as a nn.Module
    - Hex kernel topology: 6-neighbor connectivity pattern
    - Weight sharing respecting 6-fold rotational symmetry
    - Support configurable kernel size (1-ring, 2-ring)
    - He initialization adapted for 6-neighbor fan-in
    - Include optional batch normalization and activation
  step_5: >
    Write tests/unit/test_crystalline_layers.py:
    - Test output shapes for various input configurations
    - Test gradient flow (no dead gradients)
    - Test symmetry: rotating input by 60° rotates output equivalently
    - Test config validation (invalid symmetry types rejected)
  step_6: >
    Write tests/unit/test_hex_conv.py:
    - Test kernel connectivity (only 6 neighbors connected)
    - Test weight sharing (rotated weights are identical)
    - Test output shapes for variable input sizes
    - Test initialization (no zero or NaN weights)
    - Test backwards pass (finite gradients)

output_artifacts:
  - src/core/networks/crystalline_layers.py
  - src/core/networks/hex_conv.py
  - tests/unit/test_crystalline_layers.py
  - tests/unit/test_hex_conv.py

validation_command: |
  cd $PROJECT_ROOT
  pytest tests/unit/test_crystalline_layers.py tests/unit/test_hex_conv.py -v --tb=short
  # MUST: all pass, zero warnings about shape mismatches
```

### 4.3 Sub-Agent: Encoder Construction Agent

```yaml
name: encoder_agent
role: >
  Constructs the full HexSpatialEncoder module that satisfies the StateEncoder
  protocol, using hex conv layers from extraction_agent output.

tools:
  - file_read / file_write
  - pytest

input_dependencies:
  - src/core/networks/hex_conv.py (from extraction_agent)
  - src/core/networks/crystalline_layers.py (from extraction_agent)
  - src/core/protocols.py (existing — StateEncoder protocol)
  - src/core/networks/encoders.py (existing — current Conv2d encoder reference)

sequential_thinking_protocol:
  step_1: >
    Read src/core/protocols.py to extract the EXACT StateEncoder protocol:
    - Input signature (tensor shape, dtype)
    - Output signature (tensor shape, dtype)
    - Any additional protocol methods (e.g., get_output_channels)
    - Any properties the protocol expects
  step_2: >
    Read src/core/networks/encoders.py to understand the current Conv2d encoder:
    - How it's constructed (what config values it reads)
    - Its forward() implementation
    - How it handles variable map sizes
    - Any preprocessing (padding, masking) applied to input
  step_3: >
    Design HexSpatialEncoder architecture:
    - Layer 1: HexConv(in_channels=config.spatial_channels, out_channels=config.encoder_dim, ...)
    - Layer 2: CrystallineLatticeLayer(symmetry="hexagonal", ...) 
    - Layer 3: BatchNorm + activation
    - Optional: additional hex conv layers (configurable depth)
    - Output: (B, encoder_dim, H, W) matching existing contract
  step_4: >
    Create src/core/networks/hex_encoder.py:
    - HexSpatialEncoderConfig (Pydantic model)
    - HexSpatialEncoder(nn.Module) implementing StateEncoder protocol
    - All parameters from config, zero hardcoded values
    - Structured logging: log encoder selection, param count, input/output shapes
    - torch.compile compatible (no Python control flow in forward)
  step_5: >
    Write tests/unit/test_hex_encoder.py:
    - Test protocol compliance (implements all StateEncoder methods)
    - Test output shape matches Conv2d encoder for same input
    - Test with various map sizes (4x4, 8x8, 44x44)
    - Test with various channel counts (controlled by config)
    - Test parameter count is within expected range
    - Test torch.compile compatibility
    - Test deterministic output (same input → same output in eval mode)

output_artifacts:
  - src/core/networks/hex_encoder.py
  - tests/unit/test_hex_encoder.py

validation_command: |
  pytest tests/unit/test_hex_encoder.py -v --tb=short
  # MUST: all pass, encoder satisfies protocol
```

### 4.4 Sub-Agent: Integration Agent

```yaml
name: integration_agent
role: >
  Creates the encoder registry, factory, and config integration so that the
  encoder is selectable via YAML config with backwards-compatible defaults.

tools:
  - file_read / file_write / file_edit
  - pytest
  - mypy

sequential_thinking_protocol:
  step_1: >
    Read src/core/config.py to understand the existing config schema.
    Identify where network config is defined.
    Determine how to add an EncoderConfig section.
  step_2: >
    Create src/core/networks/encoder_registry.py:
    - EncoderRegistry class with register() and get() methods
    - Type-safe: register(name: str, encoder_cls: type[nn.Module], config_cls: type[BaseModel])
    - Lazy import: encoder classes loaded only when selected
    - Thread-safe singleton pattern
    - Log all registrations at DEBUG level
  step_3: >
    Create src/core/networks/encoder_factory.py:
    - create_encoder(config: EncoderConfig) -> nn.Module
    - Reads config.encoder.spatial.type to select encoder
    - Validates config against registered encoder's config schema
    - Returns instantiated encoder
    - Fallback: if type not recognized, log WARNING, use conv2d
  step_4: >
    Create configs/encoders/hex_crystalline.yaml:
      type: "hex_crystalline"
      spatial_channels: ${network.spatial_channels}  # reference parent config
      encoder_dim: ${network.encoder_dim}
      symmetry_type: "hexagonal"
      lattice_order: 6
      kernel_rings: 1
      num_layers: 2
      activation: "relu"
      batch_norm: true
      dropout: 0.0
  step_5: >
    Create configs/encoders/conv2d_baseline.yaml:
      type: "conv2d"
      spatial_channels: ${network.spatial_channels}
      encoder_dim: ${network.encoder_dim}
      kernel_size: 3
      padding: 1
      activation: "relu"
      batch_norm: true
  step_6: >
    Modify src/core/config.py (ADDITIVE ONLY):
    - Add EncoderConfig Pydantic model with type field and **kwargs
    - Add encoder field to NetworkConfig with default type="conv2d"
    - Ensure old configs without encoder field still work (default fallback)
  step_7: >
    Modify src/core/networks/model.py (MINIMAL):
    - Replace direct Conv2d instantiation with encoder_factory.create_encoder()
    - Pass encoder config section from main config
    - Log which encoder was selected at INFO level
  step_8: >
    Modify src/core/networks/encoders.py (ADDITIVE ONLY):
    - Register existing Conv2d encoder in registry at module load time
    - Add Conv2dEncoderConfig if not already a separate config class
  step_9: >
    Create configs/experiments/ab_encoder_comparison.yaml:
    - Hydra multirun config for A/B comparison
    - encoder.spatial.type: [conv2d, hex_crystalline]
    - Shared: replay buffer, MCTS config, training hyperparameters
    - Separate: network weights, optimizer state, metrics namespace
  step_10: >
    Write tests/unit/test_encoder_registry.py:
    - Test registration and retrieval
    - Test duplicate registration handling
    - Test unknown encoder type returns error or fallback
    - Test factory creates correct encoder from config
    - Test backwards compatibility: config without encoder section uses conv2d
  step_11: >
    Write tests/integration/test_hex_encoder_pipeline.py:
    - Test full forward pass: random game state → encoder → backbone → heads
    - Test full backward pass: loss.backward() produces finite gradients
    - Test checkpoint save/load preserves encoder type and weights
    - Test config-driven encoder swap: load conv2d checkpoint, switch to hex, verify shapes
    - Test both encoder types produce same output shapes for same input

output_artifacts:
  - src/core/networks/encoder_registry.py
  - src/core/networks/encoder_factory.py
  - configs/encoders/hex_crystalline.yaml
  - configs/encoders/conv2d_baseline.yaml
  - configs/experiments/ab_encoder_comparison.yaml
  - tests/unit/test_encoder_registry.py
  - tests/integration/test_hex_encoder_pipeline.py
  # Modified files:
  - src/core/config.py (additive)
  - src/core/networks/model.py (minimal)
  - src/core/networks/encoders.py (additive)
  - configs/networks/default.yaml (additive)

validation_command: |
  pytest tests/unit/test_encoder_registry.py -v --tb=short
  pytest tests/integration/test_hex_encoder_pipeline.py -v --tb=short
  mypy src/core/networks/encoder_registry.py src/core/networks/encoder_factory.py
  # MUST: all pass, mypy clean
```

### 4.5 Sub-Agent: Test Suite Agent

```yaml
name: test_agent
role: >
  Creates comprehensive test suite including property-based tests for
  hex symmetry invariance and regression tests for existing encoder.

tools:
  - file_read / file_write
  - pytest
  - hypothesis

sequential_thinking_protocol:
  step_1: >
    Audit existing test structure:
    - tests/unit/ — what naming conventions, fixtures, conftest patterns
    - tests/integration/ — how pipeline tests are structured
    - tests/conftest.py — shared fixtures, marks, parametrize patterns
    - Adopt IDENTICAL conventions for new tests
  step_2: >
    Create tests/conftest.py additions (or new conftest in tests/unit/):
    - Fixture: sample_game_state_tensor(batch_size, channels, height, width)
    - Fixture: hex_encoder_config() → HexSpatialEncoderConfig with defaults
    - Fixture: conv2d_encoder_config() → Conv2dEncoderConfig with defaults  
    - Fixture: sample_network_config(encoder_type) → full NetworkConfig
    - Mark: @pytest.mark.slow for benchmark tests
    - Mark: @pytest.mark.property for Hypothesis tests
    - Parametrize: encoder_type = ["conv2d", "hex_crystalline"]
  step_3: >
    Create tests/property/test_hex_symmetry.py using Hypothesis:
    
    Property 1 — Rotational Equivariance:
      Given: input tensor T on hex grid
      When: rotate T by 60° (hex rotation), apply hex encoder
      And: apply hex encoder to T, rotate output by 60°
      Then: results are equal within floating point tolerance
      Strategy: st.floats for tensor values, st.integers for grid sizes
    
    Property 2 — Translation Equivariance:
      Given: input tensor T, shift by (dx, dy) in hex coordinates
      When: apply hex encoder to shifted input
      Then: output is equivalently shifted
    
    Property 3 — Output Shape Invariance:
      Given: any valid (B, C, H, W) input where C = config.spatial_channels
      When: apply hex encoder
      Then: output shape is (B, config.encoder_dim, H, W)
      Strategy: B ∈ [1,8], H ∈ [4,64], W ∈ [4,64]
    
    Property 4 — Gradient Finiteness:
      Given: any valid input tensor with requires_grad=True
      When: forward pass → sum → backward
      Then: all parameter gradients are finite (no NaN, no Inf)
  step_4: >
    Create tests/benchmarks/test_encoder_performance.py:
    - Benchmark: inference latency (forward pass, 100 iterations, GPU and CPU)
    - Benchmark: memory footprint (peak GPU memory during forward + backward)
    - Benchmark: parameter count comparison
    - Benchmark: throughput (batches per second)
    - Use pytest-benchmark or manual timing with torch.cuda.synchronize()
    - Assert: hex encoder latency < 1.4 × conv2d latency (< 40% overhead)
    - Assert: hex encoder memory < 1.25 × conv2d memory (< 25% overhead)
  step_5: >
    Create regression test to lock existing conv2d behavior:
    - Save reference output for fixed random seed + conv2d encoder
    - Test: conv2d encoder with same seed produces identical output post-integration
    - This catches any accidental changes to the conv2d path

output_artifacts:
  - tests/conftest.py (modified or new conftest additions)
  - tests/property/test_hex_symmetry.py
  - tests/benchmarks/test_encoder_performance.py
  - tests/regression/test_conv2d_unchanged.py

validation_command: |
  # Full test suite
  pytest tests/ -v --tb=short -x
  
  # Property tests with more examples
  pytest tests/property/ -v --hypothesis-seed=0 --hypothesis-settings=ci
  
  # Benchmarks (informational, not blocking)
  pytest tests/benchmarks/ -v --benchmark-only
  
  # Coverage
  pytest tests/ --cov=src/core/networks --cov-report=term-missing --cov-fail-under=95
  
  # MUST: all pass, coverage >= 95% on new code
```

### 4.6 Sub-Agent: Benchmark Agent

```yaml
name: benchmark_agent
role: >
  Creates the A/B comparison runner and validates performance is acceptable.

tools:
  - bash (python, pytest, tensorboard)
  - file_read / file_write

sequential_thinking_protocol:
  step_1: >
    Create scripts/compare_encoders.py:
    - Parse command line args (num_games, tier, device, output_dir)
    - Load config for both encoder types
    - Instantiate both networks, same random init for shared layers
    - Run Tier 0 mini-game self-play with both encoders
    - Log per-game: encoding loss, policy entropy, value MSE, game outcome
    - Produce comparison CSV and matplotlib plots
  step_2: >
    Create scripts/encoder_ablation.py:
    - Ablation study runner for hex encoder hyperparameters
    - Sweep: symmetry_type, num_layers, kernel_rings, lattice_order
    - Use Hydra multirun for parameter sweep
    - Collect: convergence speed, final Elo, parameter count, latency
  step_3: >
    Add structured logging to hex encoder forward pass:
    - Log tensor shapes at each layer (DEBUG level)
    - Log gradient norms during backward (DEBUG level)
    - Log kernel weight statistics (mean, std, min, max) every N steps
    - Log symmetry violation metric (should be ~0 for correctly implemented hex conv)
  step_4: >
    Validate performance:
    - Run encoder on Tier 0 mini-game (4x4 grid) — expect < 5ms per forward pass
    - Run encoder on Tier 1 (8x8 grid) — expect < 15ms per forward pass
    - Run encoder on full map (44x44) — expect < 70ms per forward pass
    - Compare all against conv2d baseline

output_artifacts:
  - scripts/compare_encoders.py
  - scripts/encoder_ablation.py

validation_command: |
  python scripts/compare_encoders.py --num-games 100 --tier 0 --device cpu
  # MUST: both encoders complete without error, hex encoder latency < 1.4× conv2d
```

---

## SECTION 5: EXECUTION SEQUENCE

### 5.1 Dependency Graph

```
extraction_agent ──────────────────────────┐
  ├── crystalline_layers.py                │
  ├── hex_conv.py                          │
  ├── test_crystalline_layers.py           │
  └── test_hex_conv.py                     │
                                           ▼
                                    encoder_agent
                                      ├── hex_encoder.py
                                      └── test_hex_encoder.py
                                           │
                                           ▼
                                    integration_agent
                                      ├── encoder_registry.py
                                      ├── encoder_factory.py
                                      ├── configs/
                                      ├── model.py (modified)
                                      ├── config.py (modified)
                                      ├── test_encoder_registry.py
                                      └── test_hex_encoder_pipeline.py
                                           │
                                           ▼
                              ┌─────────────┴──────────────┐
                              ▼                            ▼
                        test_agent                  benchmark_agent
                          ├── test_hex_symmetry.py    ├── compare_encoders.py
                          ├── test_encoder_perf.py    └── encoder_ablation.py
                          └── test_conv2d_unchanged.py
```

### 5.2 Sequential Execution Order

```
PHASE A — Analysis (30 min)
  1. orchestrator: Read existing codebase (protocols, encoders, model, config)
  2. orchestrator: Validate assumptions about interface contracts
  3. orchestrator: Document current state (shapes, dims, config values)

PHASE B — Extraction (2-3 hours)
  4. extraction_agent: Clone Creativity-Neural-Network
  5. extraction_agent: Extract and adapt crystalline lattice layers
  6. extraction_agent: Create hex convolution module
  7. extraction_agent: Write and run unit tests for extracted layers
  8. orchestrator: VALIDATE GATE 1

PHASE C — Encoder (2-3 hours)
  9. encoder_agent: Read StateEncoder protocol and current encoder
  10. encoder_agent: Design hex encoder architecture
  11. encoder_agent: Implement HexSpatialEncoder
  12. encoder_agent: Write and run encoder unit tests
  13. orchestrator: VALIDATE GATE 2

PHASE D — Integration (3-4 hours)
  14. integration_agent: Create encoder registry and factory
  15. integration_agent: Create config schemas and YAML files
  16. integration_agent: Modify model.py to use factory
  17. integration_agent: Modify config.py to include encoder config
  18. integration_agent: Write and run integration tests
  19. orchestrator: VALIDATE GATE 3
  20. orchestrator: Run ALL existing tests (regression check)

PHASE E — Validation (2-3 hours)
  21. test_agent: Create property-based symmetry tests
  22. test_agent: Create performance benchmarks
  23. test_agent: Create regression tests for conv2d path
  24. test_agent: Run full test suite with coverage
  25. orchestrator: VALIDATE GATE 4

PHASE F — Benchmarking (1-2 hours)
  26. benchmark_agent: Create A/B comparison runner
  27. benchmark_agent: Run performance validation
  28. benchmark_agent: Create ablation runner
  29. orchestrator: VALIDATE GATE 5

PHASE G — Finalization (30 min)
  30. orchestrator: Run full CI pipeline (ruff, mypy, pytest, coverage)
  31. orchestrator: Generate integration report
  32. orchestrator: Create PR-ready commit
```

---

## SECTION 6: FILE MANIFEST

### 6.1 New Files Created

```
src/core/networks/
├── hex_conv.py                    # HexConvolution nn.Module
│                                  #   - 6-neighbor kernel topology
│                                  #   - Weight sharing with rotational symmetry
│                                  #   - Configurable kernel rings (1-ring, 2-ring)
│                                  #   - He init adapted for hex fan-in
│
├── crystalline_layers.py          # Extracted/adapted crystalline lattice layers
│                                  #   - CrystallineLatticeLayer (hex mode)
│                                  #   - Symmetry transformation functions
│                                  #   - Config dataclass for layer params
│
├── hex_encoder.py                 # HexSpatialEncoder (full encoder module)
│                                  #   - Implements StateEncoder protocol
│                                  #   - Configurable depth, dims, activation
│                                  #   - Contract: (B, 84, H, W) → (B, 256, H, W)
│
├── encoder_registry.py            # Encoder type registry
│                                  #   - register(name, cls, config_cls)
│                                  #   - get(name) → (cls, config_cls)
│                                  #   - Thread-safe singleton
│
└── encoder_factory.py             # Config-driven encoder instantiation
                                   #   - create_encoder(config) → nn.Module
                                   #   - Validates config against schema
                                   #   - Backwards-compatible defaults

configs/
├── encoders/
│   ├── hex_crystalline.yaml       # Hex encoder hyperparameters
│   └── conv2d_baseline.yaml       # Baseline encoder (extracted from current)
└── experiments/
    └── ab_encoder_comparison.yaml # Hydra multirun for A/B test

tests/
├── unit/
│   ├── test_hex_conv.py           # Hex convolution unit tests
│   ├── test_crystalline_layers.py # Crystalline layer unit tests
│   ├── test_hex_encoder.py        # Full encoder unit tests
│   └── test_encoder_registry.py   # Registry + factory unit tests
├── integration/
│   └── test_hex_encoder_pipeline.py  # Full pipeline integration tests
├── property/
│   └── test_hex_symmetry.py       # Hypothesis property-based tests
├── benchmarks/
│   └── test_encoder_performance.py   # Latency + memory benchmarks
└── regression/
    └── test_conv2d_unchanged.py   # Conv2d path regression lock

scripts/
├── compare_encoders.py            # A/B encoder comparison runner
└── encoder_ablation.py            # Hyperparameter sweep runner
```

### 6.2 Modified Files (Minimal, Additive)

```
src/core/networks/encoders.py      # ADD: register conv2d in registry at module load
src/core/networks/model.py         # REPLACE: direct Conv2d with encoder_factory call
src/core/config.py                 # ADD: EncoderConfig Pydantic model
src/core/protocols.py              # VERIFY: StateEncoder protocol sufficient (likely no change)
configs/networks/default.yaml      # ADD: encoder.spatial section with type="conv2d" default
```

---

## SECTION 7: DEEP REASONING CHECKPOINTS

At each gate, the orchestrator must perform explicit reasoning:

### Gate 1 Reasoning

```
QUESTION: Do the extracted layers produce correct shapes for 84-channel game state input?
VERIFY: instantiate CrystallineLatticeLayer(in_channels=84, ...), pass random tensor, check output
QUESTION: Are gradients flowing through hex conv layers?
VERIFY: forward → loss.sum() → backward(), check all param.grad is not None and is finite
QUESTION: Is 6-fold symmetry actually implemented correctly?
VERIFY: rotate input by 60° in hex coordinates, compare encoder output vs rotating output of unrotated input
```

### Gate 2 Reasoning

```
QUESTION: Does HexSpatialEncoder satisfy StateEncoder protocol?
VERIFY: isinstance check, or structural subtyping via mypy
QUESTION: Is the output shape EXACTLY matching the conv2d encoder?
VERIFY: for input shape (2, 84, 44, 44), both encoders produce (2, 256, 44, 44)
QUESTION: Will the residual tower accept this output without modification?
VERIFY: feed hex encoder output into first ResBlock, verify no shape error
```

### Gate 3 Reasoning

```
QUESTION: Can I load an old checkpoint (conv2d) with the new code?
VERIFY: save checkpoint with conv2d config, restart with hex config, load old checkpoint — conv2d 
        layers should load normally, hex layers should initialize fresh
QUESTION: Does the config default to conv2d if encoder section is missing?
VERIFY: load configs/networks/default.yaml WITHOUT encoder section, verify conv2d is used
QUESTION: Is the encoder truly hot-swappable via config?
VERIFY: change only encoder.spatial.type in YAML, verify different encoder is instantiated
```

### Gate 4 Reasoning

```
QUESTION: Do property tests actually verify hex-specific behavior, not just generic conv behavior?
VERIFY: run same property tests on Conv2d — they should FAIL (conv2d is not hex-equivariant)
QUESTION: Is the regression test actually locking conv2d behavior?
VERIFY: intentionally break conv2d encoder, verify regression test catches it
```

### Gate 5 Reasoning

```
QUESTION: Is the latency overhead acceptable for MCTS?
VERIFY: at 800 MCTS simulations per move with 68ms per forward pass, total search time is ~54s.
        With conv2d at 50ms, total is ~40s. Delta of 14s per move is significant.
        If overhead > 40%, consider: (a) reduce MCTS simulations, (b) optimize hex conv,
        (c) use torch.compile, (d) accept and note in report.
QUESTION: Would this encoder actually help in practice?
VERIFY: check if spatial encoding loss is lower with hex encoder on Tier 0.
        If not lower, the hypothesis is wrong — document and do not merge.
```

---

## SECTION 8: LOGGING SPECIFICATION

```yaml
logger_config:
  format: "structured_json"
  library: "structlog"
  correlation_id: true
  
  modules:
    hex_conv:
      creation: INFO    # "HexConv created: channels=84→256, symmetry=hexagonal, kernel_rings=1"
      forward: DEBUG    # "HexConv forward: input=(2,84,44,44), output=(2,256,44,44), time=2.3ms"
      gradient: DEBUG   # "HexConv backward: grad_norm=0.042, max_grad=0.15"
      
    crystalline_layers:
      creation: INFO    # "CrystallineLattice created: symmetry=hexagonal, order=6"
      symmetry: DEBUG   # "Symmetry transform applied: rotation=60°, group_element=2/6"
      
    hex_encoder:
      creation: INFO    # "HexSpatialEncoder created: layers=2, params=1,247,680"
      forward: DEBUG    # "HexEncoder forward: input=(2,84,44,44), output=(2,256,44,44)"
      
    encoder_registry:
      register: DEBUG   # "Encoder registered: name='hex_crystalline', cls=HexSpatialEncoder"
      lookup: DEBUG     # "Encoder lookup: name='hex_crystalline' → found"
      fallback: WARNING # "Encoder 'unknown' not found, falling back to 'conv2d'"
      
    encoder_factory:
      creation: INFO    # "Encoder created: type='hex_crystalline', params=1,247,680"
      config: DEBUG     # "Encoder config resolved: {symmetry_type: hexagonal, ...}"
      
    model:
      encoder_swap: INFO # "Model encoder: conv2d → hex_crystalline (config change)"
      checkpoint_load: INFO # "Checkpoint loaded: encoder_type=conv2d, migrating to hex_crystalline"
      
    benchmark:
      result: INFO      # "Encoder benchmark: hex=68ms, conv2d=50ms, overhead=36%"
      comparison: INFO  # "A/B comparison at 10K games: hex_elo=1042, conv2d_elo=1000"
```

---

## SECTION 9: DEBUGGING REFERENCE

### Common Issues and Resolutions

```yaml
issue_1:
  symptom: "RuntimeError: shape mismatch at residual tower input"
  cause: "Hex encoder output channels ≠ 256 or spatial dims changed"
  debug: |
    - Check hex_encoder forward output: assert output.shape == (B, 256, H, W)
    - Check config: encoder_dim must equal backbone input channels
    - Add assert in hex_encoder.forward() before return
  fix: "Ensure encoder config encoder_dim matches network config backbone_channels"

issue_2:
  symptom: "NaN loss after switching to hex encoder"
  cause: "Hex conv weight initialization too large/small for 6-neighbor fan-in"
  debug: |
    - Log weight statistics: model.hex_encoder[0].weight.data.std()
    - Expected std ≈ sqrt(2 / (6 * in_channels)) for He init with 6 neighbors
    - Check gradient norms: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  fix: "Use He init with fan_in = 6 * in_channels, add gradient clipping warmup"

issue_3:
  symptom: "Hex encoder produces same output as conv2d (no improvement)"
  cause: "Hex kernel actually using rectangular connectivity (bug in kernel topology)"
  debug: |
    - Run property test: test_hex_rotation_equivariance — should pass for hex, fail for conv2d
    - Visualize kernel connectivity: plot which input positions affect each output position
    - Check that only 6 hex-adjacent positions have non-zero kernel weights
  fix: "Verify hex kernel index mapping against offset coordinate hex adjacency table"

issue_4:
  symptom: "Old checkpoint fails to load with new code"
  cause: "Model state dict has different keys due to encoder restructuring"
  debug: |
    - Compare state dict keys: old_keys = torch.load(ckpt)['model'].keys()
    - Check if encoder_factory adds a prefix like 'encoder.' to parameter names
    - Log key mapping during checkpoint load
  fix: "Add key migration in checkpoint.py: strip/add prefix as needed"

issue_5:
  symptom: "Config without encoder section causes KeyError"
  cause: "EncoderConfig not defaulting when section is missing"
  debug: |
    - Load config YAML without encoder section
    - Check Pydantic model has default_factory for encoder field
    - Check Hydra resolver handles missing sections
  fix: "Add default_factory=lambda: EncoderConfig(type='conv2d') to NetworkConfig.encoder"

issue_6:
  symptom: "Property tests pass for hex encoder AND conv2d (test is wrong)"
  cause: "Rotation test not actually rotating in hex coordinates"
  debug: |
    - Hex 60° rotation in offset coords is NOT the same as 90° tensor rotation
    - Verify rotation function uses proper hex coordinate transform
    - The test should FAIL for Conv2d — if it passes, the rotation is wrong
  fix: "Implement proper hex rotation: convert to cube coords, rotate, convert back"
```

---

## SECTION 10: VERIFICATION COMMANDS

```bash
# ─── After each sub-agent completes ───

# Gate 1: Layer extraction
pytest tests/unit/test_crystalline_layers.py tests/unit/test_hex_conv.py -v

# Gate 2: Encoder construction  
pytest tests/unit/test_hex_encoder.py -v

# Gate 3: Integration
pytest tests/unit/test_encoder_registry.py -v
pytest tests/integration/test_hex_encoder_pipeline.py -v
mypy src/core/networks/ --strict

# Gate 4: Full test suite
pytest tests/ -v --tb=short -x
pytest tests/property/ --hypothesis-seed=0
pytest tests/ --cov=src/core/networks --cov-report=term-missing --cov-fail-under=95

# Gate 5: Performance
pytest tests/benchmarks/ -v --benchmark-only
python scripts/compare_encoders.py --num-games 100 --tier 0 --device cpu

# ─── Final CI pipeline ───
ruff check src/ tests/ scripts/
mypy src/ --strict
pytest tests/ -v --tb=short --cov=src/core/networks --cov-report=html --cov-fail-under=95
```

---

## SECTION 11: SESSION MEMORY (CLAUDE.md Updates)

After completing this integration, update the project CLAUDE.md with:

```markdown
## Encoder System

### Architecture
- Encoders are registered via `encoder_registry.py` and instantiated via `encoder_factory.py`
- Selection is config-driven: `configs/encoders/{type}.yaml`
- Default encoder: `conv2d` (backwards compatible)
- Available: `conv2d`, `hex_crystalline`

### Key Invariants
- StateEncoder protocol: input (B, C_in, H, W) → output (B, C_out, H, W)
- C_in and C_out are ALWAYS from config, never hardcoded
- Old checkpoints load with conv2d encoder regardless of current config
- New checkpoints include encoder_type metadata

### Testing
- Property tests in tests/property/ verify hex symmetry invariance
- Regression tests in tests/regression/ lock conv2d behavior
- Benchmarks in tests/benchmarks/ track latency overhead
- Coverage requirement: >= 95% on src/core/networks/

### Common Commands
- Run encoder tests: `pytest tests/ -k encoder -v`
- Compare encoders: `python scripts/compare_encoders.py --tier 0`
- Ablation sweep: `python scripts/encoder_ablation.py --multirun`
```

---

*End of implementation plan. All sections are self-contained and can be executed sequentially by Claude Code agents following the dependency graph in Section 5.*
