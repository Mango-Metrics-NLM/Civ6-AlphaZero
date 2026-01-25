# Changelog

All notable changes to the Civ6 AlphaZero project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Real Civ6 Bridge System
A complete bridge system enabling the AlphaZero agent to play actual Civilization VI through file-based IPC.

##### Python Bridge (`alpha_zero/bridge/`)
- **config.py**: `BridgeConfig` dataclass with validation
  - Communication directory configuration
  - Polling intervals, timeouts, retry settings
  - Map size configurations for coordinate scaling
  - Environment variable support (`CIV6_BRIDGE_*`)
- **protocol.py**: Message types and JSON schemas
  - `GameStateData`, `UnitData`, `CityData`, `TileData` for state representation
  - `LuaCommand`, `CommandMessage`, `AckMessage` for action communication
  - `LegalAction` for action masking
  - Serialization/deserialization with validation
- **file_bridge.py**: Async file-based IPC
  - `FileBridge`: Polls state.json, writes commands.json, reads ack.json
  - Heartbeat monitoring for connection health
  - Automatic retry with exponential backoff
  - `MockFileBridge` for testing without real files
- **state_converter.py**: State format transformation
  - Converts Civ6 Lua JSON to encoder-compatible dict
  - Coordinate scaling from Civ6 map to 16x16 board
  - Terrain, unit, city type mappings
  - Legal action preservation
- **action_converter.py**: Action index to Lua command
  - Maps ~483K hierarchical actions to `LuaCommand`
  - Action mask generation from legal actions
  - Validation with detailed error reporting

##### Gymnasium Environment (`alpha_zero/envs/real_civ6_env.py`)
- **RealCiv6Env**: Full Gymnasium-compliant environment
  - Dict observation space: spatial (84, 16, 16), vector (512,), action_mask
  - Discrete action space from hierarchical action encoding
  - Async bridge communication with configurable timeouts
- **MockCiv6Env**: Testing environment
  - Built-in mock state generation
  - No file I/O required
  - Compatible with all training code

##### Game Interface (`alpha_zero/interfaces/civ6_interface.py`)
- **Civ6Interface**: Implements `BaseGameInterface`
  - `connect()`, `disconnect()` for session management
  - `reset()`, `step()` for game control
  - `get_observation()`, `get_valid_actions()` for state access
  - `save_state()`, `load_state()` for checkpointing
  - Registered as "civ6" backend with `GameAdapter`

##### Civilization VI Lua Mod (`mod/AlphaZeroAI/`)
- **AlphaZeroAI.modinfo**: Mod metadata and configuration
  - File references and dependencies
  - Action criteria for enabling AI control
- **Config.lua**: Centralized configuration management
  - Communication directory, file names
  - Polling intervals, feature flags
  - Dynamic configuration overrides
- **Logger.lua**: Logging system
  - Log levels: DEBUG, INFO, WARN, ERROR
  - Console and file output
  - Scoped loggers per module
- **JsonEncoder.lua**: Pure Lua JSON encoder/decoder
  - No external dependencies (Civ6 Lua is sandboxed)
  - Handles all JSON types, escaping, unicode
- **FileIO.lua**: File operations
  - Read/write with error handling
  - Change tracking for efficient updates
  - Heartbeat, state, command, ack file management
- **StateExporter.lua**: Game state serialization
  - Map tiles with terrain, features, resources
  - Units with full stats and capabilities
  - Cities with production, buildings, population
  - Legal actions enumeration
- **ActionExecutor.lua**: Command execution
  - Unit actions: move, attack, fortify, sleep, etc.
  - City actions: production, purchase
  - Research: technology and civic selection
  - Validation and error reporting
- **Main.lua**: Entry point and orchestration
  - Event handlers for turn start/end
  - Command polling loop
  - Acknowledgment writing

##### Test Suite
- **tests/unit/test_bridge.py**: 93 unit tests for bridge components
- **tests/unit/test_state_converter.py**: State conversion tests
- **tests/unit/test_action_converter.py**: Action conversion tests
- **tests/integration/test_bridge_integration.py**: End-to-end pipeline tests
- **tests/integration/test_real_civ6_env.py**: Environment integration tests
- **tests/fixtures/game_states/**: Sample JSON fixtures
  - `turn_1_start.json`: Initial game state
  - `turn_10_mid.json`: Mid-game with units and cities
  - `turn_50_combat.json`: Combat scenario

#### Phase 3: Enhanced Training Pipeline
A comprehensive suite of training improvements designed to accelerate learning convergence and improve agent performance.

##### Curriculum Learning System (`alpha_zero/utils/curriculum_learning.py`)
- **CurriculumStage**: Dataclass defining difficulty stages with configurable parameters
  - Board size, max turns, opponent strength, starting resources
  - Win rate thresholds for automatic promotion/demotion
  - Minimum games requirement before stage transitions
- **CurriculumManager**: Orchestrates multi-stage progression
  - Three default stages: Beginner (6x6), Intermediate (8x8), Advanced (10x10)
  - Automatic stage promotion when win rate exceeds threshold
  - Automatic demotion when win rate drops below minimum threshold
  - Stage history tracking with detailed metrics
- **AdaptiveDifficultyController**: Fine-grained difficulty adjustments within stages
  - Dynamic opponent strength scaling
  - Configurable adjustment rates and bounds

##### Potential-Based Reward Shaping (`alpha_zero/utils/enhanced_rewards.py`)
- **WinConditionEvaluator**: Multi-factor game state evaluation
  - Score advantage assessment with configurable weights
  - City count and production evaluation
  - Technology progress tracking
  - Unit strength differential calculation
- **EnhancedRewardShaper**: Policy-preserving reward augmentation
  - Potential-based shaping (guarantees optimal policy preservation)
  - Configurable gamma discount factor
  - Composite potential from multiple game state factors
  - Smooth reward signals for faster credit assignment

##### Environment Tuning (`alpha_zero/utils/environment_tuning.py`)
- **EarlyTerminationConfig**: Configurable early game termination
  - Decisive advantage detection
  - Stalemate prevention
  - Turn limit enforcement with graceful handling
- **ProductionConfig**: Accelerated production mechanics
  - Production multipliers for faster game progression
  - Growth rate adjustments
  - Research speed modifications
- **TunedEnvironmentWrapper**: Gymnasium-compliant wrapper
  - Seamless integration with existing training loops
  - Configurable through dataclass parameters
  - Observation and action space preservation

##### Training Configuration (`alpha_zero/utils/training_config_enhanced.py`)
- **ParameterSchedule**: Six schedule types for hyperparameter annealing
  - Constant: Fixed value throughout training
  - Step: Discrete value changes at specified iterations
  - Linear: Smooth linear interpolation
  - Exponential: Exponential decay/growth
  - Cosine: Cosine annealing with smooth transitions
  - Warmup-Cosine: Linear warmup followed by cosine decay
- **ResourceManager**: System resource management
  - Automatic device selection (CUDA/MPS/CPU)
  - Memory-aware batch sizing
  - Worker count optimization

##### Weights & Biases Integration (`alpha_zero/utils/wandb_logger.py`)
- **WandbConfig**: Typed configuration for W&B initialization
  - Project, entity, run name, and tags configuration
  - Mode selection (online/offline/disabled)
  - Artifact logging configuration
- **WandbLogger**: Comprehensive experiment tracking
  - `init()`: Initialize W&B run with full config
  - `log()`: Generic metric logging with step tracking
  - `log_selfplay()`: Self-play statistics (games, buffer size, samples)
  - `log_training_step()`: Training metrics (loss, policy/value loss, LR)
  - `log_evaluation()`: Evaluation results (win rate, game stats)
  - `log_model()`: Model artifact versioning with metadata
  - `summary()`: Final run summary statistics
  - `finish()`: Graceful run completion

##### Enhanced Training Orchestration (`run_enhanced_training.py`)
- **Training Presets**: Pre-configured training profiles
  - Fast: Quick iteration for development (2 iterations, 2 games)
  - Balanced: Standard training (10 iterations, 20 games)
  - Full: Production training (50 iterations, 100 games)
- **Command-Line Interface**: Full argument parsing
  - All training parameters configurable via CLI
  - W&B integration flags (`--wandb`, `--wandb-project`, `--wandb-tags`)
  - Device selection and checkpoint management
- **History Tracking**: Comprehensive training history
  - Iteration-level metrics
  - Loss tracking with component breakdown
  - Curriculum stage progression
  - Final summary statistics

##### Simplified MCTS (`alpha_zero/utils/simple_mcts.py`)
- Lightweight MCTS implementation for evaluation
- Configurable simulation count
- Temperature-based action selection
- Compatible with enhanced training pipeline

#### New Run Scripts
- `run_enhanced_training.py`: Main training orchestration with all Phase 3 features
- `run_training_wandb.py`: W&B-focused training runner
- `run_civ6_ai.py`: Simple AI evaluation runner

#### Test Suite Expansion
##### Unit Tests (14 new files)
- `tests/unit/test_curriculum_learning.py`: Curriculum stage and manager tests
- `tests/unit/test_enhanced_rewards.py`: Reward shaping validation
- `tests/unit/test_environment_tuning.py`: Environment wrapper tests
- `tests/unit/test_training_config_enhanced.py`: Parameter schedule tests
- `tests/unit/test_wandb_logger.py`: W&B logger unit tests
- `tests/unit/test_enhanced_training.py`: Training function signature tests
- `tests/unit/test_enhanced_training_wandb.py`: W&B integration unit tests
- `tests/unit/test_simple_mcts.py`: Simplified MCTS tests
- `tests/unit/test_evaluate.py`: Evaluation function tests
- `tests/unit/test_self_play.py`: Self-play generation tests
- `tests/unit/test_training_loop.py`: Training loop tests
- `tests/unit/test_replay_buffer.py`: Replay buffer tests
- `tests/unit/test_integration_components.py`: Component integration tests
- `tests/unit/test_performance.py`: Performance benchmark tests

##### Integration Tests (6 new files)
- `tests/integration/test_enhanced_training.py`: End-to-end training tests
- `tests/integration/test_enhanced_training_wandb.py`: W&B integration tests
- `tests/integration/test_training_pipeline.py`: Full pipeline tests
- `tests/integration/test_curriculum_integration.py`: Curriculum system integration
- `tests/integration/test_reward_shaping_integration.py`: Reward shaping validation
- `tests/integration/test_full_training_cycle.py`: Complete training cycle tests

#### Documentation
- `PLAN_IMPROVE_LEARNING.md`: Implementation plan for Phase 3 improvements
- `TRAINING_IMPROVEMENTS_TEMPLATE.md`: Template for training experiments
- `.env.template`: Environment variable template for W&B configuration
- Updated `CLAUDE.md`: Comprehensive project documentation refresh

### Changed

#### Modified Files
- **`.gitignore`**: Added W&B directories, training artifacts, profiling files
- **`alpha_zero/utils/__init__.py`**: Exported all new Phase 3 modules
  - CurriculumStage, CurriculumManager, AdaptiveDifficultyController
  - EnhancedRewardShaper, WinConditionEvaluator
  - TunedEnvironmentWrapper, EarlyTerminationConfig, ProductionConfig
  - ParameterSchedule, ResourceManager
  - WandbLogger, WandbConfig, WANDB_AVAILABLE flag
- **`requirements.txt`**: Added new dependencies
  - `wandb>=0.15.0`: Experiment tracking
  - `gymnasium>=0.29.0`: Environment compatibility
  - `tqdm>=4.65.0`: Progress bars
  - Additional testing dependencies

### Technical Highlights

#### Design Patterns
- **Dataclass Configuration**: All configuration uses typed dataclasses for IDE support
- **Factory Pattern**: ParameterSchedule.create() for schedule instantiation
- **Wrapper Pattern**: TunedEnvironmentWrapper for non-invasive environment modification
- **Strategy Pattern**: Multiple schedule types implementing common interface
- **Optional Dependencies**: W&B integration with graceful fallback when unavailable

#### Performance Considerations
- Potential-based reward shaping preserves optimal policy (mathematically proven)
- Curriculum learning reduces sample complexity through staged difficulty
- Early termination prevents wasted computation on decided games
- Batch-aware resource management for memory efficiency

#### Testing Philosophy
- 400+ test methods across 33 test files
- Unit tests validate individual components in isolation
- Integration tests verify component interactions
- Mock-based W&B tests avoid API calls during CI
- Parametrized tests cover edge cases and boundary conditions

## [Previous Releases]

For changes prior to the Phase 3 Enhanced Training Pipeline, please refer to the git history.

---

*This changelog documents the feature/improve-learning-pipeline branch changes.*
