# Changelog

All notable changes to the Civ6 AlphaZero project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Phase 5.3: Cost Optimization for GCP Cloud Training

A comprehensive cost optimization system targeting 60-80% cost savings through spot VMs, dynamic GPU scheduling, auto-stop idle instances, and risk mitigation systems.

##### Cost Monitoring & Budget Enforcement (`alpha_zero/cloud/gcp/cost_monitor.py`)

- **CostMonitorConfig**: Configurable budget limits with environment variables (`CIV6AZ_COST_*`)
  - Daily and total budget limits in USD
  - Alert thresholds (warning at 80%, critical at 95%)
  - Check interval configuration
- **CostMonitor**: Real-time cost tracking and budget enforcement
  - GPU pricing estimates for T4, V100, P100, A100, L4
  - Alert callbacks for budget thresholds
  - Background monitoring thread
  - `can_afford()` for budget checks before launching jobs
- **Exceptions**: `BudgetExceededError` for budget limit handling

##### Dynamic GPU Scheduler (`alpha_zero/cloud/gcp/gpu_scheduler.py`)

- **GPUSchedulerConfig**: Training phase-aware GPU selection
  - Phase thresholds: Early (0-50%), Middle (50-80%), Late (80-90%), Final (90-100%)
  - Configurable GPU mappings per phase
  - Spot VM preference settings
- **DynamicGPUScheduler**: T4 → V100 upgrade strategy
  - Uses cheaper T4 GPUs for early/middle training phases
  - Automatically upgrades to V100 for final phase precision
  - Budget-aware selection with cost monitor integration
  - 42 unit tests

##### Preemption Handler (`alpha_zero/cloud/gcp/preemption_handler.py`)

- **PreemptionHandlerConfig**: Checkpoint and retry settings
  - 5-minute checkpoint intervals (configurable)
  - Exponential backoff with max 5 retries
  - GCS storage integration
- **PreemptionHandler**: Spot VM preemption recovery
  - SIGTERM signal handling for graceful shutdown
  - Automatic periodic checkpointing
  - Resume state persistence to GCS
  - Checkpoint callbacks for integration
- **Exceptions**: Enhanced `QuotaExceededError` with fallback configs

##### Idle Instance Manager (`alpha_zero/cloud/gcp/idle_manager.py`)

- **IdleInstanceConfig**: Idle detection settings
  - Configurable idle threshold (default 15 minutes)
  - GPU utilization threshold (default 10%)
  - Per-job-type policies (selfplay, training, evaluation)
  - Grace period before shutdown
- **IdleInstanceManager**: Auto-stop idle instances
  - Job registration and activity tracking
  - Idle detection by time and GPU utilization
  - Checkpoint-before-stop logic
  - Warning and stop callbacks
- **Exceptions**: `IdleTimeoutError` for recoverable shutdowns

##### Quota Manager (`alpha_zero/cloud/gcp/quota_manager.py`)

- **QuotaManagerConfig**: Quota tracking settings
  - Warning/critical thresholds (70%/90%)
  - Cache TTL for API responses
  - Fallback strategies configuration
- **QuotaManager**: GCP quota management and fallback
  - Quota usage tracking with caching
  - Fallback strategies: alternative GPUs, regions, spot VMs
  - Quota exhaustion prediction
  - Warning and exhaustion callbacks

##### Memory Guard (`alpha_zero/utils/memory_guard.py`)

- **MemoryGuardConfig**: Memory protection settings
  - Warning/critical/OOM thresholds (70%/85%/95%)
  - Batch reduction factor and minimum batch size
  - Gradient checkpointing settings
- **MemoryGuard**: GPU/CPU memory protection
  - Real-time memory monitoring (PyTorch CUDA + psutil)
  - Automatic batch size reduction on memory pressure
  - Gradient checkpointing activation
  - OOM prevention with garbage collection

##### Convergence Monitor (`alpha_zero/training/convergence_monitor.py`)

- **ConvergenceMonitorConfig**: Training monitoring settings
  - Window size for trend analysis
  - Stall detection threshold
  - Improvement thresholds
- **ConvergenceMonitor**: Training stall detection
  - Status tracking: IMPROVING, STABLE, SLOWING, STALLED, DIVERGING
  - PBT mutation recommendations
  - Curriculum advancement/retreat recommendations
  - Learning rate reduction suggestions

##### Bridge Health Checker (`alpha_zero/bridge/health_checker.py`)

- **BridgeHealthConfig**: Health monitoring settings
  - Check interval and timeout configuration
  - Circuit breaker thresholds
  - Exponential backoff settings
- **GameBridgeHealthChecker**: Resilient bridge communication
  - Circuit breaker pattern (CLOSED → OPEN → HALF_OPEN)
  - Exponential backoff retry logic
  - Health status: HEALTHY, DEGRADED, UNHEALTHY
  - Operation timeout handling
- **Exceptions**: `CircuitBreakerOpenError`, `OperationTimeoutError`

##### Unified Cost Optimization Manager (`alpha_zero/cloud/gcp/cost_optimization.py`)

- **CostOptimizationConfig**: Unified configuration
  - Combines all component configs
  - Environment variable support (`CIV6AZ_COST_OPT_*`)
  - YAML serialization
- **CostOptimizationManager**: Single interface for cost optimization
  - Coordinates all cost optimization components
  - `get_optimized_job_config()`: Returns optimal GPU/spot/region config
  - `get_savings_report()`: Comprehensive savings tracking
  - Job lifecycle management (start, activity, end)
  - Graceful initialization and shutdown

##### Test Suite (264 new tests)

- `tests/cloud/unit/test_cost_monitor.py`: Cost monitoring tests (42 tests)
- `tests/cloud/unit/test_gpu_scheduler.py`: GPU scheduler tests (39 tests)
- `tests/cloud/unit/test_preemption_handler.py`: Preemption tests (43 tests)
- `tests/cloud/unit/test_idle_manager.py`: Idle management tests (46 tests)
- `tests/unit/test_bridge_health.py`: Health checker tests (53 tests)
- All tests pass with comprehensive coverage

##### Configuration Pattern

All components follow consistent configuration patterns:

- Dataclass-based configs with validation
- Environment variable support with `CIV6AZ_` prefix
- YAML file loading support
- `from_dict()` and `to_dict()` serialization
- Thread-safe implementations with proper locking

#### Phase 4.1: Model Evaluation Framework

A comprehensive model evaluation system for measuring trained model performance against configurable baselines.

##### Evaluation Module (`alpha_zero/evaluation/`)

- **EvaluationConfig** (`config.py`): Unified configuration for evaluation
  - Experiment naming and output directory configuration
  - Board size and max turns settings
  - ELO rating integration settings
  - Environment variable support (`CIV6AZ_EVAL_*` prefix)

- **BaselineConfig** (`config.py`): Configuration for baseline opponents
  - Support for RANDOM, HEURISTIC, and NEURAL_NETWORK baseline types
  - Per-baseline game count and weight configuration
  - HeuristicWeights for customizable heuristic behavior

- **DiversityConfig** (`config.py`): Diversity tracking configuration
  - Opening move tracking with configurable depth
  - Game length distribution analysis

- **BaselinePlayer** (`baselines.py`): Protocol and implementations
  - RandomPlayer: Uniform random action selection with seeding
  - HeuristicPlayer: Configurable weighted heuristic evaluation
  - NeuralNetworkPlayer: Model checkpoint loading with optional MCTS
  - BaselineFactory: Factory pattern for baseline creation

- **ModelEvaluator** (`evaluator.py`): Main evaluation orchestrator
  - Integration with existing ELORatingSystem from tournament/elo.py
  - GameResult and EvaluationResult dataclasses
  - play_game(), evaluate_against_baseline(), evaluate_model() methods

- **DiversityTracker** (`diversity.py`): Game diversity analysis
  - Opening sequence tracking with entropy calculation
  - Game length statistics (mean, std, min, max)
  - Baseline-specific diversity breakdowns

- **ReportGenerator** (`reports.py`): Multi-format report generation
  - EvaluationReport dataclass with full evaluation results
  - JSON export with to_json() method
  - Markdown export with to_markdown() method

##### Evaluation Test Suite (62 new tests)

- `tests/unit/test_evaluation/test_config.py`: Configuration validation (10 tests)
- `tests/unit/test_evaluation/test_baselines.py`: Baseline player tests (15 tests)
- `tests/unit/test_evaluation/test_diversity.py`: Diversity tracking tests (12 tests)
- `tests/unit/test_evaluation/test_evaluator.py`: Evaluator tests (14 tests)
- `tests/unit/test_evaluation/test_reports.py`: Report generation tests (11 tests)
- `tests/integration/test_evaluation_pipeline.py`: Integration tests (11 tests)

##### Evaluation Configuration (`configs/evaluation.yaml`)

- Complete YAML configuration template
- Random and heuristic baseline definitions
- Diversity tracking settings
- ELO integration settings

#### Phase 4.2: PBT Enhancements

Enhanced Population-Based Training with temperature scheduling and crossover mutations.

##### Temperature Schedule (`alpha_zero/pbt/config.py`)

- **HyperparameterSpace** enhancements:
  - temperature_initial: Starting temperature for exploration
  - temperature_final: Ending temperature for exploitation
  - temperature_decay_steps: Steps over which temperature decays
  - Updated get_all_params() and serialization methods

##### Crossover Mutation (`alpha_zero/pbt/mutations.py`)

- **CrossoverMutation** class:
  - BLX-alpha blending crossover (default α=0.5)
  - Simulated Binary Crossover (SBX) with η=20
  - Uniform crossover with 50% probability
  - Automatic fallback to perturbation without parent

- **MutationOperator** enhancements:
  - CROSSOVER strategy in MutationStrategy enum
  - set_crossover_parent() method for parent injection
  - Integration with existing mutation infrastructure

#### Phase 4.3: Performance Optimizations

Research-based optimizations for faster training and improved sample efficiency.

##### Optimizations Module (`alpha_zero/optimizations/`)

- **PlayoutCapRandomizer** (`playout_cap.py`): Playout cap randomization
  - Geometric, uniform, and exponential cap distributions
  - Position complexity scaling for root vs non-root nodes
  - Statistics tracking (total samples, average cap, speedup estimates)
  - Configurable min/max playouts with geometric_p parameter

- **PolicyTargetPruner** (`policy_pruning.py`): Policy target pruning
  - Top-k preservation for important actions
  - Minimum probability threshold filtering
  - Maximum actions limiting
  - Entropy-based dynamic thresholding
  - Action reduction statistics tracking

- **CyclicLRScheduler** (`cyclic_lr.py`): Cyclic learning rate schedules
  - Five schedule types: triangular, triangular2, exp_range, cosine_annealing, warm_restarts
  - Momentum cycling with base_momentum and max_momentum
  - Warmup steps support
  - Iteration tracking and state serialization

- **AuxiliaryTargetGenerator** (`auxiliary_targets.py`): Auxiliary policy targets
  - Attention target generation from board positions
  - Opponent modeling for predicting opponent moves
  - Game phase prediction (early/mid/late classification)
  - Future value prediction for N-step lookahead
  - AuxiliaryLossCalculator for weighted loss computation

- **PerformanceOptimizationsConfig** (`config.py`): Unified configuration
  - Enable flags for each optimization type
  - Nested configs for all optimization parameters
  - fast_preset() and quality_preset() factory methods
  - YAML serialization and environment variable support

##### Optimizations Test Suite (148 new tests)

- `tests/unit/test_optimizations/test_playout_cap.py`: Playout cap tests (26 tests)
- `tests/unit/test_optimizations/test_policy_pruning.py`: Policy pruning tests (24 tests)
- `tests/unit/test_optimizations/test_cyclic_lr.py`: Cyclic LR tests (30 tests)
- `tests/unit/test_optimizations/test_auxiliary_targets.py`: Auxiliary targets tests (51 tests)
- `tests/unit/test_optimizations/test_config.py`: Config tests (17 tests)
- `tests/integration/test_optimizations_integration.py`: Integration tests (17 tests)

##### Optimizations Configuration (`configs/optimizations.yaml`)

- Complete YAML configuration template
- Playout cap settings with distribution parameters
- Policy pruning thresholds
- Cyclic LR schedule configuration
- Auxiliary target weights

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

#### Phase 4: Unified Training Orchestrator

A single, configurable entry point for full AlphaZero training with integrated self-play, MCTS, neural networks, reward shaping, curriculum learning, and comprehensive logging.

##### Core Orchestrator Module (`alpha_zero/orchestrator/`)

- **OrchestratorConfig** (`config.py`): Unified configuration system
  - YAML-based configuration with environment variable overrides (`CIV6AZ_` prefix)
  - Preset loading: `fast`, `balanced`, `production`
  - Full validation with warnings for invalid combinations
  - Nested configs for environment, network, MCTS, training, rewards, curriculum

- **TrainingOrchestrator** (`orchestrator.py`): Main training loop controller
  - Single entry point replacing multiple training scripts
  - Integrated self-play with MCTS-guided game generation
  - Automatic curriculum stage progression
  - Periodic evaluation and checkpointing
  - Graceful shutdown with signal handlers (SIGINT, SIGTERM)
  - Resume from checkpoint support

- **HyperparameterScheduler** (`scheduler.py`): Dynamic hyperparameter control
  - Six schedule types: constant, linear, cosine, step, warmup_cosine, exponential
  - Learning rate, batch size, temperature, and exploration scheduling
  - Step-based progression with configurable decay

- **CheckpointManager** (`checkpointer.py`): Model persistence
  - Automatic checkpoint rotation (configurable retention)
  - Best model tracking by win rate
  - Full training state save/resume
  - JSON metadata for checkpoint history

- **PaddedEnvironmentWrapper**: Curriculum board size handling
  - Pads observations for smaller curriculum boards to network's expected size
  - Consistent interface across all board sizes
  - Proper cloning support for MCTS simulations

##### Component Factories (`alpha_zero/utils/factories.py`)

- `create_network()`: Neural network instantiation with board/action size config
- `create_mcts()`: MCTS with optional SimpleMCTS fallback
- `create_environment()`: Environment creation with override support
- `create_reward_shaper()`: Reward shaping initialization
- `create_curriculum_manager()`: Curriculum system setup
- `create_replay_buffer()`: Experience replay initialization
- `create_wandb_logger()`: W&B logging setup

##### CLI Entry Point (`alpha_zero/orchestrator/__main__.py`)

- `python -m alpha_zero.orchestrator` command
- Preset selection: `--preset fast|balanced|production`
- Custom config: `--config path/to/config.yaml`
- Dry run mode: `--dry_run`
- Resume support: `--resume`

##### Configuration Files

- `configs/orchestrator.yaml`: Master configuration template
- `configs/orchestrator_fast.yaml`: Fast testing preset (6x6 board, 10 iterations)
- `configs/orchestrator_production.yaml`: Production training settings

#### RAG (Retrieval-Augmented Generation) Integration

A comprehensive RAG system for enhancing agent decision-making with external knowledge retrieval.

##### Configuration Module (`alpha_zero/rag/config/settings.py`)

- **VectorStoreConfig**: Vector database configuration
  - Support for Memory, ChromaDB, FAISS, and Pinecone backends
  - Configurable dimensions, distance metrics, and collection names
- **EmbeddingConfig**: Embedding model configuration
  - Support for OpenAI, Sentence Transformers, and Local embeddings
  - Batch size and dimension configuration
- **ProcessorConfig**: Text processing configuration
  - Chunk size and overlap settings
  - Recursive text splitting parameters
- **RAGConfig**: Unified configuration with YAML serialization
  - Environment variable loading
  - Configuration validation

##### Vector Store Abstraction (`alpha_zero/rag/stores/`)

- **VectorStore Protocol**: Abstract interface for vector databases
  - Async CRUD operations (add, search, delete, get)
  - Metadata filtering support
- **MemoryVectorStore**: In-memory implementation for testing
  - Cosine similarity and Euclidean distance metrics
  - NumPy-based efficient search
- **ChromaStore**: ChromaDB integration
  - Persistent and ephemeral collection support
- **FAISSStore**: FAISS integration
  - High-performance similarity search
- **VectorStoreFactory**: Factory pattern for store creation

##### Document Processing (`alpha_zero/rag/processors/`)

- **RecursiveTextSplitter**: Intelligent text chunking
  - Configurable separators and chunk sizes
  - Overlap for context preservation
- **TextDocumentProcessor**: Document processing pipeline
  - Async file and text processing
  - Chunk metadata tracking
- **EmbeddingGenerator Protocol**: Abstract embedding interface
- **LocalEmbeddings**: Random embeddings for testing
- **OpenAIEmbeddings**: OpenAI API integration
- **SentenceTransformerEmbeddings**: Local transformer models

##### Agent Workflow Integration (`alpha_zero/rag/agents/`)

- **RAGAgentWorkflow**: Complete workflow orchestration
  - Document ingestion and indexing
  - Query processing with context retrieval
  - Hook system for extensibility
  - Step-by-step execution tracking
- **ContextBuilder**: Context assembly strategies
  - Relevance-based, recency-based, diversity-based, and hybrid strategies
  - Deduplication and character limits
- **QueryEnhancer**: Query preprocessing
  - Keyword extraction
  - Query expansion with synonyms
  - Query normalization

##### Utilities (`alpha_zero/rag/utils/`)

- **RAGLogger**: Structured logging with correlation IDs
  - Multiple output formats (JSON, text)
  - Configurable log levels
- **MetricsCollector**: Performance metrics
  - Timer, Counter, and Gauge types
  - Context manager support for timing
- **TTLCache & LRUCache**: Caching implementations
  - Time-to-live expiration
  - Size-based eviction

##### Main Pipeline (`alpha_zero/rag/pipeline.py`)

- **RAGPipeline**: Unified interface for RAG operations
  - Text and file ingestion
  - Query with context enhancement
  - Caching and metrics collection
  - Factory methods for testing configuration
  - Async context manager support

##### RAG Test Suite (100 new tests)

- `tests/unit/test_rag/test_config.py`: Configuration tests (20 tests)
- `tests/unit/test_rag/test_vector_stores.py`: Vector store tests (22 tests)
- `tests/unit/test_rag/test_processors.py`: Processor tests (25 tests)
- `tests/unit/test_rag/test_pipeline.py`: Pipeline tests (19 tests)
- `tests/integration/test_rag/test_rag_workflow.py`: Integration tests (14 tests)

##### RAG Dependencies

- `chromadb>=0.4.0`: ChromaDB vector database
- `faiss-cpu>=1.7.0`: Facebook AI Similarity Search
- `openai>=1.0.0`: OpenAI API client
- `sentence-transformers>=2.2.0`: Local embedding models
- `pinecone-client>=2.0.0` (optional): Pinecone cloud vector database

### Fixed

#### Action Mask Indexing Bug (`alpha_zero/core/civ6_actions.py`)

- Fixed critical bug where action mask indices didn't match action execution
- Changed from using global unit/city indices to player-relative indices
- Ensures valid actions map correctly to game state modifications
- Reduced average game length from 700+ steps to 70-90 steps

#### Evaluation Environment Handling

- Fixed `'MicroCivEnv' object has no attribute '_env'` error during evaluation
- Always create fresh wrapped environment for each game
- Consistent interface between self-play and evaluation

#### Value Tensor Shape Mismatch

- Fixed shape mismatch in training loss computation
- Added `value.unsqueeze(1)` for proper [batch_size, 1] shape

#### Curriculum Manager API

- Fixed `record_game(win=...)` to `record_game(won=...)` parameter name

### Changed

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
