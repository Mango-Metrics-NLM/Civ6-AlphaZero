# Changelog

All notable changes to the Civ6 AlphaZero project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
