//! Table definitions for the Civ6 AlphaZero SpacetimeDB module.
//!
//! Design decisions:
//! - `model_version_info` is separated from `model_weights` so actors
//!   subscribe to the tiny version row (not multi-MB weight blobs).
//! - Singleton tables use `singleton_key = 0` as the unique column.
//! - All timestamps use `spacetimedb::Timestamp` for consistent ordering.
//! - Blob columns (`Vec<u8>`) store serialized PyTorch state dicts and
//!   game trajectories with optional LZMA compression.

use spacetimedb::SpacetimeType;

/// Full model weight blob storage.
///
/// Contains the serialized PyTorch state dict. Actors fetch from this
/// table explicitly after receiving a version notification.
#[spacetimedb::table(name = model_weights, public)]
pub struct ModelWeights {
    /// Model version number (unique, monotonically increasing).
    #[unique]
    pub version: u32,
    /// Training iteration when this version was created.
    pub iteration: u32,
    /// Serialized model weights (PyTorch state dict, optionally LZMA compressed).
    pub weights_blob: Vec<u8>,
    /// JSON-serialized metadata (loss, metrics, config hash, etc.).
    pub metadata_json: String,
    /// Whether this version is the current champion.
    pub is_champion: bool,
    /// When this version was created.
    pub created_at: spacetimedb::Timestamp,
}

/// Lightweight model version tracker (singleton).
///
/// Actors subscribe to this table for real-time version change notifications.
/// Only contains version numbers, not weight blobs.
#[spacetimedb::table(name = model_version_info, public)]
pub struct ModelVersionInfo {
    /// Singleton key (always 0).
    #[unique]
    pub singleton_key: u32,
    /// Current (latest) model version.
    pub current_version: u32,
    /// Champion model version (best-performing).
    pub champion_version: u32,
    /// Last update timestamp.
    pub updated_at: spacetimedb::Timestamp,
}

/// Game trajectory storage.
///
/// Stores serialized game trajectories from self-play workers.
/// Each trajectory contains training examples (state, policy, value).
#[spacetimedb::table(name = trajectories, public)]
pub struct Trajectory {
    /// Unique trajectory identifier (format: "{iteration}_{worker_id}_{game_id}").
    #[unique]
    pub trajectory_id: String,
    /// Training iteration.
    pub iteration: u32,
    /// Worker that generated this trajectory.
    pub worker_id: u32,
    /// Game identifier.
    pub game_id: String,
    /// Serialized trajectory data (GameTrajectory.to_bytes()).
    pub data_blob: Vec<u8>,
    /// Number of training examples in this trajectory.
    pub num_examples: u32,
    /// When this trajectory was submitted.
    pub created_at: spacetimedb::Timestamp,
}

/// Real-time training metrics stream.
///
/// Stores per-step training metrics for monitoring dashboards.
#[spacetimedb::table(name = training_metrics, public)]
pub struct TrainingMetrics {
    /// Training step number (unique).
    #[unique]
    pub step: u64,
    /// Total combined loss.
    pub total_loss: f64,
    /// Policy head loss.
    pub policy_loss: f64,
    /// Value head loss.
    pub value_loss: f64,
    /// Current learning rate.
    pub learning_rate: f64,
    /// When this metric was recorded.
    pub recorded_at: spacetimedb::Timestamp,
}

/// Evaluation results storage.
///
/// Stores results of candidate vs champion model evaluations.
#[spacetimedb::table(name = evaluation_results, public)]
pub struct EvaluationResult {
    /// Unique evaluation identifier.
    #[unique]
    pub evaluation_id: String,
    /// Training iteration of the candidate model.
    pub iteration: u32,
    /// Candidate win rate (0.0 to 1.0).
    pub win_rate: f64,
    /// Total games played in evaluation.
    pub games_played: u32,
    /// Whether the candidate was promoted to champion.
    pub promoted: bool,
    /// When this evaluation completed.
    pub completed_at: spacetimedb::Timestamp,
}

/// Worker heartbeat and status tracking.
///
/// Used for monitoring worker health and detecting disconnections.
#[spacetimedb::table(name = worker_heartbeats, public)]
pub struct WorkerHeartbeat {
    /// Unique worker identifier.
    #[unique]
    pub worker_id: String,
    /// Worker type ("selfplay", "trainer", "evaluator").
    pub worker_type: String,
    /// Current status ("running", "idle", "disconnected", "error").
    pub status: String,
    /// Total games completed by this worker.
    pub games_played: u32,
    /// Last heartbeat timestamp.
    pub last_heartbeat: spacetimedb::Timestamp,
    /// Client identity (for disconnect detection).
    pub identity: spacetimedb::Identity,
}

/// Global training state (singleton).
///
/// Coordinates training lifecycle across all workers.
#[spacetimedb::table(name = training_state, public)]
pub struct TrainingState {
    /// Singleton key (always 0).
    #[unique]
    pub singleton_key: u32,
    /// Whether training is currently active.
    pub is_training: bool,
    /// Whether a stop has been requested.
    pub should_stop: bool,
    /// Current training iteration.
    pub current_iteration: u32,
    /// Total games played across all workers.
    pub total_games_played: u64,
    /// Total training examples generated.
    pub total_examples: u64,
    /// Last update timestamp.
    pub updated_at: spacetimedb::Timestamp,
}

/// Full training checkpoint storage.
///
/// Stores complete training state (model + optimizer + scheduler + metadata).
#[spacetimedb::table(name = checkpoints, public)]
pub struct Checkpoint {
    /// Training iteration (unique).
    #[unique]
    pub iteration: u32,
    /// Serialized checkpoint data (torch.save format, LZMA compressed).
    pub data_blob: Vec<u8>,
    /// Model version at time of checkpoint.
    pub model_version: u32,
    /// When this checkpoint was saved.
    pub created_at: spacetimedb::Timestamp,
}

/// Training configuration snapshots.
///
/// Stores serialized training configuration for each iteration,
/// enabling reproducibility and configuration drift detection.
#[spacetimedb::table(name = training_configs, public)]
pub struct TrainingConfig {
    /// Training iteration (unique).
    #[unique]
    pub iteration: u32,
    /// JSON-serialized configuration.
    pub config_json: String,
    /// When this config was saved.
    pub created_at: spacetimedb::Timestamp,
}
