//! Worker coordination reducers.
//!
//! Handles worker heartbeats, training state management,
//! stop signals, checkpoints, and config snapshots.

use spacetimedb::ReducerContext;

use crate::tables::{Checkpoint, TrainingConfig, TrainingState, WorkerHeartbeat};

/// Worker heartbeat (upsert).
///
/// Registers a new worker or updates an existing worker's status.
/// Used for health monitoring and disconnect detection.
///
/// # Arguments
/// * `worker_id` - Unique worker identifier
/// * `worker_type` - Worker type ("selfplay", "trainer", "evaluator")
/// * `status` - Current status ("running", "idle", "error")
/// * `games_played` - Total games completed by this worker
#[spacetimedb::reducer]
pub fn worker_heartbeat(
    ctx: &ReducerContext,
    worker_id: String,
    worker_type: String,
    status: String,
    games_played: u32,
) {
    // Delete existing entry (upsert pattern)
    ctx.db.worker_heartbeats().worker_id().delete(&worker_id);

    ctx.db.worker_heartbeats().insert(WorkerHeartbeat {
        worker_id: worker_id.clone(),
        worker_type,
        status,
        games_played,
        last_heartbeat: spacetimedb::Timestamp::now(),
        identity: ctx.sender,
    });

    log::debug!("Heartbeat from worker {}", worker_id);
}

/// Update global training state.
///
/// Updates the singleton training state row with current progress.
///
/// # Arguments
/// * `is_training` - Whether training is currently active
/// * `current_iteration` - Current training iteration
/// * `total_games_played` - Total games across all workers
#[spacetimedb::reducer]
pub fn update_training_state(
    ctx: &ReducerContext,
    is_training: bool,
    current_iteration: u32,
    total_games_played: u64,
) {
    if let Some(state) = ctx.db.training_state().singleton_key().find(0) {
        ctx.db.training_state().singleton_key().delete(&0);
        ctx.db.training_state().insert(TrainingState {
            singleton_key: 0,
            is_training,
            current_iteration,
            total_games_played,
            updated_at: spacetimedb::Timestamp::now(),
            ..state
        });
    }

    log::info!(
        "Training state updated: training={}, iteration={}, games={}",
        is_training, current_iteration, total_games_played
    );
}

/// Request training to stop.
///
/// Sets `should_stop = true` in the training state singleton.
/// All workers should check this flag periodically and gracefully stop.
#[spacetimedb::reducer]
pub fn request_stop(ctx: &ReducerContext) {
    if let Some(state) = ctx.db.training_state().singleton_key().find(0) {
        ctx.db.training_state().singleton_key().delete(&0);
        ctx.db.training_state().insert(TrainingState {
            singleton_key: 0,
            should_stop: true,
            updated_at: spacetimedb::Timestamp::now(),
            ..state
        });
    }

    log::info!("Training stop requested by {:?}", ctx.sender);
}

/// Save a full training checkpoint.
///
/// Stores the complete training state for resumption.
///
/// # Arguments
/// * `iteration` - Training iteration (unique)
/// * `data_blob` - Serialized checkpoint data
/// * `model_version` - Model version at time of checkpoint
#[spacetimedb::reducer]
pub fn save_checkpoint(
    ctx: &ReducerContext,
    iteration: u32,
    data_blob: Vec<u8>,
    model_version: u32,
) {
    ctx.db.checkpoints().insert(Checkpoint {
        iteration,
        data_blob,
        model_version,
        created_at: spacetimedb::Timestamp::now(),
    });

    log::info!(
        "Checkpoint saved: iteration={}, model_version={}",
        iteration, model_version
    );
}

/// Save a training configuration snapshot.
///
/// Stores the configuration used for a given iteration,
/// enabling reproducibility and configuration auditing.
///
/// # Arguments
/// * `iteration` - Training iteration (unique)
/// * `config_json` - JSON-serialized configuration
#[spacetimedb::reducer]
pub fn save_config(ctx: &ReducerContext, iteration: u32, config_json: String) {
    ctx.db.training_configs().insert(TrainingConfig {
        iteration,
        config_json,
        created_at: spacetimedb::Timestamp::now(),
    });

    log::info!("Config saved for iteration {}", iteration);
}
