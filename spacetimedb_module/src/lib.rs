//! SpacetimeDB module for Civ6 AlphaZero distributed training.
//!
//! Provides tables and reducers for:
//! - Model weight storage and version tracking
//! - Game trajectory persistence
//! - Real-time training metrics streaming
//! - Worker coordination and heartbeats
//! - Evaluation results and champion promotion
//!
//! # Architecture
//!
//! Actors subscribe to `model_version_info` (tiny singleton row) for
//! version change notifications. They then explicitly fetch weights from
//! `model_weights` (large blobs) only when needed. This avoids pushing
//! 100MB+ weight blobs to every subscriber on every update.

pub mod reducers;
pub mod tables;

use spacetimedb::ReducerContext;

/// Initialize the module with singleton rows for version tracking and training state.
///
/// Called once when the module is first deployed.
#[spacetimedb::reducer(init)]
pub fn init(ctx: &ReducerContext) {
    // Create singleton version tracking row
    ctx.db.model_version_info().insert(tables::ModelVersionInfo {
        singleton_key: 0,
        current_version: 0,
        champion_version: 0,
        updated_at: spacetimedb::Timestamp::now(),
    });

    // Create singleton training state row
    ctx.db.training_state().insert(tables::TrainingState {
        singleton_key: 0,
        is_training: false,
        should_stop: false,
        current_iteration: 0,
        total_games_played: 0,
        total_examples: 0,
        updated_at: spacetimedb::Timestamp::now(),
    });

    log::info!("Civ6 AlphaZero module initialized");
}

/// Handle client disconnection by marking their worker heartbeat as disconnected.
///
/// Each identity maps to at most one worker, so we break after the first match.
#[spacetimedb::reducer(client_disconnected)]
pub fn on_client_disconnected(ctx: &ReducerContext) {
    let identity = ctx.sender;
    for heartbeat in ctx.db.worker_heartbeats().iter() {
        if heartbeat.identity == identity {
            ctx.db.worker_heartbeats().worker_id().delete(&heartbeat.worker_id);
            ctx.db.worker_heartbeats().insert(tables::WorkerHeartbeat {
                status: "disconnected".to_string(),
                last_heartbeat: spacetimedb::Timestamp::now(),
                ..heartbeat
            });
            log::info!("Worker {} disconnected", heartbeat.worker_id);
            break;
        }
    }
}
