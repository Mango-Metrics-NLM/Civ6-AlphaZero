//! Trajectory submission reducer.
//!
//! Handles game trajectory submissions from self-play workers.
//! Updates the global training state with trajectory counts.

use spacetimedb::ReducerContext;

use crate::tables::{Trajectory, TrainingState};

/// Submit a game trajectory from a self-play worker.
///
/// Stores the trajectory data and updates the global training state
/// with the new game and example counts.
///
/// # Arguments
/// * `trajectory_id` - Unique identifier (format: "{iteration}_{worker_id}_{game_id}")
/// * `iteration` - Training iteration
/// * `worker_id` - Worker that generated this trajectory
/// * `game_id` - Game identifier
/// * `data_blob` - Serialized trajectory data
/// * `num_examples` - Number of training examples in the trajectory
#[spacetimedb::reducer]
pub fn submit_trajectory(
    ctx: &ReducerContext,
    trajectory_id: String,
    iteration: u32,
    worker_id: u32,
    game_id: String,
    data_blob: Vec<u8>,
    num_examples: u32,
) {
    // Store the trajectory
    ctx.db.trajectories().insert(Trajectory {
        trajectory_id: trajectory_id.clone(),
        iteration,
        worker_id,
        game_id,
        data_blob,
        num_examples,
        created_at: spacetimedb::Timestamp::now(),
    });

    // Update global training state counts
    if let Some(state) = ctx.db.training_state().singleton_key().find(0) {
        ctx.db.training_state().singleton_key().delete(&0);
        ctx.db.training_state().insert(TrainingState {
            singleton_key: 0,
            total_games_played: state.total_games_played + 1,
            total_examples: state.total_examples + num_examples as u64,
            updated_at: spacetimedb::Timestamp::now(),
            ..state
        });
    }

    log::debug!(
        "Trajectory submitted: {} (iteration={}, examples={})",
        trajectory_id, iteration, num_examples
    );
}
