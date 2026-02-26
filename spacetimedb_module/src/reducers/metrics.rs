//! Metrics and evaluation reducers.
//!
//! Handles recording of training metrics and evaluation results.

use spacetimedb::ReducerContext;

use crate::tables::{EvaluationResult, TrainingMetrics};

/// Record training metrics for a step.
///
/// Stores loss values and learning rate for real-time monitoring.
/// Subscribers receive immediate notifications of new metrics.
///
/// # Arguments
/// * `step` - Training step number (unique)
/// * `total_loss` - Combined total loss
/// * `policy_loss` - Policy head loss
/// * `value_loss` - Value head loss
/// * `learning_rate` - Current learning rate
#[spacetimedb::reducer]
pub fn record_training_metrics(
    ctx: &ReducerContext,
    step: u64,
    total_loss: f64,
    policy_loss: f64,
    value_loss: f64,
    learning_rate: f64,
) {
    ctx.db.training_metrics().insert(TrainingMetrics {
        step,
        total_loss,
        policy_loss,
        value_loss,
        learning_rate,
        recorded_at: spacetimedb::Timestamp::now(),
    });

    log::debug!(
        "Metrics recorded: step={}, loss={:.4}",
        step, total_loss
    );
}

/// Record evaluation results (candidate vs champion).
///
/// Stores the outcome of a model evaluation round.
///
/// # Arguments
/// * `evaluation_id` - Unique evaluation identifier
/// * `iteration` - Training iteration of the candidate
/// * `win_rate` - Candidate win rate (0.0 to 1.0)
/// * `games_played` - Number of evaluation games
/// * `promoted` - Whether the candidate was promoted to champion
#[spacetimedb::reducer]
pub fn record_evaluation(
    ctx: &ReducerContext,
    evaluation_id: String,
    iteration: u32,
    win_rate: f64,
    games_played: u32,
    promoted: bool,
) {
    ctx.db.evaluation_results().insert(EvaluationResult {
        evaluation_id: evaluation_id.clone(),
        iteration,
        win_rate,
        games_played,
        promoted,
        completed_at: spacetimedb::Timestamp::now(),
    });

    log::info!(
        "Evaluation recorded: {} (iteration={}, win_rate={:.2}%, promoted={})",
        evaluation_id, iteration, win_rate * 100.0, promoted
    );
}
