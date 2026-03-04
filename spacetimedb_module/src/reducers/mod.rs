//! Reducer module declarations.
//!
//! Organizes reducers by domain:
//! - `models`: Weight publish, champion promotion
//! - `trajectories`: Trajectory submission
//! - `metrics`: Training metrics and evaluation results
//! - `coordination`: Worker heartbeats, training state, stop signal

pub mod coordination;
pub mod metrics;
pub mod models;
pub mod trajectories;
