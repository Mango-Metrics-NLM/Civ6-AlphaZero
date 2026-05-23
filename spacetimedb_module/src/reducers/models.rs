//! Model weight management reducers.
//!
//! Handles publishing new model weights and promoting models to champion status.
//! Updates the lightweight `model_version_info` singleton to notify subscribers.

use spacetimedb::ReducerContext;

use crate::tables::{ModelVersionInfo, ModelWeights};

/// Publish new model weights.
///
/// Inserts the weight blob and updates the version tracker singleton.
/// Version tracking is monotonically increasing — publishing an older
/// version will not regress `current_version`.
/// If `is_champion` is true, also updates the champion version.
///
/// # Arguments
/// * `version` - Model version number (must be unique)
/// * `iteration` - Training iteration when this version was created
/// * `weights_blob` - Serialized PyTorch state dict
/// * `is_champion` - Whether this is the new champion model
/// * `metadata_json` - JSON metadata (loss, metrics, etc.)
#[spacetimedb::reducer]
pub fn publish_model_weights(
    ctx: &ReducerContext,
    version: u32,
    iteration: u32,
    weights_blob: Vec<u8>,
    is_champion: bool,
    metadata_json: String,
) {
    // Insert weight blob
    ctx.db.model_weights().insert(ModelWeights {
        version,
        iteration,
        weights_blob,
        metadata_json,
        is_champion,
        created_at: spacetimedb::Timestamp::now(),
    });

    // Update version tracker singleton (monotonically increasing)
    if let Some(info) = ctx.db.model_version_info().singleton_key().find(0) {
        let new_current = if version > info.current_version {
            version
        } else {
            info.current_version
        };
        let champion_version = if is_champion {
            // Only advance champion version forward
            if version > info.champion_version { version } else { info.champion_version }
        } else {
            info.champion_version
        };

        ctx.db.model_version_info().singleton_key().delete(&0);
        ctx.db.model_version_info().insert(ModelVersionInfo {
            singleton_key: 0,
            current_version: new_current,
            champion_version,
            updated_at: spacetimedb::Timestamp::now(),
        });
    }

    log::info!(
        "Published model weights v{} (iteration={}, champion={})",
        version, iteration, is_champion
    );
}

/// Promote an existing model version to champion.
///
/// Updates the version tracker singleton with the new champion version
/// and clears the champion flag on any previous champion model.
/// The model weights must already exist in the `model_weights` table.
///
/// # Arguments
/// * `version` - Version to promote to champion
#[spacetimedb::reducer]
pub fn promote_to_champion(ctx: &ReducerContext, version: u32) {
    // Verify the version exists
    if ctx.db.model_weights().version().find(version).is_none() {
        log::error!("Cannot promote non-existent version {}", version);
        return;
    }

    // Clear champion flag on the previous champion (if any)
    if let Some(info) = ctx.db.model_version_info().singleton_key().find(0) {
        if info.champion_version > 0 && info.champion_version != version {
            if let Some(old_champ) = ctx.db.model_weights().version().find(info.champion_version) {
                if old_champ.is_champion {
                    ctx.db.model_weights().version().delete(&info.champion_version);
                    ctx.db.model_weights().insert(ModelWeights {
                        is_champion: false,
                        ..old_champ
                    });
                }
            }
        }
    }

    // Set champion flag on the new champion
    if let Some(weights) = ctx.db.model_weights().version().find(version) {
        ctx.db.model_weights().version().delete(&version);
        ctx.db.model_weights().insert(ModelWeights {
            is_champion: true,
            ..weights
        });
    }

    // Update version tracker
    if let Some(info) = ctx.db.model_version_info().singleton_key().find(0) {
        ctx.db.model_version_info().singleton_key().delete(&0);
        ctx.db.model_version_info().insert(ModelVersionInfo {
            singleton_key: 0,
            champion_version: version,
            ..info
        });
    }

    log::info!("Promoted version {} to champion", version);
}
