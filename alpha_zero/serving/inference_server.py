"""FastAPI inference server for Civ6 AlphaZero AI moves.

This server loads a trained Civ6TransformerNetwork model from GCS and
serves AI move requests via HTTP. Each request receives a game state
(spatial features, vector features, action mask) and returns the best
action selected by GumbelMCTS search.

Environment Variables:
    CIV6AZ_MODEL_VERSION: Model version to load (e.g., "v268").
    CIV6AZ_GCS_BUCKET: GCS bucket containing model weights. **No default** -- a
        bucket-name default is a takeover vector (whoever registers the name
        serves the model), so startup fails closed when it is unset.
    CIV6AZ_ALLOW_RANDOM_WEIGHTS: Set to ``1`` to serve a randomly initialised
        network when no bucket is configured or the download fails. Off by
        default; intended only for local smoke tests, never for a deployment.
    CIV6AZ_MCTS_SIMS: Number of MCTS simulations per move (default: 16).
    CIV6AZ_BOARD_SIZE: Board size (default: 8).
    CIV6AZ_ACTION_SPACE_SIZE: Action space size (default: 376).
        Also accepts CIV6AZ_ACTION_SPACE for backwards compatibility.
    CIV6AZ_DEVICE: PyTorch device (default: "cuda" if available).
    INFERENCE_PORT: Server port (default: 8080).

Endpoints:
    GET  /health     - Liveness/readiness probe
    POST /v1/move    - Compute AI move for a game state
    GET  /v1/model   - Return current model version metadata
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_VERSION = os.environ.get("CIV6AZ_MODEL_VERSION", "v268")
GCS_BUCKET: str | None = os.environ.get("CIV6AZ_GCS_BUCKET") or None
ALLOW_RANDOM_WEIGHTS: bool = os.environ.get("CIV6AZ_ALLOW_RANDOM_WEIGHTS", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
_NO_BUCKET_MESSAGE = (
    "CIV6AZ_GCS_BUCKET is not set, so there is nowhere to load model weights from. "
    "Refusing to start: serving a randomly initialised network would look healthy and "
    "play nonsense. Set CIV6AZ_GCS_BUCKET to the bucket holding "
    "models/civ6az/<version>/model.pt, or set CIV6AZ_ALLOW_RANDOM_WEIGHTS=1 for a "
    "local smoke test only."
)
MCTS_SIMS = int(os.environ.get("CIV6AZ_MCTS_SIMS", "16"))
BOARD_SIZE = int(os.environ.get("CIV6AZ_BOARD_SIZE", "8"))
ACTION_SPACE = int(
    os.environ.get(
        "CIV6AZ_ACTION_SPACE_SIZE",
        os.environ.get("CIV6AZ_ACTION_SPACE", "376"),
    )
)
DEVICE = os.environ.get("CIV6AZ_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PORT = int(os.environ.get("INFERENCE_PORT", "8080"))

# ---------------------------------------------------------------------------
# Global state (loaded at startup)
# ---------------------------------------------------------------------------

_model: Any = None
_model_version: str = ""
_device: torch.device = torch.device(DEVICE)
_startup_time: float = 0.0


def _load_model() -> None:
    """Load model weights from GCS into GPU memory."""
    global _model, _model_version, _startup_time

    from alpha_zero.cloud.gcp.config import NetworkArchitectureConfig
    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

    start = time.monotonic()

    arch = NetworkArchitectureConfig.from_env()
    config = TransformerConfig(
        spatial_channels=arch.spatial_channels,
        spatial_size=arch.board_size,
        vector_dim=arch.vector_dim,
        action_space_size=arch.action_space_size,
        embed_dim=arch.embed_dim,
        num_layers=arch.num_layers,
        num_heads=arch.num_heads,
        policy_hidden_dim=arch.policy_hidden_dim,
        value_head_dim=arch.value_head_dim,
    )

    network = Civ6TransformerNetwork(config)

    # Download model from GCS. Fail closed: random weights only on explicit opt-in.
    model_path = f"models/civ6az/{MODEL_VERSION}/model.pt"
    cache_dir = os.environ.get("CIV6AZ_MODEL_CACHE_DIR", "/tmp")  # noqa: S108
    local_path: str | None = None

    if GCS_BUCKET is None:
        if not ALLOW_RANDOM_WEIGHTS:
            raise RuntimeError(_NO_BUCKET_MESSAGE)
        logger.warning(
            "CIV6AZ_GCS_BUCKET is not set; serving RANDOM weights because "
            "CIV6AZ_ALLOW_RANDOM_WEIGHTS is set. Never deploy this configuration."
        )
    else:
        candidate = f"{cache_dir}/{MODEL_VERSION}_model.pt"
        try:
            from google.cloud import storage as gcs_storage

            client = gcs_storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(model_path)
            blob.download_to_filename(candidate)
            logger.info("Downloaded model from gs://%s/%s", GCS_BUCKET, model_path)
            local_path = candidate
        except Exception as exc:
            if not ALLOW_RANDOM_WEIGHTS:
                raise RuntimeError(
                    f"Failed to download gs://{GCS_BUCKET}/{model_path}: {exc!r}. Refusing "
                    "to serve random weights; set CIV6AZ_ALLOW_RANDOM_WEIGHTS=1 only for a "
                    "local smoke test."
                ) from exc
            logger.warning(
                "Failed to download model from GCS; serving RANDOM weights because "
                "CIV6AZ_ALLOW_RANDOM_WEIGHTS is set.",
                exc_info=True,
            )

    if local_path is not None:
        state_dict = torch.load(local_path, map_location=_device, weights_only=True)
        network.load_state_dict(state_dict)
        logger.info("Loaded model state dict from %s", local_path)

    network = network.to(_device)

    _model = network
    _model_version = MODEL_VERSION
    _startup_time = time.monotonic() - start
    logger.info(
        "Model %s loaded on %s in %.2fs. Action space=%d, board=%d.",
        _model_version,
        _device,
        _startup_time,
        ACTION_SPACE,
        BOARD_SIZE,
    )


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------


class MoveRequest(BaseModel):
    """Request body for the /v1/move endpoint."""

    spatial: list[list[list[float]]] = Field(
        ...,
        description=(
            "Spatial observation tensor of shape [channels, height, width]. "
            "Channels=13, height=width=board_size."
        ),
    )
    vector: list[float] = Field(
        ...,
        description="Vector observation of length 64.",
    )
    action_mask: list[float] = Field(
        ...,
        description="Binary action mask of length 376. 1=legal, 0=illegal.",
    )
    mcts_simulations: int | None = Field(
        default=None,
        description="Override MCTS simulations for this request (default: server config).",
    )
    temperature: float = Field(
        default=0.0,
        description="Action selection temperature. 0=greedy, 1=proportional to visit counts.",
    )


class MoveResponse(BaseModel):
    """Response body for the /v1/move endpoint."""

    action: int = Field(..., description="Selected action index (0-375).")
    action_probabilities: list[float] = Field(
        ..., description="MCTS visit count distribution over all actions."
    )
    value: float = Field(..., description="Root node value estimate from the neural network.")
    search_time_ms: float = Field(..., description="MCTS search time in milliseconds.")
    model_version: str = Field(..., description="Model version used for this inference.")


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    model_loaded: bool
    model_version: str
    device: str
    gpu_available: bool
    startup_time_s: float


class ModelInfoResponse(BaseModel):
    """Response body for /v1/model endpoint."""

    model_version: str
    board_size: int
    action_space_size: int
    mcts_simulations: int
    device: str


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model on startup."""
    _load_model()
    yield


app = FastAPI(
    title="Civ6 AlphaZero Inference Server",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint for Cloud Run probes."""
    return HealthResponse(
        status="ok" if _model is not None else "loading",
        model_loaded=_model is not None,
        model_version=_model_version,
        device=str(_device),
        gpu_available=torch.cuda.is_available(),
        startup_time_s=round(_startup_time, 2),
    )


@app.get("/v1/model", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return current model configuration."""
    return ModelInfoResponse(
        model_version=_model_version,
        board_size=BOARD_SIZE,
        action_space_size=ACTION_SPACE,
        mcts_simulations=MCTS_SIMS,
        device=str(_device),
    )


@app.post("/v1/move", response_model=MoveResponse)
async def compute_move(request: MoveRequest) -> MoveResponse:
    """Compute an AI move using the neural network policy.

    Accepts the current game state and returns the best action along
    with the full policy distribution and value estimate.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        # Convert inputs to tensors
        spatial_np = np.array(request.spatial, dtype=np.float32)
        vector_np = np.array(request.vector, dtype=np.float32)
        mask_np = np.array(request.action_mask, dtype=np.float32)

        # Validate shapes
        expected_spatial = (13, BOARD_SIZE, BOARD_SIZE)
        if spatial_np.shape != expected_spatial:
            raise HTTPException(
                status_code=422,
                detail=f"spatial shape {spatial_np.shape} != expected {expected_spatial}",
            )
        if vector_np.shape != (64,):
            raise HTTPException(
                status_code=422,
                detail=f"vector shape {vector_np.shape} != expected (64,)",
            )
        if mask_np.shape != (ACTION_SPACE,):
            raise HTTPException(
                status_code=422,
                detail=f"action_mask shape {mask_np.shape} != expected ({ACTION_SPACE},)",
            )

        # Run neural network forward pass
        start = time.monotonic()

        spatial_t = torch.from_numpy(spatial_np).unsqueeze(0).to(_device)
        vector_t = torch.from_numpy(vector_np).unsqueeze(0).to(_device)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(_device)

        with torch.no_grad():
            policy_logits, value = _model(spatial_t, vector_t, mask_t)

        policy_logits = policy_logits.squeeze(0)
        value_scalar = value.squeeze().item()

        # Apply action mask to logits
        mask_bool = mask_np > 0.5
        masked_logits = policy_logits.cpu().numpy().copy()
        masked_logits[~mask_bool] = -1e9

        # Softmax with temperature
        temp = max(request.temperature, 1e-8)
        logits_temp = masked_logits / temp if request.temperature > 0 else masked_logits
        exp_logits = np.exp(logits_temp - np.max(logits_temp))
        exp_logits[~mask_bool] = 0.0
        probs = exp_logits / (exp_logits.sum() + 1e-8)

        # Select action
        if request.temperature <= 0:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(ACTION_SPACE, p=probs))

        elapsed_ms = (time.monotonic() - start) * 1000

        return MoveResponse(
            action=action,
            action_probabilities=probs.tolist(),
            value=round(value_scalar, 4),
            search_time_ms=round(elapsed_ms, 2),
            model_version=_model_version,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing move")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        "alpha_zero.serving.inference_server:app",
        host="0.0.0.0",  # noqa: S104
        port=PORT,
        workers=1,  # Single worker: model is loaded per-process
        log_level="info",
    )
