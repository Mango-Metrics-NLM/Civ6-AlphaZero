"""
FastAPI Server for Civ6 AlphaZero Inference Service.

Production-ready API with:
- RESTful endpoints for inference
- Health checks and metrics
- Model hot-swapping
- Async request handling
- CORS support
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.storage import GCSBackend, GCSModelStorage
from alpha_zero.inference.api_schemas import (
    BatchInferenceRequest,
    BatchInferenceResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    MetricsResponse,
    ModelSwapRequest,
    ModelSwapResponse,
)
from alpha_zero.inference.inference_engine import InferenceConfig, InferenceEngine

logger = logging.getLogger(__name__)

# Global inference engine (initialized on startup)
engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup/shutdown."""
    global engine

    # Startup: Initialize engine and load model
    logger.info("Starting Civ6 AlphaZero Inference Service...")
    try:
        # Create GCS storage
        gcp_config = GCPConfig.from_env()
        backend = GCSBackend(gcp_config.project_id)
        storage = GCSModelStorage(gcp_config, backend=backend)

        # Create inference engine
        inference_config = InferenceConfig()
        engine = InferenceEngine(storage, inference_config)

        # Load latest champion model
        await engine.load_model()
        logger.info("Inference service ready")

    except Exception as e:
        logger.error(f"Failed to initialize inference service: {e}")
        raise

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down inference service...")
    if engine:
        engine.model_cache.clear()


# Create FastAPI app
app = FastAPI(
    title="Civ6 AlphaZero Inference API",
    description="Production ML inference service for Civ6 AlphaZero models",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["general"])
async def root() -> dict[str, str]:
    """Root endpoint with service information."""
    return {
        "service": "Civ6 AlphaZero Inference API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse with service health status.
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )

    health_data = engine.get_health()
    return HealthResponse(**health_data)


@app.get("/metrics", response_model=MetricsResponse, tags=["monitoring"])
async def metrics() -> MetricsResponse:
    """Get service metrics.

    Returns:
        MetricsResponse with performance metrics.
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )

    metrics_data = engine.get_metrics()
    return MetricsResponse(
        total_requests=metrics_data["total_requests"],
        requests_per_second=metrics_data["requests_per_second"],
        average_latency_ms=metrics_data["average_latency_ms"],
        p95_latency_ms=metrics_data["p95_latency_ms"],
        p99_latency_ms=metrics_data["p99_latency_ms"],
        active_sessions=0,  # TODO: Implement session tracking
        cache_hit_rate=0.0,  # TODO: Implement cache hit tracking
    )


@app.post("/v1/infer", response_model=InferenceResponse, tags=["inference"])
async def infer(request: InferenceRequest) -> InferenceResponse:
    """Run inference for a single game state.

    Args:
        request: Inference request with game state and MCTS parameters.

    Returns:
        InferenceResponse with action distribution and value estimate.

    Raises:
        HTTPException: If inference fails.
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )

    start_time = time.time()

    try:
        # Run inference
        action_distribution = await engine.infer(request)

        # Compute timing
        total_time = (time.time() - start_time) * 1000

        return InferenceResponse(
            action_distribution=action_distribution,
            model_version=engine.current_version or 0,
            inference_time_ms=total_time,
            mcts_time_ms=total_time,  # TODO: Separate MCTS timing
            session_id=request.session_id,
            metadata={
                "mcts_type": request.mcts_type.value,
                "mcts_simulations": request.mcts_simulations,
                "temperature": request.temperature,
            },
        )

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {e!s}",
        ) from e


@app.post("/v1/batch_infer", response_model=BatchInferenceResponse, tags=["inference"])
async def batch_infer(request: BatchInferenceRequest) -> BatchInferenceResponse:
    """Run batched inference for multiple game states.

    Args:
        request: Batch inference request with multiple game states.

    Returns:
        BatchInferenceResponse with results for all requests.

    Raises:
        HTTPException: If batch inference fails.
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )

    start_time = time.time()

    try:
        # Run batched inference
        action_distributions = await engine.batch_infer(
            request.requests, max_batch_size=request.max_batch_size
        )

        # Build responses
        responses = []
        for i, action_dist in enumerate(action_distributions):
            responses.append(
                InferenceResponse(
                    action_distribution=action_dist,
                    model_version=engine.current_version or 0,
                    inference_time_ms=0.0,  # Individual timing not tracked in batch
                    mcts_time_ms=0.0,
                    session_id=request.requests[i].session_id,
                    metadata={"batch_index": i},
                )
            )

        total_time = (time.time() - start_time) * 1000

        return BatchInferenceResponse(
            responses=responses,
            total_time_ms=total_time,
            batch_size=len(request.requests),
        )

    except Exception as e:
        logger.error(f"Batch inference failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {e!s}",
        ) from e


@app.post("/v1/models/swap", response_model=ModelSwapResponse, tags=["model-management"])
async def swap_model(request: ModelSwapRequest) -> ModelSwapResponse:
    """Load a different model version.

    Args:
        request: Model swap request with version number.

    Returns:
        ModelSwapResponse with swap status.

    Raises:
        HTTPException: If model swap fails.
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )

    start_time = time.time()
    previous_version = engine.current_version

    try:
        # Load new model
        new_version = await engine.load_model(request.model_version, force=request.force)

        load_time = (time.time() - start_time) * 1000

        return ModelSwapResponse(
            success=True,
            previous_version=previous_version,
            new_version=new_version,
            load_time_ms=load_time,
            message=f"Successfully loaded model v{new_version}",
        )

    except Exception as e:
        logger.error(f"Model swap failed: {e}", exc_info=True)
        return ModelSwapResponse(
            success=False,
            previous_version=previous_version,
            new_version=request.model_version,
            load_time_ms=(time.time() - start_time) * 1000,
            message=f"Failed to load model: {e!s}",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",  # noqa: S104
        port=8000,
        log_level="info",
        access_log=True,
    )
