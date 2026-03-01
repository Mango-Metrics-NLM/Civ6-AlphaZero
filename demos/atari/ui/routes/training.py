"""REST endpoints for training control and status."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from fastapi import APIRouter

from demos.atari.ui.config import AtariUIConfig

logger = logging.getLogger(__name__)


class TrainingState:
    """Tracks background training state."""

    def __init__(self) -> None:
        self.running = False
        self.stop_flag = threading.Event()
        self.iteration = 0
        self.metrics: dict[str, Any] = {}
        self.metrics_history: list[dict[str, Any]] = []
        self._task: asyncio.Task[None] | None = None


def create_training_router(config: AtariUIConfig) -> APIRouter:
    """Create the training REST API router."""
    router = APIRouter(tags=["training"])
    state = TrainingState()

    @router.post("/start")
    async def start_training(
        num_iterations: int = 10,
        episodes_per_iter: int = 2,
    ) -> dict[str, Any]:
        """Start training in the background."""
        if state.running:
            return {"status": "already_running", "iteration": state.iteration}

        state.stop_flag.clear()
        state.running = True
        state.iteration = 0
        state.metrics_history.clear()

        async def _run() -> None:
            try:
                await asyncio.to_thread(
                    _train_loop, config, state, num_iterations, episodes_per_iter
                )
            except Exception:
                logger.exception("Training loop error")
            finally:
                state.running = False

        state._task = asyncio.create_task(_run())
        return {"status": "started", "num_iterations": num_iterations}

    @router.post("/stop")
    async def stop_training() -> dict[str, str]:
        """Stop training gracefully."""
        if not state.running:
            return {"status": "not_running"}
        state.stop_flag.set()
        return {"status": "stopping"}

    @router.get("/status")
    async def training_status() -> dict[str, Any]:
        """Get current training status."""
        return {
            "running": state.running,
            "iteration": state.iteration,
            "metrics": state.metrics,
            "history": state.metrics_history[-50:],  # Last 50 entries
        }

    return router


def _train_loop(
    config: AtariUIConfig,
    state: TrainingState,
    num_iterations: int,
    episodes_per_iter: int,
) -> None:
    """Run training iterations synchronously (called in thread pool).

    Uses AtariTrainer.train() which manages env, network, MCTS, optimizer,
    and replay buffer internally. We patch the iteration count and capture
    the final summary.
    """
    from demos.atari.config import AtariConfig
    from demos.atari.trainer import AtariTrainer

    atari_config = AtariConfig.for_game(config.game)
    atari_config.training.num_iterations = num_iterations
    atari_config.training.episodes_per_iteration = episodes_per_iter
    atari_config.training.device = config.device

    trainer = AtariTrainer(atari_config)

    state.iteration = 0
    logger.info(
        "Starting training: %d iterations, %d episodes/iter, device=%s",
        num_iterations,
        episodes_per_iter,
        config.device,
    )

    try:
        summary = trainer.train()
    except Exception:
        logger.exception("Training failed")
        return

    # Extract final metrics from summary
    state.metrics = {
        "iteration": num_iterations,
        "policy_loss": float(summary.get("final_policy_loss", 0)),
        "value_loss": float(summary.get("final_value_loss", 0)),
        "total_loss": float(summary.get("final_total_loss", 0)),
        "num_examples": int(summary.get("total_examples", 0)),
    }
    state.metrics_history.append(state.metrics.copy())
    state.iteration = num_iterations

    logger.info("Training complete: %s", state.metrics)
