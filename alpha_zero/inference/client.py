"""
Python Client for Civ6 AlphaZero Inference Service.

Provides a simple interface for making inference requests.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import numpy as np

from alpha_zero.inference.api_schemas import (
    BatchInferenceRequest,
    GameState,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    MCTSType,
    MetricsResponse,
    ModelSwapRequest,
)

logger = logging.getLogger(__name__)


class InferenceClient:
    """Client for Civ6 AlphaZero Inference API.

    Example usage:
        client = InferenceClient("https://civ6az-inference.run.app")

        # Prepare game state
        spatial = np.random.randn(13, 8, 8).flatten().tolist()
        vector = np.random.randn(64).tolist()
        action_mask = [1] * 376

        game_state = GameState(
            spatial=spatial,
            vector=vector,
            action_mask=action_mask,
            turn_number=5
        )

        # Get inference
        response = client.infer(game_state)
        print(f"Recommended action: {response.action_distribution.action_id}")
        print(f"Value estimate: {response.action_distribution.value_estimate}")
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        api_key: str | None = None,
    ) -> None:
        """Initialize inference client.

        Args:
            base_url: Base URL of inference service (e.g., https://api.example.com).
            timeout: Request timeout in seconds.
            api_key: Optional API key for authentication.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Configure HTTP client
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

        self.async_client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

    def health(self) -> HealthResponse:
        """Check service health.

        Returns:
            HealthResponse with service status.

        Raises:
            httpx.HTTPError: If request fails.
        """
        response = self.client.get("/health")
        response.raise_for_status()
        return HealthResponse(**response.json())

    def metrics(self) -> MetricsResponse:
        """Get service metrics.

        Returns:
            MetricsResponse with performance metrics.

        Raises:
            httpx.HTTPError: If request fails.
        """
        response = self.client.get("/metrics")
        response.raise_for_status()
        return MetricsResponse(**response.json())

    def infer(
        self,
        game_state: GameState,
        mcts_simulations: int = 25,
        mcts_type: MCTSType = MCTSType.GUMBEL,
        temperature: float = 1.0,
        model_version: int | None = None,
        session_id: str | None = None,
    ) -> InferenceResponse:
        """Run inference for a single game state.

        Args:
            game_state: Current game state.
            mcts_simulations: Number of MCTS simulations.
            mcts_type: MCTS algorithm to use.
            temperature: Action selection temperature.
            model_version: Specific model version (None = latest).
            session_id: Optional session ID for tracking.

        Returns:
            InferenceResponse with action distribution.

        Raises:
            httpx.HTTPError: If request fails.
        """
        request = InferenceRequest(
            game_state=game_state,
            mcts_simulations=mcts_simulations,
            mcts_type=mcts_type,
            temperature=temperature,
            model_version=model_version,
            session_id=session_id,
        )

        response = self.client.post("/v1/infer", json=request.model_dump())
        response.raise_for_status()
        return InferenceResponse(**response.json())

    async def infer_async(
        self,
        game_state: GameState,
        mcts_simulations: int = 25,
        mcts_type: MCTSType = MCTSType.GUMBEL,
        temperature: float = 1.0,
        model_version: int | None = None,
        session_id: str | None = None,
    ) -> InferenceResponse:
        """Async version of infer().

        Args:
            game_state: Current game state.
            mcts_simulations: Number of MCTS simulations.
            mcts_type: MCTS algorithm to use.
            temperature: Action selection temperature.
            model_version: Specific model version (None = latest).
            session_id: Optional session ID for tracking.

        Returns:
            InferenceResponse with action distribution.

        Raises:
            httpx.HTTPError: If request fails.
        """
        request = InferenceRequest(
            game_state=game_state,
            mcts_simulations=mcts_simulations,
            mcts_type=mcts_type,
            temperature=temperature,
            model_version=model_version,
            session_id=session_id,
        )

        response = await self.async_client.post("/v1/infer", json=request.model_dump())
        response.raise_for_status()
        return InferenceResponse(**response.json())

    def batch_infer(
        self,
        requests: list[InferenceRequest],
        max_batch_size: int = 32,
    ) -> list[InferenceResponse]:
        """Run batched inference for multiple game states.

        Args:
            requests: List of inference requests.
            max_batch_size: Maximum batch size.

        Returns:
            List of InferenceResponse objects.

        Raises:
            httpx.HTTPError: If request fails.
        """
        batch_request = BatchInferenceRequest(
            requests=requests,
            max_batch_size=max_batch_size,
        )

        response = self.client.post("/v1/batch_infer", json=batch_request.model_dump())
        response.raise_for_status()

        data = response.json()
        return [InferenceResponse(**r) for r in data["responses"]]

    def swap_model(self, model_version: int, force: bool = False) -> dict[str, Any]:
        """Load a different model version.

        Args:
            model_version: Model version to load.
            force: Force reload even if already loaded.

        Returns:
            ModelSwapResponse as dict.

        Raises:
            httpx.HTTPError: If request fails.
        """
        request = ModelSwapRequest(model_version=model_version, force=force)

        response = self.client.post("/v1/models/swap", json=request.model_dump())
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def close(self) -> None:
        """Close HTTP clients."""
        self.client.close()

    async def aclose(self) -> None:
        """Close async HTTP client."""
        await self.async_client.aclose()

    def __enter__(self) -> InferenceClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    async def __aenter__(self) -> InferenceClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.aclose()


def create_random_game_state(board_size: int = 8) -> GameState:
    """Create a random game state for testing.

    Args:
        board_size: Size of the hex board.

    Returns:
        Random GameState.
    """
    spatial = np.random.randn(13, board_size, board_size).flatten().tolist()
    vector = np.random.randn(64).tolist()
    action_mask = [1] * 376  # All actions valid

    return GameState(
        spatial=spatial,
        vector=vector,
        action_mask=action_mask,
        board_size=board_size,
        turn_number=0,
    )
