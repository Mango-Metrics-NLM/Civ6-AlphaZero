"""
Load Testing Script for Inference Service.

Simulates realistic game traffic to measure:
- Throughput (requests/second)
- Latency (P50, P95, P99)
- Error rates
- Resource utilization

Usage:
    python scripts/load_test_inference.py \
        --url https://civ6az-inference.run.app \
        --concurrent-games 50 \
        --duration-seconds 300
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass

import httpx
import numpy as np

from alpha_zero.inference.api_schemas import GameState, InferenceRequest, MCTSType

logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Load test configuration."""

    base_url: str
    concurrent_games: int = 10
    duration_seconds: int = 60
    mcts_simulations: int = 25
    board_size: int = 8
    timeout_seconds: float = 30.0


@dataclass
class LoadTestResults:
    """Load test results."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies: list[float] = None
    errors: list[str] = None
    start_time: float = 0.0
    end_time: float = 0.0

    def __post_init__(self):
        if self.latencies is None:
            self.latencies = []
        if self.errors is None:
            self.errors = []

    @property
    def duration_seconds(self) -> float:
        """Total test duration."""
        return self.end_time - self.start_time

    @property
    def requests_per_second(self) -> float:
        """Average RPS."""
        return self.successful_requests / self.duration_seconds if self.duration_seconds > 0 else 0

    @property
    def error_rate(self) -> float:
        """Error rate (0-1)."""
        return self.failed_requests / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def p50_latency_ms(self) -> float:
        """P50 latency."""
        return float(np.percentile(self.latencies, 50)) if self.latencies else 0.0

    @property
    def p95_latency_ms(self) -> float:
        """P95 latency."""
        return float(np.percentile(self.latencies, 95)) if self.latencies else 0.0

    @property
    def p99_latency_ms(self) -> float:
        """P99 latency."""
        return float(np.percentile(self.latencies, 99)) if self.latencies else 0.0

    def summary(self) -> str:
        """Generate summary report."""
        return f"""
Load Test Results
==================
Duration: {self.duration_seconds:.1f}s
Total Requests: {self.total_requests}
Successful: {self.successful_requests}
Failed: {self.failed_requests}
Error Rate: {self.error_rate * 100:.2f}%

Throughput
----------
Requests/Second: {self.requests_per_second:.2f}

Latency (ms)
------------
P50: {self.p50_latency_ms:.1f}
P95: {self.p95_latency_ms:.1f}
P99: {self.p99_latency_ms:.1f}
Mean: {np.mean(self.latencies):.1f} if self.latencies else 0.0
Min: {min(self.latencies):.1f} if self.latencies else 0.0
Max: {max(self.latencies):.1f} if self.latencies else 0.0
"""


def create_random_game_state(board_size: int = 8) -> GameState:
    """Create random game state for testing."""
    spatial = np.random.randn(13, board_size, board_size).flatten().tolist()
    vector = np.random.randn(64).tolist()
    action_mask = [1] * 376

    return GameState(
        spatial=spatial,
        vector=vector,
        action_mask=action_mask,
        board_size=board_size,
        turn_number=np.random.randint(0, 50),
    )


async def run_single_game(
    client: httpx.AsyncClient,
    game_id: int,
    config: LoadTestConfig,
    results: LoadTestResults,
    stop_event: asyncio.Event,
) -> None:
    """Simulate a single game with multiple turns.

    Args:
        client: HTTP client.
        game_id: Game identifier.
        config: Load test configuration.
        results: Shared results object.
        stop_event: Event to signal test end.
    """
    turn = 0

    while not stop_event.is_set():
        # Create game state
        game_state = create_random_game_state(config.board_size)

        # Create request
        request = InferenceRequest(
            game_state=game_state,
            mcts_simulations=config.mcts_simulations,
            mcts_type=MCTSType.GUMBEL,
            temperature=1.0 if turn < 10 else 0.0,
            session_id=f"game-{game_id}",
        )

        # Send request
        start = time.time()
        try:
            response = await client.post(
                "/v1/infer",
                json=request.model_dump(),
                timeout=config.timeout_seconds,
            )
            latency_ms = (time.time() - start) * 1000

            results.total_requests += 1

            if response.status_code == 200:
                results.successful_requests += 1
                results.latencies.append(latency_ms)

                if results.successful_requests % 10 == 0:
                    logger.info(
                        f"Game {game_id} turn {turn}: {latency_ms:.1f}ms "
                        f"({results.successful_requests} total)"
                    )
            else:
                results.failed_requests += 1
                error_msg = f"HTTP {response.status_code}: {response.text[:100]}"
                results.errors.append(error_msg)
                logger.warning(f"Game {game_id} failed: {error_msg}")

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            results.total_requests += 1
            results.failed_requests += 1
            error_msg = f"{type(e).__name__}: {str(e)[:100]}"
            results.errors.append(error_msg)
            logger.error(f"Game {game_id} error: {error_msg}")

        turn += 1

        # Small delay between turns
        await asyncio.sleep(0.1)


async def run_load_test(config: LoadTestConfig) -> LoadTestResults:
    """Run load test with concurrent games.

    Args:
        config: Load test configuration.

    Returns:
        LoadTestResults with metrics.
    """
    results = LoadTestResults()
    stop_event = asyncio.Event()

    logger.info(f"Starting load test: {config.concurrent_games} concurrent games")
    logger.info(f"Target URL: {config.base_url}")
    logger.info(f"Duration: {config.duration_seconds}s")

    # Check service health
    async with httpx.AsyncClient(base_url=config.base_url) as client:
        try:
            health_response = await client.get("/health", timeout=10.0)
            health_response.raise_for_status()
            health = health_response.json()
            logger.info(f"Service health: {health['status']} (v{health['model_version']})")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return results

    # Create HTTP client
    async with httpx.AsyncClient(base_url=config.base_url) as client:
        # Start concurrent game tasks
        tasks = [
            run_single_game(client, game_id, config, results, stop_event)
            for game_id in range(config.concurrent_games)
        ]

        results.start_time = time.time()

        # Run for specified duration
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=config.duration_seconds,
            )
        except TimeoutError:
            # Expected - stop all games
            stop_event.set()
            await asyncio.gather(*tasks, return_exceptions=True)

        results.end_time = time.time()

    logger.info("Load test complete")
    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Load test inference service")

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Base URL of inference service",
    )
    parser.add_argument(
        "--concurrent-games",
        type=int,
        default=10,
        help="Number of concurrent games to simulate",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=60,
        help="Test duration in seconds",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=25,
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=8,
        help="Board size (8 or 16)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds",
    )

    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Parse arguments
    args = parse_args()

    # Create config
    config = LoadTestConfig(
        base_url=args.url,
        concurrent_games=args.concurrent_games,
        duration_seconds=args.duration_seconds,
        mcts_simulations=args.mcts_simulations,
        board_size=args.board_size,
        timeout_seconds=args.timeout,
    )

    # Run load test
    results = await run_load_test(config)

    # Print summary
    print(results.summary())

    # Print sample errors if any
    if results.errors:
        print("\nSample Errors (first 5):")
        for error in results.errors[:5]:
            print(f"  - {error}")


if __name__ == "__main__":
    asyncio.run(main())
