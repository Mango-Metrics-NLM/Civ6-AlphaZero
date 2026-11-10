"""
Inference Service Quick Start Example.

Demonstrates:
1. Service health check
2. Single inference request
3. Batch inference
4. Model version swapping
5. Performance monitoring

Usage:
    python examples/inference_quickstart.py --url https://<service-url>
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from alpha_zero.inference import (
    GameState,
    InferenceClient,
    InferenceRequest,
    MCTSType,
    create_random_game_state,
)


def example_health_check(client: InferenceClient) -> None:
    """Example 1: Check service health."""
    print("=" * 60)
    print("Example 1: Health Check")
    print("=" * 60)

    health = client.health()
    print(f"Status: {health.status}")
    print(f"Model Loaded: {health.model_loaded}")
    print(f"Model Version: {health.model_version}")
    print(f"GPU Available: {health.gpu_available}")
    print(f"Memory Usage: {health.memory_usage_mb:.1f} MB")
    print(f"Uptime: {health.uptime_seconds:.1f}s")
    print()


def example_single_inference(client: InferenceClient) -> None:
    """Example 2: Single inference request."""
    print("=" * 60)
    print("Example 2: Single Inference")
    print("=" * 60)

    # Create random game state
    game_state = create_random_game_state(board_size=8)

    # Run inference
    start = time.time()
    response = client.infer(
        game_state=game_state,
        mcts_simulations=25,
        mcts_type=MCTSType.GUMBEL,
        temperature=1.0,
        session_id="quickstart-game-1",
    )
    latency = (time.time() - start) * 1000

    # Display results
    action = response.action_distribution.action_id
    value = response.action_distribution.value_estimate
    top_actions = response.action_distribution.top_k_actions

    print(f"Recommended Action: {action}")
    print(f"Value Estimate: {value:.3f} (win prob: {(value + 1) / 2:.1%})")
    print(f"Inference Time: {latency:.1f}ms (server: {response.inference_time_ms:.1f}ms)")
    print(f"Model Version: v{response.model_version}")
    print()
    print("Top 5 Actions:")
    for i, (act_id, prob) in enumerate(top_actions, 1):
        print(f"  {i}. Action {act_id}: {prob:.2%}")
    print()


def example_batch_inference(client: InferenceClient) -> None:
    """Example 3: Batch inference for multiple game states."""
    print("=" * 60)
    print("Example 3: Batch Inference")
    print("=" * 60)

    # Create batch of requests
    batch_size = 5
    requests = []
    for i in range(batch_size):
        game_state = create_random_game_state(board_size=8)
        request = InferenceRequest(
            game_state=game_state,
            mcts_simulations=25,
            session_id=f"batch-game-{i}",
        )
        requests.append(request)

    # Run batch inference
    start = time.time()
    responses = client.batch_infer(requests, max_batch_size=32)
    total_time = (time.time() - start) * 1000

    # Display results
    print(f"Processed {len(responses)} requests in {total_time:.1f}ms")
    print(f"Average time per request: {total_time / len(responses):.1f}ms")
    print()
    print("Results:")
    for i, response in enumerate(responses):
        action = response.action_distribution.action_id
        value = response.action_distribution.value_estimate
        print(f"  Game {i}: Action {action}, Value {value:.3f}")
    print()


def example_model_swap(client: InferenceClient) -> None:
    """Example 4: Load different model version."""
    print("=" * 60)
    print("Example 4: Model Version Swap")
    print("=" * 60)

    # Get current version
    health = client.health()
    current_version = health.model_version
    print(f"Current model version: v{current_version}")

    # Try to load a different version (if available)
    target_version = current_version - 10 if current_version and current_version > 10 else None

    if target_version:
        print(f"Attempting to load v{target_version}...")
        result = client.swap_model(model_version=target_version)

        if result["success"]:
            print(f"✓ Successfully loaded v{result['new_version']}")
            print(f"  Load time: {result['load_time_ms']:.1f}ms")
        else:
            print(f"✗ Failed to load v{target_version}")
            print(f"  Error: {result['message']}")

        # Swap back to latest
        print(f"\nSwapping back to v{current_version}...")
        result = client.swap_model(model_version=current_version)
        print(f"✓ Restored v{result['new_version']}")
    else:
        print("Skipping model swap (only one version available)")

    print()


def example_performance_comparison(client: InferenceClient) -> None:
    """Example 5: Compare performance across MCTS simulation counts."""
    print("=" * 60)
    print("Example 5: Performance Comparison")
    print("=" * 60)

    game_state = create_random_game_state(board_size=8)
    simulation_counts = [16, 25, 50, 100]

    print(f"{'MCTS Sims':<12} {'Latency (ms)':<15} {'Action':<10} {'Value':<10}")
    print("-" * 60)

    for sims in simulation_counts:
        start = time.time()
        response = client.infer(game_state=game_state, mcts_simulations=sims)
        latency = (time.time() - start) * 1000

        action = response.action_distribution.action_id
        value = response.action_distribution.value_estimate

        print(f"{sims:<12} {latency:<15.1f} {action:<10} {value:<10.3f}")

    print()


def example_metrics(client: InferenceClient) -> None:
    """Example 6: Get service metrics."""
    print("=" * 60)
    print("Example 6: Service Metrics")
    print("=" * 60)

    metrics = client.metrics()
    print(f"Total Requests: {metrics.total_requests}")
    print(f"Requests/Second: {metrics.requests_per_second:.2f}")
    print(f"Active Sessions: {metrics.active_sessions}")
    print()
    print("Latency Statistics:")
    print(f"  P50: {metrics.average_latency_ms:.1f}ms")
    print(f"  P95: {metrics.p95_latency_ms:.1f}ms")
    print(f"  P99: {metrics.p99_latency_ms:.1f}ms")
    print()


def example_custom_game_state(client: InferenceClient) -> None:
    """Example 7: Create custom game state (not random)."""
    print("=" * 60)
    print("Example 7: Custom Game State")
    print("=" * 60)

    # Create a specific game state
    # In real usage, you would extract these from your game engine
    board_size = 8

    # Spatial features: [channels, height, width]
    spatial = np.zeros((13, board_size, board_size), dtype=np.float32)

    # Example: Place some terrain
    spatial[0, :, :] = 1.0  # Plains everywhere
    spatial[1, 2:4, 2:4] = 1.0  # Hills in center

    # Example: Place some units
    spatial[3, 0, 0] = 1.0  # Own warrior at (0,0)
    spatial[6, 7, 7] = 1.0  # Enemy warrior at (7,7)

    # Vector features: global state
    vector = np.zeros(64, dtype=np.float32)
    vector[0] = 0.5  # Example: 50% science progress
    vector[1] = 0.3  # Example: 30% culture progress

    # Action mask: which actions are valid
    action_mask = np.ones(376, dtype=int)
    action_mask[0] = 0  # Example: PASS action not valid

    # Create GameState
    game_state = GameState(
        spatial=spatial.flatten().tolist(),
        vector=vector.tolist(),
        action_mask=action_mask.tolist(),
        board_size=board_size,
        turn_number=10,
    )

    # Run inference
    response = client.infer(game_state=game_state, mcts_simulations=25)

    print("Custom game state inference:")
    print(f"  Action: {response.action_distribution.action_id}")
    print(f"  Value: {response.action_distribution.value_estimate:.3f}")
    print()


def main() -> None:
    """Run all examples."""
    parser = argparse.ArgumentParser(description="Inference service quick start examples")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Inference service URL",
    )
    args = parser.parse_args()

    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 8 + "Civ6 AlphaZero Inference Quick Start" + " " * 14 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    print(f"Service URL: {args.url}")
    print()

    # Create client
    try:
        with InferenceClient(args.url, timeout=30.0) as client:
            # Run examples
            example_health_check(client)
            example_single_inference(client)
            example_batch_inference(client)
            example_model_swap(client)
            example_performance_comparison(client)
            example_metrics(client)
            example_custom_game_state(client)

        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check that the service is running")
        print("2. Verify the URL is correct")
        print("3. Ensure you have network connectivity")
        raise


if __name__ == "__main__":
    main()
