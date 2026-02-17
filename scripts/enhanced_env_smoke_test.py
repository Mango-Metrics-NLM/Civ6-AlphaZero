"""Enhanced Environment Pipeline Smoke Test.

Validates the full enhanced-env pipeline locally before cloud deployment:
  1. Environment factory creates correct env types per preset
  2. Observation shapes are consistent across presets
  3. Enhanced env has all gameplay systems enabled
  4. Self-play runs with GumbelMCTS on enhanced env
  5. Vectorized env works with enhanced preset
  6. Docker container has gameplay modules (skippable)
  7. GCS path isolation prevents production data collision

Usage::

    python scripts/enhanced_env_smoke_test.py              # all tests
    python scripts/enhanced_env_smoke_test.py --skip-container  # skip Docker check
    python scripts/enhanced_env_smoke_test.py --board-size 6    # custom board size
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from typing import NamedTuple


class SmokeResult(NamedTuple):
    """Result of a single smoke test."""

    name: str
    passed: bool
    elapsed: float
    detail: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced env pipeline smoke test")
    parser.add_argument(
        "--skip-container",
        action="store_true",
        help="Skip Docker container validation (for CI without Docker)",
    )
    parser.add_argument("--board-size", type=int, default=8, help="Board size (default: 8)")
    parser.add_argument("--max-turns", type=int, default=10, help="Max turns (default: 10)")
    parser.add_argument(
        "--container-image",
        type=str,
        default=None,
        help="Container image URI (default: from CIV6AZ_CONTAINER_IMAGE_URI or latest)",
    )
    parser.add_argument(
        "--num-selfplay-games",
        type=int,
        default=2,
        help="Number of self-play games to run (default: 2)",
    )
    parser.add_argument(
        "--num-parallel-envs",
        type=int,
        default=4,
        help="Number of parallel envs for vectorized test (default: 4)",
    )
    return parser.parse_args()


def test_env_creation(board_size: int, max_turns: int) -> SmokeResult:
    """Verify create_game_env() returns correct types for both presets."""
    t0 = time.perf_counter()
    try:
        from alpha_zero.envs.env_factory import GAMEPLAY_PRESETS, create_game_env
        from alpha_zero.envs.microciv import MicroCivEnv
        from alpha_zero.envs.microciv_enhanced import EnhancedMicroCivEnv

        for preset in GAMEPLAY_PRESETS:
            env = create_game_env(
                board_size=board_size, max_turns=max_turns, gameplay_preset=preset
            )
            if preset == "default":
                assert isinstance(env, MicroCivEnv), f"Expected MicroCivEnv, got {type(env)}"
                assert not isinstance(
                    env, EnhancedMicroCivEnv
                ), "Default should not be EnhancedMicroCivEnv"
            elif preset == "enhanced":
                assert isinstance(
                    env, EnhancedMicroCivEnv
                ), f"Expected EnhancedMicroCivEnv, got {type(env)}"

        # Unknown preset should raise
        try:
            create_game_env(board_size=board_size, max_turns=max_turns, gameplay_preset="bogus")
            return SmokeResult("env_creation", False, time.perf_counter() - t0, "No ValueError")
        except ValueError:
            pass

        return SmokeResult(
            "env_creation",
            True,
            time.perf_counter() - t0,
            f"All {len(GAMEPLAY_PRESETS)} presets OK",
        )
    except Exception as e:
        return SmokeResult("env_creation", False, time.perf_counter() - t0, str(e))


def test_observation_shapes(board_size: int, max_turns: int) -> SmokeResult:
    """Both presets must produce identical observation shapes."""
    t0 = time.perf_counter()
    try:
        from alpha_zero.envs.env_factory import GAMEPLAY_PRESETS, create_game_env

        expected_spatial = (13, board_size, board_size)
        expected_vector = (64,)
        expected_mask = (376,)

        for preset in GAMEPLAY_PRESETS:
            env = create_game_env(
                board_size=board_size, max_turns=max_turns, gameplay_preset=preset
            )
            env.reset()
            obs = env.get_observation()
            assert (
                obs["spatial"].shape == expected_spatial
            ), f"{preset}: spatial {obs['spatial'].shape} != {expected_spatial}"
            assert (
                obs["vector"].shape == expected_vector
            ), f"{preset}: vector {obs['vector'].shape} != {expected_vector}"
            assert (
                obs["action_mask"].shape == expected_mask
            ), f"{preset}: mask {obs['action_mask'].shape} != {expected_mask}"

        return SmokeResult(
            "observation_shapes",
            True,
            time.perf_counter() - t0,
            f"spatial={expected_spatial}, vector={expected_vector}, mask={expected_mask}",
        )
    except Exception as e:
        return SmokeResult("observation_shapes", False, time.perf_counter() - t0, str(e))


def test_enhanced_systems(board_size: int, max_turns: int) -> SmokeResult:
    """All 5 GameplayConfig flags must be enabled for 'enhanced' preset."""
    t0 = time.perf_counter()
    try:
        from alpha_zero.envs.env_factory import create_game_env

        env = create_game_env(
            board_size=board_size, max_turns=max_turns, gameplay_preset="enhanced"
        )
        cfg = env.gameplay_config

        flags = {
            "fog_of_war": cfg.enable_fog_of_war,
            "enhanced_combat": cfg.enable_enhanced_combat,
            "economy": cfg.enable_economy,
            "city_growth": cfg.enable_city_growth,
            "terrain_effects": cfg.enable_terrain_effects,
        }
        all_on = all(flags.values())
        detail = ", ".join(f"{k}={'ON' if v else 'OFF'}" for k, v in flags.items())

        return SmokeResult("enhanced_systems", all_on, time.perf_counter() - t0, detail)
    except Exception as e:
        return SmokeResult("enhanced_systems", False, time.perf_counter() - t0, str(e))


def test_local_selfplay(board_size: int, max_turns: int, num_games: int) -> SmokeResult:
    """Run a few self-play games using GumbelMCTS on the enhanced env."""
    t0 = time.perf_counter()
    try:
        import torch

        from alpha_zero.core.civ6_network import MicroCivNetwork
        from alpha_zero.core.mcts_factory import create_mcts
        from alpha_zero.envs.env_factory import create_game_env

        # Create a temp env to get the action space size dynamically
        tmp_env = create_game_env(board_size=board_size, max_turns=1, gameplay_preset="enhanced")
        action_space_size = tmp_env.action_space.n

        network = MicroCivNetwork(
            board_size=board_size,
            action_space_size=action_space_size,
            num_res_blocks=2,
            num_filters=32,
            num_heads=4,
            policy_hidden_dim=64,
            value_head_dim=32,
        )
        network.eval()
        mcts = create_mcts("gumbel", network=network, device="cpu")

        completed = 0
        for _ in range(num_games):
            env = create_game_env(
                board_size=board_size, max_turns=max_turns, gameplay_preset="enhanced"
            )
            env.reset()
            done = False
            steps = 0
            while not done:
                with torch.no_grad():
                    action_probs, _ = mcts.search(env, add_noise=False)
                action = mcts.select_action(action_probs, temperature=0.0)
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
            completed += 1

        return SmokeResult(
            "local_selfplay",
            completed == num_games,
            time.perf_counter() - t0,
            f"{completed}/{num_games} games completed",
        )
    except Exception as e:
        return SmokeResult("local_selfplay", False, time.perf_counter() - t0, str(e))


def test_vectorized_env(board_size: int, max_turns: int, num_envs: int) -> SmokeResult:
    """Vectorized env with enhanced preset should create correct env types."""
    t0 = time.perf_counter()
    try:
        import numpy as np

        from alpha_zero.envs.microciv_enhanced import EnhancedMicroCivEnv
        from alpha_zero.envs.vectorized_env import VectorizedEnvConfig, VectorizedMicroCivEnv

        cfg = VectorizedEnvConfig(
            num_envs=num_envs,
            board_size=board_size,
            max_turns=max_turns,
            gameplay_preset="enhanced",
        )
        vec_env = VectorizedMicroCivEnv(config=cfg)
        vec_env.reset_all()

        # All sub-envs should be EnhancedMicroCivEnv
        for i in range(num_envs):
            assert isinstance(
                vec_env.get_env(i), EnhancedMicroCivEnv
            ), f"Env {i} is {type(vec_env.get_env(i))}"

        spatial, vector, mask = vec_env.get_observations()
        assert spatial.shape == (num_envs, 13, board_size, board_size)
        assert vector.shape == (num_envs, 64)
        assert mask.shape == (num_envs, 376)

        # Take one step with valid actions
        actions = np.zeros(num_envs, dtype=np.int32)
        for i in range(num_envs):
            valid = np.where(mask[i] > 0)[0]
            if len(valid) > 0:
                actions[i] = valid[0]

        rewards, terminated, truncated, infos = vec_env.step_all(actions)
        assert rewards.shape == (num_envs,)

        return SmokeResult(
            "vectorized_env",
            True,
            time.perf_counter() - t0,
            f"{num_envs} envs, shapes OK, step OK",
        )
    except Exception as e:
        return SmokeResult("vectorized_env", False, time.perf_counter() - t0, str(e))


def test_container_validation(container_image: str | None) -> SmokeResult:
    """Verify Docker container has gameplay modules and accepts --gameplay_preset."""
    import os

    t0 = time.perf_counter()
    try:
        image = container_image or os.environ.get(
            "CIV6AZ_CONTAINER_IMAGE_URI", "gcr.io/hrmtrain/civ6az-training:latest"
        )

        docker_bin = shutil.which("docker")
        if docker_bin is None:
            return SmokeResult(
                "container_validation",
                False,
                time.perf_counter() - t0,
                "Docker not found in PATH",
            )

        # Test 1: gameplay modules importable (override entrypoint to use python)
        result = subprocess.run(  # noqa: S603
            [
                docker_bin,
                "run",
                "--rm",
                "--entrypoint",
                "python",
                image,
                "-c",
                "from alpha_zero.envs.env_factory import GAMEPLAY_PRESETS; "
                "print(f'PRESETS={GAMEPLAY_PRESETS}')",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return SmokeResult(
                "container_validation",
                False,
                time.perf_counter() - t0,
                f"Import failed: {result.stderr.strip()[:200]}",
            )

        # Test 2: entrypoint accepts --gameplay_preset
        result2 = subprocess.run(  # noqa: S603
            [
                docker_bin,
                "run",
                "--rm",
                image,
                "--mode=selfplay",
                "--worker_id=0",
                "--model_version=0",
                "--iteration=0",
                "--games=1",
                "--max_turns=5",
                "--gameplay_preset=enhanced",
                "--dry_run",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result2.returncode != 0:
            return SmokeResult(
                "container_validation",
                False,
                time.perf_counter() - t0,
                f"Entrypoint failed: {result2.stderr.strip()[:200]}",
            )

        return SmokeResult(
            "container_validation",
            True,
            time.perf_counter() - t0,
            f"Image {image}: imports OK, entrypoint OK",
        )
    except subprocess.TimeoutExpired:
        return SmokeResult(
            "container_validation", False, time.perf_counter() - t0, "Timed out (120s)"
        )
    except Exception as e:
        return SmokeResult("container_validation", False, time.perf_counter() - t0, str(e))


def test_gcs_path_isolation() -> SmokeResult:
    """GCS paths with agent_id must not collide with production."""
    t0 = time.perf_counter()
    try:
        from alpha_zero.cloud.gcp.config import GCSPathConfig

        agent_id = "enhanced-smoke-v1"
        paths = GCSPathConfig(agent_id=agent_id)
        prod = GCSPathConfig()

        model = paths.model_path(1)
        traj = paths.trajectory_path(1, 0)
        ckpt = paths.checkpoint_path(100)

        # All paths must contain agent_id
        assert agent_id in model, f"Model not isolated: {model}"
        assert agent_id in traj, f"Trajectory not isolated: {traj}"
        assert agent_id in ckpt, f"Checkpoint not isolated: {ckpt}"

        # Must differ from production
        assert prod.model_path(1) != model, "Model paths collide with production!"
        assert prod.trajectory_path(1, 0) != traj, "Trajectory paths collide!"
        assert prod.checkpoint_path(100) != ckpt, "Checkpoint paths collide!"

        return SmokeResult(
            "gcs_path_isolation",
            True,
            time.perf_counter() - t0,
            f"agent_id={agent_id!r}, all paths isolated",
        )
    except Exception as e:
        return SmokeResult("gcs_path_isolation", False, time.perf_counter() - t0, str(e))


def main() -> int:
    """Run all smoke tests and report results."""
    args = _parse_args()

    print("=== Enhanced Environment Pipeline Smoke Test ===")
    print(f"Board size: {args.board_size}, Max turns: {args.max_turns}")
    print(f"Skip container: {args.skip_container}\n")

    tests: list[SmokeResult] = []

    # Run tests in order
    test_configs = [
        ("env_creation", lambda: test_env_creation(args.board_size, args.max_turns)),
        ("observation_shapes", lambda: test_observation_shapes(args.board_size, args.max_turns)),
        ("enhanced_systems", lambda: test_enhanced_systems(args.board_size, args.max_turns)),
        (
            "local_selfplay",
            lambda: test_local_selfplay(args.board_size, args.max_turns, args.num_selfplay_games),
        ),
        (
            "vectorized_env",
            lambda: test_vectorized_env(args.board_size, args.max_turns, args.num_parallel_envs),
        ),
        (
            "container_validation",
            lambda: test_container_validation(args.container_image),
        ),
        ("gcs_path_isolation", lambda: test_gcs_path_isolation()),
    ]

    total = len(test_configs)
    if args.skip_container:
        test_configs = [(n, f) for n, f in test_configs if n != "container_validation"]
        total = len(test_configs)

    for idx, (name, test_fn) in enumerate(test_configs, 1):
        label = f"[{idx}/{total}]"
        print(f"{label} {name}...", end=" ", flush=True)
        result = test_fn()
        tests.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} ({result.elapsed:.2f}s) — {result.detail}")

    # Summary
    passed = sum(1 for t in tests if t.passed)
    failed = sum(1 for t in tests if not t.passed)
    total_time = sum(t.elapsed for t in tests)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed ({total_time:.1f}s total)")

    if failed:
        print("\nFailed tests:")
        for t in tests:
            if not t.passed:
                print(f"  - {t.name}: {t.detail}")
        return 1

    print("\n=== All smoke tests PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
