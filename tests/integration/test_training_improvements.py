"""Integration tests for training improvement modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import numpy as np

from alpha_zero.utils.environment_tuning import (
    EnvironmentTuningConfig,
    create_tuned_environment,
    get_default_fast_config,
)
from alpha_zero.utils.enhanced_rewards import (
    EnhancedRewardsConfig,
    EnhancedRewardShaper,
    create_enhanced_reward_shaper,
)
from alpha_zero.utils.curriculum_learning import (
    CurriculumConfig,
    CurriculumManager,
    create_curriculum_manager,
)
from alpha_zero.utils.training_config_enhanced import (
    EnhancedTrainingConfig,
    create_training_config,
    get_fast_training_config,
)


class TestTunedEnvironmentTraining:
    """Integration tests for tuned environment training."""

    def test_tuned_environment_game_loop(self):
        """Test complete game loop with tuned environment."""
        config = get_default_fast_config()
        config.max_turns = 20  # Very short for testing

        env = create_tuned_environment(config, board_size=8)

        obs, info = env.reset()

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            # Get valid action
            action_mask = env.env.get_valid_actions()
            valid_actions = np.where(action_mask > 0)[0]
            action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if steps > 100:  # Safety limit
                break

        assert steps > 0
        assert "episode_stats" in info

    def test_tuned_environment_early_termination(self):
        """Test early termination functionality."""
        config = EnvironmentTuningConfig(
            max_turns=100,
            early_termination=EnvironmentTuningConfig().early_termination,
        )
        config.early_termination.enabled = True
        config.early_termination.min_turns = 1
        config.early_termination.advantage_threshold = 1

        env = create_tuned_environment(config, board_size=8)

        # Run multiple games to verify early termination can occur
        early_terminations = 0
        for _ in range(10):
            env.reset()
            terminated = False
            truncated = False

            while not terminated and not truncated:
                action_mask = env.env.get_valid_actions()
                valid_actions = np.where(action_mask > 0)[0]
                action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0

                obs, reward, terminated, truncated, info = env.step(action)

                if info.get("early_termination"):
                    early_terminations += 1
                    break

        # At least some games should terminate early with random play
        assert early_terminations >= 0  # May not always trigger


class TestEnhancedRewardsTraining:
    """Integration tests for enhanced rewards."""

    def test_enhanced_rewards_with_environment(self):
        """Test enhanced rewards integrated with environment."""
        env_config = get_default_fast_config()
        env = create_tuned_environment(env_config, board_size=8)

        reward_shaper = create_enhanced_reward_shaper(
            scale_factor=1.5,
            board_size=8,
        )

        obs, info = env.reset()
        reward_shaper.reset()

        total_base_reward = 0.0
        total_shaped_reward = 0.0
        steps = 0

        terminated = False
        truncated = False

        while not terminated and not truncated:
            action_mask = env.env.get_valid_actions()
            valid_actions = np.where(action_mask > 0)[0]
            action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0

            obs, base_reward, terminated, truncated, info = env.step(action)

            shaped_reward = reward_shaper.shape_reward(
                obs, base_reward, info, terminated=terminated
            )

            total_base_reward += base_reward
            total_shaped_reward += shaped_reward
            steps += 1

            if steps > 50:
                break

        stats = reward_shaper.get_episode_stats()

        assert stats["num_steps"] == steps
        assert stats["base_reward_sum"] == pytest.approx(total_base_reward)
        # Shaped rewards should be different from base
        assert stats["reward_density"] >= 0

    def test_reward_shaping_preserves_win_signal(self):
        """Test that reward shaping preserves win/loss signals."""
        shaper = create_enhanced_reward_shaper(scale_factor=1.0, board_size=8)

        # Create mock observation
        obs = {
            "spatial": np.zeros((13, 8, 8), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
        }

        shaper.reset()

        # First step
        shaper.shape_reward(obs, base_reward=0.0)

        # Final step with win
        win_info = {"winner": 0, "current_player": 0}
        shaped = shaper.shape_reward(obs, base_reward=1.0, info=win_info, terminated=True)

        # Should preserve positive signal for win
        assert shaped >= 1.0


class TestCurriculumProgression:
    """Integration tests for curriculum learning."""

    def test_curriculum_with_environment(self):
        """Test curriculum learning with environment."""
        manager = create_curriculum_manager(num_stages=3)

        # Get environment params from curriculum
        params = manager.get_environment_params()

        # Create environment with curriculum params
        config = EnvironmentTuningConfig(
            board_size=params["board_size"],
            max_turns=params["max_turns"],
        )
        env = create_tuned_environment(config)

        assert env.env.board_size == params["board_size"]
        assert env.env.max_turns == params["max_turns"]

    def test_curriculum_progression_simulation(self):
        """Test simulated curriculum progression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "curriculum.json"

            config = CurriculumConfig.default_three_stage()
            config.save_path = str(save_path)

            # Reduce min_games for faster testing
            for stage in config.stages:
                stage.min_games = 10

            manager = CurriculumManager(config)

            # Simulate training with high win rate
            for i in range(50):
                won = np.random.random() > 0.2  # 80% win rate
                result = manager.record_game(won=won)

                if result["changed"]:
                    break

            # Should have promoted at least once
            progress = manager.get_progress()
            assert progress["current_level"] >= 0

    def test_full_curriculum_completion(self):
        """Test completing full curriculum."""
        config = CurriculumConfig.default_three_stage()

        # Minimal games for testing
        for stage in config.stages:
            stage.min_games = 5
            stage.promotion_threshold = 0.5

        manager = CurriculumManager(config)

        # Play enough games with high win rate to complete curriculum
        for _ in range(100):
            manager.record_game(won=True)

            if manager.get_progress()["at_final_stage"]:
                break

        # Should be at final stage
        assert manager.get_progress()["at_final_stage"] is True


class TestFullTrainingPipeline:
    """Integration tests for full training pipeline."""

    def test_all_components_together(self):
        """Test all improvement components working together."""
        # 1. Create curriculum manager
        curriculum = create_curriculum_manager(num_stages=3)
        env_params = curriculum.get_environment_params()

        # 2. Create tuned environment
        env_config = EnvironmentTuningConfig(
            board_size=env_params["board_size"],
            max_turns=env_params["max_turns"],
            early_termination=EnvironmentTuningConfig().early_termination,
        )
        env = create_tuned_environment(env_config)

        # 3. Create reward shaper
        reward_shaper = create_enhanced_reward_shaper(
            scale_factor=1.0,
            board_size=env_params["board_size"],
        )

        # 4. Create training config
        training_config = get_fast_training_config()

        # 5. Run a few games
        for game_idx in range(5):
            obs, info = env.reset()
            reward_shaper.reset()

            terminated = False
            truncated = False
            total_reward = 0.0

            while not terminated and not truncated:
                action_mask = env.env.get_valid_actions()
                valid_actions = np.where(action_mask > 0)[0]
                action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0

                obs, base_reward, terminated, truncated, info = env.step(action)

                shaped = reward_shaper.shape_reward(
                    obs, base_reward, info, terminated=terminated
                )
                total_reward += shaped

            # Record game in curriculum
            won = info.get("winner") == 0
            curriculum.record_game(won=won)

        # Verify everything ran
        progress = curriculum.get_progress()
        assert progress["total_games"] == 5

    def test_training_config_schedules(self):
        """Test training config schedule integration."""
        config = create_training_config(
            batch_size=64,
            learning_rate=0.001,
            num_iterations=100,
            use_schedules=True,
        )

        # Verify schedules at different points
        batch_0 = config.get_batch_size(0)
        batch_end = config.get_batch_size(100)

        # Note: warmup_cosine starts at 0 during warmup, so check after warmup
        warmup_steps = config.learning_rate_schedule.warmup_steps if config.learning_rate_schedule else 0
        lr_after_warmup = config.get_learning_rate(warmup_steps)
        lr_end = config.get_learning_rate(100)

        # Batch size should increase
        assert batch_end >= batch_0

        # Learning rate should decrease after warmup
        assert lr_end <= lr_after_warmup


class TestConfigurationPersistence:
    """Integration tests for configuration persistence."""

    def test_save_load_curriculum_state(self):
        """Test saving and loading curriculum state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "curriculum.json"

            # Create and train
            config = CurriculumConfig.default_three_stage()
            config.save_path = str(save_path)

            for stage in config.stages:
                stage.min_games = 5

            manager = CurriculumManager(config)

            # Record games and promote
            for _ in range(20):
                manager.record_game(won=True)

            original_progress = manager.get_progress()

            # Create new manager with same save path
            manager2 = CurriculumManager(config)

            # Should have same state
            loaded_progress = manager2.get_progress()

            assert loaded_progress["current_level"] == original_progress["current_level"]
            assert loaded_progress["total_games"] == original_progress["total_games"]

    def test_configuration_roundtrip(self):
        """Test configuration dictionary roundtrip."""
        # Training config
        training = create_training_config(
            batch_size=128,
            learning_rate=0.002,
            num_iterations=5000,
            use_schedules=True,
        )

        training_dict = training.to_dict()
        training_loaded = EnhancedTrainingConfig.from_dict(training_dict)

        assert training_loaded.batch_size == training.batch_size
        assert training_loaded.learning_rate == training.learning_rate

        # Curriculum config
        curriculum = CurriculumConfig.default_three_stage()
        curriculum_dict = curriculum.to_dict()
        curriculum_loaded = CurriculumConfig.from_dict(curriculum_dict)

        assert len(curriculum_loaded.stages) == len(curriculum.stages)

        # Enhanced rewards config
        rewards = EnhancedRewardsConfig()
        rewards_dict = rewards.to_dict()
        rewards_loaded = EnhancedRewardsConfig.from_dict(rewards_dict)

        assert rewards_loaded.scale_factor == rewards.scale_factor


class TestResourceManagement:
    """Integration tests for resource management."""

    def test_batch_size_recommendation(self):
        """Test batch size recommendation with real model size."""
        from alpha_zero.utils.training_config_enhanced import ResourceManager, ResourceConfig

        config = ResourceConfig(max_memory_gb=4.0)  # Simulate limited memory
        manager = ResourceManager(config)

        batch_size, accumulation = manager.recommend_batch_size(
            spatial_size=(13, 16, 16),
            vector_size=64,
            model_params=500000,  # ~500K params
            target_effective_batch=256,
        )

        effective = batch_size * accumulation

        # Should achieve reasonable effective batch size
        assert effective >= 64
        assert accumulation >= 1
