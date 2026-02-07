"""Tests for the Learner module (dry-run mode, no torch required)."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from src.core.config import TrainingConfig
from src.core.training.learner import Learner, _dummy_eval_fn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    batch_size: int = 4,
    spatial_channels: int = 6,
    board_size: int = 8,
    vector_dim: int = 4,
    action_size: int = 5,
) -> dict[str, np.ndarray]:
    """Create a synthetic training batch as numpy arrays."""
    rng = np.random.default_rng(42)
    return {
        "spatial": rng.standard_normal((batch_size, spatial_channels, board_size, board_size)).astype(np.float32),
        "vector": rng.standard_normal((batch_size, vector_dim)).astype(np.float32),
        "action_mask": np.ones((batch_size, action_size), dtype=np.float32),
        "policy": np.ones((batch_size, action_size), dtype=np.float32) / action_size,
        "value": np.array([1.0, -1.0, 0.0, 1.0], dtype=np.float32),
    }


def _make_config(**overrides: object) -> TrainingConfig:
    """Create a TrainingConfig with optional overrides."""
    defaults: dict[str, object] = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


# ===================================================================
# Learner constructor tests
# ===================================================================


class TestLearnerConstructor:
    """Tests for Learner construction in dry-run mode."""

    def test_no_network_dry_run_mode(self) -> None:
        """Learner with no network enters dry-run mode."""
        config = _make_config()
        learner = Learner(config, network=None)
        assert learner.step_count == 0

    def test_step_count_starts_at_zero(self) -> None:
        """step_count is 0 immediately after construction."""
        learner = Learner(_make_config())
        assert learner.step_count == 0

    def test_network_property_is_none(self) -> None:
        """Network property returns None when no network is provided."""
        learner = Learner(_make_config(), network=None)
        assert learner.network is None

    def test_optimizer_is_none_in_dry_run(self) -> None:
        """Internal optimizer should be None in dry-run mode."""
        learner = Learner(_make_config())
        assert learner._optimizer is None

    def test_loss_fn_is_none_in_dry_run(self) -> None:
        """Internal loss function should be None in dry-run mode."""
        learner = Learner(_make_config())
        assert learner._loss_fn is None

    def test_constructor_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Dry-run mode logs a warning about the missing network or torch."""
        with caplog.at_level(logging.WARNING):
            Learner(_make_config(), network=None)
        assert any("dry-run" in record.message.lower() for record in caplog.records)

    def test_custom_config_preserved(self) -> None:
        """Configuration values are stored correctly on the Learner."""
        config = _make_config(learning_rate=0.01, weight_decay=0.001)
        learner = Learner(config)
        assert learner._config.learning_rate == 0.01
        assert learner._config.weight_decay == 0.001


# ===================================================================
# train_step dry-run tests
# ===================================================================


class TestTrainStepDryRun:
    """Tests for train_step in dry-run mode (no torch)."""

    def test_returns_dict(self) -> None:
        """train_step returns a dictionary of metrics."""
        learner = Learner(_make_config())
        batch = _make_batch()
        result = learner.train_step(batch)
        assert isinstance(result, dict)

    def test_expected_keys_present(self) -> None:
        """Result dictionary contains all expected metric keys."""
        learner = Learner(_make_config())
        batch = _make_batch()
        result = learner.train_step(batch)

        expected_keys = {"total_loss", "policy_loss", "value_loss", "step", "dry_run"}
        assert expected_keys.issubset(result.keys())

    def test_total_loss_is_zero(self) -> None:
        """Dry-run total_loss is 0.0."""
        learner = Learner(_make_config())
        result = learner.train_step(_make_batch())
        assert result["total_loss"] == 0.0

    def test_policy_loss_is_zero(self) -> None:
        """Dry-run policy_loss is 0.0."""
        learner = Learner(_make_config())
        result = learner.train_step(_make_batch())
        assert result["policy_loss"] == 0.0

    def test_value_loss_is_zero(self) -> None:
        """Dry-run value_loss is 0.0."""
        learner = Learner(_make_config())
        result = learner.train_step(_make_batch())
        assert result["value_loss"] == 0.0

    def test_dry_run_flag_is_set(self) -> None:
        """Dry-run flag is 1.0 in the result."""
        learner = Learner(_make_config())
        result = learner.train_step(_make_batch())
        assert result["dry_run"] == 1.0

    def test_step_increments_to_one(self) -> None:
        """After one train_step, step count in result is 1."""
        learner = Learner(_make_config())
        result = learner.train_step(_make_batch())
        assert result["step"] == 1.0

    def test_batch_size_reported(self) -> None:
        """Batch size is correctly reported in the metrics."""
        learner = Learner(_make_config())
        batch = _make_batch(batch_size=8)
        result = learner.train_step(batch)
        assert result["batch_size"] == 8.0

    def test_all_values_are_floats(self) -> None:
        """All metric values should be floats."""
        learner = Learner(_make_config())
        result = learner.train_step(_make_batch())
        for key, value in result.items():
            assert isinstance(value, float), f"Expected float for key '{key}', got {type(value)}"

    def test_empty_spatial_batch(self) -> None:
        """Handle a batch with empty spatial tensor gracefully."""
        learner = Learner(_make_config())
        batch = {
            "spatial": np.empty((0, 6, 8, 8), dtype=np.float32),
            "vector": np.empty((0, 4), dtype=np.float32),
            "action_mask": np.empty((0, 5), dtype=np.float32),
            "policy": np.empty((0, 5), dtype=np.float32),
            "value": np.empty((0,), dtype=np.float32),
        }
        result = learner.train_step(batch)
        assert result["batch_size"] == 0.0
        assert result["dry_run"] == 1.0


# ===================================================================
# Multiple train steps
# ===================================================================


class TestMultipleTrainSteps:
    """Tests for step_count incrementing across multiple train_step calls."""

    def test_step_count_increments(self) -> None:
        """Each train_step in dry-run mode increments step_count by 1."""
        learner = Learner(_make_config())
        batch = _make_batch()

        for i in range(1, 6):
            result = learner.train_step(batch)
            assert learner.step_count == i
            assert result["step"] == float(i)

    def test_ten_steps(self) -> None:
        """After 10 train_steps, step_count is 10."""
        learner = Learner(_make_config())
        batch = _make_batch()

        for _ in range(10):
            learner.train_step(batch)

        assert learner.step_count == 10

    def test_step_count_matches_result(self) -> None:
        """The step value in the result always matches the learner's step_count."""
        learner = Learner(_make_config())
        batch = _make_batch()

        for _ in range(7):
            result = learner.train_step(batch)
            assert result["step"] == float(learner.step_count)


# ===================================================================
# get_eval_fn tests
# ===================================================================


class TestGetEvalFn:
    """Tests for get_eval_fn in dry-run mode."""

    def test_returns_callable(self) -> None:
        """get_eval_fn returns a callable."""
        learner = Learner(_make_config())
        eval_fn = learner.get_eval_fn()
        assert callable(eval_fn)

    def test_returns_dummy_eval_fn(self) -> None:
        """In dry-run mode, get_eval_fn returns _dummy_eval_fn."""
        learner = Learner(_make_config(), network=None)
        eval_fn = learner.get_eval_fn()
        assert eval_fn is _dummy_eval_fn

    def test_dummy_eval_fn_signature(self) -> None:
        """The returned eval function accepts (spatial, vector, action_mask)."""
        learner = Learner(_make_config())
        eval_fn = learner.get_eval_fn()

        spatial = np.zeros((6, 8, 8), dtype=np.float32)
        vector = np.zeros(4, dtype=np.float32)
        action_mask = np.ones(5, dtype=np.bool_)

        result = eval_fn(spatial, vector, action_mask)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_eval_fn_output_types(self) -> None:
        """Eval function returns (NDArray, float)."""
        learner = Learner(_make_config())
        eval_fn = learner.get_eval_fn()

        spatial = np.zeros((6, 8, 8), dtype=np.float32)
        vector = np.zeros(4, dtype=np.float32)
        action_mask = np.ones(5, dtype=np.bool_)

        policy, value = eval_fn(spatial, vector, action_mask)
        assert isinstance(policy, np.ndarray)
        assert isinstance(value, float)


# ===================================================================
# _dummy_eval_fn tests
# ===================================================================


class TestDummyEvalFn:
    """Tests for the module-level _dummy_eval_fn."""

    def test_uniform_policy_over_valid_actions(self) -> None:
        """Returns uniform distribution over valid (masked) actions."""
        spatial = np.zeros((6, 8, 8), dtype=np.float32)
        vector = np.zeros(4, dtype=np.float32)
        action_mask = np.array([True, True, True, False, False], dtype=np.bool_)

        policy, _value = _dummy_eval_fn(spatial, vector, action_mask)

        # 3 valid actions -> each gets 1/3
        expected_policy = np.array([1 / 3, 1 / 3, 1 / 3, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(policy, expected_policy, atol=1e-6)

    def test_value_is_zero(self) -> None:
        """Always returns value 0.0."""
        spatial = np.zeros((6, 8, 8), dtype=np.float32)
        vector = np.zeros(4, dtype=np.float32)
        action_mask = np.ones(5, dtype=np.bool_)

        _, value = _dummy_eval_fn(spatial, vector, action_mask)
        assert value == 0.0

    def test_all_valid_actions(self) -> None:
        """When all actions are valid, uniform distribution over all."""
        spatial = np.zeros((2, 4, 4), dtype=np.float32)
        vector = np.zeros(3, dtype=np.float32)
        action_mask = np.ones(4, dtype=np.bool_)

        policy, value = _dummy_eval_fn(spatial, vector, action_mask)

        expected = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        np.testing.assert_allclose(policy, expected, atol=1e-6)
        assert value == 0.0

    def test_single_valid_action(self) -> None:
        """When only one action is valid, it gets probability 1.0."""
        spatial = np.zeros((2, 4, 4), dtype=np.float32)
        vector = np.zeros(3, dtype=np.float32)
        action_mask = np.array([False, False, True, False], dtype=np.bool_)

        policy, _ = _dummy_eval_fn(spatial, vector, action_mask)

        expected = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(policy, expected, atol=1e-6)

    def test_no_valid_actions_fallback(self) -> None:
        """When no actions are valid, falls back to uniform over all actions."""
        spatial = np.zeros((2, 4, 4), dtype=np.float32)
        vector = np.zeros(3, dtype=np.float32)
        action_mask = np.array([False, False, False, False], dtype=np.bool_)

        policy, value = _dummy_eval_fn(spatial, vector, action_mask)

        # Fallback: uniform over all 4 actions
        expected = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        np.testing.assert_allclose(policy, expected, atol=1e-6)
        assert value == 0.0

    def test_policy_sums_to_one(self) -> None:
        """Policy output always sums to 1.0 (is a valid distribution)."""
        spatial = np.zeros((6, 8, 8), dtype=np.float32)
        vector = np.zeros(4, dtype=np.float32)

        # Test several mask configurations
        masks = [
            np.array([True, True, True, True, True], dtype=np.bool_),
            np.array([True, False, False, False, False], dtype=np.bool_),
            np.array([True, True, False, True, False], dtype=np.bool_),
            np.array([False, False, False, False, False], dtype=np.bool_),
        ]

        for mask in masks:
            policy, _ = _dummy_eval_fn(spatial, vector, mask)
            assert pytest.approx(policy.sum(), abs=1e-6) == 1.0

    def test_policy_shape_matches_action_mask(self) -> None:
        """Output policy shape matches the action_mask shape."""
        spatial = np.zeros((2, 4, 4), dtype=np.float32)
        vector = np.zeros(3, dtype=np.float32)

        for action_size in [3, 5, 10, 100]:
            action_mask = np.ones(action_size, dtype=np.bool_)
            policy, _ = _dummy_eval_fn(spatial, vector, action_mask)
            assert policy.shape == (action_size,)

    def test_invalid_actions_get_zero_probability(self) -> None:
        """Actions masked as invalid should have zero probability."""
        spatial = np.zeros((2, 4, 4), dtype=np.float32)
        vector = np.zeros(3, dtype=np.float32)
        action_mask = np.array([True, False, True, False, True], dtype=np.bool_)

        policy, _ = _dummy_eval_fn(spatial, vector, action_mask)

        # Invalid actions (indices 1 and 3) should be 0.0
        assert policy[1] == 0.0
        assert policy[3] == 0.0
        # Valid actions (indices 0, 2, 4) should each be 1/3
        np.testing.assert_allclose(policy[0], 1.0 / 3, atol=1e-6)
        np.testing.assert_allclose(policy[2], 1.0 / 3, atol=1e-6)
        np.testing.assert_allclose(policy[4], 1.0 / 3, atol=1e-6)


# ===================================================================
# update_learning_rate tests
# ===================================================================


class TestUpdateLearningRate:
    """Tests for update_learning_rate in dry-run mode."""

    def test_no_op_when_optimizer_is_none(self) -> None:
        """update_learning_rate does not raise when optimizer is None."""
        learner = Learner(_make_config())
        assert learner._optimizer is None
        # Should be a no-op, not raise
        learner.update_learning_rate(0.01)

    def test_no_side_effects(self) -> None:
        """Calling update_learning_rate has no side effects in dry-run mode."""
        learner = Learner(_make_config(learning_rate=0.001))
        learner.update_learning_rate(0.05)
        # Config learning_rate is unchanged (optimizer was None, so nothing to update)
        assert learner._config.learning_rate == 0.001

    def test_multiple_calls_no_error(self) -> None:
        """Multiple calls to update_learning_rate are fine in dry-run mode."""
        learner = Learner(_make_config())
        for lr in [0.1, 0.01, 0.001, 0.0001]:
            learner.update_learning_rate(lr)
        # No exception means success


# ===================================================================
# save_state / load_state tests
# ===================================================================


class TestSaveLoadState:
    """Tests for save_state and load_state in dry-run mode."""

    def test_save_state_logs_warning_no_network(self, caplog: pytest.LogCaptureFixture, tmp_path) -> None:
        """save_state logs a warning when there is no network."""
        learner = Learner(_make_config(), network=None)
        with caplog.at_level(logging.WARNING):
            learner.save_state(str(tmp_path / "test_checkpoint.pt"))
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("cannot save" in msg.lower() for msg in warning_messages)

    def test_load_state_logs_warning_no_network(self, caplog: pytest.LogCaptureFixture, tmp_path) -> None:
        """load_state logs a warning when there is no network."""
        learner = Learner(_make_config(), network=None)
        with caplog.at_level(logging.WARNING):
            learner.load_state(str(tmp_path / "test_checkpoint.pt"))
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("cannot load" in msg.lower() for msg in warning_messages)

    def test_save_state_returns_none(self, tmp_path) -> None:
        """save_state returns None (early return) in dry-run mode."""
        learner = Learner(_make_config())
        result = learner.save_state(str(tmp_path / "nonexistent_path.pt"))
        assert result is None

    def test_load_state_returns_none(self, tmp_path) -> None:
        """load_state returns None (early return) in dry-run mode."""
        learner = Learner(_make_config())
        result = learner.load_state(str(tmp_path / "nonexistent_path.pt"))
        assert result is None

    def test_save_does_not_create_file(self, tmp_path: pytest.TempPathFactory) -> None:
        """save_state should not create any file in dry-run mode."""
        learner = Learner(_make_config())
        checkpoint_path = str(tmp_path / "should_not_exist.pt")  # type: ignore[operator]
        learner.save_state(checkpoint_path)
        import os
        assert not os.path.exists(checkpoint_path)

    def test_load_does_not_raise_for_missing_file(self, tmp_path) -> None:
        """load_state should not raise even if the file does not exist (dry-run)."""
        learner = Learner(_make_config())
        # This path does not exist, but in dry-run mode it should return early
        learner.load_state(str(tmp_path / "definitely_does_not_exist.pt"))


# ===================================================================
# _get_current_lr tests
# ===================================================================


class TestGetCurrentLr:
    """Tests for _get_current_lr in dry-run mode."""

    def test_returns_config_lr_when_no_optimizer(self) -> None:
        """Without an optimizer, returns the config learning rate."""
        config = _make_config(learning_rate=0.005)
        learner = Learner(config)
        assert learner._get_current_lr() == 0.005

    def test_default_lr(self) -> None:
        """Default learning rate is 1e-3."""
        learner = Learner(_make_config())
        assert learner._get_current_lr() == pytest.approx(1e-3)
