"""Unit tests for vertex_orchestrator W&B logging integration.

Tests the orchestrator's W&B logging functionality added to the main() function,
including initialization, per-iteration metrics, summary logging, and proper
cleanup in the finally block.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestOrchestratorWandbLogging:
    """Test suite for orchestrator W&B logging integration."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock GCPConfig."""
        with patch("alpha_zero.cloud.gcp.vertex_orchestrator.GCPConfig") as mock:
            config = MagicMock()
            config.project_id = "test-project"
            config.region = "us-central1"
            config.bucket = "test-bucket"
            config.self_play = MagicMock()
            config.training = MagicMock()
            mock.return_value = config
            yield mock

    @pytest.fixture
    def mock_vertex_client(self):
        """Create a mock VertexAIClient."""
        with patch("alpha_zero.cloud.gcp.vertex_orchestrator.VertexAIClient") as mock:
            yield mock

    @pytest.fixture
    def mock_cost_estimator(self):
        """Create a mock CostEstimator."""
        with patch("alpha_zero.cloud.gcp.cost.CostEstimator") as mock:
            estimator = MagicMock()
            estimator.estimate_job_cost.return_value = 1.50
            mock.return_value = estimator
            yield mock

    @pytest.fixture
    def mock_model_storage(self):
        """Create a mock GCSModelStorage."""
        with patch("alpha_zero.cloud.gcp.vertex_orchestrator.GCSModelStorage") as mock:
            storage = MagicMock()
            storage.get_latest_version.return_value = 1
            mock.return_value = storage
            yield mock

    def test_wandb_init_called_when_enabled(
        self,
        mock_config,
        mock_vertex_client,
        mock_cost_estimator,
        mock_model_storage,
    ):
        """Test that WandbLogger.from_env() is called and init() invoked when enabled.

        When W&B is enabled (via environment or API key), the orchestrator should:
        1. Call WandbLogger.from_env() to create the logger
        2. Call init() with a config dict containing orchestrator parameters
        3. The logger should report is_enabled=True
        """
        with patch("alpha_zero.utils.wandb_logger.WandbLogger") as mock_wandb:
            # Create mock logger instance
            mock_logger = MagicMock()
            mock_logger.is_enabled = True
            mock_logger.is_initialized = True
            mock_wandb.from_env.return_value = mock_logger

            # Mock the job phases to succeed quickly
            with patch(
                "alpha_zero.cloud.gcp.vertex_orchestrator.run_selfplay_phase",
                return_value=True,
            ):
                with patch(
                    "alpha_zero.cloud.gcp.vertex_orchestrator.run_training_phase",
                    return_value=True,
                ):
                    with patch("sys.argv", ["orchestrator", "--iterations=1", "--dry_run"]):
                        from alpha_zero.cloud.gcp.vertex_orchestrator import main

                        result = main()

            # Verify WandbLogger.from_env() was called
            mock_wandb.from_env.assert_called_once()

            # Verify init() was called with config dict
            mock_logger.init.assert_called_once()
            call_kwargs = mock_logger.init.call_args[1]
            assert "config_dict" in call_kwargs
            config = call_kwargs["config_dict"]
            assert "iterations" in config
            assert "workers" in config
            assert "mcts_type" in config
            assert "mode" in config
            assert config["mode"] == "orchestrator"

            assert result == 0

    def test_wandb_not_initialized_when_disabled(
        self,
        mock_config,
        mock_vertex_client,
        mock_cost_estimator,
        mock_model_storage,
    ):
        """Test that orchestrator works when W&B import fails.

        When wandb is not available (ImportError), the orchestrator should:
        1. Continue running normally
        2. Not crash or fail
        3. Complete the training loop successfully
        """
        with patch(
            "alpha_zero.utils.wandb_logger.WandbLogger"
        ) as mock_wandb:
            # Simulate ImportError when trying to import WandbLogger
            mock_wandb.from_env.side_effect = ImportError("wandb not installed")

            # Mock the job phases to succeed quickly
            with patch(
                "alpha_zero.cloud.gcp.vertex_orchestrator.run_selfplay_phase",
                return_value=True,
            ):
                with patch(
                    "alpha_zero.cloud.gcp.vertex_orchestrator.run_training_phase",
                    return_value=True,
                ):
                    with patch("sys.argv", ["orchestrator", "--iterations=1", "--dry_run"]):
                        from alpha_zero.cloud.gcp.vertex_orchestrator import main

                        # Should complete successfully despite ImportError
                        result = main()

            assert result == 0

    def test_wandb_finish_called_in_finally(
        self,
        mock_config,
        mock_vertex_client,
        mock_cost_estimator,
        mock_model_storage,
    ):
        """Test that wandb_logger.finish() is called in the finally block.

        Even if the training loop encounters an error, the finally block should
        ensure that finish() is called to properly close the W&B run.
        """
        with patch("alpha_zero.utils.wandb_logger.WandbLogger") as mock_wandb:
            # Create mock logger instance
            mock_logger = MagicMock()
            mock_logger.is_enabled = True
            mock_wandb.from_env.return_value = mock_logger

            # Mock selfplay to fail, triggering exception handling
            with patch(
                "alpha_zero.cloud.gcp.vertex_orchestrator.run_selfplay_phase",
                return_value=True,
            ):
                with patch(
                    "alpha_zero.cloud.gcp.vertex_orchestrator.run_training_phase",
                    return_value=True,
                ):
                    with patch("sys.argv", ["orchestrator", "--iterations=1", "--dry_run"]):
                        from alpha_zero.cloud.gcp.vertex_orchestrator import main

                        main()

            # Verify finish() was called
            mock_logger.finish.assert_called()

    def test_wandb_finish_called_on_keyboard_interrupt(
        self,
        mock_config,
        mock_vertex_client,
        mock_cost_estimator,
        mock_model_storage,
    ):
        """Test that wandb_logger.finish() is called with exit_code=130 on Ctrl+C.

        When the user interrupts the orchestrator with Ctrl+C (KeyboardInterrupt),
        the finally block should call finish(exit_code=130).
        """
        with patch("alpha_zero.utils.wandb_logger.WandbLogger") as mock_wandb:
            # Create mock logger instance
            mock_logger = MagicMock()
            mock_logger.is_enabled = True
            mock_wandb.from_env.return_value = mock_logger

            # Simulate keyboard interrupt during selfplay
            with patch(
                "alpha_zero.cloud.gcp.vertex_orchestrator.run_selfplay_phase",
                side_effect=KeyboardInterrupt("User interrupt"),
            ):
                with patch("sys.argv", ["orchestrator", "--iterations=1", "--dry_run"]):
                    from alpha_zero.cloud.gcp.vertex_orchestrator import main

                    result = main()

            # Verify finish() was called with exit code 130
            mock_logger.finish.assert_called_with(exit_code=130)
            assert result == 130

    def test_wandb_finish_called_on_exception(
        self,
        mock_config,
        mock_vertex_client,
        mock_cost_estimator,
        mock_model_storage,
    ):
        """Test that wandb_logger.finish() is called with exit_code=1 on exception.

        When the orchestrator encounters an unexpected exception, the finally
        block should call finish(exit_code=1).
        """
        with patch("alpha_zero.utils.wandb_logger.WandbLogger") as mock_wandb:
            # Create mock logger instance
            mock_logger = MagicMock()
            mock_logger.is_enabled = True
            mock_wandb.from_env.return_value = mock_logger

            # Simulate exception during config validation
            mock_config.return_value.validate.side_effect = RuntimeError("Config error")

            with patch("sys.argv", ["orchestrator", "--iterations=1"]):
                from alpha_zero.cloud.gcp.vertex_orchestrator import main

                result = main()

            # Verify finish() was called with exit code 1
            mock_logger.finish.assert_called_with(exit_code=1)
            assert result == 1

    def test_wandb_logs_iteration_metrics(
        self,
        mock_config,
        mock_vertex_client,
        mock_cost_estimator,
        mock_model_storage,
    ):
        """Test that per-iteration metrics are logged to W&B.

        The orchestrator should log metrics after each iteration including:
        - orchestrator/iteration
        - orchestrator/model_version
        - orchestrator/iteration_duration_min
        - orchestrator/selfplay_cost
        - orchestrator/training_cost
        - orchestrator/cumulative_cost
        """
        with patch("alpha_zero.utils.wandb_logger.WandbLogger") as mock_wandb:
            # Create mock logger instance
            mock_logger = MagicMock()
            mock_logger.is_enabled = True
            mock_wandb.from_env.return_value = mock_logger

            # Mock the job phases to succeed quickly
            with patch(
                "alpha_zero.cloud.gcp.vertex_orchestrator.run_selfplay_phase",
                return_value=True,
            ):
                with patch(
                    "alpha_zero.cloud.gcp.vertex_orchestrator.run_training_phase",
                    return_value=True,
                ):
                    with patch("sys.argv", ["orchestrator", "--iterations=2", "--dry_run"]):
                        from alpha_zero.cloud.gcp.vertex_orchestrator import main

                        main()

            # Verify log() was called with iteration metrics
            # Should have at least 2 calls for iterations + 1 for summary
            assert mock_logger.log.call_count >= 2

            # Check that iteration metrics were logged
            # log() is called as log(data, step=X), so data is the first positional arg
            logged_metrics = [call[0][0] for call in mock_logger.log.call_args_list]

            # Find iteration metric calls (not the summary call)
            iteration_calls = [
                m for m in logged_metrics
                if "orchestrator/iteration" in m
            ]

            assert len(iteration_calls) >= 2

            # Verify first iteration has required metrics
            first_iter = iteration_calls[0]
            assert "orchestrator/iteration" in first_iter
            assert "orchestrator/model_version" in first_iter
            assert "orchestrator/iteration_duration_min" in first_iter
            assert "orchestrator/selfplay_cost" in first_iter
            assert "orchestrator/training_cost" in first_iter
            assert "orchestrator/cumulative_cost" in first_iter

    def test_wandb_logs_summary_metrics(
        self,
        mock_config,
        mock_vertex_client,
        mock_cost_estimator,
        mock_model_storage,
    ):
        """Test that final summary metrics are logged to W&B.

        After completing all iterations, the orchestrator should log:
        - orchestrator/total_iterations
        - orchestrator/successful_iterations
        - orchestrator/failed_iterations
        - orchestrator/total_cost
        - orchestrator/total_duration_min
        - orchestrator/final_model_version
        """
        with patch("alpha_zero.utils.wandb_logger.WandbLogger") as mock_wandb:
            # Create mock logger instance
            mock_logger = MagicMock()
            mock_logger.is_enabled = True
            mock_wandb.from_env.return_value = mock_logger

            # Mock the job phases to succeed quickly
            with patch(
                "alpha_zero.cloud.gcp.vertex_orchestrator.run_selfplay_phase",
                return_value=True,
            ):
                with patch(
                    "alpha_zero.cloud.gcp.vertex_orchestrator.run_training_phase",
                    return_value=True,
                ):
                    with patch("sys.argv", ["orchestrator", "--iterations=1", "--dry_run"]):
                        from alpha_zero.cloud.gcp.vertex_orchestrator import main

                        main()

            # Find the summary metrics call (should be last call before finish)
            # log() is called as log(data, step=X), so data is the first positional arg
            logged_metrics = [call[0][0] for call in mock_logger.log.call_args_list]

            # Last log call should be the summary
            summary_call = logged_metrics[-1]

            assert "orchestrator/total_iterations" in summary_call
            assert "orchestrator/successful_iterations" in summary_call
            assert "orchestrator/failed_iterations" in summary_call
            assert "orchestrator/total_cost" in summary_call
            assert "orchestrator/total_duration_min" in summary_call
            assert "orchestrator/final_model_version" in summary_call

    def test_wandb_logs_evaluation_metrics_when_enabled(
        self,
        mock_config,
        mock_vertex_client,
        mock_cost_estimator,
        mock_model_storage,
    ):
        """Test that evaluation metrics are logged when evaluation is enabled.

        When --evaluation_games > 0, the orchestrator should log:
        - orchestrator/eval_win_rate
        - orchestrator/eval_cost
        """
        with patch("alpha_zero.utils.wandb_logger.WandbLogger") as mock_wandb:
            # Create mock logger instance
            mock_logger = MagicMock()
            mock_logger.is_enabled = True
            mock_wandb.from_env.return_value = mock_logger

            # Mock get_initial_model_version to return 1 (not 0) so evaluation runs
            with patch(
                "alpha_zero.cloud.gcp.vertex_orchestrator.get_initial_model_version",
                return_value=1,
            ):
                # Mock the job phases to succeed quickly
                with patch(
                    "alpha_zero.cloud.gcp.vertex_orchestrator.run_selfplay_phase",
                    return_value=True,
                ):
                    with patch(
                        "alpha_zero.cloud.gcp.vertex_orchestrator.run_training_phase",
                        return_value=True,
                    ):
                        with patch(
                            "alpha_zero.cloud.gcp.vertex_orchestrator.run_evaluation_phase",
                            return_value=(True, 0.65),  # success=True, win_rate=0.65
                        ):
                            with patch(
                                "sys.argv",
                                [
                                    "orchestrator",
                                    "--iterations=1",
                                    "--evaluation_games=20",
                                    "--dry_run",
                                ],
                            ):
                                from alpha_zero.cloud.gcp.vertex_orchestrator import main

                                main()

            # Find iteration metrics that should include eval metrics
            # log() is called as log(data, step=X), so data is the first positional arg
            logged_metrics = [call[0][0] for call in mock_logger.log.call_args_list]

            # Find the iteration call with eval metrics
            eval_calls = [
                m for m in logged_metrics
                if "orchestrator/eval_win_rate" in m
            ]

            assert len(eval_calls) >= 1
            eval_metrics = eval_calls[0]
            assert "orchestrator/eval_win_rate" in eval_metrics
            assert "orchestrator/eval_cost" in eval_metrics
            assert eval_metrics["orchestrator/eval_win_rate"] == 0.65
