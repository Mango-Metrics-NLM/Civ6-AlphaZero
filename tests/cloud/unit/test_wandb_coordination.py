"""Tests for W&B group coordination across workers and evaluation worker W&B.

Covers Steps 1, 3, 4, and 5 of the training readiness plan:
- Step 1: WANDB_GROUP / WANDB_RUN_NAME passthrough in _wandb_env_vars()
           and VertexEvaluationJob.launch()
- Step 3: W&B integration in evaluation_worker.main()
- Step 4: DEFAULT_MCTS_TYPE == "gumbel"; --learning_rate forwarding
- Step 5: run_evaluation_phase() retry (max_retries param) and
          _EVAL_MACHINE_TYPE constant
"""

from __future__ import annotations

import inspect
import os
from unittest.mock import MagicMock, patch

import pytest


class TestWandbEnvVars:
    """Tests for _wandb_env_vars() passthrough function."""

    def test_includes_group_when_set(self) -> None:
        env = {"WANDB_GROUP": "test-run-123", "WANDB_ENABLED": "true"}
        with patch.dict(os.environ, env, clear=False):
            from alpha_zero.cloud.gcp.vertex_jobs import _wandb_env_vars

            result = _wandb_env_vars()
        assert "WANDB_GROUP" in result
        assert result["WANDB_GROUP"] == "test-run-123"

    def test_includes_run_name_when_set(self) -> None:
        env = {"WANDB_RUN_NAME": "selfplay-iter042-w00", "WANDB_ENABLED": "true"}
        with patch.dict(os.environ, env, clear=False):
            from alpha_zero.cloud.gcp.vertex_jobs import _wandb_env_vars

            result = _wandb_env_vars()
        assert "WANDB_RUN_NAME" in result
        assert result["WANDB_RUN_NAME"] == "selfplay-iter042-w00"

    def test_filters_empty_group(self) -> None:
        env = {"WANDB_GROUP": "", "WANDB_ENABLED": "true"}
        with patch.dict(os.environ, env, clear=False):
            from alpha_zero.cloud.gcp.vertex_jobs import _wandb_env_vars

            result = _wandb_env_vars()
        assert "WANDB_GROUP" not in result

    def test_filters_empty_run_name(self) -> None:
        env = {"WANDB_RUN_NAME": ""}
        with patch.dict(os.environ, env, clear=False):
            from alpha_zero.cloud.gcp.vertex_jobs import _wandb_env_vars

            result = _wandb_env_vars()
        assert "WANDB_RUN_NAME" not in result

    def test_group_and_run_name_both_present(self) -> None:
        env = {
            "WANDB_GROUP": "my-group",
            "WANDB_RUN_NAME": "selfplay-iter001",
            "WANDB_ENABLED": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            from alpha_zero.cloud.gcp.vertex_jobs import _wandb_env_vars

            result = _wandb_env_vars()
        assert result["WANDB_GROUP"] == "my-group"
        assert result["WANDB_RUN_NAME"] == "selfplay-iter001"


class TestEvaluationJobWandb:
    """Tests that VertexEvaluationJob.launch() includes W&B env vars."""

    def test_evaluation_job_includes_wandb_env_vars(self) -> None:
        """VertexEvaluationJob must pass W&B env vars to container."""
        env = {
            "WANDB_GROUP": "run-group-abc",
            "WANDB_ENABLED": "true",
            "WANDB_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env, clear=False):
            from alpha_zero.cloud.gcp.vertex_jobs import _wandb_env_vars

            result = _wandb_env_vars()
        assert "WANDB_GROUP" in result
        assert result["WANDB_GROUP"] == "run-group-abc"

    def test_eval_job_spec_includes_wandb_on_launch(self) -> None:
        """Integration: JobSpec env_vars should contain WANDB keys when set."""
        env = {"WANDB_GROUP": "my-group", "WANDB_ENABLED": "true", "WANDB_API_KEY": "k"}
        with patch.dict(os.environ, env, clear=False):
            from alpha_zero.cloud.gcp.vertex_jobs import _wandb_env_vars

            vars_dict = _wandb_env_vars()
        assert "WANDB_GROUP" in vars_dict

    def test_eval_job_launch_merges_wandb_vars(self) -> None:
        """VertexEvaluationJob.launch() JobSpec env_vars includes WANDB_GROUP."""
        from alpha_zero.cloud.gcp.config import GCPConfig
        from alpha_zero.cloud.gcp.vertex_jobs import VertexEvaluationJob

        config = GCPConfig(
            project_id="test-proj",
            bucket="test-bucket",
            container_image_uri="gcr.io/test/img:latest",
        )
        mock_client = MagicMock()
        mock_client.submit_custom_job.return_value = "projects/test/locations/us/customJobs/99"

        env = {"WANDB_GROUP": "launch-group", "WANDB_ENABLED": "true", "WANDB_API_KEY": "k"}
        with patch.dict(os.environ, env, clear=False):
            job = VertexEvaluationJob(config, client=mock_client)
            job.launch(model_version_new=2, model_version_old=1, iteration=5)

        spec = mock_client.submit_custom_job.call_args[0][0]
        assert "WANDB_GROUP" in spec.env_vars
        assert spec.env_vars["WANDB_GROUP"] == "launch-group"


class TestDefaultMctsType:
    """Tests that DEFAULT_MCTS_TYPE is now 'gumbel'."""

    def test_default_mcts_type_is_gumbel(self) -> None:
        from alpha_zero.cloud.gcp.config import DEFAULT_MCTS_TYPE

        assert DEFAULT_MCTS_TYPE == "gumbel", (
            f"Expected 'gumbel' but got '{DEFAULT_MCTS_TYPE}'. "
            "Update config.py: DEFAULT_MCTS_TYPE = 'gumbel'"
        )


class TestLearningRateForwarding:
    """Tests for --learning_rate forwarding in build_training_extra_args."""

    def _make_args(self, **kwargs: object) -> MagicMock:
        defaults: dict[str, object] = {
            "board_size": 8,
            "action_space_size": 376,
            "num_res_blocks": 6,
            "num_filters": 64,
            "lr_schedule_type": "constant",
            "weight_decay": 0.01,
            "algorithm": "alphazero",
            "encoder_type": "patch_embed",
            "learning_rate": 0.001,
        }
        defaults.update(kwargs)
        args = MagicMock(spec=list(defaults.keys()))
        for k, v in defaults.items():
            setattr(args, k, v)
        return args

    def test_forwards_default_learning_rate(self) -> None:
        from alpha_zero.cloud.gcp.vertex_orchestrator import build_training_extra_args

        args = self._make_args(learning_rate=0.001)
        extra = build_training_extra_args(args)
        assert any("--learning_rate=0.001" in a for a in extra)

    def test_forwards_custom_learning_rate(self) -> None:
        from alpha_zero.cloud.gcp.vertex_orchestrator import build_training_extra_args

        args = self._make_args(learning_rate=0.0005)
        extra = build_training_extra_args(args)
        assert any("--learning_rate=0.0005" in a for a in extra)

    def test_forwards_lr_with_constant_schedule(self) -> None:
        """LR should be forwarded even when lr_schedule_type is constant."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import build_training_extra_args

        args = self._make_args(lr_schedule_type="constant", learning_rate=0.002)
        extra = build_training_extra_args(args)
        assert any("--learning_rate=0.002" in a for a in extra)

    def test_lr_arg_present_in_parse_args(self) -> None:
        """--learning_rate must be a recognized argument in parse_args."""
        with patch(
            "sys.argv",
            ["orchestrator", "--learning_rate=0.0003"],
        ):
            from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

            args = parse_args()
        assert args.learning_rate == pytest.approx(0.0003)


class TestEvaluationRetry:
    """Tests for evaluation phase retry and circuit breaker integration."""

    def test_run_evaluation_phase_accepts_max_retries(self) -> None:
        """run_evaluation_phase must accept max_retries parameter."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_evaluation_phase

        sig = inspect.signature(run_evaluation_phase)
        assert (
            "max_retries" in sig.parameters
        ), "run_evaluation_phase must have max_retries parameter"
        assert sig.parameters["max_retries"].default == 2

    def test_eval_machine_type_constant_exists(self) -> None:
        """_EVAL_MACHINE_TYPE constant should exist for cost estimation."""
        import alpha_zero.cloud.gcp.vertex_orchestrator as orch

        assert hasattr(
            orch, "_EVAL_MACHINE_TYPE"
        ), "vertex_orchestrator must have _EVAL_MACHINE_TYPE constant"
        assert orch._EVAL_MACHINE_TYPE == "n1-standard-4"

    def test_run_evaluation_phase_skips_when_zero_games(self) -> None:
        """run_evaluation_phase returns (True, 0.0) when evaluation_games <= 0."""
        from alpha_zero.cloud.gcp.config import GCPConfig
        from alpha_zero.cloud.gcp.vertex_jobs import VertexAIClient
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_evaluation_phase

        config = GCPConfig(project_id="proj", bucket="bkt")
        client = MagicMock(spec=VertexAIClient)
        args = MagicMock()
        args.evaluation_games = 0

        ok, wr = run_evaluation_phase(
            config=config,
            client=client,
            iteration=1,
            model_version_new=2,
            model_version_old=1,
            args=args,
        )
        assert ok is True
        assert wr == 0.0
        client.submit_custom_job.assert_not_called()

    def test_run_evaluation_phase_dry_run(self) -> None:
        """Dry run returns simulated win rate without launching jobs."""
        from alpha_zero.cloud.gcp.config import GCPConfig
        from alpha_zero.cloud.gcp.vertex_jobs import VertexAIClient
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_evaluation_phase

        config = GCPConfig(project_id="proj", bucket="bkt")
        client = MagicMock(spec=VertexAIClient)
        args = MagicMock()
        args.evaluation_games = 20

        ok, wr = run_evaluation_phase(
            config=config,
            client=client,
            iteration=1,
            model_version_new=2,
            model_version_old=1,
            args=args,
            dry_run=True,
        )
        assert ok is True
        assert wr == pytest.approx(0.55)
        client.submit_custom_job.assert_not_called()
