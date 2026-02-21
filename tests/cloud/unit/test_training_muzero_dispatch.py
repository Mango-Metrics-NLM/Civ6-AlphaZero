"""Tests for training_worker MuZero dispatch in main().

Verifies that when --algorithm=muzero is passed, main() correctly
routes to create_muzero_model() and the MuZero training path instead
of the AlphaZero path. Also includes regression guards for default behaviour.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from alpha_zero.cloud.gcp.training_worker import parse_args

# ---------------------------------------------------------------------------
# parse_args verification
# ---------------------------------------------------------------------------


class TestMuZeroTrainingDispatchParseArgs:
    """Verify parse_args routes muzero correctly."""

    def test_algorithm_muzero_accepted(self) -> None:
        with patch(
            "sys.argv",
            [
                "prog",
                "--iteration=0",
                "--trajectory_iterations=0",
                "--algorithm=muzero",
            ],
        ):
            args = parse_args()
        assert args.algorithm == "muzero"

    def test_default_algorithm_is_alphazero(self) -> None:
        with patch(
            "sys.argv",
            ["prog", "--iteration=0", "--trajectory_iterations=0"],
        ):
            args = parse_args()
        assert args.algorithm == "alphazero"

    def test_muzero_specific_args_accepted(self) -> None:
        """MuZero-specific training args should be parsed."""
        with patch(
            "sys.argv",
            [
                "prog",
                "--iteration=0",
                "--trajectory_iterations=0",
                "--algorithm=muzero",
                "--muzero_unroll_steps=3",
                "--muzero_td_steps=5",
                "--muzero_discount=0.99",
            ],
        ):
            args = parse_args()
        assert args.muzero_unroll_steps == 3
        assert args.muzero_td_steps == 5
        assert args.muzero_discount == 0.99


# ---------------------------------------------------------------------------
# main() dispatch routing
# ---------------------------------------------------------------------------


class TestMainMuZeroTrainingDispatch:
    """Verify main() dispatches to the correct model creator based on --algorithm."""

    @patch("alpha_zero.cloud.gcp.training_worker.save_final_model")
    @patch("alpha_zero.cloud.gcp.training_worker.train_muzero_epoch")
    @patch("alpha_zero.cloud.gcp.training_worker.create_muzero_model")
    @patch("alpha_zero.cloud.gcp.training_worker.prepare_dataset")
    @patch("alpha_zero.cloud.gcp.training_worker.load_trajectories")
    @patch("alpha_zero.cloud.gcp.training_worker.GCPConfig")
    def test_muzero_dispatches_to_muzero_model(
        self,
        mock_config_cls: MagicMock,
        mock_load_traj: MagicMock,
        mock_prepare: MagicMock,
        mock_create_muzero: MagicMock,
        mock_train_muzero: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """When algorithm=muzero, main() should call create_muzero_model."""
        import torch

        # Setup mocks
        mock_config_cls.return_value = MagicMock()
        mock_load_traj.return_value = [
            {"spatial": [[0.0] * 832], "vector": [[0.0] * 64], "policies": [[0.01] * 376], "actions": [0], "rewards": [1.0]}
        ]

        # Create minimal dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_prepare.return_value = mock_dataset

        # Model mock
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = [torch.zeros(2, requires_grad=True)]
        mock_create_muzero.return_value = mock_model

        mock_train_muzero.return_value = {"loss": 0.5}

        from alpha_zero.cloud.gcp.training_worker import main

        with patch(
            "sys.argv",
            [
                "prog",
                "--iteration=0",
                "--trajectory_iterations=0",
                "--algorithm=muzero",
                "--dry_run",
                "--epochs=1",
            ],
        ):
            main()

        mock_create_muzero.assert_called_once()

    @patch("alpha_zero.cloud.gcp.training_worker.save_final_model")
    @patch("alpha_zero.cloud.gcp.training_worker.create_model")
    @patch("alpha_zero.cloud.gcp.training_worker.prepare_dataset")
    @patch("alpha_zero.cloud.gcp.training_worker.load_trajectories")
    @patch("alpha_zero.cloud.gcp.training_worker.GCPConfig")
    def test_alphazero_dispatches_to_standard_model(
        self,
        mock_config_cls: MagicMock,
        mock_load_traj: MagicMock,
        mock_prepare: MagicMock,
        mock_create_model: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """Regression: default algorithm should use create_model (AlphaZero)."""
        import torch

        mock_config_cls.return_value = MagicMock()
        mock_load_traj.return_value = [
            {"spatial": [[0.0] * 832], "vector": [[0.0] * 64], "policies": [[0.01] * 376], "actions": [0], "rewards": [1.0]}
        ]
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_prepare.return_value = mock_dataset

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = [torch.zeros(2, requires_grad=True)]
        mock_create_model.return_value = mock_model

        from alpha_zero.cloud.gcp.training_worker import main

        with patch(
            "sys.argv",
            [
                "prog",
                "--iteration=0",
                "--trajectory_iterations=0",
                "--dry_run",
                "--epochs=1",
            ],
        ):
            main()

        mock_create_model.assert_called_once()
