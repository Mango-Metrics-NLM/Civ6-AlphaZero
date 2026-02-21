"""Tests for selfplay_worker MuZero dispatch in main().

Verifies that when --mcts_type=muzero is passed, main() correctly
routes to load_muzero_model() + run_muzero_selfplay() instead of the
AlphaZero path. Also includes regression guards for the default path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from alpha_zero.cloud.gcp.selfplay_worker import parse_args

# ---------------------------------------------------------------------------
# parse_args verification
# ---------------------------------------------------------------------------


class TestMuZeroDispatchParseArgs:
    """Verify parse_args routes muzero correctly."""

    def test_mcts_type_muzero_accepted(self) -> None:
        with patch(
            "sys.argv",
            ["prog", "--worker_id=0", "--model_version=1", "--iteration=0", "--mcts_type=muzero"],
        ):
            args = parse_args()
        assert args.mcts_type == "muzero"

    def test_default_mcts_type_is_gumbel(self) -> None:
        with patch(
            "sys.argv",
            ["prog", "--worker_id=0", "--model_version=1", "--iteration=0"],
        ):
            args = parse_args()
        assert args.mcts_type == "gumbel"

    def test_gumbel_mcts_type_preserved(self) -> None:
        """Regression: gumbel still works."""
        with patch(
            "sys.argv",
            ["prog", "--worker_id=0", "--model_version=1", "--iteration=0", "--mcts_type=gumbel"],
        ):
            args = parse_args()
        assert args.mcts_type == "gumbel"


# ---------------------------------------------------------------------------
# main() dispatch routing
# ---------------------------------------------------------------------------


class TestMainMuZeroDispatch:
    """Verify main() dispatches to correct model loader and selfplay function."""

    @patch("alpha_zero.cloud.gcp.selfplay_worker.GCSTrajectoryStorage")
    @patch("alpha_zero.cloud.gcp.selfplay_worker.run_muzero_selfplay")
    @patch("alpha_zero.cloud.gcp.selfplay_worker.load_muzero_model")
    @patch("alpha_zero.cloud.gcp.selfplay_worker.GCPConfig")
    def test_muzero_dispatches_to_muzero_functions(
        self,
        mock_config_cls: MagicMock,
        mock_load_muzero: MagicMock,
        mock_run_muzero: MagicMock,
        mock_traj_storage: MagicMock,
    ) -> None:
        """When mcts_type=muzero, main() should call load_muzero_model and run_muzero_selfplay."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_load_muzero.return_value = mock_model
        mock_run_muzero.return_value = [{"spatial": [], "vector": [], "policies": [], "actions": [], "rewards": [], "algorithm": "muzero"}]
        mock_config_cls.return_value = MagicMock()

        from alpha_zero.cloud.gcp.selfplay_worker import main

        with patch(
            "sys.argv",
            [
                "prog",
                "--worker_id=0",
                "--model_version=1",
                "--iteration=0",
                "--games=1",
                "--mcts_type=muzero",
                "--dry_run",
            ],
        ):
            result = main()

        assert result == 0
        mock_load_muzero.assert_called_once()
        mock_run_muzero.assert_called_once()

    @patch("alpha_zero.cloud.gcp.selfplay_worker.GCSTrajectoryStorage")
    @patch("alpha_zero.cloud.gcp.selfplay_worker.run_selfplay")
    @patch("alpha_zero.cloud.gcp.selfplay_worker.load_model")
    @patch("alpha_zero.cloud.gcp.selfplay_worker.GCPConfig")
    def test_gumbel_dispatches_to_alphazero_functions(
        self,
        mock_config_cls: MagicMock,
        mock_load_model: MagicMock,
        mock_run_selfplay: MagicMock,
        mock_traj_storage: MagicMock,
    ) -> None:
        """Regression: mcts_type=gumbel should use load_model and run_selfplay."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_load_model.return_value = mock_model
        mock_run_selfplay.return_value = [{"spatial": [], "vector": [], "policies": [], "actions": [], "rewards": []}]
        mock_config_cls.return_value = MagicMock()

        from alpha_zero.cloud.gcp.selfplay_worker import main

        with patch(
            "sys.argv",
            [
                "prog",
                "--worker_id=0",
                "--model_version=1",
                "--iteration=0",
                "--games=1",
                "--mcts_type=gumbel",
                "--dry_run",
            ],
        ):
            result = main()

        assert result == 0
        mock_load_model.assert_called_once()
        mock_run_selfplay.assert_called_once()
