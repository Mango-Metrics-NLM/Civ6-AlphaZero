"""
Unit tests for self-play worker.

Tests the selfplay_worker module used for Vertex AI cloud training.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from alpha_zero.cloud.gcp.config import GCPConfig, GCSPathConfig
from alpha_zero.cloud.gcp.storage import ModelMetadata
from tests.mocks.cloud_mocks import MockStorageBackend

# =============================================================================
# Mock Classes
# =============================================================================


class MockModel(nn.Module):
    """Simple mock model for testing."""

    def __init__(self, action_space_size: int = 100) -> None:
        super().__init__()
        self.linear = nn.Linear(10, action_space_size)
        self.value_head = nn.Linear(10, 1)
        self.action_space_size = action_space_size

    def forward(
        self, spatial: torch.Tensor, vector: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = spatial.shape[0]
        x = torch.randn(batch, 10)
        policy = self.linear(x)
        if action_mask is not None:
            policy = policy.masked_fill(~action_mask.bool(), float("-inf"))
        policy = torch.softmax(policy, dim=-1)
        value = torch.tanh(self.value_head(x)).squeeze(-1)
        return policy, value

    def predict(
        self, spatial: torch.Tensor, vector: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(spatial, vector, action_mask)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_backend() -> MockStorageBackend:
    """Create mock storage backend."""
    return MockStorageBackend()


@pytest.fixture
def gcp_config() -> GCPConfig:
    """Create test GCP config."""
    return GCPConfig(
        project_id="test-project",
        region="us-central1",
        bucket="test-bucket",
        paths=GCSPathConfig(),
    )


@pytest.fixture
def mock_model() -> MockModel:
    """Create mock model for testing."""
    return MockModel(action_space_size=100)


# =============================================================================
# parse_args Tests
# =============================================================================


class TestParseArgs:
    """Tests for argument parsing."""

    def test_required_args_missing(self) -> None:
        """Test that required arguments raise error when missing."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        with patch.object(sys, "argv", ["selfplay_worker"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_required_args_provided(self) -> None:
        """Test parsing with all required arguments."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        test_args = [
            "selfplay_worker",
            "--worker_id=5",
            "--model_version=10",
            "--iteration=20",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.worker_id == 5
        assert args.model_version == 10
        assert args.iteration == 20
        assert args.games == 100  # default
        assert args.mcts_simulations == 400  # default
        assert args.dry_run is False  # default

    def test_parse_args_with_custom_values(self) -> None:
        """Test parsing with custom argument values."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        test_args = [
            "selfplay_worker",
            "--worker_id=1",
            "--model_version=5",
            "--iteration=10",
            "--games=50",
            "--mcts_simulations=100",
            "--dry_run",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.worker_id == 1
        assert args.games == 50
        assert args.mcts_simulations == 100
        assert args.dry_run is True

    def test_parse_args_mcts_config(self) -> None:
        """Test parsing MCTS configuration arguments."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        test_args = [
            "selfplay_worker",
            "--worker_id=1",
            "--model_version=1",
            "--iteration=1",
            "--c_puct=2.0",
            "--dirichlet_alpha=0.5",
            "--temperature=0.5",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.c_puct == 2.0
        assert args.dirichlet_alpha == 0.5
        assert args.temperature == 0.5


# =============================================================================
# load_model Tests
# =============================================================================


class TestLoadModel:
    """Tests for model loading."""

    def test_dry_run_creates_dummy_model(self, gcp_config: GCPConfig) -> None:
        """Test that dry run creates a functional model."""
        from alpha_zero.cloud.gcp.selfplay_worker import load_model

        # dry-run uses 13 spatial channels, 8x8 board, 376 actions (production defaults)
        model = load_model(gcp_config, version=1, dry_run=True)

        assert model is not None
        # Verify model can do forward pass with correct production-default shapes
        spatial = torch.randn(1, 13, 8, 8)
        vector = torch.randn(1, 64)
        mask = torch.ones(1, 376)
        model.eval()
        with torch.no_grad():
            policy, value = model.predict(spatial, vector, mask)

        assert policy.shape == (1, 376)
        assert value.shape == (1,)

    def test_load_model_from_gcs_with_config(
        self, gcp_config: GCPConfig, mock_backend: MockStorageBackend
    ) -> None:
        """Test loading model from GCS with model config."""
        from alpha_zero.cloud.gcp.selfplay_worker import load_model
        from alpha_zero.cloud.gcp.storage import GCSModelStorage
        from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

        # Create a real model and save its state dict
        config = TransformerConfig(
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=100,
        )
        real_model = Civ6TransformerNetwork(config=config)

        # Prepare checkpoint with model config
        import io

        checkpoint = {
            "state_dict": real_model.state_dict(),
            "model_config": {
                "spatial_channels": 13,
                "spatial_size": 8,
                "vector_dim": 64,
                "action_space_size": 100,
            },
        }
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        model_bytes = buffer.getvalue()

        # Pre-populate mock storage
        mock_backend.upload_blob("test-bucket", "models/civ6az/v001/model.pt", model_bytes)

        # Create metadata
        import json

        meta = ModelMetadata(version=1, iteration=10, elo=1500.0)
        meta_bytes = json.dumps(meta.to_dict()).encode()
        mock_backend.upload_blob("test-bucket", "models/civ6az/v001/metadata.json", meta_bytes)

        # Load model
        with patch.object(GCSModelStorage, "__init__", lambda self, cfg: None):
            storage = GCSModelStorage(gcp_config)
            storage.config = gcp_config
            storage.backend = mock_backend
            storage._bucket = gcp_config.bucket
            storage._paths = gcp_config.paths

            with patch(
                "alpha_zero.cloud.gcp.selfplay_worker.GCSModelStorage", return_value=storage
            ):
                model = load_model(gcp_config, version=1, dry_run=False)

        assert model is not None
        assert model.config.spatial_channels == 13
        assert model.config.spatial_size == 8

    def test_load_model_legacy_format(
        self, gcp_config: GCPConfig, mock_backend: MockStorageBackend
    ) -> None:
        """Test loading legacy model without config."""
        from alpha_zero.cloud.gcp.selfplay_worker import load_model
        from alpha_zero.cloud.gcp.storage import GCSModelStorage
        from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

        # Create a model matching pipeline architecture (NetworkArchitectureConfig defaults)
        config = TransformerConfig(
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            num_layers=6,
            embed_dim=64,
        )
        real_model = Civ6TransformerNetwork(config=config)

        import io

        buffer = io.BytesIO()
        torch.save(real_model.state_dict(), buffer)  # Legacy format - just state_dict
        model_bytes = buffer.getvalue()

        mock_backend.upload_blob("test-bucket", "models/civ6az/v001/model.pt", model_bytes)

        import json

        meta = ModelMetadata(version=1, iteration=10)
        meta_bytes = json.dumps(meta.to_dict()).encode()
        mock_backend.upload_blob("test-bucket", "models/civ6az/v001/metadata.json", meta_bytes)

        with patch.object(GCSModelStorage, "__init__", lambda self, cfg: None):
            storage = GCSModelStorage(gcp_config)
            storage.config = gcp_config
            storage.backend = mock_backend
            storage._bucket = gcp_config.bucket
            storage._paths = gcp_config.paths

            with patch(
                "alpha_zero.cloud.gcp.selfplay_worker.GCSModelStorage", return_value=storage
            ):
                # Should load but warn about missing config
                model = load_model(gcp_config, version=1, dry_run=False)

        assert model is not None


# =============================================================================
# run_selfplay Tests
# =============================================================================


class TestRunSelfplay:
    """Tests for self-play execution."""

    def test_dry_run_returns_trajectories(self, mock_model: MockModel) -> None:
        """Test that dry run returns valid trajectory format."""
        from alpha_zero.cloud.gcp.selfplay_worker import run_selfplay

        trajectories = run_selfplay(
            model=mock_model,
            num_games=5,
            mcts_sims=10,
            dry_run=True,
        )

        assert len(trajectories) == 5

        # Verify new format with spatial/vector
        for traj in trajectories:
            assert "spatial" in traj
            assert "vector" in traj
            assert "policies" in traj
            assert "actions" in traj
            assert "players" in traj
            assert "rewards" in traj

            # Verify spatial and vector have same length
            assert len(traj["spatial"]) == len(traj["vector"])
            assert len(traj["spatial"]) == len(traj["policies"])
            assert len(traj["spatial"]) == len(traj["players"])
            assert len(traj["spatial"]) == len(traj["rewards"])

            # Dry-run payloads follow the real reward semantics: signed
            # per player-to-move, drawn from the {-1, 0, +1} alphabet.
            assert all(r in (-1.0, 0.0, 1.0) for r in traj["rewards"])

    def test_dry_run_trajectory_shapes(self, mock_model: MockModel) -> None:
        """Test that dry run trajectory shapes are correct."""
        from alpha_zero.cloud.gcp.selfplay_worker import run_selfplay

        trajectories = run_selfplay(
            model=mock_model,
            num_games=2,
            mcts_sims=10,
            dry_run=True,
        )

        for traj in trajectories:
            # spatial should be flattened to list
            assert isinstance(traj["spatial"][0], list)
            # vector should be flattened to list
            assert isinstance(traj["vector"][0], list)
            # policies should be list
            assert isinstance(traj["policies"][0], list)

    def test_mcts_config_override(self, mock_model: MockModel) -> None:
        """Test that MCTS config override is applied."""
        from alpha_zero.cloud.gcp.selfplay_worker import run_selfplay

        mcts_override = {
            "c_puct": 2.5,
            "dirichlet_alpha": 0.5,
        }

        # Dry run still works with override
        trajectories = run_selfplay(
            model=mock_model,
            num_games=1,
            mcts_sims=10,
            mcts_config_override=mcts_override,
            dry_run=True,
        )

        assert len(trajectories) == 1

    @patch("alpha_zero.envs.microciv.MicroCivEnv")
    @patch("alpha_zero.core.mcts_factory.create_mcts")
    def test_run_selfplay_gumbel_configuration(
        self, mock_create_mcts: MagicMock, mock_env_cls: MagicMock, mock_model: MockModel
    ) -> None:
        """Test that Gumbel MCTS configuration is correctly passed."""
        from alpha_zero.cloud.gcp.selfplay_worker import run_selfplay

        # Mock environment
        mock_env = mock_env_cls.return_value
        mock_env.get_observation.return_value = {
            "spatial": MagicMock(flatten=lambda: MagicMock(tolist=lambda: [])),
            "vector": MagicMock(flatten=lambda: MagicMock(tolist=lambda: [])),
        }
        mock_env.step.return_value = (None, 0.5, True, False, {})

        # Mock MCTS
        mock_mcts = mock_create_mcts.return_value
        mock_mcts.search.return_value = (MagicMock(tolist=lambda: []), {})
        mock_mcts.select_action.return_value = 0

        # Run with Gumbel MCTS and specific num_actions
        run_selfplay(
            model=mock_model,
            num_games=1,
            mcts_sims=50,
            mcts_type="gumbel",
            gumbel_num_actions=8,
            dry_run=False,
        )

        # Verify create_mcts was called with correct parameters
        mock_create_mcts.assert_called_once()
        call_kwargs = mock_create_mcts.call_args[1]

        assert call_kwargs["variant"] == "gumbel"
        assert call_kwargs["num_simulations"] == 50
        assert call_kwargs["num_sampled_actions"] == 8

    @patch("alpha_zero.envs.microciv.MicroCivEnv")
    @patch("alpha_zero.core.mcts_factory.create_mcts")
    def test_run_selfplay_gumbel_defaults(
        self, mock_create_mcts: MagicMock, mock_env_cls: MagicMock, mock_model: MockModel
    ) -> None:
        """Test that Gumbel MCTS defaults to sims if num_actions not provided (legacy behavior)."""
        from alpha_zero.cloud.gcp.selfplay_worker import run_selfplay

        # Mock environment
        mock_env = mock_env_cls.return_value
        mock_env.get_observation.return_value = {
            "spatial": MagicMock(flatten=lambda: MagicMock(tolist=lambda: [])),
            "vector": MagicMock(flatten=lambda: MagicMock(tolist=lambda: [])),
        }
        mock_env.step.return_value = (None, 0.5, True, False, {})

        # Mock MCTS
        mock_mcts = mock_create_mcts.return_value
        mock_mcts.search.return_value = (MagicMock(tolist=lambda: []), {})
        mock_mcts.select_action.return_value = 0

        # Run with Gumbel MCTS but without explicit gumbel_num_actions (using function default 16)
        run_selfplay(
            model=mock_model,
            num_games=1,
            mcts_sims=50,
            mcts_type="gumbel",
            # gumbel_num_actions defaults to 16
            dry_run=False,
        )

        # Verify create_mcts was called with default 16
        mock_create_mcts.assert_called_once()
        call_args = mock_create_mcts.call_args[1]
        assert call_args["variant"] == "gumbel"
        assert call_args["num_sampled_actions"] == 16


# =============================================================================
# save_trajectories Tests
# =============================================================================


class TestSaveTrajectories:
    """Tests for trajectory saving."""

    def test_dry_run_returns_none(self, gcp_config: GCPConfig) -> None:
        """Test that dry run doesn't save and returns None."""
        from alpha_zero.cloud.gcp.selfplay_worker import save_trajectories

        result = save_trajectories(
            config=gcp_config,
            trajectories=[{"spatial": [], "vector": [], "policies": [], "rewards": []}],
            iteration=1,
            worker_id=0,
            model_version=1,
            dry_run=True,
        )

        assert result is None

    def test_save_calls_storage(
        self, gcp_config: GCPConfig, mock_backend: MockStorageBackend
    ) -> None:
        """Test that save uses GCS storage correctly."""
        from alpha_zero.cloud.gcp.selfplay_worker import save_trajectories
        from alpha_zero.cloud.gcp.storage import GCSTrajectoryStorage

        mock_storage = MagicMock(spec=GCSTrajectoryStorage)
        mock_storage.save_trajectories.return_value = "gs://test-bucket/path"

        with patch(
            "alpha_zero.cloud.gcp.selfplay_worker.GCSTrajectoryStorage",
            return_value=mock_storage,
        ):
            result = save_trajectories(
                config=gcp_config,
                trajectories=[
                    {"spatial": [[0.0]], "vector": [[0.0]], "policies": [[0.1]], "rewards": [0.5]}
                ],
                iteration=5,
                worker_id=2,
                model_version=3,
                dry_run=False,
            )

        assert result == "gs://test-bucket/path"
        mock_storage.save_trajectories.assert_called_once()


# =============================================================================
# main Tests
# =============================================================================


class TestMain:
    """Tests for main entry point."""

    def test_main_dry_run_success(self) -> None:
        """Test that dry run completes successfully."""
        from alpha_zero.cloud.gcp.selfplay_worker import main

        test_args = [
            "selfplay_worker",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
            "--games=2",
            "--mcts_simulations=10",
            "--dry_run",
        ]

        with patch.object(sys, "argv", test_args):
            exit_code = main()

        assert exit_code == 0

    def test_main_keyboard_interrupt(self) -> None:
        """Test that KeyboardInterrupt returns code 130."""
        from alpha_zero.cloud.gcp.selfplay_worker import main

        test_args = [
            "selfplay_worker",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
            "--dry_run",
        ]

        with patch.object(sys, "argv", test_args):
            with patch(
                "alpha_zero.cloud.gcp.selfplay_worker.load_model",
                side_effect=KeyboardInterrupt(),
            ):
                exit_code = main()

        assert exit_code == 130

    def test_main_exception(self) -> None:
        """Test that unhandled exception returns code 1."""
        from alpha_zero.cloud.gcp.selfplay_worker import main

        test_args = [
            "selfplay_worker",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
            "--dry_run",
        ]

        with patch.object(sys, "argv", test_args):
            with patch(
                "alpha_zero.cloud.gcp.selfplay_worker.load_model",
                side_effect=RuntimeError("Test error"),
            ):
                exit_code = main()

        assert exit_code == 1


# =============================================================================
# action_mask Storage Tests
# =============================================================================


class TestSelfplayActionMaskStorage:
    """Tests that selfplay trajectories include action_masks."""

    def test_new_trajectory_declares_action_masks(self) -> None:
        """Every self-play path starts from a trajectory that has the field.

        This used to grep ``run_selfplay``'s source text for the string
        "action_masks" -- a check that could pass while the value was never
        stored, and that broke as soon as the append moved behind a shared
        helper. All three paths now build their trajectory here, so asserting
        on the constructor covers each of them behaviorally.
        """
        from alpha_zero.cloud.gcp.selfplay_worker import new_trajectory

        assert "action_masks" in new_trajectory()
        assert "action_masks" in new_trajectory(algorithm="muzero")

    def test_append_step_stores_the_mask(self) -> None:
        """A committed step records the observation's action mask verbatim."""
        import numpy as np

        from alpha_zero.cloud.gcp.selfplay_worker import append_step, new_trajectory

        trajectory = new_trajectory()
        mask = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        append_step(
            trajectory,
            observation={
                "spatial": np.zeros((1, 2, 2), dtype=np.float32),
                "vector": np.zeros(3, dtype=np.float32),
                "action_mask": mask,
            },
            policy=np.array([0.5, 0.0, 0.5], dtype=np.float32),
            action=0,
            player=1,
        )

        assert trajectory["action_masks"] == [[1.0, 0.0, 1.0]]
        # Every per-step field advances together, so nothing can go ragged.
        assert {
            len(trajectory[f])
            for f in ("spatial", "vector", "action_masks", "policies", "actions", "players")
        } == {1}

    def test_dry_run_trajectories_carry_masks(self) -> None:
        """Dry runs must produce the same shape as real self-play.

        A dry-run trajectory previously carried a 100-wide policy against a
        376-wide action space and no mask at all.
        """
        from alpha_zero.cloud.gcp.config import NetworkArchitectureConfig
        from alpha_zero.cloud.gcp.selfplay_worker import dummy_trajectory

        expected_width = NetworkArchitectureConfig().action_space_size
        for algorithm in (None, "muzero"):
            traj = dummy_trajectory(board_size=8, algorithm=algorithm)
            assert len(traj["action_masks"][0]) == expected_width
            assert len(traj["policies"][0]) == expected_width
            assert len(traj["action_masks"]) == len(traj["spatial"])
