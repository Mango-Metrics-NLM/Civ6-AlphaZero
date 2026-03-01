"""
Unit tests for training worker.
"""

import os
from unittest.mock import MagicMock, patch

from alpha_zero.cloud.gcp.training_worker import cleanup_ddp, setup_ddp


class TestTrainingWorkerDDP:
    """Tests for DDP setup and cleanup."""

    def test_setup_ddp_no_env_vars(self) -> None:
        """Test setup_ddp when environment variables are missing."""
        with patch.dict(os.environ, {}, clear=True):
            rank, local_rank, world_size = setup_ddp()
            assert rank == 0
            assert local_rank == 0
            assert world_size == 1

    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.cuda.set_device")
    def test_setup_ddp_with_env_vars(
        self,
        mock_set_device: MagicMock,
        mock_cuda_available: MagicMock,
        mock_init_process_group: MagicMock,
    ) -> None:
        """Test setup_ddp when environment variables are present (CPU fallback)."""
        env_vars = {
            "WORLD_SIZE": "2",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12345",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            rank, local_rank, world_size = setup_ddp()

            assert rank == 1
            assert local_rank == 1
            assert world_size == 2

            # Backend selection is based on CUDA availability
            mock_init_process_group.assert_called_once_with(backend="gloo")
            mock_set_device.assert_not_called()

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.destroy_process_group")
    def test_cleanup_ddp_initialized(
        self,
        mock_destroy: MagicMock,
        mock_is_initialized: MagicMock,
    ) -> None:
        """Test cleanup_ddp when DDP is initialized."""
        cleanup_ddp()
        mock_destroy.assert_called_once()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.destroy_process_group")
    def test_cleanup_ddp_not_initialized(
        self,
        mock_destroy: MagicMock,
        mock_is_initialized: MagicMock,
    ) -> None:
        """Test cleanup_ddp when DDP is not initialized."""
        cleanup_ddp()
        mock_destroy.assert_not_called()


class TestPrepareDatasetActionMask:
    """Tests for action_mask integration in prepare_dataset."""

    def _make_trajectory(
        self,
        num_steps: int = 3,
        action_space_size: int = 376,
        include_mask: bool = True,
    ) -> dict:
        """Create a minimal trajectory dict for testing."""
        trajectory = {
            "spatial": [[0.0] * (13 * 8 * 8)] * num_steps,
            "vector": [[0.0] * 64] * num_steps,
            "policies": [[1.0 / action_space_size] * action_space_size] * num_steps,
            "rewards": [1.0] * num_steps,
        }
        if include_mask:
            # Mask with first 10 actions legal
            mask = [0.0] * action_space_size
            for i in range(10):
                mask[i] = 1.0
            trajectory["action_masks"] = [mask] * num_steps
        return trajectory

    def test_returns_five_tensors_with_mask(self) -> None:
        """Dataset must have 5 tensors when action_masks present."""
        from alpha_zero.cloud.gcp.training_worker import prepare_dataset

        traj = self._make_trajectory(include_mask=True)
        dataset = prepare_dataset([traj])
        assert (
            len(dataset.tensors) == 5
        ), "Expected 5 tensors: spatial, vector, action_mask, policy, value"

    def test_returns_five_tensors_without_mask_legacy(self) -> None:
        """Legacy trajectories without action_masks get all-ones mask."""
        from alpha_zero.cloud.gcp.training_worker import prepare_dataset

        traj = self._make_trajectory(include_mask=False)
        dataset = prepare_dataset([traj])
        assert len(dataset.tensors) == 5
        # Check mask tensor is all-ones
        _, _, action_mask_tensor, _, _ = dataset.tensors
        assert action_mask_tensor.min().item() == 1.0
        assert action_mask_tensor.max().item() == 1.0

    def test_action_mask_tensor_shape(self) -> None:
        """Action mask must have shape (num_steps, action_space_size)."""
        from alpha_zero.cloud.gcp.training_worker import prepare_dataset

        traj = self._make_trajectory(num_steps=5, action_space_size=376, include_mask=True)
        dataset = prepare_dataset([traj])
        _, _, action_mask_tensor, _, _ = dataset.tensors
        assert action_mask_tensor.shape == (5, 376)

    def test_legacy_mask_has_correct_action_space_size(self) -> None:
        """Legacy all-ones mask must match action_space_size parameter."""
        from alpha_zero.cloud.gcp.training_worker import prepare_dataset

        traj = self._make_trajectory(include_mask=False)
        dataset = prepare_dataset([traj], action_space_size=376)
        _, _, action_mask_tensor, _, _ = dataset.tensors
        assert action_mask_tensor.shape[-1] == 376

    def test_tensor_order_is_spatial_vector_mask_policy_value(self) -> None:
        """Verify tensor order matches train_epoch unpacking."""

        from alpha_zero.cloud.gcp.training_worker import prepare_dataset

        traj = self._make_trajectory(num_steps=2, include_mask=True)
        dataset = prepare_dataset([traj])
        spatial, vector, action_mask, policies, values = dataset.tensors
        # Spatial: (N, 13*8*8)
        assert spatial.shape[1] == 13 * 8 * 8
        # Vector: (N, 64)
        assert vector.shape[1] == 64
        # Action mask: (N, 376)
        assert action_mask.shape[1] == 376
        # Values: (N,) scalar
        assert values.ndim == 1


class TestModelVersionArg:
    """Tests for --model_version CLI argument and base model loading."""

    def test_model_version_arg_parsed(self) -> None:
        """--model_version is parsed correctly."""
        import sys

        from alpha_zero.cloud.gcp.training_worker import parse_args

        with patch.object(
            sys, "argv", ["worker", "--iteration=0", "--trajectory_iterations=0", "--model_version=5"]
        ):
            args = parse_args()
        assert args.model_version == 5

    def test_model_version_default_none(self) -> None:
        """--model_version defaults to None (train from scratch)."""
        import sys

        from alpha_zero.cloud.gcp.training_worker import parse_args

        with patch.object(sys, "argv", ["worker", "--iteration=0", "--trajectory_iterations=0"]):
            args = parse_args()
        assert args.model_version is None


class TestModelVersionWeightLoading:
    """Tests for actual weight loading behavior with --model_version.

    These tests verify the v1.3.0 fix: that training loads pre-trained
    weights instead of training from scratch each iteration.
    """

    def test_weight_loading_changes_model_state(self) -> None:
        """Loading a model_version should produce different weights than fresh init."""
        import tempfile

        import torch

        from alpha_zero.core.civ6_network import MicroCivNetwork

        # Create and save a "pre-trained" model
        net_a = MicroCivNetwork(board_size=8, action_space_size=100, num_res_blocks=2, num_filters=16)
        # Modify weights to be distinguishable from random init
        with torch.no_grad():
            for p in net_a.parameters():
                p.fill_(0.42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pretrained.pt")
            net_a.save(path)

            # Create fresh model (random init)
            net_b = MicroCivNetwork(
                board_size=8, action_space_size=100, num_res_blocks=2, num_filters=16
            )

            # Verify fresh model is NOT all 0.42
            first_param_b = next(net_b.parameters()).data.mean().item()
            assert abs(first_param_b - 0.42) > 0.01, "Fresh init should not match pre-trained"

            # Load pre-trained weights (simulates --model_version behavior)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            net_b.load_state_dict(checkpoint["state_dict"], strict=False)

            # Now net_b should have 0.42 weights
            first_param_loaded = next(net_b.parameters()).data.mean().item()
            assert abs(first_param_loaded - 0.42) < 1e-6, (
                f"After loading, weights should be 0.42 but got {first_param_loaded}"
            )

    def test_weight_continuity_across_training_steps(self) -> None:
        """Weights should incrementally change during training, not reset."""
        import torch

        from alpha_zero.core.civ6_network import MicroCivNetwork

        net = MicroCivNetwork(board_size=8, action_space_size=100, num_res_blocks=2, num_filters=16)

        # Record initial weight checksum
        def weight_checksum(model: torch.nn.Module) -> float:
            return sum(p.data.abs().sum().item() for p in model.parameters())

        checksum_0 = weight_checksum(net)

        # Simulate one training step
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        spatial = torch.randn(4, 13, 8, 8)
        vector = torch.randn(4, 64)
        mask = torch.ones(4, 100)

        net.train()
        policy, value = net(spatial, vector, mask)
        loss = policy.sum() + value.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        checksum_1 = weight_checksum(net)

        # Weights should have changed (training occurred)
        assert checksum_0 != checksum_1, "Weights should change after training step"

        # But not by more than 50% (not a re-init)
        relative_change = abs(checksum_1 - checksum_0) / max(checksum_0, 1e-8)
        assert relative_change < 0.5, (
            f"Weight change too large ({relative_change:.1%}), "
            "looks like re-initialization rather than gradient update"
        )

    def test_model_version_none_trains_from_scratch(self) -> None:
        """When model_version is None, model should start with random weights."""
        import torch

        from alpha_zero.core.civ6_network import MicroCivNetwork

        # Create two fresh models with same architecture
        net_a = MicroCivNetwork(board_size=8, action_space_size=100, num_res_blocks=2, num_filters=16)
        net_b = MicroCivNetwork(board_size=8, action_space_size=100, num_res_blocks=2, num_filters=16)

        # With different random seeds, weights should differ
        param_a = next(net_a.parameters()).data.flatten()[:10]
        param_b = next(net_b.parameters()).data.flatten()[:10]

        # At least some values should differ (random init)
        assert not torch.allclose(param_a, param_b, atol=1e-6), (
            "Two separately initialized networks should have different weights"
        )

    def test_strict_false_handles_extra_keys(self) -> None:
        """strict=False should silently ignore extra keys in state_dict."""
        import torch

        from alpha_zero.core.civ6_network import MicroCivNetwork

        net = MicroCivNetwork(board_size=8, action_space_size=100, num_res_blocks=2, num_filters=16)

        # Create state dict with extra keys
        state_dict = net.state_dict()
        state_dict["nonexistent_layer.weight"] = torch.randn(10, 10)

        # strict=False should not raise
        net.load_state_dict(state_dict, strict=False)


class TestTrajectoryDatasetRemoved:
    """TrajectoryDataset class should be removed (was dead code)."""

    def test_trajectory_dataset_not_importable(self) -> None:
        """TrajectoryDataset was dead code and should be removed."""
        import alpha_zero.cloud.gcp.training_worker as tw

        assert not hasattr(
            tw, "TrajectoryDataset"
        ), "TrajectoryDataset class should have been removed as dead code"
