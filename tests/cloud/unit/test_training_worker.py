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
        assert len(dataset.tensors) == 5, (
            "Expected 5 tensors: spatial, vector, action_mask, policy, value"
        )

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
        spatial, vector, action_mask, _policies, values = dataset.tensors
        # Spatial: (N, 13*8*8)
        assert spatial.shape[1] == 13 * 8 * 8
        # Vector: (N, 64)
        assert vector.shape[1] == 64
        # Action mask: (N, 376)
        assert action_mask.shape[1] == 376
        # Values: (N,) scalar
        assert values.ndim == 1


class TestTrajectoryDatasetRemoved:
    """TrajectoryDataset class should be removed (was dead code)."""

    def test_trajectory_dataset_not_importable(self) -> None:
        """TrajectoryDataset was dead code and should be removed."""
        import alpha_zero.cloud.gcp.training_worker as tw

        assert not hasattr(tw, "TrajectoryDataset"), (
            "TrajectoryDataset class should have been removed as dead code"
        )
