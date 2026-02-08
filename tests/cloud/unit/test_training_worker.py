"""
Unit tests for training worker.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

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
    @patch("torch.cuda.set_device")
    def test_setup_ddp_with_env_vars(
        self,
        mock_set_device: MagicMock,
        mock_init_process_group: MagicMock,
    ) -> None:
        """Test setup_ddp when environment variables are present."""
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
            
            mock_init_process_group.assert_called_once_with(backend="nccl")
            mock_set_device.assert_called_once_with(1)

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
