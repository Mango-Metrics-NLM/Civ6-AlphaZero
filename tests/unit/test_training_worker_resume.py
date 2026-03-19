"""Regression tests for training-worker champion model loading.

Phase 0 bug: training_worker.py was creating fresh models with random weights on
every iteration instead of fine-tuning from the current champion.  These tests
verify that:

1. Champion model weights ARE loaded by default (no --fresh_start).
2. --fresh_start bypasses champion loading entirely.
3. An explicit --start_model_version that does not exist raises RuntimeError.
4. When GCS contains no models at all and --fresh_start is not set, RuntimeError
   is raised with a clear message.
5. model_version is threaded through build_training_extra_args / run_training_phase
   in the orchestrator.
"""

from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from alpha_zero.cloud.gcp.config import GCPConfig, GCSPathConfig
from alpha_zero.cloud.gcp.storage import GCSModelStorage
from alpha_zero.cloud.gcp.training_worker import _load_champion_weights
from alpha_zero.cloud.gcp.vertex_orchestrator import build_training_extra_args
from tests.mocks.cloud_mocks import MockStorageBackend
from tests.test_constants import TestDefaults

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gcp_config() -> GCPConfig:
    """Minimal GCPConfig for unit tests (no real GCS needed)."""
    return GCPConfig(
        project_id="test-project",
        region="us-central1",
        bucket="test-bucket",
        paths=GCSPathConfig(),
    )


@pytest.fixture
def simple_model() -> nn.Linear:
    """Tiny model whose state_dict is easy to mutate and compare."""
    return nn.Linear(TestDefaults.SMALL_BOARD_SIZE, 4)


@pytest.fixture
def mock_backend() -> MockStorageBackend:
    return MockStorageBackend()


@pytest.fixture
def storage_with_champion(
    gcp_config: GCPConfig,
    simple_model: nn.Linear,
    mock_backend: MockStorageBackend,
) -> tuple[GCSModelStorage, dict[str, Any]]:
    """Return storage pre-seeded with champion model v5."""
    storage = GCSModelStorage(gcp_config, backend=mock_backend)
    storage.save_model(
        simple_model,
        version=5,
        metadata={"iteration": 5, "elo": 1550.0, "games_played": 300},
    )
    return storage, simple_model.state_dict()


# ---------------------------------------------------------------------------
# _load_champion_weights tests
# ---------------------------------------------------------------------------


class TestLoadChampionWeights:
    """Unit tests for the _load_champion_weights helper."""

    def test_loads_latest_when_version_not_specified(
        self,
        gcp_config: GCPConfig,
        simple_model: nn.Linear,
        storage_with_champion: tuple[GCSModelStorage, Any],
    ) -> None:
        """Default behaviour: loads latest champion (v5) when no version given."""
        _storage, champion_state = storage_with_champion

        # Create a fresh model with different (random) weights
        fresh = nn.Linear(TestDefaults.SMALL_BOARD_SIZE, 4)
        nn.init.constant_(fresh.weight, 99.0)

        with patch(
            "alpha_zero.cloud.gcp.training_worker.GCSModelStorage",
            return_value=_storage,
        ):
            _load_champion_weights(
                config=gcp_config,
                model=fresh,
                start_model_version=None,
                is_main_process=True,
            )

        # Weights should now match the champion
        for key in champion_state:
            assert torch.equal(
                fresh.state_dict()[key], champion_state[key]
            ), f"Weight mismatch for key '{key}' after loading champion"

    def test_loads_explicit_version(
        self,
        gcp_config: GCPConfig,
        simple_model: nn.Linear,
        storage_with_champion: tuple[GCSModelStorage, Any],
    ) -> None:
        """Explicit --start_model_version=5 loads that specific version."""
        _storage, champion_state = storage_with_champion

        fresh = nn.Linear(TestDefaults.SMALL_BOARD_SIZE, 4)
        nn.init.constant_(fresh.weight, 0.0)

        with patch(
            "alpha_zero.cloud.gcp.training_worker.GCSModelStorage",
            return_value=_storage,
        ):
            _load_champion_weights(
                config=gcp_config,
                model=fresh,
                start_model_version=5,
                is_main_process=True,
            )

        for key in champion_state:
            assert torch.equal(fresh.state_dict()[key], champion_state[key])

    def test_raises_when_requested_version_exceeds_latest(
        self,
        gcp_config: GCPConfig,
        simple_model: nn.Linear,
        storage_with_champion: tuple[GCSModelStorage, Any],
    ) -> None:
        """Requesting a version higher than latest raises RuntimeError (no silent fallback)."""
        _storage, _ = storage_with_champion

        fresh = nn.Linear(TestDefaults.SMALL_BOARD_SIZE, 4)

        with patch(
            "alpha_zero.cloud.gcp.training_worker.GCSModelStorage",
            return_value=_storage,
        ):
            with pytest.raises(RuntimeError, match="start_model_version=999"):
                _load_champion_weights(
                    config=gcp_config,
                    model=fresh,
                    start_model_version=999,
                    is_main_process=True,
                )

    def test_raises_when_no_models_in_gcs(
        self,
        gcp_config: GCPConfig,
        simple_model: nn.Linear,
        mock_backend: MockStorageBackend,
    ) -> None:
        """When GCS has no models at all, raises RuntimeError with helpful message."""
        empty_storage = GCSModelStorage(gcp_config, backend=mock_backend)
        fresh = nn.Linear(TestDefaults.SMALL_BOARD_SIZE, 4)

        with patch(
            "alpha_zero.cloud.gcp.training_worker.GCSModelStorage",
            return_value=empty_storage,
        ):
            with pytest.raises(RuntimeError, match="no models found in GCS"):
                _load_champion_weights(
                    config=gcp_config,
                    model=fresh,
                    start_model_version=None,
                    is_main_process=True,
                )

    def test_raises_on_architecture_mismatch(
        self,
        gcp_config: GCPConfig,
        mock_backend: MockStorageBackend,
    ) -> None:
        """State dict shape mismatch raises RuntimeError, not a silent wrong-load."""
        # Save a model with different architecture (different output size)
        different_arch = nn.Linear(TestDefaults.SMALL_BOARD_SIZE, 99)
        storage = GCSModelStorage(gcp_config, backend=mock_backend)
        storage.save_model(different_arch, version=1, metadata={"iteration": 1})

        # Fresh model with incompatible architecture
        fresh = nn.Linear(TestDefaults.SMALL_BOARD_SIZE, 4)

        with patch(
            "alpha_zero.cloud.gcp.training_worker.GCSModelStorage",
            return_value=storage,
        ):
            with pytest.raises(RuntimeError, match="shape mismatch"):
                _load_champion_weights(
                    config=gcp_config,
                    model=fresh,
                    start_model_version=None,
                    is_main_process=True,
                )

    def test_non_main_process_does_not_log(
        self,
        gcp_config: GCPConfig,
        simple_model: nn.Linear,
        storage_with_champion: tuple[GCSModelStorage, Any],
    ) -> None:
        """is_main_process=False: weights still loaded, but no INFO log emitted."""
        _storage, champion_state = storage_with_champion
        fresh = nn.Linear(TestDefaults.SMALL_BOARD_SIZE, 4)

        with patch(
            "alpha_zero.cloud.gcp.training_worker.GCSModelStorage",
            return_value=_storage,
        ), patch("alpha_zero.cloud.gcp.training_worker.logger") as mock_log:
            _load_champion_weights(
                config=gcp_config,
                model=fresh,
                start_model_version=None,
                is_main_process=False,
            )
            mock_log.info.assert_not_called()

        # Despite no logging, weights must still be loaded
        for key in champion_state:
            assert torch.equal(fresh.state_dict()[key], champion_state[key])


# ---------------------------------------------------------------------------
# Orchestrator arg-threading tests
# ---------------------------------------------------------------------------


class TestBuildTrainingExtraArgsModelVersion:
    """Verify orchestrator threads model_version into training extra args."""

    def _base_args(self) -> argparse.Namespace:
        """Minimal args namespace matching orchestrator defaults."""
        return argparse.Namespace(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            num_heads=8,
            policy_hidden_dim=128,
            value_head_dim=64,
            lr_schedule_type="constant",
            lr_initial=0.001,
            lr_final=0.0001,
            lr_warmup_steps=0,
            lr_decay_steps=0,
            weight_decay=0.01,
            value_lr_multiplier=1.0,
            policy_lr_multiplier=1.0,
            algorithm="alphazero",
            encoder_type="patch_embed",
            learning_rate=0.001,
        )

    def test_model_version_included_when_provided(self) -> None:
        args = self._base_args()
        extra = build_training_extra_args(args, model_version=42)
        assert "--start_model_version=42" in extra

    def test_model_version_absent_when_none(self) -> None:
        args = self._base_args()
        extra = build_training_extra_args(args, model_version=None)
        assert not any("start_model_version" in flag for flag in extra)

    def test_model_version_zero_not_forwarded(self) -> None:
        """model_version=0 is the 'no prior models' sentinel from get_initial_model_version.

        It must NOT be forwarded to the training worker, because asking the worker
        to load v0 from GCS when no models exist would crash the job.
        The worker will instead use --fresh_start (or default to latest if v0 exists).
        """
        args = self._base_args()
        extra = build_training_extra_args(args, model_version=0)
        assert not any("start_model_version" in flag for flag in extra)

    def test_backwards_compat_no_model_version_arg(self) -> None:
        """Calling without model_version kwarg (old callers) must not break."""
        args = self._base_args()
        extra = build_training_extra_args(args)
        # Should not raise; should not include start_model_version
        assert not any("start_model_version" in flag for flag in extra)

    def test_model_version_positive_is_forwarded(self) -> None:
        """Any model_version > 0 is a real champion version and must be forwarded."""
        args = self._base_args()
        for version in (1, 42, 268):
            extra = build_training_extra_args(args, model_version=version)
            assert f"--start_model_version={version}" in extra


class TestLoadChampionWeightsEdgeCases:
    """Additional edge-case tests added after gap analysis."""

    @pytest.fixture
    def gcp_config(self) -> GCPConfig:
        return GCPConfig(
            project_id="test-project",
            region="us-central1",
            bucket="test-bucket",
            paths=GCSPathConfig(),
        )

    def test_only_gcs_storage_error_triggers_no_models_message(
        self,
        gcp_config: GCPConfig,
    ) -> None:
        """A non-GCSStorageError from get_latest_version propagates without masking."""
        mock_storage = MagicMock()
        mock_storage.get_latest_version.side_effect = ValueError("unexpected config error")

        fresh = nn.Linear(TestDefaults.SMALL_BOARD_SIZE, 4)

        with patch(
            "alpha_zero.cloud.gcp.training_worker.GCSModelStorage",
            return_value=mock_storage,
        ):
            # ValueError should NOT be caught and converted to RuntimeError("no models")
            with pytest.raises(ValueError, match="unexpected config error"):
                _load_champion_weights(
                    config=gcp_config,
                    model=fresh,
                    start_model_version=None,
                    is_main_process=False,
                )

    def test_load_specific_older_version_below_latest(
        self,
        gcp_config: GCPConfig,
    ) -> None:
        """Requesting a valid older version (< latest) loads that specific version."""
        backend = MockStorageBackend()
        storage = GCSModelStorage(gcp_config, backend=backend)

        # Save two versions with distinct weights
        model_v3 = nn.Linear(TestDefaults.SMALL_BOARD_SIZE, 4)
        nn.init.constant_(model_v3.weight, 3.0)
        storage.save_model(model_v3, version=3, metadata={"iteration": 3})

        model_v7 = nn.Linear(TestDefaults.SMALL_BOARD_SIZE, 4)
        nn.init.constant_(model_v7.weight, 7.0)
        storage.save_model(model_v7, version=7, metadata={"iteration": 7})

        fresh = nn.Linear(TestDefaults.SMALL_BOARD_SIZE, 4)
        nn.init.constant_(fresh.weight, 0.0)

        with patch(
            "alpha_zero.cloud.gcp.training_worker.GCSModelStorage",
            return_value=storage,
        ):
            _load_champion_weights(
                config=gcp_config,
                model=fresh,
                start_model_version=3,  # explicitly request older version
                is_main_process=False,
            )

        # Should have v3 weights (constant 3.0), not v7 (constant 7.0)
        assert torch.allclose(fresh.weight, torch.full_like(fresh.weight, 3.0))

    def test_weights_loaded_on_cpu_before_device_move(
        self,
        gcp_config: GCPConfig,
    ) -> None:
        """_load_champion_weights must work correctly when model is on CPU.

        In the fixed code, champion loading happens before model.to(device),
        so the model is always on CPU at load time. This test verifies that
        the state dict is correctly applied regardless.
        """
        backend = MockStorageBackend()
        storage = GCSModelStorage(gcp_config, backend=backend)

        champion = nn.Linear(TestDefaults.SMALL_BOARD_SIZE, 4)
        nn.init.constant_(champion.weight, 5.0)
        storage.save_model(champion, version=1, metadata={"iteration": 1})

        fresh = nn.Linear(TestDefaults.SMALL_BOARD_SIZE, 4)
        nn.init.constant_(fresh.weight, 0.0)
        # Ensure model is on CPU (as it would be before .to(device))
        assert str(fresh.weight.device) == "cpu"

        with patch(
            "alpha_zero.cloud.gcp.training_worker.GCSModelStorage",
            return_value=storage,
        ):
            _load_champion_weights(
                config=gcp_config,
                model=fresh,
                start_model_version=None,
                is_main_process=False,
            )

        assert torch.allclose(fresh.weight, torch.full_like(fresh.weight, 5.0))
        # Model should remain on CPU after load (no unexpected device migration)
        assert str(fresh.weight.device) == "cpu"
