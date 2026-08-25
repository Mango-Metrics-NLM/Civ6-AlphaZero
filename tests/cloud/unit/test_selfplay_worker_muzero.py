"""
Unit tests for MuZero self-play worker paths.

Tests the MuZero-specific functions in selfplay_worker.py:
- parse_args with mcts_type=muzero
- load_muzero_model
- run_muzero_selfplay
- main() routing to MuZero path
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import patch

import pytest
import torch

from alpha_zero.cloud.gcp.config import GCPConfig, GCSPathConfig

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def gcp_config() -> GCPConfig:
    """Create test GCP config."""
    return GCPConfig(
        project_id="test-project",
        region="us-central1",
        bucket="test-bucket",
        paths=GCSPathConfig(),
    )


# =============================================================================
# parse_args — MuZero flag Tests
# =============================================================================


class TestParseArgsMuZero:
    """Test that --mcts_type=muzero is accepted by arg parser."""

    def test_muzero_mcts_type_accepted(self) -> None:
        """Test parsing with mcts_type=muzero."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        test_args = [
            "selfplay_worker",
            "--worker_id=1",
            "--model_version=1",
            "--iteration=1",
            "--mcts_type=muzero",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.mcts_type == "muzero"

    def test_muzero_mcts_type_with_other_args(self) -> None:
        """Test muzero type alongside other MCTS arguments."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        test_args = [
            "selfplay_worker",
            "--worker_id=1",
            "--model_version=5",
            "--iteration=10",
            "--mcts_type=muzero",
            "--mcts_simulations=200",
            "--max_turns=50",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.mcts_type == "muzero"
        assert args.mcts_simulations == 200
        assert args.max_turns == 50

    def test_default_mcts_type_is_not_muzero(self) -> None:
        """Test that default mcts_type is NOT muzero (backwards compat)."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        test_args = [
            "selfplay_worker",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.mcts_type != "muzero"


# =============================================================================
# load_muzero_model Tests
# =============================================================================


class TestLoadMuZeroModel:
    """Tests for MuZero model loading."""

    def test_dry_run_creates_muzero_model(self, gcp_config: GCPConfig) -> None:
        """Test that dry run creates a functional MuZero model."""
        from alpha_zero.cloud.gcp.selfplay_worker import load_muzero_model

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            dry_run=True,
        )

        assert model is not None
        # Verify it's a MuZero network
        from alpha_zero.core.muzero_network import MuZeroNetwork

        assert isinstance(model, MuZeroNetwork)

    def test_dry_run_model_can_forward(self, gcp_config: GCPConfig) -> None:
        """Test that dry-run MuZero model can perform forward pass."""
        from alpha_zero.cloud.gcp.selfplay_worker import load_muzero_model

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            dry_run=True,
        )

        model.eval()
        spatial = torch.randn(1, 13, 8, 8)
        vector = torch.randn(1, 64)

        with torch.no_grad():
            policy, value = model(spatial, vector)

        assert policy.shape[0] == 1
        assert value.shape == (1,)

    def test_dry_run_model_initial_inference(self, gcp_config: GCPConfig) -> None:
        """Test that dry-run MuZero model can do initial inference."""
        from alpha_zero.cloud.gcp.selfplay_worker import load_muzero_model

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            dry_run=True,
        )

        model.eval()
        spatial = torch.randn(1, 13, 8, 8)
        vector = torch.randn(1, 64)

        with torch.no_grad():
            output = model.initial_inference(spatial, vector)

        assert output.hidden_state is not None
        assert output.policy_logits.shape[1] == 376
        assert output.value.shape == (1,)

    def test_dry_run_model_recurrent_inference(self, gcp_config: GCPConfig) -> None:
        """Test that dry-run MuZero model can do recurrent inference."""
        from alpha_zero.cloud.gcp.selfplay_worker import load_muzero_model

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            dry_run=True,
        )

        model.eval()
        spatial = torch.randn(1, 13, 8, 8)
        vector = torch.randn(1, 64)

        with torch.no_grad():
            output = model.initial_inference(spatial, vector)
            action = torch.tensor([0])
            next_hidden, _reward, policy, _value = model.recurrent_inference(
                output.hidden_state, action
            )

        assert next_hidden.shape == output.hidden_state.shape
        assert policy.shape[1] == 376

    def test_custom_architecture_params(self, gcp_config: GCPConfig) -> None:
        """Test dry run with custom architecture parameters."""
        from alpha_zero.cloud.gcp.selfplay_worker import load_muzero_model

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=16,
            action_space_size=500,
            num_res_blocks=4,
            num_filters=32,
            dry_run=True,
        )

        assert model.config.spatial_size == 16
        assert model.config.action_space_size == 500
        assert model.config.num_res_blocks == 4
        assert model.config.hidden_state_channels == 32


# =============================================================================
# run_muzero_selfplay Tests
# =============================================================================


class TestRunMuZeroSelfplay:
    """Tests for MuZero self-play execution."""

    def test_dry_run_returns_trajectories(self, gcp_config: GCPConfig) -> None:
        """Test that MuZero dry run returns non-empty trajectories."""
        from alpha_zero.cloud.gcp.selfplay_worker import (
            load_muzero_model,
            run_muzero_selfplay,
        )

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            dry_run=True,
        )

        trajectories = run_muzero_selfplay(
            model=model,
            num_games=3,
            mcts_sims=10,
            board_size=8,
            max_turns=30,
            dry_run=True,
        )

        assert len(trajectories) == 3

    def test_dry_run_trajectory_format(self, gcp_config: GCPConfig) -> None:
        """Test that MuZero dry run trajectory has expected keys."""
        from alpha_zero.cloud.gcp.selfplay_worker import (
            load_muzero_model,
            run_muzero_selfplay,
        )

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            dry_run=True,
        )

        trajectories = run_muzero_selfplay(
            model=model,
            num_games=1,
            mcts_sims=10,
            board_size=8,
            max_turns=30,
            dry_run=True,
        )

        traj = trajectories[0]
        # Standard keys
        assert "spatial" in traj
        assert "vector" in traj
        assert "policies" in traj
        assert "actions" in traj
        assert "players" in traj
        assert "rewards" in traj
        # MuZero-specific keys
        assert "reward_targets" in traj
        assert "value_targets" in traj
        # Dry-run payloads follow the real reward semantics: signed per
        # player-to-move, drawn from the {-1, 0, +1} alphabet.
        assert all(r in (-1.0, 0.0, 1.0) for r in traj["rewards"])
        assert all(v in (-1.0, 0.0, 1.0) for v in traj["value_targets"])

    def test_dry_run_trajectory_lengths_consistent(self, gcp_config: GCPConfig) -> None:
        """Test that all trajectory arrays have consistent lengths."""
        from alpha_zero.cloud.gcp.selfplay_worker import (
            load_muzero_model,
            run_muzero_selfplay,
        )

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            dry_run=True,
        )

        trajectories = run_muzero_selfplay(
            model=model,
            num_games=2,
            mcts_sims=10,
            board_size=8,
            max_turns=30,
            dry_run=True,
        )

        for traj in trajectories:
            n = len(traj["spatial"])
            assert len(traj["vector"]) == n
            assert len(traj["policies"]) == n
            assert len(traj["rewards"]) == n


# =============================================================================
# main() MuZero Routing Tests
# =============================================================================


class TestMainMuZeroRouting:
    """Tests for the main() function MuZero routing."""

    def test_main_dry_run_muzero_success(self) -> None:
        """Test that main() with --mcts_type=muzero --dry_run completes."""
        from alpha_zero.cloud.gcp.selfplay_worker import main

        test_args = [
            "selfplay_worker",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
            "--games=2",
            "--mcts_simulations=10",
            "--mcts_type=muzero",
            "--dry_run",
        ]

        with patch.object(sys, "argv", test_args):
            exit_code = main()

        assert exit_code == 0

    def test_main_muzero_error_returns_1(self) -> None:
        """Test that MuZero errors return code 1."""
        from alpha_zero.cloud.gcp.selfplay_worker import main

        test_args = [
            "selfplay_worker",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
            "--mcts_type=muzero",
            "--dry_run",
        ]

        with patch.object(sys, "argv", test_args):
            with patch(
                "alpha_zero.cloud.gcp.selfplay_worker.load_muzero_model",
                side_effect=RuntimeError("MuZero model load failed"),
            ):
                exit_code = main()

        assert exit_code == 1

    def test_main_muzero_keyboard_interrupt(self) -> None:
        """Test that KeyboardInterrupt in MuZero path returns 130."""
        from alpha_zero.cloud.gcp.selfplay_worker import main

        test_args = [
            "selfplay_worker",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
            "--mcts_type=muzero",
            "--dry_run",
        ]

        with patch.object(sys, "argv", test_args):
            with patch(
                "alpha_zero.cloud.gcp.selfplay_worker.load_muzero_model",
                side_effect=KeyboardInterrupt(),
            ):
                exit_code = main()

        assert exit_code == 130


# =============================================================================
# Config Filtering Tests (v15 fix regression)
# =============================================================================


class TestConfigFiltering:
    """Tests for MuZeroConfig filtering of unknown fields."""

    def test_config_filtering_strips_unknown_fields(self) -> None:
        """Config filter should strip ViT fields like patch_size."""
        import dataclasses as _dc

        from alpha_zero.core.muzero_network import MuZeroConfig

        valid_fields = {f.name for f in _dc.fields(MuZeroConfig)}

        # Simulate a checkpoint config with ViT-specific fields
        checkpoint_config = {
            "spatial_size": 8,
            "vector_dim": 64,
            "action_space_size": 376,
            "num_res_blocks": 6,
            "hidden_state_channels": 64,
            # ViT fields that should be stripped:
            "patch_size": 4,
            "embed_dim": 256,
            "transformer_depth": 12,
        }

        unknown_keys = [k for k in checkpoint_config if k not in valid_fields]
        filtered = {k: v for k, v in checkpoint_config.items() if k in valid_fields}

        assert "patch_size" in unknown_keys
        assert "embed_dim" in unknown_keys
        assert "transformer_depth" in unknown_keys
        assert "patch_size" not in filtered
        assert "embed_dim" not in filtered
        assert "spatial_size" in filtered
        assert "action_space_size" in filtered

        # Verify filtered config can construct MuZeroConfig
        config = MuZeroConfig(**filtered)
        assert config.spatial_size == 8
        assert config.action_space_size == 376

    def test_valid_fields_pass_through_unchanged(self) -> None:
        """All recognized MuZeroConfig fields should pass through filtering."""
        import dataclasses as _dc

        from alpha_zero.core.muzero_network import MuZeroConfig

        valid_fields = {f.name for f in _dc.fields(MuZeroConfig)}

        # All valid fields
        checkpoint_config = {
            "spatial_size": 16,
            "vector_dim": 128,
            "action_space_size": 500,
            "num_res_blocks": 4,
            "hidden_state_channels": 32,
        }

        filtered = {k: v for k, v in checkpoint_config.items() if k in valid_fields}
        assert filtered == checkpoint_config  # Nothing stripped


# =============================================================================
# Architecture Mismatch Fallback Tests (v16 threshold fix)
# =============================================================================


class TestArchitectureMismatchFallback:
    """Tests for the >20% architecture-mismatch fallback in ``load_muzero_model``.

    These tests exercise the *production loader* path
    (``selfplay_worker.load_muzero_model``) end-to-end with a stubbed
    ``GCSModelStorage.load_model``, rather than just recomputing the
    mismatch ratio on the side — the previous version of this test did
    not actually verify that the fallback branch ran, only that the
    arithmetic Copilot reviewed would have crossed the threshold.
    """

    @staticmethod
    def _muzero_kwargs() -> dict[str, int]:
        return {
            "spatial_size": 8,
            "vector_dim": 64,
            "action_space_size": 376,
            "num_res_blocks": 6,
            "hidden_state_channels": 64,
        }

    @staticmethod
    def _muzero_config_dict_from_kwargs(kwargs: dict[str, int]) -> dict[str, Any]:
        """Build the saved-config dict used by load_muzero_model's reconstruction path."""
        # Mirror the fields load_muzero_model wires up explicitly so the
        # reconstructed MuZeroConfig matches the production loader's view.
        return {
            "spatial_channels": 13,  # SPATIAL_CHANNELS used by the loader
            **kwargs,
        }

    def test_mismatch_above_threshold_triggers_fallback(
        self,
        gcp_config: GCPConfig,
    ) -> None:
        """Stubbed checkpoint with >20% mismatched keys → loader returns fresh model.

        Verified by injecting unique-id parameters into the fake state_dict
        and asserting they are NOT present in the returned model — i.e. the
        loader silently dropped the bad checkpoint and built a fresh
        ``MuZeroNetwork`` from the same config.
        """
        from alpha_zero.cloud.gcp import selfplay_worker
        from alpha_zero.cloud.gcp.storage import ModelMetadata
        from alpha_zero.core.muzero_network import MuZeroConfig, MuZeroNetwork

        kwargs = self._muzero_kwargs()
        cfg = MuZeroConfig(**kwargs)
        baseline = MuZeroNetwork(config=cfg)
        real_state = baseline.state_dict()

        # Drop > 20% of real keys and add foreign keys so the loader's
        # mismatch_ratio crosses the 0.2 threshold.
        keys_to_remove = list(real_state.keys())[: max(1, len(real_state) // 3)]
        fake_state = {k: v for k, v in real_state.items() if k not in keys_to_remove}
        # SENTINEL: a key the fresh model will never have. Used to assert the
        # loader did NOT consume these weights.
        sentinel_key = "vit_encoder.layer_999.weight"
        fake_state[sentinel_key] = torch.full((4, 4), 7.0)
        for i in range(len(keys_to_remove)):
            fake_state[f"vit_encoder.layer_{i}.weight"] = torch.full((4, 4), float(i + 1))

        fake_metadata = ModelMetadata(version=999, iteration=0, elo=None)
        fake_cfg_dict = self._muzero_config_dict_from_kwargs(kwargs)

        with patch.object(selfplay_worker, "GCSModelStorage") as MockStorage:
            MockStorage.return_value.load_model.return_value = (
                fake_state,
                fake_metadata,
                fake_cfg_dict,
            )
            model = selfplay_worker.load_muzero_model(
                config=gcp_config,
                version=999,
                board_size=kwargs["spatial_size"],
                action_space_size=kwargs["action_space_size"],
                num_res_blocks=kwargs["num_res_blocks"],
                num_filters=kwargs["hidden_state_channels"],
                dry_run=False,
            )

        # The fallback should have returned a fresh MuZeroNetwork. We don't
        # compare weights directly (the baseline ``baseline`` used different
        # randomness) — instead we assert the foreign keys never reached the
        # returned model's state_dict.
        returned_keys = set(model.state_dict().keys())
        assert sentinel_key not in returned_keys, (
            "Sentinel key from fake checkpoint leaked into returned model — "
            "fallback branch did not run"
        )
        for i in range(len(keys_to_remove)):
            assert f"vit_encoder.layer_{i}.weight" not in returned_keys, (
                f"Foreign key vit_encoder.layer_{i}.weight leaked into returned "
                "model — fallback branch did not run"
            )
        # And the returned model should be in eval mode (loader sets this).
        assert not model.training

    def test_mismatch_below_threshold_loads_normally(
        self,
        gcp_config: GCPConfig,
    ) -> None:
        """0% mismatch → loader returns the model with the supplied weights."""
        from alpha_zero.cloud.gcp import selfplay_worker
        from alpha_zero.cloud.gcp.storage import ModelMetadata
        from alpha_zero.core.muzero_network import MuZeroConfig, MuZeroNetwork

        kwargs = self._muzero_kwargs()
        cfg = MuZeroConfig(**kwargs)
        baseline = MuZeroNetwork(config=cfg)
        # Set a recognisable scalar in one parameter so we can verify it was
        # actually adopted by the returned model (i.e. fallback did NOT fire).
        with torch.no_grad():
            first_param_name = next(iter(baseline.state_dict()))
            baseline.state_dict()[first_param_name].fill_(0.5)
        real_state = baseline.state_dict()

        fake_metadata = ModelMetadata(version=42, iteration=0, elo=None)
        fake_cfg_dict = self._muzero_config_dict_from_kwargs(kwargs)

        with patch.object(selfplay_worker, "GCSModelStorage") as MockStorage:
            MockStorage.return_value.load_model.return_value = (
                real_state,
                fake_metadata,
                fake_cfg_dict,
            )
            model = selfplay_worker.load_muzero_model(
                config=gcp_config,
                version=42,
                board_size=kwargs["spatial_size"],
                action_space_size=kwargs["action_space_size"],
                num_res_blocks=kwargs["num_res_blocks"],
                num_filters=kwargs["hidden_state_channels"],
                dry_run=False,
            )

        # The sentinel value we injected must be present in the returned model
        # — proving the supplied state_dict was loaded, not discarded.
        returned = model.state_dict()[first_param_name]
        assert torch.all(returned == 0.5), (
            "Loaded model parameter does not carry the sentinel value 0.5 — "
            "fallback branch fired unexpectedly"
        )
        assert not model.training


# =============================================================================
# MCTS Search API Regression Tests (v14 bug prevention)
# =============================================================================


class TestMCTSSearchAPI:
    """Regression tests ensuring MCTS search uses (obs, action_mask) API."""

    def test_muzero_mcts_search_signature(self) -> None:
        """MuZeroMCTS.search must accept (obs, action_mask), NOT (env)."""
        import inspect

        from alpha_zero.core.civ6_mcts import MuZeroMCTS

        sig = inspect.signature(MuZeroMCTS.search)
        param_names = list(sig.parameters.keys())

        # 'self' is first, then obs-like, then action_mask
        assert len(param_names) >= 3, (
            f"MuZeroMCTS.search should have at least 3 params (self, obs, action_mask), "
            f"got: {param_names}"
        )
        # Should NOT accept a single 'env' parameter (the v14 bug pattern)
        assert "env" not in param_names, (
            "MuZeroMCTS.search should NOT accept 'env' — use obs + action_mask instead"
        )

    def test_search_call_in_selfplay_uses_obs_and_mask(self) -> None:
        """Verify the selfplay code calls mcts.search(obs, action_mask).

        Locates ``selfplay_worker.py`` by introspecting the imported module
        instead of a hard-coded absolute path, so the test runs in any
        checkout, worktree, or CI environment.
        """
        import ast
        import inspect
        import pathlib

        from alpha_zero.cloud.gcp import selfplay_worker

        source_path = pathlib.Path(inspect.getfile(selfplay_worker))
        assert source_path.is_file(), f"Resolved source path does not exist: {source_path}"

        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Only the MuZero self-play loop must use the (obs, action_mask)
        # signature — the legacy Civ6 path correctly calls ``mcts.search(env)``.
        # Scope the AST search to functions named ``run_muzero_selfplay``.
        muzero_funcs = [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == "run_muzero_selfplay"
        ]
        assert muzero_funcs, "run_muzero_selfplay function not found in selfplay_worker.py"

        mcts_search_calls: list[ast.Call] = []
        for func in muzero_funcs:
            for node in ast.walk(func):
                if isinstance(node, ast.Call):
                    fn = node.func
                    if (
                        isinstance(fn, ast.Attribute)
                        and fn.attr == "search"
                        and isinstance(fn.value, ast.Name)
                        and fn.value.id == "mcts"
                    ):
                        mcts_search_calls.append(node)

        assert mcts_search_calls, "No mcts.search(...) calls found inside run_muzero_selfplay"

        for call in mcts_search_calls:
            # Should have 2 positional args (obs, action_mask), NOT 1 (env).
            assert len(call.args) == 2, (
                f"mcts.search in run_muzero_selfplay should be called with 2 args "
                f"(obs, action_mask), got {len(call.args)} args at line {call.lineno}"
            )
