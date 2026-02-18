"""Unit tests for encoder_type pipeline plumbing.

Tests the propagation of encoder_type through the Civ6 AlphaZero cloud pipeline:
- vertex_orchestrator: build_*_extra_args functions include encoder_type
- training_worker: create_model accepts encoder_type parameter
- selfplay_worker: parse_args recognizes encoder_type
- evaluation_worker: parse_args recognizes encoder_type

Validates that the encoder_type setting flows correctly from orchestrator
through to all workers, enabling hex-aware spatial encoding.
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest
import torch

from alpha_zero.cloud.gcp.vertex_orchestrator import (
    build_evaluation_extra_args,
    build_selfplay_extra_args,
    build_training_extra_args,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_args() -> argparse.Namespace:
    """Create base orchestrator args with all required fields for build_*_extra_args."""
    return argparse.Namespace(
        # Selfplay args
        mcts_type="gumbel",
        mcts_simulations=16,
        gumbel_num_actions=16,
        max_turns=30,
        board_size=8,
        action_space_size=376,
        num_res_blocks=6,
        num_filters=64,
        num_heads=8,
        policy_hidden_dim=128,
        value_head_dim=64,
        num_parallel_games=1,
        max_batch_size=64,
        reward_shaping=False,
        reward_scale=1.0,
        algorithm="alphazero",
        gameplay_preset="default",
        encoder_type="patch_embed",
        # Training args
        lr_schedule_type="constant",
        lr_initial=0.001,
        lr_final=0.0001,
        lr_warmup_steps=0,
        lr_decay_steps=0,
        weight_decay=0.01,
        value_lr_multiplier=1.0,
        policy_lr_multiplier=1.0,
    )


# =============================================================================
# build_selfplay_extra_args Tests
# =============================================================================


class TestSelfplayExtraArgs:
    """Tests for build_selfplay_extra_args encoder_type propagation."""

    def test_includes_encoder_type_when_hex_crystalline(
        self, base_args: argparse.Namespace
    ) -> None:
        """Includes --encoder_type=hex_crystalline when non-default."""
        base_args.encoder_type = "hex_crystalline"
        extra = build_selfplay_extra_args(base_args)

        assert "--encoder_type=hex_crystalline" in extra

    def test_excludes_encoder_type_when_default(self, base_args: argparse.Namespace) -> None:
        """Does NOT include encoder_type when default (patch_embed)."""
        base_args.encoder_type = "patch_embed"
        extra = build_selfplay_extra_args(base_args)

        # Should not contain any encoder_type arg when default
        assert not any("--encoder_type" in a for a in extra)

    def test_selfplay_args_backwards_compatible(self, base_args: argparse.Namespace) -> None:
        """Selfplay args work without encoder_type field (getattr fallback)."""
        delattr(base_args, "encoder_type")
        extra = build_selfplay_extra_args(base_args)

        # Should not crash, should not include encoder_type
        assert isinstance(extra, list)
        assert not any("--encoder_type" in a for a in extra)


# =============================================================================
# build_training_extra_args Tests
# =============================================================================


class TestTrainingExtraArgs:
    """Tests for build_training_extra_args encoder_type propagation."""

    def test_includes_encoder_type_when_hex_crystalline(
        self, base_args: argparse.Namespace
    ) -> None:
        """Includes --encoder_type=hex_crystalline when non-default."""
        base_args.encoder_type = "hex_crystalline"
        extra = build_training_extra_args(base_args)

        assert "--encoder_type=hex_crystalline" in extra

    def test_excludes_encoder_type_when_default(self, base_args: argparse.Namespace) -> None:
        """Does NOT include encoder_type when default (patch_embed)."""
        base_args.encoder_type = "patch_embed"
        extra = build_training_extra_args(base_args)

        # Should not contain any encoder_type arg when default
        assert not any("--encoder_type" in a for a in extra)

    def test_training_args_backwards_compatible(self, base_args: argparse.Namespace) -> None:
        """Training args work without encoder_type field (getattr fallback)."""
        delattr(base_args, "encoder_type")
        extra = build_training_extra_args(base_args)

        # Should not crash, should not include encoder_type
        assert isinstance(extra, list)
        assert not any("--encoder_type" in a for a in extra)


# =============================================================================
# build_evaluation_extra_args Tests
# =============================================================================


class TestEvaluationExtraArgs:
    """Tests for build_evaluation_extra_args encoder_type propagation."""

    def test_includes_encoder_type_when_hex_crystalline(
        self, base_args: argparse.Namespace
    ) -> None:
        """Includes --encoder_type=hex_crystalline when non-default."""
        base_args.encoder_type = "hex_crystalline"
        extra = build_evaluation_extra_args(base_args)

        assert "--encoder_type=hex_crystalline" in extra

    def test_excludes_encoder_type_when_default(self, base_args: argparse.Namespace) -> None:
        """Does NOT include encoder_type when default (patch_embed)."""
        base_args.encoder_type = "patch_embed"
        extra = build_evaluation_extra_args(base_args)

        # Should not contain any encoder_type arg when default
        assert not any("--encoder_type" in a for a in extra)

    def test_evaluation_args_backwards_compatible(self, base_args: argparse.Namespace) -> None:
        """Evaluation args work without encoder_type field (getattr fallback)."""
        delattr(base_args, "encoder_type")
        extra = build_evaluation_extra_args(base_args)

        # Should not crash, should not include encoder_type
        assert isinstance(extra, list)
        assert not any("--encoder_type" in a for a in extra)


# =============================================================================
# training_worker.create_model Tests
# =============================================================================


class TestTrainingWorkerCreateModel:
    """Tests for training_worker.create_model encoder_type parameter."""

    def test_create_model_with_hex_crystalline(self) -> None:
        """create_model creates model with hex_crystalline encoder."""
        from alpha_zero.cloud.gcp.training_worker import create_model

        model = create_model(
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            num_heads=8,
            policy_hidden_dim=128,
            value_head_dim=64,
            encoder_type="hex_crystalline",
        )

        # Check that model has the correct config
        assert hasattr(model, "config")
        assert model.config.encoder_type == "hex_crystalline"

    def test_create_model_default_encoder(self) -> None:
        """create_model with default encoder_type creates patch_embed model."""
        from alpha_zero.cloud.gcp.training_worker import create_model

        model = create_model(
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            num_heads=8,
            policy_hidden_dim=128,
            value_head_dim=64,
        )

        # Check that model uses default encoder
        assert hasattr(model, "config")
        assert model.config.encoder_type == "patch_embed"

    def test_create_model_hex_encoder_has_correct_architecture(self) -> None:
        """Model created with hex_crystalline has correct spatial encoder."""
        from alpha_zero.cloud.gcp.training_worker import create_model

        model = create_model(
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            encoder_type="hex_crystalline",
        )

        # Verify the model is a valid torch.nn.Module
        assert isinstance(model, torch.nn.Module)

        # Verify config matches input
        assert model.config.spatial_size == 8
        assert model.config.action_space_size == 376
        assert model.config.num_layers == 6
        assert model.config.embed_dim == 64
        assert model.config.encoder_type == "hex_crystalline"

    def test_create_model_patch_embed_encoder_has_correct_architecture(self) -> None:
        """Model created with patch_embed has correct spatial encoder."""
        from alpha_zero.cloud.gcp.training_worker import create_model

        model = create_model(
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            encoder_type="patch_embed",
        )

        # Verify the model is a valid torch.nn.Module
        assert isinstance(model, torch.nn.Module)

        # Verify config matches input
        assert model.config.spatial_size == 8
        assert model.config.action_space_size == 376
        assert model.config.num_layers == 6
        assert model.config.embed_dim == 64
        assert model.config.encoder_type == "patch_embed"


# =============================================================================
# Worker CLI Argument Parsing Tests
# =============================================================================


class TestTrainingWorkerParseArgs:
    """Tests for training_worker parse_args recognizing encoder_type."""

    def test_parse_args_recognizes_encoder_type(self) -> None:
        """Training worker parse_args accepts --encoder_type flag."""
        from alpha_zero.cloud.gcp.training_worker import parse_args

        test_args = [
            "--iteration=1",
            "--trajectory_iterations=0,1",
            "--encoder_type=hex_crystalline",
        ]

        with patch("sys.argv", ["training_worker.py", *test_args]):
            args = parse_args()

        assert hasattr(args, "encoder_type")
        assert args.encoder_type == "hex_crystalline"

    def test_parse_args_encoder_type_default(self) -> None:
        """Training worker encoder_type defaults to patch_embed."""
        from alpha_zero.cloud.gcp.training_worker import parse_args

        test_args = [
            "--iteration=1",
            "--trajectory_iterations=0,1",
        ]

        with patch("sys.argv", ["training_worker.py", *test_args]):
            args = parse_args()

        assert args.encoder_type == "patch_embed"


class TestSelfplayWorkerParseArgs:
    """Tests for selfplay_worker parse_args recognizing encoder_type."""

    def test_parse_args_recognizes_encoder_type(self) -> None:
        """Selfplay worker parse_args accepts --encoder_type flag."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        test_args = [
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
            "--encoder_type=hex_crystalline",
        ]

        with patch("sys.argv", ["selfplay_worker.py", *test_args]):
            args = parse_args()

        assert hasattr(args, "encoder_type")
        assert args.encoder_type == "hex_crystalline"

    def test_parse_args_encoder_type_default(self) -> None:
        """Selfplay worker encoder_type defaults to patch_embed."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        test_args = [
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
        ]

        with patch("sys.argv", ["selfplay_worker.py", *test_args]):
            args = parse_args()

        assert args.encoder_type == "patch_embed"


class TestEvaluationWorkerParseArgs:
    """Tests for evaluation_worker parse_args recognizing encoder_type."""

    def test_parse_args_recognizes_encoder_type(self) -> None:
        """Evaluation worker parse_args accepts --encoder_type flag."""
        from alpha_zero.cloud.gcp.evaluation_worker import parse_args

        test_args = [
            "--model_version_new=2",
            "--model_version_old=1",
            "--iteration=1",
            "--encoder_type=hex_crystalline",
        ]

        with patch("sys.argv", ["evaluation_worker.py", *test_args]):
            args = parse_args()

        assert hasattr(args, "encoder_type")
        assert args.encoder_type == "hex_crystalline"

    def test_parse_args_encoder_type_default(self) -> None:
        """Evaluation worker encoder_type defaults to patch_embed."""
        from alpha_zero.cloud.gcp.evaluation_worker import parse_args

        test_args = [
            "--model_version_new=2",
            "--model_version_old=1",
            "--iteration=1",
        ]

        with patch("sys.argv", ["evaluation_worker.py", *test_args]):
            args = parse_args()

        assert args.encoder_type == "patch_embed"


# =============================================================================
# Integration Tests
# =============================================================================


class TestEncoderTypePipelineIntegration:
    """Integration tests for encoder_type propagation through the full pipeline."""

    def test_hex_encoder_flows_through_selfplay_training_eval(
        self, base_args: argparse.Namespace
    ) -> None:
        """encoder_type=hex_crystalline propagates through all build_*_extra_args."""
        base_args.encoder_type = "hex_crystalline"

        selfplay_args = build_selfplay_extra_args(base_args)
        training_args = build_training_extra_args(base_args)
        eval_args = build_evaluation_extra_args(base_args)

        # All three should include the encoder_type flag
        assert "--encoder_type=hex_crystalline" in selfplay_args
        assert "--encoder_type=hex_crystalline" in training_args
        assert "--encoder_type=hex_crystalline" in eval_args

    def test_default_encoder_not_in_args(self, base_args: argparse.Namespace) -> None:
        """Default encoder_type is omitted from all extra_args for backwards compat."""
        base_args.encoder_type = "patch_embed"

        selfplay_args = build_selfplay_extra_args(base_args)
        training_args = build_training_extra_args(base_args)
        eval_args = build_evaluation_extra_args(base_args)

        # None should include encoder_type when using default
        assert not any("--encoder_type" in a for a in selfplay_args)
        assert not any("--encoder_type" in a for a in training_args)
        assert not any("--encoder_type" in a for a in eval_args)

    def test_create_model_respects_encoder_type_from_args(self) -> None:
        """Model creation respects encoder_type parsed from CLI args."""
        from alpha_zero.cloud.gcp.training_worker import create_model, parse_args

        test_args = [
            "--iteration=1",
            "--trajectory_iterations=0,1",
            "--encoder_type=hex_crystalline",
            "--board_size=16",
            "--action_space_size=376",
        ]

        with patch("sys.argv", ["training_worker.py", *test_args]):
            args = parse_args()

        # Create model using parsed args
        model = create_model(
            board_size=args.board_size,
            action_space_size=args.action_space_size,
            encoder_type=args.encoder_type,
        )

        # Verify encoder_type was applied
        assert model.config.encoder_type == "hex_crystalline"
        assert model.config.spatial_size == 16
        assert model.config.action_space_size == 376
