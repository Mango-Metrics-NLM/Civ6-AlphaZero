"""Unit tests for training_civ6 stability, numerical parity, and M3 helpers."""

from __future__ import annotations

import warnings
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from alpha_zero.core.civ6_network import MicroCivNetwork, compute_loss
from alpha_zero.envs.vectorized_env import VectorizedEnvConfig, VectorizedMicroCivEnv
from alpha_zero.evaluation.scoring import summarize
from alpha_zero.training_civ6 import (
    ReplayDataset,
    TrainingConfig,
    TrainingExample,
    _masked_policy_loss,
    evaluate_against_random,
    train,
    train_epoch,
)
from alpha_zero.utils.constants import MCTS, TRAINING, discrete_action_count


def _make_example(
    board_size: int = 4,
    action_size: int = 16,
    value: float = 0.25,
) -> TrainingExample:
    return TrainingExample(
        spatial=np.arange(13 * board_size * board_size, dtype=np.float32).reshape(
            13, board_size, board_size
        ),
        vector=np.ones(64, dtype=np.float32),
        action_mask=np.ones(action_size, dtype=np.float32),
        policy=np.full(action_size, 1.0 / action_size, dtype=np.float32),
        value=value,
    )


class TestReplayDatasetStability:
    def test_handles_negative_stride_fortran_float64_and_readonly(self) -> None:
        spatial = np.asfortranarray(np.arange(13 * 4 * 4, dtype=np.float64).reshape(13, 4, 4))[
            ::-1, :, :
        ].copy(order="F")
        vector = np.arange(64, dtype=np.float64)[::-1]
        action_mask = np.ones(16, dtype=np.float64)
        policy = np.full(16, 1.0 / 16.0, dtype=np.float32)

        for arr in (spatial, vector, action_mask, policy):
            arr.setflags(write=False)

        ex = TrainingExample(
            spatial=spatial,
            vector=vector,
            action_mask=action_mask,
            policy=policy,
            value=0.5,
        )
        ds = ReplayDataset([ex])

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            item = ds[0]

        for key in ("spatial", "vector", "action_mask", "policy", "value"):
            assert item[key].dtype == torch.float32
        assert torch.allclose(
            item["spatial"], torch.from_numpy(np.array(spatial, dtype=np.float32))
        )
        assert torch.allclose(item["vector"], torch.from_numpy(np.array(vector, dtype=np.float32)))
        assert item["value"].item() == pytest.approx(0.5)


class TestTrainEpochStability:
    def test_empty_dataloader_returns_zero_losses(self) -> None:
        net = MicroCivNetwork(board_size=4, action_space_size=16, num_res_blocks=1, num_filters=4)
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        loader: DataLoader = DataLoader(ReplayDataset([]), batch_size=4)
        cfg = TrainingConfig(device="cpu")

        losses = train_epoch(net, opt, loader, cfg)
        assert losses == {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

    def test_no_deprecation_warning_and_exact_parity_with_compute_loss(self) -> None:
        torch.manual_seed(0)
        net = MicroCivNetwork(board_size=4, action_space_size=16, num_res_blocks=1, num_filters=8)
        ex1 = _make_example(board_size=4, action_size=16, value=1.0)
        ex2 = _make_example(board_size=4, action_size=16, value=-1.0)
        ex1.action_mask[8:] = 0.0
        ex2.action_mask[8:] = 0.0
        loader: DataLoader = DataLoader(ReplayDataset([ex1, ex2]), batch_size=2, shuffle=False)
        cfg = TrainingConfig(device="cpu", num_players=2, gradient_clip=1.0)

        net.train()
        batch = next(iter(loader))
        logits, val = net(batch["spatial"], batch["vector"], batch["action_mask"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ref_total, ref_policy, ref_value = compute_loss(
                logits, val, batch["policy"], batch["value"], batch["action_mask"]
            )

        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            losses = train_epoch(net, opt, loader, cfg)

        assert losses["policy_loss"] == pytest.approx(ref_policy.item(), rel=1e-6, abs=1e-6)
        assert losses["value_loss"] == pytest.approx(ref_value.item(), rel=1e-6, abs=1e-6)
        assert losses["total_loss"] == pytest.approx(ref_total.item(), rel=1e-6, abs=1e-6)

    def test_batch_size_one_batchnorm_safe_and_restores_training_mode(self) -> None:
        class _TinyBNNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn2d = nn.BatchNorm2d(4)
                self.bn1d = nn.BatchNorm1d(4)
                self.policy = nn.Linear(4, 16)
                self.value = nn.Linear(4, 1)

            def forward(
                self,
                spatial: torch.Tensor,
                vector: torch.Tensor,
                action_mask: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                s = self.bn2d(spatial[:, :4, :1, :1]).flatten(1)
                h = self.bn1d(s + vector[:, :4] * 0.0)
                return self.policy(h), self.value(h).squeeze(-1)

        net = _TinyBNNet()
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        loader: DataLoader = DataLoader(ReplayDataset([_make_example()]), batch_size=1)
        cfg = TrainingConfig(device="cpu", num_players=2)

        losses = train_epoch(net, opt, loader, cfg)
        assert np.isfinite(losses["total_loss"])
        assert net.training is True
        assert net.bn2d.training is True
        assert net.bn1d.training is True

    def test_masked_policy_loss_uses_prior_epsilon(self) -> None:
        logits = torch.zeros(1, 4)
        target = torch.zeros(1, 4)
        mask = torch.ones(1, 4)
        loss = _masked_policy_loss(logits, target, mask)
        assert torch.isfinite(loss)
        assert MCTS.PRIOR_EPSILON > 0


class TestTrainLoopDropLastAndEvalActionType:
    @pytest.mark.parametrize("num_examples,expected_batch_sizes", [(5, [4]), (1, [1])])
    def test_train_drop_last_and_sample_multiplier(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        num_examples: int,
        expected_batch_sizes: list[int],
    ) -> None:
        seen_batch_sizes: list[int] = []

        def _fake_train_epoch(network, optimizer, dataloader, config):
            for batch in dataloader:
                seen_batch_sizes.append(int(batch["spatial"].shape[0]))
            return {"total_loss": 0.1, "policy_loss": 0.05, "value_loss": 0.05}

        examples = [_make_example(board_size=4, action_size=376) for _ in range(num_examples)]
        cfg = TrainingConfig(
            board_size=4,
            max_turns=2,
            num_simulations=2,
            num_res_blocks=1,
            num_filters=4,
            num_games_per_iteration=1,
            num_iterations=1,
            num_epochs=1,
            batch_size=4,
            min_replay_size=1,
            eval_interval=999,
            save_interval=999,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "log"),
            seed=7,
            device="cpu",
        )

        monkeypatch.setattr("alpha_zero.training_civ6.run_self_play", lambda *a, **kw: examples)
        monkeypatch.setattr("alpha_zero.training_civ6.train_epoch", _fake_train_epoch)
        monkeypatch.setattr(
            "alpha_zero.training_civ6.evaluate_against_random",
            lambda *a, **kw: summarize(1, 0, 0),
        )
        assert TRAINING.SAMPLE_MULTIPLIER == 100

        train(cfg, dry_run=False)
        assert seen_batch_sizes == expected_batch_sizes

    def test_evaluate_against_random_passes_python_int_action(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        step_action_types: list[type] = []

        class _StubEnv:
            def __init__(self, **kwargs) -> None:
                self.current_player = 1
                self.action_space = SimpleNamespace(n=4)

            def reset(self, seed=None):
                return {}, {}

            def get_valid_actions(self):
                return np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)

            def step(self, action):
                step_action_types.append(type(action))
                return {}, 0.0, True, False, {"winner": 0}

        monkeypatch.setattr("alpha_zero.training_civ6.MicroCivEnv", _StubEnv)
        net = MicroCivNetwork(board_size=4, action_space_size=4, num_res_blocks=1, num_filters=4)
        cfg = TrainingConfig(board_size=4, max_turns=2, num_simulations=2, seed=3, device="cpu")
        summary = evaluate_against_random(net, cfg, num_games=1, iteration=0)
        assert summary.wins + summary.draws + summary.losses == 1
        assert step_action_types == [int]


class TestMypyRuntimeHelpers:
    def test_discrete_action_count_valid_and_invalid(self) -> None:
        assert discrete_action_count(SimpleNamespace(n=np.int64(376))) == 376
        assert type(discrete_action_count(SimpleNamespace(n=np.int64(12)))) is int
        assert type(discrete_action_count(SimpleNamespace(n=np.int32(7)))) is int
        assert type(discrete_action_count(SimpleNamespace(n=np.uint8(5)))) is int

        for bad in (
            SimpleNamespace(),
            SimpleNamespace(n=None),
            SimpleNamespace(n=True),
            SimpleNamespace(n=np.bool_(True)),
            SimpleNamespace(n=3.5),
            SimpleNamespace(n=np.float32(3.5)),
            SimpleNamespace(n=np.float16(3.5)),
            SimpleNamespace(n=Decimal("3.5")),
            SimpleNamespace(n="10"),
            SimpleNamespace(n="abc"),
        ):
            with pytest.raises(TypeError):
                discrete_action_count(bad)

        with pytest.raises(ValueError, match="positive"):
            discrete_action_count(SimpleNamespace(n=0))

    def test_rag_client_without_chromadb_initializes_none_attributes(self) -> None:
        from alpha_zero.utils import rag_client

        with patch.object(rag_client, "CHROMA_AVAILABLE", False):
            client = rag_client.RAGClient(rag_client.RAGConfig(persist_directory="unused"))
            assert client.client is None
            assert client.collection is None
            assert client.add_documents(["doc"]) is False
            assert client.query("q") == []

    def test_vectorized_env_get_active_envs_returns_python_ints(self) -> None:
        venv = VectorizedMicroCivEnv(VectorizedEnvConfig(num_envs=3, board_size=4, max_turns=4))
        venv.reset_all()
        active = venv.get_active_envs()
        assert active == [0, 1, 2]
        assert all(type(idx) is int for idx in active)
