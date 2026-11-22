"""Regression and AQA tests for GM rules/bridge Lua parity and Dual-GPU E2E invariants.

Covers:
1. Seeded determinism of ``run_selfplay`` and ``run_muzero_selfplay`` without relying on
   process-global numpy RNG state (covering both CPU and live CUDA devices).
2. Un-mocked multi-device (CPU + every live ``cuda:N``) ML training steps, finite
   gradients, strict ``eval()`` deterministic replay, illegal-action mask zeroing,
   and cross-device checkpoint save/load equivalence.
3. Un-mocked ``alpha_zero.rules`` -> ``ResearchRegistry`` -> bridge converters ->
   ``Civ6NetworkV2`` device forward pass -> ``lupa`` Lua 5.1 command execution when
   ``lupa`` is installed (failing cleanly under ``CIV6AZ_REQUIRE_LUA=1``, zero skips).
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

from alpha_zero.bridge import tech_registry as tr
from alpha_zero.bridge.action_converter import ActionConverter
from alpha_zero.bridge.protocol import ActionType, GameStateData, LuaCommand
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.cloud.gcp.selfplay_worker import run_muzero_selfplay, run_selfplay
from alpha_zero.cloud.gcp.training_worker import create_model, create_muzero_model
from alpha_zero.core.civ6_actions import ActionCategory, ActionSpaceConfig
from alpha_zero.core.civ6_network import Civ6NetworkV2, NetworkConfig
from alpha_zero.core.muzero_network import MuZeroConfig, MuZeroNetwork
from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.rules import BUILTIN, content_sha256, load_builtin, parse_ruleset
from tests.fixtures import lua_harness as h

REPO_ROOT = Path(__file__).resolve().parents[2]
_BOARD_SIZE = 6
_MAX_TURNS = 6
_MCTS_SIMS = 2
_SEED = 4242


def _live_devices() -> list[str]:
    """Return ``['cpu', 'cuda:0', ...]`` dynamically for whatever hardware is present."""
    if not torch.cuda.is_available():
        return ["cpu"]
    return ["cpu", *[f"cuda:{idx}" for idx in range(torch.cuda.device_count())]]


@contextlib.contextmanager
def _preserve_rng_state() -> Iterator[None]:
    """Snapshot and restore process-global numpy and torch RNG states, CUDA generators included."""
    np_state = np.random.get_state()
    cuda = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda):
        try:
            yield
        finally:
            np.random.set_state(np_state)


def _trajectory_fingerprint(trajectories: list[dict[str, Any]]) -> list[dict[str, list[Any]]]:
    """Normalize replayed trajectory fields for exact determinism comparison."""
    return [
        {
            "actions": [int(a) for a in traj["actions"]],
            "policies": [np.asarray(p, dtype=np.float32).tolist() for p in traj["policies"]],
            "rewards": [float(r) for r in traj["rewards"]],
        }
        for traj in trajectories
    ]


def test_run_selfplay_and_muzero_seed_determinism_without_global_rng() -> None:
    """``seed=`` drives both sequential AlphaZero and MuZero self-play end-to-end."""
    with _preserve_rng_state():
        torch.manual_seed(0)
        az_model = create_model(
            board_size=_BOARD_SIZE,
            num_res_blocks=1,
            num_filters=16,
            num_heads=2,
        ).eval()
        mz_model = create_muzero_model(
            board_size=_BOARD_SIZE,
            num_res_blocks=1,
            num_filters=16,
        ).eval()

        first_az = run_selfplay(
            model=az_model,
            num_games=2,
            mcts_sims=_MCTS_SIMS,
            mcts_type="gumbel",
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            device="cpu",
            seed=_SEED,
        )
        np.random.random(512)
        torch.rand(512)
        second_az = run_selfplay(
            model=az_model,
            num_games=2,
            mcts_sims=_MCTS_SIMS,
            mcts_type="gumbel",
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            device="cpu",
            seed=_SEED,
        )
        other_az = run_selfplay(
            model=az_model,
            num_games=2,
            mcts_sims=_MCTS_SIMS,
            mcts_type="gumbel",
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            device="cpu",
            seed=_SEED + 100,
        )

        assert first_az and all(traj["actions"] for traj in first_az)
        assert _trajectory_fingerprint(first_az) == _trajectory_fingerprint(second_az)
        assert _trajectory_fingerprint(first_az) != _trajectory_fingerprint(other_az)

        first_mz = run_muzero_selfplay(
            model=mz_model,
            num_games=2,
            mcts_sims=_MCTS_SIMS,
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            device="cpu",
            seed=_SEED,
        )
        np.random.random(512)
        torch.rand(512)
        second_mz = run_muzero_selfplay(
            model=mz_model,
            num_games=2,
            mcts_sims=_MCTS_SIMS,
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            device="cpu",
            seed=_SEED,
        )
        other_mz = run_muzero_selfplay(
            model=mz_model,
            num_games=2,
            mcts_sims=_MCTS_SIMS,
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            device="cpu",
            seed=_SEED + 100,
        )

        assert first_mz and all(traj["actions"] for traj in first_mz)
        assert _trajectory_fingerprint(first_mz) == _trajectory_fingerprint(second_mz)
        assert _trajectory_fingerprint(first_mz) != _trajectory_fingerprint(other_mz)

    # Caller override (`mcts_config_override={"seed": 999}`) must be honoured, while `seed=None`
    # leaves `"seed"` out of `create_mcts` kwargs.
    fake_env = MagicMock()
    fake_env.get_observation.return_value = {
        "spatial": np.zeros((13, _BOARD_SIZE, _BOARD_SIZE), dtype=np.float32),
        "vector": np.zeros((64,), dtype=np.float32),
        "action_mask": np.ones((376,), dtype=np.float32),
    }
    fake_env.step.return_value = (None, 0.0, True, False, {"winner": 0})
    fake_mcts = MagicMock()
    fake_mcts.search.return_value = (np.ones((376,), dtype=np.float32) / 376.0, 0.0)
    fake_mcts.select_action.return_value = 0

    with (
        patch("alpha_zero.envs.env_factory.create_game_env", return_value=fake_env),
        patch("alpha_zero.core.mcts_factory.create_mcts", return_value=fake_mcts) as mcts_mock,
    ):
        run_selfplay(
            model=MagicMock(spec=torch.nn.Module),
            num_games=1,
            mcts_sims=2,
            mcts_type="gumbel",
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            seed=None,
        )
        assert "seed" not in mcts_mock.call_args.kwargs

        mcts_mock.reset_mock()
        run_selfplay(
            model=MagicMock(spec=torch.nn.Module),
            num_games=1,
            mcts_sims=2,
            mcts_type="gumbel",
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            mcts_config_override={"seed": 999},
            seed=_SEED,
        )
        assert mcts_mock.call_args.kwargs["seed"] == 999

    if torch.cuda.is_available():
        gpu_script = (
            "import numpy as np, torch\n"
            "from alpha_zero.cloud.gcp.selfplay_worker import run_muzero_selfplay, run_selfplay\n"
            "from alpha_zero.cloud.gcp.training_worker import create_model, create_muzero_model\n"
            # The backend is NOT configured here: the seeded worker call has to do it.
            # A fresh process starts with the deterministic flags off.
            "assert not torch.are_deterministic_algorithms_enabled()\n"
            "for idx in range(torch.cuda.device_count()):\n"
            "    dev = f'cuda:{idx}'\n"
            "    torch.manual_seed(0)\n"
            "    az = create_model(board_size=6, num_res_blocks=1, num_filters=16, num_heads=2).to(dev).eval()\n"
            "    mz = create_muzero_model(board_size=6, num_res_blocks=1, num_filters=16).to(dev).eval()\n"
            "    a1 = [t['actions'] for t in run_selfplay(model=az, num_games=1, mcts_sims=2, mcts_type='gumbel', board_size=6, max_turns=4, device=dev, seed=4242)]\n"
            "    np.random.random(256); torch.rand(256)\n"
            "    a2 = [t['actions'] for t in run_selfplay(model=az, num_games=1, mcts_sims=2, mcts_type='gumbel', board_size=6, max_turns=4, device=dev, seed=4242)]\n"
            "    assert a1 and a1[0] and a1 == a2, (dev, a1, a2)\n"
            "    assert torch.are_deterministic_algorithms_enabled(), 'the seeded run left the backend as it was'\n"
            "    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark\n"
            "    m1 = [t['actions'] for t in run_muzero_selfplay(model=mz, num_games=1, mcts_sims=2, board_size=6, max_turns=4, device=dev, seed=4242)]\n"
            "    np.random.random(256); torch.rand(256)\n"
            "    m2 = [t['actions'] for t in run_muzero_selfplay(model=mz, num_games=1, mcts_sims=2, board_size=6, max_turns=4, device=dev, seed=4242)]\n"
            "    assert m1 and m1[0] and m1 == m2, (dev, m1, m2)\n"
            "print('CUDA_SELFPLAY_DETERMINISM_OK')\n"
        )
        env = os.environ.copy()
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        proc = subprocess.run(
            [sys.executable, "-c", gpu_script],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
        assert "CUDA_SELFPLAY_DETERMINISM_OK" in proc.stdout


def test_unmocked_multidevice_ml_training_and_checkpoint_invariants(tmp_path: Path) -> None:
    """Exercise un-mocked train/eval/mask/checkpoint invariants across CPU and live CUDA GPUs."""
    devices = _live_devices()
    env = MicroCivEnv(board_size=8, max_turns=12, num_players=2, seed=2026)
    obs, _ = env.reset()
    spatial_np = np.asarray(obs["spatial"], dtype=np.float32)
    vector_np = np.asarray(obs["vector"], dtype=np.float32)
    mask_np = np.asarray(obs["action_mask"], dtype=np.float32)
    spatial_channels, board_h, board_w = spatial_np.shape
    assert board_h == board_w == 8
    vector_dim = vector_np.shape[0]
    action_space_size = mask_np.shape[0]

    with _preserve_rng_state():
        for enc_type in ("patch_embed", "hex_crystalline"):
            cfg = TransformerConfig(
                spatial_channels=spatial_channels,
                spatial_size=8,
                vector_dim=vector_dim,
                action_space_size=action_space_size,
                num_layers=1,
                embed_dim=16,
                num_heads=2,
                policy_hidden_dim=32,
                value_head_dim=16,
                encoder_type=enc_type,
                dropout=0.0,
                attention_dropout=0.0,
                path_dropout=0.0,
            )
            for dev in devices:
                torch.manual_seed(77)
                net = Civ6TransformerNetwork(config=cfg).to(dev)
                net.train()
                w_before = next(net.parameters()).detach().clone()
                opt = torch.optim.AdamW(net.parameters(), lr=1e-3)

                sp_t = torch.from_numpy(spatial_np).unsqueeze(0).to(dev)
                vec_t = torch.from_numpy(vector_np).unsqueeze(0).to(dev)
                mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(dev)

                logits, value = net(sp_t, vec_t, mask_t)
                target_policy = mask_t / mask_t.sum(dim=-1, keepdim=True)
                policy_loss = (
                    -(target_policy * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                )
                value_loss = torch.nn.functional.mse_loss(value, torch.full_like(value, 0.5))
                loss = policy_loss + value_loss
                assert torch.isfinite(loss).item()

                opt.zero_grad()
                loss.backward()
                for param_name, param in net.named_parameters():
                    if param.grad is not None:
                        assert torch.isfinite(param.grad).all().item(), f"bad grad: {param_name}"
                opt.step()
                assert not torch.equal(w_before, next(net.parameters()).detach())

                net.eval()
                with torch.no_grad():
                    logits_a, val_a = net(sp_t, vec_t, mask_t)
                    logits_b, val_b = net(sp_t, vec_t, mask_t)
                    assert torch.equal(logits_a, logits_b)
                    assert torch.equal(val_a, val_b)
                    probs = torch.softmax(logits_a, dim=-1)
                    assert torch.isclose(probs.sum(), torch.tensor(1.0, device=dev), atol=1e-5)
                    illegal = mask_t == 0
                    assert torch.all(probs[illegal] == 0).item()

            ckpt_path = tmp_path / f"transformer_{enc_type}.pt"
            src = Civ6TransformerNetwork(config=cfg).to(devices[-1]).eval()
            src.save(str(ckpt_path))
            dst = Civ6TransformerNetwork.load(str(ckpt_path), device=devices[0]).eval()
            with torch.no_grad():
                sp_src = torch.from_numpy(spatial_np).unsqueeze(0).to(devices[-1])
                vec_src = torch.from_numpy(vector_np).unsqueeze(0).to(devices[-1])
                l_src, v_src = src(sp_src, vec_src)
                l_dst, v_dst = dst(sp_src.to(devices[0]), vec_src.to(devices[0]))
                assert torch.allclose(l_src.cpu(), l_dst.cpu(), atol=1e-4, rtol=1e-4)
                assert torch.allclose(v_src.cpu(), v_dst.cpu(), atol=1e-4, rtol=1e-4)

        mz_cfg = MuZeroConfig(
            spatial_channels=spatial_channels,
            spatial_size=8,
            vector_dim=vector_dim,
            action_space_size=action_space_size,
            num_res_blocks=1,
            hidden_state_channels=16,
        )
        valid_action = int(np.flatnonzero(mask_np)[0])
        for dev in devices:
            torch.manual_seed(88)
            mz = MuZeroNetwork(config=mz_cfg).to(dev)
            mz.train()
            w_before = next(mz.parameters()).detach().clone()
            opt = torch.optim.AdamW(mz.parameters(), lr=1e-3)
            sp_t = torch.from_numpy(spatial_np).unsqueeze(0).to(dev)
            vec_t = torch.from_numpy(vector_np).unsqueeze(0).to(dev)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(dev)

            h0, p0, v0 = mz.initial_inference(sp_t, vec_t, mask_t)
            act_t = torch.tensor([valid_action], dtype=torch.long, device=dev)
            h1, r1, p1, v1 = mz.recurrent_inference(h0, act_t, mask_t)
            assert h1.shape == h0.shape
            loss = (
                torch.nn.functional.mse_loss(v0, torch.zeros_like(v0))
                + torch.nn.functional.mse_loss(v1, torch.zeros_like(v1))
                + torch.nn.functional.mse_loss(r1, torch.zeros_like(r1))
                - torch.log_softmax(p0, dim=-1)[:, valid_action].mean()
                - torch.log_softmax(p1, dim=-1)[:, valid_action].mean()
            )
            assert torch.isfinite(loss).item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            assert not torch.equal(w_before, next(mz.parameters()).detach())


def test_unmocked_rules_bridge_lua_and_device_roundtrip(tmp_path: Path) -> None:
    """Exercise rules hashing/CLI, Lua bundle gate, bridge converters, and Lua 5.1 -> GPU."""
    spec = load_builtin("science-v1-draft")
    expected_sha = BUILTIN["science-v1-draft"].sha256
    assert content_sha256(spec) == expected_sha
    parsed_dict = yaml.safe_load(
        (REPO_ROOT / "alpha_zero" / "rules" / "data" / "science_v1_draft.yaml").read_text(
            encoding="utf-8"
        )
    )
    reversed_yaml = yaml.safe_dump(parsed_dict, sort_keys=False).replace("\n", "\r\n")
    assert content_sha256(parse_ruleset(reversed_yaml)) == expected_sha

    bundle_proc = subprocess.run(
        [sys.executable, "scripts/build_lua_bundle.py", "--check"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert bundle_proc.returncode == 0, bundle_proc.stdout + bundle_proc.stderr

    out_yaml = tmp_path / "imported_gameinfo.yaml"
    imp_ok = subprocess.run(
        [
            sys.executable,
            "-m",
            "alpha_zero.rules",
            "import-gameinfo",
            "tests/fixtures/gameinfo/synthetic_export.json",
            "--out",
            str(out_yaml),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert imp_ok.returncode == 0 and out_yaml.is_file(), imp_ok.stderr

    refused_out = REPO_ROOT / f"_tmp_aqa_refused_{tmp_path.name}.yaml"
    try:
        imp_bad = subprocess.run(
            [
                sys.executable,
                "-m",
                "alpha_zero.rules",
                "import-gameinfo",
                "tests/fixtures/gameinfo/synthetic_export.json",
                "--out",
                str(refused_out),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert imp_bad.returncode != 0
        assert not refused_out.exists()
    finally:
        if refused_out.exists():
            refused_out.unlink()

    reg = tr.from_ruleset(spec, ("CIVIC_CODE_OF_LAWS", "CIVIC_FOREIGN_TRADE"))
    sc = StateConverter(board_size=4, research_registry=reg)
    ac = ActionConverter(
        board_size=4,
        action_space_config=ActionSpaceConfig(
            board_size=4,
            max_units=5,
            max_cities=2,
            num_production_items=10,
        ),
        research_registry=reg,
    )

    writing_slot = reg.tech_ids.index("writing")
    if importlib.util.find_spec("lupa") is None:
        if h.lua_required():
            pytest.fail(f"{h.REQUIRE_ENV}=1 but lupa is not installed")
        # Even in CPU environments without lupa, validate the Python research converter pipeline.
        sample = GameStateData.from_dict(
            {
                "turn_id": 1,
                "player_id": 0,
                "map": {"width": 4, "height": 4, "tiles": []},
                "units": [],
                "cities": [],
                "legal_actions": [
                    {"action": "end_turn"},
                    {
                        "action": "research_tech",
                        "items": [writing_slot],
                        "item_names": ["TECH_WRITING"],
                    },
                ],
                "players": [
                    {
                        "id": 0,
                        "civilization": "CIVILIZATION_ROME",
                        "techs_researched": ["TECH_POTTERY"],
                        "civics_researched": ["CIVIC_CODE_OF_LAWS"],
                    }
                ],
            }
        )
        converted = sc.convert(sample)
        mask = ac.generate_action_mask(sample.legal_actions, converted)
        writing_idx = ac.action_space.category_offsets[ActionCategory.RESEARCH_TECH] + writing_slot
        assert mask[writing_idx] == 1.0
        cmd = ac.convert(writing_idx, converted, sample.legal_actions)
        assert cmd == LuaCommand(action_type=ActionType.RESEARCH_TECH, item_id="TECH_WRITING")
        # The Python half held, but neither the network nor the Lua ran. The harness
        # reports that as a skip (a failure under CIV6AZ_REQUIRE_LUA=1), never as a pass.
        h.import_lua51()
        raise AssertionError("unreachable: import_lua51 skips or fails without lupa")

    runtime = h.load_mod()
    parsed = GameStateData.from_dict(h.export_state(runtime))
    converted = sc.convert(parsed)
    spatial_t, vector_t = sc.convert_and_encode(parsed)
    mask = ac.generate_action_mask(parsed.legal_actions, converted)

    writing_idx = ac.action_space.category_offsets[ActionCategory.RESEARCH_TECH] + writing_slot
    assert mask[writing_idx] == 1.0

    target_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    sp_batch = spatial_t.unsqueeze(0) if spatial_t.ndim == 3 else spatial_t
    vec_batch = vector_t.unsqueeze(0) if vector_t.ndim == 1 else vector_t
    net = (
        Civ6NetworkV2(
            config=NetworkConfig(
                spatial_channels=int(sp_batch.shape[1]),
                spatial_size=int(sp_batch.shape[2]),
                vector_dim=int(vec_batch.shape[1]),
                action_space_size=int(mask.shape[0]),
                num_res_blocks=1,
                num_filters=16,
                num_attention_heads=2,
                dropout=0.0,
                attention_dropout=0.0,
            )
        )
        .to(target_dev)
        .eval()
    )
    # The policy is restricted to the research the game offered, and the command is built
    # from the index the network's own output selects, not from a number written here.
    research_start = ac.action_space.category_offsets[ActionCategory.RESEARCH_TECH]
    research_mask = np.zeros_like(mask)
    research_mask[research_start : research_start + len(reg.tech_ids)] = mask[
        research_start : research_start + len(reg.tech_ids)
    ]
    assert research_mask.sum() >= 1.0, "premise: the game offers at least one technology"
    with torch.no_grad():
        logits, val = net(
            sp_batch.float().to(target_dev),
            vec_batch.float().to(target_dev),
            torch.from_numpy(research_mask.astype(np.float32)).unsqueeze(0).to(target_dev),
        )
        probs = torch.softmax(logits, dim=-1)
        assert torch.isfinite(probs).all() and torch.isfinite(val).all()
        assert torch.isclose(probs.sum(), torch.tensor(1.0, device=target_dev), atol=1e-5)
        chosen = int(torch.argmax(probs, dim=-1).item())
    assert research_mask[chosen] == 1.0, "the network chose an action outside the offered research"

    cmd = ac.convert(chosen, converted, parsed.legal_actions)
    assert cmd.action_type == ActionType.RESEARCH_TECH
    assert chosen == writing_idx, "the stand-in game offers TECH_WRITING only"
    assert cmd == LuaCommand(action_type=ActionType.RESEARCH_TECH, item_id="TECH_WRITING")
    ok, err, _msg = h.execute(runtime, cmd.to_dict())
    assert ok is True and err is None
    assert runtime.eval("Civ6Stub.applyRequests()") == 1
    refreshed = GameStateData.from_dict(h.export_state(runtime))
    assert refreshed.players[0].current_research == "TECH_WRITING"


def test_unmocked_live_civ6_parity_and_windows_portability_invariants(tmp_path: Path) -> None:
    """AQA regression covering live-Civ6 Lua parity fixes and Windows portability invariants."""
    import json
    from pathlib import PurePosixPath, PureWindowsPath

    from alpha_zero import civ6_local
    from alpha_zero.bridge import gameinfo_capture
    from alpha_zero.bridge.config import BridgeConfig
    from alpha_zero.bridge.log_tailer import LogTailer
    from alpha_zero.local import gameplay_config

    assert civ6_local.Civ6LocalConfig is gameplay_config.Civ6LocalConfig
    assert civ6_local.search_mode_refusal is gameplay_config.search_mode_refusal

    win_shell = gameinfo_capture._for_a_shell(
        PureWindowsPath(r"C:\ProgramData\%TEMP%\gameinfo.json"), windows=True
    )
    assert win_shell.startswith("'") and win_shell.endswith("'")
    posix_shell = gameinfo_capture._for_a_shell(PurePosixPath("/tmp/two words/snap.json"))
    assert posix_shell == '"/tmp/two words/snap.json"'

    cfg = BridgeConfig.from_live_env(comm_dir=tmp_path)
    assert cfg.comm_dir == tmp_path
    tailer = LogTailer(log_path=tmp_path / "Lua.log")
    tailer._backlog_state = {"turn_id": 4, "player_id": 0}
    tailer._handle_heartbeat('{"turn": 4}')
    assert tailer.current_state() == {"turn_id": 4, "player_id": 0}
    assert tailer.stats["backlog_states_used"] == 1

    if importlib.util.find_spec("lupa") is None:
        if h.lua_required():
            pytest.fail(f"{h.REQUIRE_ENV}=1 but lupa is not installed")
        h.import_lua51()
        raise AssertionError("unreachable: import_lua51 skips or fails without lupa")

    # 1. LLP64-safe integer formatting + -0.0 normalization
    rt = h.load_mod()
    for big_int in (2**31, -(2**31), 2**32, 2**53 - 1):
        encoded = rt.eval(f"AlphaZeroJson.Encode({big_int})")
        assert json.loads(encoded[0] if isinstance(encoded, tuple) else encoded) == big_int
    neg_zero = rt.eval("AlphaZeroJson.Encode(-0.0)")
    assert (neg_zero[0] if isinstance(neg_zero, tuple) else neg_zero) == "0"

    # 2. Sole end-turn blocker authority (LF-H1): UI.CanEndTurn() == true cannot override blocker
    rt.execute(
        "Civ6Stub.canEndTurn = true\n"
        "Civ6Stub.notifications[0] = { { id = 1, blocking = EndTurnBlockingTypes.ENDTURN_BLOCKING_PRODUCTION, dismissed = false } }\n"
    )
    ok_end, err_end, _ = h.execute(rt, {"action_type": "end_turn"})
    assert (ok_end, err_end) == (False, "END_TURN_BLOCKED")

    # 3. GetCurrentProductionTypeHash fallback & short YieldTypes keys & locked production refusal
    rt_live = h.load_mod(
        setup=(
            "local bq = Players[0]:GetCities():FindID(301):GetBuildQueue()\n"
            "bq.CurrentlyBuilding = function() return nil end\n"
            "Civ6Stub.city(301).building = 'UNIT_WARRIOR'\n"
            "Civ6Stub.cities[301].canProduce = { UNIT_WARRIOR = 'now', BUILDING_GRANARY = 'locked' }\n"
            "YieldTypes = { FOOD = 0, PRODUCTION = 1, GOLD = 2, SCIENCE = 3, CULTURE = 4, FAITH = 5 }\n"
            "Map.GetPlot(1, 1).GetYield = function(_, y) return y + 5 end\n"
        )
    )
    exported = h.export_state(rt_live)
    assert exported["cities"][0]["current_production"] == "UNIT_WARRIOR"
    tile_1_1 = next(t for t in exported["map"]["tiles"] if (t["x"], t["y"]) == (1, 1))
    assert tile_1_1["yields"] == {
        "food": 5,
        "production": 6,
        "gold": 7,
        "science": 8,
        "culture": 9,
        "faith": 10,
    }
    produce_offers = [a for a in exported["legal_actions"] if a["action"] == "produce"]
    assert produce_offers and produce_offers[0]["items"] == ["UNIT_WARRIOR"]
    ok_locked, err_locked, _ = h.execute(
        rt_live, {"action_type": "city_produce", "city_id": 301, "item_id": "BUILDING_GRANARY"}
    )
    assert (ok_locked, err_locked) == (False, "OPERATION_REFUSED")
