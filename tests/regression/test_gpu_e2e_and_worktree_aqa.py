"""Regression and AQA tests covering live/mocked GPU behavior, worktree pruning, and RAG safety.

All tests execute unconditionally on both CPU-only CI runners and multi-GPU workstations
without adding ``pytest.skip`` sites: hardware-specific branches inspect
``torch.cuda.is_available()`` dynamically and always verify mocked equivalents too.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from alpha_zero.core.civ6_network import (
    Civ6Network,
    Civ6NetworkV2,
    MicroCivNetwork,
    NetworkConfig,
    SpatialSelfAttention,
)
from alpha_zero.local import devices as local_devices
from alpha_zero.local.pinned_env import TorchPin
from alpha_zero.multiplayer.network_heads import NPlayerNetworkAdapter
from alpha_zero.utils.agent_tools import CodeExplorer
from alpha_zero.utils.rag_client import RAGClient, RAGConfig
from scripts import audit_import_graph as aig
from scripts import release_reproduction as rr


def _live_devices() -> list[str]:
    """Return ``['cpu', 'cuda:0', ...]`` dynamically for whatever hardware is present."""
    if not torch.cuda.is_available():
        return ["cpu"]
    return ["cpu", *[f"cuda:{idx}" for idx in range(torch.cuda.device_count())]]


def test_torch_report_code_and_device_problem_live_and_mocked() -> None:
    """Verify live child torch/GPU probe plus mocked multi-GPU and failure modes."""
    probe_script = (
        f"import json\ninfo = {{}}\n{local_devices.TORCH_REPORT_CODE}\nprint(json.dumps(info))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe_script],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    torch_report = info["torch"]
    assert isinstance(torch_report, dict)
    assert torch_report["version"] == str(torch.__version__)

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        assert torch_report["cuda_available"] is True
        assert torch_report["device_count"] == count
        assert torch_report["smoke"] == [None] * count
        requested = {f"arm_{idx}": f"cuda:{idx}" for idx in range(count)}
        assert (
            local_devices.device_problem(
                requested,
                torch_report,
                driver_report=info.get("nvidia_smi"),
            )
            is None
        )
        oob_problem = local_devices.device_problem({"train": f"cuda:{count}"}, torch_report)
        assert oob_problem is not None and f"cuda:{count}" in oob_problem
    else:
        assert torch_report["cuda_available"] is False
        assert local_devices.device_problem({"train": "cpu"}, torch_report) is None

    # Mocked dual-GPU validation (always runs on CPU CI too)
    pin = TorchPin("torch==2.11.0+cu128", "https://download.pytorch.org/whl/cu128")
    mock_torch_ok: dict[str, Any] = {
        "version": "2.11.0+cu128",
        "cuda": "12.8",
        "cuda_available": True,
        "device_count": 2,
        "devices": ["NVIDIA GeForce RTX 5060 Ti", "NVIDIA GeForce RTX 5060"],
        "capabilities": [[12, 0], [12, 0]],
        "arch_list": ["sm_90", "sm_120"],
        "cudnn": 91900,
        "smoke": [None, None],
    }
    mock_smi_ok = [
        "0, NVIDIA GeForce RTX 5060 Ti, 00000000:01:00.0, 616.92",
        "1, NVIDIA GeForce RTX 5060, 00000000:06:00.0, 616.92",
    ]
    assert (
        local_devices.device_problem(
            {"train": "cuda:0", "arena": "cuda:1"},
            mock_torch_ok,
            torch_pin=pin,
            visible="0,1",
            expected_gpus={0: "NVIDIA GeForce RTX 5060 Ti", 1: "NVIDIA GeForce RTX 5060"},
            driver_report=mock_smi_ok,
        )
        is None
    )
    assert local_devices.driver_versions(mock_smi_ok) == ("616.92", "616.92")

    smi_err = {"error": "CalledProcessError", "returncode": 12, "detail": "NVML mismatch"}
    assert local_devices.driver_versions(smi_err) is None
    assert (
        local_devices.device_problem(
            {"train": "cuda:0"},
            mock_torch_ok,
            torch_pin=pin,
            driver_report=smi_err,
        )
        is not None
    )
    bad_smoke = {**mock_torch_ok, "smoke": [None, "RuntimeError: illegal instruction"]}
    smoke_problem = local_devices.device_problem({"arena": "cuda:1"}, bad_smoke, torch_pin=pin)
    assert smoke_problem is not None and "illegal instruction" in smoke_problem


def test_network_fp16_masking_and_narrow_index_checkpoint_roundtrip_all_devices(
    tmp_path: Path,
) -> None:
    """Verify fp16 masking, cross-device checkpoints, and narrow index widening across devices."""
    cfg = NetworkConfig(
        spatial_channels=4,
        spatial_size=6,
        vector_dim=8,
        num_res_blocks=2,
        num_filters=8,
        fc_hidden_dim=16,
        action_space_size=24,
        value_head_hidden=8,
        dropout=0.0,
        num_attention_heads=2,
        use_attention=True,
        attention_dropout=0.0,
    )
    devices = _live_devices()
    for dev in devices:
        for dtype in (torch.float32, torch.float16):
            v1 = Civ6Network(config=cfg).to(device=dev, dtype=dtype).eval()
            v2 = Civ6NetworkV2(config=cfg).to(device=dev, dtype=dtype).eval()
            micro = MicroCivNetwork(
                board_size=6,
                action_space_size=24,
                num_res_blocks=1,
                num_filters=8,
                spatial_channels=4,
                num_players=3,
                vector_dim=8,
            ).to(device=dev, dtype=dtype)
            micro.eval()
            adapter = NPlayerNetworkAdapter(v1, num_players=3, value_hidden_dim=8).to(
                device=dev, dtype=dtype
            )
            adapter.eval()

            spatial = torch.randn(2, 4, 6, 6, device=dev, dtype=dtype)
            vector = torch.randn(2, 8, device=dev, dtype=dtype)
            mask = torch.zeros(2, 24, device=dev, dtype=dtype)
            mask[:, :4] = 1.0

            for logits in (
                v1(spatial, vector, mask)[0],
                v2(spatial, vector, mask)[0],
                v2.forward_with_categories(spatial, vector, mask)[0],
                micro(spatial, vector, mask)[0],
                adapter(spatial, vector, mask)[0],
            ):
                assert logits.device.type == torch.device(dev).type
                assert logits.dtype == dtype
                assert torch.isfinite(logits).all()
                probs = torch.softmax(logits, dim=-1)
                assert torch.isfinite(probs).all()
                assert torch.all(probs[:, 4:] == 0)

        # Narrow integer index widening + corrupt partial rejection on each live device
        src_attn = SpatialSelfAttention(8, num_heads=2, spatial_size=3).to(dev).eval()
        for narrow_dtype in (torch.int8, torch.uint8, torch.int16, torch.int32):
            sd = {k: v.clone() for k, v in src_attn.state_dict().items()}
            sd["relative_position_index"] = sd["relative_position_index"].to(dtype=narrow_dtype)
            dst_attn = SpatialSelfAttention(8, num_heads=2, spatial_size=None).to(dev).eval()
            dst_attn.load_state_dict(sd, strict=True)
            assert dst_attn.relative_position_index is not None
            assert dst_attn.relative_position_index.dtype == torch.int64
            x = torch.randn(1, 8, 3, 3, device=dev)
            assert torch.allclose(dst_attn(x), src_attn(x))

        eager_attn = SpatialSelfAttention(8, num_heads=2, spatial_size=4).to(dev)
        orig_index = eager_attn.relative_position_index.detach().clone()
        partial_bad: dict[str, Any] = {
            k: v.clone()
            for k, v in eager_attn.state_dict().items()
            if k != "relative_pos_bias_table"
        }
        partial_bad["relative_position_index"] = torch.full(
            (256,), 999, dtype=torch.int64, device=dev
        )
        with pytest.raises(RuntimeError):
            eager_attn.load_state_dict(partial_bad, strict=False)
        assert torch.equal(eager_attn.relative_position_index, orig_index)

    # Cross-device checkpoint roundtrip (last device -> first device)
    ckpt_path = tmp_path / "nested" / "v2_cross_device.pt"
    src_dev = devices[-1]
    dst_dev = devices[0]
    src_v2 = Civ6NetworkV2(config=cfg).to(src_dev).eval()
    src_v2.save(str(ckpt_path), metadata={"src_device": src_dev})
    loaded_v2 = Civ6NetworkV2.load(str(ckpt_path), device=dst_dev).eval()
    sample_s = torch.randn(1, 4, 6, 6, device=src_dev)
    sample_v = torch.randn(1, 8, device=src_dev)
    with torch.no_grad():
        assert torch.allclose(
            src_v2(sample_s, sample_v)[0].cpu(),
            loaded_v2(sample_s.to(dst_dev), sample_v.to(dst_dev))[0].cpu(),
            atol=1e-3,
            rtol=1e-3,
        )


def test_release_reproduction_and_device_env_wiring() -> None:
    """Verify device_plan env wiring and case-insensitive MCTS_* condition differences."""
    env: dict[str, str] = {}
    pin = TorchPin("torch==2.11.0+cu128", "https://download.pytorch.org/whl/cu128")
    plan = local_devices.apply_device_environment(
        env,
        cuda_visible_devices="0,1",
        requested={"train": "cuda:0", "arena": "cuda:1"},
        torch_pin=pin,
        expected_gpus={0: "GPU-A", 1: "GPU-B"},
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["CUDA_DEVICE_ORDER"] == local_devices.PCI_BUS_ORDER
    assert plan["source"] == local_devices.SOURCE_CARD

    base_record = {
        "device_plan": {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "torch_pin": pin.as_dict(),
            "requested": {"train": "cuda:0", "arena": "cuda:1"},
            "expected_gpus": {"0": "GPU-A", "1": "GPU-B"},
            "torch": {
                "version": "2.11.0+cu128",
                "cuda": "12.8",
                "cudnn": 91900,
                "devices": ["GPU-A", "GPU-B"],
                "capabilities": [[12, 0], [12, 0]],
            },
        },
        "preflight": {"nvidia_smi": ["0, GPU-A, bus0, 616.92", "1, GPU-B, bus1, 616.92"]},
        "torch_wheel": {"url": "https://example.invalid/torch.whl", "sha256": "a" * 64},
        "env_var_names": ["PATH", "mcts_simulations"],
    }
    attempt = rr.Attempt(
        run_id="a1-attempt",
        role="attempt",
        sha="a" * 40,
        record=base_record,
        run_json_sha256="b" * 64,
        completed=False,
        devices_registered=True,
    )
    rerun_clean = rr.Attempt(
        run_id="a1-rerun",
        role="rerun",
        sha="a" * 40,
        record={**base_record, "env_var_names": ["PATH"]},
        run_json_sha256="c" * 64,
        completed=True,
        devices_registered=True,
    )
    diffs = rr.condition_differences(attempt, rerun_clean)
    assert any("MCTS_* overrides" in d for d in diffs)


def test_import_graph_and_code_explorer_worktree_and_pep405_venv_isolation(tmp_path: Path) -> None:
    """Both audit_import_graph and CodeExplorer must skip PEP 405 venvs and git worktrees."""
    (tmp_path / "alpha_zero" / "pkg").mkdir(parents=True)
    (tmp_path / "alpha_zero" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "alpha_zero" / "pkg" / "__init__.py").write_text(
        "from alpha_zero.pkg.unused import SentinelTarget\n", encoding="utf-8"
    )
    real_target = tmp_path / "alpha_zero" / "pkg" / "unused.py"
    real_target.write_text("class SentinelTarget:\n    pass\n", encoding="utf-8")

    # Planted duplicate copies under PEP 405 venvs and nested git worktrees
    for folder in (".venv-aqa", "custom_pep405_env", ".claude/worktrees/aqa-wt", "nested_wt"):
        base = tmp_path / folder
        pkg_dir = base / "alpha_zero" / "pkg"
        pkg_dir.mkdir(parents=True)
        if folder == "custom_pep405_env":
            (base / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
        elif folder == "nested_wt":
            (base / ".git").write_text("gitdir: ../.git/worktrees/aqa\n", encoding="utf-8")
        (base / "alpha_zero" / "__init__.py").write_text("", encoding="utf-8")
        (pkg_dir / "__init__.py").write_text(
            "from alpha_zero.pkg.unused import SentinelTarget\n", encoding="utf-8"
        )
        (pkg_dir / "unused.py").write_text("class SentinelTarget:\n    pass\n", encoding="utf-8")

    reports = {r.module: r for r in aig.audit(tmp_path)}
    assert "alpha_zero.pkg.unused" in reports
    assert reports["alpha_zero.pkg.unused"].path == "alpha_zero/pkg/unused.py"

    explorer = CodeExplorer(tmp_path)
    py_files = explorer.find_python_files()
    assert real_target in py_files
    assert all(
        all(
            part not in p.parts
            for part in (".venv-aqa", "custom_pep405_env", ".claude", "nested_wt")
        )
        for p in py_files
    )


def test_rag_client_graceful_degradation_on_native_panic_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RAGClient must survive PyO3-style BaseException panics without swallowing KeyboardInterrupt."""
    import alpha_zero.utils.rag_client as rc_mod

    class NativeRustPanic(BaseException):
        pass

    def _panic(*_args: Any, **_kwargs: Any) -> Any:
        raise NativeRustPanic("panicked at rust/sqlite/src/db.rs")

    monkeypatch.setattr(rc_mod, "CHROMA_AVAILABLE", True)
    monkeypatch.setattr(rc_mod, "chromadb", SimpleNamespace(PersistentClient=_panic), raising=False)

    client = RAGClient(RAGConfig(persist_directory=str(tmp_path / "rag_db")))
    assert client.client is None
    assert client.collection is None
    assert client.add_documents(["doc"]) is False
    assert client.query("query") == []
    assert client.count() == 0

    def _interrupt(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        rc_mod, "chromadb", SimpleNamespace(PersistentClient=_interrupt), raising=False
    )
    with pytest.raises(KeyboardInterrupt):
        RAGClient(RAGConfig(persist_directory=str(tmp_path / "rag_db_2")))


def test_mlflow_evaluation_result_logs_all_15_metrics_and_matches_docs() -> None:
    """log_evaluation_result emits all 15 metrics and matches C4 & Agent.md counts."""
    import alpha_zero.monitoring.mlflow_logger as ml_mod

    logged: dict[str, float] = {}
    fake_mlflow = SimpleNamespace(
        log_metrics=lambda metrics, step=None: logged.update(metrics),
    )
    ml_logger = ml_mod.MLflowLogger(ml_mod.MLflowConfig(enabled=True, fail_silently=False))
    ml_logger._mlflow = fake_mlflow

    fake_result = SimpleNamespace(
        win_rate=SimpleNamespace(
            wins=6,
            losses=2,
            draws=2,
            total_games=10,
            win_rate=60.0,
            score_rate=70.0,
            draw_rate=20.0,
            loss_rate=20.0,
        ),
        avg_game_length=42.5,
        avg_cities_won=1.5,
        avg_cities_lost=0.5,
        avg_units_won=3.0,
        avg_units_lost=1.0,
        total_evaluation_time=12.25,
        value_correlation=0.65,
    )

    emitted = ml_logger.log_evaluation_result(fake_result, prefix="eval", step=3)
    expected_keys = {
        "eval/wins",
        "eval/losses",
        "eval/draws",
        "eval/total_games",
        "eval/win_rate",
        "eval/score_rate",
        "eval/draw_rate",
        "eval/loss_rate",
        "eval/avg_game_length",
        "eval/avg_cities_won",
        "eval/avg_cities_lost",
        "eval/avg_units_won",
        "eval/avg_units_lost",
        "eval/total_evaluation_time",
        "eval/value_correlation",
    }
    assert set(emitted) == expected_keys
    assert set(logged) == expected_keys
    assert len(emitted) == 15
    assert emitted["eval/win_rate"] == pytest.approx(0.6)
    assert emitted["eval/score_rate"] == pytest.approx(0.7)
    assert emitted["eval/draw_rate"] == pytest.approx(0.2)
    assert emitted["eval/loss_rate"] == pytest.approx(0.2)

    repo_root = Path(__file__).resolve().parents[2]
    c4_text = (repo_root / "docs" / "architecture" / "C4_ARCHITECTURE.md").read_text(
        encoding="utf-8"
    )
    mon_agent = (repo_root / "alpha_zero" / "monitoring" / "Agent.md").read_text(encoding="utf-8")
    assert "15 metrics" in c4_text and "14 metrics" not in c4_text
    assert "`score_rate`" in c4_text
    assert "15 metrics" in mon_agent and "`score_rate`" in mon_agent


def test_governance_validators_and_single_source_walk_invariants(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Debt measurement shares AST discovery and Agent.md warning baseline is zero."""
    from scripts import measure_debt as md
    from scripts import validate_agents as va

    assert md.iter_python_files is aig.iter_python_files
    assert md.SKIP_PARTS is aig.SKIP_PARTS

    repo_root = Path(__file__).resolve().parents[2]
    baseline = json.loads(
        (repo_root / "tests" / "governance" / "agent_warning_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    assert baseline["max_warnings"] == 0
    assert baseline["docs_with_warnings"] == []

    rc = va.main(["--json", "--max-warnings", str(baseline["max_warnings"])])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["warning_count"] == 0
    assert payload["issues"] == []
