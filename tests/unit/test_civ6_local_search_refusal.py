"""The live search mode refuses until an action translator exists (GM-B5, Part D).

``--mcts`` in ``--mode civ6`` builds ``Civ6MctsPolicy`` over a MicroCiv simulator whose
observation (13 channels) and action space (376 actions) are not the live network's (84
channels, 88,562 actions), and whose action indexes mean other things in the live action
converter. Every turn of such a run raises and ends the turn unplayed. Until the bridge
action translator (GM-B6) exists the mode refuses, before it touches the game, the network
or the IPC directory; the gap that makes it unusable (GM-B5) is named in the message.

Side effects are not mocked away but made loud: a recorder that raises ``BaseException``
(which ``main``'s own ``except Exception`` cannot swallow) replaces each of them, so a run
that gets past the refusal fails the test where it got to.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from alpha_zero import civ6_local
from alpha_zero.bridge import live_bridge
from alpha_zero.bridge.agent_loop import GreedyNetworkPolicy
from alpha_zero.bridge.mcts_policy import Civ6MctsPolicy
from alpha_zero.civ6_local import Civ6LocalConfig, create_parser, main

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_TOKENS = ("GM-B6", "GM-B5", "--mcts")


class _ReachedSideEffect(BaseException):
    """Raised by a recorder: the run went past where the refusal should have stopped it."""


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(civ6_local.ENV_PREFIX):
            monkeypatch.delenv(name)
    monkeypatch.delenv("CIV6_BRIDGE_TRANSPORT", raising=False)


@pytest.fixture
def side_effects(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make the first side effect of a bridge run raise, and record that it was reached."""
    reached: list[str] = []

    def network(*args: Any, **kwargs: Any) -> Any:
        reached.append("_build_network_for_bridge")
        raise _ReachedSideEffect("_build_network_for_bridge")

    def bridge(*args: Any, **kwargs: Any) -> Any:
        reached.append("LiveBridge")
        raise _ReachedSideEffect("LiveBridge")

    monkeypatch.setattr(civ6_local, "_build_network_for_bridge", network)
    monkeypatch.setattr(live_bridge, "LiveBridge", bridge)
    return reached


def _bridge_args(tmp_path: Path) -> list[str]:
    return [
        "--bridge-ipc-dir",
        str(tmp_path / "ipc"),
        "--civ6-log-path",
        str(tmp_path / "Lua.log"),
    ]


def _call(fn: Any, *args: Any) -> Any:
    """Run ``fn``; a run that reaches a side effect is a failed test, not an error."""
    try:
        return fn(*args)
    except _ReachedSideEffect as reached:
        pytest.fail(f"the search mode got past the refusal and reached {reached}")


def _error_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]


class TestRefusalText:
    @pytest.mark.parametrize(
        ("mode", "use_mcts", "refused"),
        [
            ("civ6", True, True),
            ("civ6", False, False),
            ("microciv", True, False),
            ("microciv", False, False),
            ("web", True, False),
            ("web", False, False),
        ],
    )
    def test_only_the_live_bridge_search_mode_is_refused(
        self, mode: str, use_mcts: bool, refused: bool
    ) -> None:
        text = civ6_local.search_mode_refusal(Civ6LocalConfig(mode=mode, use_mcts=use_mcts))
        if refused:
            assert isinstance(text, str)
        else:
            assert text is None

    def test_the_message_is_one_line_naming_both_gaps_and_the_flag(self) -> None:
        text = civ6_local.search_mode_refusal(Civ6LocalConfig(mode="civ6", use_mcts=True))
        assert text is not None
        assert text.strip() == text
        assert "\n" not in text
        for token in REQUIRED_TOKENS:
            assert token in text

    def test_other_options_do_not_change_what_is_refused(self) -> None:
        for kwargs in ({"num_simulations": 1}, {"board_size": 8}, {"checkpoint_path": "x.pt"}):
            cfg = Civ6LocalConfig(mode="civ6", use_mcts=True, **kwargs)
            assert civ6_local.search_mode_refusal(cfg) is not None


class TestMain:
    def test_search_mode_returns_one_and_logs_the_message(
        self,
        tmp_path: Path,
        side_effects: list[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        argv = ["--mode", "civ6", "--mcts", *_bridge_args(tmp_path)]
        with caplog.at_level(logging.ERROR):
            rc = _call(main, argv)

        assert rc == 1
        assert side_effects == []
        assert not (tmp_path / "ipc").exists()
        config = civ6_local.build_config(create_parser().parse_args(argv))
        text = civ6_local.search_mode_refusal(config)
        assert text is not None
        messages = _error_messages(caplog)
        assert any(text in m for m in messages), messages
        for token in REQUIRED_TOKENS:
            assert any(token in m for m in messages), token

    def test_dry_run_of_the_search_mode_is_refused_too(
        self,
        tmp_path: Path,
        side_effects: list[str],
        caplog: pytest.LogCaptureFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with caplog.at_level(logging.ERROR):
            rc = _call(main, ["--mode", "civ6", "--mcts", "--dry-run", *_bridge_args(tmp_path)])
        out = capsys.readouterr().out

        assert rc == 1
        assert "Configuration valid" not in out
        assert any("GM-B6" in m for m in _error_messages(caplog))

    @pytest.mark.parametrize("flags", [[], ["--no-mcts"]])
    def test_the_greedy_dry_run_is_unchanged(
        self,
        flags: list[str],
        tmp_path: Path,
        side_effects: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = _call(main, ["--mode", "civ6", *flags, "--dry-run", *_bridge_args(tmp_path)])
        out = capsys.readouterr().out

        assert rc == 0
        assert "Configuration valid" in out
        assert "use_mcts: False" in out
        assert side_effects == []

    @pytest.mark.parametrize("mode", ["microciv", "web"])
    def test_other_modes_accept_the_flag(
        self, mode: str, side_effects: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = _call(main, ["--mode", mode, "--mcts", "--dry-run"])
        out = capsys.readouterr().out

        assert rc == 0
        assert "Configuration valid" in out
        assert "use_mcts: True" in out

    def test_the_preset_is_refused_with_the_flag(
        self, tmp_path: Path, side_effects: list[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR):
            rc = _call(main, ["--preset", "civ6", "--mcts", "--dry-run"])
        assert rc == 1
        assert any("GM-B6" in m for m in _error_messages(caplog))


class TestEnvironmentSwitch:
    def test_env_var_with_the_mode_flag_is_refused(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        side_effects: list[str],
        caplog: pytest.LogCaptureFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("CIV6AZ_LOCAL_USE_MCTS", "1")
        with caplog.at_level(logging.ERROR):
            rc = _call(main, ["--mode", "civ6", "--dry-run", *_bridge_args(tmp_path)])

        assert rc == 1
        assert "Configuration valid" not in capsys.readouterr().out
        assert any("GM-B6" in m for m in _error_messages(caplog))

    def test_env_var_and_env_mode_are_refused_without_flags(
        self,
        monkeypatch: pytest.MonkeyPatch,
        side_effects: list[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("CIV6AZ_LOCAL_MODE", "civ6")
        monkeypatch.setenv("CIV6AZ_LOCAL_USE_MCTS", "true")
        with caplog.at_level(logging.ERROR):
            rc = _call(main, ["--dry-run"])

        assert rc == 1
        assert any("GM-B6" in m for m in _error_messages(caplog))

    def test_no_mcts_on_the_command_line_wins_over_the_variable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        side_effects: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("CIV6AZ_LOCAL_USE_MCTS", "1")
        rc = _call(main, ["--mode", "civ6", "--no-mcts", "--dry-run", *_bridge_args(tmp_path)])

        assert rc == 0
        assert "use_mcts: False" in capsys.readouterr().out

    def test_the_variable_does_not_touch_other_modes(
        self, monkeypatch: pytest.MonkeyPatch, side_effects: list[str]
    ) -> None:
        monkeypatch.setenv("CIV6AZ_LOCAL_USE_MCTS", "1")
        assert _call(main, ["--mode", "microciv", "--dry-run"]) == 0


class TestRunCiv6Bridge:
    def test_search_mode_returns_one_before_any_side_effect(
        self, tmp_path: Path, side_effects: list[str]
    ) -> None:
        cfg = Civ6LocalConfig(
            mode="civ6",
            use_mcts=True,
            bridge_ipc_dir=str(tmp_path / "ipc"),
            civ6_log_path=str(tmp_path / "Lua.log"),
        )
        rc = _call(civ6_local.run_civ6_bridge, cfg)

        assert rc == 1
        assert side_effects == []
        assert not (tmp_path / "ipc").exists(), "the IPC directory was created"
        assert not (tmp_path / "Lua.log").exists()

    def test_greedy_mode_is_not_refused(self, tmp_path: Path, side_effects: list[str]) -> None:
        """The refusal is the search mode's alone: the greedy run still builds its network."""
        cfg = Civ6LocalConfig(
            mode="civ6",
            use_mcts=False,
            bridge_ipc_dir=str(tmp_path / "ipc"),
            civ6_log_path=str(tmp_path / "Lua.log"),
        )
        with pytest.raises(_ReachedSideEffect, match="_build_network_for_bridge"):
            civ6_local.run_civ6_bridge(cfg)
        assert side_effects == ["_build_network_for_bridge"]


class TestPolicyChoiceIsUnchanged:
    """The refusal sits in front of the choice; the choice itself is still made as before."""

    def test_search_policy_when_asked_for(self) -> None:
        cfg = Civ6LocalConfig(mode="civ6", use_mcts=True, num_simulations=4)
        assert isinstance(civ6_local._build_bridge_policy(cfg, object()), Civ6MctsPolicy)

    def test_greedy_policy_otherwise(self) -> None:
        cfg = Civ6LocalConfig(mode="civ6", use_mcts=False)
        assert isinstance(civ6_local._build_bridge_policy(cfg, object()), GreedyNetworkPolicy)


class TestHelp:
    def test_mcts_help_names_the_gap(self) -> None:
        actions = [a for a in create_parser()._actions if "--mcts" in a.option_strings]
        assert len(actions) == 1
        assert "GM-B6" in (actions[0].help or "")


class TestDocumentedCommand:
    def test_the_documented_command_exits_one_and_names_the_gap(self, tmp_path: Path) -> None:
        """``python -m alpha_zero.civ6_local --mode civ6 --mcts``, in a scratch IPC directory.

        Before the refusal this command starts the bridge and waits for a game that is not
        there; the timeout turns that into a failure instead of a hang.
        """
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(civ6_local.ENV_PREFIX) and k != "CIV6_BRIDGE_TRANSPORT"
        }
        env["PYTHONPATH"] = str(ROOT)
        env.setdefault("OMP_NUM_THREADS", "2")
        command = [
            sys.executable,
            "-m",
            "alpha_zero.civ6_local",
            "--mode",
            "civ6",
            "--mcts",
            *_bridge_args(tmp_path),
        ]
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pytest.fail("the documented command started the bridge instead of refusing")

        output = result.stdout + result.stderr
        assert result.returncode == 1, output[-2000:]
        assert "GM-B6" in output
        assert not (tmp_path / "ipc").exists()
