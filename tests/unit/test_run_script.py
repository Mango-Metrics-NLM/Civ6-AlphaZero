import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# The runner scripts live in scripts/, which is not on sys.path; bootstrap it the way
# tests/scripts/ does. The module name stays bare so existing mock.patch targets hold.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run_civ6_ai import main  # noqa: E402


@pytest.fixture
def mock_subprocess():
    with patch("subprocess.run") as mock:
        yield mock


@pytest.fixture
def mock_sys_exit():
    with patch("sys.exit") as mock:
        yield mock


def test_run_script_defaults(mock_subprocess, mock_sys_exit):
    """Test that default arguments construct the correct command."""
    with patch.object(sys, "argv", ["run_civ6_ai.py"]):
        main()

    mock_subprocess.assert_called_once()
    args, _ = mock_subprocess.call_args
    cmd = args[0]

    assert cmd[0] == sys.executable
    assert "-m" in cmd
    assert "alpha_zero.training_civ6" in cmd
    assert "--num_games=10" in cmd
    assert "--num_iterations=10" in cmd
    assert "--log_dir=logs/civ6" in cmd
    assert "--debug" not in cmd


def test_run_script_debug(mock_subprocess, mock_sys_exit):
    """Test that --debug flag is passed through."""
    with patch.object(sys, "argv", ["run_civ6_ai.py", "--debug"]):
        main()

    args, _ = mock_subprocess.call_args
    cmd = args[0]
    assert "--debug" in cmd


def test_run_script_custom_log_dir(mock_subprocess, mock_sys_exit):
    """Test custom log directory."""
    with patch.object(sys, "argv", ["run_civ6_ai.py", "--log-dir=custom_logs"]):
        main()

    args, _ = mock_subprocess.call_args
    cmd = args[0]
    assert "--log_dir=custom_logs" in cmd


def test_run_script_passthrough(mock_subprocess, mock_sys_exit):
    """Test that unknown args are passed through."""
    with patch.object(sys, "argv", ["run_civ6_ai.py", "--some-arg=value"]):
        main()

    args, _ = mock_subprocess.call_args
    cmd = args[0]
    assert "--some-arg=value" in cmd


def test_run_script_failure(mock_subprocess, mock_sys_exit):
    """Test error handling."""
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "cmd")

    with patch.object(sys, "argv", ["run_civ6_ai.py"]):
        main()

    mock_sys_exit.assert_called_with(1)
