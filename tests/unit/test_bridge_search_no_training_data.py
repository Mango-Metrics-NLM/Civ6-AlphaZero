"""GM-B6b (D), charter NG-1: playing with the search produces no training data.

Two kinds of evidence. (1) The loop plays a few game turns in a temporary working directory
and nothing appears there (no replay shard, checkpoint, log or evidence file). (2) The four
modules that carry the search into the bridge import no replay buffer, trainer or storage module:
checked on their text (module-level imports, with ``ast``) and on what importing them loads.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures.bridge_search_standin import play

ROOT = Path(__file__).resolve().parents[2]
BRIDGE = ROOT / "alpha_zero" / "bridge"
MODULES = ["command_policy", "search_translator", "mcts_policy", "policy_factory"]
FORBIDDEN_ROOTS = (
    "alpha_zero.storage",
    "alpha_zero.training",
    "alpha_zero.distributed",
    "alpha_zero.training_civ6",
    "alpha_zero.cloud",
)


def _forbidden(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in FORBIDDEN_ROOTS)


def module_level_imports(path: Path) -> list[str]:
    """Absolute names imported at module level (``if``/``try`` blocks count; functions do not)."""
    package = "alpha_zero.bridge"
    names: list[str] = []

    def visit(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                    module = f"{base}.{node.module}" if node.module else base
                else:
                    module = node.module or ""
                names.append(module)
                names.extend(f"{module}.{alias.name}" for alias in node.names)
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                for field in ("body", "orelse", "finalbody"):
                    visit(getattr(node, field, []))
                for handler in getattr(node, "handlers", []):
                    visit(handler.body)

    visit(ast.parse(path.read_text(encoding="utf-8")).body)
    return names


class TestNothingIsWritten:
    def test_a_few_game_turns_leave_the_working_directory_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runtime, _, loop, _, code = play(max_game_turns=3, max_commands_per_turn=4)
        assert loop.stats()["search_decisions"] >= 6, "premise: the search really played"
        assert loop.stats()["game_turns_played"] == 3
        assert int(runtime.eval("Civ6Stub.turn")) == 15
        assert code == 0
        assert list(tmp_path.rglob("*")) == []

    def test_the_check_would_see_a_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control: a file written by anything in the process is seen by the listing above."""
        monkeypatch.chdir(tmp_path)
        play(max_game_turns=1, max_commands_per_turn=1)
        (tmp_path / "replay.pkl").write_bytes(b"x")
        assert [p.name for p in tmp_path.rglob("*")] == ["replay.pkl"]


class TestNoTrainingImports:
    @pytest.mark.parametrize("module", MODULES)
    def test_module_level_imports_name_no_training_or_storage_module(self, module: str) -> None:
        path = BRIDGE / f"{module}.py"
        assert path.is_file(), f"{path} does not exist"
        imported = module_level_imports(path)
        assert any(name.startswith("alpha_zero") for name in imported), "premise: it was read"
        assert [name for name in imported if _forbidden(name)] == []

    @pytest.mark.parametrize("module", MODULES)
    def test_importing_the_module_loads_none_of_them(self, module: str) -> None:
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            f"import alpha_zero.bridge.{module}\n"
            f"roots = {FORBIDDEN_ROOTS!r}\n"
            "bad = sorted(n for n in sys.modules if any(n == r or n.startswith(r + '.')"
            " for r in roots))\n"
            "print('LOADED', bad)\n"
        )
        # Use -E -P instead of -I so environment/cwd isolation holds without
        # disabling user site-packages (Windows --user installs of numpy/yaml).
        done = subprocess.run(
            [sys.executable, "-E", "-P", "-c", code], capture_output=True, text=True, timeout=100
        )
        assert done.returncode == 0, done.stderr[-2000:]
        assert done.stdout.strip().splitlines()[-1] == "LOADED []"

    def test_the_scan_sees_a_forbidden_import(self, tmp_path: Path) -> None:
        """Control: the same reader reports module-level, relative and from-imports, not bodies."""
        sample = tmp_path / "sample.py"
        sample.write_text(
            "import alpha_zero.storage.factory\n"
            "from alpha_zero import training_civ6\n"
            "from alpha_zero.distributed import actor\n"
            "try:\n    from alpha_zero.cloud.gcp import arena\nexcept ImportError:\n    pass\n"
            "def late():\n    from alpha_zero.storage.config import StorageConfig\n",
            encoding="utf-8",
        )
        found = [name for name in module_level_imports(sample) if _forbidden(name)]
        assert "alpha_zero.storage.factory" in found
        assert "alpha_zero.training_civ6" in found
        assert "alpha_zero.distributed.actor" in found
        assert "alpha_zero.cloud.gcp" in found
        assert "alpha_zero.storage.config" not in found, "an import inside a function is late"
