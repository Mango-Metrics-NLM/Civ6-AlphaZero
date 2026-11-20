"""``alpha_zero.rules`` is a leaf: stdlib, ``yaml`` and itself only (GM-K1)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import alpha_zero.rules as rules_pkg

RULES_DIR = Path(rules_pkg.__file__).resolve().parent
OWN = "alpha_zero.rules"
ALLOWED_THIRD_PARTY = {"yaml"}
DYNAMIC_IMPORTS = {"__import__", "import_module"}


def _allowed(module: str) -> bool:
    root = module.split(".")[0]
    if root in sys.stdlib_module_names or root in ALLOWED_THIRD_PARTY:
        return True
    return module == OWN or module.startswith(OWN + ".")


def import_violations(path: Path) -> list[str]:
    """Every import in ``path`` outside the allowed set, as ``file:line imports X``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level >= 2:
                names = ["." * node.level + (node.module or "")]
            elif node.level == 0:
                names = [node.module or ""]
        elif isinstance(node, ast.Call):
            func = node.func
            callee = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if callee in DYNAMIC_IMPORTS:
                names = [f"<dynamic {callee}>"]
        found += [f"{path.name}:{node.lineno} imports {n}" for n in names if not _allowed(n)]
    return found


def test_every_module_is_discovered() -> None:
    names = {p.name for p in RULES_DIR.glob("*.py")}
    assert {"spec.py", "loader.py", "validate.py", "queries.py", "registry.py"} <= names
    assert {"builtin.py", "__main__.py", "__init__.py"} <= names
    assert {"pacing.py", "gameinfo.py"} <= names


def test_rules_package_imports_only_stdlib_yaml_and_itself() -> None:
    bad: list[str] = []
    for path in sorted(RULES_DIR.rglob("*.py")):
        bad += import_violations(path)
    assert not bad, "rules is a leaf package; offending imports: " + "; ".join(bad)


def test_checker_flags_a_planted_third_party_import(tmp_path: Path) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text("import os\nimport numpy\n")
    assert import_violations(planted) == ["planted.py:2 imports numpy"]


def test_checker_flags_a_planted_sibling_package_import(tmp_path: Path) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text("from alpha_zero.utils import x\n")
    assert import_violations(planted) == ["planted.py:1 imports alpha_zero.utils"]


def test_checker_flags_function_local_dynamic_and_escaping_relative(tmp_path: Path) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text(
        "def f():\n    import torch\n    return __import__('numpy')\nfrom ..core import y\n"
    )
    found = import_violations(planted)
    assert len(found) == 3, found
    assert any("torch" in item for item in found)
    assert any("dynamic" in item for item in found)
    assert any("..core" in item for item in found)


def test_checker_accepts_the_allowed_forms(tmp_path: Path) -> None:
    ok = tmp_path / "ok.py"
    ok.write_text(
        "from __future__ import annotations\nimport yaml\nimport json\n"
        "from alpha_zero.rules.spec import X\nfrom . import spec\nfrom importlib import resources\n"
    )
    assert import_violations(ok) == []
