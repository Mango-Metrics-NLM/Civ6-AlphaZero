"""``python -m alpha_zero.rules`` and what the wheel ships."""

from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from alpha_zero.rules import builtin
from alpha_zero.rules.__main__ import main
from alpha_zero.rules.builtin import BUILTIN, read_builtin_text
from alpha_zero.rules.loader import content_sha256, parse_ruleset
from tests.unit.test_rules._fixtures import BASE, mutate

REPO_ROOT = Path(__file__).resolve().parents[3]


def write(tmp_path: Path, text: str, name: str = "r.yaml") -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_validate_builtin_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", "science-v1-draft"]) == 0
    out = capsys.readouterr().out
    assert "science-v1-draft: valid" in out
    assert "available at the start: agriculture, mining, writing" in out
    assert "index fingerprint:" in out


def test_hash_builtin_prints_id_and_pin(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["hash", "science-v1-draft"]) == 0
    assert (
        capsys.readouterr().out.strip() == f"science-v1-draft {BUILTIN['science-v1-draft'].sha256}"
    )


def test_validate_and_hash_a_valid_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = write(tmp_path, BASE)
    assert main(["validate", target]) == 0
    assert "tiny-test: valid" in capsys.readouterr().out
    assert main(["hash", target]) == 0
    expected = content_sha256(parse_ruleset(BASE))
    assert capsys.readouterr().out.strip() == f"tiny-test {expected}"


def test_unresolved_prerequisite_exits_one_and_names_the_rule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = write(tmp_path, mutate("prerequisites: [alpha]", "prerequisites: [ghost]"))
    assert main(["validate", target]) == 1
    err = capsys.readouterr().err
    assert "unresolved-prerequisite" in err and "'ghost'" in err


def test_every_violated_rule_is_listed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    text = mutate("prerequisites: [alpha]", "prerequisites: [ghost]")
    text = mutate("    cost: 10\n", "    cost: 0\n", text)
    assert main(["validate", write(tmp_path, text)]) == 1
    err = capsys.readouterr().err
    assert "unresolved-prerequisite" in err and "non-positive-cost" in err


def test_malformed_yaml_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", write(tmp_path, "key: [unclosed")]) == 1
    assert "error:" in capsys.readouterr().err
    assert main(["hash", write(tmp_path, mutate("cost: 20", "cost: 2.5"), "f.yaml")]) == 1
    assert "$.techs[1].cost" in capsys.readouterr().err


def test_hash_mismatch_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    text = read_builtin_text("science-v1-draft").replace("cost: 180", "cost: 181", 1)
    monkeypatch.setattr(builtin, "read_builtin_text", lambda ruleset_id: text)
    assert main(["validate", "science-v1-draft"]) == 1
    assert "hashes to" in capsys.readouterr().err


@pytest.mark.parametrize("target", ["no-such-ruleset", "science-v1", "does/not/exist.yaml"])
def test_unknown_target_exits_two(target: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", target]) == 2
    err = capsys.readouterr().err
    assert target in err and "science-v1-draft" in err


def test_a_directory_is_not_a_target(tmp_path: Path) -> None:
    assert main(["hash", str(tmp_path)]) == 2


@pytest.mark.parametrize("argv", [[], ["validate"], ["frobnicate", "x"], ["validate", "a", "b"]])
def test_usage_errors_exit_two(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2


def test_a_non_utf8_file_is_a_clean_failure_not_a_traceback(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"schema_version: 1\nid: \xff\xfe\n")
    assert main(["validate", str(path)]) == 1


def test_an_os_error_while_reading_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import alpha_zero.rules.__main__ as cli

    def boom(path: Path) -> None:
        raise PermissionError("denied-by-test")

    monkeypatch.setattr(cli, "load_ruleset", boom)
    assert main(["validate", write(tmp_path, BASE)]) == 1
    assert "denied-by-test" in capsys.readouterr().err


def test_a_broken_builtin_table_entry_is_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = BUILTIN["science-v1-draft"]
    monkeypatch.setitem(
        BUILTIN, "science-v1-draft", builtin.BuiltinRuleset(entry.resource, entry.sha256, "frozen")
    )
    assert main(["validate", "science-v1-draft"]) == 1


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.run(
        [sys.executable, "-m", "alpha_zero.rules", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=60,
    )


def test_python_dash_m_runs_for_real(tmp_path: Path) -> None:
    ok = _run("validate", "science-v1-draft")
    assert ok.returncode == 0 and "science-v1-draft: valid" in ok.stdout
    bad = _run("validate", write(tmp_path, mutate("prerequisites: [alpha]", "prerequisites: [x]")))
    assert bad.returncode == 1 and "unresolved-prerequisite" in bad.stderr
    assert _run("validate", "nope").returncode == 2
    assert _run().returncode == 2


# ---- packaging ------------------------------------------------------------------


def test_pyproject_ships_the_rule_data() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = config["tool"]["setuptools"]["package-data"]["alpha_zero"]
    assert "rules/data/*.yaml" in patterns
    assert "py.typed" in patterns
    root = REPO_ROOT / "alpha_zero"
    shipped = {str(p.relative_to(root)) for p in (root / "rules" / "data").glob("*.yaml")}
    assert shipped, "no data to ship"
    for entry in BUILTIN.values():
        rel = f"rules/data/{entry.resource}"
        assert any(fnmatch.fnmatch(rel, pat) for pat in patterns), rel
