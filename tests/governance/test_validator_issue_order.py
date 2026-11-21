"""The validators' own sorting and serialising, on reports that have something to sort.

On a clean checkout ``validate_kb``, ``validate_openspec`` and ``check_doc_sync`` report
nothing, so "the output is sorted" holds whether or not they sort. Here each validator's
collector is replaced by one that hands back several findings in a deliberately wrong
order, and the validator's real ``main`` has to put them, and the keys of each JSON
object, in order. Removing the production ``issues.sort`` or ``sort_keys=True`` fails
these tests (checked by doing so).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"

#: (file, line, code, message), in an order no sort key would produce.
SHUFFLED = [
    ("zz/last.md", 9, "X2", "second code on the same line"),
    ("aa/first.md", 30, "X1", "later line"),
    ("zz/last.md", 9, "X1", "first code on the same line"),
    ("aa/first.md", None, "X9", "no line sorts before any line"),
    ("aa/first.md", 4, "X1", "b message"),
    ("aa/first.md", 4, "X1", "a message"),
]
EXPECTED = sorted(SHUFFLED, key=lambda row: (row[0], row[1] or 0, row[2], row[3]))


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_order_{name}", SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _issues(module: ModuleType) -> list[Any]:
    return [
        module.Issue(code=code, message=message, file=REPO_ROOT / file, line=line)
        for file, line, code, message in SHUFFLED
    ]


def _kb(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, list[str]]:
    module = _load("validate_kb")
    monkeypatch.setattr(
        module, "_validate_ledger", lambda _ledger, issues, **_kw: issues.extend(_issues(module))
    )
    monkeypatch.setattr(module, "_validate_sessions", lambda _issues_list: None)
    return module, ["--no-git"]


def _openspec(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, list[str]]:
    module = _load("validate_openspec")
    monkeypatch.setattr(
        module, "_validate_all", lambda issues, **_kw: issues.extend(_issues(module))
    )
    return module, []


def _doc_sync(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, list[str]]:
    module = _load("check_doc_sync")
    monkeypatch.setattr(
        module, "_check_versions", lambda _root, _pyproject, issues: issues.extend(_issues(module))
    )
    return module, []


VALIDATORS = {"validate_kb": _kb, "validate_openspec": _openspec, "check_doc_sync": _doc_sync}


@pytest.mark.parametrize("name", sorted(VALIDATORS))
def test_json_findings_and_keys_come_out_sorted(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module, extra = VALIDATORS[name](monkeypatch)
    assert module.main([*extra, "--json"]) == 1
    text = capsys.readouterr().out
    payload = json.loads(text)
    ours = [row for row in payload if row["code"].startswith("X")]
    got = [(Path(r["file"]).as_posix(), r["line"], r["code"], r["message"]) for r in ours]
    expected = [(f, line, code, message) for f, line, code, message in EXPECTED]
    assert [g[1:] for g in got] == [e[1:] for e in expected]
    assert [g[0].endswith(e[0]) for g, e in zip(got, expected, strict=True)] == [True] * len(got)
    # Keys of every object in the order sort_keys=True writes them.
    for block in re.findall(r"\{[^{}]*\}", text):
        keys = re.findall(r'^\s*"([a-z_]+)":', block, flags=re.M)
        assert keys == sorted(keys), f"{name}: keys not sorted in {block!r}"
    assert json.dumps(payload, indent=2, sort_keys=True) == text.strip()


@pytest.mark.parametrize("name", sorted(VALIDATORS))
def test_text_findings_come_out_sorted(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module, extra = VALIDATORS[name](monkeypatch)
    assert module.main(extra) == 1
    lines = [line for line in capsys.readouterr().err.splitlines() if " X" in line]
    codes_and_messages = [line.split(": ", 1)[1] for line in lines]
    assert codes_and_messages == [f"{code} {message}" for _f, _l, code, message in EXPECTED]
