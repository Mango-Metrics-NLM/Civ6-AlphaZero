"""Unit tests for ``scripts/check_release_readiness.py`` (R1-R6 positive & negative)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import check_release_readiness as crr

SHA_ONE = "a" * 40


def _seed_ready_repo(root: Path) -> None:
    for rel in crr.REQUIRED_PUBLIC_FILES:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {rel}\n\nPlaceholder content.\n", encoding="utf-8")

    (root / "README.md").write_text(
        "# Civ6 AlphaZero\n\n**Status:** Active research.\n\nThis project is not affiliated with Firaxis.\n",
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [2.0.1] - 2026-11-25\n- Release.\n",
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(
        'cff-version: 1.2.0\nversion: "2.0.1"\ndate-released: "2026-11-25"\n',
        encoding="utf-8",
    )
    (root / "SECURITY.md").write_text(
        "# Security\n\n## Supported versions\n| 2.0.x | Yes |\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "civ6-alphazero"\nversion = "2.0.1"\n'
        'description = "Test"\nreadme = "README.md"\nrequires-python = ">=3.11"\n'
        '[project.urls]\nHomepage = "https://example.invalid"\n'
        "[tool.coverage.report]\nfail_under = 85\n",
        encoding="utf-8",
    )
    init_path = root / "alpha_zero" / "__init__.py"
    init_path.parent.mkdir(parents=True, exist_ok=True)
    init_path.write_text('__version__ = "2.0.1"\n', encoding="utf-8")

    (root / ".gitignore").write_text(
        "\n".join(crr.REQUIRED_GITIGNORE_TOKENS) + "\n", encoding="utf-8"
    )
    (root / ".dockerignore").write_text(
        "\n".join(crr.REQUIRED_DOCKERIGNORE_TOKENS) + "\n", encoding="utf-8"
    )
    (root / ".gitleaks.toml").write_text(
        '[allowlist]\npaths = ["^eval_results/"]\n', encoding="utf-8"
    )
    (root / "coverage-floors.json").write_text(
        json.dumps({"floors": {"alpha_zero/cloud": {"minimum": 59, "measured": 60.2}}}),
        encoding="utf-8",
    )
    baseline = root / "docs" / "evidence" / "tech_debt_baseline" / "baseline.json"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(
        json.dumps(
            {
                "files_over_1000_lines": 22,
                "noqa_comments": 70,
                "pytest_skip_sites": 150,
                "unsafe_torch_load_calls": 8,
            }
        ),
        encoding="utf-8",
    )
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "release_play_path.txt").write_text(
        "alpha_zero/training/**\n", encoding="utf-8"
    )
    (root / "scripts" / "local_muzero_launcher.py").write_text(
        "import subprocess\ndef run(cmd):\n    return subprocess.run(cmd, shell=False)\n",
        encoding="utf-8",
    )
    wf_dir = root / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    for wf in ("ci.yml", "release.yml", "security-scan.yml"):
        (wf_dir / wf).write_text(
            f"jobs:\n  test:\n    steps:\n      - uses: actions/checkout@{SHA_ONE}\n",
            encoding="utf-8",
        )


def test_seed_repo_passes_with_skip_git_and_json_is_sorted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_ready_repo(tmp_path)
    rc = crr.main(["--root", str(tmp_path), "--json", "--skip-git", "--verbose"])
    assert rc == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["ok"] is True
    assert parsed["summary"] == "release-readiness: OK"
    assert [c["code"] for c in parsed["checks"]] == ["R1", "R2", "R3", "R4", "R5", "R6"]
    assert parsed["checks"][-1]["status"] == "SKIP"
    assert "[PASS] R1 (public-surface)" in captured.err


def test_r2_fails_on_version_mismatch(tmp_path: Path) -> None:
    _seed_ready_repo(tmp_path)
    (tmp_path / "alpha_zero" / "__init__.py").write_text(
        '__version__ = "2.0.0"\n', encoding="utf-8"
    )
    result = crr.evaluate_release_readiness(tmp_path, skip_git=True)
    assert result["ok"] is False
    r2 = next(c for c in result["checks"] if c["code"] == "R2")
    assert r2["status"] == "FAIL"
    assert any("2.0.0" in d for d in r2["details"])


def test_r3_fails_on_missing_ignore_or_catchall_gitleaks(tmp_path: Path) -> None:
    _seed_ready_repo(tmp_path)
    (tmp_path / ".gitleaks.toml").write_text('[allowlist]\npaths = [".*"]\n', encoding="utf-8")
    result = crr.evaluate_release_readiness(tmp_path, skip_git=True)
    assert result["ok"] is False
    r3 = next(c for c in result["checks"] if c["code"] == "R3")
    assert r3["status"] == "FAIL"
    assert any("overly broad" in d for d in r3["details"])


def test_r4_fails_when_measured_below_floor(tmp_path: Path) -> None:
    _seed_ready_repo(tmp_path)
    (tmp_path / "coverage-floors.json").write_text(
        json.dumps({"floors": {"alpha_zero/cloud": {"minimum": 60, "measured": 58.0}}}),
        encoding="utf-8",
    )
    result = crr.evaluate_release_readiness(tmp_path, skip_git=True)
    r4 = next(c for c in result["checks"] if c["code"] == "R4")
    assert r4["status"] == "FAIL"
    assert any("measured" in d for d in r4["details"])


def test_r5_fails_on_unpinned_workflow_and_shell_true_launcher(tmp_path: Path) -> None:
    _seed_ready_repo(tmp_path)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text(
        "jobs:\n  test:\n    steps:\n      - uses: actions/checkout@v4\n", encoding="utf-8"
    )
    (tmp_path / "scripts" / "local_muzero_launcher.py").write_text(
        "import subprocess\nsubprocess.Popen(['echo'], shell=True)\n", encoding="utf-8"
    )
    result = crr.evaluate_release_readiness(tmp_path, skip_git=True)
    r5 = next(c for c in result["checks"] if c["code"] == "R5")
    assert r5["status"] == "FAIL"
    assert any("unpinned GitHub Action" in d for d in r5["details"])
    assert any("shell=False" in d for d in r5["details"])


def test_r6_fails_cleanly_on_non_git_directory(tmp_path: Path) -> None:
    _seed_ready_repo(tmp_path)
    rc = crr.main(["--root", str(tmp_path), "--quiet"])
    assert rc == 1
