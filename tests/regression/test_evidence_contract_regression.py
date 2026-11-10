"""Regression gate for the evidence-artifact contract (``docs/evidence/SCHEMA.md``).

Every committed measurement under ``docs/evidence/`` must say what code produced
it and whether that code predates the 2026-09-04 search fixes. Without this,
pre-fix numbers (the encoder A/B, the iteration-zero run, the draw sweep) were
quoted in the README, the runbook and the model card as if they measured
strength. Each check is a pure predicate over paths or parsed JSON so that the
negative controls can drive the same predicate with planted defects.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = REPO_ROOT / "docs" / "evidence"
INDEX = EVIDENCE / "README.md"
FIX_COMMIT = "e9d0572"
EXEMPT_DIRS = frozenset({"tech_debt_baseline"})
EPOCHS = frozenset({"pre-2026-09-04", "post-2026-09-04"})
REQUIRED = (
    "schema_version",
    "git_sha",
    "generated_utc",
    "search_epoch",
    "seed",
    "command",
    "hardware",
    "games_total",
    "claims",
    "non_claims",
)
BANNER_MARK = "**Pre-fix artifact (superseded).**"
_SHA = re.compile(r"^[0-9a-f]{7,40}$")


def evidence_dirs(root: Path = EVIDENCE) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name not in EXEMPT_DIRS)


def report_files(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.glob("*.md")
        if p.name == "report.md" or p.name.endswith("_report.md") or p.name == "README.md"
    )


def evidence_blocks(directory: Path) -> list[tuple[Path, dict]]:
    found = []
    for p in sorted(directory.glob("*.json")):
        data = json.loads(p.read_text())
        if isinstance(data, dict) and isinstance(data.get("evidence"), dict):
            found.append((p, data["evidence"]))
    return found


def block_problems(block: dict) -> list[str]:
    """Return every contract violation in one evidence block."""
    problems = [f"missing '{k}'" for k in REQUIRED if k not in block]
    if "git_sha" in block and not _SHA.match(str(block["git_sha"])):
        problems.append("git_sha is not 7-40 hex")
    if block.get("search_epoch") not in EPOCHS:
        problems.append(f"search_epoch {block.get('search_epoch')!r} not in {sorted(EPOCHS)}")
    if block.get("seed") is None and not block.get("seed_null_reason"):
        problems.append("seed is null without seed_null_reason")
    if not isinstance(block.get("games_total"), int):
        problems.append("games_total is not an int")
    for key in ("claims", "non_claims"):
        if not isinstance(block.get(key), list):
            problems.append(f"{key} is not a list")
    return problems


def indexed_dirs(index_text: str) -> set[str]:
    return set(re.findall(r"\]\(([a-z0-9_]+)/", index_text))


def _is_ancestor(sha: str, of: str) -> bool | None:
    if shutil.which("git") is None:
        return None
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=REPO_ROOT, capture_output=True
    )
    if probe.returncode != 0:
        return None  # shallow clone or unknown object: cannot decide
    res = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, of], cwd=REPO_ROOT, capture_output=True
    )
    return res.returncode == 0


class TestEvidenceContract:
    def test_every_dir_has_exactly_one_report(self) -> None:
        dirs = evidence_dirs()
        assert dirs, "no evidence directories found -- the check would be vacuous"
        bad = {d.name: [p.name for p in report_files(d)] for d in dirs if len(report_files(d)) != 1}
        assert not bad, f"directories without exactly one report: {bad}"

    def test_every_dir_has_a_valid_evidence_block(self) -> None:
        failures: dict[str, list[str]] = {}
        for d in evidence_dirs():
            blocks = evidence_blocks(d)
            if not blocks:
                failures[d.name] = ["no JSON file carries an 'evidence' block"]
                continue
            for path, block in blocks:
                problems = block_problems(block)
                if problems:
                    failures[str(path.relative_to(REPO_ROOT))] = problems
        assert not failures, failures

    def test_every_dir_is_indexed(self) -> None:
        listed = indexed_dirs(INDEX.read_text())
        missing = sorted(d.name for d in evidence_dirs() if d.name not in listed)
        assert not missing, f"not listed in docs/evidence/README.md: {missing}"

    def test_pre_fix_reports_carry_the_banner(self) -> None:
        checked = 0
        for d in evidence_dirs():
            epochs = {b["search_epoch"] for _, b in evidence_blocks(d) if "search_epoch" in b}
            if "pre-2026-09-04" not in epochs:
                continue
            for report in report_files(d):
                checked += 1
                assert BANNER_MARK in report.read_text(), f"{report} lacks the superseded banner"
        assert checked > 0, "no pre-fix report was inspected -- the check would be vacuous"

    def test_pre_fix_sha_is_an_ancestor_of_the_fix_commit(self) -> None:
        decided = 0
        for d in evidence_dirs():
            for path, block in evidence_blocks(d):
                if block.get("search_epoch") != "pre-2026-09-04":
                    continue
                verdict = _is_ancestor(str(block["git_sha"]), FIX_COMMIT)
                if verdict is None:
                    continue
                decided += 1
                assert verdict, (
                    f"{path}: labelled pre-fix but {block['git_sha']} is not before {FIX_COMMIT}"
                )
        if decided == 0:
            pytest.skip("git history unavailable (shallow clone); ancestry not checked")


class TestNegativeControls:
    """Each planted defect must be caught by the same predicate the gate uses."""

    def test_missing_git_sha_is_reported(self) -> None:
        block = dict.fromkeys(REQUIRED, 1)
        block.update(search_epoch="pre-2026-09-04", claims=[], non_claims=[], seed=1)
        del block["git_sha"]
        assert "missing 'git_sha'" in block_problems(block)

    def test_bad_epoch_is_reported(self) -> None:
        block = dict.fromkeys(REQUIRED, 1)
        block.update(git_sha="abc1234", search_epoch="post-fix", claims=[], non_claims=[], seed=1)
        assert any("search_epoch" in p for p in block_problems(block))

    def test_null_seed_needs_a_reason(self) -> None:
        block = dict.fromkeys(REQUIRED, 1)
        block.update(
            git_sha="abc1234", search_epoch="pre-2026-09-04", claims=[], non_claims=[], seed=None
        )
        assert "seed is null without seed_null_reason" in block_problems(block)
        block["seed_null_reason"] = "predates --seed"
        assert "seed is null without seed_null_reason" not in block_problems(block)

    def test_unindexed_dir_is_detected(self, tmp_path: Path) -> None:
        (tmp_path / "planted_run").mkdir()
        assert "planted_run" not in indexed_dirs("| [`other/`](other/report.md) |")

    def test_dir_without_report_is_detected(self, tmp_path: Path) -> None:
        d = tmp_path / "planted_run"
        d.mkdir()
        (d / "notes.md").write_text("# not a report\n")
        assert report_files(d) == []
