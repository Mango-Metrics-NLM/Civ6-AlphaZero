"""Tests for ``MCTS_*`` override reproduction, case-insensitivity, and bundle run-id safety."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.evaluation import reproduction as rp
from alpha_zero.evaluation.a1_protocol import NON_REPRODUCING_LABEL
from scripts import build_release_bundle as builder
from scripts import release_reproduction as rr
from scripts import verify_release_evidence as vre
from scripts.check_branch_direction import Git
from tests.scripts import release_evidence_fixture as fx
from tests.scripts.git_fixture import GitRepo
from tests.scripts.test_verify_release_evidence import verify
from tests.scripts.test_verify_release_evidence_reproduction import (
    A1,
    R2,
    SHA,
    attempt,
    entries,
    killed_at_once,
    non_reproducing_reasons,
    rerun,
    stored_record,
    verdicts,
)


def _write_minimal_run(run_dir: Path, card_id: Any, sha: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "card_id": card_id,
        "sha": sha,
        "status": "COMPLETED",
        "env_var_names": ["PATH"],
        "total_priced_games": 2,
        "invocations": [
            {
                "name": "train",
                "priced_games": 2,
                "exit_code": 0,
                "argv": ["python", "-m", "alpha_zero.training_civ6"],
                "launched_argv": ["python", "-m", "alpha_zero.training_civ6", "--max_games", "2"],
            }
        ],
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. EMPTY comparison after an MCTS_* override VOID
# ---------------------------------------------------------------------------


def test_empty_comparison_after_mcts_override_is_still_non_reproducing(tmp_path: Path) -> None:
    """Clearing ``MCTS_*`` overrides forces ``NON_REPRODUCING`` even on ``NOTHING TO COMPARE``."""
    specs = [killed_at_once(env_var_names=["MCTS_C_VISIT", "PATH"]), rerun()]
    release = fx.make_release(tmp_path, specs)
    assert stored_record(release)[rp.RECORD_STATUS] == rp.STATUS_NOTHING_TO_COMPARE
    assert non_reproducing_reasons(release) == [
        "MCTS_* overrides: the attempt had ['MCTS_C_VISIT'], the re-run []"
    ]
    assert verdicts(release) == ["VOID", "INCONCLUSIVE"]
    assert entries(release)[R2]["verdict"]["label"] == NON_REPRODUCING_LABEL
    assert release.bundle["learning_claimed"] is False
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out

    entries(release)[A1]["reproduction"]["outcome"] = rr.OUTCOME_EMPTY
    entries(release)[R2]["verdict"] = {"verdict": "PASS", "reasons": ["tampered"]}
    release.save()
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert "recomputed 'NON_REPRODUCING'" in out
    assert "stored verdict 'PASS' != recomputed 'INCONCLUSIVE'" in out


# ---------------------------------------------------------------------------
# 2. condition_differences & _mcts_overrides unit tests
# ---------------------------------------------------------------------------


class TestMctsConditions:
    def test_cleared_mcts_override_is_a_difference(self) -> None:
        first = attempt(record={"env_var_names": ["MCTS_C_VISIT", "PATH"]})
        second = attempt(R2, record={"env_var_names": ["PATH"]})
        assert rr.condition_differences(first, second) == [
            "MCTS_* overrides: the attempt had ['MCTS_C_VISIT'], the re-run []"
        ]

    def test_changed_and_added_mcts_overrides_are_differences(self) -> None:
        first = attempt(record={"env_var_names": ["MCTS_C_VISIT"]})
        second = attempt(R2, record={"env_var_names": ["MCTS_ROLLOUT_DEPTH"]})
        assert rr.condition_differences(first, second) == [
            "MCTS_* overrides: the attempt had ['MCTS_C_VISIT'], the re-run ['MCTS_ROLLOUT_DEPTH']"
        ]
        clean = attempt(record={"env_var_names": ["PATH"]})
        assert rr.condition_differences(clean, second) == [
            "MCTS_* overrides: the attempt had [], the re-run ['MCTS_ROLLOUT_DEPTH']"
        ]

    def test_case_insensitive_mcts_prefix_and_sorting(self) -> None:
        first = attempt(record={"env_var_names": ["mcts_c_visit", "MCTS_EPSILON", "path"]})
        second = attempt(R2, record={"env_var_names": ["PATH"]})
        assert rr.condition_differences(first, second) == [
            "MCTS_* overrides: the attempt had ['MCTS_EPSILON', 'mcts_c_visit'], the re-run []"
        ]

    def test_identical_reordered_and_non_mcts_vars_are_not_differences(self) -> None:
        first = attempt(record={"env_var_names": ["MCTS_B", "MCTS_A", "PATH"]})
        second = attempt(R2, record={"env_var_names": ["MCTS_A", "MCTS_B", "HOME"]})
        assert rr.condition_differences(first, second) == []
        non_a = attempt(record={"env_var_names": ["PATH", "HOME"]})
        non_b = attempt(R2, record={"env_var_names": ["OMP_NUM_THREADS"]})
        assert rr.condition_differences(non_a, non_b) == []

    @pytest.mark.parametrize(
        "bad",
        [None, "MCTS_C_VISIT", b"MCTS_C_VISIT", 123, {"MCTS_C_VISIT": "1"}],
    )
    def test_missing_or_non_sequence_env_var_names_backwards_compatibility(self, bad: Any) -> None:
        legacy = attempt()
        malformed = attempt(R2, record={"env_var_names": bad})
        assert rr.condition_differences(legacy, malformed) == []
        assert rr.condition_differences(malformed, legacy) == []
        with_override = attempt(R2, record={"env_var_names": ["MCTS_C_VISIT"]})
        assert rr.condition_differences(malformed, with_override) == [
            "MCTS_* overrides: the attempt had [], the re-run ['MCTS_C_VISIT']"
        ]


# ---------------------------------------------------------------------------
# 3. verify_release_evidence lowercase mcts_* detection
# ---------------------------------------------------------------------------


def test_check_run_json_detects_lowercase_mcts_override() -> None:
    record = {
        "schema_version": 1,
        "card_id": A1,
        "sha": SHA,
        "status": "COMPLETED",
        "env_var_names": ["mcts_c_visit", "McTs_Rollout_Depth", "PATH"],
        "total_priced_games": 0,
        "invocations": [
            {
                "name": "train",
                "priced_games": 0,
                "exit_code": 0,
                "argv": ["python"],
                "launched_argv": ["python", "--max_games", "0"],
            }
        ],
    }
    findings = vre.Findings()
    facts = vre.check_run_json(record, A1, f"run {A1} (A1)", findings)
    assert findings.problems == []
    assert any(
        "MCTS_* override(s) ['McTs_Rollout_Depth', 'mcts_c_visit']" in item
        for item in facts.interruptions
    )


def test_verify_end_to_end_detects_lowercase_mcts_override_in_bundle(tmp_path: Path) -> None:
    # Single clean run mutated on disk to lowercase mcts_c_visit must fail as VOID.
    single = fx.make_release(tmp_path / "single", [fx.RunSpec(env_var_names=["PATH"])])
    run_entry = single.bundle["runs"][0]["run_json"]
    run_path = single.bundle_dir / run_entry["path"]
    run_data = json.loads(run_path.read_text(encoding="utf-8"))
    run_data["env_var_names"] = ["mcts_c_visit", "PATH"]
    run_path.write_text(json.dumps(run_data, indent=2), encoding="utf-8", newline="\n")
    fx.rehash(single.bundle_dir, run_entry)
    single.save()

    code, out = verify(single)
    assert code == vre.EXIT_FAIL
    assert "mcts_c_visit" in out
    assert "recomputed 'VOID'" in out

    # Two-run VOID + clean re-run where attempt recorded lowercase mcts_c_visit verifies cleanly.
    release = fx.make_release(
        tmp_path / "rerun",
        [fx.RunSpec(run_id=A1, env_var_names=["MCTS_C_VISIT", "PATH"]), rerun()],
    )
    a1_entry = entries(release)[A1]
    a1_run_path = release.bundle_dir / a1_entry["run_json"]["path"]
    a1_run_data = json.loads(a1_run_path.read_text(encoding="utf-8"))
    a1_run_data["env_var_names"] = ["mcts_c_visit", "PATH"]
    a1_run_path.write_text(json.dumps(a1_run_data, indent=2), encoding="utf-8", newline="\n")
    fx.rehash(release.bundle_dir, a1_entry["run_json"])

    rec_entry = a1_entry["reproduction"]
    rec_path = release.bundle_dir / rec_entry["path"]
    rec_data = json.loads(rec_path.read_text(encoding="utf-8"))
    rec_data[rp.RECORD_ATTEMPT][rp.RUN_JSON_SHA256] = a1_entry["run_json"]["sha256"]
    rec_path.write_text(
        json.dumps(rec_data, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    fx.rehash(release.bundle_dir, rec_entry)
    release.save()

    findings = vre.verify(
        release.bundle_dir / vre.BUNDLE_FILE,
        repo_root=release.repo.path,
        git=Git(release.repo.path),
        api=fx.fake_api(release.rc_sha),
        repo=fx.REPO,
    )
    assert findings.problems == []
    assert findings.missing == []
    assert any("mcts_c_visit" in note for note in findings.notes)
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out


def test_fixture_write_run_treats_lowercase_mcts_override_as_void(tmp_path: Path) -> None:
    """``release_evidence_fixture.write_run`` matches verifier case-insensitivity for ``mcts_*``."""
    specs = [fx.RunSpec(run_id=A1, env_var_names=["mcts_c_visit", "PATH"]), rerun()]
    release = fx.make_release(tmp_path, specs)
    assert verdicts(release) == ["VOID", "INCONCLUSIVE"]
    assert entries(release)[A1]["reproduction"]["outcome"] == rr.OUTCOME_NON_REPRODUCING
    assert non_reproducing_reasons(release) == [
        "MCTS_* overrides: the attempt had ['mcts_c_visit'], the re-run []"
    ]
    assert entries(release)[R2]["verdict"]["label"] == NON_REPRODUCING_LABEL
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out


# ---------------------------------------------------------------------------
# 4. build_release_bundle card_id / run_id & _copy path-traversal guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    [
        "../escape",
        "..",
        "a/../b",
        "a\\..\\b",
        "",
        ".hidden",
        "-flag",
        "a/b",
        "a1\n",
        "a1\r",
    ],
)
def test_build_release_bundle_rejects_invalid_and_traversal_card_ids(
    tmp_path: Path, bad_id: str
) -> None:
    repo = GitRepo(tmp_path / "repo")
    run_sha = repo.commit(
        "init",
        {
            "pyproject.toml": fx.PYPROJECT,
            "docs/EVALUATION.md": "# empty\n",
        },
    )
    repo.tag(fx.RC_TAG, run_sha)
    run_dir = tmp_path / "runs" / "bad"
    _write_minimal_run(run_dir, bad_id, run_sha)

    staging_dir = tmp_path / "stage"
    with pytest.raises(builder.BuildError):
        builder.collect_run(
            builder.RunInput("A1", run_dir),
            staging_dir,
            Git(repo.path),
            run_sha,
        )
    assert not staging_dir.exists()
    assert not (tmp_path / "escape").exists()

    out_dir = tmp_path / "evidence" / vre.bundle_dir_name(fx.RELEASE)
    code = builder.main(
        [
            "--release",
            fx.RELEASE,
            "--rc",
            fx.RC_TAG,
            "--run",
            f"A1={run_dir}",
            "--repo-root",
            str(repo.path),
            "--out",
            str(out_dir),
        ]
    )
    assert code == builder.EXIT_REFUSED
    assert not out_dir.exists()
    assert not (tmp_path / "evidence" / "escape").exists()


def test_copy_rejects_path_traversal(tmp_path: Path) -> None:
    src = tmp_path / "source.json"
    src.write_text("{}", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    for bad_rel in ("../escape/run.json", ".", "", "a\\b/run.json"):
        with pytest.raises(builder.BuildError):
            builder._copy(src, bundle_dir, bad_rel)
    assert not (tmp_path / "escape").exists()
