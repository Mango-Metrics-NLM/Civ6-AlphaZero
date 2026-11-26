"""The release verifier applies the re-run rule itself (section 7.1, "After a VOID").

After an interruption the same card is run again under a new ``card_id`` at the
same seed, and that re-run is first a reproduction check: ``python -m
alpha_zero.evaluation.reproduction compare`` writes a record, and "that file goes
in the bundle beside both attempts". These tests hold the verifier
(``scripts/verify_release_evidence.py``, through ``scripts/release_reproduction.py``)
to what the record must show: a reproducing re-run is judged, a re-run that
differs or ran under other conditions cannot PASS, an empty comparison is said to
be empty, and a record that was not made from the bundled runs fails.

Bundles come from ``release_evidence_fixture``: an interrupted attempt with a
re-run gets its record by default (the arena and iteration digests taken from the
bundled files, the checkpoint digests stand-ins). Each test asserts the values the
bundle stores, then the verifier's exit code and message.

Kept apart from ``test_verify_release_evidence.py`` so neither file grows past the
size the debt ratchet holds test modules to.

Negative controls (2026-11-15, all re-run after the third round). Each defect was
planted in the named file, the planted text was read back from disk, this module
and ``test_verify_release_evidence_ordering.py`` were run and the listed tests were
seen to FAIL, and the file was restored and checked byte-identical (sha256) before
the next one. In ``scripts/release_reproduction.py`` unless another file is named;
"+N" counts further failing tests that are not named:

1.  ``review`` accepts an attempt without a record ->
    ``test_an_interrupted_attempt_without_a_record_fails``,
    ``test_a_run_voided_by_an_override_needs_a_record``,
    ``test_an_attempt_interrupted_in_its_first_invocation_is_an_attempt`` (+1).
2.  ``outcome`` ignores the digests' differences ->
    ``test_a_differing_rerun_cannot_pass``,
    ``test_a_differing_checkpoint_alone_cannot_pass``,
    ``test_a_record_claiming_reproduced_over_differing_digests_fails`` (+8).
3.  ``scripts/verify_release_evidence.py``: ``judge`` does not pass
    ``non_reproducing`` -> 19 tests, ``test_a_differing_rerun_cannot_pass`` and
    every "is non-reproducing" test among them (the stored INCONCLUSIVE was
    "recomputed" as PASS).
4.  ``scripts/verify_release_evidence.py``: ``compare_verdict`` does not compare
    the label -> ``test_a_differing_rerun_cannot_pass``.
5.  ``check_record`` does not recompute the status ->
    ``test_a_record_claiming_reproduced_over_differing_digests_fails`` (+1).
6.  ``check_record`` does not hold the digests to the bundled files -> all four
    ``test_a_record_that_contradicts_the_bundled_components_fails`` cases (+1): a
    record that leaves out the arm that differs verified with exit 0 and PASS.
7.  ``_bundle_contradictions`` checks one direction only -> the ``a missing arm``
    case.
8.  ``check_record`` does not check the identities -> the six
    ``test_a_record_for_other_runs_fails`` cases and
    ``TestWhichAttempts::test_a_null_identity_never_matches``; only the "is a
    string" half removed -> the latter alone.
9.  ``condition_differences`` does not compare the ``sha`` ->
    ``test_another_commit_is_non_reproducing``,
    ``TestConditions::test_the_sha_is_always_compared``.
10. ... compares no ``device_plan`` key -> the four
    ``test_another_device_plan_is_non_reproducing`` cases and the eleven
    ``test_every_named_*_key_is_compared`` cases.
11. ... skips the driver -> both ``test_another_driver_is_non_reproducing`` cases,
    both ``test_an_unreadable_driver_is_non_reproducing`` cases,
    ``test_a_run_json_without_the_new_keys_still_verifies`` and one unit test;
    compares the first GPU's driver only -> both
    ``test_another_driver_is_non_reproducing`` cases.
12. ... does not compare the torch wheel ->
    ``test_another_torch_wheel_is_non_reproducing_when_both_runs_record_one`` and
    its unit test; counts a wheel only one run records as a difference -> those
    two and ``TestConditions::test_records_that_are_not_conditions_are_ignored``.
13. ``alpha_zero/evaluation/reproduction.py``: ``_shows_a_result`` counts the
    seed-fixed checkpoint -> ``test_only_checkpoint_0_is_an_empty_comparison``,
    ``test_another_commit_is_non_reproducing``.
14. ``needs_record`` ignores ``control_void`` ->
    ``test_an_interrupted_control_void_is_a_second_draw``,
    ``TestWhichAttempts::test_needs_record``; true for every launched attempt ->
    ``test_a_control_void_needs_no_record`` (+7); without the "was interrupted"
    condition -> two unit tests only (in a bundle the judged attempt is the last
    of its role, so nothing follows it to be compared with).
15. ``rerun_of`` takes the last later COMPLETED attempt ->
    ``test_the_rerun_is_the_next_completed_attempt`` (+6 in the ordering module).
16. ``launched`` reads ``invocations`` (the finished ones) ->
    ``test_an_attempt_interrupted_in_its_first_invocation_is_an_attempt``,
    ``test_only_checkpoint_0_is_an_empty_comparison``,
    ``test_another_commit_is_non_reproducing`` and four unit tests.
17. ``scripts/verify_release_evidence.py``: the "launched nothing" problem removed
    -> ``test_an_attempt_that_launched_nothing_must_not_be_listed``.
18. ``review`` accepts a ``reproduction`` entry no comparison applies to -> both
    ``test_a_record_no_comparison_applies_to_fails`` cases.
19. ``review`` does not compare the stored outcome ->
    ``test_a_stored_outcome_that_is_not_the_recomputed_one_fails`` (+1).
20. ``scripts/build_release_bundle.py``: ``reproduction_lines`` returns nothing ->
    ``test_only_checkpoint_0_is_an_empty_comparison``,
    ``test_the_report_shows_the_label_and_the_comparison``.
21. ``scripts/verify_release_evidence.py``, the ``Attempt`` it builds: no arena
    paths -> 32 tests; no iterations path -> 29; ``control_void`` always false ->
    ``test_an_interrupted_control_void_is_a_second_draw`` (+4);
    ``devices_registered`` always false -> the five driver tests.
22. ``_recompute`` lets a misshapen digest raise ->
    ``test_a_record_with_digests_of_another_shape_fails_and_does_not_crash``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.evaluation import reproduction as rp
from alpha_zero.evaluation.a1_protocol import NON_REPRODUCING_LABEL
from scripts import build_release_bundle as builder
from scripts import release_reproduction as rr
from scripts import verify_release_evidence as vre
from tests.scripts import release_evidence_fixture as fx
from tests.scripts.test_verify_release_evidence import verify

A1 = "a1-20261111"
R2 = "a1-20261111-r2"
R3 = "a1-20261111-r3"
A1R = {"run_id": "a1r-20261112", "role": "A1-R"}
RECORD_PATH = "a1-20261111.reproduction.json"
MCTS_OVERRIDE = ["MCTS_C_VISIT", "PATH"]


def interrupted(**changes: Any) -> fx.RunSpec:
    """Training and the primary arm finished; interrupted inside the symmetry arm."""
    return fx.RunSpec(**{"run_id": A1, "status": "VOID", "recorded": 2, **changes})


def rerun(**changes: Any) -> fx.RunSpec:
    return fx.RunSpec(**{"run_id": R2, **changes})


def entries(release: fx.Release) -> dict[str, dict[str, Any]]:
    return {run["run_id"]: run for run in release.bundle["runs"]}


def verdicts(release: fx.Release) -> list[str]:
    return [run["verdict"]["verdict"] for run in release.bundle["runs"]]


def stored_record(release: fx.Release, run_id: str = A1) -> dict[str, Any]:
    path = release.bundle_dir / entries(release)[run_id]["reproduction"]["path"]
    return dict(json.loads(path.read_text(encoding="utf-8")))


def digest_of(record: dict[str, Any], side: str) -> dict[str, Any]:
    return dict(record[rp.RECORD_DIGESTS][side])


# ---------------------------------------------------------------------------
# REPRODUCED: the re-run is judged
# ---------------------------------------------------------------------------


def test_a_reproducing_rerun_is_judged(tmp_path: Path) -> None:
    release = fx.make_release(tmp_path, [interrupted(), rerun(), fx.RunSpec(**A1R)])
    void, judged = entries(release)[A1], entries(release)[R2]
    reference = void["reproduction"]
    assert reference == {
        "path": RECORD_PATH,
        "sha256": vre.sha256_file(release.bundle_dir / RECORD_PATH),
        "rerun": R2,
        "outcome": rr.OUTCOME_REPRODUCED,
        "reasons": [],
    }
    record = stored_record(release)
    assert record[rp.RECORD_STATUS] == rp.STATUS_REPRODUCED
    # checkpoint_0 and one per 5 iterations, the primary arm, 20 iteration records.
    assert record["components"] == {"checkpoints": 5, "arenas": 1, "iterations": 20}
    assert record[rp.RECORD_ATTEMPT]["card_id"] == A1 and record[rp.RECORD_RERUN]["card_id"] == R2
    assert "reproduction" not in judged
    assert verdicts(release) == ["VOID", "PASS", "PASS"]
    assert "label" not in judged["verdict"]
    assert release.bundle["learning_claimed"] is True
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out
    assert f"run {R2} (A1): verdict PASS" in out


def test_the_stored_reasons_are_prose_and_are_not_compared(tmp_path: Path) -> None:
    release = fx.make_release(tmp_path, [interrupted(), rerun()])
    entries(release)[A1]["reproduction"]["reasons"] = ["reworded by hand"]
    release.save()
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out


def test_an_interrupted_attempt_without_a_record_fails(tmp_path: Path) -> None:
    release = fx.make_release(tmp_path, [interrupted(compare_record=False), rerun()])
    assert "reproduction" not in entries(release)[A1]
    assert verdicts(release) == ["VOID", "PASS"]
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert f"run {A1} (A1) reproduction record: missing" in out
    assert f"{R2} is its re-run" in out and "--compare FILE" in out


def test_a_record_that_is_not_in_the_bundle_is_missing(tmp_path: Path) -> None:
    release = fx.make_release(tmp_path, [interrupted(), rerun()])
    (release.bundle_dir / RECORD_PATH).unlink()
    code, out = verify(release)
    assert code == vre.EXIT_MISSING
    assert f"{RECORD_PATH} does not exist" in out


def test_an_edited_record_fails_its_hash(tmp_path: Path) -> None:
    release = fx.make_release(tmp_path, [interrupted(), rerun()])
    path = release.bundle_dir / RECORD_PATH
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert "reproduction record: a1-20261111.reproduction.json does not hash" in out


# ---------------------------------------------------------------------------
# DIFFERS: the re-run is reported, labelled, and cannot PASS
# ---------------------------------------------------------------------------


def test_a_differing_rerun_cannot_pass(tmp_path: Path) -> None:
    """The re-run played other games in the primary arm (the same counts: it would PASS)."""
    release = fx.make_release(tmp_path, [interrupted(), rerun(primary_rotation=1)])
    reference, verdict = entries(release)[A1]["reproduction"], entries(release)[R2]["verdict"]
    assert stored_record(release)[rp.RECORD_STATUS] == rp.STATUS_DIFFERS
    assert reference["outcome"] == rr.OUTCOME_NON_REPRODUCING and reference["rerun"] == R2
    assert len(reference["reasons"]) == 1 and "arena primary" in reference["reasons"][0]
    assert verdict["verdict"] == "INCONCLUSIVE"
    assert verdict["label"] == NON_REPRODUCING_LABEL == "non-reproducing re-run"
    assert verdict["primary_hull"][0] > 0.5, "the premise: the arm itself would PASS"
    assert verdict["mde_alpha025_80pct"] > 0 and verdict["hull_ratio"] >= 1
    assert any(f"it does not reproduce {A1}" in reason for reason in verdict["reasons"])
    assert release.bundle["learning_claimed"] is False
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out  # recorded honestly, it ships with the claim withheld
    assert f"run {R2} (A1): verdict INCONCLUSIVE" in out

    stored = dict(verdict)
    # Restoring the PASS the arm would have been.
    verdict.update(verdict="PASS", mde_alpha025_80pct=None, hull_ratio=None)
    verdict["mde_alpha025_80pct_hull"] = None
    del verdict["label"]
    release.save()
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert "stored verdict 'PASS' != recomputed 'INCONCLUSIVE'" in out
    # Keeping INCONCLUSIVE but dropping the label.
    verdict.clear()
    verdict.update({key: value for key, value in stored.items() if key != "label"})
    release.save()
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert "verdict label None != 'non-reproducing re-run'" in out


def test_a_differing_checkpoint_alone_cannot_pass(tmp_path: Path) -> None:
    """Nothing in the bundle anchors the checkpoint digests, but they still decide."""
    release = fx.make_release(tmp_path, [interrupted(), rerun(checkpoint_salt="other weights")])
    reference = entries(release)[A1]["reproduction"]
    assert reference["outcome"] == rr.OUTCOME_NON_REPRODUCING
    assert [r.split(":")[0] for r in reference["reasons"]] == [
        f"checkpoint checkpoint_{n}.pt" for n in (10, 15, 20, 5)
    ]
    assert entries(release)[R2]["verdict"]["label"] == NON_REPRODUCING_LABEL
    assert verdicts(release) == ["VOID", "INCONCLUSIVE"]
    assert verify(release)[0] == vre.EXIT_PASS


def test_the_claim_is_withheld_after_a_non_reproducing_rerun(tmp_path: Path) -> None:
    """A1-R ran (and passed) before A1's re-run started; nothing may start after that re-run."""
    specs = [interrupted(), fx.RunSpec(**A1R), rerun(primary_rotation=1)]
    release = fx.make_release(tmp_path, specs)
    assert verdicts(release) == ["VOID", "PASS", "INCONCLUSIVE"]
    assert release.bundle["learning_claimed"] is False
    assert verify(release)[0] == vre.EXIT_PASS
    release.bundle["learning_claimed"] = True
    release.save()
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert "do not permit the claim" in out


def test_a_record_claiming_reproduced_over_differing_digests_fails(tmp_path: Path) -> None:
    """The record's status is not trusted: it is recomputed from the record's own digests."""
    release = fx.make_release(tmp_path, [interrupted(), rerun(primary_rotation=1)])
    void, judged = entries(release)[A1], entries(release)[R2]
    path = release.bundle_dir / RECORD_PATH
    record = stored_record(release)
    record[rp.RECORD_STATUS], record[rp.RECORD_DIFFERENCES] = rp.STATUS_REPRODUCED, []
    path.write_text(json.dumps(record), encoding="utf-8")
    fx.rehash(release.bundle_dir, void["reproduction"])
    void["reproduction"].update(outcome=rr.OUTCOME_REPRODUCED, reasons=[])
    judged["verdict"].update(verdict="PASS", mde_alpha025_80pct=None, hull_ratio=None)
    judged["verdict"]["mde_alpha025_80pct_hull"] = None
    del judged["verdict"]["label"]
    release.save()
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert "status 'REPRODUCED', but its own digests give 'DIFFERS'" in out
    assert "its differences are not the ones its own digests give" in out
    assert "stored outcome 'REPRODUCED'" in out and "recomputed 'NON_REPRODUCING'" in out
    assert "stored verdict 'PASS' != recomputed 'INCONCLUSIVE'" in out


def test_a_stored_outcome_that_is_not_the_recomputed_one_fails(tmp_path: Path) -> None:
    release = fx.make_release(tmp_path, [interrupted(), rerun()])
    reference = entries(release)[A1]["reproduction"]
    reference["outcome"] = rr.OUTCOME_EMPTY
    release.save()
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert "stored outcome 'EMPTY'" in out and "recomputed 'REPRODUCED'" in out
    reference.update(outcome=rr.OUTCOME_REPRODUCED, rerun="a1-other")
    release.save()
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert "against 'a1-other' != recomputed 'REPRODUCED' against 'a1-20261111-r2'" in out


# ---------------------------------------------------------------------------
# The record is held to the bundled runs
# ---------------------------------------------------------------------------


def _hide_the_arm(record: dict[str, Any]) -> None:
    """Leave the differing arm out of the attempt's digest, so the record says REPRODUCED."""
    del record[rp.RECORD_DIGESTS][rp.RECORD_ATTEMPT][rp.COMPONENT_ARENAS]["primary"]
    record[rp.RECORD_STATUS], record[rp.RECORD_DIFFERENCES] = rp.STATUS_REPRODUCED, []


def _other_arena_games(record: dict[str, Any]) -> None:
    """Both sides agree with each other, but not with the bundled result.json."""
    for side in (rp.RECORD_ATTEMPT, rp.RECORD_RERUN):
        rows = record[rp.RECORD_DIGESTS][side][rp.COMPONENT_ARENAS]["primary"]["rows"]
        rows[0]["games"][0]["result"] = "loss"


def _other_iteration_records(record: dict[str, Any]) -> None:
    for side in (rp.RECORD_ATTEMPT, rp.RECORD_RERUN):
        record[rp.RECORD_DIGESTS][side][rp.COMPONENT_ITERATIONS]["3"]["seed"] = 7


def _an_arm_the_bundle_omits(record: dict[str, Any]) -> None:
    """The record digests an arm of the attempt that is not bundled for it."""
    arenas = {side: record[rp.RECORD_DIGESTS][side][rp.COMPONENT_ARENAS] for side in _SIDES}
    arenas[rp.RECORD_ATTEMPT]["symmetry"] = arenas[rp.RECORD_RERUN]["symmetry"]


_SIDES = (rp.RECORD_ATTEMPT, rp.RECORD_RERUN)


@pytest.mark.parametrize(
    ("edit", "rotation", "message"),
    [
        pytest.param(
            _hide_the_arm,
            1,
            f"attempt: arenas ['primary'] are bundled for {A1} but missing from the record",
            id="a missing arm",
        ),
        pytest.param(
            _other_arena_games,
            0,
            f"attempt: arenas ['primary'] differ from the files bundled for {A1}",
            id="other arena games",
        ),
        pytest.param(
            _other_iteration_records,
            0,
            f"rerun: iterations ['3'] differ from the files bundled for {R2}",
            id="other iteration records",
        ),
        pytest.param(
            _an_arm_the_bundle_omits,
            0,
            f"attempt: arenas ['symmetry'] are in the record but not bundled for {A1}",
            id="an arm the bundle omits",
        ),
    ],
)
def test_a_record_that_contradicts_the_bundled_components_fails(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None], rotation: int, message: str
) -> None:
    """The record is self-consistent and says REPRODUCED; only the bundled files show otherwise.

    The first case is the one that matters: a re-run whose primary arm differs,
    and a record that leaves that arm out. Its status, the stored outcome and the
    stored PASS all agree with each other.
    """
    specs = [interrupted(edit_record=edit), rerun(primary_rotation=rotation)]
    release = fx.make_release(tmp_path, specs)
    record = stored_record(release)
    recomputed = rp.compare_status(*(digest_of(record, side) for side in _SIDES))
    assert (
        recomputed
        == (rp.STATUS_REPRODUCED, [])
        == (
            record[rp.RECORD_STATUS],
            record[rp.RECORD_DIFFERENCES],
        )
    )
    assert entries(release)[A1]["reproduction"]["outcome"] == rr.OUTCOME_REPRODUCED
    assert verdicts(release) == ["VOID", "PASS"]
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert message in out


@pytest.mark.parametrize("side", _SIDES)
@pytest.mark.parametrize("key", ["card_id", "sha", rp.RUN_JSON_SHA256])
def test_a_record_for_other_runs_fails(tmp_path: Path, side: str, key: str) -> None:
    def other_run(record: dict[str, Any]) -> None:
        record[side][key] = "f" * 40

    release = fx.make_release(tmp_path, [interrupted(edit_record=other_run), rerun()])
    assert verdicts(release) == ["VOID", "PASS"]
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert f"{side}.{key} '{'f' * 40}' is not" in out and "the record is for other runs" in out


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda r: r.update({rp.RECORD_SCHEMA_KEY: 2}), "schema_version 2 != 1"),
        (lambda r: r.pop(rp.RECORD_SCHEMA_KEY), "schema_version None != 1"),
        (lambda r: r.pop(rp.RECORD_DIGESTS), "the record holds no digests for both runs"),
        (lambda r: r.update({rp.RECORD_RERUN: None}), "rerun: the record names no run"),
    ],
)
def test_a_record_of_another_shape_fails(
    tmp_path: Path, edit: Callable[[dict[str, Any]], Any], message: str
) -> None:
    """A record written before identities and digests existed cannot be recomputed."""
    release = fx.make_release(tmp_path, [interrupted(edit_record=edit), rerun()])
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert message in out


def test_a_record_with_digests_of_another_shape_fails_and_does_not_crash(tmp_path: Path) -> None:
    """A record is input: an arm digest that is a list must fail the bundle, not the verifier."""

    def misshapen(record: dict[str, Any]) -> None:
        for side in _SIDES:
            record[rp.RECORD_DIGESTS][side][rp.COMPONENT_ARENAS]["primary"] = ["not", "a", "digest"]

    release = fx.make_release(tmp_path, [interrupted(edit_record=misshapen), rerun()])
    reference = entries(release)[A1]["reproduction"]
    assert reference["outcome"] == rr.OUTCOME_NON_REPRODUCING
    assert "the record's digests cannot be compared (AttributeError" in reference["reasons"][0]
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert "Traceback" not in out and "cannot verify the bundle" not in out
    assert "status 'REPRODUCED', but its own digests give 'DIFFERS'" in out
    assert f"attempt: arenas ['primary'] differ from the files bundled for {A1}" in out


# ---------------------------------------------------------------------------
# Another sha, device_plan or driver: non-reproducing, whatever the comparison says
# ---------------------------------------------------------------------------


def killed_at_once(**changes: Any) -> fx.RunSpec:
    """Interrupted before the first iteration finished: only ``checkpoint_0`` exists."""
    return interrupted(recorded=0, iterations_written=0, **changes)


def non_reproducing_reasons(release: fx.Release) -> list[str]:
    reference = entries(release)[A1]["reproduction"]
    assert reference["outcome"] == rr.OUTCOME_NON_REPRODUCING, reference
    assert entries(release)[R2]["verdict"]["verdict"] == "INCONCLUSIVE"
    assert entries(release)[R2]["verdict"]["label"] == NON_REPRODUCING_LABEL
    return list(reference["reasons"])


def test_another_commit_is_non_reproducing(tmp_path: Path) -> None:
    """The re-run ran at the rc, not at the attempt's ``sha`` -- even an empty comparison."""
    specs = [killed_at_once(), rerun(at_rc=True)]
    release = fx.make_release(tmp_path, specs, after_run={"alpha_zero/unrelated.py": "X = 2\n"})
    assert release.run_sha != release.rc_sha
    assert stored_record(release)[rp.RECORD_STATUS] == rp.STATUS_NOTHING_TO_COMPARE
    assert non_reproducing_reasons(release) == [
        f"sha: the attempt ran at {release.run_sha}, the re-run at {release.rc_sha}"
    ]
    assert verify(release)[0] == vre.EXIT_PASS


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (
            lambda p: {**p, "CUBLAS_WORKSPACE_CONFIG": ":16:8"},
            'device_plan.CUBLAS_WORKSPACE_CONFIG: the attempt had ":4096:8", the re-run ":16:8"',
        ),
        (
            lambda p: {**p, "requested": {"train": "cuda:0", "primary": "cuda:0"}},
            "device_plan.requested: the attempt had null, the re-run",
        ),
        (
            lambda p: {**p, "torch": {**p["torch"], "cuda": "12.9"}},
            'device_plan.torch.cuda: the attempt had "12.8", the re-run "12.9"',
        ),
        (
            lambda p: {**p, "torch": {**p["torch"], "cudnn": 91002}},
            "device_plan.torch.cudnn: the attempt had null, the re-run 91002",
        ),
    ],
)
def test_another_device_plan_is_non_reproducing(tmp_path: Path, edit: Any, reason: str) -> None:
    """A key the registration does not fix, so the re-run is not VOID: it just cannot PASS."""
    release = fx.make_release(tmp_path, [interrupted(), rerun(edit_device_plan=edit)])
    assert stored_record(release)[rp.RECORD_STATUS] == rp.STATUS_REPRODUCED
    (found,) = non_reproducing_reasons(release)
    assert found.startswith(reason)
    assert verify(release)[0] == vre.EXIT_PASS


def test_where_a_value_came_from_is_not_a_condition(tmp_path: Path) -> None:
    def inherited(plan: dict[str, Any]) -> dict[str, Any]:
        return {**plan, "source": "inherited"}

    release = fx.make_release(tmp_path, [interrupted(), rerun(edit_device_plan=inherited)])
    assert entries(release)[A1]["reproduction"]["outcome"] == rr.OUTCOME_REPRODUCED
    assert verdicts(release) == ["VOID", "PASS"]
    assert verify(release)[0] == vre.EXIT_PASS


@pytest.mark.parametrize(
    ("before", "after", "shown"),
    [
        (fx.DRIVER, "580.10", "('576.88', '576.88'), the re-run under ('580.10', '580.10')"),
        # One GPU only: every GPU's driver is compared, not the first one's.
        (
            "581.15",
            ("581.15", "581.42"),
            "('581.15', '581.15'), the re-run under ('581.15', '581.42')",
        ),
    ],
)
def test_another_driver_is_non_reproducing(
    tmp_path: Path, before: str, after: Any, shown: str
) -> None:
    release = fx.make_release(tmp_path, [interrupted(driver=before), rerun(driver=after)])
    assert non_reproducing_reasons(release) == [
        f"driver: the attempt ran under NVIDIA driver {shown}"
    ]
    assert verify(release)[0] == vre.EXIT_PASS


@pytest.mark.parametrize(("void_driver", "rerun_driver"), [(None, fx.DRIVER), (fx.DRIVER, None)])
def test_an_unreadable_driver_is_non_reproducing(
    tmp_path: Path, void_driver: str | None, rerun_driver: str | None
) -> None:
    """A driver that was never recorded cannot be shown to be the same one."""
    specs = [interrupted(driver=void_driver), rerun(driver=rerun_driver)]
    release = fx.make_release(tmp_path, specs)
    (found,) = non_reproducing_reasons(release)
    who = "the attempt" if void_driver is None else "the re-run"
    assert found.startswith(f"driver: the NVIDIA driver version of {who} could not be read")
    assert verify(release)[0] == vre.EXIT_PASS


def _wheel(sha256: str) -> Callable[[dict[str, Any]], None]:
    def record_wheel(run_json: dict[str, Any]) -> None:
        run_json["torch_wheel"] = {"url": "https://example.invalid/torch.whl", "sha256": sha256}

    return record_wheel


def test_another_torch_wheel_is_non_reproducing_when_both_runs_record_one(tmp_path: Path) -> None:
    """Section 7.1 "Devices" names the same torch wheel; pip does not always report it."""
    one, other = "1" * 64, "2" * 64
    specs = [interrupted(edit_run_json=_wheel(one)), rerun(edit_run_json=_wheel(other))]
    release = fx.make_release(tmp_path / "both", specs)
    assert non_reproducing_reasons(release) == [
        f"torch_wheel: the attempt installed the wheel with sha256 {one}, the re-run {other}"
    ]
    assert verify(release)[0] == vre.EXIT_PASS
    for name, specs in {
        "same": [interrupted(edit_run_json=_wheel(one)), rerun(edit_run_json=_wheel(one))],
        "attempt only": [interrupted(edit_run_json=_wheel(one)), rerun()],
        "re-run only": [interrupted(), rerun(edit_run_json=_wheel(other))],
    }.items():
        release = fx.make_release(tmp_path / name, specs)
        assert entries(release)[A1]["reproduction"]["outcome"] == rr.OUTCOME_REPRODUCED, name
        assert verdicts(release) == ["VOID", "PASS"]
        assert verify(release)[0] == vre.EXIT_PASS


# ---------------------------------------------------------------------------
# NOTHING TO COMPARE: the re-run is judged, and the bundle says so
# ---------------------------------------------------------------------------


def test_only_checkpoint_0_is_an_empty_comparison(tmp_path: Path) -> None:
    release = fx.make_release(tmp_path, [killed_at_once(), rerun()])
    record = stored_record(release)
    assert record["components"] == {"checkpoints": 1, "arenas": 0, "iterations": 0}
    assert list(digest_of(record, rp.RECORD_ATTEMPT)[rp.COMPONENT_CHECKPOINTS]) == [
        rp.SEED_FIXED_CHECKPOINT
    ]
    assert record[rp.RECORD_STATUS] == rp.STATUS_NOTHING_TO_COMPARE
    reference = entries(release)[A1]["reproduction"]
    assert (reference["outcome"], reference["reasons"]) == (rr.OUTCOME_EMPTY, [])
    assert verdicts(release) == ["VOID", "PASS"]
    assert "label" not in entries(release)[R2]["verdict"]
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out
    report = builder.render_report(release.bundle)
    assert (
        f"Reproduction check against `{R2}` (`{RECORD_PATH}`): the comparison was empty" in report
    )
    assert "NOTHING TO COMPARE" in report and "`checkpoint_0`" in report


def test_the_report_shows_the_label_and_the_comparison(tmp_path: Path) -> None:
    release = fx.make_release(tmp_path, [interrupted(), rerun(primary_rotation=1)])
    report = builder.render_report(release.bundle)
    assert f"| `{R2}` | A1 | 20261111 |" in report
    assert "| INCONCLUSIVE (non-reproducing re-run) |" in report
    assert f"## `{R2}` (A1): INCONCLUSIVE (non-reproducing re-run)" in report
    assert "the re-run does not reproduce this attempt" in report and "cannot PASS" in report
    assert "INCONCLUSIVE (non-reproducing re-run)" in builder.index_row(
        release.bundle, "2026-12-01"
    )
    assert vre._claim_lines(report) == [], "the report must not word a learning claim"
    reproduced = fx.make_release(tmp_path / "ok", [interrupted(), rerun()])
    text = builder.render_report(reproduced.bundle)
    assert f"Reproduction check against `{R2}` (`{RECORD_PATH}`): REPRODUCED" in text
    assert "non-reproducing" not in text


# ---------------------------------------------------------------------------
# Which attempts need a record
# ---------------------------------------------------------------------------


def test_a_control_void_needs_no_record(tmp_path: Path) -> None:
    """A symmetry VOID: the code changes before the re-run, which is a second draw."""
    specs = [fx.RunSpec(run_id=A1, symmetry=list(fx.LOPSIDED)), rerun()]
    release = fx.make_release(tmp_path, specs)
    assert verdicts(release) == ["VOID", "PASS"]
    assert all("reproduction" not in run for run in release.bundle["runs"])
    assert not list(release.bundle_dir.glob(f"*{rr.RECORD_SUFFIX}"))
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out


def test_an_interrupted_control_void_is_a_second_draw(tmp_path: Path) -> None:
    """Interrupted after its judged arms had shown a symmetry VOID: still a second draw."""
    specs = [fx.RunSpec(run_id=A1, status="VOID", symmetry=list(fx.LOPSIDED)), rerun()]
    release = fx.make_release(tmp_path, specs)
    assert "reproduction" not in entries(release)[A1]
    assert verdicts(release) == ["VOID", "PASS"]
    assert verify(release)[0] == vre.EXIT_PASS


def test_a_run_voided_by_an_override_needs_a_record(tmp_path: Path) -> None:
    """COMPLETED, but VOID for an ``MCTS_*`` variable: an interruption-class VOID."""
    release = fx.make_release(
        tmp_path / "ok", [fx.RunSpec(run_id=A1, env_var_names=MCTS_OVERRIDE), rerun()]
    )
    reference = entries(release)[A1]["reproduction"]
    assert reference["outcome"] == rr.OUTCOME_NON_REPRODUCING
    assert reference["reasons"] == [
        "MCTS_* overrides: the attempt had ['MCTS_C_VISIT'], the re-run []"
    ]
    assert stored_record(release)["components"] == {"checkpoints": 5, "arenas": 7, "iterations": 20}
    assert verdicts(release) == ["VOID", "INCONCLUSIVE"]
    assert entries(release)[R2]["verdict"]["label"] == NON_REPRODUCING_LABEL
    assert release.bundle["learning_claimed"] is False
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out
    entries(release)[R2]["verdict"] = {"verdict": "PASS", "reasons": ["tampered"]}
    release.save()
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert "stored verdict 'PASS' != recomputed 'INCONCLUSIVE'" in out
    bare = [fx.RunSpec(run_id=A1, env_var_names=MCTS_OVERRIDE, compare_record=False), rerun()]
    code, out = verify(fx.make_release(tmp_path / "bare", bare))
    assert code == vre.EXIT_FAIL
    assert "reproduction record: missing" in out and "MCTS_* override(s)" in out


def test_the_rerun_is_the_next_completed_attempt(tmp_path: Path) -> None:
    """What section 7.1 allows, with three attempts started one after another.

    Interrupted; a re-run that reproduces it and that a control arm VOIDs; then,
    after the arena fix, a second draw at the rc, which is the one judged.
    """
    specs = [interrupted(), rerun(symmetry=list(fx.LOPSIDED)), fx.RunSpec(run_id=R3, at_rc=True)]
    arena_fix = {"alpha_zero/core/mcts/base.py": "BASE = 2\n"}
    release = fx.make_release(tmp_path, specs, after_run=arena_fix)
    assert release.run_sha != release.rc_sha
    at_rc = [run["git_sha"] == release.rc_sha for run in release.bundle["runs"]]
    assert at_rc == [False, False, True]
    assert verdicts(release) == ["VOID", "VOID", "PASS"]
    reference = entries(release)[A1]["reproduction"]
    assert (reference["rerun"], reference["outcome"]) == (R2, rr.OUTCOME_REPRODUCED)
    assert "reproduction" not in entries(release)[R2]
    assert "reproduction" not in entries(release)[R3]
    assert "label" not in entries(release)[R3]["verdict"]
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out
    assert f"run {R3} (A1): verdict PASS" in out


@pytest.mark.parametrize("on", [R2, A1])
def test_a_record_no_comparison_applies_to_fails(tmp_path: Path, on: str) -> None:
    """A ``reproduction`` entry nothing recomputes is a stored claim: it may not be listed."""
    void = fx.RunSpec(run_id=A1, symmetry=list(fx.LOPSIDED)) if on == A1 else interrupted()
    release = fx.make_release(tmp_path, [void, rerun()])
    path = release.bundle_dir / "extra.reproduction.json"
    path.write_text("{}", encoding="utf-8")
    entry = {"path": path.name, "rerun": R2, "outcome": rr.OUTCOME_REPRODUCED, "reasons": []}
    fx.rehash(release.bundle_dir, entry)
    entries(release)[on]["reproduction"] = entry
    release.save()
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    why = "it was not interrupted" if on == R2 else "its re-run is a second draw"
    assert f"run {on} (A1) reproduction record: listed, but no comparison applies" in out
    assert why in out


# ---------------------------------------------------------------------------
# What is an attempt
# ---------------------------------------------------------------------------


def test_an_attempt_that_launched_nothing_must_not_be_listed(tmp_path: Path) -> None:
    """Refused at its device check: ``launched`` is empty, so it played nothing."""
    refused = fx.RunSpec(run_id=A1, status="FAILED", recorded=0, iterations_written=0, launched=[])
    release = fx.make_release(tmp_path, [refused, rerun()])
    assert "reproduction" not in entries(release)[A1]
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert f"run {A1} (A1): run.json records no launched invocation" in out
    assert "not an attempt; do not list it" in out


def test_an_attempt_interrupted_in_its_first_invocation_is_an_attempt(tmp_path: Path) -> None:
    """Nothing finished, but training had started: ``launched`` names it, so a record is due."""
    killed = fx.RunSpec(run_id=A1, status="VOID", recorded=0, iterations_written=7)
    release = fx.make_release(tmp_path / "ok", [killed, rerun()])
    run_json = json.loads(
        (release.bundle_dir / entries(release)[A1]["run_json"]["path"]).read_text(encoding="utf-8")
    )
    assert (run_json["launched"], run_json["invocations"]) == (["train"], [])
    assert entries(release)[A1]["reproduction"]["outcome"] == rr.OUTCOME_REPRODUCED
    assert stored_record(release)["components"] == {"checkpoints": 2, "arenas": 0, "iterations": 7}
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out
    assert "not an attempt" not in out
    bare = fx.RunSpec(
        run_id=A1, status="VOID", recorded=0, iterations_written=7, compare_record=False
    )
    code, out = verify(fx.make_release(tmp_path / "bare", [bare, rerun()]))
    assert code == vre.EXIT_FAIL
    assert "reproduction record: missing" in out


def test_a_run_json_without_the_new_keys_still_verifies(tmp_path: Path) -> None:
    """``launched`` and ``preflight`` are additive: an old record is read."""
    release = fx.make_release(tmp_path / "one", [fx.RunSpec(legacy_run_json=True)])
    run_json = json.loads(
        (release.bundle_dir / release.bundle["runs"][0]["run_json"]["path"]).read_text("utf-8")
    )
    assert not {"launched", "preflight"} & set(run_json) and "started_at" in run_json
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out
    assert "verdict PASS" in out
    # An old interrupted attempt counts as launched; its driver was never recorded,
    # so under a GPU registration it cannot be shown to reproduce.
    specs = [interrupted(legacy_run_json=True), rerun(legacy_run_json=True)]
    release = fx.make_release(tmp_path / "two", specs)
    (found,) = non_reproducing_reasons(release)
    assert (
        "driver: the NVIDIA driver version of the attempt and the re-run could not be read" in found
    )
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out


# ---------------------------------------------------------------------------
# release_reproduction, directly
# ---------------------------------------------------------------------------

SHA = "a" * 40
SMI = ["0, NVIDIA GeForce RTX 5060 Ti, 00000000:01:00.0, 576.88"]
PLAN = {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "source": "card",
    "torch_pin": {"spec": "torch==2.11.0+cu128"},
    "requested": {"train": "cuda:0"},
    "expected_gpus": {"0": "NVIDIA GeForce RTX 5060 Ti"},
    "torch": {"version": "2.11.0+cu128", "cuda": "12.8", "cudnn": 91002, "devices": ["x"]},
}


def attempt(run_id: str = A1, **changes: Any) -> rr.Attempt:
    """A run on the registered devices; ``record`` keys and fields are overridden by name."""
    record: dict[str, Any] = {
        "device_plan": PLAN,
        "preflight": {"nvidia_smi": SMI},
        "launched": ["train"],
        "started_at": "2026-11-11T08:00:00Z",
    }
    record.update(changes.pop("record", {}))
    fields: dict[str, Any] = {"sha": SHA, "completed": True, "devices_registered": True, **changes}
    return rr.Attempt(
        run_id, fields.pop("role", "A1"), record=record, run_json_sha256="0" * 64, **fields
    )


class TestConditions:
    def test_the_same_conditions_are_no_difference(self) -> None:
        assert rr.condition_differences(attempt(), attempt(R2)) == []

    def test_records_that_are_not_conditions_are_ignored(self) -> None:
        """A ``torch_wheel`` only one run records; ``source`` says where a value came from."""
        other = attempt(
            R2,
            record={
                "torch_wheel": {"url": "https://example.invalid/torch.whl", "sha256": "1" * 64},
                "device_plan": {**PLAN, "source": "inherited"},
                "started_at": "2026-11-12T00:00:00Z",
            },
        )
        assert rr.condition_differences(attempt(), other) == []

    def test_the_torch_wheel_is_compared_when_both_runs_record_its_hash(self) -> None:
        def wheel(sha256: Any) -> dict[str, Any]:
            return {"torch_wheel": {"url": "https://example.invalid/t.whl", "sha256": sha256}}

        first = attempt(record=wheel("1" * 64))
        (found,) = rr.condition_differences(first, attempt(R2, record=wheel("2" * 64)))
        assert found.startswith(
            f"torch_wheel: the attempt installed the wheel with sha256 {'1' * 64}"
        )
        for unrecorded in (wheel("1" * 64), wheel(None), wheel(""), {"torch_wheel": None}, {}):
            assert rr.condition_differences(first, attempt(R2, record=unrecorded)) == []

    def test_the_sha_is_always_compared(self) -> None:
        (found,) = rr.condition_differences(attempt(), attempt(R2, sha="b" * 40))
        assert found == f"sha: the attempt ran at {SHA}, the re-run at {'b' * 40}"

    @pytest.mark.parametrize("key", rr.PLAN_KEYS)
    def test_every_named_plan_key_is_compared(self, key: str) -> None:
        other = attempt(R2, record={"device_plan": {**PLAN, key: "changed"}})
        (found,) = rr.condition_differences(attempt(), other)
        assert found.startswith(f"device_plan.{key}: ")

    @pytest.mark.parametrize("key", rr.PLAN_TORCH_KEYS)
    def test_every_named_torch_key_is_compared(self, key: str) -> None:
        plan = {**PLAN, "torch": {**PLAN["torch"], key: "changed"}}
        (found,) = rr.condition_differences(attempt(), attempt(R2, record={"device_plan": plan}))
        assert found.startswith(f"device_plan.torch.{key}: ")

    def test_a_plan_on_one_side_only_is_a_difference(self) -> None:
        other = attempt(R2, record={"device_plan": None})
        assert rr.condition_differences(attempt(), other) == [
            "device_plan: only one of the two runs records one"
        ]
        neither = attempt(record={"device_plan": None})
        assert rr.condition_differences(neither, other) == []

    def test_the_driver_is_compared_only_where_devices_are_registered(self) -> None:
        newer = {"preflight": {"nvidia_smi": [SMI[0].replace("576.88", "580.10")]}}
        (found,) = rr.condition_differences(attempt(), attempt(R2, record=newer))
        assert found.startswith("driver: the attempt ran under NVIDIA driver ('576.88',)")
        unread = {"preflight": {"nvidia_smi": {"error": "FileNotFoundError"}}}
        (found,) = rr.condition_differences(attempt(), attempt(R2, record=unread))
        assert "of the re-run could not be read" in found
        cpu = {"devices_registered": False}
        for record in (newer, unread, {"preflight": None}):
            assert rr.condition_differences(attempt(**cpu), attempt(R2, record=record, **cpu)) == []


class TestWhichAttempts:
    @pytest.mark.parametrize(
        ("record", "expected"),
        [
            ({}, True),  # written before the key existed
            ({"launched": []}, False),
            ({"launched": None}, False),
            ({"launched": ["train"]}, True),
            ({"launched": [], "invocations": [{"name": "train"}]}, False),
        ],
    )
    def test_launched(self, record: dict[str, Any], expected: bool) -> None:
        assert rr.launched(record) is expected

    def test_needs_record(self) -> None:
        why = ("run.json status is 'VOID'",)
        assert rr.needs_record(attempt(interruptions=why)) is True
        assert rr.needs_record(attempt()) is False
        assert rr.needs_record(attempt(control_void=True)) is False
        assert rr.needs_record(attempt(interruptions=why, control_void=True)) is False
        nothing = attempt(interruptions=why, record={"launched": []})
        assert rr.needs_record(nothing) is False

    def test_a_null_identity_never_matches(self) -> None:
        """Not even an attempt whose own ``sha`` is unknown: null is not an identity."""
        unknown, other = attempt(sha=None), attempt(R2)
        blocks = {
            side: {"card_id": run.run_id, "sha": run.sha, rp.RUN_JSON_SHA256: run.run_json_sha256}
            for side, run in zip(_SIDES, (unknown, other), strict=True)
        }
        record = {rp.RECORD_SCHEMA_KEY: 1, **blocks, rp.RECORD_DIGESTS: {s: {} for s in _SIDES}}
        problems = rr.check_record(record, unknown, other)
        assert [p for p in problems if "the record is for other runs" in p] == [
            f"attempt.sha None is not {A1}'s (None): the record is for other runs"
        ]
        problems = rr.check_record({**record, rp.RECORD_ATTEMPT: None}, unknown, other)
        assert "attempt: the record names no run (was a run.json beside it?)" in problems


def test_small_helpers() -> None:
    assert rr.arena_directory("E:\\civ6az-data\\runs\\a1-20261111\\arena\\positive_control") == (
        "positive_control"
    )
    assert rr.arena_directory("/data/runs/a1-tiny/arena/primary") == "primary"
    assert rr.record_name(A1) == RECORD_PATH
    named = {rp.RECORD_ATTEMPT: {"card_id": A1}, rp.RECORD_RERUN: {"card_id": R2}}
    assert rr.record_run_ids(named) == (A1, R2)
    assert rr.record_run_ids({**named, rp.RECORD_RERUN: None}) is None
    assert rr.record_run_ids({**named, rp.RECORD_RERUN: {"card_id": None}}) is None
    assert rr.record_run_ids("not a record") is None
    assert set(rr.OUTCOMES) == {"REPRODUCED", "EMPTY", "NON_REPRODUCING"}
