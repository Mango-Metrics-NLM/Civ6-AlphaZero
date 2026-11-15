"""The re-run rule cannot be ordered around (section 7.1, "After a VOID").

``scripts/release_reproduction.py`` requires the compare record of an interrupted
attempt against its re-run: the first COMPLETED attempt of the role that started
after it. Two routes around that were open, and these tests close them:

* **Give the re-run an earlier start.** Interrupt X, re-run as Y after a code
  change, and record Y as started before X (a clock set back), or leave a
  ``started_at`` out so the order fell back to the one the bundle lists. X then had
  no later attempt, needed no record, and Y could PASS. Now every listed attempt
  must record a readable ``started_at``, the bundle's order decides nothing, and an
  interrupted attempt with no later COMPLETED attempt fails a role that is judged.
* **Draw again after a re-run that did not reproduce.** X interrupted, Y does not
  reproduce X and is VOID by a control arm, Z is a second draw that would be judged
  PASS. Section 7.1 lets no A1-family run start after a non-reproducing re-run
  until it is re-registered, so any attempt of either role that started later fails
  the bundle.

Also here: two interrupted attempts before one re-run need a record each.

``started_at`` is the operator's own clock, so this is an accident net like the
rest of the verifier: it makes the honest order the only one that verifies, not a
forged clock impossible.

Negative controls (2026-11-15). Each defect was planted in
``scripts/release_reproduction.py``, the planted text was read back from disk, this
module and ``test_verify_release_evidence_reproduction.py`` were run and the listed
tests were seen to FAIL, and the file was restored and checked byte-identical
(sha256) before the next one:

1. ``rerun_of`` counts an attempt that started in the same second as later
   (``>=``) -> ``test_an_interrupted_attempt_must_be_followed_by_its_rerun[the same
   second]`` and ``TestOrder::test_only_started_at_orders_the_attempts`` failed.
2. ``_order_problems`` does not require a readable ``started_at`` -> the five
   ``test_every_listed_attempt_needs_a_readable_started_at`` cases failed.
3. ``_order_problems`` does not require an interrupted attempt to be followed by
   its re-run -> both ``test_an_interrupted_attempt_must_be_followed_by_its_rerun``
   cases failed (the completed attempt verified with exit 0 and PASS), and the
   five ``started_at`` cases above.
4. ``_order_problems`` reports that also for a role without a non-VOID attempt
   (the ``judged`` condition dropped) ->
   ``test_a_role_without_a_judged_attempt_is_not_told_twice`` failed.
5. ``_later_run_problems`` reports nothing ->
   ``test_no_second_draw_after_a_rerun_that_did_not_reproduce`` and
   ``test_the_other_role_may_not_start_after_it_either`` failed (exit 0, with a
   second draw judged PASS).
6. ``_later_run_problems`` looks at the re-run's own role only ->
   ``test_the_other_role_may_not_start_after_it_either`` failed.
7. ``_later_run_problems`` counts the same start as "after" (``<`` for ``<=``), so
   a non-reproducing re-run flags itself -> 17 tests failed: the three bundle tests
   here that hold one, and every "is non-reproducing" test of the other module.
8. ``review`` reviews one record per re-run (a second interrupted attempt with the
   same re-run is skipped) ->
   ``test_two_interrupted_attempts_and_one_rerun_need_a_record_each`` failed.
9. ``rerun_of`` takes the last later COMPLETED attempt, not the first -> both
   ``TestOrder`` ordering tests and
   ``test_no_second_draw_after_a_rerun_that_did_not_reproduce`` failed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.evaluation.a1_protocol import NON_REPRODUCING_LABEL
from scripts import release_reproduction as rr
from scripts import verify_release_evidence as vre
from tests.scripts import release_evidence_fixture as fx
from tests.scripts.test_verify_release_evidence import verify
from tests.scripts.test_verify_release_evidence_reproduction import (
    A1,
    A1R,
    R2,
    R3,
    attempt,
    entries,
    interrupted,
    rerun,
    verdicts,
)

EARLY, LATE = "2026-11-11T22:00:00Z", "2026-11-12T09:30:00Z"
FOLLOWED = "An interrupted attempt must be followed by its re-run"


# ---------------------------------------------------------------------------
# started_at orders the attempts, and nothing else does
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rerun_first", [False, True])
def test_the_order_the_bundle_lists_attempts_in_decides_nothing(
    tmp_path: Path, rerun_first: bool
) -> None:
    specs = [interrupted(started_at=EARLY), rerun(started_at=LATE)]
    release = fx.make_release(tmp_path, specs[::-1] if rerun_first else specs)
    assert [run["run_id"] for run in release.bundle["runs"]] == (
        [R2, A1] if rerun_first else [A1, R2]
    )
    reference = entries(release)[A1]["reproduction"]
    assert (reference["rerun"], reference["outcome"]) == (R2, rr.OUTCOME_REPRODUCED)
    assert entries(release)[R2]["verdict"]["verdict"] == "PASS"
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out


@pytest.mark.parametrize(
    ("completed_at", "interrupted_at"),
    [
        pytest.param(EARLY, LATE, id="an earlier start"),
        pytest.param(LATE, LATE, id="the same second"),
    ],
)
def test_an_interrupted_attempt_must_be_followed_by_its_rerun(
    tmp_path: Path, completed_at: str, interrupted_at: str
) -> None:
    """The route: the "re-run" is recorded as started before the attempt it replaces.

    The interrupted attempt then has nothing after it to be compared with, and
    before this rule that meant no record was needed and the completed run could
    PASS. Whichever order the bundle lists them in.
    """
    specs = [rerun(started_at=completed_at), interrupted(started_at=interrupted_at)]
    for name, listed in (("as run", specs), ("reversed", specs[::-1])):
        release = fx.make_release(tmp_path / name, listed)
        assert "reproduction" not in entries(release)[A1]
        assert entries(release)[R2]["verdict"]["verdict"] == "PASS"
        code, out = verify(release)
        assert code == vre.EXIT_FAIL, out
        assert f"run {A1} (A1): it is VOID (run.json status is 'VOID', not 'COMPLETED')" in out
        assert f"no COMPLETED attempt of A1 started after it (started_at '{interrupted_at}')" in out
        assert f"yet {R2} is judged. {FOLLOWED}" in out


def test_a_role_without_a_judged_attempt_is_not_told_twice(tmp_path: Path) -> None:
    """An interrupted attempt alone: the role has no non-VOID attempt, which says it all."""
    code, out = verify(fx.make_release(tmp_path, [interrupted()]))
    assert code == vre.EXIT_FAIL
    assert "A1 has no non-VOID attempt" in out
    assert FOLLOWED not in out


def _without_started_at(run_json: dict[str, Any]) -> None:
    del run_json["started_at"]


def _started_at(value: Any) -> Callable[[dict[str, Any]], None]:
    def edit(run_json: dict[str, Any]) -> None:
        run_json["started_at"] = value

    return edit


@pytest.mark.parametrize(
    ("edit", "shown"),
    [
        pytest.param(_without_started_at, "None", id="missing"),
        pytest.param(_started_at(None), "None", id="null"),
        pytest.param(_started_at("yesterday"), "'yesterday'", id="not a time"),
        pytest.param(_started_at(""), "''", id="empty"),
        pytest.param(_started_at(1762848000), "1762848000", id="a number"),
    ],
)
def test_every_listed_attempt_needs_a_readable_started_at(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None], shown: str
) -> None:
    """The other route: no ``started_at``, so the order fell back to the bundle's.

    A clean single attempt fails for it too, and beside an interrupted attempt the
    run without a start is not "later", so it cannot stand in as the re-run.
    """
    code, out = verify(fx.make_release(tmp_path / "one", [fx.RunSpec(edit_run_json=edit)]))
    assert code == vre.EXIT_FAIL
    problem = f"(A1): run.json started_at {shown} is not a timestamp"
    assert (
        f"run {A1} {problem}" in out and "every listed attempt must record when it started" in out
    )
    for name, listed in (
        ("re-run first", [rerun(edit_run_json=edit), interrupted()]),
        ("as run", [interrupted(), rerun(edit_run_json=edit)]),
    ):
        release = fx.make_release(tmp_path / name, listed)
        assert "reproduction" not in entries(release)[A1]
        code, out = verify(release)
        assert code == vre.EXIT_FAIL
        assert f"run {R2} {problem}" in out
        assert f"yet {R2} is judged. {FOLLOWED}" in out


# ---------------------------------------------------------------------------
# Nothing starts after a non-reproducing re-run
# ---------------------------------------------------------------------------


def test_no_second_draw_after_a_rerun_that_did_not_reproduce(tmp_path: Path) -> None:
    """X interrupted; Y does not reproduce X and a control arm VOIDs it; Z would PASS."""
    specs = [
        interrupted(),
        rerun(primary_rotation=1, symmetry=list(fx.LOPSIDED)),
        fx.RunSpec(run_id=R3),
    ]
    release = fx.make_release(tmp_path, specs)
    assert verdicts(release) == ["VOID", "VOID", "PASS"]
    assert entries(release)[A1]["reproduction"]["outcome"] == rr.OUTCOME_NON_REPRODUCING
    assert entries(release)[R2]["verdict"]["label"] == NON_REPRODUCING_LABEL
    assert "label" not in entries(release)[R3]["verdict"]
    code, out = verify(release)
    assert code == vre.EXIT_FAIL, out
    assert (
        f"run {R3} (A1): it started (started_at '2026-11-13T08:00:00Z') after {R2}, a "
        "non-reproducing re-run (started_at '2026-11-12T08:00:00Z')"
    ) in out
    assert "a new registration is required first" in out
    assert "this verifier does not model the bundle of that registration yet" in out
    # Neither the interrupted attempt nor the re-run itself is "after" it.
    assert f"run {A1} (A1): it started" not in out and f"run {R2} (A1): it started" not in out


def test_the_other_role_may_not_start_after_it_either(tmp_path: Path) -> None:
    """ "No further A1-family run": A1-R is one. Before the re-run started, it stands."""
    before = [interrupted(), fx.RunSpec(**A1R), rerun(primary_rotation=1)]
    release = fx.make_release(tmp_path / "before", before)
    assert verdicts(release) == ["VOID", "PASS", "INCONCLUSIVE"]
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out
    after = [interrupted(), rerun(primary_rotation=1), fx.RunSpec(**A1R)]
    release = fx.make_release(tmp_path / "after", after)
    assert verdicts(release) == ["VOID", "INCONCLUSIVE", "PASS"]
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert f"run {A1R['run_id']} (A1-R): it started" in out
    assert f"after {R2}, a non-reproducing re-run" in out


# ---------------------------------------------------------------------------
# Two interrupted attempts, one re-run
# ---------------------------------------------------------------------------


def second_interruption(**changes: Any) -> fx.RunSpec:
    """Interrupted like the first, but its primary arm played other games than the re-run's."""
    return fx.RunSpec(
        **{"run_id": R2, "status": "VOID", "recorded": 2, "primary_rotation": 1, **changes}
    )


def test_two_interrupted_attempts_and_one_rerun_need_a_record_each(tmp_path: Path) -> None:
    """The first reproduces, the second does not: the re-run still cannot PASS."""
    specs = [interrupted(), second_interruption(), fx.RunSpec(run_id=R3)]
    release = fx.make_release(tmp_path / "both", specs)
    first, second = (entries(release)[run_id]["reproduction"] for run_id in (A1, R2))
    assert (first["rerun"], first["outcome"]) == (R3, rr.OUTCOME_REPRODUCED)
    assert (second["rerun"], second["outcome"]) == (R3, rr.OUTCOME_NON_REPRODUCING)
    assert second["path"] == f"{R2}.reproduction.json" != first["path"]
    judged = entries(release)[R3]["verdict"]
    assert (judged["verdict"], judged["label"]) == ("INCONCLUSIVE", NON_REPRODUCING_LABEL)
    assert any(f"it does not reproduce {R2}: " in reason for reason in judged["reasons"])
    assert not any(f"it does not reproduce {A1}: " in reason for reason in judged["reasons"])
    assert verdicts(release) == ["VOID", "VOID", "INCONCLUSIVE"]
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out
    assert f"run {R3} (A1): verdict INCONCLUSIVE" in out
    # Each needs its own record: leaving out the one that differs stores a PASS.
    for name, listed, missing in (
        ("first", [interrupted(compare_record=False), second_interruption(), specs[2]], A1),
        ("second", [interrupted(), second_interruption(compare_record=False), specs[2]], R2),
    ):
        release = fx.make_release(tmp_path / name, listed)
        stored = "PASS" if missing == R2 else "INCONCLUSIVE"
        assert entries(release)[R3]["verdict"]["verdict"] == stored
        code, out = verify(release)
        assert code == vre.EXIT_FAIL
        assert f"run {missing} (A1) reproduction record: missing" in out


# ---------------------------------------------------------------------------
# release_reproduction, directly
# ---------------------------------------------------------------------------

WHY = ("run.json status is 'VOID', not 'COMPLETED'",)


def at(run_id: str, started: Any, **changes: Any) -> rr.Attempt:
    return attempt(run_id, record={"started_at": started}, **changes)


class TestOrder:
    def test_the_rerun_is_the_first_later_completed_attempt(self) -> None:
        first = at("a", "2026-11-11T08:00:00Z", interruptions=WHY, completed=False)
        second = at("b", "2026-11-12T08:00:00Z", interruptions=WHY, completed=False)
        third, fourth = at("c", "2026-11-13T08:00:00Z"), at("d", "2026-11-14T08:00:00Z")
        for runs in ([first, second, third, fourth], [fourth, third, second, first]):
            assert rr.rerun_of(first, runs) is third and rr.rerun_of(second, runs) is third
            assert rr.rerun_of(third, runs) is fourth and rr.rerun_of(fourth, runs) is None
            assert rr.record_rerun(third, runs) is None  # it needs no record
        other_role = at("r", "2026-11-15T08:00:00Z", role="A1-R")
        assert rr.record_rerun(first, [first, other_role]) is None
        assert rr.record_rerun(first, [first, other_role, third]) is third

    def test_only_started_at_orders_the_attempts(self) -> None:
        void = at("void", "2026-11-11T22:00:00Z", interruptions=WHY, completed=False)
        early = at("early", "2026-11-11T08:00:00Z")
        later = at("later", "2026-11-12T08:00:00+00:00")
        latest = at("latest", "2026-11-13T08:00:00Z")
        assert rr.rerun_of(void, [latest, later, early, void]) is later
        assert rr.rerun_of(void, [early, void]) is None
        # The same second is not later, whichever is listed first.
        tie = at("tie", "2026-11-11T22:00:00Z")
        assert rr.rerun_of(void, [void, tie]) is None and rr.rerun_of(void, [tie, void]) is None
        # An attempt without a readable start is neither earlier nor later.
        for unreadable in (None, "soon", 20261112):
            unknown = at("unknown", unreadable)
            assert rr.rerun_of(void, [void, unknown]) is None
            assert rr.rerun_of(void, [unknown, void, later]) is later
            assert rr.rerun_of(at("lost", unreadable, interruptions=WHY), [later]) is None
        # A naive timestamp is read as UTC; another offset is converted.
        assert rr.rerun_of(void, [void, at("naive", "2026-11-11T22:00:01")]) is not None
        assert rr.rerun_of(void, [void, at("east", "2026-11-12T00:30:00+03:00")]) is None

    def test_void(self) -> None:
        assert attempt().void is False
        assert attempt(interruptions=WHY).void is True and attempt(control_void=True).void is True
