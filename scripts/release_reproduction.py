#!/usr/bin/env python3
"""The re-run rule of section 7.1 ("After a VOID"), applied to a release bundle.

``docs/EVALUATION.md`` section 7.1 lets an interrupted confirmatory run be run
again at the same seed under a new ``card_id``, and makes that re-run first a
**reproduction check**: everything the interrupted attempt finished is compared
with the re-run's by ``python -m alpha_zero.evaluation.reproduction compare
<attempt dir> <re-run dir> --out FILE``, and "that file goes in the bundle beside
both attempts". What the comparison says decides whether the re-run may PASS:

* ``REPRODUCED`` -- the re-run is judged;
* ``NOTHING TO COMPARE`` (the attempt finished nothing beyond ``checkpoint_0``,
  which the seed alone fixes) -- the re-run is judged, and the bundle says the
  comparison was empty (:data:`OUTCOME_EMPTY`);
* ``DIFFERS``, or another ``sha``, ``device_plan``, torch wheel, NVIDIA driver
  or ``MCTS_*`` override set (``env_var_names``) -- the re-run is a
  "non-reproducing re-run" and cannot PASS
  (:data:`OUTCOME_NON_REPRODUCING`; ``a1_protocol.compute_verdict`` reads a PASS
  row as INCONCLUSIVE). No further A1-family run may start until section 7.1 is
  re-registered, so an attempt listed as started after such a re-run fails the
  bundle (:func:`review`).

After a symmetry or positive-control VOID the code changed, so the re-run is a
second draw and no comparison applies (nothing here checks that the code did
change). An attempt whose ``run.json`` records no launched invocation played
nothing: it is not an attempt and is not listed.

**Order.** A role's attempts are ordered by ``run.json``'s ``started_at`` and by
nothing else. Every listed attempt must record a readable one, and no two
attempts of a role may record the same one, so the order the bundle lists
attempts in decides nothing. The role's non-VOID attempt is its last: every other
listed attempt of the role must have started strictly before it ("each role has
exactly one non-VOID attempt, and that attempt is the one judged"; nothing of a
role runs after it). That is what makes an interrupted attempt be followed by its
re-run, and what keeps a second draw from being dated at or before a
non-reproducing re-run (:func:`_order_problems`).

This module is that rule and nothing else. It takes plain data -- an
:class:`Attempt` per listed run, built by the caller from files it has already
read and hashed -- and returns problems as strings, so
``scripts/verify_release_evidence.py`` (which recomputes every stored outcome
with :func:`review`) and ``scripts/build_release_bundle.py`` (which writes them)
share one implementation, and neither's own types leak in here. Both import it
lazily: it imports ``alpha_zero``, which the verifier's ``--scope`` path must not.

What a record can and cannot show (:func:`check_record`):

* its arena and iteration digests are held to the files the bundle carries for
  both runs, in both directions, so a record that leaves out a bundled arm, or
  was made from other files, fails. What the bundle may omit (an arm that wrote
  ``result.json`` after the last invocation ``run.json`` records, the iteration
  records of an attempt killed during training) the record can omit with it;
* its ``status`` and ``differences`` are recomputed from its own digests
  (``reproduction.compare_status``);
* its **checkpoint tensor digests are not anchored to anything**: checkpoints are
  too large to bundle, so the record is the only account of them;
* so :data:`OUTCOME_EMPTY` means the record shows nothing beyond ``checkpoint_0``,
  not that the attempt is proved to have finished nothing more: for an attempt
  killed during training it is reachable by omission.

Limits no check here can close: an attempt that is simply not listed (an
interrupted one, or a completed re-run that did not reproduce) is invisible to a
bundle verifier, and ``started_at`` is the operator's own
record, written by the operator's own machine. The order rules refuse the
orderings they name; a forged clock that stays consistent with them is not
detectable, and a ``started_at`` is not checked against anything outside the
bundle (not the commit date of its ``sha``). Not modelled: the bundle of the new
registration that section 7.1 requires after a non-reproducing re-run (a new
confirmatory seed pair, listing both attempts).

Conservative readings, where section 7.1 is silent or looser (the maintainer's to
overrule): ``docs/evidence/SCHEMA.md``, "Conservative readings".
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

from alpha_zero.evaluation import reproduction
from alpha_zero.local.devices import driver_versions

logger = logging.getLogger(__name__)

__all__ = [
    "OUTCOMES",
    "OUTCOME_EMPTY",
    "OUTCOME_NON_REPRODUCING",
    "OUTCOME_REPRODUCED",
    "RECORD_SUFFIX",
    "Attempt",
    "Outcome",
    "arena_directory",
    "bundle_reference",
    "check_record",
    "condition_differences",
    "launched",
    "needs_record",
    "outcome",
    "record_name",
    "record_rerun",
    "record_run_ids",
    "rerun_of",
    "review",
    "verdict_reasons",
]

#: What a bundle's ``reproduction`` entry says of the re-run (``outcome``).
OUTCOME_REPRODUCED = "REPRODUCED"
#: The comparison was empty (``NOTHING TO COMPARE``): the re-run is still judged.
OUTCOME_EMPTY = "EMPTY"
#: The re-run differs, or ran under other conditions: it cannot PASS.
OUTCOME_NON_REPRODUCING = "NON_REPRODUCING"
OUTCOMES = (OUTCOME_REPRODUCED, OUTCOME_EMPTY, OUTCOME_NON_REPRODUCING)

#: A record's file name in the bundle root is ``<VOID attempt's run_id>`` + this.
RECORD_SUFFIX = ".reproduction.json"

# ``run.json`` keys read here (``alpha_zero/local/run_card.py`` writes them).
LAUNCHED_KEY = "launched"
STARTED_AT_KEY = "started_at"
DEVICE_PLAN_KEY = "device_plan"
PREFLIGHT_KEY = "preflight"
NVIDIA_SMI_KEY = "nvidia_smi"
PLAN_TORCH_KEY = "torch"
#: ``{url, sha256}`` of the torch wheel pip installed, when pip reported it.
TORCH_WHEEL_KEY = "torch_wheel"
WHEEL_SHA256_KEY = "sha256"
ENV_VAR_NAMES_KEY = "env_var_names"
MCTS_ENV_PREFIX = "MCTS_"

#: ``device_plan`` keys a re-run must share with the attempt it replaces. Named,
#: never the whole mapping: ``source`` says where a value came from, not what it was.
PLAN_KEYS: tuple[str, ...] = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUBLAS_WORKSPACE_CONFIG",
    "torch_pin",
    "requested",
    "expected_gpus",
)
#: The same, inside ``device_plan.torch`` (the pinned child's own report).
PLAN_TORCH_KEYS: tuple[str, ...] = ("version", "cuda", "cudnn", "devices", "capabilities")

# The bundle's ``reproduction`` entry.
REFERENCE_RERUN = "rerun"
REFERENCE_OUTCOME = "outcome"
REFERENCE_REASONS = "reasons"

_SIDES = (reproduction.RECORD_ATTEMPT, reproduction.RECORD_RERUN)
#: The component kinds a bundle carries files for (checkpoints are not bundled).
_BUNDLED_KINDS = (reproduction.COMPONENT_ARENAS, reproduction.COMPONENT_ITERATIONS)


@dataclass(frozen=True)
class Attempt:
    """One listed attempt, as the re-run rule reads it.

    ``record`` is the attempt's ``run.json`` and ``run_json_sha256`` the hash of
    that file as bundled. ``interruptions`` are the interruption-class reasons it
    is VOID (``run.json`` not COMPLETED, a resumed iteration record, an
    ``MCTS_*`` override, a ``device_plan`` other than the registered one);
    ``control_void`` says its judged arms alone make it VOID (the symmetry arm or
    the positive control). ``devices_registered`` says section 7.1 registers
    devices at the attempt's commit. ``iterations`` and ``arenas`` are the
    **bundled** copies of its ``iterations.jsonl`` and of each arm's
    ``result.json``, the arms keyed by their directory name in the run directory
    (:func:`arena_directory`), which is how a compare record names them.
    """

    run_id: str
    role: str
    sha: str | None
    record: Mapping[str, Any]
    run_json_sha256: str
    completed: bool
    interruptions: tuple[str, ...] = ()
    control_void: bool = False
    devices_registered: bool = False
    iterations: Path | None = None
    arenas: Mapping[str, Path] = field(default_factory=dict)

    @property
    def void(self) -> bool:
        """VOID: not COMPLETED, another interruption-class reason, or a control arm.

        So an attempt that is not VOID is always a COMPLETED one.
        """
        return not self.completed or bool(self.interruptions) or self.control_void


@dataclass(frozen=True)
class Outcome:
    """What a comparison means for the re-run: one of :data:`OUTCOMES`, and why."""

    kind: str
    reasons: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Which attempts need a record, and against which re-run
# ---------------------------------------------------------------------------


def launched(record: Mapping[str, Any]) -> bool:
    """Whether ``run.json`` records a launched invocation.

    ``run_card`` writes ``launched`` before each child starts, so an attempt
    interrupted inside its first invocation has launched something, and one that
    was refused at its preflight or device check has not. A ``run.json`` written
    before the key existed counts as launched.
    """
    if LAUNCHED_KEY not in record:
        return True
    return bool(record[LAUNCHED_KEY])


def needs_record(attempt: Attempt) -> bool:
    """Whether the attempt's re-run is a reproduction check (section 7.1).

    True for an attempt that is VOID for an interruption-class reason and launched
    something. False for a symmetry or positive-control VOID, even an interrupted
    one: the code changes before its re-run, which is then a second draw.
    """
    return bool(attempt.interruptions) and not attempt.control_void and launched(attempt.record)


def _started(attempt: Attempt) -> datetime | None:
    value = attempt.record.get(STARTED_AT_KEY)
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def rerun_of(attempt: Attempt, role_attempts: Sequence[Attempt]) -> Attempt | None:
    """The attempt's re-run: the first COMPLETED attempt of its role that started after it.

    "After" is by ``run.json``'s ``started_at`` and strict: an attempt that started
    in the same second is not later, and one without a readable ``started_at`` is
    neither earlier nor later than anything (:func:`review` reports it). The order
    of ``role_attempts`` decides nothing: between two later attempts that record
    the same start the smaller ``run_id`` is taken, and :func:`review` refuses such
    a tie anyway, so a bundle cannot reorder its way to another answer. The re-run
    need not be the attempt that ends up judged: an interrupted attempt's re-run
    can itself be VOID by a control arm, and a second draw follows it.
    """
    mine = _started(attempt)
    if mine is None:
        return None
    later = [
        (start, other)
        for other in role_attempts
        if other is not attempt
        and other.completed
        and (start := _started(other)) is not None
        and start > mine
    ]
    return min(later, key=lambda item: (item[0], item[1].run_id))[1] if later else None


def record_rerun(attempt: Attempt, attempts: Sequence[Attempt]) -> Attempt | None:
    """The re-run whose comparison with ``attempt`` the bundle must hold, or ``None``.

    ``attempts`` is every listed attempt. ``None`` when no comparison applies: the
    attempt needs no record (:func:`needs_record`), or no COMPLETED attempt of its
    role started after it (then the role's non-VOID attempt, if it has one, did not
    start after it either, which :func:`review` reports).
    """
    if not needs_record(attempt):
        return None
    return rerun_of(attempt, [other for other in attempts if other.role == attempt.role])


def _no_comparison(attempt: Attempt) -> str:
    """Why :func:`record_rerun` is ``None`` for this attempt."""
    if attempt.control_void:
        return "it is a symmetry or positive-control VOID, so its re-run is a second draw"
    if not attempt.interruptions:
        return "it was not interrupted"
    if not launched(attempt.record):
        return "it launched nothing"
    return f"no COMPLETED attempt of {attempt.role} started after it"


# ---------------------------------------------------------------------------
# The conditions a reproduction is claimed under
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> str:
    """JSON text equal exactly when two values are the same JSON value (1 is not 1.0)."""
    return json.dumps(value, sort_keys=True, default=str)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _driver(attempt: Attempt) -> tuple[str, ...] | None:
    return driver_versions(_mapping(attempt.record.get(PREFLIGHT_KEY)).get(NVIDIA_SMI_KEY))


def _mcts_overrides(attempt: Attempt) -> tuple[str, ...]:
    names = attempt.record.get(ENV_VAR_NAMES_KEY)
    if not isinstance(names, Sequence) or isinstance(names, str | bytes):
        return ()
    return tuple(sorted(str(n) for n in names if str(n).upper().startswith(MCTS_ENV_PREFIX)))


def condition_differences(attempt: Attempt, rerun: Attempt) -> list[str]:
    """How the re-run's conditions depart from the attempt's; empty when they do not.

    Section 7.1 claims a reproduction only "at the VOID attempt's ``sha`` with the
    same ``device_plan`` (torch pin, device order, GPUs) and the same NVIDIA
    driver", and its Devices paragraph adds "the same torch wheel";
    ``docs/evidence/SCHEMA.md`` ("Conservative readings") adds that clearing or
    changing ``MCTS_*`` overrides makes the re-run non-reproducing. Compared: the
    ``sha``; when both runs record a ``device_plan``, its :data:`PLAN_KEYS` and its
    ``torch`` report's :data:`PLAN_TORCH_KEYS` (a plan on one side only is a
    difference); the sha256 of the installed torch wheel, when **both** runs record
    one (pip does not always report it, so a run without the record is not held to
    it); when section 7.1 registers devices for either run, the driver version
    of every GPU -- a driver that cannot be read on either side is a difference,
    because it cannot be shown to be the same; and the sorted ``MCTS_*`` names
    in ``run.json``'s ``env_var_names`` (absent or non-sequence records count as
    none).
    """
    found: list[str] = []
    if attempt.sha != rerun.sha:
        found.append(f"sha: the attempt ran at {attempt.sha}, the re-run at {rerun.sha}")
    mine, theirs = (run.record.get(DEVICE_PLAN_KEY) for run in (attempt, rerun))
    if isinstance(mine, Mapping) != isinstance(theirs, Mapping):
        found.append("device_plan: only one of the two runs records one")
    elif isinstance(mine, Mapping) and isinstance(theirs, Mapping):
        pairs = [(key, mine.get(key), theirs.get(key)) for key in PLAN_KEYS]
        torch_mine, torch_theirs = (
            _mapping(mine.get(PLAN_TORCH_KEY)),
            _mapping(theirs.get(PLAN_TORCH_KEY)),
        )
        pairs += [
            (f"{PLAN_TORCH_KEY}.{key}", torch_mine.get(key), torch_theirs.get(key))
            for key in PLAN_TORCH_KEYS
        ]
        found += [
            f"device_plan.{key}: the attempt had {_canonical(a)}, the re-run {_canonical(b)}"
            for key, a, b in pairs
            if _canonical(a) != _canonical(b)
        ]
    wheels = [
        _mapping(run.record.get(TORCH_WHEEL_KEY)).get(WHEEL_SHA256_KEY) for run in (attempt, rerun)
    ]
    if all(isinstance(wheel, str) and wheel for wheel in wheels) and wheels[0] != wheels[1]:
        found.append(
            f"{TORCH_WHEEL_KEY}: the attempt installed the wheel with sha256 {wheels[0]}, the "
            f"re-run {wheels[1]}"
        )
    if attempt.devices_registered or rerun.devices_registered:
        drivers = {"the attempt": _driver(attempt), "the re-run": _driver(rerun)}
        unread = [who for who, versions in drivers.items() if versions is None]
        if unread:
            found.append(
                f"driver: the NVIDIA driver version of {' and '.join(unread)} could not be read "
                f"from run.json's {PREFLIGHT_KEY}.{NVIDIA_SMI_KEY}, so it cannot be shown to be "
                "the same"
            )
        elif drivers["the attempt"] != drivers["the re-run"]:
            found.append(
                f"driver: the attempt ran under NVIDIA driver {drivers['the attempt']}, the "
                f"re-run under {drivers['the re-run']}"
            )
    mcts_mine, mcts_theirs = _mcts_overrides(attempt), _mcts_overrides(rerun)
    if mcts_mine != mcts_theirs:
        found.append(
            f"{MCTS_ENV_PREFIX}* overrides: the attempt had {list(mcts_mine)}, "
            f"the re-run {list(mcts_theirs)}"
        )
    return found


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def arena_directory(out: str) -> str:
    """An arm's directory name in its run directory: the last part of its ``--out``.

    The registered ``--out`` is ``<D>\\arena\\<name>`` (a Windows path; either
    separator is accepted), and ``reproduction.run_digest`` names an arm by that
    directory.
    """
    return PureWindowsPath(out).name


def record_name(run_id: str) -> str:
    """The bundle-relative path of the record for the VOID attempt ``run_id``."""
    return f"{run_id}{RECORD_SUFFIX}"


def record_run_ids(record: Any) -> tuple[str, str] | None:
    """``(attempt card_id, re-run card_id)`` a record names, or ``None`` when it names none."""
    sides = [_mapping(_mapping(record).get(side)).get("card_id") for side in _SIDES]
    if not all(isinstance(card_id, str) and card_id for card_id in sides):
        return None
    return str(sides[0]), str(sides[1])


def _bundled_digest(run: Attempt) -> dict[str, dict[str, Any]]:
    """The arena and iteration digests of the files the bundle carries for ``run``."""
    arenas: dict[str, Any] = {}
    for name, path in run.arenas.items():
        try:
            arenas[name] = reproduction.arena_digest(path)
        except (OSError, ValueError) as exc:
            # Absent from the digest, so the record's entry for it has no counterpart.
            logger.warning("bundled arena %s of %s cannot be digested: %s", name, run.run_id, exc)
    iterations: dict[str, Any] = {}
    if run.iterations is not None:
        iterations = reproduction.iteration_digests(run.iterations, {})
    return {
        reproduction.COMPONENT_ARENAS: arenas,
        reproduction.COMPONENT_ITERATIONS: iterations,
    }


def _bundle_contradictions(digest: Mapping[str, Any], run: Attempt) -> list[str]:
    """How a record's digest of ``run`` disagrees with the files bundled for ``run``."""
    bundled = _bundled_digest(run)
    found: list[str] = []
    for kind in _BUNDLED_KINDS:
        recorded, held = _mapping(digest.get(kind)), bundled[kind]
        only_record = sorted(set(recorded) - set(held))
        only_bundle = sorted(set(held) - set(recorded))
        changed = sorted(
            name
            for name in set(recorded) & set(held)
            if _canonical(recorded[name]) != _canonical(held[name])
        )
        if only_record:
            found.append(f"{kind} {only_record} are in the record but not bundled for {run.run_id}")
        if only_bundle:
            found.append(
                f"{kind} {only_bundle} are bundled for {run.run_id} but missing from the record"
            )
        if changed:
            found.append(f"{kind} {changed} differ from the files bundled for {run.run_id}")
    return found


def _recompute(digests: Mapping[str, Any]) -> tuple[str, list[str]]:
    """``compare_status`` of a record's two digests; digests of another shape are a difference.

    A record is input, so a digest that is not shaped like one must fail the
    record, never the caller.
    """
    try:
        return reproduction.compare_status(*(_mapping(digests.get(side)) for side in _SIDES))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("a compare record's digests cannot be compared: %r", exc)
        return reproduction.STATUS_DIFFERS, [
            f"the record's digests cannot be compared ({type(exc).__name__}: {exc})"
        ]


def check_record(record: Any, attempt: Attempt, rerun: Attempt) -> list[str]:
    """Why a compare record cannot be used for this attempt and re-run; empty when it can.

    * its schema version is ``reproduction.RECORD_SCHEMA_VERSION``;
    * it names these two runs: each side's ``card_id``, ``sha`` and
      ``run_json_sha256`` are strings and are the bundle's (a null never matches,
      whatever the bundle's attempt holds);
    * its arena and iteration digests equal the digests of the files the bundle
      carries for each run, in both directions (checkpoint digests cannot be
      checked: the bundle holds no checkpoints);
    * its ``status`` and ``differences`` are what ``reproduction.compare_status``
      gives for its own digests.
    """
    if not isinstance(record, Mapping):
        return ["the record is not a JSON object"]
    problems: list[str] = []
    version = record.get(reproduction.RECORD_SCHEMA_KEY)
    if version != reproduction.RECORD_SCHEMA_VERSION:
        problems.append(
            f"{reproduction.RECORD_SCHEMA_KEY} {version!r} != {reproduction.RECORD_SCHEMA_VERSION} "
            "(written by another version of `reproduction compare`?)"
        )
    runs = dict(zip(_SIDES, (attempt, rerun), strict=True))
    for side, run in runs.items():
        identity = record.get(side)
        if not isinstance(identity, Mapping):
            problems.append(f"{side}: the record names no run (was a run.json beside it?)")
            continue
        expected = {
            "card_id": run.run_id,
            "sha": run.sha,
            reproduction.RUN_JSON_SHA256: run.run_json_sha256,
        }
        problems += [
            f"{side}.{key} {identity.get(key)!r} is not {run.run_id}'s ({want!r}): the record "
            "is for other runs"
            for key, want in expected.items()
            if not isinstance(identity.get(key), str) or identity.get(key) != want
        ]
    digests = _mapping(record.get(reproduction.RECORD_DIGESTS))
    if not all(isinstance(digests.get(side), Mapping) for side in _SIDES):
        problems.append(f"the record holds no {reproduction.RECORD_DIGESTS} for both runs")
        return problems
    for side, run in runs.items():
        problems += [f"{side}: {p}" for p in _bundle_contradictions(digests[side], run)]
    status, differences = _recompute(digests)
    stored = record.get(reproduction.RECORD_STATUS)
    if stored != status:
        problems.append(f"status {stored!r}, but its own digests give {status!r}")
    if record.get(reproduction.RECORD_DIFFERENCES) != differences:
        problems.append("its differences are not the ones its own digests give")
    return problems


def outcome(record: Any, attempt: Attempt, rerun: Attempt) -> Outcome:
    """What the record means for the re-run, recomputed from the record's digests.

    Any difference between the digests, or between the two runs' conditions
    (:func:`condition_differences`), makes the re-run non-reproducing -- also when
    the comparison itself was empty. Otherwise an empty comparison is
    :data:`OUTCOME_EMPTY` and anything else :data:`OUTCOME_REPRODUCED`. A record
    without digests cannot show a reproduction, so it is non-reproducing.
    """
    status, differences = _recompute(_mapping(_mapping(record).get(reproduction.RECORD_DIGESTS)))
    reasons = [*differences, *condition_differences(attempt, rerun)]
    if reasons:
        return Outcome(OUTCOME_NON_REPRODUCING, tuple(reasons))
    if status == reproduction.STATUS_NOTHING_TO_COMPARE:
        return Outcome(OUTCOME_EMPTY)
    return Outcome(OUTCOME_REPRODUCED)


def verdict_reasons(attempt: Attempt, result: Outcome) -> list[str]:
    """``compute_verdict(non_reproducing=...)`` reasons for the re-run of ``attempt``."""
    return [f"it does not reproduce {attempt.run_id}: {reason}" for reason in result.reasons]


def bundle_reference(written: Mapping[str, str], rerun: Attempt, result: Outcome) -> dict[str, Any]:
    """The VOID attempt's ``reproduction`` entry; ``written`` is the copy's ``{path, sha256}``."""
    return {
        **written,
        REFERENCE_RERUN: rerun.run_id,
        REFERENCE_OUTCOME: result.kind,
        REFERENCE_REASONS: list(result.reasons),
    }


# ---------------------------------------------------------------------------
# A whole bundle
# ---------------------------------------------------------------------------


def _started_text(attempt: Attempt) -> str:
    return f"{STARTED_AT_KEY} {attempt.record.get(STARTED_AT_KEY)!r}"


def _order_problems(attempts: Sequence[Attempt]) -> list[str]:
    """What is wrong with when the listed attempts started (module docstring, "Order").

    * Every listed attempt records a readable ``started_at``.
    * No two attempts of one role record the same instant. ``run_card`` writes
      ``started_at`` to the second and two runs of a role do not start in one, so
      an honest bundle has no tie; a tie would let the listing pick the re-run.
    * In a role that has a non-VOID attempt, every other listed attempt of the
      role started strictly before it. Section 7.1 judges a role's one non-VOID
      attempt, and nothing of the role runs after it. This is one rule for two
      routes: an interrupted attempt dated after the attempt that is judged (it
      would have nothing after it to be compared with), and a second draw dated at
      or before a non-reproducing re-run (it would escape
      :func:`_later_run_problems`, which the last attempt of a role cannot).

    A pair with an unreadable ``started_at`` is reported by the first rule only.
    """
    problems = [
        f"run {attempt.run_id} ({attempt.role}): run.json {_started_text(attempt)} is not a "
        "timestamp. The re-run rule orders a role's attempts by it, so every listed attempt "
        "must record when it started"
        for attempt in attempts
        if _started(attempt) is None
    ]
    for role in dict.fromkeys(attempt.role for attempt in attempts):
        timed = [
            (start, attempt)
            for attempt in attempts
            if attempt.role == role and (start := _started(attempt)) is not None
        ]
        first_at: dict[datetime, Attempt] = {}
        for start, attempt in timed:
            earlier = first_at.setdefault(start, attempt)
            if earlier is not attempt:
                problems.append(
                    f"run {attempt.run_id} ({role}): it records the same start as "
                    f"{earlier.run_id} ({_started_text(attempt)}). Two attempts of one role do "
                    f"not start in the same second, and a role's attempts are ordered by "
                    f"{STARTED_AT_KEY} alone, so a tie would leave their order to the listing"
                )
        for end, judged in timed:
            if judged.void:
                continue
            problems += [
                f"run {other.run_id} ({role}): it did not start before {judged.run_id}, the "
                f"role's non-VOID attempt ({_started_text(other)} against "
                f"{_started_text(judged)}). Section 7.1 judges the one non-VOID attempt of a "
                "role and nothing of that role runs after it, so every other listed attempt "
                "must have started strictly before it. An interrupted attempt in particular "
                "must be followed by its re-run and compared with it ('After a VOID')"
                for start, other in timed
                if other is not judged and start >= end
            ]
    return problems


def _later_run_problems(attempts: Sequence[Attempt], reruns: Sequence[Attempt]) -> list[str]:
    """Attempts listed as started after a non-reproducing re-run: section 7.1 allows none.

    "No further A1-family run starts until this section is re-registered with a
    new confirmatory seed pair", so any attempt of either role that started later
    was made under a registration that no longer stood.
    """
    problems: list[str] = []
    for rerun in reruns:
        cutoff = _started(rerun)
        for other in attempts:
            start = _started(other)
            if cutoff is None or start is None or start <= cutoff:
                continue
            problems.append(
                f"run {other.run_id} ({other.role}): it started ({_started_text(other)}) after "
                f"{rerun.run_id}, a non-reproducing re-run ({_started_text(rerun)}). Section 7.1 "
                "('After a VOID') lets no further A1-family run start until the section is "
                "re-registered with a new confirmatory seed pair: a new registration is required "
                "first, and this verifier does not model the bundle of that registration yet"
            )
    return problems


def review(
    listed: Sequence[tuple[Attempt, Any]], load: Callable[[Any, str], Any]
) -> tuple[list[str], dict[str, list[str]]]:
    """Apply the rule to every listed attempt: ``(problems, non-reproducing re-runs)``.

    ``listed`` pairs each attempt with its bundle entry's ``reproduction`` value
    (``None`` when it has none); the order of ``listed`` decides nothing.
    ``load(reference, label)`` returns the parsed record a reference names, or
    ``None`` when it cannot be read; the loader reports why, and re-hashes the file.

    For each attempt whose re-run is a reproduction check a record is required,
    checked (:func:`check_record`) and its outcome recomputed (:func:`outcome`);
    the stored ``outcome`` and ``rerun`` must equal the recomputed ones (the
    stored ``reasons`` are prose and are not compared). An entry on an attempt no
    comparison applies to is a problem too: it would be a stored claim nothing
    recomputes. So are an unreadable or tied ``started_at``, an attempt that did
    not start before its role's non-VOID attempt (:func:`_order_problems`) and any
    attempt started after a non-reproducing re-run (:func:`_later_run_problems`;
    its cutoff is the re-run's ``started_at``, not when it finished). The second
    value maps a re-run's ``run_id`` to why it is a non-reproducing re-run
    (:func:`verdict_reasons`).
    """
    attempts = [attempt for attempt, _ in listed]
    problems = _order_problems(attempts)
    non_reproducing: dict[str, list[str]] = {}
    flagged: list[Attempt] = []  # the non-reproducing re-runs, once each
    for attempt, reference in listed:
        label = f"run {attempt.run_id} ({attempt.role}) reproduction record"
        rerun = record_rerun(attempt, attempts)
        if rerun is None:
            if reference is not None:
                problems.append(
                    f"{label}: listed, but no comparison applies to this attempt "
                    f"({_no_comparison(attempt)})"
                )
            continue
        if reference is None:
            problems.append(
                f"{label}: missing. The attempt is VOID ({'; '.join(attempt.interruptions)}) and "
                f"{rerun.run_id} is its re-run, so the bundle must hold the file written by "
                "`python -m alpha_zero.evaluation.reproduction compare <attempt dir> "
                "<re-run dir> --out FILE` (build_release_bundle.py --compare FILE; section "
                "7.1, 'After a VOID')"
            )
            continue
        record = load(reference, label)
        if record is None:
            continue
        problems += [f"{label}: {p}" for p in check_record(record, attempt, rerun)]
        result = outcome(record, attempt, rerun)
        stored = _mapping(reference)
        if (stored.get(REFERENCE_OUTCOME), stored.get(REFERENCE_RERUN)) != (
            result.kind,
            rerun.run_id,
        ):
            problems.append(
                f"{label}: stored outcome {stored.get(REFERENCE_OUTCOME)!r} against "
                f"{stored.get(REFERENCE_RERUN)!r} != recomputed {result.kind!r} against "
                f"{rerun.run_id!r} ({'; '.join(result.reasons) or 'no difference'})"
            )
        if result.kind == OUTCOME_NON_REPRODUCING:
            non_reproducing.setdefault(rerun.run_id, []).extend(verdict_reasons(attempt, result))
            if not any(other is rerun for other in flagged):
                flagged.append(rerun)
        logger.info("%s: %s against %s", label, result.kind, rerun.run_id)
    return problems + _later_run_problems(attempts, flagged), non_reproducing
