#!/usr/bin/env python3
"""Build a release evidence bundle from run-card run directories (plan section 6, LF-D8).

The producer for what ``scripts/verify_release_evidence.py`` checks. It reads each
attempt's run directory exactly as ``scripts/run_card.py`` left it --
``run.json``, ``<D>/ckpt/iterations.jsonl``, the checkpoints and every arena arm's
``<D>/arena/<arm>/result.json`` -- and writes
``docs/evidence/release_vX_Y_Z/``:

* a copy of each attempt's ``run.json``, ``iterations.jsonl`` and arm
  ``result.json`` files under ``<run_id>/`` (VOID attempts included: list every
  attempt, ``--run A1=<first> --run A1=<re-run>``);
* a copy of each ``--compare`` record as ``<VOID attempt's run_id>.reproduction.json``
  in the bundle root: the output of ``python -m alpha_zero.evaluation.reproduction
  compare <VOID attempt dir> <re-run dir> --out FILE``, which section 7.1 ("After a
  VOID") requires for every interrupted attempt that has a re-run. The record
  names both runs, so it is matched to them by ``card_id``;
* ``bundle.json`` (``docs/evidence/SCHEMA.md``, "Release evidence bundle"): the
  sha256 of every copy, each arm's summary (``verify_release_evidence.arm_summary``),
  each attempt's verdict (``a1_protocol.compute_verdict``, or VOID by status; a
  re-run that does not reproduce its attempt is labelled and cannot PASS), the
  checkpoints' digests, ``games_total`` and the evidence block;
* ``report.md``, verdict-neutral: it never says the model learns unless the
  judged A1 and A1-R both PASS;
* the row in ``docs/evidence/README.md`` (added if absent, never rewritten).

Nothing here re-implements a check. The run facts, interruptions and summaries
come from the verifier's own helpers, and before anything is published the
staged bundle is run through ``verify_release_evidence.verify`` with every check
but the rc's CI check runs, which need the GitHub API (run the verifier with
``--repo OWNER/NAME`` afterwards for those). A bundle that fails is not written.

It refuses to overwrite: an existing bundle directory is never touched.

Usage::

    python scripts/build_release_bundle.py --release 2.1.0 --rc v2.1.0-rc.1 \\
        --run A1=E:\\civ6az-data\\runs\\a1-20261111 \\
        --run A1-R=E:\\civ6az-data\\runs\\a1r-20261112

    # after an interruption: both attempts, and the compare record between them
    python scripts/build_release_bundle.py --release 2.1.0 --rc v2.1.0-rc.1 \\
        --run A1=E:\\civ6az-data\\runs\\a1-20261111 \\
        --run A1=E:\\civ6az-data\\runs\\a1-20261111-r2 --compare compare.json

Exit status: 0 built, 1 the staged bundle failed verification, 2 refused (the
bundle exists, an input is missing or unreadable, the registration cannot be read).
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import verify_release_evidence as vre

EXIT_BUILT, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
CHECKPOINT_FLAGS = ("--checkpoint", "--opponent-checkpoint")
JUDGED = (vre.PRIMARY, vre.SYMMETRY, vre.POSITIVE_CONTROL)


class BuildError(Exception):
    """An input is missing or unreadable, or the bundle may not be written."""


@dataclass
class RunInput:
    role: str
    run_dir: Path


@dataclass
class Collected:
    """One attempt, read: its ``runs[]`` entry (without a verdict) and what judges it."""

    entry: dict[str, Any]
    #: ``release_reproduction.Attempt``: what the re-run rule reads of this attempt. Its
    #: bundled file paths are left out: nothing here holds a record to the bundled
    #: files, the verifier does that on the staged bundle.
    attempt: Any
    #: The judged arms' per-game records (primary, symmetry, positive control);
    #: ``None`` when they never all ran (VOID by status).
    judged: tuple[Any, ...] | None
    interruptions: list[str] = field(default_factory=list)


def parse_run(spec: str) -> RunInput:
    """``ROLE=DIR`` -> :class:`RunInput`; ROLE is ``A1`` or ``A1-R``."""
    role, sep, directory = spec.partition("=")
    if not sep or role not in vre.ROLES or not directory:
        raise argparse.ArgumentTypeError(f"--run takes ROLE=DIR with ROLE in {list(vre.ROLES)}")
    return RunInput(role, Path(directory))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"{path}: {exc}") from exc


def _copy(src: Path, bundle_dir: Path, rel: str) -> dict[str, str]:
    """Copy ``src`` byte for byte to ``bundle_dir/rel``; return ``{path, sha256}``."""
    try:
        target = vre._component_path(bundle_dir, rel)
    except ValueError as exc:
        raise BuildError(str(exc)) from exc
    resolved_root = bundle_dir.resolve()
    resolved_target = target.resolve()
    if not resolved_target.is_relative_to(resolved_root) or resolved_target == resolved_root:
        raise BuildError(f"bundle path {rel!r} escapes {bundle_dir}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, target)
    return {"path": rel, "sha256": vre.sha256_file(target)}


def _local(registered_path: str, card_dir: str, run_dir: Path) -> Path:
    """A registered ``<D>``-relative path, found under the run directory as given.

    ``<D>`` is where run_card ran (``E:\\civ6az-data\\runs\\<card>``); the run
    directory passed here may be that one or a copy of it elsewhere.
    """
    if registered_path.startswith(card_dir):
        rest = registered_path[len(card_dir) :].replace("\\", "/").strip("/")
        return run_dir.joinpath(*[p for p in rest.split("/") if p])
    return Path(registered_path)


def collect_run(run: RunInput, bundle_dir: Path, git: Any, rc_sha: str) -> Collected:
    """One attempt, its components copied under ``bundle_dir/<run_id>/``.

    The entry has no verdict yet: a re-run's verdict depends on its comparison with
    the attempt it replaces (:func:`attach_records`, then :func:`set_verdicts`).
    """
    from alpha_zero.evaluation.a1_protocol import compute_verdict
    from alpha_zero.evaluation.a1_registration import RegistrationError, run_directory
    from alpha_zero.evaluation.benchmark import checkpoint_opponent_id
    from alpha_zero.utils.data_root import DataRootError, validate_run_id
    from scripts import release_reproduction as rr

    run_json = run.run_dir / "run.json"
    if not run_json.is_file():
        raise BuildError(f"{run.run_dir}: no run.json (is it a run_card run directory?)")
    record = _read_json(run_json)
    run_id = str(record.get("card_id", "")) if isinstance(record, dict) else ""
    label = f"run {run_id or '?'} ({run.role})"
    facts_findings = vre.Findings()
    facts = vre.check_run_json(record, run_id, label, facts_findings)
    if facts.sha is None or facts_findings.problems:
        raise BuildError("; ".join(facts_findings.problems) or f"{label}: run.json has no sha")
    try:
        validate_run_id(run_id)
    except DataRootError as exc:
        raise BuildError(f"{label}: {exc}") from exc
    if "\n" in run_id or "\r" in run_id:
        raise BuildError(f"{label}: invalid run id {run_id!r}: must not contain newlines")
    if not rr.launched(record):
        raise BuildError(
            f"{label}: run.json records no launched invocation, so it played nothing: it is "
            "not an attempt and is not listed (section 7.1, 'After a VOID'); drop its --run"
        )
    if not git.exists(facts.sha):
        raise BuildError(f"{label}: commit {facts.sha[:12]} is not in this repository")
    try:
        reg = vre.read_registration(git, facts.sha, run_id, run.role)
    except (RegistrationError, ValueError) as exc:
        raise BuildError(f"{label}: section 7.1 at {facts.sha[:12]}: {exc}") from exc
    # A run on other devices than registered is VOID, as the verifier reads it.
    vre.check_registered_devices(facts, reg)
    card_dir = run_directory(run_id)
    base = run_id

    entry: dict[str, Any] = {
        "run_id": run_id,
        "role": run.role,
        "seed": reg.seed,
        "git_sha": facts.sha,
        "games": facts.games,
        "approval_ref": facts.approval_ref,
        "run_json": _copy(run_json, bundle_dir, f"{base}/run.json"),
    }

    finished = [n for n in facts.names if facts.exit_codes.get(n) == 0]
    trained = facts.completed or vre.TRAIN in finished
    ckpt_dir = reg.flag(vre.TRAIN, "--checkpoint_dir")
    iterations = _local(ckpt_dir, card_dir, run.run_dir) / "iterations.jsonl" if ckpt_dir else None
    if iterations is not None and iterations.is_file():
        entry["iterations"] = _copy(iterations, bundle_dir, f"{base}/iterations.jsonl")
        # Only for the interruptions it adds (resumed / superseding lines).
        vre.check_iterations(
            iterations.read_text(encoding="utf-8"), reg, facts.sha, label, vre.Findings(),
            facts, require_all=trained,
        )  # fmt: skip
    elif trained:
        raise BuildError(f"{label}: training finished, but {iterations} does not exist")

    # Checkpoints: not copied (too large to commit); their digests are recorded and
    # each arm must have played exactly the file its registered command names.
    digests: dict[str, str] = {}
    arms: dict[str, Any] = {}
    games: dict[str, list[dict[str, Any]]] = {}
    for name in reg.arms:
        invocation = name.replace("_", "-")
        args = reg.invocations[invocation]
        out = reg.flag(invocation, "--out")
        result = _local(out, card_dir, run.run_dir) / "result.json" if out else None
        if result is None or not result.is_file():
            if facts.completed:
                raise BuildError(f"{label}: arm {name}: {result} does not exist")
            continue
        envelope = _read_json(result)
        row = envelope["results"][0]
        for flag in CHECKPOINT_FLAGS:
            path = vre._flag(args, flag)
            if path is None:
                continue
            local = _local(path, card_dir, run.run_dir)
            rel = (
                local.relative_to(run.run_dir).as_posix()
                if local.is_relative_to(run.run_dir)
                else path
            )
            if rel not in digests and local.is_file():
                digests[rel] = checkpoint_opponent_id(local).split(":", 1)[1]
            played = envelope["candidate"]["id"] if flag == "--checkpoint" else row["opponent_id"]
            if rel in digests and played != f"checkpoint:{digests[rel]}":
                raise BuildError(f"{label}: arm {name} did not play {rel} as it is on disk")
        written = _copy(result, bundle_dir, f"{base}/arena/{name}/result.json")
        arms[name] = {**written, **vre.arm_summary(row, vre.ARM_CONFIDENCE.get(name, 0.95))}
        games[name] = list(row["games"])
    entry["arms"] = arms
    entry["checkpoints"] = dict(sorted(digests.items()))
    judged = tuple(games[n] for n in JUDGED) if all(n in games for n in JUDGED) else None
    attempt = rr.Attempt(
        run_id,
        run.role,
        facts.sha,
        record,
        entry["run_json"]["sha256"],
        facts.completed,
        tuple(facts.interruptions),
        # A symmetry or positive-control VOID: the judged arms alone make it VOID.
        control_void=judged is not None and compute_verdict(*judged).verdict == vre.VOID,
        devices_registered=reg.devices is not None,
    )
    return Collected(entry, attempt, judged, list(facts.interruptions))


def attach_records(
    collected: Sequence[Collected], compare: Sequence[Path], bundle_dir: Path
) -> dict[str, list[str]]:
    """Copy each ``--compare`` record beside the attempts it names and reference it.

    Returns ``{re-run's run_id: why it is a non-reproducing re-run}``. The record
    names both runs (``attempt.card_id``, ``rerun.card_id``); it must be the
    comparison of a listed interrupted attempt with that attempt's re-run. An
    interrupted attempt left without a record is not refused here: the verifier
    fails the staged bundle for it.

    Raises:
        BuildError: a record that names no runs, names runs that are not an
            interrupted attempt and its re-run among ``--run``, or repeats one.
    """
    from scripts import release_reproduction as rr

    attempts = [c.attempt for c in collected]
    non_reproducing: dict[str, list[str]] = {}
    for source in compare:
        record = _read_json(source)
        ids = rr.record_run_ids(record)
        if ids is None:
            raise BuildError(
                f"--compare {source}: the record names no runs; write it with `python -m "
                "alpha_zero.evaluation.reproduction compare <attempt dir> <re-run dir> --out FILE` "
                "on two run_card run directories"
            )
        void = next((c for c in collected if c.attempt.run_id == ids[0]), None)
        rerun = rr.record_rerun(void.attempt, attempts) if void is not None else None
        if void is None or rerun is None or rerun.run_id != ids[1]:
            raise BuildError(
                f"--compare {source}: it compares {ids[0]!r} with {ids[1]!r}, which are not an "
                "interrupted attempt and its re-run among the --run directories"
            )
        if "reproduction" in void.entry:
            raise BuildError(f"--compare {source}: a second record for {ids[0]!r}")
        result = rr.outcome(record, void.attempt, rerun)
        written = _copy(source, bundle_dir, rr.record_name(ids[0]))
        void.entry["reproduction"] = rr.bundle_reference(written, rerun, result)
        if result.kind == rr.OUTCOME_NON_REPRODUCING:
            non_reproducing.setdefault(rerun.run_id, []).extend(
                rr.verdict_reasons(void.attempt, result)
            )
    return non_reproducing


def set_verdicts(collected: Sequence[Collected], non_reproducing: dict[str, list[str]]) -> None:
    """Store each attempt's verdict, the re-run rule applied (section 7.1, "After a VOID")."""
    from alpha_zero.evaluation.a1_protocol import compute_verdict

    for item in collected:
        if item.judged is None:  # the judged arms never all ran: VOID by status
            item.entry["verdict"] = {"verdict": vre.VOID, "reasons": list(item.interruptions)}
            continue
        item.entry["verdict"] = compute_verdict(
            *item.judged,
            interruptions=item.interruptions,
            non_reproducing=non_reproducing.get(item.entry["run_id"], ()),
        ).to_dict()


def judged_verdicts(runs: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Role -> the verdict of its one non-VOID attempt (roles without one are absent)."""
    out: dict[str, str] = {}
    for role in vre.ROLES:
        valid = [r["verdict"]["verdict"] for r in runs if r["role"] == role]
        valid = [v for v in valid if v != vre.VOID]
        if len(valid) == 1:
            out[role] = valid[0]
    return out


def verdict_text(run: dict[str, Any]) -> str:
    """A run's verdict as the report and the index show it: with its label, when it has one."""
    verdict = run["verdict"]
    label = verdict.get("label")
    return f"{verdict['verdict']} ({label})" if label else str(verdict["verdict"])


def reproduction_lines(run: dict[str, Any]) -> list[str]:
    """What the report says of a VOID attempt's reproduction check (section 7.1)."""
    from scripts import release_reproduction as rr

    entry = run.get("reproduction")
    if not entry:
        return []
    said = {
        rr.OUTCOME_REPRODUCED: "REPRODUCED: everything this attempt finished is in the re-run "
        "and equal, so the re-run is judged.",
        rr.OUTCOME_EMPTY: "the comparison was empty (NOTHING TO COMPARE): this attempt finished "
        "nothing beyond `checkpoint_0`, which the seed alone fixes, so nothing of the result "
        "had been seen and the re-run is judged.",
        rr.OUTCOME_NON_REPRODUCING: "the re-run does not reproduce this attempt. It is reported "
        "as a non-reproducing re-run and cannot PASS (a PASS row is read as INCONCLUSIVE).",
    }
    lines = [
        f"- Reproduction check against `{entry['rerun']}` (`{entry['path']}`): "
        f"{said.get(entry['outcome'], entry['outcome'])}"
    ]
    return lines + [f"  - {reason}" for reason in entry.get("reasons") or []]


def render_report(bundle: dict[str, Any]) -> str:
    """``report.md``: the verdicts and arms, worded so it claims nothing it may not."""
    lines = [
        f"# Release v{bundle['release']} evidence",
        "",
        f"Release candidate `{bundle['rc_tag']}` (`{bundle['rc_sha'][:12]}`), judged under "
        f"`docs/EVALUATION.md` section 7.1 (sha256 `{bundle['protocol_sha256'][:16]}...`).",
        "Built by `scripts/build_release_bundle.py`; checked by "
        "`scripts/verify_release_evidence.py`, which recomputes every number below.",
        "",
        "| Run | Role | Seed | Commit | Games | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for run in bundle["runs"]:
        lines.append(
            f"| `{run['run_id']}` | {run['role']} | {run['seed']} | `{run['git_sha'][:12]}` | "
            f"{run['games']} | {verdict_text(run)} |"
        )
    for run in bundle["runs"]:
        lines += ["", f"## `{run['run_id']}` ({run['role']}): {verdict_text(run)}", ""]
        lines += [f"- {reason}" for reason in run["verdict"].get("reasons") or []]
        lines += reproduction_lines(run)
        lines += ["", "| Arm | W/D/L | Score | Hull |", "|---|---|---|---|"]
        for name, arm in run["arms"].items():
            low, high = arm["hull"]
            lines.append(
                f"| {name} | {arm['wins']}/{arm['draws']}/{arm['losses']} | "
                f"{arm['score_rate']:.4f} | {arm['confidence']:.0%} [{low:.4f}, {high:.4f}] |"
            )
    claim = (
        "The judged A1 and A1-R both PASS: `learning_claimed` is true."
        if bundle["learning_claimed"]
        else "`learning_claimed` is false: the verdicts are recorded and the claim is withheld."
    )
    lines += ["", "## Claim", "", claim, ""]
    return "\n".join(lines)


def index_row(bundle: dict[str, Any], date: str) -> str:
    """The ``docs/evidence/README.md`` row for the bundle."""
    name = vre.bundle_dir_name(str(bundle["release"]))
    verdicts = ", ".join(f"{r['role']} `{r['run_id']}` {verdict_text(r)}" for r in bundle["runs"])
    seeds = ", ".join(str(s) for s in dict.fromkeys(r["seed"] for r in bundle["runs"]))
    return (
        f"| [`{name}/`]({name}/report.md) | {date} (`{bundle['rc_sha'][:7]}`) | post-fix | "
        f"{seeds} | release v{bundle['release']} evidence under section 7.1: {verdicts}; "
        f"{bundle['games_total']:,} games | current |"
    )


def add_index_row(index: Path, row: str, name: str) -> str | None:
    """Insert ``row`` after the index table's last row; return the old text (None: unchanged)."""
    text = index.read_text(encoding="utf-8")
    if f"]({name}/" in text:
        return None
    lines = text.split("\n")
    last = max(i for i, line in enumerate(lines) if line.startswith("| [`"))
    lines.insert(last + 1, row)
    index.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return text


def build(
    *,
    release: str,
    rc_tag: str,
    runs: Sequence[RunInput],
    repo_root: Path,
    compare: Sequence[Path] = (),
    out: Path | None = None,
    date: str | None = None,
    command: str = "python scripts/build_release_bundle.py",
) -> tuple[Path, vre.Findings]:
    """Stage, verify and publish the bundle. Returns ``(bundle dir, findings)``.

    The findings are the verifier's on the staged bundle (check runs excluded);
    the bundle is published only when they hold no problem and nothing missing.

    Raises:
        BuildError: the bundle directory exists, or an input cannot be read.
    """
    from alpha_zero.evaluation.a1_protocol import protocol_sha256
    from alpha_zero.evaluation.benchmark_evidence import collect_evidence
    from scripts.check_branch_direction import Git

    repo_root = repo_root.resolve()
    name = vre.bundle_dir_name(release)
    out = out or repo_root / "docs" / "evidence" / name
    if out.name != name:
        raise BuildError(f"the bundle directory must be named {name}, not {out.name}")
    if out.exists():
        raise BuildError(f"{out} exists; a bundle is never overwritten (remove it by hand)")
    git = Git(repo_root)
    if not git.exists(f"refs/tags/{rc_tag}"):
        raise BuildError(f"tag {rc_tag} is not in {repo_root} (fetch tags)")
    rc_sha = git.resolve(f"refs/tags/{rc_tag}")

    out.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".staging-", dir=out.parent))
    staged = staging_root / name
    index = repo_root / vre.EVIDENCE_INDEX
    old_index: str | None = None
    try:
        collected = [collect_run(run, staged, git, rc_sha) for run in runs]
        set_verdicts(collected, attach_records(collected, compare, staged))
        entries = [item.entry for item in collected]
        verdicts = judged_verdicts(entries)
        claimed = all(verdicts.get(role) == "PASS" for role in vre.CLAIM_ROLES)
        total = sum(int(e["games"] or 0) for e in entries)
        a1_seeds = [e["seed"] for e in entries if e["role"] == "A1"]
        evidence = collect_evidence(
            a1_seeds[0] if a1_seeds else None,
            total,
            command,
            claims=["the A1 (and, when listed, A1-R) verdict under docs/EVALUATION.md section 7.1"],
            non_claims=[
                "strength under any other protocol, board size or opponent",
                "any learning claim unless learning_claimed is true",
            ],
        )
        bundle = {
            "bundle_schema_version": vre.BUNDLE_SCHEMA_VERSION,
            "release": release,
            "rc_tag": rc_tag,
            "rc_sha": rc_sha,
            "protocol_sha256": protocol_sha256(
                (repo_root / vre.EVALUATION_MD).read_text(encoding="utf-8")
            ),
            "runs": entries,
            "games_total": total,
            "learning_recorded": True,
            "learning_claimed": claimed,
            "evidence": evidence,
        }
        (staged / vre.BUNDLE_FILE).write_text(
            json.dumps(bundle, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        (staged / vre.REPORT_FILE).write_text(render_report(bundle), encoding="utf-8", newline="\n")
        if index.is_file():
            row = index_row(bundle, date or datetime.now(UTC).date().isoformat())
            old_index = add_index_row(index, row, name)
        findings = vre.verify(
            staged / vre.BUNDLE_FILE, repo_root=repo_root, git=git, api=None, repo=""
        )
        # The rc's check runs need the GitHub API; the verifier reads them afterwards.
        findings.missing = [m for m in findings.missing if not m.startswith("check runs:")]
        if findings.problems or findings.missing:
            if old_index is not None:
                index.write_text(old_index, encoding="utf-8", newline="\n")
            return out, findings
        staged.rename(out)
        return out, findings
    except BaseException:
        if old_index is not None:
            index.write_text(old_index, encoding="utf-8", newline="\n")
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", required=True, help="X.Y.Z (pyproject.toml's version)")
    parser.add_argument("--rc", required=True, help="the highest rc tag, e.g. v2.1.0-rc.1")
    parser.add_argument(
        "--run",
        type=parse_run,
        action="append",
        required=True,
        help="ROLE=DIR, one per attempt (VOID attempts too); ROLE is A1 or A1-R",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        action="append",
        default=[],
        metavar="FILE",
        help="a `reproduction compare --out` record: one per interrupted attempt that has a "
        "re-run (section 7.1, 'After a VOID'); matched to the attempts by card_id",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=None, help="default: docs/evidence/<name>")
    parser.add_argument("--date", default=None, help="the index row's date (default: today, UTC)")
    args = parser.parse_args(argv)
    command = shlex.join(["python", "scripts/build_release_bundle.py", *(argv or sys.argv[1:])])
    try:
        out, findings = build(
            release=args.release,
            rc_tag=args.rc,
            runs=args.run,
            repo_root=args.repo_root,
            compare=args.compare,
            out=args.out,
            date=args.date,
            command=command,
        )
    except BuildError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    for note in findings.notes:
        print(f"build_release_bundle: {note}")
    if findings.problems or findings.missing:
        for item in [*findings.missing, *findings.problems]:
            print(f"::error::build_release_bundle: {item}", file=sys.stderr)
        print("build_release_bundle: the staged bundle failed verification; nothing written")
        return EXIT_FAILED
    print(f"build_release_bundle: wrote {out}")
    print(
        "Next: python scripts/verify_release_evidence.py "
        f"{(out / vre.BUNDLE_FILE).as_posix()} --repo OWNER/NAME  (adds the rc's check runs)"
    )
    return EXIT_BUILT


if __name__ == "__main__":
    sys.exit(main())
