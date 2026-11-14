"""Does a seeded re-run reproduce the attempt it replaces? (A1 "After a VOID")

``docs/EVALUATION.md`` section 7.1 lets an interrupted confirmatory run be run
again at the same seed under a new ``card_id``, on the grounds that deterministic
algorithms make the re-run replay the interrupted attempt bit for bit. That is a
claim about two directories on disk, and this module is what checks it: it
reduces a run-card run directory to a *digest* and compares the digest of the
interrupted attempt (the **reference**) with the digest of its re-run (the
**candidate**).

What a digest holds (:func:`run_digest`), for the layout the run cards write
(``configs/run_cards/a1.yaml``):

* ``<run_dir>/ckpt/checkpoint_<N>.pt`` -- one sha256 per ``state_dict`` tensor
  (:func:`checkpoint_digest`). Tensors are hashed, not the file: a checkpoint's
  bytes also carry its metadata, the saving device and the file's own stem (torch
  names the archive root after it), none of which is the trained network.
* ``<run_dir>/arena/<arm>/result.json`` -- the protocol, the seed and every
  opponent row's per-game records. The statistics beside them are recomputed
  from those records, and the timestamps and host fields are not results.
* ``<run_dir>/ckpt/iterations.jsonl`` -- every field of every iteration record
  except the ones in :data:`ITERATION_RECORD_EXCLUDED_FIELDS`.

The comparison is **one-directional** (:func:`compare_digests`): everything the
reference holds must be in the candidate and equal. What only the candidate holds
is ignored, because a re-run finishes more than the attempt it reproduces. A
caller checking two complete runs against each other compares in both directions
(``scripts/check_determinism.py`` does).

Three rules keep a pass from being vacuous:

* a component that exists but cannot be read (a checkpoint torn by the
  interruption, a ``result.json`` that does not parse) is reported as a
  difference, never skipped: an unverifiable component has not been shown to
  reproduce;
* a checkpoint with no tensors is unreadable, not "equal to anything";
* :func:`compared_components` says how many components the reference offered, and
  the CLI exits :data:`EXIT_NOTHING_TO_COMPARE` -- not 0 -- when that is none.

CLI::

    python -m alpha_zero.evaluation.reproduction compare <attempt_dir> <rerun_dir> [--out r.json]

prints ``REPRODUCED`` (exit 0), ``DIFFERS`` and the differences (exit 1) or
``NOTHING TO COMPARE`` (exit 2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from alpha_zero.training.run_record import ITERATIONS_FILE, read_iteration_records
from alpha_zero.utils.checkpoint import extract_state_dict, safe_load_checkpoint

logger = logging.getLogger(__name__)

__all__ = [
    "ARENA_DIR_NAME",
    "ARENA_RESULT_FILE",
    "CHECKPOINT_DIR_NAME",
    "CHECKPOINT_FILE_RE",
    "COMPONENT_ARENAS",
    "COMPONENT_CHECKPOINTS",
    "COMPONENT_ITERATIONS",
    "COMPONENT_KINDS",
    "EXIT_DIFFERS",
    "EXIT_NOTHING_TO_COMPARE",
    "EXIT_REPRODUCED",
    "ITERATION_RECORD_EXCLUDED_FIELDS",
    "STATUS_DIFFERS",
    "STATUS_NOTHING_TO_COMPARE",
    "STATUS_REPRODUCED",
    "checkpoint_digest",
    "compare_digests",
    "compared_components",
    "component_counts",
    "main",
    "run_digest",
    "tensor_digest",
]

# ---------------------------------------------------------------------------
# Layout of a run-card run directory (configs/run_cards/a1.yaml)
# ---------------------------------------------------------------------------

#: ``--checkpoint_dir`` of the registered training command, relative to the run directory.
CHECKPOINT_DIR_NAME = "ckpt"
#: Parent of every arena arm's ``--out``, relative to the run directory.
ARENA_DIR_NAME = "arena"
#: The envelope ``alpha_zero.evaluation.benchmark`` writes into an arm's ``--out``.
ARENA_RESULT_FILE = "result.json"
#: Numbered checkpoints, as ``training_civ6`` names them. ``latest.pt`` is left out:
#: it is a second copy of the last numbered checkpoint, written only by a run that
#: finished, so an interrupted attempt never has one.
CHECKPOINT_FILE_RE = re.compile(r"^checkpoint_(\d+)\.pt$")

# ---------------------------------------------------------------------------
# Digest shape
# ---------------------------------------------------------------------------

#: Bumped when the digest's shape changes; digests of different versions never compare.
DIGEST_SCHEMA_VERSION = 1
SCHEMA_KEY = "digest_schema_version"

COMPONENT_CHECKPOINTS = "checkpoints"
COMPONENT_ARENAS = "arenas"
COMPONENT_ITERATIONS = "iterations"
#: The comparable component kinds, in the order differences are reported.
COMPONENT_KINDS = (COMPONENT_CHECKPOINTS, COMPONENT_ARENAS, COMPONENT_ITERATIONS)
#: ``{"<kind>/<name>": reason}`` for components that exist but could not be read.
UNREADABLE = "unreadable"

#: Iteration-record fields that are not compared, as key paths into one
#: ``iterations.jsonl`` line. These are the wall-clock fields
#: ``training_civ6`` writes, plus the one identity label built from them:
#:
#: * ``started_at`` -- UTC time the training process started (``_open_iteration_log``);
#: * ``self_play.seconds`` -- wall-clock duration of the self-play phase (``train``);
#: * ``run_id`` -- ``CIV6AZ_RUN_ID`` (the card id, which section 7.1 *requires* a
#:   re-run to change) or, without one, ``train-<seed>-<UTC stamp>``.
#:
#: Everything else is compared, including ``git_sha``, ``seed``, ``resumed_from``,
#: ``start_iteration`` and ``supersedes_earlier_record``.
ITERATION_RECORD_EXCLUDED_FIELDS: tuple[tuple[str, ...], ...] = (
    ("started_at",),
    ("self_play", "seconds"),
    ("run_id",),
)

#: Opponent ids of this form are ``checkpoint:<sha256 of the file>``. The file hash
#: is not compared (see the module docstring); the tensors behind it are.
CHECKPOINT_OPPONENT_PREFIX = "checkpoint:"
CHECKPOINT_OPPONENT_KIND = "checkpoint"

#: At most this many tensors, fields or names are spelled out in one difference.
MAX_NAMED_DIFFERENCES = 5

_FIELD_SEPARATOR = b"\x00"
_HASH_CHUNK_BYTES = 1 << 20

STATUS_REPRODUCED = "REPRODUCED"
STATUS_DIFFERS = "DIFFERS"
STATUS_NOTHING_TO_COMPARE = "NOTHING TO COMPARE"
EXIT_REPRODUCED = 0
EXIT_DIFFERS = 1
#: Also the exit for a run directory that does not exist: never a pass.
EXIT_NOTHING_TO_COMPARE = 2


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def tensor_digest(tensor: torch.Tensor) -> str:
    """sha256 over a tensor's dtype, shape and raw bytes (moved to CPU, contiguous).

    The bytes are the machine's native byte order: digests compare runs on one
    machine, which is the only reproduction section 7.1 claims.
    """
    data = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(data.dtype).encode("utf-8"))
    digest.update(_FIELD_SEPARATOR)
    digest.update(",".join(str(size) for size in data.shape).encode("utf-8"))
    digest.update(_FIELD_SEPARATOR)
    # A byte view, so every dense dtype (bool, bfloat16, ...) hashes the same way.
    raw = data.reshape(-1).view(torch.uint8)
    for start in range(0, raw.numel(), _HASH_CHUNK_BYTES):
        digest.update(raw[start : start + _HASH_CHUNK_BYTES].cpu().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_digest(path: str | os.PathLike[str]) -> dict[str, str]:
    """``{state_dict key: sha256}`` for every tensor of a training checkpoint.

    The file is loaded with ``weights_only=True`` semantics onto the CPU and is
    never loaded unsafely, whatever the environment says: a comparison tool has
    no business executing code from the files it compares.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        UnsafeCheckpointError: the safe loader rejected the file (which is also
            how a torn file surfaces).
        KeyError: the file holds no recognisable ``state_dict``.
        ValueError: the ``state_dict`` is empty or holds a non-tensor entry.
    """
    source = Path(path)
    logger.debug("digesting checkpoint %s", source)
    loaded = safe_load_checkpoint(source, map_location="cpu", strict_security=True)
    if not isinstance(loaded.data, dict):
        raise ValueError(
            f"{source}: expected a checkpoint mapping, got {type(loaded.data).__name__}"
        )
    state = extract_state_dict(loaded.data)
    digests: dict[str, str] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"{source}: state_dict entry {key!r} is {type(value).__name__}, not a tensor"
            )
        digests[str(key)] = tensor_digest(value)
    if not digests:
        # An empty map would compare equal to any other empty map.
        raise ValueError(f"{source}: state_dict holds no tensors")
    return digests


def _checkpoint_number(name: str) -> int:
    match = CHECKPOINT_FILE_RE.match(name)
    return int(match.group(1)) if match else -1


def _checkpoint_digests(directory: Path, unreadable: dict[str, str]) -> dict[str, dict[str, str]]:
    if not directory.is_dir():
        return {}
    names = sorted(
        (p.name for p in directory.iterdir() if p.is_file() and CHECKPOINT_FILE_RE.match(p.name)),
        key=_checkpoint_number,
    )
    digests: dict[str, dict[str, str]] = {}
    for name in names:
        try:
            digests[name] = checkpoint_digest(directory / name)
        except Exception as exc:  # reported as a difference by compare_digests
            unreadable[f"{COMPONENT_CHECKPOINTS}/{name}"] = _reason(exc)
            logger.warning("checkpoint %s is unreadable: %s", directory / name, _reason(exc))
    return digests


# ---------------------------------------------------------------------------
# Arena arms
# ---------------------------------------------------------------------------


def _opponent_label(opponent_id: object) -> str:
    """An opponent row's comparable name: a baseline's id, or just ``checkpoint``."""
    text = str(opponent_id)
    return CHECKPOINT_OPPONENT_KIND if text.startswith(CHECKPOINT_OPPONENT_PREFIX) else text


def _arena_digest(result_file: Path) -> dict[str, Any]:
    """The comparable part of one ``result.json``. Raises ``ValueError`` if it has none."""
    envelope = json.loads(result_file.read_text(encoding="utf-8"))
    if not isinstance(envelope, dict):
        raise ValueError("the envelope is not a JSON object")
    results = envelope.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("the envelope has no 'results' rows")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(results):
        games = row.get("games") if isinstance(row, dict) else None
        if not isinstance(games, list) or not games:
            raise ValueError(f"results[{index}] has no per-game records ('games')")
        rows.append({"opponent": _opponent_label(row.get("opponent_id")), "games": games})
    candidate = envelope.get("candidate")
    evidence = envelope.get("evidence")
    return {
        "protocol": envelope.get("protocol"),
        "seed": evidence.get("seed") if isinstance(evidence, dict) else None,
        "candidate_kind": candidate.get("kind") if isinstance(candidate, dict) else None,
        "rows": rows,
    }


def _arena_digests(directory: Path, unreadable: dict[str, str]) -> dict[str, dict[str, Any]]:
    if not directory.is_dir():
        return {}
    digests: dict[str, dict[str, Any]] = {}
    for arm in sorted(p for p in directory.iterdir() if p.is_dir()):
        result_file = arm / ARENA_RESULT_FILE
        if not result_file.is_file():
            # The benchmark writes result.json last: no file means the arm never finished.
            logger.debug("arena arm %s has no %s; absent from the digest", arm, ARENA_RESULT_FILE)
            continue
        logger.debug("digesting arena arm %s", arm)
        try:
            digests[arm.name] = _arena_digest(result_file)
        except Exception as exc:  # reported as a difference by compare_digests
            unreadable[f"{COMPONENT_ARENAS}/{arm.name}"] = _reason(exc)
            logger.warning("arena result %s is unreadable: %s", result_file, _reason(exc))
    return digests


# ---------------------------------------------------------------------------
# Iteration records
# ---------------------------------------------------------------------------


def _without_excluded(record: Mapping[str, Any]) -> dict[str, Any]:
    """A deep copy of ``record`` with :data:`ITERATION_RECORD_EXCLUDED_FIELDS` removed."""
    cleaned: dict[str, Any] = json.loads(json.dumps(record))
    for path in ITERATION_RECORD_EXCLUDED_FIELDS:
        holder: Any = cleaned
        for key in path[:-1]:
            holder = holder.get(key) if isinstance(holder, dict) else None
        if isinstance(holder, dict):
            holder.pop(path[-1], None)
    return cleaned


def _iteration_digests(record_file: Path, unreadable: dict[str, str]) -> dict[str, dict[str, Any]]:
    if not record_file.is_file():
        return {}
    logger.debug("digesting iteration records %s", record_file)
    try:
        records = read_iteration_records(record_file)
    except Exception as exc:  # reported as a difference by compare_digests
        unreadable[COMPONENT_ITERATIONS] = _reason(exc)
        logger.warning("%s is unreadable: %s", record_file, _reason(exc))
        return {}
    digests: dict[str, dict[str, Any]] = {}
    for record in records:
        # Keyed by iteration; a later line for the same iteration supersedes the
        # earlier one (run_record: "readers wanting one line per iteration take the last").
        digests[str(record.get("iteration"))] = _without_excluded(record)
    return digests


# ---------------------------------------------------------------------------
# Run digest
# ---------------------------------------------------------------------------


def _reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def run_digest(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """The digest of a run-card run directory (see the module docstring).

    Each of ``checkpoints``, ``arenas`` and ``iterations`` maps a component's name
    (a checkpoint file name, an arm's directory name, an iteration number) to its
    digest; a component the run never produced is simply absent. ``unreadable``
    names the components that exist but could not be read. The digest is plain
    JSON data.

    Raises:
        NotADirectoryError: ``run_dir`` is not a directory.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")
    logger.info("digesting run directory %s", root)
    unreadable: dict[str, str] = {}
    checkpoint_dir = root / CHECKPOINT_DIR_NAME
    digest: dict[str, Any] = {
        SCHEMA_KEY: DIGEST_SCHEMA_VERSION,
        COMPONENT_CHECKPOINTS: _checkpoint_digests(checkpoint_dir, unreadable),
        COMPONENT_ARENAS: _arena_digests(root / ARENA_DIR_NAME, unreadable),
        COMPONENT_ITERATIONS: _iteration_digests(checkpoint_dir / ITERATIONS_FILE, unreadable),
        UNREADABLE: unreadable,
    }
    logger.info("%s: %s, %d unreadable", root, component_counts(digest), len(unreadable))
    return digest


def _components(digest: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    components = digest.get(kind)
    return components if isinstance(components, Mapping) else {}


def component_counts(digest: Mapping[str, Any]) -> dict[str, int]:
    """How many readable components of each kind ``digest`` holds."""
    return {kind: len(_components(digest, kind)) for kind in COMPONENT_KINDS}


def compared_components(reference: Mapping[str, Any]) -> int:
    """How many components :func:`compare_digests` checks for this ``reference``.

    Zero means there is nothing to compare: no difference will be found, and that
    is not a reproduction.
    """
    return sum(component_counts(reference).values())


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> str:
    """JSON text that is equal exactly when two values are the same JSON value.

    Unlike ``==`` it tells ``1`` from ``1.0`` and ``true`` from ``1``, and it calls
    two NaNs equal -- a NaN loss that repeats is a reproduction.
    """
    return json.dumps(value, sort_keys=True)


def _flatten(value: Any, prefix: str = "") -> dict[str, str]:
    """``{dotted path: canonical leaf}``; empty containers are leaves."""
    if isinstance(value, Mapping) and value:
        flat: dict[str, str] = {}
        for key in sorted(value, key=str):
            flat.update(_flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
        return flat
    if isinstance(value, list) and value:
        flat = {}
        for index, item in enumerate(value):
            flat.update(_flatten(item, f"{prefix}[{index}]"))
        return flat
    return {prefix: _canonical(value)}


def _named(names: Sequence[str]) -> str:
    """The first few names, and how many more there are."""
    shown = ", ".join(names[:MAX_NAMED_DIFFERENCES])
    extra = len(names) - MAX_NAMED_DIFFERENCES
    return f"{shown} (+{extra} more)" if extra > 0 else shown


def _field_differences(reference: Any, candidate: Any) -> list[str]:
    """Per-path differences between two JSON values, in path order."""
    ref, cand = _flatten(reference), _flatten(candidate)
    found: list[str] = []
    for path in sorted(ref.keys() | cand.keys()):
        label = path or "value"
        if path not in cand:
            found.append(f"{label} is only in the reference ({ref[path]})")
        elif path not in ref:
            found.append(f"{label} is only in the candidate ({cand[path]})")
        elif ref[path] != cand[path]:
            found.append(f"{label}: reference {ref[path]}, candidate {cand[path]}")
    return found


def _absent(kind: str, name: str, candidate: Mapping[str, Any]) -> str:
    """Why a reference component has no counterpart in the candidate."""
    reason = _components(candidate, UNREADABLE).get(f"{kind}/{name}")
    if reason is None and kind == COMPONENT_ITERATIONS:
        reason = _components(candidate, UNREADABLE).get(COMPONENT_ITERATIONS)
    return (
        "missing from the candidate"
        if reason is None
        else f"unreadable in the candidate ({reason})"
    )


def _compare_checkpoints(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    differences: list[str] = []
    theirs = _components(candidate, COMPONENT_CHECKPOINTS)
    for name, tensors in _components(reference, COMPONENT_CHECKPOINTS).items():
        logger.debug("comparing checkpoint %s", name)
        if name not in theirs:
            differences.append(
                f"checkpoint {name}: {_absent(COMPONENT_CHECKPOINTS, name, candidate)}"
            )
            continue
        other = theirs[name]
        only_reference = [key for key in tensors if key not in other]
        only_candidate = [key for key in other if key not in tensors]
        changed = [key for key in tensors if key in other and tensors[key] != other[key]]
        if only_reference:
            differences.append(
                f"checkpoint {name}: {len(only_reference)} tensor(s) only in the reference: "
                f"{_named(only_reference)}"
            )
        if only_candidate:
            differences.append(
                f"checkpoint {name}: {len(only_candidate)} tensor(s) only in the candidate: "
                f"{_named(only_candidate)}"
            )
        if changed:
            differences.append(
                f"checkpoint {name}: {len(changed)} of {len(tensors)} tensors differ: "
                f"{_named(changed)}"
            )
    return differences


def _compare_arena(name: str, mine: Mapping[str, Any], other: Mapping[str, Any]) -> list[str]:
    differences: list[str] = []
    for key in ("protocol", "seed", "candidate_kind"):
        fields = _field_differences(mine.get(key), other.get(key))
        if fields:
            differences.append(f"arena {name}: {key} differs: {_named(fields)}")
    my_rows, other_rows = list(mine.get("rows") or []), list(other.get("rows") or [])
    if len(my_rows) != len(other_rows):
        differences.append(
            f"arena {name}: the reference has {len(my_rows)} opponent row(s), "
            f"the candidate {len(other_rows)}"
        )
    for index, (my_row, other_row) in enumerate(zip(my_rows, other_rows, strict=False)):
        where = f"arena {name}, opponent row {index} ({my_row.get('opponent')})"
        if my_row.get("opponent") != other_row.get("opponent"):
            differences.append(f"{where}: the candidate's opponent is {other_row.get('opponent')}")
        my_games, other_games = list(my_row.get("games") or []), list(other_row.get("games") or [])
        if len(my_games) != len(other_games):
            differences.append(
                f"{where}: the reference has {len(my_games)} games, the candidate "
                f"{len(other_games)}"
            )
        changed = [
            game
            for game, (a, b) in enumerate(zip(my_games, other_games, strict=False))
            if _canonical(a) != _canonical(b)
        ]
        if changed:
            first = changed[0]
            differences.append(
                f"{where}: {len(changed)} of {len(my_games)} games differ; the first is game "
                f"index {first}: reference {_canonical(my_games[first])}, "
                f"candidate {_canonical(other_games[first])}"
            )
    return differences


def _compare_arenas(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    differences: list[str] = []
    theirs = _components(candidate, COMPONENT_ARENAS)
    for name, mine in _components(reference, COMPONENT_ARENAS).items():
        logger.debug("comparing arena arm %s", name)
        if name not in theirs:
            differences.append(f"arena {name}: {_absent(COMPONENT_ARENAS, name, candidate)}")
            continue
        differences.extend(_compare_arena(name, mine, theirs[name]))
    return differences


def _compare_iterations(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    differences: list[str] = []
    theirs = _components(candidate, COMPONENT_ITERATIONS)
    for name, record in _components(reference, COMPONENT_ITERATIONS).items():
        logger.debug("comparing iteration record %s", name)
        if name not in theirs:
            differences.append(
                f"iteration record {name}: {_absent(COMPONENT_ITERATIONS, name, candidate)}"
            )
            continue
        fields = _field_differences(record, theirs[name])
        if fields:
            differences.append(
                f"iteration record {name}: {len(fields)} field(s) differ: {_named(fields)}"
            )
    return differences


def compare_digests(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    """Every way ``candidate`` fails to reproduce ``reference``; empty when it does.

    ``reference`` is the interrupted attempt, ``candidate`` its re-run. For every
    component the reference holds, the candidate must hold an equal one: a missing
    or unreadable counterpart is a difference. Components only the candidate holds
    are ignored. A component of the *reference* that could not be read is a
    difference too -- it cannot be shown to reproduce.

    An empty result is a reproduction only if :func:`compared_components` of the
    reference is non-zero.
    """
    mine, theirs = reference.get(SCHEMA_KEY), candidate.get(SCHEMA_KEY)
    if mine != DIGEST_SCHEMA_VERSION or theirs != DIGEST_SCHEMA_VERSION:
        return [
            f"digest schema versions cannot be compared: reference {mine!r}, candidate "
            f"{theirs!r}, this code {DIGEST_SCHEMA_VERSION}"
        ]
    differences = [
        f"{name}: unreadable in the reference ({reason}); it cannot be shown to reproduce"
        for name, reason in _components(reference, UNREADABLE).items()
    ]
    differences.extend(_compare_checkpoints(reference, candidate))
    differences.extend(_compare_arenas(reference, candidate))
    differences.extend(_compare_iterations(reference, candidate))
    logger.info(
        "compared %d component(s): %d difference(s)",
        compared_components(reference),
        len(differences),
    )
    return differences


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m alpha_zero.evaluation.reproduction",
        description="Check that a re-run reproduces the attempt it replaces.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser(
        "compare",
        help="compare an interrupted attempt's run directory with its re-run's",
        description=(
            "Everything the attempt left behind (checkpoints, finished arena arms, "
            "iteration records) must be in the re-run and equal. Exit 0 REPRODUCED, "
            "1 DIFFERS, 2 NOTHING TO COMPARE."
        ),
    )
    compare.add_argument("attempt_dir", type=Path, help="the interrupted attempt (the reference)")
    compare.add_argument("rerun_dir", type=Path, help="the same-seed re-run (the candidate)")
    compare.add_argument("--out", type=Path, default=None, help="also write the outcome as JSON")
    return parser.parse_args(argv)


def _compare_command(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "tool": "alpha_zero.evaluation.reproduction compare",
        "attempt_dir": str(args.attempt_dir),
        "rerun_dir": str(args.rerun_dir),
    }
    try:
        reference, candidate = run_digest(args.attempt_dir), run_digest(args.rerun_dir)
    except NotADirectoryError as exc:
        status, exit_code = STATUS_NOTHING_TO_COMPARE, EXIT_NOTHING_TO_COMPARE
        payload.update({"error": str(exc), "components_compared": 0, "differences": []})
        print(f"ERROR: {exc}", file=sys.stderr)
    else:
        differences = compare_digests(reference, candidate)
        compared = compared_components(reference)
        if differences:
            status, exit_code = STATUS_DIFFERS, EXIT_DIFFERS
        elif compared == 0:
            status, exit_code = STATUS_NOTHING_TO_COMPARE, EXIT_NOTHING_TO_COMPARE
        else:
            status, exit_code = STATUS_REPRODUCED, EXIT_REPRODUCED
        payload.update(
            {
                "components_compared": compared,
                "components": component_counts(reference),
                "rerun_components": component_counts(candidate),
                "differences": differences,
            }
        )
    payload["status"] = status
    print(status)
    for difference in payload["differences"]:
        print(f"  {difference}")
    if status == STATUS_REPRODUCED:
        print(f"  {payload['components_compared']} component(s) compared: {payload['components']}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info("wrote %s", args.out)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit status."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    return _compare_command(args)


if __name__ == "__main__":
    sys.exit(main())
