#!/usr/bin/env python3
"""Check that a seeded A1-shaped run repeats bit for bit in a new process (determinism harness).

Section 7.1 of ``docs/EVALUATION.md`` registers the A1 runs on GPUs (training on
``cuda:0``, every arena arm on ``cuda:1``) and lets an interrupted run be run
again at the same seed, on the grounds that the re-run replays it exactly. A
re-run is a *new process*, in the pinned torch build, at A1's network and search
shape. This harness checks exactly that, at a size that takes minutes: it runs a
small version of the A1 pipeline twice, **each repeat in its own subprocess**,
and compares what the two left on disk with
:mod:`alpha_zero.evaluation.reproduction`, in both directions.

One repeat (the ``--worker`` mode) is section 7.1's own commands at this check's
sizes, built by :func:`training_argv` and :func:`arena_argv`:

* the registered training command (``--mcts gumbel --gumbel_k K --gumbel_sims N
  --bootstrap none --eval_games 0``, default network size) on ``--train-device``,
  checkpointing every iteration;
* the registered primary arm: the last checkpoint against ``checkpoint_0.pt``
  under the same search, on ``--arena-device``.

The children inherit this process's environment unchanged, so
``CUDA_VISIBLE_DEVICES``, ``CUDA_DEVICE_ORDER``, ``CUBLAS_WORKSPACE_CONFIG`` and
every ``CIV6AZ_*`` variable reach them as the operator (or ``run_card``) set them.
``PYTHONHASHSEED`` is deliberately not set: a real re-run does not set it either.

This is a *harness*. ``REPRODUCED`` says this machine, this torch build and this
device pair repeated this pipeline once across two processes; it is evidence for
section 7.1's reproduction claim, not a proof for every run. Anything else exits
non-zero: ``DIFFERS`` (with the differences), ``INCOMPLETE`` (nothing differed,
but a checkpoint or the arena was not compared, training never changed the
weights, or a CUDA repeat ran without deterministic algorithms enforced) and
``WORKER_FAILED`` (with the end of the worker's log).

Usage:
    python -m scripts.check_determinism --train-device cuda:0 --arena-device cuda:1 \
        --out determinism.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Run as a script, sys.path[0] is scripts/, so make this checkout win over any
# separately installed copy of the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha_zero.evaluation import reproduction
from alpha_zero.training.run_record import ITERATIONS_FILE, read_iteration_records
from alpha_zero.utils.device import cuda_ordinal, resolve_device_str
from alpha_zero.utils.game_budget import EXIT_CODE_REFUSED, add_budget_arguments, enforce_cli

logger = logging.getLogger("check_determinism")

REPO_ROOT = Path(__file__).resolve().parents[1]
#: ``python -m`` target of this file: how a worker is launched (and how run cards launch it).
MODULE = "scripts.check_determinism"
HARNESS = "check_determinism"
#: This harness's cost model in ``alpha_zero.utils.game_budget``.
RUNNER = "determinism_check"

#: The pipeline is run this many times, each in its own process.
REPEATS = 2
REPEAT_NAMES = ("run_a", "run_b")
#: The arena arm a worker plays; named after section 7.1's arm of the same shape.
ARENA_ARM = "primary"
#: A worker checkpoints every iteration, so ``checkpoint_0.pt`` and a trained one exist.
SAVE_INTERVAL = 1
#: A1 trains in every iteration because 24 games fill ``min_replay_size`` at once.
#: A handful of games would not, and a check that never trains says nothing about
#: training, so the worker lowers the threshold instead of raising the game count.
MIN_REPLAY_SIZE = 1
#: Not one of section 7.1's registered seeds, so the check is no A1-family run.
DEFAULT_SEED = 20261114
#: Device requests that name no device: ``resolve_device_str`` may pick any.
AUTO_DEVICE_NAMES = ("", "auto")
WORKER_REPORT_FILE = "worker.json"
#: The trainer's ``log_dir``, kept inside the repeat's run directory.
TRAINER_LOG_DIR_NAME = "log"
WORKER_LOG_TAIL_LINES = 40

STATUS_REPRODUCED = "REPRODUCED"
STATUS_DIFFERS = "DIFFERS"
STATUS_INCOMPLETE = "INCOMPLETE"
STATUS_WORKER_FAILED = "WORKER_FAILED"
EXIT_REPRODUCED = 0
EXIT_NOT_REPRODUCED = 1


class DeviceUnavailableError(RuntimeError):
    """An explicit device was requested and this machine cannot run it."""


# ---------------------------------------------------------------------------
# The pipeline: section 7.1's commands at this check's sizes
# ---------------------------------------------------------------------------


def training_argv(args: argparse.Namespace, checkpoint_dir: Path) -> list[str]:
    """Section 7.1's training command for ``alpha_zero.training_civ6`` at this check's sizes.

    Flag for flag the registered command; only the seed, the sizes, the device,
    ``--save_interval`` and the checkpoint directory differ
    (``tests/scripts/test_check_determinism.py`` holds the two together). ``args``
    here and below are :func:`resolved_arguments`: both devices are concrete.
    """
    return [
        *("--seed", str(args.seed)),
        *("--board_size", str(args.board_size)),
        *("--max_turns", str(args.max_turns)),
        *("--num_iterations", str(args.iterations)),
        *("--num_games", str(args.games)),
        *("--mcts", "gumbel"),
        *("--gumbel_k", str(args.k)),
        *("--gumbel_sims", str(args.n)),
        *("--device", str(args.train_device)),
        *("--bootstrap", "none"),
        *("--save_interval", str(SAVE_INTERVAL)),
        *("--eval_games", "0"),
        *("--checkpoint_dir", str(checkpoint_dir)),
    ]


def arena_argv(args: argparse.Namespace, run_dir: Path) -> list[str]:
    """Section 7.1's primary arm for ``alpha_zero.evaluation.benchmark`` at this check's sizes.

    The last checkpoint against ``checkpoint_0.pt``. An approved over-cap check
    passes its approval on, because the benchmark prices its own argv.
    """
    checkpoint_dir = run_dir / reproduction.CHECKPOINT_DIR_NAME
    argv = [
        "--opponents=",
        *("--seed", str(args.seed)),
        *("--board-size", str(args.board_size)),
        *("--max-turns", str(args.max_turns)),
        *("--sims", str(args.n)),
        *("--gumbel-k", str(args.k)),
        *("--mcts-type", "gumbel"),
        *("--device", str(args.arena_device)),
        *("--checkpoint", str(checkpoint_dir / f"checkpoint_{args.iterations}.pt")),
        *("--opponent-checkpoint", str(checkpoint_dir / "checkpoint_0.pt")),
        *("--n-games", str(args.arena_games)),
        *("--out", str(run_dir / reproduction.ARENA_DIR_NAME / ARENA_ARM)),
    ]
    if args.approval_ref:
        argv += ["--max_games", str(args.max_games), "--approval_ref", str(args.approval_ref)]
    return argv


def budget_inputs(args: argparse.Namespace) -> dict[str, int]:
    """Resolved ``determinism_check`` cost-model inputs for parsed ``args``.

    A worker is one repeat; the check is :data:`REPEATS` of them.
    """
    return {
        "repeats": 1 if args.worker is not None else REPEATS,
        "iterations": args.iterations,
        "games": args.games,
        "arena_games": args.arena_games,
    }


def environment_report(
    train_device: str, arena_device: str, *, probe_gpus: bool = True
) -> dict[str, Any]:
    """The torch build and devices a repeat ran on.

    ``probe_gpus=False`` asks nothing of CUDA, so the parent can describe the
    build without holding a CUDA context while its workers run.
    """
    import torch

    def gpu_name(device: str) -> str | None:
        ordinal = cuda_ordinal(device)
        if ordinal is None or not probe_gpus:
            return None
        return str(torch.cuda.get_device_name(ordinal))

    cudnn: int | None = None
    if probe_gpus:
        try:
            cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        except Exception:  # a broken cuDNN install must not hide the comparison
            cudnn = None
    return {
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cudnn": cudnn,
        "gpu_names": {"train": gpu_name(train_device), "arena": gpu_name(arena_device)},
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def run_worker(args: argparse.Namespace) -> int:
    """One repeat: train, then the arena, into ``args.worker`` (a new run directory)."""
    # Imported here: only a worker trains or plays, the parent just compares.
    from alpha_zero import training_civ6
    from alpha_zero.evaluation import benchmark

    run_dir: Path = args.worker
    checkpoint_dir = run_dir / reproduction.CHECKPOINT_DIR_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    train_args = training_civ6.build_parser().parse_args(training_argv(args, checkpoint_dir))
    config = dataclasses.replace(
        training_civ6.config_from_args(train_args, str(checkpoint_dir)),
        min_replay_size=MIN_REPLAY_SIZE,
        # The trainer's default is relative to the working directory (the checkout).
        log_dir=str(run_dir / TRAINER_LOG_DIR_NAME),
    )
    logger.info("worker %s: training on %s", run_dir.name, config.device)
    training_civ6.train(config)
    records = read_iteration_records(checkpoint_dir / ITERATIONS_FILE)
    untrained = [r.get("iteration") for r in records if r.get("train") is None]
    if len(records) != args.iterations or untrained:
        raise RuntimeError(
            f"expected {args.iterations} trained iteration(s); {len(records)} were recorded "
            f"and {untrained} did not train, so training determinism was not exercised"
        )

    logger.info("worker %s: arena on %s", run_dir.name, args.arena_device)
    status = int(benchmark.main(arena_argv(args, run_dir)))
    if status != 0:
        return status
    report = environment_report(str(args.train_device), str(args.arena_device))
    (run_dir / WORKER_REPORT_FILE).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


# ---------------------------------------------------------------------------
# The check: two workers, compared both ways
# ---------------------------------------------------------------------------


def resolve_run_device(requested: str, flag: str) -> str:
    """The device a phase will run on; an explicit device this machine lacks is refused.

    ``resolve_device_str`` falls back to CPU with a warning. A determinism result
    labelled with a GPU it never ran on is worse than no result, so anything but
    ``auto`` must resolve to itself -- which also refuses a malformed ``cuda:x``,
    which names no ordinal and would otherwise fall back unnoticed.
    """
    asked = str(requested).strip().lower()
    try:
        resolved = resolve_device_str(requested)
    except ValueError as exc:
        raise DeviceUnavailableError(f"{flag} {requested}: {exc}") from exc
    if asked not in AUTO_DEVICE_NAMES and resolved != asked:
        raise DeviceUnavailableError(
            f"{flag} {requested} was requested but {resolved!r} is what this machine can run"
        )
    return resolved


def resolved_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """A copy of ``args`` with both devices resolved to what the repeats will run on.

    An unset ``--arena-device`` follows the training device. The parser leaves it
    unset rather than filling it in, so a run card can tell "not given" from
    "given" and require both on a card that pins a torch build
    (``alpha_zero.local.run_card`` reads ``train_device`` and ``arena_device``).

    Raises:
        DeviceUnavailableError: an explicit device this machine cannot run.
    """
    train = resolve_run_device(args.train_device, "--train-device")
    arena = train
    if args.arena_device is not None:
        arena = resolve_run_device(args.arena_device, "--arena-device")
    return argparse.Namespace(**{**vars(args), "train_device": train, "arena_device": arena})


def worker_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    """The command line of one repeat: this module, in worker mode, in a new process."""
    command = [
        *(sys.executable, "-m", MODULE),
        *("--worker", str(run_dir)),
        *("--train-device", str(args.train_device)),
        *("--arena-device", str(args.arena_device)),
        *("--seed", str(args.seed)),
        *("--board_size", str(args.board_size)),
        *("--max_turns", str(args.max_turns)),
        *("--k", str(args.k)),
        *("--n", str(args.n)),
        *("--iterations", str(args.iterations)),
        *("--games", str(args.games)),
        *("--arena-games", str(args.arena_games)),
    ]
    if args.approval_ref:
        command += ["--max_games", str(args.max_games), "--approval_ref", str(args.approval_ref)]
    return command


def _run_worker_process(command: list[str], log: Path) -> int:
    """Run one worker to completion, output to ``log``; kill it if we are interrupted.

    No ``env``: the child inherits this process's environment unchanged.
    """
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
        try:
            return proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise


def _log_tail(log: Path, lines: int = WORKER_LOG_TAIL_LINES) -> list[str]:
    """The last ``lines`` lines of a worker's log, or ``[]`` if it cannot be read."""
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


def _worker_report(run_dir: Path) -> dict[str, Any] | None:
    try:
        report = json.loads((run_dir / WORKER_REPORT_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return report if isinstance(report, dict) else None


def coverage_gaps(
    digests: Sequence[Mapping[str, Any]],
    iterations: int,
    reports: Sequence[Mapping[str, Any] | None],
    *,
    cuda: bool,
) -> list[str]:
    """Why an empty difference list would not be a reproduction, or ``[]``.

    Every repeat must have offered the pre-training checkpoint, the last trained
    one and the arena, and training must have changed the weights: equal
    checkpoints that are all the seeded initialisation say nothing about training.
    A repeat on a CUDA device must also report deterministic algorithms enforced
    (``reports`` are the workers' :func:`environment_report`): section 7.1 rests
    the reproduction on that enforcement, so two GPU runs that happened to agree
    without it are not evidence for it.
    """
    first, last = "checkpoint_0.pt", f"checkpoint_{iterations}.pt"
    gaps: list[str] = []
    for name, digest, report in zip(REPEAT_NAMES, digests, reports, strict=True):
        checkpoints = digest[reproduction.COMPONENT_CHECKPOINTS]
        gaps += [f"{name}: {c} was not compared" for c in (first, last) if c not in checkpoints]
        if ARENA_ARM not in digest[reproduction.COMPONENT_ARENAS]:
            gaps.append(f"{name}: arena {ARENA_ARM} was not compared")
        if first in checkpoints and checkpoints.get(last) == checkpoints[first]:
            gaps.append(f"{name}: {last} equals {first}, so training changed no weight")
        if cuda and (report or {}).get("deterministic_algorithms") is not True:
            gaps.append(f"{name}: ran on CUDA without deterministic algorithms enforced")
    return gaps


def compare_runs(work_dir: Path, iterations: int, *, cuda: bool = False) -> dict[str, Any]:
    """Digest both repeats and compare them in both directions.

    ``cuda`` says whether a phase ran on a CUDA device (see :func:`coverage_gaps`).
    """
    digests = [reproduction.run_digest(work_dir / name) for name in REPEAT_NAMES]
    first, second = digests
    differences = [
        f"{REPEAT_NAMES[0]} -> {REPEAT_NAMES[1]}: {difference}"
        for difference in reproduction.compare_digests(first, second)
    ]
    # Both ways: a component only the second repeat produced is a difference too.
    differences += [
        f"{REPEAT_NAMES[1]} -> {REPEAT_NAMES[0]}: {difference}"
        for difference in reproduction.compare_digests(second, first)
    ]
    reports = [_worker_report(work_dir / name) for name in REPEAT_NAMES]
    if reports[0] != reports[1]:
        differences.append(f"the repeats report different environments: {reports}")
    gaps = coverage_gaps(digests, iterations, reports, cuda=cuda)
    if differences:
        status = STATUS_DIFFERS
    elif gaps:
        status = STATUS_INCOMPLETE
    else:
        status = STATUS_REPRODUCED
    return {
        "status": status,
        "components_compared": reproduction.component_counts(first),
        "differences": differences,
        "coverage_gaps": gaps,
        "environment": reports[0],
    }


def check(args: argparse.Namespace, work_dir: Path) -> dict[str, Any]:
    """Run both repeats under ``work_dir`` and return the outcome."""
    for name in REPEAT_NAMES:
        run_dir = work_dir / name
        log = work_dir / f"{name}.log"
        logger.info("repeat %s: starting a worker (log: %s)", name, log)
        code = _run_worker_process(worker_command(args, run_dir), log)
        if code != 0:
            logger.error("repeat %s: the worker exited with status %d", name, code)
            return {
                "status": STATUS_WORKER_FAILED,
                "failed_repeat": name,
                "worker_exit_code": code,
                "worker_log_tail": _log_tail(log),
                "components_compared": dict.fromkeys(reproduction.COMPONENT_KINDS, 0),
                "differences": [],
                "environment": None,
            }
    on_cuda = any(cuda_ordinal(d) is not None for d in (args.train_device, args.arena_device))
    return compare_runs(work_dir, args.iterations, cuda=on_cuda)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1; got {value}")
    return value


def _positive_even_int(text: str) -> int:
    value = _positive_int(text)
    if value % 2:
        raise argparse.ArgumentTypeError(f"must be even (the arena pairs seats); got {value}")
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    p.add_argument("--train-device", default="cpu", help="training and self-play device")
    p.add_argument("--arena-device", default=None, help="arena device (default: --train-device)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--board_size", type=_positive_int, default=8)
    p.add_argument("--max_turns", type=_positive_int, default=60)
    p.add_argument("--k", type=_positive_int, default=8, help="Gumbel sampled actions")
    p.add_argument("--n", type=_positive_int, default=24, help="simulations per move")
    p.add_argument("--iterations", type=_positive_int, default=2, help="training iterations")
    p.add_argument("--games", type=_positive_int, default=2, help="self-play games per iteration")
    p.add_argument("--arena-games", type=_positive_even_int, default=4)
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="where the two repeats are written (kept). Default: a temporary directory, "
        "removed after a REPRODUCED check unless --keep.",
    )
    p.add_argument("--keep", action="store_true", help="keep the temporary work directory")
    p.add_argument("--out", type=Path, default=None, help="write JSON here")
    # One repeat, run by the check itself in a new process.
    p.add_argument("--worker", type=Path, default=None, help=argparse.SUPPRESS)
    add_budget_arguments(p)
    # --arena-device is left unset here on purpose: see resolved_arguments().
    return p.parse_args(argv)


def _claim_work_dir(requested: Path | None) -> tuple[Path, bool]:
    """``(work directory, whether it is ours to remove)``; refuses to reuse a repeat."""
    if requested is None:
        return Path(tempfile.mkdtemp(prefix=f"{HARNESS}_")), True
    # Absolute: the workers run from the repository root, not from this process's cwd.
    requested = requested.resolve()
    taken = [name for name in REPEAT_NAMES if (requested / name).exists()]
    if taken:
        raise FileExistsError(
            f"{requested} already holds {', '.join(taken)}; give the check an unused --work-dir"
        )
    requested.mkdir(parents=True, exist_ok=True)
    return requested, False


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    # Game budget: refuse before a process is started or any game is played.
    # A worker prices itself too, so --worker is no way around the cap.
    priced = enforce_cli(RUNNER, budget_inputs(args), args)
    try:
        # Workers resolve their devices too: one started by hand must not fall back to CPU.
        run_args = resolved_arguments(args)
        if args.worker is not None:
            return run_worker(run_args)
        work_dir, temporary = _claim_work_dir(args.work_dir)
    except (DeviceUnavailableError, FileExistsError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_CODE_REFUSED

    outcome = check(run_args, work_dir)
    reproduced = outcome["status"] == STATUS_REPRODUCED
    removed = temporary and reproduced and not args.keep
    if removed:
        shutil.rmtree(work_dir, ignore_errors=True)
    environment = outcome.pop("environment") or environment_report(
        run_args.train_device, run_args.arena_device, probe_gpus=False
    )
    payload: dict[str, Any] = {
        "harness": HARNESS,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        # The devices the repeats ran on, not the ones asked for (``args``):
        # ``auto`` and an unset --arena-device resolve, and an unavailable
        # explicit device was refused above.
        "train_device": run_args.train_device,
        "arena_device": run_args.arena_device,
        "repeats": REPEATS,
        "games_priced": priced,
        "work_dir": None if removed else str(work_dir),
        **environment,
        **outcome,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        logger.info("wrote %s", args.out)
    print(text)
    return EXIT_REPRODUCED if reproduced else EXIT_NOT_REPRODUCED


if __name__ == "__main__":
    sys.exit(main())
