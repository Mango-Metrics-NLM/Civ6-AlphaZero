"""Run-card driver: price, pin, isolate and execute a local run (LF-E6, LF-G11).

A *run card* is a YAML file in ``configs/run_cards/`` naming a git SHA and a
list of runner invocations. :func:`run_card` turns it into a recorded run:

1. **Price** every invocation with the runner's *own* argv parser and
   ``game_budget`` cost model, and refuse the card -- before anything is
   created or spawned -- when the sum exceeds ``max_games`` without an
   ``approval_ref``. Each invocation is also checked against the cap the child
   will itself enforce, so a card cannot pass here and be refused mid-run.
2. **Prepare** the data root (:func:`prepare_data_root`): validate it, and
   ACL-harden it only when this driver creates it.
3. **Isolate** (:func:`~alpha_zero.local.isolation.enter_local_mode`) and **pin**
   (:func:`~alpha_zero.local.pinned_env.prepare_pinned_run`).
4. **Preflight** a real child of the pinned interpreter that runs the pinned
   code's own :func:`~alpha_zero.local.pinned_env.assert_running_pinned`, then
   re-verify what it reports (prefix, ``alpha_zero.__file__``, version) with
   THIS checkout's code (:func:`verify_preflight`), so a pinned SHA whose check
   is weaker or broken cannot vouch for itself.
5. **Run** the invocations sequentially (no shell, ``cwd=src``,
   ``PYTHONNOUSERSITE=1``, temp under the data root), one log per invocation.
   Each child is launched with ``--max_games <its priced games>`` appended
   (argparse keeps the last value), so the child's own budget check caps it at
   what this driver priced.

``<root>/runs/<card_id>/run.json`` is written at the start (``RUNNING``), after
every invocation, and at the end (``COMPLETED``/``FAILED``/``VOID``). An
interruption -- Ctrl+C, Ctrl+Break (``SIGBREAK``) or ``SIGTERM`` -- makes the
card ``VOID``; so does finding a ``RUNNING`` record from an earlier attempt,
because a resumed card is not the card that was priced and pinned. A run
directory is single-use. ``run.json`` records environment variable **names**
only; values never reach it.

Windows semantics, stated plainly: ``TerminateProcess`` (Task Manager, a hard
kill, and what most Windows tooling means by "SIGTERM") cannot be caught, so
the driver gets no chance to write ``VOID``. **A lingering ``RUNNING`` record
must be read as VOID** -- the next attempt of the card rewrites it so. A
hard-killed driver can also orphan the child it was waiting on; containing
children in a Job object is planned, not done.

Pricing uses this checkout's parsers, defaults and cost models (and resolves a
relative ``--config`` from the driver's cwd) while the children run the pinned
SHA's from ``cwd=src``, so the two can price an invocation differently. The
appended ``--max_games <priced>`` keeps the card total safe: a child that
prices itself higher than the driver did refuses to start (exit 2, card
FAILED). The one exception is an invocation whose own argv carries
``--approval_ref``: ``game_budget`` lets an approved run exceed its cap, so
that invocation is bounded by its approval, not by this driver.

Public surface: :func:`load_card`, :func:`price_card`, :func:`prepare_data_root`,
:func:`refuse_placeholder_sha`, :func:`refuse_mcts_overrides`,
:func:`refuse_unresolvable_sha`, :func:`run_card`, :func:`main`, :data:`RUNNERS`.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alpha_zero.local import devices
from alpha_zero.local.card import Invocation, RunCard, RunCardError, load_card
from alpha_zero.local.devices import TORCH_REPORT_CODE
from alpha_zero.local.isolation import enter_local_mode
from alpha_zero.local.pinned_env import (
    PinnedEnv,
    PinnedEnvError,
    Runner,
    _pyproject_version,
    child_env,
    prepare_pinned_run,
    resolve_commit,
)
from alpha_zero.utils import game_budget
from alpha_zero.utils.checkpoint import GIT_SHA_ENV
from alpha_zero.utils.data_root import (
    ENV_VAR as DATA_ROOT_ENV,
)
from alpha_zero.utils.data_root import (
    RUN_ID_MAX_LENGTH,
    SUBDIRS,
    DataRootError,
    assert_safe_data_root,
    harden_data_root,
    resolve_data_root,
    scoped_temp_env,
    validate_run_id,
)
from alpha_zero.utils.device import cuda_ordinal

logger = logging.getLogger(__name__)

RUN_JSON = "run.json"
#: ``run.json`` schema. Additive keys (``probe``, ``device_plan.*``, ``error_tail``)
#: do not bump it: readers accept unknown keys and ``verify_release_evidence``
#: requires exactly this value.
SCHEMA_VERSION = 1
STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_VOID = "VOID"
#: A ``--probe`` rehearsal: pinned, preflighted, device-checked, nothing launched.
STATUS_PROBED = "PROBED"
#: The attribute every runner's parser stores its ``--device`` under.
DEVICE_ATTRIBUTE = "device"
#: The logger whose "using cpu" fallback warnings are about the driver, not the run.
DEVICE_LOGGER_NAME = "alpha_zero.utils.device"
PROBE_PREFIX = "probe-"
PROBE_STAMP_FORMAT = "%Y%m%dT%H%M%S%f"
#: Lines of a failed child's log copied into ``run.json`` (``error_tail``) and printed,
#: so the reason -- a deterministic-algorithm error names the offending op -- is not
#: left in a log file the operator has to find.
ERROR_TAIL_LINES = 40
EXIT_INTERRUPTED = 130
#: Run in the pinned child. Imports the PINNED ``assert_running_pinned`` -- a SHA
#: that predates it cannot pass, which is the fail-closed answer for a card.
PREFLIGHT_CODE = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "import alpha_zero\n"
    "from alpha_zero.local.pinned_env import PinnedEnv, assert_running_pinned\n"
    "assert_running_pinned(PinnedEnv(python=Path(sys.argv[1]), src=Path(sys.argv[2])))\n"
    "info = {'alpha_zero_file': alpha_zero.__file__,"
    " 'version': str(alpha_zero.__version__), 'prefix': sys.prefix}\n"
    + TORCH_REPORT_CODE
    + "print(json.dumps(info))\n"
)


class CardResumedError(RunCardError):
    """The card's run directory already holds a run record."""


# ---------------------------------------------------------------------------
# Runner registry: how to price an argv and how to launch it
# ---------------------------------------------------------------------------

BudgetInputs = tuple[Mapping[str, Any], argparse.Namespace]


def _training_civ6_inputs(argv: Sequence[str]) -> BudgetInputs:
    from alpha_zero import training_civ6 as tc

    args = tc.build_parser().parse_args(list(argv))
    # The checkpoint dir does not affect the price; resolving it would validate a root.
    config = tc.config_from_args(args, checkpoint_dir="unused-for-pricing")
    return tc.training_budget_inputs(config, args.dry_run), args


def _benchmark_inputs(argv: Sequence[str]) -> BudgetInputs:
    from alpha_zero.evaluation import benchmark

    args = benchmark._parse_args(list(argv))
    return benchmark.benchmark_budget_inputs(args), args


def _orchestrator_inputs(argv: Sequence[str]) -> BudgetInputs:
    from alpha_zero.orchestrator.__main__ import create_parser, load_config

    args = create_parser().parse_args(list(argv))
    config = load_config(args)
    return game_budget.orchestrator_budget_inputs(config, args.dry_run), args


def _calibrate_throughput_inputs(argv: Sequence[str]) -> BudgetInputs:
    # By name: scripts/ is outside the type-checked package (``mypy alpha_zero/``).
    calibrate_throughput = importlib.import_module("scripts.calibrate_throughput")
    args = calibrate_throughput._parse_args(list(argv))
    return {"games": args.games}, args


def _determinism_check_inputs(argv: Sequence[str]) -> BudgetInputs:
    # By name: scripts/ is outside the type-checked package (``mypy alpha_zero/``).
    check_determinism = importlib.import_module("scripts.check_determinism")
    args = check_determinism._parse_args(list(argv))
    return check_determinism.budget_inputs(args), args


@dataclass(frozen=True)
class RunnerSpec:
    """A runner the driver can price and launch: ``python -m <module> <argv>``."""

    module: str
    budget_inputs: Callable[[Sequence[str]], BudgetInputs]
    #: The parser attributes that hold this runner's device(s). One for most
    #: runners; a runner that trains on one device and judges on another has two.
    device_attributes: tuple[str, ...] = (DEVICE_ATTRIBUTE,)


#: Runners a card may name. A ``game_budget`` runner absent here is refused.
RUNNERS: dict[str, RunnerSpec] = {
    "training_civ6": RunnerSpec("alpha_zero.training_civ6", _training_civ6_inputs),
    "benchmark": RunnerSpec("alpha_zero.evaluation.benchmark", _benchmark_inputs),
    "orchestrator": RunnerSpec("alpha_zero.orchestrator", _orchestrator_inputs),
    # T0b's wall-clock calibration; children run it from the pinned src (cwd=src).
    "calibrate_throughput": RunnerSpec(
        "scripts.calibrate_throughput", _calibrate_throughput_inputs
    ),
    # T0b's cross-process reproduction check (section 7.1 "Devices"): it trains on one
    # device and judges on another, so both are read and both are held to the device check.
    "determinism_check": RunnerSpec(
        "scripts.check_determinism",
        _determinism_check_inputs,
        device_attributes=("train_device", "arena_device"),
    ),
}


# ---------------------------------------------------------------------------
# Card model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PricedInvocation:
    invocation: Invocation
    games: int


@dataclass(frozen=True)
class PricedCard:
    card: RunCard
    invocations: tuple[PricedInvocation, ...]

    @property
    def total(self) -> int:
        return sum(p.games for p in self.invocations)


@dataclass
class InvocationResult:
    name: str
    runner: str
    argv: list[str]
    priced_games: int
    launched_argv: list[str] = field(default_factory=list)
    exit_code: int | None = None
    duration_s: float | None = None
    log: str = ""


@dataclass
class CardResult:
    card_id: str
    status: str
    run_dir: Path
    total_priced_games: int
    invocations: list[InvocationResult] = field(default_factory=list)
    error: str | None = None
    error_tail: list[str] = field(default_factory=list)
    device_plan: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Devices (the logic lives in alpha_zero.local.devices; these bind it to a card)
# ---------------------------------------------------------------------------


def _parsed(inv: Invocation) -> BudgetInputs:
    """``inv``'s argv through its runner's own parser. Raises :class:`RunCardError`."""
    if inv.runner not in game_budget.priced_runners():
        raise RunCardError(
            f"invocation {inv.name!r}: unknown runner {inv.runner!r}; "
            f"supported: {', '.join(sorted(RUNNERS))}"
        )
    spec = RUNNERS.get(inv.runner)
    if spec is None:
        raise RunCardError(
            f"invocation {inv.name!r}: runner {inv.runner!r} has a cost model but run cards "
            f"cannot resolve its argv yet; supported: {', '.join(sorted(RUNNERS))}"
        )
    try:
        with _driver_device_warnings_silenced():
            return spec.budget_inputs(inv.argv)
    except SystemExit as exc:  # argparse rejected the argv
        raise RunCardError(
            f"invocation {inv.name!r}: {inv.runner} rejected argv {list(inv.argv)} ({exc})"
        ) from None


@contextlib.contextmanager
def _driver_device_warnings_silenced() -> Iterator[None]:
    """Mute ``utils.device``'s fallback warnings while the DRIVER resolves an argv.

    Pricing the trainer builds its config, which resolves the device in this
    interpreter. The driver's torch is not the run's: on a CPU-only driver that
    prints "device 'cuda:0' requested but CUDA is unavailable; using cpu" for a
    card that will run on the GPU in its pinned environment -- the very message
    this module exists to make trustworthy. What the run really has is decided by
    the pinned child's preflight (``devices.device_problem``), not here.
    """
    device_logger = logging.getLogger(DEVICE_LOGGER_NAME)
    previous = device_logger.level
    device_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        device_logger.setLevel(previous)


def invocation_devices(inv: Invocation) -> dict[str, str | None]:
    """The device(s) ``inv`` will run on, as its runner's own parser resolves them.

    Reading the parsed namespace -- not the argv tokens -- is what makes this
    agree with the child: argparse accepts abbreviations (``--dev cuda:1``) and
    applies defaults (the benchmark's is ``cpu``; the trainer's is unset, which
    it resolves to ``auto``). ``None`` means the runner leaves that device unset.

    Keys are the invocation's name for a single-device runner and
    ``<name>.<attribute>`` when a runner has several
    (:attr:`RunnerSpec.device_attributes`), so each one is checked and recorded.
    """
    _, args = _parsed(inv)
    attributes = RUNNERS[inv.runner].device_attributes
    found: dict[str, str | None] = {}
    for attribute in attributes:
        key = inv.name if len(attributes) == 1 else f"{inv.name}.{attribute}"
        device = getattr(args, attribute, None)
        found[key] = None if device is None else str(device)
    return found


def resolved_devices(card: RunCard) -> dict[str, str | None]:
    """``{invocation (or invocation.attribute): its parser-resolved device, or None}``."""
    resolved: dict[str, str | None] = {}
    for inv in card.invocations:
        resolved.update(invocation_devices(inv))
    return resolved


def requested_devices(card: RunCard) -> dict[str, str]:
    """The devices of ``card`` that resolve to a value, by invocation."""
    return {name: device for name, device in resolved_devices(card).items() if device is not None}


def refuse_implicit_devices(card: RunCard) -> None:
    """Refuse a torch-pinned card that leaves any invocation's device to ``auto`` or a default.

    A static property of the card, so it is refused before anything is created
    (see :func:`alpha_zero.local.devices.implicit_device_problem`).
    """
    problem = devices.implicit_device_problem(resolved_devices(card), torch_pin=card.torch_pin)
    if problem is not None:
        raise RunCardError(f"card {card.card_id!r}: {problem}")


def device_problem(card: RunCard, torch_report: Any, *, visible: str | None = None) -> str | None:
    """Why the pinned environment cannot honour the card's devices, or ``None``.

    ``visible`` is the ``CUDA_VISIBLE_DEVICES`` in force for the children; it
    defaults to the card's own value. See :func:`alpha_zero.local.devices.device_problem`.
    """
    return devices.device_problem(
        requested_devices(card),
        torch_report,
        torch_pin=card.torch_pin,
        visible=card.cuda_visible_devices if visible is None else visible,
        expected_gpus=card.gpus,
    )


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def price_invocation(inv: Invocation) -> int:
    """Games one invocation can play, priced by its runner's own parser.

    Public so ``scripts/verify_release_evidence.py`` prices a registered card the
    way this module does. Raises :class:`RunCardError` for an unknown runner or
    an argv the parser rejects.
    """
    resolved, args = _parsed(inv)
    # The cap the child itself will enforce with this argv -- refuse now, not mid-card.
    games = game_budget.enforce(
        inv.runner,
        resolved,
        max_games=int(getattr(args, "max_games", game_budget.DEFAULT_MAX_GAMES)),
        approval_ref=getattr(args, "approval_ref", None),
    )
    if games < 0:
        raise RunCardError(f"invocation {inv.name!r}: {inv.runner} could not be priced")
    return games


def price_card(card: RunCard) -> PricedCard:
    """Price every invocation and refuse a card whose SUM is over its cap.

    Raises :class:`~alpha_zero.utils.game_budget.GameBudgetExceededError` (or a
    subclass) for a budget refusal and :class:`RunCardError` for an unknown
    runner or an argv its parser rejects. Has no side effects.
    """
    label = f"run card {card.card_id}"
    ref = (card.approval_ref or "").strip()
    if card.max_games > game_budget.DEFAULT_MAX_GAMES and not ref:
        raise game_budget.GameBudgetCapRaisedError(label, card.max_games)
    priced = PricedCard(
        card=card,
        invocations=tuple(PricedInvocation(inv, price_invocation(inv)) for inv in card.invocations),
    )
    if priced.total > card.max_games:
        if not ref:
            detail = ", ".join(f"{p.invocation.name}={p.games:,}" for p in priced.invocations)
            raise game_budget.GameBudgetExceededError(
                label, priced.total, card.max_games, f"Per invocation: {detail}."
            )
        logger.warning(
            "%s: %s games exceeds max_games %s; proceeding on approval %s",
            label,
            f"{priced.total:,}",
            f"{card.max_games:,}",
            ref,
        )
    return priced


def is_placeholder_sha(sha: str) -> bool:
    """``True`` for an all-zero sha: a card that prices but is not yet pinned to a commit."""
    return bool(sha) and set(sha) == {"0"}


def refuse_placeholder_sha(card: RunCard) -> None:
    """Refuse a card whose ``sha`` is a placeholder, before anything is created.

    A registered card (``configs/run_cards/a1.yaml``) ships with an all-zero
    ``sha`` until its registration commit is merged. Pricing it is useful; running
    it would claim the card's single-use run directory and write a FAILED record
    for a run that never happened (T13 review I2).
    """
    if is_placeholder_sha(card.sha):
        raise RunCardError(
            f"run card {card.card_id}: placeholder sha {card.sha!r}; pin the merged commit "
            "that carries its registration before running it (--price_only still works)"
        )


def refuse_mcts_overrides(environ: Mapping[str, str] | None = None) -> None:
    """Refuse to start while any ``MCTS_*`` variable is set, before anything is created.

    ``core/mcts_constants.py`` reads ``MCTS_*`` overrides, so one would change the
    search a registered card runs, and section 7.1 makes such a run VOID -- but only
    once it has finished. No card needs one, so the driver refuses up front
    (final review M-3). Unset the variable to proceed.
    """
    env = os.environ if environ is None else environ
    names = sorted(n for n in env if n.upper().startswith("MCTS_"))
    if names:
        raise RunCardError(
            f"MCTS_* override(s) {names} are set; a card run under them is VOID "
            "(docs/EVALUATION.md section 7.1). Unset them before running a card."
        )


def refuse_unresolvable_sha(
    card: RunCard, *, repo_root: Path, runner: Runner | None = None
) -> None:
    """Refuse a well-formed ``sha`` that names no commit, before anything is created.

    Pinning resolves the sha again later; resolving it here first means a typo,
    an ambiguous abbreviation or an unfetched merge commit is refused before the
    single-use run directory is claimed (T13 rereview N1). ``--price_only`` never
    calls this.
    """
    try:
        resolve_commit(card.sha, runner=runner, repo_root=repo_root)
    except PinnedEnvError as exc:
        raise RunCardError(
            f"run card {card.card_id}: sha {card.sha!r} does not name a commit in "
            f"{repo_root} ({exc}); fetch it or fix the card"
        ) from None


# ---------------------------------------------------------------------------
# Data root, run record, subprocesses
# ---------------------------------------------------------------------------


def prepare_data_root(root: Path, *, repo_root: Path) -> Path:
    """Validate ``root``; ACL-harden it only if this call creates it. Returns it resolved."""
    root = Path(root).resolve()
    assert_safe_data_root(root, repo_root)
    if root.exists():
        logger.info("data root %s already exists; not changing its ACL", root)
        for sub in SUBDIRS:
            (root / sub).mkdir(exist_ok=True)
    else:
        harden_data_root(root)
    return root


def _write_record(path: Path, record: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def probe_run_name(card_id: str, *, now: datetime | None = None) -> str:
    """The throwaway run directory name for a ``--probe`` of ``card_id``.

    ``probe-<card_id>-<UTC stamp to the microsecond>``, so two probes of one card
    never collide and neither can be mistaken for (or claim) the card's own
    directory. The card id is shortened if needed to keep the name a valid run id.
    """
    stamp = (now or datetime.now(UTC)).strftime(PROBE_STAMP_FORMAT)
    budget = RUN_ID_MAX_LENGTH - len(PROBE_PREFIX) - len(stamp) - 1  # 1 for the separator
    return validate_run_id(f"{PROBE_PREFIX}{card_id[:budget]}-{stamp}")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _claim_run_dir(run_dir: Path) -> None:
    """Refuse a reused run directory; VOID a record an earlier attempt left RUNNING."""
    record_path = run_dir / RUN_JSON
    if record_path.exists():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
        status = record.get("status")
        if status == STATUS_RUNNING:
            record.update(
                status=STATUS_VOID,
                void_reason="resumed: an earlier attempt never finished; a card cannot resume",
                finished_at=_now(),
            )
            _write_record(record_path, record)
            raise CardResumedError(f"{run_dir}: earlier attempt never finished; marked VOID")
        raise CardResumedError(
            f"{run_dir}: already ran (status {status}); run directories are single-use -- "
            "give the card a new card_id"
        )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise CardResumedError(f"{run_dir} is not empty; run directories are single-use")
    run_dir.mkdir(parents=True, exist_ok=True)


def _run_invocation(cmd: list[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> int:
    """Run one child to completion, output to ``log``; kill it if we are interrupted."""
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(  # noqa: S603 - argv from a validated card, pinned python, no shell
            cmd, cwd=cwd, env=dict(env), stdout=handle, stderr=subprocess.STDOUT
        )
        try:
            return proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise


def _log_tail(log: Path, lines: int = ERROR_TAIL_LINES) -> list[str]:
    """The last ``lines`` lines of a child's log, or ``[]`` if it cannot be read."""
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


def device_report_lines(plan: Mapping[str, Any] | None) -> list[str]:
    """Human-readable lines for a ``device_plan``: the torch build, then each invocation's GPU."""
    if not plan:
        return []
    torch_info = plan.get("torch")
    if not isinstance(torch_info, Mapping):
        torch_info = {}
    names = list(torch_info.get("devices") or [])
    lines = [
        f"torch {torch_info.get('version', torch_info.get('error', 'unknown'))} "
        f"(cuda {torch_info.get('cuda')}); "
        f"{devices.ENV_CUDA_VISIBLE_DEVICES}={plan.get(devices.ENV_CUDA_VISIBLE_DEVICES)!r} "
        f"({plan.get('source')}), order {plan.get(devices.ENV_CUDA_DEVICE_ORDER)}"
    ]
    for name, device in (plan.get("requested") or {}).items():
        index = cuda_ordinal(device)
        gpu = names[index] if index is not None and index < len(names) else "-"
        lines.append(f"  {name:<24} {device:<8} {gpu}")
    return lines


def _preflight(env: PinnedEnv, child: Mapping[str, str]) -> dict[str, Any]:
    result = subprocess.run(  # noqa: S603 - fixed code string, the pinned python, no shell
        [str(env.python), "-c", PREFLIGHT_CODE, str(env.python), str(env.src)],
        cwd=env.src,
        env=dict(child),
        capture_output=True,
        text=True,
        check=False,
    )
    record: dict[str, Any] = {"exit_code": result.returncode}
    if result.returncode == 0:
        record.update(json.loads(result.stdout.strip().splitlines()[-1]))
    else:
        record["output"] = (result.stdout + result.stderr)[-4000:]
    return record


def verify_preflight(env: PinnedEnv, report: Mapping[str, Any]) -> str | None:
    """Re-check the child's preflight report with THIS checkout's rules.

    Returns ``None`` when the report proves the child ran the pinned interpreter
    and the pinned ``alpha_zero``; otherwise the reason it does not. Mirrors
    :func:`~alpha_zero.local.pinned_env.assert_running_pinned` on purpose: the
    pinned SHA's copy of that check is code under test, not a trusted oracle.
    """
    if report.get("exit_code") != 0:
        return "the pinned preflight itself failed"
    try:
        prefix = Path(str(report["prefix"])).resolve()
        file = Path(str(report["alpha_zero_file"])).resolve()
        version = str(report["version"])
    except KeyError as exc:
        return f"preflight report is missing {exc}"
    venv, src = env.venv.resolve(), env.src.resolve()
    if prefix != venv:
        return f"child prefix {prefix} is not the pinned venv {venv}"
    if not any(root == file or root in file.parents for root in (src, venv)):
        return f"child imported alpha_zero from {file}, outside the pinned src and venv"
    expected = _pyproject_version(env.src)
    if version != expected:
        return f"child alpha_zero version {version!r} != pinned pyproject version {expected!r}"
    return None


def _pip_freeze(env: PinnedEnv, child: Mapping[str, str], runner: Runner | None) -> list[str]:
    cmd = [str(env.python), "-m", "pip", "freeze", "--all"]
    if runner is not None:
        out = runner(cmd, cwd=env.src, env=child)
    else:
        out = subprocess.run(  # noqa: S603 - the pinned python's own pip, no shell
            cmd, cwd=env.src, env=dict(child), capture_output=True, text=True, check=True
        ).stdout
    return [line for line in out.splitlines() if line.strip()]


@contextlib.contextmanager
def _signals_interrupt() -> Iterator[None]:
    """Turn SIGBREAK (Ctrl+Break) and SIGTERM into KeyboardInterrupt while a card runs."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    names = [n for n in ("SIGBREAK", "SIGTERM") if hasattr(signal, n)]
    previous = {n: signal.signal(getattr(signal, n), _raise) for n in names}
    try:
        yield
    finally:
        for n, handler in previous.items():
            signal.signal(getattr(signal, n), handler)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_card(
    card_path: str | Path,
    *,
    data_root: Path | None = None,
    repo_root: Path | None = None,
    pinned_runner: Runner | None = None,
    probe: bool = False,
) -> CardResult:
    """Price, pin, isolate and run a card; return its :class:`CardResult`.

    ``probe=True`` is a rehearsal: everything up to and including the preflight
    and the device check runs in a throwaway ``runs/probe-<card_id>-<stamp>``
    directory, nothing is launched, and the card's own single-use run directory
    is not claimed. Its status is ``PROBED`` (or ``FAILED`` with the reason).

    ``data_root`` defaults to ``CIV6AZ_DATA_ROOT``. ``pinned_runner`` is the
    ``pinned_env`` subprocess seam (tests pass a fake that builds no wheel).
    Budget and card errors are raised before anything is created. A
    ``KeyboardInterrupt`` (also raised for SIGBREAK/SIGTERM) marks the card VOID
    and is re-raised; any other setup error marks it FAILED and is re-raised; a
    child exiting non-zero marks it FAILED and stops the card.
    """
    repo = Path(repo_root) if repo_root is not None else _default_repo_root()
    card = load_card(card_path)
    priced = price_card(card)  # refuses before anything exists
    refuse_implicit_devices(card)  # a card that pins torch must name every device
    refuse_placeholder_sha(card)  # likewise: a placeholder must not claim the run dir
    refuse_mcts_overrides()  # an MCTS_* override would VOID the run only after it ran
    refuse_unresolvable_sha(card, repo_root=repo, runner=pinned_runner)  # nor a typo

    root_arg = data_root if data_root is not None else resolve_data_root(required=True)
    if root_arg is None:  # unreachable: required=True raises
        raise RunCardError(f"{DATA_ROOT_ENV} is not set")
    root = prepare_data_root(Path(root_arg), repo_root=repo)
    run_name = probe_run_name(card.card_id) if probe else card.card_id
    run_dir = root / "runs" / run_name
    _claim_run_dir(run_dir)
    logs = run_dir / "logs"
    logs.mkdir()
    record_path = run_dir / RUN_JSON

    results = [
        InvocationResult(
            name=p.invocation.name,
            runner=p.invocation.runner,
            argv=list(p.invocation.argv),
            priced_games=p.games,
        )
        for p in priced.invocations
    ]
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "card_id": card.card_id,
        "card_path": str(card.path) if card.path else None,
        "requested_sha": card.sha,
        "sha": None,
        "wheel_sha256": None,
        "max_games": card.max_games,
        "approval_ref": card.approval_ref,
        "total_priced_games": priced.total,
        "status": STATUS_RUNNING,
        "started_at": _now(),
        "invocations": [],
        "probe": probe,
    }
    _write_record(record_path, record)

    def finish(status: str, **extra: Any) -> CardResult:
        record.update(status=status, finished_at=_now(), **extra)
        _write_record(record_path, record)
        logger.info("run card %s: %s (%s)", card.card_id, status, record_path)
        ran = [r for r in results if r.exit_code is not None]
        return CardResult(
            card.card_id,
            status,
            run_dir,
            priced.total,
            ran,
            error=record.get("error"),
            error_tail=list(record.get("error_tail") or []),
            device_plan=record.get("device_plan"),
        )

    try:
        with _signals_interrupt():
            base = enter_local_mode(card.card_id, run_dir)
            env = prepare_pinned_run(
                card.sha,
                run_dir,
                envs_root=root / "envs",
                runner=pinned_runner,
                repo_root=repo,
                torch_pin=card.torch_pin,
            )
            child = child_env(base)
            child.update(scoped_temp_env(root))
            child[DATA_ROOT_ENV] = str(root)
            # The child runs from a `git archive` export with no .git; this is how its
            # checkpoints and evaluations learn their commit (utils.checkpoint.read_git_sha).
            child[GIT_SHA_ENV] = env.sha
            device_plan = devices.apply_device_environment(
                child,
                cuda_visible_devices=card.cuda_visible_devices,
                requested=requested_devices(card),
                torch_pin=card.torch_pin,
                expected_gpus=card.gpus,
            )
            record.update(
                sha=env.sha,
                wheel_sha256=env.wheel_sha256,
                python=str(env.python),
                src=str(env.src),
                device_plan=device_plan,
                env_var_names=sorted(child),
                pip_freeze=_pip_freeze(env, child, pinned_runner),
            )
            preflight = _preflight(env, child)
            record["preflight"] = preflight
            record["device_plan"]["torch"] = preflight.get("torch")
            _write_record(record_path, record)
            problem = verify_preflight(env, preflight)
            if problem is not None:
                return finish(STATUS_FAILED, error=f"preflight: {problem}")
            problem = device_problem(
                card,
                preflight.get("torch"),
                visible=device_plan[devices.ENV_CUDA_VISIBLE_DEVICES],
            )
            if problem is not None:
                logger.error("run card %s: devices: %s", card.card_id, problem)
                return finish(STATUS_FAILED, error=f"devices: {problem}")
            logger.info("run card %s: devices ok -- torch %s", card.card_id, preflight.get("torch"))
            if probe:
                return finish(STATUS_PROBED)

            for result in results:
                log = logs / f"{result.name}.log"
                # Cap the child at what was priced here (argparse keeps the last value).
                result.launched_argv = [*result.argv, "--max_games", str(result.priced_games)]
                cmd = [str(env.python), "-m", RUNNERS[result.runner].module, *result.launched_argv]
                started = time.monotonic()
                result.exit_code = _run_invocation(cmd, cwd=env.src, env=child, log=log)
                result.duration_s = round(time.monotonic() - started, 3)
                result.log = str(log.relative_to(run_dir))
                record["invocations"].append(asdict(result))
                _write_record(record_path, record)
                if result.exit_code != 0:
                    return finish(
                        STATUS_FAILED,
                        error=f"{result.name} exited {result.exit_code} (see {result.log})",
                        error_tail=_log_tail(log),
                    )
            return finish(STATUS_COMPLETED)
    except KeyboardInterrupt:
        finish(STATUS_VOID, void_reason="interrupted (KeyboardInterrupt/SIGBREAK/SIGTERM)")
        raise
    except BaseException as exc:
        finish(STATUS_FAILED, error=f"{type(exc).__name__}: {exc}")
        raise


# ---------------------------------------------------------------------------
# CLI (scripts/run_card.py is a thin wrapper around this)
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """``run_card CARD [--data_root D] [--price_only | --probe]``.

    Exit 0 ok, 1 failed, 2 refused, 130 VOID.
    """
    parser = argparse.ArgumentParser(description="Price, pin and run a run card.")
    parser.add_argument("card", type=Path, help="Path to a configs/run_cards/*.yaml card")
    parser.add_argument("--data_root", type=Path, default=None, help=f"Default: ${DATA_ROOT_ENV}")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--price_only", action="store_true", help="Price the card and exit; create nothing"
    )
    mode.add_argument(
        "--probe",
        action="store_true",
        help="Rehearse: build the pinned env, preflight it and check the card's devices in a "
        "throwaway run directory; launch nothing and leave the card's own directory unclaimed",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.price_only:
            priced = price_card(load_card(args.card))
            for p in priced.invocations:
                print(f"{p.invocation.name:<24} {p.invocation.runner:<16} {p.games:>8,}")
            print(f"{'total':<41} {priced.total:>8,} (max_games {priced.card.max_games:,})")
            return 0
        result = run_card(args.card, data_root=args.data_root, probe=args.probe)
    except (game_budget.GameBudgetExceededError, RunCardError, DataRootError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return game_budget.EXIT_CODE_REFUSED
    except PinnedEnvError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("VOID: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    print(f"{result.status}: {result.run_dir / RUN_JSON}")
    if result.error:
        print(f"  reason: {result.error}", file=sys.stderr)
    for line in result.error_tail:
        print(f"  | {line}", file=sys.stderr)
    if args.probe:
        for line in device_report_lines(result.device_plan):
            print(line)
    return 0 if result.status in (STATUS_COMPLETED, STATUS_PROBED) else 1
