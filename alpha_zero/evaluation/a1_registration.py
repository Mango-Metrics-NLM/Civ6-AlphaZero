"""The A1 registration's commands, read from ``docs/EVALUATION.md`` section 7.1 (LF-A1, LF-D8).

Section 7.1 holds the registered commands as machine-readable fenced blocks
(``<!-- a1-commands:base -->`` and ``<!-- a1-commands:branch-60 -->``) and one
slot, "Branch in force", that says which positive-control branch applies.
:func:`registered_invocations` turns them into the argument lists a run card's
``run.json`` must record, so ``scripts/verify_release_evidence.py`` can compare a
run with what was registered **at the run's own commit**, token for token, and
``tests/scripts/test_a1_registration.py`` can hold the cards and the plan to the
same text.

Two facts are read from prose rather than from a block, so they are constants
here: :data:`A1R_SEED` and :data:`DATA_ROOT`. So are the verdict table's
thresholds (:func:`alpha_zero.evaluation.a1_protocol.verdict_from_bounds`).
:data:`REGISTRATION_DIGEST` records the section 7.1 digest they were last
confirmed against; ``tests/regression/test_a1_registration_digest_regression.py``
holds it equal to the registration log's last row, so any re-registration makes
a person re-confirm them in the same change.
"""

from __future__ import annotations

import re

__all__ = [
    "A1R_SEED",
    "BENCH_ENTRY",
    "DATA_ROOT",
    "REGISTRATION_DIGEST",
    "TRAIN_ENTRY",
    "RegistrationError",
    "branch_in_force",
    "command_block",
    "flag_value",
    "registered_invocations",
    "run_directory",
]

#: The section 7.1 digest the constants below (and the verdict table) were last
#: confirmed against. Must equal the registration log's last row.
REGISTRATION_DIGEST = "9853c719b3549822cfa3f6215615ff0b12d637a1194847141b359417ba1e002e"
#: A1-R's seed: "every ``--seed 20261111`` replaced by ``--seed 20261112``".
A1R_SEED = 20261112
#: ``<D>`` is ``<DATA_ROOT>\runs\<card>`` ("A card therefore runs with ``--data_root E:\civ6az-data``").
DATA_ROOT = "E:\\civ6az-data"

TRAIN_ENTRY = ("python", "-m", "alpha_zero.training_civ6")
BENCH_ENTRY = ("python", "scripts/benchmark_suite.py")
_BRANCH_RE = re.compile(r"\*\*Branch in force: ([^*]+?)\.\*\*")


class RegistrationError(ValueError):
    """Section 7.1's command blocks or branch slot cannot be read as registered."""


def command_block(section: str, block: str) -> list[tuple[str, list[str]]]:
    """``[(name, tokens)]`` from the ``<!-- a1-commands:<block> -->`` fenced block.

    ``section`` is the registered text (``a1_protocol.registered_protocol``). Each
    command is a ``# name`` line followed by one command line; tokens are split on
    whitespace (registered commands carry no quoting).

    Raises:
        RegistrationError: if the marker does not occur exactly once, is not
            followed by a ``text`` fence, a command is quoted, or the block is not
            ``# name`` / command pairs.
    """
    section = section.replace("\r\n", "\n")
    marker = f"<!-- a1-commands:{block} -->"
    if section.count(marker) != 1:
        raise RegistrationError(f"{marker} must occur exactly once in section 7.1")
    fence = section.find("```text\n", section.index(marker))
    if fence < 0:
        raise RegistrationError(f"{marker} is not followed by a text fence")
    start = fence + len("```text\n")
    end = section.find("```", start)
    lines = [ln for ln in section[start:end].splitlines() if ln.strip()]
    if not lines or len(lines) % 2 or not all(ln.startswith("# ") for ln in lines[::2]):
        raise RegistrationError(f"{marker}: expected '# name' / command line pairs")
    out = []
    for name, command in zip(lines[::2], lines[1::2], strict=True):
        if "'" in command or '"' in command:
            raise RegistrationError(f"{marker}: {name[2:]!r} is quoted")
        out.append((name[2:].strip(), command.split()))
    return out


def branch_in_force(section: str) -> int:
    """The positive-control branch section 7.1 says is in force: 40 or 60.

    Raises:
        RegistrationError: while the slot still reads "pending T0b", or when it
            is missing, repeated, or anything but 40 or 60.
    """
    found = _BRANCH_RE.findall(section)
    if len(found) != 1:
        raise RegistrationError(f"expected one 'Branch in force' slot, found {len(found)}")
    value = found[0].strip()
    if value not in ("40", "60"):
        raise RegistrationError(
            f"'Branch in force: {value}': the positive-control branch is not registered yet"
        )
    return int(value)


def run_directory(card_id: str, data_root: str = DATA_ROOT) -> str:
    """``<D>`` for a card: ``<data_root>\\runs\\<card_id>``."""
    return f"{data_root}\\runs\\{card_id}"


def registered_invocations(
    section: str, *, card_id: str, seed: int, data_root: str = DATA_ROOT
) -> dict[str, list[str]]:
    """One A1-family card's registered invocations: name -> runner arguments, in order.

    The base block, with the "branch-60" commands substituted when section 7.1
    says branch 60 is in force; ``<D>`` written out by :func:`run_directory`;
    every ``--seed`` value replaced by ``seed``. The entry point
    (:data:`TRAIN_ENTRY` / :data:`BENCH_ENTRY`) is dropped, so each value compares
    directly with a ``run.json`` invocation's ``argv``.

    Raises:
        RegistrationError: as :func:`command_block` and :func:`branch_in_force`,
            or for a command with another entry point.
    """
    commands = dict(command_block(section, "base"))
    if branch_in_force(section) == 60:
        branch = dict(command_block(section, "branch-60"))
        unknown = sorted(set(branch) - set(commands))
        if unknown:
            raise RegistrationError(f"branch-60 names unregistered invocations {unknown}")
        commands.update(branch)
    directory = run_directory(card_id, data_root)
    out: dict[str, list[str]] = {}
    for name, tokens in commands.items():
        if tuple(tokens[:3]) == TRAIN_ENTRY:
            args = tokens[3:]
        elif tuple(tokens[:2]) == BENCH_ENTRY:
            args = tokens[2:]
        else:
            raise RegistrationError(f"{name!r} has an unregistered entry point: {tokens[:3]}")
        args = [a.replace("<D>", directory) for a in args]
        out[name] = [
            str(seed) if i > 0 and args[i - 1] == "--seed" else a for i, a in enumerate(args)
        ]
    return out


def flag_value(args: list[str], flag: str) -> str | None:
    """The value after the last ``flag`` in ``args`` (argparse keeps the last one)."""
    value = None
    for i, arg in enumerate(args[:-1]):
        if arg == flag:
            value = args[i + 1]
    return value
