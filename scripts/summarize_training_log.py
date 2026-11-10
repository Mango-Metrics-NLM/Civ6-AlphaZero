#!/usr/bin/env python3
"""Turn a ``training_civ6`` run log into a committable evidence artifact.

The project had no way to produce run evidence: ``analyse_training_run.py``
reads MLflow, and ``alpha_zero.training_civ6`` does not log there, so a local
training run left nothing behind but a terminal scrollback. This parses that
scrollback into the same shape as ``docs/evidence/hexcrystal_ab/`` -- a JSON
with a provenance block and a markdown report -- so a run can be cited later.

Stdlib only, matching the governance validators, so it runs anywhere the
training run itself ran.

Usage::

    python scripts/summarize_training_log.py run.log \\
        --out docs/evidence/my_run --title "My run" \\
        --config board_size=8 --config max_turns=20

Exit codes: 0 on success, 1 if the log contains no parseable iteration.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The log is written by logging with a "%(asctime)s [%(levelname)s] " prefix;
# every pattern below therefore matches on a suffix, not a whole line.
RE_PARAMS = re.compile(r"Network parameters:\s*([\d,]+)")
RE_ACTIONS = re.compile(r"Action space size:\s*(\d+)")
RE_DEVICE = re.compile(r"Device:\s*(\S+)")
RE_PLAYERS = re.compile(
    r"Players:\s*(\d+)\s*\|\s*spatial channels:\s*(\d+)\s*\|\s*"
    r"value head:\s*(\S+)\s*\|\s*scoring:\s*(\S+)"
)
RE_ITER = re.compile(r"Iteration (\d+)/(\d+)")
RE_SELFPLAY = re.compile(r"Self-play:\s*(\d+) examples from (\d+) games \(([\d.]+)s\)")
RE_EPOCH = re.compile(
    r"Epoch (\d+):\s*loss=([\d.eE+-]+),\s*policy=([\d.eE+-]+),\s*value=([\d.eE+-]+)"
)
RE_TRAIN_DONE = re.compile(r"Training complete \(([\d.]+)s\)")
RE_WINRATE = re.compile(r"Win rate vs random:\s*([\d.]+)%")
# The draw-aware twin written by training_civ6.evaluate_against_random (E1):
#   Eval score rate vs random: 0.5250 (95% CI [0.3921, 0.6579], draw rate 0.2500,
#   win rate 0.4000, n=20)
RE_SCORERATE = re.compile(
    r"Eval score rate vs random:\s*([\d.]+) \(95% CI \[([\d.]+), ([\d.]+)\], "
    r"draw rate ([\d.]+), win rate ([\d.]+), n=(\d+)\)"
)
RE_EXIT = re.compile(r"^EXIT=(\d+)")
RE_REAL = re.compile(r"^real\s+(\d+)m([\d.]+)s")


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _torch_version() -> str | None:
    try:
        import torch
    except Exception:
        return None
    return str(torch.__version__)


def parse_log(text: str) -> dict[str, Any]:
    """Extract per-iteration metrics from a training log.

    Iterations are keyed off the ``Iteration i/N`` banner; everything after a
    banner and before the next one belongs to that iteration. A run killed
    mid-iteration therefore yields a final entry with missing fields rather
    than being dropped, which is the honest representation.
    """
    result: dict[str, Any] = {"iterations": [], "exit_code": None, "wall_clock_seconds": None}
    current: dict[str, Any] | None = None

    for raw in text.splitlines():
        # tqdm writes progress with carriage returns; take the last segment.
        line = raw.split("\r")[-1]

        if (m := RE_PARAMS.search(line)) and "network_parameters" not in result:
            result["network_parameters"] = int(m.group(1).replace(",", ""))
        if (m := RE_ACTIONS.search(line)) and "action_space_size" not in result:
            result["action_space_size"] = int(m.group(1))
        if (m := RE_DEVICE.search(line)) and "device" not in result:
            result["device"] = m.group(1)
        if (m := RE_PLAYERS.search(line)) and "num_players" not in result:
            result["num_players"] = int(m.group(1))
            result["spatial_channels"] = int(m.group(2))
            result["value_head"] = m.group(3)
            result["scoring_method"] = m.group(4)

        if m := RE_ITER.search(line):
            current = {
                "iteration": int(m.group(1)),
                "total_iterations": int(m.group(2)),
                "epochs": [],
            }
            result["iterations"].append(current)
            continue

        if current is None:
            continue

        if m := RE_SELFPLAY.search(line):
            current["selfplay_examples"] = int(m.group(1))
            current["selfplay_games"] = int(m.group(2))
            current["selfplay_seconds"] = float(m.group(3))
        elif m := RE_EPOCH.search(line):
            current["epochs"].append(
                {
                    "epoch": int(m.group(1)),
                    "total_loss": float(m.group(2)),
                    "policy_loss": float(m.group(3)),
                    "value_loss": float(m.group(4)),
                }
            )
        elif m := RE_TRAIN_DONE.search(line):
            current["training_seconds"] = float(m.group(1))
        elif m := RE_WINRATE.search(line):
            # Stored as a fraction; the log prints a percentage.
            current["win_rate_vs_random"] = float(m.group(1)) / 100.0
        elif m := RE_SCORERATE.search(line):
            current["score_rate_vs_random"] = float(m.group(1))
            current["score_ci_low"] = float(m.group(2))
            current["score_ci_high"] = float(m.group(3))
            current["draw_rate_vs_random"] = float(m.group(4))
            current["eval_games"] = int(m.group(6))
        elif m := RE_EXIT.match(line):
            result["exit_code"] = int(m.group(1))
        elif m := RE_REAL.match(line):
            result["wall_clock_seconds"] = int(m.group(1)) * 60 + float(m.group(2))

    return result


def _parse_config(pairs: list[str]) -> dict[str, Any]:
    """Parse repeated ``key=value`` flags, coercing numbers where possible."""
    config: dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key:
            continue
        text = value.strip()
        for caster in (int, float):
            try:
                config[key] = caster(text)
                break
            except ValueError:
                continue
        else:
            config[key] = text
    return config


def render_markdown(data: dict[str, Any], title: str, notes: str) -> str:
    """Render the report, including the limitations section the convention requires."""
    prov = data["provenance"]
    lines = [
        f"# {title}",
        "",
        f"Generated {prov['timestamp_utc']} from commit `{prov.get('git_revision') or 'unknown'}`.",
        "",
        "## Per-iteration results",
        "",
        "| Iter | Self-play games | Examples | Self-play (s) | Train (s) | Final loss "
        "| Win rate vs random | Score rate (95% CI) | Draw rate |",
        "|-----:|----------------:|---------:|--------------:|----------:|-----------:"
        "|-------------------:|--------------------:|----------:|",
    ]
    for it in data["iterations"]:
        epochs = it.get("epochs") or []
        final_loss = f"{epochs[-1]['total_loss']:.4f}" if epochs else "-"
        win = it.get("win_rate_vs_random")
        score = it.get("score_rate_vs_random")
        draw = it.get("draw_rate_vs_random")
        if score is None:
            score_cell = "-"
        else:
            score_cell = f"{score:.3f} [{it['score_ci_low']:.3f}, {it['score_ci_high']:.3f}]"
        lines.append(
            f"| {it['iteration']} "
            f"| {it.get('selfplay_games', '-')} "
            f"| {it.get('selfplay_examples', '-')} "
            f"| {it.get('selfplay_seconds', '-')} "
            f"| {it.get('training_seconds', '-')} "
            f"| {final_loss} "
            f"| {'-' if win is None else f'{win:.1%}'} "
            f"| {score_cell} "
            f"| {'-' if draw is None else f'{draw:.1%}'} |"
        )

    lines += ["", "## Run", "", "| Setting | Value |", "|---------|-------|"]
    for key, value in sorted(prov.get("config", {}).items()):
        lines.append(f"| `{key}` | {value} |")
    for key in ("device", "python_version", "torch_version", "git_revision"):
        if prov.get(key):
            lines.append(f"| {key} | `{prov[key]}` |")
    for key in ("network_parameters", "action_space_size", "spatial_channels", "num_players"):
        if data.get(key) is not None:
            lines.append(f"| {key} | {data[key]} |")
    if data.get("wall_clock_seconds") is not None:
        lines.append(f"| wall clock | {data['wall_clock_seconds']:.1f}s |")
    lines.append(f"| exit code | {data.get('exit_code')} |")

    lines += ["", "## Scope and limitations", "", notes.strip(), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("log", type=Path, help="Path to the training run log")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--title", default="Training run evidence")
    parser.add_argument("--basename", default="run", help="Base filename for the two artifacts")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Run setting to record in provenance (repeatable)",
    )
    parser.add_argument(
        "--notes",
        default=(
            "State the scale, what was and was not measured, and what a reader "
            "must not conclude from this run."
        ),
        help="Contents of the 'Scope and limitations' section",
    )
    args = parser.parse_args(argv)

    if not args.log.is_file():
        print(f"error: no such log file: {args.log}", file=sys.stderr)
        return 1

    data = parse_log(args.log.read_text(errors="replace"))
    if not data["iterations"]:
        print(f"error: no iterations parsed from {args.log}", file=sys.stderr)
        return 1

    data["provenance"] = {
        "config": _parse_config(args.config),
        "device": data.get("device"),
        "git_revision": _git_revision(),
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "torch_version": _torch_version(),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"{args.basename}_result.json"
    md_path = args.out / f"{args.basename}_report.md"
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(data, args.title, args.notes))

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
