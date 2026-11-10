"""The power table in ``docs/EVALUATION.md`` is recomputed, not trusted.

The document quotes minimum detectable effects and implied Elo per (games,
draw-rate) cell. If ``scoring.minimum_detectable_effect`` changes, or someone
edits a cell by hand, the table and the code disagree silently. This test parses
the Markdown table and recomputes every cell from the library, at the precision
the document prints.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from alpha_zero.evaluation import scoring

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "EVALUATION.md"
HEADER = re.compile(r"^\|\s*games\s*\|(.+)\|\s*$")
ROW = re.compile(r"^\|\s*(\d+)\s*\|(.+)\|\s*$")
CELL = re.compile(r"([0-9.]+)\s*\(\+(\d+) Elo\)")


def _elo(p: float) -> float:
    return -400.0 * math.log10(1.0 / p - 1.0)


def parse_power_table(text: str) -> tuple[list[float], list[tuple[int, list[tuple[float, int]]]]]:
    lines = text.splitlines()
    draw_rates: list[float] = []
    rows: list[tuple[int, list[tuple[float, int]]]] = []
    in_table = False
    for line in lines:
        head = HEADER.match(line)
        if head:
            draw_rates = [float(x) for x in re.findall(r"draw rate ([0-9.]+)", head.group(1))]
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("|---") or line.startswith("| ---"):
            continue
        row = ROW.match(line)
        if not row:
            if rows:
                break
            continue
        cells = [(float(m), int(e)) for m, e in CELL.findall(row.group(2))]
        rows.append((int(row.group(1)), cells))
    return draw_rates, rows


def test_every_cell_matches_the_library() -> None:
    draw_rates, rows = parse_power_table(DOC.read_text())
    assert len(draw_rates) == 3, draw_rates
    assert len(rows) >= 5, "table not found or too short -- the check would be vacuous"
    mismatches = []
    for n, cells in rows:
        assert len(cells) == len(draw_rates), (n, cells)
        for d, (mde_doc, elo_doc) in zip(draw_rates, cells, strict=True):
            mde = scoring.minimum_detectable_effect(n, d)
            elo = round(_elo(0.5 + mde))
            if f"{mde:.3f}" != f"{mde_doc:.3f}" or abs(elo - elo_doc) > 1:
                mismatches.append((n, d, mde_doc, f"{mde:.3f}", elo_doc, elo))
    assert not mismatches, mismatches


def test_games_for_fourteen_elo_matches_the_text() -> None:
    """The prose quotes ~1,160 (one-sided) and ~1,470 (two-sided lower bound)."""
    sigma = math.sqrt((1 - 0.7) / 4)
    one_sided = math.ceil(((1.645 + 0.842) * sigma / 0.02) ** 2)
    two_sided = math.ceil(((1.96 + 0.842) * sigma / 0.02) ** 2)
    text = DOC.read_text()
    assert f"{one_sided:,}" in text, one_sided
    assert f"{two_sided:,}" in text, two_sided


@pytest.mark.parametrize("perturbed", ["0.233 (+175 Elo)", "0.999 (+175 Elo)"])
def test_negative_control_detects_a_hand_edited_cell(perturbed: str) -> None:
    text = DOC.read_text().replace("0.233 (+175 Elo)", perturbed, 1)
    draw_rates, rows = parse_power_table(text)
    n, cells = rows[0]
    mde_doc, _ = cells[0]
    matches = f"{scoring.minimum_detectable_effect(n, draw_rates[0]):.3f}" == f"{mde_doc:.3f}"
    assert matches == (perturbed == "0.233 (+175 Elo)")
