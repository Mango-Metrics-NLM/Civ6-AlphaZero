#!/usr/bin/env python3
"""Render ``.github/rulesets/*.json`` into request bodies for the GitHub REST API.

Used by ``scripts/maintainer/branch_cutover.sh``; never calls the API itself.

The committed files are the *final* state of each ruleset, in the shape GitHub's
own export produces (``name``, ``target``, ``enforcement``, ``bypass_actors``,
``conditions``, ``rules``; github/docs ``.github/rulesets/README.md``). Two
things are filled in or removed at apply time:

* the maintainer's numeric user id, which the files carry as the sentinel
  ``{{MAINTAINER_USER_ID}}`` so no personal id is committed. A sentinel that
  survives rendering is refused here, and is a string where the API wants an
  integer, so a raw file POSTed by hand is rejected by GitHub rather than
  applied with a wrong actor;
* ``Branch Direction``, which cut-over step 4 omits (the check does not exist
  until T6 is merged into ``dev``) and step 5 adds back with a ``PUT``;
* ``Release Evidence`` (``main`` only), which steps 4 and 5 omit and step 9a
  (``add-release-evidence``, after T14 is merged into ``dev``) adds back.

Every required status check carries ``integration_id`` 15368, the GitHub
Actions app (confirmed from this repository's own check runs:
``app.id=15368, slug=github-actions``). Without it a ruleset context is also
satisfied by a commit *status* of the same name, which any token with
``statuses: write`` -- the agent's Write PAT included -- can post.

``RepositoryRole`` actor id 5 is the repository Admin role.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SENTINEL = "{{MAINTAINER_USER_ID}}"
BRANCH_DIRECTION = "Branch Direction"
RULESET_DIR = Path(__file__).resolve().parents[2] / ".github" / "rulesets"
#: Cut-over step 4 POSTs in this order (S3.7).
CUTOVER_ORDER = ("dev", "qa", "main", "qa-main-merge", "release-tags", "legacy-name")
#: Step 5 PUTs Branch Direction back into these.
BRANCH_DIRECTION_RULESETS = ("dev", "qa", "main")


class RulesetError(ValueError):
    """A ruleset cannot be rendered or located."""


def load(name: str, directory: Path = RULESET_DIR) -> dict[str, Any]:
    path = directory / f"{name}.json"
    if not path.is_file():
        raise RulesetError(f"no ruleset file {path}")
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return doc


def render(
    doc: dict[str, Any], *, maintainer_id: int, without_checks: Iterable[str] = ()
) -> dict[str, Any]:
    """A request body: the sentinel replaced, ``without_checks`` dropped."""
    if isinstance(maintainer_id, bool) or not isinstance(maintainer_id, int) or maintainer_id <= 0:
        raise RulesetError(f"maintainer id must be a positive integer, got {maintainer_id!r}")
    out = copy.deepcopy(doc)
    for actor in out.get("bypass_actors", []):
        if actor.get("actor_id") == SENTINEL:
            actor["actor_id"] = maintainer_id
    dropped = set(without_checks)
    for rule in out.get("rules", []):
        if rule.get("type") == "required_status_checks":
            params = rule["parameters"]
            params["required_status_checks"] = [
                c for c in params["required_status_checks"] if c["context"] not in dropped
            ]
            if not params["required_status_checks"]:
                raise RulesetError(
                    f"{doc.get('name')}: dropping {sorted(dropped)} leaves no checks"
                )
    if SENTINEL in json.dumps(out):
        raise RulesetError(f"{doc.get('name')}: {SENTINEL} survived rendering")
    return out


def ruleset_id(listing: Sequence[dict[str, Any]], name: str) -> int:
    """The id of the one repository ruleset called ``name`` in a ``GET .../rulesets`` listing."""
    matches = [
        r
        for r in listing
        if r.get("name") == name and r.get("source_type", "Repository") == "Repository"
    ]
    if len(matches) != 1:
        raise RulesetError(f"expected exactly one ruleset named {name!r}, found {len(matches)}")
    return int(matches[0]["id"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("render", help="print a request body for one ruleset")
    rp.add_argument("name", choices=CUTOVER_ORDER)
    rp.add_argument("--maintainer-id", type=int, required=True)
    rp.add_argument("--without-check", action="append", default=[])
    rp.add_argument("--dir", type=Path, default=RULESET_DIR)
    ip = sub.add_parser("id-of", help="read a GET .../rulesets listing on stdin, print one id")
    ip.add_argument("name")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "render":
            body = render(
                load(args.name, args.dir),
                maintainer_id=args.maintainer_id,
                without_checks=args.without_check,
            )
            print(json.dumps(body, indent=2))
        else:
            print(ruleset_id(json.load(sys.stdin), args.name))
    except (RulesetError, json.JSONDecodeError) as exc:
        print(f"rulesets: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
