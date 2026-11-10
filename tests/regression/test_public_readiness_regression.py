"""Public-readiness gate: what must be true of the tracked tree before it is public.

Each check pins a finding from the 2026-11-10 public-release review
(``docs/audits/2026-11-10-public-release-review.md`` 3) rather than a hypothetical:

1.  **Personal e-mail addresses** appeared in four tracked files, one of them the
    Code of Conduct, whose private channel must not be a personal inbox.
2.  **The GCP project number and compute service-account address** appeared in five
    files. Project *ids* are not secrets; the project number, service name and
    region together derive the unauthenticated endpoint URL, so those are zero
    tolerance everywhere, archive included. **Bucket names** were code defaults
    (``inference_server.py``), which is a takeover vector: whoever registers the
    name serves the model. Zero tolerance outside ``*.example.*`` files and history.
3.  **Project-id mentions** are allowed in docs but ratcheted: the file count may
    only fall, and every exemption must name a file that still contains it.
4.  **Tracked binaries** over 100 KB need a stated reason (2.2 MB of ``.npz`` replays
    and a W&B dashboard JSON were tracked with none).
5.  **README URLs** must equal ``docs/public_links.toml``: six Hugging Face badges
    pointed at repositories that return 404.
6.  **CITATION.cff** must parse, carry the standard keys and agree with pyproject.
7.  **Lint-excluded directories** still have to byte-compile: "ungated" is not a
    licence to ship syntax errors.
8.  **Packaging metadata** must be present, and the ``all`` extra must not pull a
    ``git+`` dependency (it made ``pip install '.[all]'`` fail for every reader).
9.  **CLAUDE.md** must not state a 3.10 floor (pyproject says 3.11) and its footer
    date must not predate its newest entry.
10. **Forbidden claims** -- numbers and phrases the committed evidence does not
    support -- may not appear in maintained prose; README must open with a status
    and a non-affiliation statement.
11. **Absolute personal paths** (``/home/<name>``, ``C:\\Users\\<name>``) belong to
    the author's machines, not the repository.

House style follows ``test_repo_hygiene_regression.py`` and
``test_suite_wiring_regression.py``: every rule is a pure predicate over text or
paths; each predicate is run on planted input in
:class:`TestTheGateItselfDiscriminates` so it is proven to fail before anyone relies
on it; every allowlist entry needs a reason and fails when stale; and every tree
scan asserts it inspected more than zero files.

Data lives in ``public_readiness_manifest.json`` next to this file. Three checks
were ``xfail(strict=True)`` until the README, CLAUDE.md and CI wording landed;
they are ordinary tests now.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).with_name("public_readiness_manifest.json")
PUBLIC_LINKS_PATH = REPO_ROOT / "docs" / "public_links.toml"
README_PATH = REPO_ROOT / "README.md"
CLAUDE_MD_PATH = REPO_ROOT / "CLAUDE.md"
CITATION_PATH = REPO_ROOT / "CITATION.cff"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

#: This gate's own two files spell every forbidden string on purpose (controls and
#: patterns), so every tree scan skips exactly these and nothing else; a test below
#: pins the set so it cannot quietly widen into a general exemption.
_SELF_PATHS = frozenset(
    {
        Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
        MANIFEST_PATH.resolve().relative_to(REPO_ROOT).as_posix(),
    }
)

_BINARY_SNIFF_BYTES = 8192
_LARGE_BINARY_BYTES = 100 * 1024
_MIN_REASON_CHARS = 20
_README_STATUS_LINES = 40

# ---------------------------------------------------------------------------
# Manifest and tracked-file helpers
# ---------------------------------------------------------------------------


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _git_ls_files() -> list[str]:
    """Tracked paths, or skip when git is unavailable."""
    if shutil.which("git") is None:  # pragma: no cover - environment guard
        pytest.skip("git is not available")
    proc = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=False
    )
    paths = [p.decode("utf-8") for p in proc.stdout.split(b"\0") if p]
    if not paths:  # pragma: no cover - not a git checkout
        pytest.skip("not a git working tree")
    return paths


def _is_binary(path: Path) -> bool:
    """NUL in the first 8 KB -- the same sniff ``git diff`` uses."""
    try:
        with path.open("rb") as fh:
            return b"\0" in fh.read(_BINARY_SNIFF_BYTES)
    except OSError:  # pragma: no cover - vanished between ls-files and read
        return False


@dataclass(frozen=True)
class _TextFile:
    rel: str
    text: str


def _tracked_text_files() -> list[_TextFile]:
    out: list[_TextFile] = []
    for rel in _git_ls_files():
        path = REPO_ROOT / rel
        if rel in _SELF_PATHS or not path.is_file() or _is_binary(path):
            continue
        out.append(_TextFile(rel, path.read_text(encoding="utf-8", errors="replace")))
    assert out, "no tracked text files inspected -- the scan is vacuous"
    return out


def _excluded(rel: str, prefixes: tuple[str, ...] | list[str]) -> bool:
    return any(rel == p or rel.startswith(p) for p in prefixes)


def _allowlist_problems(name: str, entries: dict[str, str], still_matching: set[str]) -> list[str]:
    """Why an allowlist is unacceptable: a stale path, or a placeholder reason.

    ``still_matching`` is the set of tracked paths that currently trip the rule.
    An exemption for a path that no longer trips it is stale -- it hides the
    paths that really do -- and an exemption with no sentence of reasoning is a
    wired-off gate with a comment on it.
    """
    problems = []
    for rel, reason in sorted(entries.items()):
        if rel not in still_matching:
            problems.append(
                f"{name}[{rel!r}] is stale: the file no longer trips the rule "
                "(or no longer exists); delete the entry"
            )
        if len(reason.strip()) < _MIN_REASON_CHARS:
            problems.append(f"{name}[{rel!r}] needs a real reason, not {reason!r}")
    return problems


# ---------------------------------------------------------------------------
# 1. Personal e-mail addresses
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_EMAIL_SAFE_DOMAIN_SUFFIXES = (".invalid", "gserviceaccount.com")


def _personal_emails(text: str) -> set[str]:
    """Addresses that could belong to a person.

    ``noreply@`` senders, RFC 2606 ``example.`` domains, ``.invalid`` and service
    accounts are not people and are excluded.
    """
    found: set[str] = set()
    for match in _EMAIL_RE.findall(text):
        local, domain = match.rsplit("@", 1)
        domain_l = domain.lower()
        if local.lower().startswith("noreply"):
            continue
        if domain_l.startswith("example.") or ".example." in f".{domain_l}":
            continue
        if domain_l.endswith(_EMAIL_SAFE_DOMAIN_SUFFIXES):
            continue
        found.add(match)
    return found


class TestNoPersonalEmail:
    def test_no_personal_email_in_tracked_text(self) -> None:
        manifest = _manifest()
        allow = manifest["email_allowlist"]
        offenders: dict[str, set[str]] = {}
        for tf in _tracked_text_files():
            hits = _personal_emails(tf.text)
            if hits:
                offenders[tf.rel] = hits
        problems = _allowlist_problems("email_allowlist", allow, set(offenders))
        unlisted = {rel: sorted(h) for rel, h in offenders.items() if rel not in allow}
        assert not unlisted and not problems, (
            f"personal e-mail addresses in tracked files: {unlisted}; "
            "the Code of Conduct channel is GitHub Security Advisories, not an inbox. "
            f"Allowlist problems: {problems}"
        )


# ---------------------------------------------------------------------------
# 2. Zero tolerance: project number, SA address, bucket literals
# ---------------------------------------------------------------------------


def _forbidden_hits(text: str, patterns: list[str]) -> list[str]:
    return [p for p in patterns if re.search(p, text)]


def _bucket_literal_hits(text: str, literals: list[str]) -> list[str]:
    return [lit for lit in literals if re.search(rf"\b{re.escape(lit)}\b", text)]


_BUCKET_SCAN_EXCLUDED_PREFIXES = ("docs/archive/", "CHANGELOG.md", "kb/")


def _bucket_scan_skips(rel: str) -> bool:
    return _excluded(rel, _BUCKET_SCAN_EXCLUDED_PREFIXES) or ".example." in Path(rel).name


class TestZeroToleranceIdentifiers:
    def test_project_number_and_sa_address_appear_nowhere(self) -> None:
        """Archive included: redacting an identifier is not 'maintaining' a document."""
        patterns = _manifest()["forbidden_always"]
        assert patterns, "forbidden_always is empty -- the check is vacuous"
        offenders = {
            tf.rel: hits
            for tf in _tracked_text_files()
            if (hits := _forbidden_hits(tf.text, patterns))
        }
        assert not offenders, (
            f"project number / service-account address in tracked files: {offenders}. "
            "Together with the service name and region these derive the public endpoint URL."
        )

    def test_bucket_literals_only_in_examples_and_history(self) -> None:
        manifest = _manifest()
        literals = manifest["bucket_literals"]
        allow = manifest["bucket_literal_allowlist"]
        assert literals, "bucket_literals is empty -- the check is vacuous"
        inspected = 0
        offenders: dict[str, list[str]] = {}
        for tf in _tracked_text_files():
            if _bucket_scan_skips(tf.rel):
                continue
            inspected += 1
            hits = _bucket_literal_hits(tf.text, literals)
            if hits:
                offenders[tf.rel] = hits
        assert inspected > 0
        problems = _allowlist_problems("bucket_literal_allowlist", allow, set(offenders))
        unlisted = {rel: hits for rel, hits in offenders.items() if rel not in allow}
        assert not unlisted and not problems, (
            f"bucket names outside *.example.* / history: {unlisted}. A bucket name as a "
            "default is a takeover vector; use a placeholder or a required variable. "
            f"Allowlist problems: {problems}"
        )


# ---------------------------------------------------------------------------
# 3. Project-id ratchet
# ---------------------------------------------------------------------------


def _ratchet_problems(
    files_with_id: set[str], allowlist: dict[str, str], baseline: int
) -> list[str]:
    problems = _allowlist_problems("project_id_allowlist", allowlist, files_with_id)
    counted = sorted(files_with_id - set(allowlist))
    if len(counted) > baseline:
        problems.append(
            f"{len(counted)} files mention the project id, baseline is {baseline}: "
            f"{counted}. The count may only fall; lower project_id_baseline_count when it does."
        )
    return problems


class TestProjectIdRatchet:
    def test_project_id_file_count_does_not_grow(self) -> None:
        manifest = _manifest()
        project_id = manifest["project_id"]
        files_with_id = {tf.rel for tf in _tracked_text_files() if project_id in tf.text}
        problems = _ratchet_problems(
            files_with_id, manifest["project_id_allowlist"], manifest["project_id_baseline_count"]
        )
        assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# 4. Tracked binaries
# ---------------------------------------------------------------------------


def _large_binaries(paths: list[str], root: Path) -> set[str]:
    out = set()
    for rel in paths:
        path = root / rel
        if path.is_file() and path.stat().st_size > _LARGE_BINARY_BYTES and _is_binary(path):
            out.add(rel)
    return out


class TestTrackedBinaries:
    def test_large_binaries_need_a_reason(self) -> None:
        allow = _manifest()["binary_allowlist"]
        large = _large_binaries(_git_ls_files(), REPO_ROOT)
        problems = _allowlist_problems("binary_allowlist", allow, large)
        unlisted = sorted(large - set(allow))
        assert not unlisted and not problems, (
            f"tracked binaries over 100 KB without a reason: {unlisted}; {problems}"
        )


# ---------------------------------------------------------------------------
# 5. README links == docs/public_links.toml
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"\((https?://[^)\s]+)\)")


def _readme_urls(text: str) -> set[str]:
    return set(_URL_RE.findall(text))


def _link_manifest_problems(entries: list[dict]) -> list[str]:
    problems = []
    for i, entry in enumerate(entries):
        for key in ("url", "purpose", "verified_on"):
            if not str(entry.get(key, "")).strip():
                problems.append(f"links[{i}] is missing {key!r}")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(entry.get("verified_on", ""))):
            problems.append(f"links[{i}].verified_on must be YYYY-MM-DD")
    return problems


class TestReadmeLinks:
    def test_readme_urls_match_public_links_manifest(self) -> None:
        readme = _readme_urls(README_PATH.read_text(encoding="utf-8"))
        data = tomllib.loads(PUBLIC_LINKS_PATH.read_text(encoding="utf-8"))
        entries = data.get("links", [])
        problems = _link_manifest_problems(entries)
        listed = {e["url"] for e in entries if "url" in e}
        assert readme, "README has no URLs -- the comparison is vacuous"
        missing = sorted(readme - listed)
        stale = sorted(listed - readme)
        assert not missing and not stale and not problems, (
            f"README URLs not in docs/public_links.toml: {missing}; manifest URLs not in "
            f"README: {stale}; {problems}. Every external link carries a purpose and a "
            "verified_on date, because six Hugging Face badges pointed at 404s."
        )


# ---------------------------------------------------------------------------
# 6. CITATION.cff
# ---------------------------------------------------------------------------

_CITATION_KEYS = ("cff-version", "title", "authors", "version", "repository-code", "license")


def _citation_problems(cff: object, pyproject_version: str) -> list[str]:
    if not isinstance(cff, dict):
        return ["CITATION.cff did not parse to a mapping"]
    problems = [f"CITATION.cff is missing {key!r}" for key in _CITATION_KEYS if key not in cff]
    if str(cff.get("version")) != pyproject_version:
        problems.append(
            f"CITATION.cff version {cff.get('version')!r} != pyproject {pyproject_version!r}"
        )
    return problems


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


class TestCitation:
    def test_citation_parses_and_matches_pyproject(self) -> None:
        assert CITATION_PATH.is_file(), "CITATION.cff is missing"
        cff = yaml.safe_load(CITATION_PATH.read_text(encoding="utf-8"))
        problems = _citation_problems(cff, _pyproject()["project"]["version"])
        assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# 7. Lint-excluded directories still byte-compile
# ---------------------------------------------------------------------------


def _ruff_excluded_dirs(pyproject: dict) -> list[str]:
    raw = pyproject.get("tool", {}).get("ruff", {}).get("extend-exclude", [])
    return [entry.removeprefix("./").rstrip("/") for entry in raw]


def _compile_problems(paths: list[Path], root: Path) -> list[str]:
    problems = []
    for path in paths:
        try:
            compile(path.read_bytes(), str(path), "exec")
        except SyntaxError as exc:  # pragma: no cover - exercised by the control
            problems.append(f"{path.relative_to(root).as_posix()}: {exc.msg} (line {exc.lineno})")
    return problems


class TestExcludedDirsCompile:
    def test_every_py_under_a_lint_excluded_dir_compiles(self) -> None:
        dirs = _ruff_excluded_dirs(_pyproject())
        assert dirs, "ruff extend-exclude is empty; nothing to check (parser drift?)"
        targets = [
            REPO_ROOT / rel
            for rel in _git_ls_files()
            if rel.endswith(".py") and any(rel.startswith(f"{d}/") for d in dirs)
        ]
        assert targets, f"no tracked .py files under {dirs} -- vacuous"
        problems = _compile_problems(targets, REPO_ROOT)
        assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# 8. pyproject metadata
# ---------------------------------------------------------------------------

_PROJECT_KEYS = ("readme", "license", "authors", "urls", "classifiers", "keywords")


def _metadata_problems(pyproject: dict) -> list[str]:
    project = pyproject.get("project", {})
    problems = [f"[project] is missing {key!r}" for key in _PROJECT_KEYS if key not in project]
    if "license" in project and not isinstance(project["license"], str):
        problems.append("[project].license must be an SPDX string, not a table")
    return problems


def _git_extras_in_all(pyproject: dict) -> list[str]:
    """Extras reachable from ``all`` whose requirements point at a git URL."""
    extras = pyproject.get("project", {}).get("optional-dependencies", {})
    name = pyproject.get("project", {}).get("name", "")
    offenders = []
    for req in extras.get("all", []):
        match = re.fullmatch(rf"{re.escape(name)}\[([\w-]+)\]", req.strip())
        if match is None:
            if "git+" in req:
                offenders.append(req)
            continue
        extra = match.group(1)
        if any("git+" in dep for dep in extras.get(extra, [])):
            offenders.append(f"{extra} (via all)")
    return offenders


class TestPyprojectMetadata:
    def test_project_table_carries_public_metadata(self) -> None:
        problems = _metadata_problems(_pyproject())
        assert not problems, "\n".join(problems)

    def test_all_extra_has_no_git_dependency(self) -> None:
        """``eval`` pins ``langfuse-eval-harness`` to a git SHA; ``all`` must not pull it.

        It did until P2/P3: ``pip install -e '.[all]'`` failed for every reader on
        that one line. The extra itself may keep the git pin; ``all`` may not.
        """
        offenders = _git_extras_in_all(_pyproject())
        assert not offenders, f"`all` pulls git dependencies: {offenders}"


# ---------------------------------------------------------------------------
# 9. CLAUDE.md
# ---------------------------------------------------------------------------

_PY310_RE = re.compile(r"3\.10")
_DATE_RE = re.compile(r"\b(2026-\d{2}-\d{2})\b")
_ALLOW_MARKER = "doc-sync: allow"
_FOOTER_RE = re.compile(r"Last updated:\s*(\d{4}-\d{2}-\d{2})")


def _py310_lines(text: str) -> list[int]:
    return [
        i
        for i, line in enumerate(text.splitlines(), start=1)
        if _PY310_RE.search(line) and _ALLOW_MARKER not in line
    ]


def _footer_date_problems(text: str) -> list[str]:
    dates = _DATE_RE.findall(text)
    footer = _FOOTER_RE.search(text)
    if footer is None:
        return ["no 'Last updated: YYYY-MM-DD' footer"]
    if not dates:
        return ["no 2026-MM-DD dates found (parser drift?)"]
    newest = max(dates)
    if footer.group(1) < newest:
        return [f"footer says {footer.group(1)} but the newest entry is dated {newest}"]
    return []


class TestClaudeMd:
    def test_no_python_310_floor(self) -> None:
        lines = _py310_lines(CLAUDE_MD_PATH.read_text(encoding="utf-8"))
        assert not lines, (
            f"CLAUDE.md mentions 3.10 on lines {lines}; pyproject requires >=3.11. "
            f"Append '<!-- {_ALLOW_MARKER} -->' on a line that must keep it."
        )

    def test_footer_date_is_not_older_than_the_newest_entry(self) -> None:
        problems = _footer_date_problems(CLAUDE_MD_PATH.read_text(encoding="utf-8"))
        assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# 10. Forbidden claims and README status
# ---------------------------------------------------------------------------


def _claim_hits(text: str, patterns: list[str]) -> list[str]:
    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


def _readme_status_problems(text: str) -> list[str]:
    head = "\n".join(text.splitlines()[:_README_STATUS_LINES])
    problems = []
    if "Status" not in head:
        problems.append(f"README's first {_README_STATUS_LINES} lines carry no 'Status'")
    if "not affiliated" not in head.lower():
        problems.append(f"README's first {_README_STATUS_LINES} lines carry no 'not affiliated'")
    return problems


class TestForbiddenClaims:
    def test_no_forbidden_claims_in_maintained_prose(self) -> None:
        manifest = _manifest()
        patterns = manifest["forbidden_claims"]
        exclusions = manifest["claim_scan_exclusions"]
        assert patterns
        inspected = 0
        offenders: dict[str, list[str]] = {}
        for tf in _tracked_text_files():
            if _excluded(tf.rel, tuple(exclusions)):
                continue
            inspected += 1
            hits = _claim_hits(tf.text, patterns)
            if hits:
                offenders[tf.rel] = hits
        assert inspected > 0
        assert not offenders, (
            f"claims the committed evidence does not support: {offenders}. "
            "Quote a measurement from docs/evidence/ or drop the number."
        )

    def test_claim_scan_exclusions_are_justified(self) -> None:
        exclusions = _manifest()["claim_scan_exclusions"]
        bad = [k for k, v in exclusions.items() if len(v.strip()) < _MIN_REASON_CHARS]
        missing = [k for k in exclusions if not (REPO_ROOT / k).exists()]
        assert not bad and not missing, f"unjustified: {bad}; nonexistent: {missing}"

    def test_readme_opens_with_status_and_non_affiliation(self) -> None:
        problems = _readme_status_problems(README_PATH.read_text(encoding="utf-8"))
        assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# 11. Absolute personal paths
# ---------------------------------------------------------------------------

_USER_PATH_RES = (
    re.compile(r"C:\\\\?Users\\\\?(?!<)[A-Za-z]"),
    re.compile(r"/home/(?!user/|<)[a-z][a-z0-9_-]*/"),
    re.compile(r"/Users/(?!<)[a-z][a-z0-9_-]*/"),
)


def _user_path_hits(text: str) -> list[str]:
    return sorted({m.group(0) for rx in _USER_PATH_RES for m in rx.finditer(text)})


class TestNoPersonalPaths:
    def test_no_absolute_user_paths_outside_archive(self) -> None:
        allow = _manifest()["user_path_allowlist"]
        offenders: dict[str, list[str]] = {}
        for tf in _tracked_text_files():
            if _excluded(tf.rel, ("docs/archive/",)):
                continue
            hits = _user_path_hits(tf.text)
            if hits:
                offenders[tf.rel] = hits
        problems = _allowlist_problems("user_path_allowlist", allow, set(offenders))
        unlisted = {rel: h for rel, h in offenders.items() if rel not in allow}
        assert not unlisted and not problems, (
            f"absolute personal paths: {unlisted}; use $HOME / <user> placeholders. {problems}"
        )


# ---------------------------------------------------------------------------
# Negative controls: every predicate must fail on planted input
# ---------------------------------------------------------------------------


class TestTheGateItselfDiscriminates:
    def test_self_exclusion_is_exactly_this_gate(self) -> None:
        """The only files a scan may skip wholesale are this test and its manifest."""
        assert _SELF_PATHS == {
            "tests/regression/test_public_readiness_regression.py",
            "tests/regression/public_readiness_manifest.json",
        }
        rels = {tf.rel for tf in _tracked_text_files()}
        assert not (rels & _SELF_PATHS)
        assert "README.md" in rels and "pyproject.toml" in rels, "scan lost ordinary files"

    def test_personal_email_is_found_and_safe_senders_are_not(self) -> None:
        assert _personal_emails("mail someone.real@gmail.com now") == {"someone.real@gmail.com"}
        assert not _personal_emails(
            "noreply@github.com user@example.com x@sub.example.org "
            "bot@service.invalid 123-compute@developer.gserviceaccount.com"
        )

    def test_project_number_and_sa_patterns_fire(self) -> None:
        patterns = _manifest()["forbidden_always"]
        assert _forbidden_hits("project 1019219508001 here", patterns)
        assert _forbidden_hits("sa: 123456789012-compute@developer.gserviceaccount.com", patterns)
        assert _forbidden_hits("x@hrmtrain.iam.gserviceaccount.com", patterns)
        assert not _forbidden_hits("sa-name@project.iam.gserviceaccount.com", patterns)

    def test_bucket_literal_is_word_bounded(self) -> None:
        literals = _manifest()["bucket_literals"]
        assert _bucket_literal_hits("gs://civ6az-training/models", literals) == ["civ6az-training"]
        assert not _bucket_literal_hits("civ6az-trainingfoo and alphazero-training", literals)

    def test_bucket_scan_skips_examples_and_history_only(self) -> None:
        assert _bucket_scan_skips("configs/vertex_training.example.yaml")
        assert _bucket_scan_skips("docs/archive/old.md")
        assert not _bucket_scan_skips("configs/vertex_training.yaml")
        assert not _bucket_scan_skips("alpha_zero/serving/inference_server.py")

    def test_ratchet_rejects_growth_and_stale_exemptions(self) -> None:
        assert _ratchet_problems({"a.md", "b.md"}, {}, baseline=1)
        assert _ratchet_problems({"a.md"}, {"gone.md": "a perfectly adequate reason here"}, 5)
        assert _ratchet_problems({"a.md"}, {"a.md": "todo"}, 5)
        assert not _ratchet_problems({"a.md"}, {"a.md": "a perfectly adequate reason here"}, 0)

    def test_large_binary_is_detected_and_large_text_is_not(self, tmp_path: Path) -> None:
        (tmp_path / "blob.bin").write_bytes(b"\0" * (_LARGE_BINARY_BYTES + 1))
        (tmp_path / "big.txt").write_text("x" * (_LARGE_BINARY_BYTES + 1))
        (tmp_path / "small.bin").write_bytes(b"\0" * 10)
        assert _large_binaries(["blob.bin", "big.txt", "small.bin"], tmp_path) == {"blob.bin"}

    def test_readme_url_extraction_and_manifest_shape(self) -> None:
        text = "[![b](https://img.shields.io/x.svg)](https://example.com/a) plain https://no.paren"
        assert _readme_urls(text) == {"https://img.shields.io/x.svg", "https://example.com/a"}
        assert _link_manifest_problems([{"url": "https://a", "purpose": "", "verified_on": "x"}])
        assert not _link_manifest_problems(
            [{"url": "https://a", "purpose": "badge", "verified_on": "2026-11-10"}]
        )

    def test_citation_problems_fire(self) -> None:
        good = dict.fromkeys(_CITATION_KEYS, "x") | {"version": "1.8.0"}
        assert not _citation_problems(good, "1.8.0")
        assert _citation_problems(good, "1.9.0")
        assert _citation_problems({k: v for k, v in good.items() if k != "license"}, "1.8.0")
        assert _citation_problems("not a mapping", "1.8.0")

    def test_syntax_error_in_excluded_dir_is_caught(self, tmp_path: Path) -> None:
        bad = tmp_path / "broken.py"
        bad.write_text("def f(:\n    pass\n")
        good = tmp_path / "fine.py"
        good.write_text("x = 1\n")
        assert _compile_problems([bad, good], tmp_path) and not _compile_problems([good], tmp_path)

    def test_excluded_dirs_parse_from_pyproject(self) -> None:
        dirs = _ruff_excluded_dirs({"tool": {"ruff": {"extend-exclude": ["./examples", "x/"]}}})
        assert dirs == ["examples", "x"]
        assert "examples" in _ruff_excluded_dirs(_pyproject()), "live config lost ./examples?"

    def test_metadata_problems_fire(self) -> None:
        full = {"project": dict.fromkeys(_PROJECT_KEYS, "v")}
        assert not _metadata_problems(full)
        assert _metadata_problems({"project": {k: "v" for k in _PROJECT_KEYS if k != "urls"}})
        table = {"project": {**full["project"], "license": {"text": "MIT"}}}
        assert _metadata_problems(table)

    def test_git_extra_reachable_from_all_is_found(self) -> None:
        py = {
            "project": {
                "name": "pkg",
                "optional-dependencies": {
                    "eval": ["thing @ git+https://example.com/r@abc"],
                    "web": ["fastapi"],
                    "all": ["pkg[eval]", "pkg[web]"],
                },
            }
        }
        assert _git_extras_in_all(py) == ["eval (via all)"]
        py["project"]["optional-dependencies"]["all"] = ["pkg[web]"]
        assert not _git_extras_in_all(py)

    def test_py310_lines_respect_the_allow_marker(self) -> None:
        assert _py310_lines("ok\npython 3.10+\n") == [2]
        assert not _py310_lines(f"python 3.10+ <!-- {_ALLOW_MARKER} -->\n")

    def test_footer_date_ordering(self) -> None:
        assert _footer_date_problems("| 2026-09-04 | x |\n*Last updated: 2026-07-25*")
        assert not _footer_date_problems("| 2026-09-04 | x |\n*Last updated: 2026-09-04*")

    def test_claim_patterns_fire(self) -> None:
        patterns = _manifest()["forbidden_claims"]
        for text in (
            "+31.4 Elo",
            "+31.4  Elo",
            "13x faster",
            "5\u201310x",
            "5-10x",
            "the LLM eval gate",
            "Hub integration complete",
        ):
            assert _claim_hits(text, patterns), text
        assert not _claim_hits("a 113x number and 5 to 10 times", patterns)

    def test_readme_status_predicate(self) -> None:
        assert _readme_status_problems("# T\n" * 50)
        assert not _readme_status_problems(
            "# T\n\n## Status\n\nThis project is not affiliated with 2K.\n"
        )
        buried = "x\n" * _README_STATUS_LINES + "Status not affiliated"
        assert _readme_status_problems(buried), "status buried past line 40 must not count"

    def test_user_path_patterns(self) -> None:
        assert _user_path_hits("cd /home/alice/proj") == ["/home/alice/"]
        assert _user_path_hits(r"C:\Users\bob\x") and _user_path_hits("/Users/carol/x")
        assert not _user_path_hits(r"/home/user/x /home/<name>/y C:\Users\<user>\z /Users/<me>/w")

    def test_allowlist_problems_fire(self) -> None:
        assert _allowlist_problems("x", {"gone": "a perfectly adequate reason here"}, set())
        assert _allowlist_problems("x", {"here": "todo"}, {"here"})
        assert not _allowlist_problems("x", {"here": "a perfectly adequate reason here"}, {"here"})
