"""Tests for ``scripts/diff_audit_baseline.py`` (the pip-audit baseline gate).

Every exit-code branch is exercised against synthetic pip-audit JSON and a
synthetic baseline, so the gate is proven to discriminate before CI relies on
it: a finding outside the baseline fails, a covered one passes, an expired entry
passes-with-warning unless ``--strict``, and a malformed baseline is an input
error rather than a silent allow.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "diff_audit_baseline.py"
_spec = importlib.util.spec_from_file_location("diff_audit_baseline", _SCRIPT)
assert _spec is not None and _spec.loader is not None
dab = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = dab
_spec.loader.exec_module(dab)

TODAY = "2026-11-10"


def _audit(tmp_path: Path, name: str, *deps: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps({"dependencies": list(deps), "fixes": []}), encoding="utf-8")
    return str(path)


def _vuln(package: str, version: str, vuln_id: str, *aliases: str) -> dict:
    return {
        "name": package,
        "version": version,
        "vulns": [{"id": vuln_id, "fix_versions": ["9.9.9"], "aliases": list(aliases)}],
    }


def _baseline(tmp_path: Path, *entries: str) -> str:
    path = tmp_path / "baseline.toml"
    path.write_text("schema = 1\n" + "\n".join(entries), encoding="utf-8")
    return str(path)


def _entry(vuln_id: str, package: str, expires: str, reason: str = "accepted for test") -> str:
    return (
        f'[[ignore]]\nid = "{vuln_id}"\npackage = "{package}"\n'
        f'reason = "{reason}"\nexpires = {expires}\n'
    )


def test_clean_audit_and_empty_baseline_pass(tmp_path: Path) -> None:
    audit = _audit(tmp_path, "a.json", {"name": "numpy", "version": "2.0.0", "vulns": []})
    assert dab.main([audit, _baseline(tmp_path), "--today", TODAY]) == 0


def test_a_finding_outside_the_baseline_fails(tmp_path: Path, capsys) -> None:
    """Negative control for the whole gate: the unlisted finding must exit 1."""
    audit = _audit(tmp_path, "a.json", _vuln("urllib3", "1.26.0", "GHSA-aaaa-bbbb-cccc"))
    assert dab.main([audit, _baseline(tmp_path), "--today", TODAY]) == 1
    err = capsys.readouterr().err
    assert "new finding not in baseline" in err
    assert "GHSA-aaaa-bbbb-cccc" in err


def test_a_covered_finding_passes(tmp_path: Path, capsys) -> None:
    audit = _audit(tmp_path, "a.json", _vuln("urllib3", "1.26.0", "GHSA-aaaa-bbbb-cccc"))
    baseline = _baseline(tmp_path, _entry("GHSA-aaaa-bbbb-cccc", "urllib3", "2027-01-01"))
    assert dab.main([audit, baseline, "--today", TODAY]) == 0
    assert "accepted" in capsys.readouterr().out


def test_coverage_matches_on_an_alias_and_normalised_name(tmp_path: Path) -> None:
    """A baseline written with the CVE id and ``PyYAML`` spelling still covers."""
    audit = _audit(tmp_path, "a.json", _vuln("pyyaml", "5.3", "PYSEC-2020-176", "CVE-2020-14343"))
    baseline = _baseline(tmp_path, _entry("cve-2020-14343", "PyYAML", "2027-01-01"))
    assert dab.main([audit, baseline, "--today", TODAY]) == 0


def test_coverage_requires_the_package_to_match_too(tmp_path: Path) -> None:
    """The same advisory id on a different package is a different finding."""
    audit = _audit(tmp_path, "a.json", _vuln("requests", "2.0", "GHSA-aaaa-bbbb-cccc"))
    baseline = _baseline(tmp_path, _entry("GHSA-aaaa-bbbb-cccc", "urllib3", "2027-01-01"))
    assert dab.main([audit, baseline, "--today", TODAY]) == 1


def test_expired_entry_warns_but_passes_by_default(tmp_path: Path, capsys) -> None:
    audit = _audit(tmp_path, "a.json", _vuln("urllib3", "1.26.0", "GHSA-aaaa-bbbb-cccc"))
    baseline = _baseline(tmp_path, _entry("GHSA-aaaa-bbbb-cccc", "urllib3", "2026-01-01"))
    assert dab.main([audit, baseline, "--today", TODAY]) == 0
    assert "WARNING expired baseline entry" in capsys.readouterr().err


def test_expired_entry_fails_under_strict(tmp_path: Path, capsys) -> None:
    """Negative control for ``--strict``: the same input must now exit 1."""
    audit = _audit(tmp_path, "a.json", _vuln("urllib3", "1.26.0", "GHSA-aaaa-bbbb-cccc"))
    baseline = _baseline(tmp_path, _entry("GHSA-aaaa-bbbb-cccc", "urllib3", "2026-01-01"))
    assert dab.main([audit, baseline, "--today", TODAY, "--strict"]) == 1
    assert "ERROR expired baseline entry" in capsys.readouterr().err


def test_unused_entry_is_only_a_warning(tmp_path: Path, capsys) -> None:
    audit = _audit(tmp_path, "a.json", {"name": "numpy", "version": "2.0.0", "vulns": []})
    baseline = _baseline(tmp_path, _entry("GHSA-dead-dead-dead", "numpy", "2027-01-01"))
    assert dab.main([audit, baseline, "--today", TODAY]) == 0
    assert "matched no finding" in capsys.readouterr().err


def test_multiple_reports_are_merged(tmp_path: Path, capsys) -> None:
    clean = _audit(tmp_path, "container.json", {"name": "numpy", "version": "2.0.0", "vulns": []})
    dirty = _audit(tmp_path, "dev.json", _vuln("urllib3", "1.26.0", "GHSA-aaaa-bbbb-cccc"))
    assert dab.main([clean, dirty, _baseline(tmp_path), "--today", TODAY]) == 1
    assert "[dev.json]" in capsys.readouterr().err


def test_skipped_dependency_contributes_nothing(tmp_path: Path) -> None:
    audit = _audit(tmp_path, "a.json", {"name": "local-pkg", "skip_reason": "not on PyPI"})
    assert dab.main([audit, _baseline(tmp_path), "--today", TODAY]) == 0


def test_bare_list_report_shape_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps([_vuln("urllib3", "1.26.0", "GHSA-aaaa-bbbb-cccc")]))
    assert dab.main([str(path), _baseline(tmp_path), "--today", TODAY]) == 1


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(
            '[[ignore]]\nid = "GHSA-x"\npackage = "p"\nexpires = 2027-01-01\n', id="no-reason"
        ),
        pytest.param(
            '[[ignore]]\nid = "GHSA-x"\npackage = "p"\nreason = ""\nexpires = 2027-01-01\n',
            id="empty-reason",
        ),
        pytest.param('[[ignore]]\nid = "GHSA-x"\npackage = "p"\nreason = "r"\n', id="no-expiry"),
        pytest.param(
            '[[ignore]]\nid = "GHSA-x"\npackage = "p"\nreason = "r"\nexpires = "soon"\n',
            id="bad-expiry",
        ),
        pytest.param(
            '[[ignore]]\nid = "GHSA-x"\npackage = "p"\nreason = "r"\nexpires = 2027-01-01\n'
            'note = "extra"\n',
            id="unknown-key",
        ),
    ],
)
def test_malformed_baseline_entry_is_an_input_error(tmp_path: Path, entry: str, capsys) -> None:
    """A half-written exemption must not silently allow anything (exit 2, not 0)."""
    audit = _audit(tmp_path, "a.json", {"name": "numpy", "version": "2.0.0", "vulns": []})
    assert dab.main([audit, _baseline(tmp_path, entry), "--today", TODAY]) == 2
    assert "ERROR" in capsys.readouterr().err


def test_unreadable_report_is_an_input_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert dab.main([str(bad), _baseline(tmp_path), "--today", TODAY]) == 2


def test_missing_baseline_file_means_empty_baseline(tmp_path: Path) -> None:
    """No baseline file yet is a legitimate state: nothing is accepted."""
    audit = _audit(tmp_path, "a.json", _vuln("urllib3", "1.26.0", "GHSA-aaaa-bbbb-cccc"))
    assert dab.main([audit, str(tmp_path / "absent.toml"), "--today", TODAY]) == 1


def test_the_committed_baseline_parses() -> None:
    """The real file must load, or the CI step fails for a reason unrelated to findings."""
    real = _SCRIPT.parents[1] / ".security" / "pip-audit-baseline.toml"
    assert real.exists()
    dab.load_baseline(real)  # raises InputError on a malformed entry
