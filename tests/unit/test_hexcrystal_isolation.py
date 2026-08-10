"""Dependency-isolation guard for the hexcrystal subpackage.

``alpha_zero.hexcrystal`` promises to depend on torch and the standard library
and nothing else, so it can be vendored or reused outside this repository. That
promise is easy to break by accident: a single convenience import of numpy, or a
helper pulled from ``alpha_zero.core``, silently reintroduces the whole training
stack.

These tests enforce the contract from the outside. They run the import in a
*subprocess* with an import blocker installed on ``sys.meta_path``, because the
parent pytest process has already imported numpy (via conftest) and cannot
un-import it.

Note on the blocker: it must raise ``ModuleNotFoundError``, not a bare
``ImportError``. Torch guards its own optional numpy import with
``except ModuleNotFoundError``, so a plain ``ImportError`` propagates out of
``import torch`` itself and the test would fail for the wrong reason.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Third-party packages the subpackage must never require.
_FORBIDDEN = ("numpy", "gymnasium")

_BLOCKER_PREAMBLE = """
import sys, importlib.abc

BLOCKED = {blocked!r}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            # ModuleNotFoundError specifically: torch catches this subclass when
            # probing for numpy, whereas a bare ImportError escapes and breaks
            # `import torch` before the code under test ever runs.
            raise ModuleNotFoundError(f"blocked by test: {{fullname}}")
        return None


sys.meta_path.insert(0, _Blocker())
"""


def _run_isolated(
    body: str, blocked: tuple[str, ...] = _FORBIDDEN
) -> subprocess.CompletedProcess[str]:
    """Execute *body* in a subprocess with *blocked* packages unimportable."""
    script = _BLOCKER_PREAMBLE.format(blocked=set(blocked)) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_subpackage_imports_without_numpy_or_gymnasium() -> None:
    """The subpackage imports cleanly when numpy and gymnasium are unavailable."""
    result = _run_isolated(
        """
        import alpha_zero.hexcrystal as hx

        assert hx.HexSpatialEncoder is not None
        assert hx.PatchEmbed is not None
        print("IMPORT_OK")
        """
    )

    assert result.returncode == 0, (
        f"isolated import failed.\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "IMPORT_OK" in result.stdout


def test_forward_pass_without_numpy_or_gymnasium() -> None:
    """A real forward pass runs in the isolated environment.

    Importing is not enough — a lazily-imported dependency would only surface
    when tensors actually flow through the module.
    """
    result = _run_isolated(
        """
        import torch
        from alpha_zero.hexcrystal import HexEncoderConfig, HexSpatialEncoder

        config = HexEncoderConfig(
            in_channels=13, spatial_size=8, embed_dim=64, num_output_tokens=4
        )
        encoder = HexSpatialEncoder(config)
        out = encoder(torch.randn(2, 13, 8, 8))

        assert tuple(out.shape) == (2, 4, 64), out.shape
        assert torch.isfinite(out).all()
        print("FORWARD_OK")
        """
    )

    assert result.returncode == 0, (
        f"isolated forward pass failed.\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "FORWARD_OK" in result.stdout


def test_forbidden_modules_absent_from_sys_modules() -> None:
    """Nothing in the import graph pulls the forbidden packages in.

    Belt-and-braces alongside the meta-path blocker: this would also catch a
    dependency that imports successfully but should not be there at all.
    """
    result = _run_isolated(
        """
        import sys
        import alpha_zero.hexcrystal  # noqa: F401

        leaked = sorted(m for m in ("numpy", "gymnasium") if m in sys.modules)
        print("LEAKED:" + ",".join(leaked))
        """
    )

    assert result.returncode == 0, result.stderr
    assert "LEAKED:\n" in result.stdout or result.stdout.strip().endswith("LEAKED:"), (
        f"forbidden modules were imported: {result.stdout.strip()}"
    )


def test_root_package_does_not_import_training_stack() -> None:
    """``import alpha_zero`` alone must not pull numpy or gymnasium.

    This is what makes the isolated subpackage import possible: the root
    package resolves its public API lazily (PEP 562).
    """
    result = _run_isolated(
        """
        import alpha_zero

        assert alpha_zero.__version__
        print("ROOT_OK")
        """
    )

    assert result.returncode == 0, (
        f"root import pulled a blocked dependency.\n--- stderr ---\n{result.stderr}"
    )
    assert "ROOT_OK" in result.stdout


@pytest.mark.parametrize(
    "module",
    [
        "alpha_zero.hexcrystal.hex_conv",
        "alpha_zero.hexcrystal.crystalline_layers",
        "alpha_zero.hexcrystal.hex_encoder",
        "alpha_zero.hexcrystal.encoder_registry",
        "alpha_zero.hexcrystal.patch_embed",
    ],
)
def test_each_module_imports_in_isolation(module: str) -> None:
    """Every module in the subpackage is individually torch-only."""
    result = _run_isolated(
        f"""
        import {module}  # noqa: F401
        print("OK")
        """
    )

    assert result.returncode == 0, (
        f"{module} is not dependency-isolated.\n--- stderr ---\n{result.stderr}"
    )
