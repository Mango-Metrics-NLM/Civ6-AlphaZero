"""The inference deploy script renders the Cloud Run template instead of shipping placeholders.

Review on PR #132 found that ``scripts/deploy_inference.sh`` rewrote only the ``image:``
line, in place on the tracked manifest, so ``YOUR_GCP_PROJECT``, ``YOUR_BUCKET`` and the
service-account address reached ``gcloud run services replace`` verbatim, and that a
``CIV6AZ_INFERENCE_SERVICE`` override changed the URL lookup but not the deployed name.
``RENDER_ONLY=1`` exercises the render step without docker or gcloud.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "deploy_inference.sh"
TEMPLATE = REPO_ROOT / "cloud-run-inference.yaml"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="bash script; the deploy path runs on Linux/macOS",
)


def _run(tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    base = {k: v for k, v in os.environ.items() if not k.startswith("CIV6AZ_")}
    # Hide gcloud so the project fallback cannot pick up a developer's default.
    base["PATH"] = "/usr/bin:/bin"
    base.update({"RENDER_ONLY": "1", "TMPDIR": str(tmp_path), **env})
    return subprocess.run(
        ["bash", str(SCRIPT), "cloud-run", "v9"],
        capture_output=True,
        text=True,
        env=base,
        check=False,
    )


def test_render_substitutes_every_placeholder_and_leaves_the_template_alone(
    tmp_path: Path,
) -> None:
    before = TEMPLATE.read_bytes()
    result = _run(
        tmp_path,
        CIV6AZ_VERTEX_PROJECT="demo-proj",
        CIV6AZ_GCS_BUCKET="demo-bucket",
        CIV6AZ_INFERENCE_SERVICE="svc-x",
    )
    assert result.returncode == 0, result.stderr
    rendered = Path(result.stdout.strip().splitlines()[-1])
    text = rendered.read_text(encoding="utf-8")
    assert "YOUR_" not in text
    assert "\n  name: svc-x\n" in text
    assert "serviceAccountName: alphazero-inference@demo-proj.iam.gserviceaccount.com" in text
    assert "image: us-central1-docker.pkg.dev/demo-proj/civ6az/alphazero-inference:v9" in text
    assert "value: demo-bucket" in text
    assert TEMPLATE.read_bytes() == before, "the tracked template must never be edited"


def test_service_account_override_is_honoured(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        CIV6AZ_VERTEX_PROJECT="demo-proj",
        CIV6AZ_GCS_BUCKET="demo-bucket",
        CIV6AZ_INFERENCE_SERVICE_ACCOUNT="runner@other.iam.gserviceaccount.com",
    )
    assert result.returncode == 0, result.stderr
    text = Path(result.stdout.strip().splitlines()[-1]).read_text(encoding="utf-8")
    assert "serviceAccountName: runner@other.iam.gserviceaccount.com" in text


@pytest.mark.parametrize(
    ("env", "needle"),
    [
        ({"CIV6AZ_VERTEX_PROJECT": "demo-proj"}, "CIV6AZ_GCS_BUCKET"),
        ({"CIV6AZ_GCS_BUCKET": "demo-bucket"}, "CIV6AZ_VERTEX_PROJECT"),
    ],
)
def test_missing_required_settings_refuse_to_render(
    tmp_path: Path, env: dict[str, str], needle: str
) -> None:
    result = _run(tmp_path, **env)
    assert result.returncode != 0
    assert needle in result.stderr


def test_a_surviving_placeholder_refuses_to_render(tmp_path: Path) -> None:
    """Negative control: a template field the renderer does not know must stop the deploy."""
    planted = tmp_path / "template.yaml"
    planted.write_text(
        TEMPLATE.read_text(encoding="utf-8") + "\n# region: YOUR_REGION\n", encoding="utf-8"
    )
    result = _run(
        tmp_path,
        CIV6AZ_VERTEX_PROJECT="demo-proj",
        CIV6AZ_GCS_BUCKET="demo-bucket",
        CIV6AZ_CLOUD_RUN_MANIFEST=str(planted),
    )
    assert result.returncode == 3
    assert "YOUR_REGION" in result.stderr
