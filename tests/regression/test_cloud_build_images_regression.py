"""Cloud Build producers and their consumers must agree, and nested substitutions must expand.

Two defects of one shape, both found by review on PR #132 after the identifier scrub:

1. ``_IMAGE`` embedded ``$PROJECT_ID`` inside a user substitution. Build triggers expand
   that, but a manual ``gcloud builds submit --config=cloudbuild.yaml`` does so only when
   ``options.dynamicSubstitutions`` is true, so the image reference kept the literal
   ``$PROJECT_ID`` and was invalid.
2. The builds pushed ``civ6az/training`` and ``civ6az/inference`` while every consumer
   (the production launcher, the inference deploy script, the Cloud Run and Kubernetes
   manifests, the container build wrapper) pulled ``alphazero-training`` /
   ``alphazero-inference``. Following build-then-launch pulled a different repository.

Both predicates are pure functions over text so the negative controls drive the same code.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

PRODUCERS = (
    "cloudbuild.yaml",
    "dockerfiles/cloudbuild.yaml",
    "infra/cloudbuild-gameplay.yaml",
)
#: (file, regex capturing the default image path) for every consumer that pulls an image
#: by default. Each regex captures ``<repo>/<image>`` after the registry host and project.
CONSUMERS = (
    ("scripts/launch_prod_v1.6.0.sh", r"docker\.pkg\.dev/\$\{PROJECT\}/([\w-]+/[\w-]+):"),
    ("cloud-run-inference.yaml", r"docker\.pkg\.dev/[\w$-]+/([\w-]+/[\w-]+):"),
    ("k8s/inference-deployment.yaml", r"docker\.pkg\.dev/[\w$-]+/([\w-]+/[\w-]+):"),
)
#: The inference deploy script builds its path from variables; these are its defaults.
DEPLOY_SCRIPT = "scripts/deploy_inference.sh"

# `<host>-docker.pkg.dev/<project>/<repo>/<image>` with the project and repo possibly
# written as substitutions (`$PROJECT_ID`, `${_AR_REPO}`).
_PRODUCED = re.compile(r"docker\.pkg\.dev/[^/\s'\"]+/([^/\s'\"]+)/([\w-]+)")


def needs_dynamic_substitutions(doc: dict) -> bool:
    """A user substitution whose value references another variable needs dynamic expansion."""
    subs = doc.get("substitutions") or {}
    return any("$" in str(v) for v in subs.values())


def dynamic_substitution_problems(doc: dict) -> list[str]:
    if needs_dynamic_substitutions(doc) and not (doc.get("options") or {}).get(
        "dynamicSubstitutions"
    ):
        nested = sorted(k for k, v in (doc.get("substitutions") or {}).items() if "$" in str(v))
        return [f"substitutions {nested} reference variables but dynamicSubstitutions is not true"]
    return []


def produced_images(text: str, ar_repo_default: str = "civ6az") -> set[str]:
    """Every `<repo>/<image>` pushed or named by a Cloud Build config."""
    images = set()
    for repo, image in _PRODUCED.findall(text):
        repo = ar_repo_default if repo.startswith("$") else repo
        images.add(f"{repo}/{image}")
    return images


def consumed_image(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def deploy_script_image(text: str) -> str:
    repo = re.search(r'REPO="\$\{CIV6AZ_ARTIFACT_REPO:-([\w-]+)\}"', text)
    image = re.search(r'IMAGE_REPO_NAME="\$\{CIV6AZ_INFERENCE_IMAGE:-([\w-]+)\}"', text)
    assert repo and image, "deploy script defaults not found -- the check would be vacuous"
    return f"{repo.group(1)}/{image.group(1)}"


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _all_produced() -> set[str]:
    produced: set[str] = set()
    for rel in PRODUCERS:
        produced |= produced_images(_read(rel))
    assert produced, "no produced images parsed -- the check would be vacuous"
    return produced


class TestCloudBuildConfigs:
    def test_nested_substitutions_are_expanded(self) -> None:
        problems = {
            rel: dynamic_substitution_problems(yaml.safe_load(_read(rel))) for rel in PRODUCERS
        }
        assert not {k: v for k, v in problems.items() if v}, problems

    def test_every_consumer_pulls_an_image_some_build_pushes(self) -> None:
        produced = _all_produced()
        missing = {}
        for rel, pattern in CONSUMERS:
            image = consumed_image(_read(rel), pattern)
            assert image, f"{rel}: default image not found -- the check would be vacuous"
            if image not in produced:
                missing[rel] = image
        deploy = deploy_script_image(_read(DEPLOY_SCRIPT))
        if deploy not in produced:
            missing[DEPLOY_SCRIPT] = deploy
        assert not missing, f"consumers pull images no build pushes: {missing}; built: {produced}"


class TestNegativeControls:
    """Each planted defect must be caught by the predicate the gate uses."""

    def test_nested_substitution_without_the_option_is_caught(self) -> None:
        doc = yaml.safe_load(
            "substitutions:\n  _IMAGE: us-central1-docker.pkg.dev/$PROJECT_ID/r/i\n"
            "options:\n  logging: CLOUD_LOGGING_ONLY\n"
        )
        assert dynamic_substitution_problems(doc)
        doc["options"]["dynamicSubstitutions"] = True
        assert not dynamic_substitution_problems(doc)

    def test_plain_substitutions_need_no_option(self) -> None:
        doc = yaml.safe_load("substitutions:\n  _TAG: v1\noptions: {}\n")
        assert not dynamic_substitution_problems(doc)

    def test_a_mismatched_consumer_is_caught(self) -> None:
        produced = produced_images(
            "images: ['us-central1-docker.pkg.dev/$PROJECT_ID/civ6az/alphazero-training:x']"
        )
        consumer = consumed_image(
            "image: us-central1-docker.pkg.dev/p/civ6az/training:latest",
            CONSUMERS[1][1],
        )
        assert consumer == "civ6az/training"
        assert consumer not in produced

    def test_substituted_repository_resolves_to_its_default(self) -> None:
        text = "- '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_AR_REPO}/alphazero-training:t'"
        assert produced_images(text) == {"civ6az/alphazero-training"}
