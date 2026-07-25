"""Compatibility shim — all package metadata lives in ``pyproject.toml``.

This file is intentionally minimal. It is retained only because
``dockerfiles/Dockerfile.base`` copies ``setup.py`` during the image build; the
version, dependencies, and extras are defined once, in ``pyproject.toml``.
"""

from setuptools import setup

setup()
