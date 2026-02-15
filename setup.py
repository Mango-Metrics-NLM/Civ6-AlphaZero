"""Thin shim for editable installs.

All package metadata is defined in pyproject.toml.
This file exists only for backwards compatibility with
``pip install -e .`` on older pip versions.
"""

from setuptools import setup

setup()
