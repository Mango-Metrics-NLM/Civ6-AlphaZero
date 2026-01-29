"""Setup script for Civ6 AlphaZero."""

from setuptools import find_packages, setup

setup(
    name="alpha_zero",
    version="0.1.0",
    description="AlphaZero-based AI for Civilization VI",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=1.13.0",
        "numpy>=1.23.0",
        "gymnasium>=0.29.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
    },
)
