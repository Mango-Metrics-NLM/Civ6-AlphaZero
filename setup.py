"""Setup script for Civ6 AlphaZero."""

from setuptools import setup, find_packages

setup(
    name="alpha_zero",
    version="0.2.0",
    description="AlphaZero-based AI for Civilization VI with RAG integration",
    packages=find_packages(),
    python_requires=">=3.11",
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
            "pytest-asyncio>=0.21.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
        "rag": [
            "chromadb>=0.4.0",
            "faiss-cpu>=1.7.0",
            "openai>=1.0.0",
            "sentence-transformers>=2.2.0",
        ],
        "rag-full": [
            "chromadb>=0.4.0",
            "faiss-cpu>=1.7.0",
            "openai>=1.0.0",
            "sentence-transformers>=2.2.0",
            "pinecone-client>=2.0.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
            "chromadb>=0.4.0",
            "faiss-cpu>=1.7.0",
            "openai>=1.0.0",
            "sentence-transformers>=2.2.0",
        ],
    },
)
