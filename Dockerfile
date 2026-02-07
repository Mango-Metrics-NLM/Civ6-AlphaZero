# Civ6 AlphaZero Training Container
# Supports both self-play and training workers via --mode flag
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="civ6az" \
      description="Civ6 AlphaZero self-play and training worker"

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
# Use slim requirements (no RAG/dev tools) for smaller container
COPY requirements-container.txt .
RUN pip install --no-cache-dir -r requirements-container.txt

# Copy the alpha_zero package
COPY alpha_zero/ alpha_zero/

# Copy setup files if they exist (for package install)
COPY setup.py pyproject.toml* ./
RUN pip install --no-cache-dir -e . 2>/dev/null || true

# Entrypoint routes to correct worker based on --mode flag
ENTRYPOINT ["python", "-m", "alpha_zero.cloud.gcp.entrypoint"]
CMD ["--mode=selfplay", "--dry_run", "--worker_id=0", "--model_version=0", "--iteration=0", "--games=1"]
