# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for code-indexer
#
# Stage 1: builder – install dependencies into a virtual environment.
# Stage 2: runtime – copy only the venv and app code for a minimal image.
#
# This approach keeps the final image small by excluding build tools
# (gcc, git, etc.) that are only needed at install time.
#
# Build:
#   docker build -f docker/Dockerfile -t code-indexer:latest .
#
# Run API server:
#   docker run -p 8000:8000 --env-file .env code-indexer:latest
#
# Run CLI:
#   docker run --rm -v $(pwd)/myrepo:/repo code-indexer:latest \
#     code-indexer index /repo
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS builder

WORKDIR /build

# Install system dependencies needed to compile C extensions
# (required by tree-sitter grammar packages and tokenizers).
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment so we can copy just the venv in stage 2.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the package definition first (for layer caching).
COPY pyproject.toml ./

# Install runtime dependencies.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e ".[sentence_transformer]" || \
    pip install --no-cache-dir -e "."

# Copy source code.
COPY src/ ./src/

# Install the package itself.
RUN pip install --no-cache-dir -e .

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime image
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS runtime

LABEL maintainer="code-indexer"
LABEL description="Production-ready codebase indexing for RAG and code generation"

WORKDIR /app

# Copy only the virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy application source.
COPY --from=builder /build/src ./src

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose the API port.
EXPOSE 8000

# Default: run the API server.
CMD ["code-indexer", "serve", "--host", "0.0.0.0", "--port", "8000"]
