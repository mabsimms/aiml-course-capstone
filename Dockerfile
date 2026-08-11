ARG PYTHON_IMAGE=python:3.13.13-slim-bookworm

# Stage 1 - builder
FROM ${PYTHON_IMAGE} AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /uvx /usr/local/bin/

# Copy in dependency base
WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install and cache packages
RUN mkdir -p dependencies/wheels

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --extra serving --no-editable

# Step 2 - runtime
FROM ${PYTHON_IMAGE} AS runtime

# Copy in the core dependency set
RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --home-dir /app --create-home app

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GPU_MODE=force_cpu \
    GIT_PYTHON_REFRESH=quiet

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER app

EXPOSE 8001


HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3).read()" || exit 1

# MODEL_MANIFEST_PATH MUST be set
ENTRYPOINT ["uvicorn", "capstone.app:app", "--host", "0.0.0.0", "--port", "8001"]
