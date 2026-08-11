# ZolexAI API.
#
# Build context is the REPO ROOT, not apps/api — the workflow definitions live
# at the top level and are the API's source of truth for what it can run:
#   docker build -f apps/api/Dockerfile .
#
# Python 3.12 rather than 3.13: every dependency here ships prebuilt wheels for
# 3.12, so the image builds without a compiler toolchain. The code supports
# 3.11-3.13 and CI may run any of them.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Dependencies ─────────────────────────────────────────────────────────
# Copied and installed BEFORE the source so a code change does not invalidate
# the dependency layer — the difference between a 5-second and a 2-minute build.
COPY apps/api/pyproject.toml ./pyproject.toml
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p app && touch app/__init__.py && pip install .

# ── Application ──────────────────────────────────────────────────────────
COPY apps/api/app ./app
COPY apps/api/migrations ./migrations
COPY apps/api/alembic.ini ./alembic.ini

# Read at startup and validated before the app serves anything.
COPY workflow-definitions /workflow-definitions
ENV WORKFLOW_DEFINITIONS_DIR=/workflow-definitions

# Never run as root: a container escape should not land on a privileged user.
RUN useradd --create-home --uid 10001 zolexai && chown -R zolexai:zolexai /app
USER zolexai

EXPOSE 8000

# Liveness only — it must not fail because PostgreSQL blipped, or the
# orchestrator would restart every API container over a dependency wobble.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/v1/health/live')"

# One worker per container. Scale by adding containers, not processes: the API
# is stateless, so a load balancer in front of N containers is the whole story.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
