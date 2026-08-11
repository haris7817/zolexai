# ZolexAI generation worker.
#
#   docker build -f apps/worker/Dockerfile .        (context = repo root)
#
# This image contains NO database driver and NO storage SDK. The worker reaches
# the platform only through the internal HTTP API, and every file it touches
# comes from a presigned URL scoped to one job. A compromised node therefore
# cannot read the database or enumerate another user's media.
#
# M2 will add a CUDA base image and model dependencies here. Nothing above this
# layer — the API, the schema, the frontend — has to change for that.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY apps/worker/pyproject.toml ./pyproject.toml
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p worker && touch worker/__init__.py && pip install .

COPY apps/worker/worker ./worker

RUN useradd --create-home --uid 10002 zolexai && chown -R zolexai:zolexai /app
USER zolexai

# No EXPOSE and no health endpoint: the worker listens on nothing. It pulls
# work and reports progress outbound, which is what lets it run behind NAT on
# any GPU provider without inbound connectivity.
#
# Liveness is observed centrally instead — the API tracks heartbeats, and any
# job whose lease expires is requeued regardless of why the worker went quiet.

CMD ["python", "-m", "worker.main"]
