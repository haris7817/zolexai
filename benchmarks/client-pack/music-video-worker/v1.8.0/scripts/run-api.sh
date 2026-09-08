#!/usr/bin/env sh
set -eu

exec uvicorn zolex_music_worker.api:app --host "${ZOLEX_API_HOST:-0.0.0.0}" --port "${ZOLEX_API_PORT:-8080}"

