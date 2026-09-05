#!/usr/bin/env bash
# Installs the ZolexAI worker + VPS tunnel on the GPU node under supervisord.
# Safe to run before the VPS has authorised the tunnel key or the token is
# set: the tunnel retries, the worker waits for the tunnel's health check.
set -euo pipefail

ENVF=/workspace/zolexai/.env.gpu-worker
if [ ! -f "$ENVF" ]; then
cat > "$ENVF" <<'ENV'
# ZolexAI GPU worker — ltx-6000-2 (Vast, RTX PRO 6000, non-persistent). Token appended separately.
WORKER_NAME=ltx-6000-2
RUNTIME=ltx
RUNTIMES=ltx,ltx_comfy,character_replacement
API_BASE_URL=http://127.0.0.1:18000
USE_REDIS_WAKEUP=false
MAX_CONCURRENCY=1
LTX_REPO_DIR=/workspace/ltx2-benchmark
LTX_QUANTIZATION=nvfp4-prequant
LTX_MAX_SECONDS=30
LTX_COMFY_BASE_URL=http://127.0.0.1:8189
LTX_COMFY_MODELS_DIR=/workspace/ComfyUI-ltx/models
LTX_COMFY_INPUT_DIR=/workspace/ComfyUI-ltx/input
ENABLE_H3=false
WORKSPACE_DIR=/workspace/zolexai-worker
REQUEST_TIMEOUT_SECONDS=20
DOWNLOAD_TIMEOUT_SECONDS=300
UPLOAD_TIMEOUT_SECONDS=900
LOG_FORMAT=json
ENV
chmod 600 "$ENVF"
echo "wrote $ENVF (no token yet)"
else
echo "$ENVF exists; left as is"
fi
mkdir -p /workspace/zolexai-worker

cat > /opt/supervisor-scripts/zolexai-tunnel.sh <<'SH'
#!/usr/bin/env bash
# SSH tunnel to the VPS API: local 127.0.0.1:18000 -> VPS 127.0.0.1:8100.
# supervisord restarts it when it drops; ExitOnForwardFailure makes a refused
# forward a clean exit rather than a silent half-tunnel.
exec ssh \
  -i /root/.ssh/zolexai_prod_tunnel \
  -N \
  -L 127.0.0.1:18000:127.0.0.1:8100 \
  -o BatchMode=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=accept-new \
  zolexai-gpu-tunnel@76.13.27.242
SH
chmod +x /opt/supervisor-scripts/zolexai-tunnel.sh

cat > /opt/supervisor-scripts/zolexai-worker.sh <<'SH'
#!/usr/bin/env bash
# The ZolexAI worker. Waits for the tunnel's health check (supervisord has no
# dependency ordering) and for a token in the env file, then execs the worker
# in its own venv so supervisord signals reach the real process.
set -euo pipefail
ENVF=/workspace/zolexai/.env.gpu-worker
until curl -sf --max-time 5 http://127.0.0.1:18000/api/v1/health >/dev/null 2>&1; do
  echo "waiting for the VPS tunnel on 127.0.0.1:18000 ..."; sleep 10
done
until grep -q '^WORKER_API_TOKEN=.\+' "$ENVF"; do
  echo "waiting for WORKER_API_TOKEN in $ENVF ..."; sleep 30
done
set -a; source "$ENVF"; set +a
cd /workspace/zolexai/apps/worker
exec .venv/bin/python -m worker.main
SH
chmod +x /opt/supervisor-scripts/zolexai-worker.sh

cat > /etc/supervisor/conf.d/zolexai-tunnel.conf <<'CONF'
[program:zolexai-tunnel]
command=/opt/supervisor-scripts/zolexai-tunnel.sh
autostart=true
autorestart=true
startsecs=5
startretries=1000000
stopasgroup=true
killasgroup=true
stdout_logfile=/tmp/zolexai-tunnel.log
stderr_logfile=/tmp/zolexai-tunnel.log
redirect_stderr=true
CONF

cat > /etc/supervisor/conf.d/zolexai-worker.conf <<'CONF'
[program:zolexai-worker]
command=/opt/supervisor-scripts/zolexai-worker.sh
autostart=true
autorestart=true
startsecs=20
stopwaitsecs=330
stopasgroup=true
killasgroup=true
stdout_logfile=/tmp/zolexai-ltx-worker.log
stderr_logfile=/tmp/zolexai-ltx-worker.log
redirect_stderr=true
CONF

supervisorctl reread >/dev/null
supervisorctl update >/dev/null
sleep 6
supervisorctl status | grep zolexai
echo "--- tunnel log ---"; tail -3 /tmp/zolexai-tunnel.log 2>/dev/null || true
echo "--- worker log ---"; tail -2 /tmp/zolexai-ltx-worker.log 2>/dev/null || true
echo NODE_WORKER_SETUP_DONE
