#!/usr/bin/env bash
#
# Keeps the GPU node's two hand-started processes alive.
#
# ## Why this exists
#
# The GPU box runs two bare processes that nothing supervises: the worker
# (`python -m worker.main`) and the SSH tunnel that carries its API traffic to
# the VPS. Neither is under supervisord — supervisord on this instance manages
# vast.ai's own services and has never known about ours.
#
# On 30 Aug 2026 the tunnel died. The worker stayed up and healthy and simply
# could not reach the API, so it logged `api_transport_retry`,
# `heartbeat_failed` and `claim_failed` in a loop while every customer
# generation sat in `queued`. Nothing restarted it, nothing alerted, and the
# only signal that reached anyone was a customer asking why nothing was
# rendering. The GPU sat at 0% the whole time.
#
# The worker at least retries its API calls forever. The tunnel just stays
# dead. That asymmetry is the whole bug: a five-second blip became an outage
# because no one was watching the one process with no recovery of its own.
#
# ## What it does
#
# Every INTERVAL seconds:
#
#   * asks the API for its health THROUGH the tunnel — not `pgrep`, because a
#     half-open SSH session leaves a live process forwarding nothing, which is
#     exactly the failure that looks fine from the outside. Two consecutive
#     failures and the tunnel is killed and rebuilt.
#   * checks the worker process exists, and starts it if not. A missing worker
#     process holds no job by definition, so there is nothing to drain first.
#
# Deliberately NOT a health check on the worker: it owns its own retry loop,
# and killing a process mid-render to fix a transient API error would destroy
# a customer's video to solve a problem that resolves itself.
#
# ## Running it
#
#     setsid nohup bash /workspace/keepalive.sh > /tmp/zolexai-keepalive.log 2>&1 < /dev/null &
#
# `touch /workspace/keepalive.paused` stops it acting without stopping it
# running — take that out before a deliberate restart during a deploy, or it
# will race you and start the worker again on the old code.
#
set -uo pipefail

INTERVAL="${KEEPALIVE_INTERVAL:-20}"
REPO=/workspace/src/zolexai
PAUSE_FILE=/workspace/keepalive.paused
LOCK=/tmp/zolexai-keepalive.lock

HEALTH_URL=http://127.0.0.1:18000/api/v1/health
TUNNEL_MATCH="18000:127.0.0.1:8100"
WORKER_MATCH="python -m worker.main"

TUNNEL_LOG=/tmp/tunnel.log
WORKER_LOG=/tmp/zolexai-ltx-worker.log

# One keepalive per box. Two would fight over the same restarts.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another keepalive holds $LOCK; exiting"
  exit 0
fi

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

start_tunnel() {
  pkill -f "$TUNNEL_MATCH" 2>/dev/null
  sleep 2
  setsid nohup ssh -i /root/.ssh/zolexai_prod_tunnel -N \
    -L 127.0.0.1:18000:127.0.0.1:8100 \
    -o BatchMode=yes -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    zolexai-gpu-tunnel@76.13.27.242 \
    > "$TUNNEL_LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
  sleep 5
}

start_worker() {
  setsid nohup bash -lc \
    "cd $REPO; set -a; source .env.gpu-worker; set +a; exec apps/worker/.venv/bin/python -m worker.main" \
    > "$WORKER_LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
  sleep 8
}

log "keepalive started (interval ${INTERVAL}s)"
health_failures=0

while true; do
  if [ -f "$PAUSE_FILE" ]; then
    sleep "$INTERVAL"
    continue
  fi

  # ── The tunnel, checked by what it is FOR ────────────────────────────
  code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 8 "$HEALTH_URL" 2>/dev/null)
  if [ "$code" = "200" ]; then
    if [ "$health_failures" -gt 0 ]; then
      log "tunnel healthy again (was failing ${health_failures}x)"
    fi
    health_failures=0
  else
    health_failures=$((health_failures + 1))
    log "api health through tunnel returned '${code:-none}' (${health_failures})"
    # Two in a row, so one slow response during a render does not trigger a
    # rebuild of a tunnel that was never broken.
    if [ "$health_failures" -ge 2 ]; then
      log "restarting tunnel"
      start_tunnel
      after=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 8 "$HEALTH_URL" 2>/dev/null)
      log "tunnel restarted, health now '${after:-none}'"
      [ "$after" = "200" ] && health_failures=0
    fi
  fi

  # ── The worker ───────────────────────────────────────────────────────
  if ! pgrep -f "$WORKER_MATCH" > /dev/null 2>&1; then
    log "worker process missing — starting it"
    start_worker
    if pgrep -f "$WORKER_MATCH" > /dev/null 2>&1; then
      log "worker started"
    else
      log "WORKER FAILED TO START — see $WORKER_LOG"
    fi
  fi

  sleep "$INTERVAL"
done
