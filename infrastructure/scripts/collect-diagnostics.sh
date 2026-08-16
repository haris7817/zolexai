#!/usr/bin/env bash
# ZolexAI VPS diagnostics — evidence bundle for the intermittent API drops.
#
# Context: the public API at :8100 refuses connections for a short window
# roughly twice an hour (first measured 16 Aug 2026; the SSH tunnel was ruled
# out — the refusals originate on the VPS itself). The two leading suspects
# are container restarts (OOM or crash) and something host-side competing for
# the port or the memory. This script collects the evidence for both without
# guessing, so the cause is read from data rather than symptoms.
#
# Usage (as root on the VPS):
#
#     bash /opt/zolexai/infrastructure/scripts/collect-diagnostics.sh
#
# It writes a tar.gz under /tmp and prints the path. Nothing is uploaded
# anywhere; copy it off the box yourself.
#
# SAFETY: this script never reads /opt/zolexai/.env and never runs a bare
# `docker inspect` (whose JSON embeds every container's environment —
# secrets included). Container state is extracted through explicit --format
# templates that name only status fields.

set -u  # not -e: a partially collected bundle beats none, so probes may fail

PROJECT_DIR="${PROJECT_DIR:-/opt/zolexai}"
COMPOSE=(docker compose --env-file .env -f infrastructure/compose/docker-compose.prod.yml)
SERVICES=(api worker web postgres redis minio)
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="/tmp/zolexai-diagnostics-${STAMP}"
mkdir -p "${OUT}"

cd "${PROJECT_DIR}" || { echo "cannot cd ${PROJECT_DIR}" >&2; exit 1; }

note() { printf '== %s\n' "$*"; }

# ── 1. Host at a glance ──────────────────────────────────────────────────
{
  note "collected (UTC)"; date -u
  note "uptime";          uptime
  note "kernel";          uname -a
  note "memory";          free -h
  note "swap";            swapon --show
  note "disk";            df -h
  note "vmstat (3 samples)"; vmstat 1 3
} > "${OUT}/host.txt" 2>&1

# ── 2. Who owns :8100, right now ─────────────────────────────────────────
# A refusal means nothing was listening at that instant. If this shows a
# different owner than the API container's proxy, that is the whole story.
{
  note "listeners on :8100";  ss -ltnp | grep -E ':8100\b' || echo "NOTHING LISTENING on 8100"
  note "all listeners";       ss -ltnp
} > "${OUT}/ports.txt" 2>&1

# ── 3. Container state — restart counts are the headline number ─────────
# RestartCount > 0 or OOMKilled=true on the api container closes the case.
{
  "${COMPOSE[@]}" ps
  echo
  for service in "${SERVICES[@]}"; do
    container="$("${COMPOSE[@]}" ps -q "${service}" 2>/dev/null)"
    [ -n "${container}" ] || { echo "${service}: no container"; continue; }
    docker inspect \
      --format "${service}: status={{.State.Status}} restarts={{.RestartCount}} oom_killed={{.State.OOMKilled}} exit_code={{.State.ExitCode}} started={{.State.StartedAt}} finished={{.State.FinishedAt}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}" \
      "${container}"
  done
} > "${OUT}/containers.txt" 2>&1

# ── 4. Docker's own event log, last 48h ──────────────────────────────────
# Every die / oom / restart / health transition, with timestamps to line up
# against the observed refusal windows. This is historical — it returns
# immediately, it does not wait for new events.
docker events --since 48h --until 1s \
  --filter type=container \
  --format '{{.Time}} {{.Actor.Attributes.name}} {{.Action}} exitCode={{index .Actor.Attributes "exitCode"}}' \
  > "${OUT}/docker-events.txt" 2>&1

# ── 5. Kernel OOM killer — the suspect docker events cannot see ─────────
# If the kernel killed a process inside a container (rather than docker
# OOM-killing the container), only dmesg records it.
{
  dmesg -T 2>/dev/null | grep -iE 'out of memory|oom[-_ ]kill|killed process' \
    || echo "no OOM-killer lines in dmesg"
} > "${OUT}/oom.txt" 2>&1

# ── 6. The docker daemon's journal, last 48h ─────────────────────────────
# Daemon-level trouble (restarts of dockerd itself, live-restore events,
# network plugin errors) shows up here and nowhere else.
journalctl -u docker --since "-48h" --no-pager > "${OUT}/dockerd-journal.txt" 2>&1

# ── 7. Host scheduling — anything that fires twice an hour? ─────────────
# The drops recur on a suspiciously regular cadence. A cron job or systemd
# timer with a matching period (certbot, backups, log rotation) would explain
# the rhythm.
{
  note "systemd timers";  systemctl list-timers --all --no-pager
  note "root crontab";    crontab -l 2>/dev/null || echo "no root crontab"
  note "/etc/cron.d";     ls -la /etc/cron.d/ 2>/dev/null
} > "${OUT}/schedules.txt" 2>&1

# ── 8. Service logs around the incidents ─────────────────────────────────
# Same logs the runbook already reads on every deploy; timestamps let them be
# lined up against the events above.
"${COMPOSE[@]}" logs --since 48h --timestamps --tail 2000 api   > "${OUT}/logs-api.txt"    2>&1
"${COMPOSE[@]}" logs --since 48h --timestamps --tail 500  web   > "${OUT}/logs-web.txt"    2>&1
"${COMPOSE[@]}" logs --since 48h --timestamps --tail 500  worker > "${OUT}/logs-worker.txt" 2>&1

# ── 9. nginx, if it fronts the port on this host ─────────────────────────
if [ -f /var/log/nginx/error.log ]; then
  tail -n 300 /var/log/nginx/error.log > "${OUT}/nginx-error.txt" 2>&1
fi

tar -czf "${OUT}.tar.gz" -C /tmp "$(basename "${OUT}")"
rm -rf "${OUT}"

echo
echo "Diagnostics bundle: ${OUT}.tar.gz"
echo "Contains NO secrets (no .env, no container environments)."
echo "Copy it off the VPS and hand it over for analysis."
