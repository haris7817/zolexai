# ZolexAI — RTX 5090 + VPS Worker Deployment Runbook

**Last updated:** 13 August 2026
**Purpose:** Internal production-style testing of real GPU generation through the ZolexAI production API.

This is the M2 companion to [`production-runbook.md`](./production-runbook.md), which
documents the M1 stack (mock worker, no GPU). It is written from what was actually
done, after it was done.

> **IMPORTANT:** Never put private SSH keys, passwords, production API tokens, or
> other secrets in this document or Git.

---

## 1. Current Deployment State

### VPS

Production project:

```text
/opt/zolexai
```

Current deployed commit:

```text
516b455
```

Production endpoints:

```text
Web:     https://zolexai.com
API:     https://zolexai.com/api/v1/*
Storage: https://storage.zolexai.com
```

Production services:

```text
web
api
worker
postgres
redis
minio
```

Loopback ports:

```text
Web      127.0.0.1:3100
API      127.0.0.1:8100
MinIO    127.0.0.1:9000
Console  127.0.0.1:9001
```

CloudPanel/Nginx owns public ports 80/443.

Public access to `/api/v1/internal/*` remains blocked.

Production-only files:

```text
infrastructure/compose/docker-compose.prod.yml
infrastructure/minio-cors.xml
```

These files are intentionally untracked and must not accidentally be
committed/deleted.

---

## 2. RTX 5090 Environment

GPU host:

```text
Vast.ai RTX 5090
```

Current GPU setup:

```text
GPU:     NVIDIA GeForce RTX 5090
VRAM:    ~31 GB
Driver:  595.84
CUDA:    13.2
PyTorch: 2.13.0+cu132
```

LTX repository:

```text
/workspace/ltx2-benchmark
```

ZolexAI repository:

```text
/workspace/zolexai
```

Worker Python environment:

```text
/workspace/zolexai/.venv-worker
```

Worker workspace:

```text
/workspace/zolexai-worker
```

Current GPU worker:

```text
WORKER_NAME=ltx-5090-1
RUNTIME=ltx
RUNTIMES=ltx
MAX_CONCURRENCY=1
```

Current LTX configuration:

```text
LTX_QUANTIZATION=nvfp4-prequant
LTX_MAX_SECONDS=30
```

A direct 60-second LTX pass can OOM the RTX 5090.

Long generations therefore use:

```text
segment
→ generate
→ normalize
→ stitch
→ verify
```

---

## 3. Secure RTX → VPS API Connection

The production internal API is **NOT** exposed publicly.

A dedicated VPS user was created:

```text
zolexai-gpu-tunnel
```

SSH restriction config:

```text
/etc/ssh/sshd_config.d/99-zolexai-gpu-tunnel.conf
```

Configuration:

```text
Match User zolexai-gpu-tunnel
    PasswordAuthentication no
    PubkeyAuthentication yes
    AllowTcpForwarding local
    PermitOpen 127.0.0.1:8100
    X11Forwarding no
    PermitTTY no
    AllowAgentForwarding no
    GatewayPorts no
```

Validate:

```bash
sshd -t
```

---

## 4. RTX Tunnel Key

RTX private key:

```text
/root/.ssh/zolexai_prod_tunnel
```

**NEVER print or share this private key.**

Public key is installed on VPS under:

```text
/home/zolexai-gpu-tunnel/.ssh/authorized_keys
```

Recorded public-key fingerprint:

```text
SHA256:UgkneO+nHoZsegkoetAyOiPCPNZ0b6tuZE0ntf6va3g
```

---

## 5. Start Secure Tunnel on RTX

Run on RTX:

```bash
nohup ssh \
  -i /root/.ssh/zolexai_prod_tunnel \
  -N \
  -L 127.0.0.1:18000:127.0.0.1:8100 \
  -o BatchMode=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  zolexai-gpu-tunnel@76.13.27.242 \
  >/tmp/zolexai-tunnel.log 2>&1 &
```

Check tunnel:

```bash
ss -ltnp | grep 18000
```

Test production API through tunnel:

```bash
curl --max-time 10 -sS \
  http://127.0.0.1:18000/api/v1/health
```

Expected:

```json
{
  "status": "ok",
  "environment": "production",
  "checks": {
    "database": true,
    "redis": true,
    "storage": true,
    "workflows": true
  }
}
```

---

## 6. GitHub Access on RTX

Dedicated read-only deploy key:

```text
/root/.ssh/zolexai_github
```

SSH config:

```text
Host github-zolexai
    HostName github.com
    User git
    IdentityFile /root/.ssh/zolexai_github
    IdentitiesOnly yes
```

Test:

```bash
ssh -T github-zolexai
```

Repository:

```text
/workspace/zolexai
```

Current checkout:

```text
516b455
```

---

## 7. RTX Worker Environment

Worker virtualenv was created separately:

```bash
cd /workspace/zolexai

uv venv /workspace/zolexai/.venv-worker

uv pip install \
  --python /workspace/zolexai/.venv-worker/bin/python \
  -e /workspace/zolexai/apps/worker
```

Worker adapters available:

```text
harness
ltx
mock
```

---

## 8. GPU Worker Environment File

File:

```text
/workspace/zolexai/.env.gpu-worker
```

Permissions:

```bash
chmod 600 /workspace/zolexai/.env.gpu-worker
```

Configuration:

```ini
WORKER_NAME=ltx-5090-1

RUNTIME=ltx
RUNTIMES=ltx

API_BASE_URL=http://127.0.0.1:18000

USE_REDIS_WAKEUP=false
MAX_CONCURRENCY=1

LTX_REPO_DIR=/workspace/ltx2-benchmark
LTX_QUANTIZATION=nvfp4-prequant
LTX_MAX_SECONDS=30

WORKSPACE_DIR=/workspace/zolexai-worker

REQUEST_TIMEOUT_SECONDS=20
DOWNLOAD_TIMEOUT_SECONDS=300
UPLOAD_TIMEOUT_SECONDS=900

LOG_FORMAT=json

WORKER_API_TOKEN=<PRODUCTION_WORKER_API_TOKEN>
```

Never commit this file with the production token.

The GPU does **NOT** need:

```text
PostgreSQL credentials
MinIO credentials
Redis credentials
```

Media transfers happen using API-generated presigned URLs.

---

## 9. Worker Token Verification

We initially received:

```text
401 worker_api_rejected
```

because the RTX Worker API token did not exactly match production.

**Do not display the actual token.**

### VPS

```bash
bash -lc '
set -a
source /opt/zolexai/.env
set +a

printf "length=%s\n" "${#WORKER_API_TOKEN}"
printf "%s" "$WORKER_API_TOKEN" | sha256sum
'
```

### RTX

```bash
bash -lc '
set -a
source /workspace/zolexai/.env.gpu-worker
set +a

printf "length=%s\n" "${#WORKER_API_TOKEN}"
printf "%s" "$WORKER_API_TOKEN" | sha256sum
'
```

Both must have exactly the same:

```text
length
SHA256
```

Current production token length:

```text
96
```

The actual token/hash should not be stored in public documentation.

---

## 10. Verify Worker Configuration

RTX:

```bash
cd /workspace/zolexai

set -a
source .env.gpu-worker
set +a

/workspace/zolexai/.venv-worker/bin/python - <<'PY'
from worker.core.config import WorkerSettings

s = WorkerSettings()

print("worker_name:", s.worker_name)
print("runtime:", s.runtime)
print("runtimes:", s.runtime_list)
print("api_base_url:", s.api_base_url)
print("max_concurrency:", s.max_concurrency)
print("redis_wakeup:", s.use_redis_wakeup)
print("ltx_repo:", s.ltx_repo_dir)
print("quantization:", s.ltx_quantization)
print("token_present:", bool(s.worker_api_token))
PY
```

Expected:

```text
worker_name: ltx-5090-1
runtime: ltx
runtimes: ['ltx']
api_base_url: http://127.0.0.1:18000
max_concurrency: 1
redis_wakeup: False
ltx_repo: /workspace/ltx2-benchmark
quantization: nvfp4-prequant
token_present: True
```

---

## 11. Start RTX Worker

First ensure there is no old test worker:

```bash
pgrep -af "worker.main"
```

Then:

```bash
cd /workspace/zolexai

nohup bash -lc '
  cd /workspace/zolexai

  set -a
  source .env.gpu-worker
  set +a

  exec /workspace/zolexai/.venv-worker/bin/python -m worker.main
' >/tmp/zolexai-ltx-worker.log 2>&1 &
```

Verify:

```bash
sleep 7

pgrep -af "worker.main"

tail -n 120 /tmp/zolexai-ltx-worker.log
```

Latest successful startup:

```json
{
  "message": "worker_ready",
  "worker_name": "ltx-5090-1",
  "runtime": "ltx",
  "runtimes": ["ltx"],
  "workflows": [
    "text-to-video",
    "image-to-video",
    "video-to-video",
    "extend-video",
    "music",
    "music-video"
  ],
  "max_concurrency": 1
}
```

---

## 12. VPS Workflow Routing for Internal GPU Testing

Current Git production definitions normally route through:

```text
runtime: mock
```

For internal RTX testing, we locally changed these three workflows:

```text
workflow-definitions/text-to-video.yaml
workflow-definitions/image-to-video.yaml
workflow-definitions/extend-video.yaml
```

to:

```yaml
execution:
  # SERVER TEST ROUTING ONLY — do not commit this change.
  runtime: ltx
```

Current real GPU test workflows:

```text
Text-to-Video
Image-to-Video
Video Extension
```

---

## 13. Workflow Backup

Before changing routing:

```bash
cd /opt/zolexai

mkdir -p /root/zolexai-ltx-routing-backup-2026-08-12

cp workflow-definitions/text-to-video.yaml \
  /root/zolexai-ltx-routing-backup-2026-08-12/

cp workflow-definitions/image-to-video.yaml \
  /root/zolexai-ltx-routing-backup-2026-08-12/

cp workflow-definitions/extend-video.yaml \
  /root/zolexai-ltx-routing-backup-2026-08-12/
```

---

## 14. Important: API Image Must Be Rebuilt

We discovered that merely changing the YAML files on the VPS and recreating the
API did **NOT** update runtime routing.

The workflow definitions are copied into the API image.

Therefore after changing workflow YAML:

```bash
cd /opt/zolexai

COMPOSE="docker compose \
  --env-file /opt/zolexai/.env \
  -f infrastructure/compose/docker-compose.prod.yml"

$COMPOSE build api

$COMPOSE up -d \
  --no-deps \
  --force-recreate \
  api
```

Wait:

```bash
sleep 8
```

Verify:

```bash
$COMPOSE ps api
```

Health:

```bash
curl -sS \
  http://127.0.0.1:8100/api/v1/health
```

---

## 15. Verify Runtime Inside API Container

This is the important final routing check:

```bash
docker exec zolexai-prod-api-1 sh -lc '
for f in text-to-video image-to-video extend-video; do

  echo "----- $f -----"

  grep -n -A4 "^execution:" \
    "/workflow-definitions/$f.yaml"

  echo

done
'
```

Latest confirmed state:

```text
text-to-video  → runtime: ltx
image-to-video → runtime: ltx
extend-video   → runtime: ltx
```

---

## 16. Current VPS Git Status

Current expected status:

```text
 M workflow-definitions/extend-video.yaml
 M workflow-definitions/image-to-video.yaml
 M workflow-definitions/text-to-video.yaml

?? infrastructure/compose/docker-compose.prod.yml
?? infrastructure/minio-cors.xml
```

Meaning:

```text
3 modified YAML files = temporary internal GPU routing
2 untracked files      = intentional production-only configuration
```

Do **NOT** blindly run:

```bash
git add .
git clean -fd
git reset --hard
```

---

## 17. Current Request Flow

```text
User Browser
     ↓
zolexai.com
     ↓
Production FastAPI
     ↓
Job created
     ↓
Worker control API
     ↓
Secure RTX SSH tunnel
     ↓
ltx-5090-1
     ↓
LTX 2.5 / RTX 5090
     ↓
Generated output
     ↓
Presigned upload
     ↓
Production storage
     ↓
ZolexAI Media/Result
```

---

## 18. RTX Worker Monitoring

Follow worker live:

```bash
tail -f /tmp/zolexai-ltx-worker.log
```

Recent logs:

```bash
tail -n 200 /tmp/zolexai-ltx-worker.log
```

GPU usage:

```bash
nvidia-smi
```

Worker process:

```bash
pgrep -af "worker.main"
```

---

## 19. VPS Monitoring

API:

```bash
cd /opt/zolexai

docker compose \
  --env-file /opt/zolexai/.env \
  -f infrastructure/compose/docker-compose.prod.yml \
  logs --tail=200 api
```

Mock worker:

```bash
docker compose \
  --env-file /opt/zolexai/.env \
  -f infrastructure/compose/docker-compose.prod.yml \
  logs --tail=100 worker
```

An LTX-routed job should be claimed by `ltx-5090-1`, not the VPS mock worker.

---

## 20. Troubleshooting: 401 Worker Registration

Symptom:

```text
worker_api_rejected
status_code: 401
```

Check:

```text
WORKER_API_TOKEN
```

Compare hash/length between VPS and RTX without displaying the token.

After correcting the token, restart the RTX worker.

---

## 21. Troubleshooting: Old Worker

Check:

```bash
pgrep -af "worker.main"
```

Inspect:

```bash
ps -fp <PID>
```

Working directory:

```bash
readlink -f /proc/<PID>/cwd
```

Non-secret environment:

```bash
tr '\0' '\n' < /proc/<PID>/environ \
  | grep -E \
  '^(WORKER_NAME|RUNTIME|RUNTIMES|API_BASE_URL|MAX_CONCURRENCY)='
```

Stop only the stale worker:

```bash
kill <PID>
```

---

## 22. Troubleshooting: API Still Shows Mock

Check:

```bash
docker exec zolexai-prod-api-1 sh -lc '
grep -n -A4 "^execution:" \
/workflow-definitions/text-to-video.yaml
'
```

If it shows `runtime: mock`, the API image needs rebuilding:

```bash
$COMPOSE build api
$COMPOSE up -d --no-deps --force-recreate api
```

---

## 23. Roll Back GPU Routing

To return to original mock routing:

```bash
cd /opt/zolexai

cp \
/root/zolexai-ltx-routing-backup-2026-08-12/text-to-video.yaml \
workflow-definitions/text-to-video.yaml

cp \
/root/zolexai-ltx-routing-backup-2026-08-12/image-to-video.yaml \
workflow-definitions/image-to-video.yaml

cp \
/root/zolexai-ltx-routing-backup-2026-08-12/extend-video.yaml \
workflow-definitions/extend-video.yaml
```

Rebuild API:

```bash
COMPOSE="docker compose \
  --env-file /opt/zolexai/.env \
  -f infrastructure/compose/docker-compose.prod.yml"

$COMPOSE build api

$COMPOSE up -d \
  --no-deps \
  --force-recreate \
  api
```

Then verify inside the container that the workflows are back to `runtime: mock`.

---

## 24. Stop RTX Worker

Find:

```bash
pgrep -af "worker.main"
```

Stop:

```bash
kill <PID>
```

Verify:

```bash
pgrep -af "worker.main" || echo "no worker running"
```

---

## 25. Stop SSH Tunnel

Find:

```bash
pgrep -af \
"127.0.0.1:18000:127.0.0.1:8100"
```

Stop:

```bash
kill <PID>
```

Verify:

```bash
ss -ltnp | grep 18000 || echo "tunnel stopped"
```

---

## 26. Security Rules

Always preserve:

- Never display `WORKER_API_TOKEN`.
- Never display SSH private keys.
- Do not put DB credentials on GPU.
- Do not put MinIO credentials on GPU.
- Keep FastAPI 8100 private.
- Keep `/api/v1/internal/*` blocked publicly.
- Use the restricted SSH tunnel user.
- Keep GitHub deploy key read-only.
- Keep GPU concurrency at 1 during current testing.
- Do not commit temporary LTX production-test routing.
- Do not delete production-only untracked files.
- Do not force-push production.
- Production deployment remains manual.

---

## 27. Current LTX Duration Limit

Current:

```text
LTX_MAX_SECONDS=30
```

Safe architecture for a 60-second request:

```text
30 sec generation
+
30 sec generation
+
stitch
```

Example arbitrary long duration:

```text
73 seconds
=
30 + 30 + 13
```

This avoids a dangerous single 60-second GPU invocation.

---

## 28. Current Workflow Status

### Real GPU / Internal Production Testing

```text
Text-to-Video     READY
Image-to-Video    READY
Video Extension   READY
```

### Remaining M2

```text
Video-to-Video      IN PROGRESS / REMAINING
Music Video         IN PROGRESS / REMAINING
Music Generation    IN PROGRESS / REMAINING
```

> Superseded in code as of 13 Aug 2026 — see §32 below. The three above are
> implemented and locally tested; none has run on the GPU yet, so this section
> stays accurate for *deployed* status and §32 records the code status.

---

## 29. Release Boundary

Current GPU deployment is for:

```text
internal testing
development
quality validation
production-style pipeline testing
```

It should not yet be treated as the final public commercial launch.

Commercial/public LTX usage should only be enabled after the licensing position
is separately confirmed — see [`ltx-2.5-licensing-review.md`](./ltx-2.5-licensing-review.md).

---

## 30. Daily Health Check

### VPS

```bash
cd /opt/zolexai

COMPOSE="docker compose \
  --env-file /opt/zolexai/.env \
  -f infrastructure/compose/docker-compose.prod.yml"

$COMPOSE ps

curl -sS \
  http://127.0.0.1:8100/api/v1/health
```

### RTX

Tunnel:

```bash
curl --max-time 10 -sS \
  http://127.0.0.1:18000/api/v1/health
```

Worker:

```bash
pgrep -af "worker.main"

tail -n 50 \
  /tmp/zolexai-ltx-worker.log
```

GPU:

```bash
nvidia-smi
```

---

## 31. Latest Known Good State

```text
VPS commit:               516b455
API:                      healthy
Web:                      healthy
PostgreSQL:               healthy
Redis:                    healthy
MinIO:                    healthy
Public internal routes:   blocked

RTX tunnel:               working
RTX worker:               ltx-5090-1
RTX runtime:              ltx
RTX concurrency:          1
Worker token:             verified/matched

Text-to-Video routing:    ltx
Image-to-Video routing:   ltx
Extend Video routing:     ltx

API image:                rebuilt with LTX test routing
Real GPU website testing: enabled for the 3 above workflows
```

---

## 32. Code changes since this state was captured (13 Aug 2026)

Recorded here so the runbook above is not silently stale. **Nothing below is
deployed**, and none of it has run on the GPU — the deployed state remains §31.

**New workflows on the LTX runtime.** The adapter now handles `video-to-video`
and `music-video` in addition to the three above. Both take their duration from
the uploaded file rather than from a request parameter.

**New runtime: `music`.** §7's adapter list is now `harness, ltx, mock, music`.
It refuses every job until `MUSIC_LAUNCHER` points at a model wrapper, because no
music model has been selected yet. Nothing else about the node changes.

**GPU testing no longer requires §12–§15.** The smoke scripts build an
`AdapterJob` directly and call the adapter, bypassing the API, the database, the
storage layer and workflow routing entirely — so **no YAML edit and no API image
rebuild** is needed to exercise a workflow on the GPU:

```bash
cd /workspace/zolexai/apps/worker
PY=/workspace/zolexai/.venv-worker/bin/python

# Video to Video — duration comes from the source, so DURATION is ignored
MODE=restyle VIDEO=/path/clip.mp4 $PY scripts/ltx_smoke.py as a charcoal sketch

# …with the optional reference image
MODE=restyle VIDEO=/path/clip.mp4 REFERENCE=/path/look.png $PY scripts/ltx_smoke.py …

# Music Video — duration comes from the track
MODE=music-video AUDIO=/path/song.mp3 $PY scripts/ltx_smoke.py a dancer, hard side light

# Force multi-pass chaining on a short input, to check seams cheaply
MODE=restyle VIDEO=/path/10s.mp4 MAX_SEGMENT_SECONDS=5 $PY scripts/ltx_smoke.py …

# Music — only once a model wrapper exists
MUSIC_LAUNCHER="uv run python -m zolexai_music" $PY scripts/music_smoke.py an upbeat pop song
```

`MODE` is one of `text`, `image`, `extend`, `restyle`, `music-video`. The older
`IMAGE=` / `VIDEO=` invocations still work and infer the mode.

**First thing to verify on the GPU.** Video-to-Video conditions each pass on
several stills lifted from the source, which assumes the pipeline's
`--image PATH FRAME_IDX STRENGTH` argument may be repeated. That is **not yet
confirmed on hardware.** If a restyle run fails on the second `--image`, set
`v2v_keyframes: 1` in `workflow-definitions/video-to-video.yaml` — the
single-still form is the documented fallback.

**When these workflows are eventually routed for a website test**, §12–§15 apply
unchanged, with one addition: `video-to-video.yaml`, `music-video.yaml` and
`music.yaml` still carry M1's

```yaml
  output_content_type: image/png
  output_kind: image
```

Those two lines must be **removed at the same time** as the runtime is switched,
or the API will sign the worker's upload for the wrong content type and the job
will fail after the render has already been paid for.
