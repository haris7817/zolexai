# ZolexAI — GPU Worker + VPS Deployment Runbook

**Last updated:** 14 August 2026
**Purpose:** Internal production-style testing of real GPU generation through the ZolexAI production API.

> **Migrated 14 Aug 2026.** Production moved from the RTX 5090 (32 GB) to an
> RTX PRO 6000 Blackwell (96 GB); the 5090 instance was destroyed the same day.
> Sections below describe the **current** box. §33 records the migration itself
> and the traps that only surface when building one of these from scratch.

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

## 2. GPU Environment

GPU host:

```text
Vast.ai RTX PRO 6000 Blackwell Workstation
Instance ID   47698594
SSH           ssh -i ~/.ssh/zolexai_vast -p 24194 root@121.158.120.137
```

The IP and port are read from the Vast console's `>_` button and **change without
warning while the instance keeps running**. A connection timeout means re-read
the address, not that the box died — confirm by instance ID.

Current GPU setup:

```text
GPU:      NVIDIA RTX PRO 6000 Blackwell
VRAM:     95.6 GB (97887 MiB)
Driver:   580.159.03
CUDA:     13.0 (system nvcc) / 13.2 (torch)  ← see §34, this mismatch matters
PyTorch:  2.13.0+cu132
Capability: sm_120 (same as the 5090 — the same wheels work)
```

Superseded hardware, kept for reading older measurements in context:

```text
GPU:     NVIDIA GeForce RTX 5090   (destroyed 14 Aug 2026)
VRAM:    ~31 GB
Driver:  595.84
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
WORKER_NAME=ltx-6000-1
RUNTIME=ltx
RUNTIMES=ltx
MAX_CONCURRENCY=1
```

Current LTX configuration:

```text
LTX_QUANTIZATION=nvfp4-prequant
LTX_MAX_SECONDS=30
```

> **Both of those values are inherited from the 5090 and have NOT been
> re-measured on this card.** `LTX_MAX_SECONDS=30` was the measured single-pass
> ceiling at 32 GB (60s hard-OOMed); NVFP4 was forced by the same limit. On 96 GB
> neither constraint necessarily applies — but the numbers were reached by
> measurement, so raising them requires measurement, not assumption. The same
> goes for `_PIXEL_BUDGET` and the `_DIMENSIONS` grids in `adapters/ltx.py`.

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
924e1de   (17 Aug 2026 — see §38)
```

> The `516b455` recorded elsewhere in this document was already stale before
> that deploy: the box was actually on `38d70de`. Only this line was verified
> against the box, so treat the others as historical until someone checks them.

---

## 7. GPU Worker Environment

Worker virtualenv was created separately:

```bash
cd /workspace/zolexai

uv venv /workspace/zolexai/.venv-worker

uv pip install \
  --python /workspace/zolexai/.venv-worker/bin/python \
  -e /workspace/zolexai/apps/worker
```

The worker venv has **no torch** — it shells out to the LTX pipelines, which
carry their own environment at `/workspace/ltx2-benchmark/.venv`. A
`ModuleNotFoundError: torch` from `.venv-worker` is expected, not a fault.

Worker adapters available:

```text
harness
ltx
mock
music
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
WORKER_NAME=ltx-6000-1

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
worker_name: ltx-6000-1
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
  "worker_name": "ltx-6000-1",
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

An LTX-routed job should be claimed by `ltx-6000-1`, not the VPS mock worker.

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

GPU tunnel:               working
GPU worker:               ltx-6000-1
GPU runtime:              ltx
GPU concurrency:          1
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

---

## 33. Migration: RTX 5090 → RTX PRO 6000 (14 Aug 2026)

Done live, with no service gap. The 5090 kept serving until the new box had
completed a real customer job; rollback was always one command (restart the old
worker) until the instance was destroyed.

Order that worked:

```text
1. verify new box            nvidia-smi, disk, no preinstalled torch
2. base tooling              git ffmpeg tmux curl rsync jq + uv
3. fresh SSH keys            github deploy key, VPS tunnel key, peer key
4. clone repo, open tunnel   confirm /api/v1/health before anything else
5. LTX repo code by rsync    from the old box — carries the local patch
6. LTX weights from HF       ~89 GB, far faster than rsync (see below)
7. build ltx-kernels         see §34 — this is the step that fails
8. parity smoke test         same settings as the old card, then EYEBALL it
9. start new worker          different WORKER_NAME so both can coexist
10. stop old worker          new box now serves alone, old box still rollback
11. one real job via the website, confirmed in the browser
12. rescue test fixtures, then destroy the old instance
```

**Measured result.** A 10s 896×512 text-to-video went from ~60s on the 5090 to
**34s end-to-end** (browser to stored asset) on the PRO 6000. The gain is not
raw compute — the cards are within ~10% on TFLOPS — it is 96 GB removing the
per-job weight reload the 32 GB card had to do.

**Transfer speeds, measured.** rsync between two Vast boxes ran at 8–12 MB/s and
degraded to 14 kB/s under contention (44-hour ETA). HuggingFace to the same box
ran at 33 MB/s single-stream and ~70 MB/s with the CLI's parallel connections.
**Pull weights from HuggingFace; use rsync only for things HF does not have** —
repo code, local commits, test fixtures.

**What must be carried by rsync, not re-cloned.** `ltx2-benchmark` commit
`d434411` (the `_build_transformer` meta-tensor fix) exists on no remote. Losing
it reintroduces solid-green output that passes ffprobe. After any rebuild:

```bash
grep -c "materialize only the tensors" \
  packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py   # must print 1
```

**Never `echo >> authorized_keys` without checking for a trailing newline.** The
existing key had none, so the appended key fused onto it — one line, two key
tokens, both invalid. Caught only because the session was still open. Verify:

```bash
wc -l < ~/.ssh/authorized_keys                       # lines
grep -o "ssh-ed25519\|ssh-rsa" ~/.ssh/authorized_keys | wc -l   # keys
```

Those two numbers must be equal.

---

## 34. Building a GPU box from scratch

### 34.1 Model weights are on a gated HuggingFace repo

`Lightricks/LTX-2.5` returns **401 without a token**, and the token is *not*
recoverable from a running box — the checkpoints live in the repo tree, not in
`~/.cache/huggingface`, and `HF_HOME` is redirected. You need a fresh token from
an account that has accepted the licence at `huggingface.co/Lightricks/LTX-2.5`.

```bash
read -rsp "HF token: " T; echo; mkdir -p /root/.cache/huggingface
printf '%s' "$T" > /root/.cache/huggingface/token
chmod 600 /root/.cache/huggingface/token; export HF_TOKEN="$T"; unset T
```

**`hf download --exclude` is a trap.** Given several patterns it consumes all but
the first as *positional filenames*, warns `Ignoring --exclude since filenames
have been explicitly set`, and downloads exactly what you meant to skip. Pass the
files you want positionally instead:

```bash
export HF_XET_HIGH_PERFORMANCE=1     # HF_HUB_ENABLE_HF_TRANSFER is deprecated
M=/workspace/ltx2-benchmark/models/ltx-2.5
hf download Lightricks/LTX-2.5 --local-dir "$M" \
  diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
  diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  model_patches/ltx-2.5-duration-head-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
```

The repo carries **no config JSONs** — configs come from the repo code. Nothing
else needs fetching.

### 34.2 `ltx-kernels` will not build against the system CUDA

NVFP4 requires a compiled extension that is **not** in the repo and does not
survive a `.venv`-excluded rsync. Building it with the system `nvcc` fails:

```text
error: #error "CUDA compiler and CUDA toolkit headers are incompatible"
RuntimeError: ltx-kernels not built; NVFP4 quantization requires the nvfp4 extension.
```

The cause is a version split: system `nvcc` is **13.0**, torch ships **13.2**
headers. Torch's own wheels include a matching compiler. Point `CUDA_HOME` at it:

```bash
V=/workspace/ltx2-benchmark/.venv/lib/python3.11/site-packages/nvidia/cu13
export CUDA_HOME="$V" CUDA_PATH="$V" PATH="$V/bin:$PATH"
cd /workspace/ltx2-benchmark
TORCH_CUDA_ARCH_LIST="12.0" uv sync --group kernels
```

`TORCH_CUDA_ARCH_LIST` is **12.0** (sm_120), *not* the `10.0` the error message
suggests — that value is for datacenter Blackwell. Both the 5090 and the PRO 6000
are sm_120. Verify by import, not by build exit code:

```bash
.venv/bin/python -c "import ltx_kernels; print('ltx_kernels OK')"
```

**NVFP4 may be unnecessary here.** It was forced by the 5090's 32 GB. At 96 GB the
bf16 distilled checkpoint fits and needs no compiled extension at all — that is
the fallback if this build ever breaks again, and it is already on disk.

### 34.3 Base image quirks

- The image auto-activates a venv called `main`; `unset VIRTUAL_ENV` (and add it
  to `~/.bashrc`) or packages land in the wrong environment.
- It ships its own `uv`, which shadows a fresh install. Use `/root/.local/bin/uv`.
- Every SSH login attaches to one shared tmux (`[ssh_tmux]`).

### 34.4 Validation is visual, not automatic

`ffprobe` exit 0 proves nothing about picture content — the `to_empty()` bug
produced solid-green video that passed every automated check. **Download the
smoke-test output and look at it** before cutting production over.

Test fixtures for comparable re-tests live at `/workspace/fixtures/`
(`tune-source.mp4`, `ext-source.mp4`, `test-photo.jpg`) — carried across from the
5090 so any re-measurement is directly comparable to earlier results.

---

## 35. Not yet done on this box

> 🔴 **OPEN PRODUCTION BUG — read
> [`issue-triton-na-kernel.md`](./issue-triton-na-kernel.md) before touching video.**
> Text to Video **fails at 30s and 60s** on this card (10s/15s/20s pass) with
> `Triton Error [CUDA]: invalid argument` in the video VAE's *fallback*
> neighbourhood-attention kernel. Music Video fails too and has been un-routed to
> `mock`. Found 14 Aug 2026, after the migration was declared complete — the
> parity test used 10s, which passes.

```text
Dev checkpoint           42 GB on disk, NEVER RUN
30s / 60s video          BROKEN — see issue-triton-na-kernel.md
music-video workflow     un-routed to mock, fails on GPU
i2v / extend at 30s+     presumed broken, NOT CONFIRMED
Peak VRAM under load     video not measured on this card (music is: 23.9 GB)
LTX_MAX_SECONDS          still 30, inherited from the 5090
_PIXEL_BUDGET / grids    still sized for 32 GB
```

The dev checkpoint is the reason this card was bought: it is the non-distilled
transformer that exposes guidance scale, negative prompt and step count, which is
the actual fix for the client's *"something missing in every video"*. Until it
has generated something here, **no claim about improved prompt adherence is
GPU-proven** — see [[ltx-quality-tuning-options]] in memory and §29 above.

---

## 36. Music service (ACE-Step) — live since 14 Aug 2026

Music generation went live in production on the PRO 6000. Measured end to end,
browser to stored asset: **15 seconds for a 2-minute song.**

### 36.1 Install

```bash
cd /workspace
git clone https://github.com/ace-step/ACE-Step-1.5.git /workspace/acestep-benchmark
cd /workspace/acestep-benchmark
git checkout 6d467e4          # the commit the model choice was benchmarked on
uv sync                       # pins torch 2.10.0+cu128 — a DIFFERENT pin from LTX's
                              # 2.13.0+cu132. Separate venvs; do not merge them.
```

Weights **auto-download on first request** (~18 GB), so there is nothing to fetch
by hand and nothing worth rsyncing from another box.

### 36.2 `ACESTEP_CONFIG_PATH` is not optional

Left unset, the service silently loads **`acestep-v15-turbo`** — the smaller
non-XL DiT. That is *not* the model ZolexAI evaluated and selected. It still
generates music, so nothing fails; the quality and the VRAM figures are simply
different, and nobody notices.

```bash
hf download ACE-Step/acestep-v15-xl-turbo \
  --local-dir /workspace/acestep-benchmark/checkpoints/acestep-v15-xl-turbo
export ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo
```

Confirm in the log after the first request:

```text
[API Server] Primary model loaded: acestep-v15-xl-turbo
```

With the XL loaded, peak VRAM is **23,972 MiB** — within 70 MiB of the figure
measured on the 5090, so every number in the original benchmark carries over.

### 36.3 The vLLM memory reservation sizes itself correctly

The `gpu_memory_utilization: 0.700` recorded from the 5090 is **not a constant**
and needs no override on a bigger card. `gpu_config.py` caps the reservation to
what the LM actually needs (`usable_for_lm = min(usable_for_lm, total_target_gb)`)
and then converts to a ratio. On 96 GB it logs:

```text
Adaptive LM memory allocation: target=8.0GB, ratio=0.182, total_gpu=95.0GB
```

0.182 × 95 ≈ 17 GB, not the 67 GB a fixed fraction would have taken. **Music and
video coexist safely**: ~24 GB resident for music plus ~28 GB peak for video is
~52 of 96 GB.

The service holds its weights **resident between requests** by design. That is
the scheduling constraint to reason about, not per-job usage.

### 36.4 Provider bug found on first GPU contact

`worker/music/acestep.py` `_poll_once` returned entries as soon as `result` was
non-empty. The service writes a **progress record into `result` while still
generating**, and only fills in `file` once audio exists — so the provider
grabbed the partial record and then failed in `_download` with
`the music service reported success but returned no audio`.

Fixed by only treating entries that carry `file` as finished. **This fix is not
yet committed** (14 Aug 2026) — a rebuild from git reintroduces it.

### 36.5 Output bitrate

The service writes 128 kbps MP3, which is too low for a paid product. The
platform's post-processing re-encodes after loudness normalisation and the
delivered file lands nearer **187 kbps**, so this is less urgent than it looks —
but the service-side setting is still worth raising.

### 36.6 Lyric density is a band, not a ceiling

Verified on this box. Too *few* lines is as bad as too many: 5 lines at 120s
produced an **82-second instrumental intro** plus wordless "oh" padding; 9 lines
at the same duration brought vocals in at 30s with every line sung. The rule of
~1 line per 15–20s is what `line_budget()` in `worker/music/lyrics.py` already
enforces, so generated sheets are correctly sized — the risk is only with
customer-supplied lyrics.

Note the service returns **2 takes per request and they differ**; a given take
may omit a line the other includes. Do not promise exact lyric fidelity.

To check sung lyrics without listening, transcribe:

```bash
uv venv /workspace/whisper-venv
uv pip install --python /workspace/whisper-venv/bin/python faster-whisper
# device="cpu", compute_type="int8" — the GPU path fails on a missing libcublas,
# and CPU transcribes a 2-minute track in seconds.
```

Whisper is reliable for *"did this verse appear at all"* and weak for *"were the
words exactly right"* on sung audio.

---

## 37. Process supervision

**This box does not run systemd.** `systemctl` exists but PID 1 is `bash`, so
units never start. The base image runs **supervisord**, which manages its own
services (`pyworker`, `syncthing`, `tensorboard`) — ZolexAI's three processes are
registered the same way.

```text
/opt/supervisor-scripts/zolexai-tunnel.sh   → /etc/supervisor/conf.d/zolexai-tunnel.conf
/opt/supervisor-scripts/zolexai-worker.sh   → /etc/supervisor/conf.d/zolexai-worker.conf
/opt/supervisor-scripts/zolexai-music.sh    → /etc/supervisor/conf.d/zolexai-music.conf
```

```bash
supervisorctl status | grep zolexai
supervisorctl restart zolexai-worker
supervisorctl reread && supervisorctl update   # after editing a .conf
```

Log paths are unchanged from the manual era: `/tmp/zolexai-tunnel.log`,
`/tmp/zolexai-ltx-worker.log`, `/tmp/acestep-api.log`.

Three details that matter:

- **The worker script waits on the tunnel's health check** before exec'ing.
  Supervisord has no dependency ordering, and a worker that starts first just
  fails registration and thrashes.
- **`stopasgroup` / `killasgroup` on worker and music.** The music service is a
  parent `uv run` wrapping the real python; without these, supervisord kills the
  wrapper and orphans a process still holding 24 GB of VRAM.
- **`ACESTEP_CONFIG_PATH` lives in the script**, not in a shell you happened to
  export it in — see §36.2.

Verified 14 Aug 2026 by killing the worker and music service: both returned to
`RUNNING` with new PIDs in ~43s and the worker re-registered on its own.

**Still unverified: reboot survival.** Supervisord itself is started by Vast's
entrypoint and already manages base-image services, so it should come back — but
the honest test is stopping and starting the instance from the Vast console, and
that has not been done.

---

## 38. Deploy: worker side of V2V transform + audio conditioning (17 Aug 2026)

`38d70de` → `924e1de` on the GPU node only. The VPS was **not** touched.

```bash
cd /workspace/zolexai && git pull --ff-only
supervisorctl restart zolexai-worker
```

**This deploy changes nothing a customer can see, by design.** The new
video-to-video engine is selected by `execution.v2v_engine`, which lives in the
workflow YAML — and the API serves workflow definitions **baked into its image**
(§14). Until that image is rebuilt, production claims carry no such key and the
worker takes the same still-conditioned restyle path it always has. Deploying
the worker first is what makes the activation step reversible: the code is
already proven on the node before anything starts routing to it.

Verified after restart:

- `worker_draining` reported `active_jobs: 0` — nothing was interrupted.
- `worker_ready`: `ltx-6000-1`, runtimes `["ltx", "music"]`, all six workflows.
- New modules import in `.venv-worker`; all three optional weight files
  (`transformer_dev`, `distilled_lora`, `union_control_lora`) are present.
- **End-to-end through the real adapter on the real GPU**: a `video-to-video`
  job with `v2v_engine: transform` built its control clip, rendered on
  `ltx_pipelines.ic_lora`, restored the source audio and validated — 1024x576,
  8.0s against an 8.0s source, 737 KiB, **61s**. This was the last untested
  join: the model invocation and the surrounding pipeline had only ever been
  proven separately.

Rollback is the previous commit plus a restart:

```bash
cd /workspace/zolexai && git checkout 38d70de
supervisorctl restart zolexai-worker
```

**Not yet done — the activation half.** Rebuilding the API image ships
`v2v_engine: transform` and switches every video-to-video job onto the new
engine. `audio_conditioning` for music video stays off in the YAML regardless;
it is ~4x the compute and is a pricing decision.
