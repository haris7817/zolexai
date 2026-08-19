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

**Corrected 19 Aug 2026 against the live VPS.** This section long described
three workflows and one edit. It is now SIX workflows and TWO edits per file —
found while deploying §43, because the recorded expectation no longer matched
what `git status` actually printed.

```text
workflow-definitions/text-to-video.yaml     runtime: ltx
workflow-definitions/image-to-video.yaml    runtime: ltx
workflow-definitions/extend-video.yaml      runtime: ltx
workflow-definitions/video-to-video.yaml    runtime: ltx  (+ v2v_engine: transform)
workflow-definitions/music-video.yaml       runtime: ltx
workflow-definitions/music.yaml             runtime: music   <- its own runtime
```

Each file carries BOTH of these edits, not just the first:

```yaml
execution:
  # SERVER TEST ROUTING ONLY — do not commit this change.
  runtime: ltx          # was: mock

  # ...and the two M1 mock-output lines are DELETED:
  #   output_content_type: image/png
  #   output_kind: image
```

**The deletion is not optional and is easy to miss.** Those two lines are the
M1 placeholder that makes the API sign the worker's upload as a PNG; leaving
them in place while routing to a real GPU signs a video as an image. They are
still committed in the repo (M2 has not removed them), so every production
YAML edit removes them locally.

`video-to-video.yaml` additionally moves the committed `v2v_engine: transform`
key up beside `runtime`, and `music-video.yaml` / `video-to-video.yaml` have
had their long explanatory comment blocks stripped locally. Both are cosmetic —
the keys resolve identically — but it means the on-VPS files no longer carry
the "how to enable audio conditioning / person lock" notes. The repo does.

Current real GPU workflows: all six.

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

Current expected status (**verified on the box 19 Aug 2026** — this listed
three YAMLs until then, see §12):

```text
 M workflow-definitions/extend-video.yaml
 M workflow-definitions/image-to-video.yaml
 M workflow-definitions/music-video.yaml
 M workflow-definitions/music.yaml
 M workflow-definitions/text-to-video.yaml
 M workflow-definitions/video-to-video.yaml

?? infrastructure/compose/docker-compose.prod.yml
?? infrastructure/minio-cors.xml
```

Meaning:

```text
6 modified YAML files = production GPU routing (+ mock-output removal, §12)
2 untracked files      = intentional production-only configuration
```

Before any `git stash` on this checkout, print the local edits and confirm
they are only those two known changes — a stash you cannot describe is a stash
you should not pop:

```bash
runuser -u zolexai -- git --no-pager diff workflow-definitions/   | grep -E "^(diff|[+-][^+-])"
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

## 39. Deploy: worker side of the guided tier (18 Aug 2026)

`1fe0fa2` → `dad546b` on the GPU node only, same shape as §38: the new code is
unreachable until a workflow YAML sets `execution.generation_engine: guided`,
and no shipped YAML does — the tier is documented commented-out in
`text-to-video.yaml` / `image-to-video.yaml`. Deploying the worker first keeps
the eventual activation reversible.

Procedure was §38's:

```bash
cd /workspace/zolexai && git pull --ff-only
supervisorctl restart zolexai-worker
```

Verified after restart:

- `worker_draining` reported `active_jobs: 0` — nothing interrupted.
- `worker_ready`: `ltx-6000-1`, runtimes `["ltx", "music"]`, all six workflows.
- `_GUIDED` imports in the worker venv: `ltx_pipelines.ti2vid_two_stages`,
  landings `(121,)`, pass ceiling 5.0s.
- **End-to-end through the real adapter on the real GPU**: a text-to-video job
  with `generation_engine: guided` rendered on the dev transformer +
  distilled LoRA, validated video+audio — 1024x576, 5.013s, 667 KiB,
  **130.6s wall**. The default path's argv is byte-identical and stays pinned
  by the suite (540 passed on this commit).

Rollback is the previous commit plus a restart:

```bash
cd /workspace/zolexai && git checkout 1fe0fa2
supervisorctl restart zolexai-worker
```

**The VPS half (owner-performed).** The VPS checkout carries deliberate
uncommitted YAML edits (`runtime: mock` → `ltx` on three workflows), and this
push touches two of those same files with comment blocks — a bare
`git pull --ff-only` will refuse. The sequence that preserves the routing:

```bash
cd /opt/zolexai
runuser -u zolexai -- git stash
runuser -u zolexai -- git pull --ff-only   # → 92f23e8 (or later)
runuser -u zolexai -- git stash pop        # re-applies the runtime flips; the
                                           #   new comments don't touch the
                                           #   runtime line, so it merges clean
runuser -u zolexai -- git status           # expect the same three YAMLs
                                           #   modified, nothing else
```

Then the §14 image rebuild, because the API bakes `workflow-definitions/` in
(canonical invocation from §14; migrations are not needed — this release has
no schema changes):

```bash
cd /opt/zolexai
COMPOSE="docker compose \
  --env-file /opt/zolexai/.env \
  -f infrastructure/compose/docker-compose.prod.yml"

$COMPOSE build api
$COMPOSE up -d --no-deps --force-recreate api
sleep 8
$COMPOSE ps api
curl -sS http://127.0.0.1:8100/api/v1/health

docker exec zolexai-prod-api-1 grep -n -A6 '^execution:' \
  /workflow-definitions/video-to-video.yaml   # must show v2v_engine: transform
docker exec zolexai-prod-api-1 grep -n -A2 '^execution:' \
  /workflow-definitions/text-to-video.yaml    # must still show runtime: ltx
```

What that rebuild activates: **video-to-video switches to the transform
engine** (`v2v_engine: transform` is committed in the repo YAML). Music-video
`audio_conditioning` and the guided tier stay OFF — both are commented in the
YAML and each is a separate pricing/quality decision. The worker at `dad546b`
already serves all three paths, so later activations are YAML-only. Gate after
the rebuild: one real video-to-video job through zolexai.com, eyes on the
result.

## 40. Person lock: what a GPU node needs before the flag can be enabled

`execution.v2v_person_lock` (video-to-video) mattes the people in each pass and
carries their own pixels through the restyle, so a customer's subject stops
coming back with a different face and a different skin tone. The worker code
ships with the checkout; the **model environment needs three packages**, once
per node:

```bash
cd /workspace/ltx2-benchmark
uv pip install kornia timm transformers
```

`transformers` pulls the matting model (`ZhengPeng7/BiRefNet`, MIT) from
HuggingFace on first use and caches it; `kornia` and `timm` are its own
imports and it fails on them at load time without them. Verify the whole path
without touching a workflow — this is the matter's CLI, run exactly the way the
worker runs it:

```bash
cd /workspace/ltx2-benchmark
uv run python /workspace/zolexai/apps/worker/scripts/person_matte.py \
  --source <any clip>.mp4 --dest /tmp/matte.mp4 \
  --start-seconds 0 --duration-seconds 8.04 \
  --width 1024 --height 576 --fps 24 --frames 193
ffprobe -v error -show_entries stream=width,height,nb_frames \
  -of csv=p=0 /tmp/matte.mp4     # must be exactly 1024,576,193
```

Those three numbers are the whole contract: the matte is merged with the edge
map frame for frame, so a matte of a different size or length protects the
wrong pixels rather than simply protecting fewer.

**No file needs copying anywhere.** The worker resolves the script from its own
checkout (`settings.person_matte_argv`), so a node that has pulled the worker
already has the matter. `PERSON_MATTE_COMMAND` overrides it for a node that
keeps the script elsewhere.

**Cost, measured 18 Aug 2026** at 1024x576, 193-frame pass: 61s against the
edge-only path's ~54, plus the matting pass itself (~40s on this card for 193
frames). Call it a third more wall-clock per section.

**Still to prove before enabling it in a workflow:** the whole path has been
verified on ONE 8-second window. A chained multi-pass job, several people in
frame, occlusion and fast motion are all unmeasured, and the matter's behaviour
in those cases decides whether the seam is invisible or obvious.

## 41. Deploy: Director / Idea mode for Text to Video (18 Aug 2026)

`201a0de` → `0a1025f` on the GPU node. Same two-half shape as §38 and §39,
with one difference worth stating plainly: **this release's activation switch
is a public `settings` flag, not a private `execution` key.** The worker half
is inert without it — the API refuses `prompt_mode` on any workflow whose
served definition lacks `settings.prompt_modes`, so until the image is
rebuilt the feature is not merely unrouted, it is unreachable and invisible.

Research and every measurement behind it:
[`research-2026-08-18-director-idea-mode.md`](./research-2026-08-18-director-idea-mode.md).

### 41.1 The GPU half (done)

```bash
cd /workspace/zolexai && git pull --ff-only
supervisorctl restart zolexai-worker
```

Verified after restart:

- `worker_draining` reported `active_jobs: 0` — nothing interrupted. (A
  customer image-to-video job had completed nine minutes earlier; this node
  is serving real traffic, so the drain check is not a formality.)
- `worker_ready`: `ltx-6000-1`, runtimes `["ltx", "music"]`, all six workflows.
- `worker.director` imports in `.venv-worker`; `settings.director_gemma_root`
  resolves to an existing directory.
- **End-to-end through the real adapter on the real GPU**, from the deployed
  checkout: a text-to-video job with `prompt_mode: director` planned a scene
  locally and rendered it with spoken dialogue — see §41.3.

Rollback is the previous commit plus a restart:

```bash
cd /workspace/zolexai && git checkout 201a0de
supervisorctl restart zolexai-worker
```

### 41.2 The planner checkpoint is a per-node prerequisite

Director mode plans with a **generative** Gemma instruct checkpoint — the one
LTX 2.5 names as its own prompt enhancer, Apache 2.0, so one download serves
both roles. It is NOT part of the LTX weight set fetched in §34.1 and a node
without it fails every director job (and only director jobs):

```bash
export HF_XET_HIGH_PERFORMANCE=1
hf download google/gemma-4-e2b-it \
  --local-dir /workspace/ltx2-benchmark/models/gemma-4-e2b-it   # ~10 GB
```

`DIRECTOR_GEMMA_DIR` overrides the location; `DIRECTOR_PLANNER_COMMAND`
overrides the whole invocation. The planner runs in the LTX environment
(`transformers` + `torch` already live there — nothing new to install) and
exits before the render starts, so its ~10.3 GB never stacks on a render peak.

### 41.3 Measured on this node

| | |
|---|---|
| Planning (subprocess, incl. model load) | ~35–45 s per job |
| Planner peak VRAM | 10.3 GB, released before rendering |
| 20 s director video, end to end | ~78 s |
| 30 s | ~112 s |
| 60 s (two chained sections) | ~203 s |

All on the distilled tier — **the guided tier is not involved and must not be
enabled for this feature**; its 5 s pass ceiling would chop a conversation
into twelve seams.

### 41.4 The VPS half (owner-performed) — what actually turns it on

The VPS checkout carries the deliberate uncommitted `runtime: mock` → `ltx`
edits (§16), and this push touches `text-to-video.yaml`, so a bare
`git pull --ff-only` will refuse. The sequence that preserves the routing is
§39's:

```bash
cd /opt/zolexai
runuser -u zolexai -- git stash
runuser -u zolexai -- git pull --ff-only    # → 0a1025f (or later)
runuser -u zolexai -- git stash pop         # re-applies the runtime flips;
                                            #   this release only ADDS a
                                            #   settings key, so it merges clean
runuser -u zolexai -- git status            # expect the same three YAMLs
                                            #   modified, nothing else
```

Then the §14 image rebuild, because the API bakes `workflow-definitions/` in
(no migrations — this release has no schema changes):

```bash
cd /opt/zolexai
COMPOSE="docker compose \
  --env-file /opt/zolexai/.env \
  -f infrastructure/compose/docker-compose.prod.yml"

$COMPOSE build api web
$COMPOSE up -d --no-deps --force-recreate api web
sleep 8
$COMPOSE ps api web
curl -sS http://127.0.0.1:8100/api/v1/health
```

**`web` is rebuilt too, and that is not optional here.** The frontend renders
the mode toggle from its own build-time read of the YAML
(`catalog.server.ts`), and it seeds the client query cache from that read — a
web image built before this commit would ship `prompt_modes: false` to every
browser and the API's correct answer would never displace the stale seed.
That is exactly how `settings.lyrics` shipped wrong on 17 Aug.

Verify the switch actually flipped — inside the container, not on disk:

```bash
docker exec zolexai-prod-api-1 grep -n -A8 '^settings:' \
  /workflow-definitions/text-to-video.yaml     # must show prompt_modes: true

curl -sS http://127.0.0.1:8100/api/v1/workflows \
  | python3 -c 'import json,sys; print([w["settings"] for w in json.load(sys.stdin)["workflows"] if w["id"]=="text-to-video"])'
                                               # must contain prompt_modes: True
```

Gate before telling anyone: one real Text to Video job through zolexai.com in
**Idea (Director)** mode, and one in **Standard** — with sound on, eyes and
ears on both. Standard is the regression check; its request is byte-identical
to what it was before this release.

Rollback of the VPS half is a YAML edit, not a redeploy: set
`settings.prompt_modes: false` in `text-to-video.yaml`, rebuild `api` and
`web`. The toggle disappears and every in-flight director job still completes,
because the worker keeps serving the parameter it was already given.

---

## 42. Deploy: automatic lyrics in the customer's language (19 Aug 2026)

`725fcc5` → `6f3c60c` on the GPU node. **Worker-only — the VPS needs nothing.**
Unlike §39 and §41 there is no second half: no workflow YAML changes, no API or
web rebuild, no new public `settings` flag. Music is already routed to this box
(§12), so the release is live the moment the worker restarts.

### 42.1 What it fixes, and how often it was firing

Blank lyrics plus any language but English **failed the job outright**. Not a
missing feature — a deliberate refusal in `MusicAdapter`, correct in intent
(English lyrics labelled Spanish is worse than an error) but with nothing
multilingual behind it.

This was not theoretical. Every non-English music job on this box had failed:

```text
2026-08-17T14:39:54   lyrics writer cannot write 'es'
2026-08-17T15:08:58   lyrics writer cannot write 'es'
2026-08-17T15:21:22   lyrics writer cannot write 'es'
2026-08-18T12:04:14   lyrics writer cannot write 'es'
```

Of the music jobs that reached `music_planned`, **six selected Spanish and one
English**. The feature was failing for the large majority of the people using
it.

### 42.2 The deploy

```bash
# key first — the release is inert without it, and adding it after the restart
# means a second restart
cp /workspace/zolexai/.env.gpu-worker{,.bak-$(date +%Y%m%d-%H%M%S)}
printf '\nCEREBRAS_API_KEY=%s\n' "$KEY" >> /workspace/zolexai/.env.gpu-worker
chmod 600 /workspace/zolexai/.env.gpu-worker

cd /workspace/zolexai && git pull --ff-only
supervisorctl restart zolexai-worker
```

Verified after restart:

- `worker_draining` reported `active_jobs: 0` — nothing interrupted.
- `worker_ready`: `ltx-6000-1`, runtimes `["ltx", "music"]`, all six workflows.
- The chain resolves in `.venv-worker` as `['cerebras', 'template']`,
  `available: True`, supported languages **ANY** (empty set = a language model
  has no list to give). Before the key it was `['en']`.
- All fourteen offered languages written from the deployed checkout against the
  live API — `scripts/lyrics_smoke.py`, 0 problems, ~1.5 s median.

Rollback is the previous commit plus a restart; the key can stay, it is inert
without the code:

```bash
cd /workspace/zolexai && git checkout 725fcc5
supervisorctl restart zolexai-worker
```

### 42.3 `CEREBRAS_API_KEY` is a per-node prerequisite

A node without it does not fail — it **silently reverts to the old behaviour**:
the chain reports the hosted writer unavailable, falls back to the English-only
template bank, and refuses every other language exactly as before. That is the
designed degradation, and it is also the trap: the release looks deployed and
changes nothing. Check it directly rather than inferring it from the commit:

```bash
grep -c '^CEREBRAS_API_KEY=.' /workspace/zolexai/.env.gpu-worker   # must be 1
```

The key is worker-only. It never reaches the browser, job parameters, job
metadata, or the logs — the logs record latency, token counts and the model
name, none of which identify the credential.

### 42.4 Two things measured here that documentation would not have told us

**Reasoning tokens are billed against `max_completion_tokens`.** Cerebras serves
exactly two public models. Both write all fourteen languages — but
`gpt-oss-120b` spent 386–696 tokens on a hidden reasoning channel against a
540-token budget and returned **empty content with `finish_reason: stop` and no
error of any kind**. Nine of fourteen languages "failed" with no diagnosable
cause until the raw response was dumped. `_REASONING_HEADROOM` in
`worker/music/cerebras.py` reserves for it. Any reasoning model on any
OpenAI-shaped API has this trap; the symptom is silence, not an error.

The default is `gemma-4-31b` — it reaches the same result without needing the
reserve. `CEREBRAS_LYRICS_MODEL` overrides it (`CEREBRAS_AI_MODEL` is accepted
as an alias, because a deployment already used that name and config that is
silently ignored is worse than config that is rejected).

**A capitalised word is not a proper noun.** `salient_details` collects every
capitalised token as a must-keep detail, so "Two people falling in love" made
`Two` mandatory and the model wedged it verbatim into a Spanish chorus —
`estamos Two en un baile fiel`. Only visible in a real song; every mocked test
passed. `singable_details` filters genre words, style words and English number
words, applied once in the adapter so the writer and the reviewer agree on what
a detail is.

### 42.5 Gate before telling anyone

One real Music job through zolexai.com with **Lyrics blank** and **Lyrics
language: Spanish** — ears on it, confirm it sings Spanish. Then one with
pasted lyrics and one instrumental as the regression checks: neither may call
the lyrics service at all, and `grep -c cerebras_lyrics_attempt` over that
window must be `0` for both.

---

## 43. Deploy: Director / Idea mode for Image to Video (19 Aug 2026)

`e716a08` → `a1fd9a8` on the GPU node. Same two-half shape as §41, and the
same warning applies with more force: **the activation switch is a public
`settings` flag**, so the worker half deployed below is inert — invisible,
not merely unrouted — until the VPS rebuilds `api` **and** `web`.

Research and every measurement:
[`research-2026-08-19-i2v-director.md`](./research-2026-08-19-i2v-director.md).

### 43.1 The GPU half (done)

```bash
cd /workspace/zolexai && git pull --ff-only
supervisorctl restart zolexai-worker
```

Verified after restart: `worker_draining` reported `active_jobs: 0` on every
restart (the box is serving real traffic — a customer director job in Spanish
had run at 09:10 the same morning); `worker_ready` as `ltx-6000-1` with all
six workflows; `wants_director` true for image-to-video and text-to-video and
false for extend/v2v/music-video; the Text-to-Video planning brief asserted
byte-identical to the anchored one's prefix.

Rollback is the previous commit plus a restart:

```bash
cd /workspace/zolexai && git checkout e716a08
supervisorctl restart zolexai-worker
```

### 43.2 What the deploy measurement found, and why there are four commits

The feature was deployed, measured on the box, **found broken in the exact way
it was designed to prevent**, fixed, and re-measured. That sequence is the
point of deploying to a GPU before telling anyone.

The planning brief tells the model, in capitals, never to invent visible
detail — the image is the visual truth. Given a photograph of a woman in a
yellow raincoat beside a silver robot on a park bench, the hosted planner
returned a "beige linen blouse", a robot in "white ceramic plating", and the
scene moved to "a modern minimalist study"; the local checkpoint invented a
silver jumpsuit and a futuristic classroom. Both then wrote those inventions
into `continuity` — the block the compiler restates at the end of every
section. The anti-drift mechanism was repeating a description of a different
woman, once per pass, against the customer's own image.

`_ground_visual_claims` now enforces in code what the prompt only asked for: a
visual claim survives only if the supplied text supports it (the idea, plus
the measured image facts when the vision step ran). Re-measured on the same
input: appearances empty, scene replaced by a pointer at the opening frame,
and the single grounded fact — the count of people — surviving.

### 43.3 The image-facts step is OFF and needs no node prerequisite

`DIRECTOR_VISION_ENABLED` defaults to false. Nothing to download: it would
reuse the §41.2 Gemma checkpoint. It is off because whether that checkpoint
accepts image input is unmeasured, and after 43.2 it is also **not load
bearing** — an ungrounded description is discarded rather than rendered, so
the feature is correct without it. Turning it on is a measurement session
first, then one environment variable.

### 43.4 Measured on this node

| | |
|---|---|
| Planning (Cerebras, hosted) | ~1 s |
| Planning (local fallback, incl. model load) | ~30 s |
| 10 s I2V director video, 2 chained sections, end to end | **52.9 s** |
| Output verified | h264 1024x576 + aac, 9.96 s, both streams |
| Plan rejections (hosted) | 0 |

Distilled tier throughout — **the guided tier must not be enabled for this
feature**, same as §41.3.

### 43.5 The VPS half — DONE 19 Aug 2026

Performed by the owner the same day, from `725fcc5` (the VPS was thirteen
commits behind, so this also brought the auto-lyrics release of §42 — which
needs nothing here, being worker-only). Both YAMLs auto-merged and the stash
popped clean, because the pull touches only the `settings:` blocks while every
local edit is in `execution:`:

```text
Auto-merging workflow-definitions/image-to-video.yaml
Auto-merging workflow-definitions/text-to-video.yaml
Dropped refs/stash@{0}
```

Verified after the rebuild — all four, and the fourth is the one that matters:

```text
API image  /workflow-definitions/image-to-video.yaml   prompt_modes: true
routing    same file                                   runtime: ltx
API serves GET /api/v1/workflows                       'prompt_modes': True
WEB image  /app/workflow-definitions/image-to-video…   prompt_modes: true
```

The procedure, for the next time. Identical in shape to §41.4, and this
release touches `image-to-video.yaml` and `text-to-video.yaml` (a comment
only), so the stash dance still applies:

```bash
cd /opt/zolexai
runuser -u zolexai -- git stash
runuser -u zolexai -- git pull --ff-only    # → a1fd9a8 (or later)
runuser -u zolexai -- git stash pop         # re-applies the runtime flips
runuser -u zolexai -- git status            # expect the same YAMLs, nothing else
```

Then the §14 rebuild — no migrations, this release has no schema changes:

```bash
cd /opt/zolexai
COMPOSE="docker compose \
  --env-file /opt/zolexai/.env \
  -f infrastructure/compose/docker-compose.prod.yml"

$COMPOSE build api web
$COMPOSE up -d --no-deps --force-recreate api web
sleep 8
$COMPOSE ps api web
curl -sS http://127.0.0.1:8100/api/v1/health
```

**`web` is not optional**, for the reason spelled out in §41.4:
`catalog.server.ts` reads the YAML at build time and seeds the client query
cache from it, so a stale web image ships `prompt_modes: false` to every
browser and the API's correct answer never displaces it.

Verify the switch inside the container, not on disk:

```bash
docker exec zolexai-prod-api-1 grep -n -A12 '^settings:' \
  /workflow-definitions/image-to-video.yaml    # must show prompt_modes: true

curl -sS http://127.0.0.1:8100/api/v1/workflows \
  | python3 -c 'import json,sys; print([w["settings"] for w in json.load(sys.stdin)["workflows"] if w["id"]=="image-to-video"])'
                                               # must contain prompt_modes: True
```

### 43.6 Gate before telling anyone

Through zolexai.com, with sound on:

1. **Image to Video, Idea (Director)** — upload a photo of one or two people,
   idea "they discuss <something>". Then look at the thing no test can check:
   **is it the same person as the photograph, at the start AND at the end?**
   That is the whole promise of the mode and the one open question §43.2's fix
   does not close by itself.
2. **Image to Video, Standard** — the regression check. Its request is
   byte-identical to what it was before this release.
3. **Text to Video, Idea (Director)** — the other regression check. This
   release refactored shared director code, and the T2V brief is asserted
   unchanged; confirm a planned scene still speaks.

Rollback of the VPS half is a YAML edit, not a redeploy: set
`settings.prompt_modes: false` in `image-to-video.yaml`, rebuild `api` and
`web`. The toggle disappears from Image to Video, Text to Video keeps its own,
and any in-flight director job still completes because the worker keeps
serving the parameter it was already given.

---

## 44. Deploy: V2V reference identity + frame-exact seam delivery (19 Aug 2026)

Full findings and measurements:
[`research-2026-08-19-v2v-reference-identity-and-lipsync.md`](./research-2026-08-19-v2v-reference-identity-and-lipsync.md).
Commits `2ba698e` (worker), `7d99e41` (YAML + API test), `e6e26dc` (web).

### 44.1 What changed, per half

**Worker (DONE 19 Aug, this box):** frame-exact section delivery on the V2V
path — active for every V2V job the moment the worker restarted, no flag —
and `v2v_reference_identity`, which only activates when the API sends the
flag AND the job carries a reference image. Deployed by `git pull --ff-only`
to `e6e26dc` and `supervisorctl restart zolexai-worker`; the worker drained
(0 active jobs), came back `worker_ready` with all six workflows, and the
checkout imports the identity constants.

**VPS (PENDING — being done by hand):** the flag rides in the baked YAML, so
this is the standard §16 dance, with the extra care that
`video-to-video.yaml` now changes in Git UNDER the local edits:

1. `cd /opt/zolexai && git diff` — read the six locally-modified YAMLs FIRST
   (§16: two edits each — runtime flip + mock-output deletion).
2. `git stash && git pull && git stash pop` — video-to-video.yaml is the one
   that may conflict; its resolution must keep BOTH the local runtime edits
   AND the new `v2v_reference_identity: true` + the new help line.
3. Rebuild **api AND web** (`docker compose ... build api web` then
   `up -d --no-deps --force-recreate api web`) — api for the YAML, web for
   the Reuse-Settings inputs fix and the SSR catalog's copy of the help text.
4. Verify inside the container:
   `docker exec zolexai-prod-api-1 grep -A2 v2v_reference_identity /workflow-definitions/video-to-video.yaml`
   and that `/api/v1/workflows/video-to-video` serves the new help line
   ("The person in the result follows this image").

### 44.2 Node prerequisite

Identity jobs matte every pass: BiRefNet in the LTX environment
(`scripts/person_matte.py`, §40). Already proven on THIS box — the 19 Aug
matrix runs used it. A future node without the weights fails identity jobs
with a clear internal detail rather than delivering the source person; that
is deliberate (`test_a_matting_failure_fails_the_job_not_the_promise`).

### 44.3 Gate before telling anyone

Through zolexai.com, with sound on, a V2V job with a clip of ONE person
speaking + a reference photo of a different person:

1. The output person resembles the reference at the START and the END.
2. The mouth still articulates the source's speech — no frozen face.
3. A V2V job WITHOUT a reference behaves as before (restyle, source audio).
4. Reuse Settings on the finished job keeps the reference image thumbnail.

Known limits, shipped knowingly: a multi-person source has ALL people
re-imagined toward the reference (help text warns); identity adds ~27s of
matting per pass; the recorded recommendation for a consent checkbox
(reference-engine pattern) is NOT yet implemented.

Rollback, worker half: `git revert` or checkout `883ff86`, restart worker.
Rollback, VPS half: remove `v2v_reference_identity: true` from the baked
YAML and rebuild api — the worker then never receives the flag and V2V is
exactly the 18 Aug transform engine (the seam fix stays, and should).

---

## 45. Deploy: 60-second continuity + Director-aware Extend (20 Aug 2026)

Two-half deploy in the §41/§43 shape, with one structural difference worth
reading twice: **this release has real API-side code** — the Extend lineage
walk lives in `apps/api/app/services/generation.py` — so the VPS `api`
rebuild is not just re-baking YAML this time. Research and every measurement:
[`research-2026-08-20-60s-continuity-and-director-extend.md`](./research-2026-08-20-60s-continuity-and-director-extend.md).

What the release does:

- **60s T2V/I2V renders as two 30-second sections** instead of one 60s pass
  (`execution.max_segment_seconds: 30` in both YAMLs). Measured reason: a
  single 60s pass returned a departed man to the kitchen for the final twelve
  seconds while the audio said he was gone; 30s passes are the measured-clean
  regime, and 2×30 is ~33% FASTER than 1×60 (191s vs 285s). 5/15/30s argv is
  byte-identical — pinned by test.
- **Departures are state** (`DirectorEvent.exits`): a character who leaves is
  out of every cast and constancy sentence from that point on, the scene is
  restated as who REMAINS, and people-count continuity facts are dropped.
- **Extend is Director-aware**: the API walks `source asset → producing job`
  at creation, stores `director_lineage` (mode, language, idea, prior
  seconds, identity image) in the extend job's params, and the worker plans a
  CONTINUATION — same language, story moves forward, no re-telling, the
  original I2V upload carried as the `identity_image` identity anchor.
  Works retroactively for every Director video ever generated; a source with
  no ancestry extends byte-identically to before.

### 45.1 Split-deploy behaviour (safe in either order)

Worker first, API-half pending (the state after 45.2): 60s jobs still arrive
without the ceiling → still one pass, but the presence-aware captions are
already active (the caption contradiction is gone even single-pass); extends
arrive without lineage → byte-identical standard extends. Nothing breaks;
the full feature turns on when the VPS api image is rebuilt.

### 45.2 The GPU half

```bash
cd /workspace/zolexai && git pull --ff-only
supervisorctl restart zolexai-worker
```

Verify: `worker_draining` reports `active_jobs: 0`; `worker_ready` lists all
six workflows; and in `.venv-worker`,
`from worker.director import continuation_lineage` imports.

Rollback: checkout the previous commit, restart.

### 45.3 The VPS half — DONE 20 Aug 2026

Performed by the owner the same day, from `013c6d1` — which meant the pull
ALSO brought the §44 V2V reference-identity release whose VPS half was
pending, plus the `e6e26dc` web fix (Reuse Settings input restoration). So
`web` was rebuilt after all (for `e6e26dc`, not for this release), and the
rebuild activated `v2v_reference_identity: true` alongside this release's
ceiling. All in-container checks passed: `max_segment_seconds: 30` in both
generation YAMLs, `runtime: ltx` survived the stash dance, the v2v flag
baked. The procedure, for the next time:

The §16 stash dance (SIX locally-modified YAMLs, TWO edits each — read the
diff before stashing), because this push touches `text-to-video.yaml` and
`image-to-video.yaml`. Both edits land inside `execution:` this time, next to
the local `runtime:` flips — they auto-merge because the stash's hunks and
the pull's hunks touch different lines, but READ the `git stash pop` output
and run the verification below regardless.

```bash
cd /opt/zolexai
runuser -u zolexai -- git --no-pager diff workflow-definitions/ | grep -E "^(diff|[+-][^+-])"
runuser -u zolexai -- git stash
runuser -u zolexai -- git pull --ff-only
runuser -u zolexai -- git stash pop
runuser -u zolexai -- git status --short   # same six YAMLs + two untracked files
grep -n "max_segment_seconds\|runtime:" workflow-definitions/text-to-video.yaml
grep -n "max_segment_seconds\|runtime:" workflow-definitions/image-to-video.yaml
# each must show BOTH max_segment_seconds: 30 AND runtime: ltx
```

Then rebuild **api** (§14). **`web` is NOT required this time** — the YAML
change is `execution:`-private and stripped from every public response, and
no web source changed. Rebuilding it anyway is harmless.

```bash
cd /opt/zolexai
COMPOSE="docker compose --env-file /opt/zolexai/.env -f infrastructure/compose/docker-compose.prod.yml"
$COMPOSE build api
$COMPOSE up -d --no-deps --force-recreate api
sleep 8 && $COMPOSE ps api
curl -sS http://127.0.0.1:8100/api/v1/health

docker exec zolexai-prod-api-1 grep -n "max_segment_seconds" \
  /workflow-definitions/text-to-video.yaml /workflow-definitions/image-to-video.yaml
docker exec zolexai-prod-api-1 grep -n -A2 '^execution:' \
  /workflow-definitions/image-to-video.yaml   # runtime: ltx survived
```

No migrations — the lineage lives in the existing `request_params` JSONB.

### 45.4 Incident on go-live night: the two-image decoder cell (FIXED)

The first production 60s Image-to-Video after activation (job `2502edeb`)
crashed in pass 2: `CUBLAS_STATUS_INTERNAL_ERROR` → illegal memory access in
the VAE decoder. The new geometry made I2V's second pass carry TWO
conditioning images for the first time at 720 frames — the seam frame plus
the mid-window identity anchor — and every "720 conditioned = safe" cell in
the tables was a SINGLE-image measurement.

Probed the same night with `frame_probe2.py` (adapter's own command builder,
production two-image shape), 1024x576:

    FAIL: 720 (deterministic — the production crash), 736 (the
          render-extra-and-trim dodge does NOT work for this cell family)
    PASS: 120, 240, 360

Fix: `_TWO_IMAGE_SAFE_FRAMES` — the identity anchor rides only measured
two-image counts; everywhere else the pass carries the seam frame alone
(`identity_anchor_skipped` in the log), which is the single-image shape the
60s validation ran clean. Consequence to know when reading an
identity-drift report: **30-second chain passes carry no photo anchor** —
identity there rides the seam frame and the captions. Growing the set is a
`frame_probe2.py` measurement, not an opinion.

### 45.5 Gate before telling anyone

Through zolexai.com, sound on:

1. **T2V Director, 60s**, an idea where someone LEAVES partway ("...midway
   through, X walks out and does not come back"). Watch the second half: the
   departed person stays gone, no flash back, dialogue never restarts. The
   worker log shows `passes: 2` in `longform_plan` and a `director_sections`
   line with `departed` filled for section 2.
2. **T2V standard, 30s** — regression; request and render byte-identical.
3. **Extend a Director video +10s** — the extension continues the
   conversation in the same language (job detail shows
   `parameters.director_lineage`); extend the RESULT again and check
   `prior_seconds` accumulated.
4. **Extend a plain uploaded video** — regression: no `director_lineage` in
   the job, behaviour unchanged.

Rollback, worker half: previous commit + restart. Rollback, VPS half: revert
the two `max_segment_seconds` lines in the baked YAML and rebuild api (60s
returns to single-pass); the lineage code is inert for any video without
Director ancestry and needs no separate switch.
