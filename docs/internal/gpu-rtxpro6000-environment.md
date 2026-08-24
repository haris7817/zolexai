# RTX PRO 6000 WS — machine inventory

**Recorded:** 24 August 2026, 09:04 UTC · Vast.ai instance `C.48538452`
**Measured on the machine**, not copied from the listing.

---

## 1. Identity and licence-relevant location

| | |
|---|---|
| Instance / container | `48538452` · host `9669e7b98a1f` |
| Machine ID / host ID | `148006` / `657491` |
| Public IP | `149.106.247.248` (static), SSH on port `40008` |
| **Geolocation (provider record)** | **`Israel, IL`** |
| Corroboration | ipinfo: Petah Tikva, Central District, AS6810 Bezeq |
| Price | `$1.296142779/hr` |
| Reliability | `0.9063` |
| Verification | `verified` |

**Licence consequence.** Israel is **not** among the MiniMax H3 Excluded
Territories (EU, UK, South Korea, US), so the community licence covers
deployment here by default and **no application is required**. This was taken
from Vast's own machine record via `vastai show instance`, not inferred from IP
geolocation or the billing country. See
[h3-rtxpro6000-runtime-research.md](./h3-rtxpro6000-runtime-research.md) §1.1 —
our previous documents had this backwards and would have blocked here.

## 2. GPU

```text
NVIDIA RTX PRO 6000 Blackwell Workstation Edition
97887 MiB total (95.6 GB) · 2 MiB in use · no compute processes
driver 595.84 · compute capability 12.0 (sm_120) · P8 idle · 32 °C
power limit 600.00 W
```

**The 600 W limit confirms the full Workstation Edition, not the 300 W Max-Q.**
That was the open question about which variant Vast's "WS" label meant: this is
the full-throughput card, so the production LTX ceilings should transfer and the
34 s / 121-frame reference is the right yardstick.

`compute_capability 12.0` is Blackwell — NVFP4 is available, and **any wheel
must be CUDA ≥ 12.8**; a cu124 build installs cleanly and then fails at runtime
with "no kernel image is available".

## 3. Host

```text
Ubuntu 24.04.4 LTS · kernel 7.0.0-30-generic
Intel Core Ultra 7 270K Plus · 24 cores / 24 threads (no SMT)
RAM   125 GiB total · 120 GiB available
Swap  8.0 GiB          ← small; pinned memory cannot use it anyway
shm   62 GiB
Disk  879 GB overlay, 874 GB free
Python 3.12.3 · uv 0.12.5 · CUDA toolkit 12.8.93 (nvcc present)
```

CUDA install is **partial but sufficient**: nvcc, dev headers, cudart, nvrtc,
cuBLAS, cuDNN, cuFFT, cuSPARSE, cuSOLVER, cuRAND, NCCL, NPP, nvJPEG all present.
Forward-compat libs exist but are correctly disabled (host driver is newer).

Preinstalled and useful: `git`/`git-lfs`, `rclone`, `rsync`, `magic-wormhole`,
`ffmpeg`, `jq`, `conda`, `huggingface-hub` CLI, `vastai` CLI, build-essential.

**No HuggingFace token is configured** (`credentials.huggingface: false`).

## 4. Persistence — read this before trusting the box

```text
workspace_is_volume : false
Volumes             : none
```

`/workspace` is **not** a volume. Nothing here survives instance destruction.
Combined with the fact that both feature branches are still local-only on the
dev machine, no artefact produced on this box is backed up by anything.

**Rule for this rental:** results leave the box after each phase. Upload is
~390 Mbps, so it costs minutes.

## 5. Directory layout created

```text
/workspace/src/          LTX-2 upstream, our repo when it arrives
/workspace/envs/         ltx25/ and h3/ — separate, never merged
/workspace/models/ltx25/ /workspace/models/h3/
/workspace/cache/huggingface/     HF_HOME, same filesystem so it can symlink
/workspace/results/      benchmark output, pulled off after each phase
/workspace/logs/
```

## 6. Disk budget against 874 GB free

| Item | Estimate |
|---|---:|
| LTX-2 source + env | ~8 GB |
| LTX checkpoints | ~19 GB |
| H3, **selective** fetch (shared components once) | ~144 GB |
| H3 runtime env | ~10 GB |
| Benchmark outputs + intermediates | ~80 GB |
| ACE-Step, logs, headroom | ~40 GB |
| **Total** | **~301 GB**, leaving ~573 GB |

The trap to avoid: the H3 repository is **~354 GB** because the `fl2va` and
`ref2va` folders duplicate the shared transformer, text encoder and VAEs. A
naive `snapshot_download` plus a copying cache would approach 708 GB and leave
nothing. Fetch with explicit `allow_patterns`, `HF_HOME` on the same filesystem.

## 7. LTX baseline restored

```text
/workspace/src/LTX-2 @ 400fd31054597515f47125691032c04b1c3ee24e
working tree clean
```

`400fd31` is **still the tip of upstream `main`** (16 Aug 2026), so our audited
baseline and current upstream are the same commit. Phase 4 needs no
reconciliation.

### 7.1 A defect in the runbook, found on the first command

The GPU-day runbook's Phase 1 integrity check is:

```bash
grep -c "materialize only the tensors" \
  packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py   # must print 1
```

Against a clean checkout of the audited baseline it **prints 0**. The file
exists (55,772 bytes) and the tree is clean — the string simply does not occur
anywhere in the repository at `400fd31`.

The nearest real code is
`packages/ltx-core/src/ltx_core/model/transformer/model.py:161`, which discusses
parameters "created on the meta device and only later materialized" — the same
area as the `to_empty` green-screen bug recorded from the old rig.

So the check as written is invalid: it would have fired a spurious **STOP** on
the first command of GPU day. Whether our green-screen patch is still needed
against this upstream is an empirical question, and the Phase 6 smoke render
answers it definitively — a green or black output means it is.

**Action:** the grep gate should be replaced by the smoke render's visual
inspection, which is what actually tests the condition.

## 8. Reproduce this machine

```bash
ssh -p 40008 -i ~/.ssh/zolexai_vast root@149.106.247.248
# image: vastai/base-image_cuda-12.8.1-auto/jupyter
# env:   /venv/main (conda) or uv at /.uv
export HF_HOME=/workspace/cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # SGLang recommends this for H3 decode
```

## 9. Open at time of writing

1. **Our repository is not on the box** — `dual-engine-benchmark-prep` does not
   exist on `origin`, so it cannot be cloned. Phases 3, 6 and everything
   downstream are blocked on that push.
2. LTX Python environment not yet built; checkpoints not yet downloaded.
3. H3 not yet installed; no HF token on the box.
