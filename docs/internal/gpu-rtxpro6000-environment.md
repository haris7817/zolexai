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

## 9. Phase 6 — LTX verified on this card

All timings are wall-clock from `scripts/ltx_smoke.py`, which drives the real
`LtxAdapter` against the real model with no platform, no API and no routing.
Every output was inspected, not merely validated.

| Smoke | Wall | Output | Result |
|---|---:|---|---|
| **T2V** 5 s | **28.2 s** | 1024x576, 24 fps, 5.013 s | PASS |
| **I2V** 5 s | 24.7 s | 1024x576, 4.97 s | PASS |
| **Extend** +5 s | 24.7 s | 1024x576, **10.004 s** from a 5 s source | PASS |
| **V2V** default | 25.9 s | 1024x576, 5.013 s | ran, but see §9.2 |
| **V2V** transform engine | **61.8 s** | 1024x576, 5.0 s | PASS, real restyle |

**Against the reference.** Production recorded ~34 s for 121 frames at
1024x576; this card does the same work in **28.2 s**, about 17% faster. That is
consistent with the 600 W Workstation Edition and means the recorded ceilings
are conservative here, not optimistic.

### 9.1 The T2V output, inspected

```text
1024x576  yuv420p  24 fps  120 frames  5.013 s  834 KiB
mean luma Y          = 80.80     (black would be ~16)
mean interframe delta = 2.256    (0.000 would be frozen)
max  interframe delta = 3.204
```

Frames 0, 60 and 119 were pulled off the box and looked at. Photographic dawn
light on open water, shoreline and rock detail, ripples evolving frame to
frame, scene coherent from first to last with no drift into artefact. **Not
green, not black, not corrupted** — which settles the `to_empty` question
empirically and replaces the runbook's broken grep gate (§7.1).

120 delivered frames is not an 8k+1 violation. `conforming_frames()` snaps the
request up to the lattice — 121 for a 5 s ask — and the deliverable is then
trimmed to the requested duration. The model received a conforming count; had
it not, it would have crashed rather than rendered.

### 9.2 The finding that matters for the benchmark

**The default V2V path did not restyle.** Given "as a charcoal sketch, heavy
graphite texture", it returned a clip visually indistinguishable from its
source: same dawn colour, same photographic look, no graphite anywhere. The
smoke reported `SMOKE TEST PASSED` because the adapter and model agreed and the
file validated — the failure is only visible in the pixels.

This is **not a defect**. The default path is still-conditioned restyle at
`v2v_structure_strength: 0.45`, which anchors hard to source stills. The strong
restyle is the opt-in transform engine (`v2v_engine: transform`,
`ltx_pipelines.ic_lora` + Union Control LoRA). Run with that, the same prompt
stripped the colour to monochrome while holding the geometry — shoreline, water
and composition unchanged.

**Consequence:** C-group (6 cases) and D-group (5 cases) are 11 of 41 benchmark
cases, and on the default path they would measure a near-passthrough rather than
a restyle. The benchmark harness already sets `v2v_engine: transform`; this
confirms that is required, not preferential. It costs **2.4x** (61.8 s against
25.9 s), which is the documented LoRA rule — a LoRA drops quantization entirely
and the unquantized 22B is fitted instead.

The transform restyle is monochrome but reads as a desaturated photograph
rather than a drawing, so stylistic strength for "charcoal sketch" is partial.
That is a quality question for the benchmark to score, not a gate.

### 9.3 Weights actually required

The README's five files are **not** sufficient for our adapter.

| File | Size | Why |
|---|---:|---|
| `distilled-transformer-nvfp4` | 17.44 GB | `ltx_quantization` defaults to `nvfp4-prequant` |
| `distilled-transformer-bf16` | 39.13 GB | any LoRA path drops quantization |
| `gemma4-12b-with-proj` | 24.46 GB | text encoder |
| `video-vae` / `audio-vae` | 1.37 / 0.34 GB | |
| `latent-spatial-upscaler-x2` | 0.93 GB | |
| `duration-head` | <0.01 GB | |
| `ltx-2.3-22b-ic-lora-union-control-ref0.5` | 0.61 GB | transform engine — **from `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control`, a different repo** |

Total 84 GB. Disk after: 96 GB of 879 GB.

### 9.4 Two more runbook gaps

- **`ltx-kernels` is never mentioned.** NVFP4 refuses to run without it, so the
  first real render dies. Building it is not optional on this path.
- **Its build needs the venv toolchain, not the system one.** The system nvcc is
  12.8 and torch is cu132, which fails with a CUDA version mismatch. The 13.2
  nvcc ships inside the venv at
  `.venv/lib/python3.11/site-packages/nvidia/cu13`. And upstream's own hint says
  `TORCH_CUDA_ARCH_LIST='10.0'` — datacenter Blackwell. **This card is sm_120**,
  so it must be `12.0` or the kernels build for an architecture the card cannot
  run.

```bash
export CUDA_HOME=/workspace/src/LTX-2/.venv/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=12.0
uv pip install -e packages/ltx-kernels --no-build-isolation
```

### 9.5 Not yet run

Music video / A2V. It needs a real track, and the golden `benchmark-song` is
still unacquired — generating it with our own ACE-Step is the next GPU task.

## 10. Open at time of writing

1. **Music video / A2V not smoked** — needs a real track. Generating the golden
   `benchmark-song` with our own ACE-Step is the next GPU task, and it doubles
   as asset acquisition (2 of the 19 golden assets).
2. **H3 not installed.** Licence is clear (§1) and the repo is ungated, so
   nothing blocks it but the work itself.
3. **Golden media 0 of 19 acquired** — still the calendar-time blocker for the
   real benchmark, unchanged by anything measured here.

Resolved during this session: the repository is on the box at `c16f82d`, a
HuggingFace token with gated-repo scope is installed at
`/workspace/cache/huggingface/token`, and both feature branches now exist on
`origin` so the box is no longer anybody's only copy.
