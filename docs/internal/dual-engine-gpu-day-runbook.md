# GPU day — the checklist

**Purpose:** the next session should execute, not design. Everything below is
decided; what is left is measurement.

**Branch:** `dual-engine-benchmark-prep`. **Nothing here deploys.** Production
routing is untouched and stays untouched until the decision table in §Phase 9
exists.

Companions: [ltx-h3-hybrid-benchmark.md](ltx-h3-hybrid-benchmark.md) (why the
hybrid is shaped the way it is), [golden-benchmark-pack.md](golden-benchmark-pack.md)
(the frozen inputs), [ltx-h3-comparison-framework.md](ltx-h3-comparison-framework.md)
(scoring and cases), [research-ltx25-zolexai-audit.md](research-ltx25-zolexai-audit.md)
(the verified LTX baseline).

---

## Before the card is switched on

The golden media does not exist yet. **Shoot or generate it first** — it is
the only remaining blocker that costs calendar time rather than GPU time, and
arriving at a rented GPU without it wastes the rental.

```bash
uv run python apps/worker/scripts/golden_pack.py --status     # 19 assets, all pending
# ... acquire, then for each file:
uv run python apps/worker/scripts/golden_pack.py --hash benchmarks/assets/<path>
# paste the hash into assets.manifest.json, set acquisition: acquired, record provenance
uv run python apps/worker/scripts/golden_pack.py --verify     # must exit 0
```

The song is the long pole: generate it with our own ACE-Step service so the
rights are unambiguous and the lyric sheet is known rather than transcribed.

---

## Phase 0 · Machine verification

Record every value; they go into every `RunRecord`.

```bash
nvidia-smi                      # GPU model, driver, total VRAM
nvcc --version                  # CUDA toolkit
free -g && df -h /workspace     # host RAM, disk (weights are ~100 GB across both engines)
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Gate: at least ~200 GB free disk, and VRAM at or above the purchased spec.

## Phase 1 - Restore and verify LTX

LTX is the baseline; nothing else matters if it moved.

**Everything in this phase was executed on a rented RTX PRO 6000 Blackwell
Workstation Edition on 24 August 2026, and the corrections below are what that
run produced.** Three of the commands that used to be here were wrong.

```bash
cd /workspace/src/LTX-2 && git log -1        # must be 400fd31 - still tip of upstream main
uv run python -c "import natten; print(natten.__version__)"     # 0.21.7
```

> **Removed: the `grep -c "materialize only the tensors"` gate.** That string
> occurs **nowhere** in the audited baseline. The file it named exists and the
> tree is clean, so the check printed `0` and would have fired a spurious STOP
> on the first command of GPU day. The condition it was reaching for - the
> `to_empty` green-frame regression - is tested properly by looking at the
> render below, which is the only thing that ever actually tested it.

### 1.1 Build `ltx-kernels` - not optional on the NVFP4 path

`ltx_quantization` defaults to `nvfp4-prequant`, and NVFP4 refuses to run
without the compiled extension: *"ltx-kernels not built; NVFP4 quantization
requires the nvfp4 extension."* The first real render dies here otherwise.

Two traps, both hit on this machine:

- **The system nvcc is the wrong one.** It is CUDA 12.8 while torch is cu132,
  and the build fails with a CUDA version mismatch. The 13.2 compiler ships
  *inside the venv*.
- **Upstream's own hint says `TORCH_CUDA_ARCH_LIST='10.0'`** - that is
  datacenter Blackwell (B100/B200). The RTX PRO 6000 WS is **sm_120**. Build
  with `10.0` and the kernels are for an architecture this card cannot run.

```bash
cd /workspace/src/LTX-2
export CUDA_HOME=$PWD/.venv/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH          # nvcc 13.2, not the system 12.8
export TORCH_CUDA_ARCH_LIST=12.0          # sm_120 - NOT 10.0
uv pip install -e packages/ltx-kernels --no-build-isolation
```

Verified stack: `torch 2.13.0+cu132`, CUDA 13.2, `natten 0.21.7`,
`ltx-kernels 1.2.0`, driver 595.84, compute capability 12.0.

### 1.2 The checkpoints - the README's five are not enough

Our adapter references **nine** assets, and one lives in a different
repository. Downloading only the README set means the transform engine and the
guided tier both fail at run time rather than at setup.

| File | Size | Needed for |
|---|---:|---|
| `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4` | 17.44 GB | the default - `nvfp4-prequant` |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16` | 39.13 GB | any LoRA path (a LoRA drops quantization) |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16` | 24.46 GB | text encoder |
| `vae/ltx-2.5-video-vae-bf16` | 1.37 GB | |
| `vae/ltx-2.5-audio-vae-bf16` | 0.34 GB | A2V |
| `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0` | 0.93 GB | two-stage |
| `model_patches/ltx-2.5-duration-head-bf16` | <0.01 GB | |
| `loras/ltx-2.3-22b-ic-lora-union-control-ref0.5` | 0.61 GB | **transform engine - from `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control`** |
| `diffusion_models/ltx-2.5-22b-dev-transformer-bf16` | - | guided tier, when that is tested |

`Lightricks/LTX-2.5` is `gated: auto` - accept the terms and use a token with
**read access to gated repos**, or every file 401s in a way that looks exactly
like a bad scope.

### 1.3 The render, and its actual CLI

`ltx_smoke.py` takes **the prompt as a positional argument and everything else
from the environment**. It has no `--seconds/--width/--height` flags; the old
invocation in this runbook could never have run.

```bash
cd /workspace/src/zolexai/apps/worker && . .venv/bin/activate
export LTX_REPO_DIR=/workspace/src/LTX-2 LTX_MODEL_DIR=/workspace/models/ltx25
uv run python -m pytest tests/test_ltx_golden.py -q       # 11 shapes, must pass

DURATION=5s ASPECT_RATIO=16:9 python scripts/ltx_smoke.py "a koi pond at dawn"
```

**Measured baseline on this card: 28.2 s** for 5 s at 1024x576 - against the
~34 s recorded on production, i.e. about 17% faster, consistent with the 600 W
Workstation Edition. Treat 34 s as the ceiling and 28 s as this card's figure.

Other modes, same script (note EXECUTION takes a JSON object):

```bash
IMAGE=/path/still.png DURATION=5s   python scripts/ltx_smoke.py "the camera pushes in"
MODE=extend  VIDEO=/path/clip.mp4 DURATION=5s python scripts/ltx_smoke.py "it continues"
MODE=restyle VIDEO=/path/clip.mp4 EXECUTION={"v2v_engine":"transform"} \
    python scripts/ltx_smoke.py "as a charcoal sketch"
MODE=music-video AUDIO=/path/song.wav python scripts/ltx_smoke.py "a singer, hard side light"
```

Measured: I2V 24.7 s - extend 24.7 s (10.004 s out of a 5 s source) - V2V
default 25.9 s - V2V transform 61.8 s.

### 1.4 Inspect the render. Do not trust "PASSED"

The smoke prints `SMOKE TEST PASSED` when the adapter and model agree and the
file validates. That is not the same as the model having done the job - see
1.5, where a restyle passed while changing nothing.

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,width,height,pix_fmt -of default=nw=1 out.mp4
# black? luma near 16.  frozen? interframe delta near 0.
ffmpeg -v error -i out.mp4 -vf "tblend=all_mode=difference,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-" -f null -
```

Then **actually look at frames** - first, middle, last. The 24 Aug reference
render: Y mean 80.80, interframe delta 2.256, photographic dawn water,
coherent start to finish, no green and no black.

Delivered frame counts are trimmed to the requested duration, so a 5 s ask
yields 120 frames while the model received the conforming 121. That is
`conforming_frames()` doing its job, not an 8k+1 violation.

**STOP if** the golden test fails, the render is green/black/frozen, or
wall-clock is far from 28 s on this card.

### 1.5 V2V: the default path does not restyle

**This is a benchmark-validity finding, not a tuning note.**

Given *"as a charcoal sketch, heavy graphite texture"*, the **default** V2V path
returned a clip visually indistinguishable from its source - same colour, same
photographic look, no graphite anywhere. Nothing errored. The file validated.
The smoke reported PASSED. The failure existed only in the pixels.

That is the documented behaviour of the default: still-conditioned restyle at
`v2v_structure_strength: 0.45`, which anchors hard to source stills. The strong
restyle is the opt-in transform engine - `v2v_engine: transform`,
`ltx_pipelines.ic_lora` plus the Union Control LoRA. Run that way, the same
prompt stripped the colour to monochrome and held the geometry exactly.

> **Benchmark rule.** C-group (6 cases) and D-group (5 cases) - 11 of the 41 -
> **must** run with `v2v_engine: transform`. On the default path they measure a
> near-passthrough, and any LTX-vs-H3 conclusion drawn from them would be a
> statement about one config key wearing a model's name. The harness already
> sets it; this is why it must stay set.
>
> Where it is informative, record **both** as separate cells - the default is a
> legitimate *structure-preserving* mode, it is simply not a restyle. Compare
> H3 against the LTX mode that semantically matches the task.

Cost of the transform engine: **2.4x** (61.8 s against 25.9 s). That is the
documented LoRA rule - a LoRA drops quantization entirely and the unquantized
22B is fitted instead.

## Phase 2 · Install the H3 runtime

Do **not** guess the build. Decide it here, from what the purchased card
actually has, and record the decision.

- bf16 is ~110 GB (61.7 GB DiT + 48 GB Qwen3-VL-32B text encoder) and does not
  fit a 96 GB card. A quantized build is mandatory, not an optimisation.
- Official deployment paths: SGLang, vLLM, diffusers, ComfyUI. The
  service-shaped ones match how we already run ACE-Step.
- Sparse attention is not in the initial open-source release; full attention
  only.

Record exactly, into the result document: H3 model revision, quantization
build, runtime and version, CUDA, PyTorch, attention backend, and whether the
weights are the `fl2va` head, the `ref2va` head, or both.

**STOP if** the licence application has not been approved. This is a real
gate, but it turns on WHERE THIS GPU PHYSICALLY IS. The Applicable
Territory is worldwide **excluding** the EU, UK, South Korea and the US.
Outside those four, the community licence covers us by default; inside any of
them, an approved application is required first. Confirm the location from the
provider's machine record, not from IP geolocation or the billing country.
(Corrected 24 Aug 2026 — the earlier wording here was inverted. See
`h3-rtxpro6000-runtime-research.md` §1.1.)

## Phase 3 · H3 smoke

One of each, shortest supported length, before any comparison:

1. T2V (FL2VA, no images)
2. I2V (FL2VA, first frame)
3. Ref2VA with a subject image
4. Ref2VA with supplied audio in `fully_copy` — confirm the output audio IS
   the input waveform and that the mouth moves to it
5. Ref2VA video continuation

**STOP if** reference semantics differ from what
`worker/providers/h3_prompt.py` and `h3.py` assume. Fix the compiler first —
benchmarking a wrong request shape measures our bug, not the model.

## Phase 4 · Short A/B, LTX vs H3

Five seconds, one case per group, both engines, one run each. This is a
plumbing check, not a result: confirm both engines produce valid media, the
manifests match what the dry run predicted, and timings are in the expected
order of magnitude.

```bash
uv run python scripts/dual_engine_bench.py --dry-run --group A   # compare against reality
```

## Phase 5 · The high-value A/B/C

Three cells per case: LTX, H3, hybrid. Priority order — spend the budget top
down and stop when it runs out:

1. **D1–D5** reference-person V2V (the strongest a-priori H3 case)
2. **E1–E3** music video (lip-sync, `fully_copy` vs LTX a2vid)
3. **B1–B6, B8** image-to-video
4. **A2, A3, A5, A6** premium text-to-video

**STOP hybrid for a workflow** as soon as it is clearly behind direct H3 on
two consecutive cases. The question was never "can hybrid work" — it is
whether it earns a second inference pass.

## Phase 6 · Long form

30 s and 60 s from one global plan: A8, A9, B7, B8, H1, H2, E2, E3. Record
sections, seams, and per-section length for each engine — they differ by
design, and that difference is the finding.

## Phase 7 · Reliability

Repeat the shortlisted cells: 3 runs minimum, 5 on launch-critical paths.
Record the failure class every time, not just the count. A cell that passes
six times in isolation and fails five of fifteen under co-tenancy has already
happened here once.

## Phase 8 · Performance and cost

Fill in wall-clock, VRAM, host RAM and the model-switch penalty:

```text
LTX loaded → LTX generate → unload → H3 load → H3 generate
```

Measure the switch explicitly; a hybrid pays it on every job. Then compute
cost per 5 s / 30 s / 60 s / minute-of-music-video, per strategy, at the
actual GPU price. Keep quality and cost in separate columns.

## Phase 9 · The decision

Fill in the decision table. Every row cites quality, speed, VRAM, reliability
and cost. A mixed outcome is expected; "one model wins everything" should be
distrusted unless the table says so.

Only then: change `_AUTO_ROUTES`, with the evidence in the commit message.

---

## Stop conditions, in one place

| Condition | Action |
|---|---|
| An asset hash differs from the manifest | **Stop the comparison.** The inputs are not the same inputs. |
| The LTX golden argv test fails | **Stop.** The baseline moved; nothing measured against it means anything. |
| H3 repeatedly OOMs | **Stop tuning quality.** Fix runtime, build or memory first. |
| H3 reference semantics differ from our compiler | **Stop.** Correct the compiler, then re-run anything already measured. |
| Hybrid trails direct H3 on two consecutive cases in a workflow | **Stop hybrid for that workflow.** Spend the time on cells that can still change the answer. |
| A cell fails more than 1 run in 5 | Record it and treat reliability as the finding; do not average it away. |
| Licence not approved | **Stop.** H3 does not run. |

## The first commands

```bash
# 1  machine - record everything; it goes into every RunRecord
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit,compute_cap --format=csv
nvcc --version && free -g && df -h /workspace

# 2  LTX tree integrity - 400fd31, and NO grep gate (it was invalid)
cd /workspace/src/LTX-2 && git log -1 --oneline

# 3  LTX env + the NVFP4 kernels (1.1) - sm_120, venv nvcc
uv sync --extra natten
export CUDA_HOME=$PWD/.venv/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0
uv pip install -e packages/ltx-kernels --no-build-isolation

# 4  checkpoints - all nine (1.2), Union Control from its own repo
hf download Lightricks/LTX-2.5 <the eight> --local-dir /workspace/models/ltx25
hf download Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control \
    ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors --local-dir /workspace/models/ltx25/loras

# 5  structural gate, no GPU needed - catches a moved baseline instantly
cd /workspace/src/zolexai/apps/worker && . .venv/bin/activate
python -m pytest tests/test_ltx_golden.py tests/test_providers.py tests/test_hybrid.py tests/test_golden_pack.py -q

# 6  the real render (1.3), then LOOK at it (1.4)
DURATION=5s ASPECT_RATIO=16:9 python scripts/ltx_smoke.py "a koi pond at dawn"

# 7  what the benchmark expects, so reality can be compared against it
python scripts/dual_engine_bench.py --dry-run --out /workspace/plans.json
python scripts/dual_engine_bench.py --skeleton /workspace/results.json

# 8  the pack must verify before any comparison is scored
python scripts/golden_pack.py --verify

# 9  install the H3 runtime - build chosen in Phase 2 and recorded in results.json

# 10 H3 PROVIDER-NATIVE reproduction BEFORE any ZolexAI request

# 11 first real cell: D1, all strategies, one run each
```
