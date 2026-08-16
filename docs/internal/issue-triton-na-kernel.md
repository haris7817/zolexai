# Issue: Triton neighbourhood-attention kernel fails on longer generations

**Opened:** 14 August 2026
**Updated:** 16 August 2026 — **root-caused and fixed**
**Status:** 🟢 **RESOLVED** — cause found, fix applied, ceilings re-measured
**Severity:** was High — customer-visible failures on the most-used workflow
**Box:** RTX PRO 6000 Blackwell (`sm_120`), instance `47698594`

---

## 0. Resolution (16 August 2026)

**NATTEN was never installed. It is a declared, pinned, optional dependency of
`ltx-core` — and the default VAE decode path expects it.**

```toml
# packages/ltx-core/pyproject.toml
[project.optional-dependencies]
natten = [
    # Pinned wheel + matching torch so DiffVAE does not hit TokPerm IMA on
    # older PyTorch/NVIDIA stacks.
    "natten==0.21.7+torch2130cu132; sys_platform == 'linux' and ...",
    "torch==2.13.0; ...",
]
```

Two things in that block deserved attention long before today. The comment
names **IMA — illegal memory access** — the exact second error in §4.3, and says
the pin exists to prevent it. And the box already ran `torch 2.13.0+cu132`,
precisely what the wheel is built for.

The mechanism, confirmed from the pipeline's own documentation:

```
--diffvae-optimization chunked_eager   (the DEFAULT)
    → "deferred stage-4, W-chunks=4, cutlass-fna (or Triton/eager fallback)"
    → NATTEN absent, so the fallback is taken
    → fallback_na → triton_na → _na3d_kernel → cannot launch at larger shapes
    → LTX_MAX_SECONDS=10 → six passes for a 60s video → five seams
```

FNA is Fused Neighbourhood Attention: NATTEN's kernel. The default configuration
assumes it is present.

### Install

```bash
cd /workspace/ltx2-benchmark
uv sync --extra natten --group kernels     # --group kernels is REQUIRED
```

**`--group kernels` is not optional.** `uv sync --extra natten` alone resolves
the lockfile for the named extras only and **uninstalls `ltx-kernels`** — the
NVFP4 CUDA extension production depends on — along with the whole
`nvidia-cutlass-dsl` stack. A dry run caught this; without it the install would
have taken every video generation on the box down. Always:

```bash
uv sync --extra natten --group kernels --dry-run
```

and require the output to be exactly `+ natten` with no `-` lines.

194 MB, prebuilt for `sm_120`, no compilation, torch untouched.

### Measured after (single pass, 60s, `nvfp4-prequant`)

| Aspect | Grid before | Ceiling before | Grid now | Ceiling now |
|---|---|---|---|---|
| 16:9 | 896×512 | 20s | **1024×576** | **60s** |
| 9:16 | 512×896 | 10s | **576×1024** | **60s** |
| 1:1 | 640×640 | ≥15s | **768×768** | **60s** |
| 4:5 | 512×640 | ≥15s | 512×640 | **60s** |

**Every duration the product offers is now one pass on every aspect ratio**, and
the grids are 29–44% larger rather than smaller. A 60s render went from six
passes and five seams to one pass and none.

That matters far beyond speed. Action replay, identity drift, dialogue restart
and the visible pause were all reported at ~10-second intervals — the pass
boundary. A symptom that occurs *at* a boundary cannot occur when there is no
boundary.

### The failure that remains, and it is a different one

`896×512` at 60s now fails in the VAE's **MLP**, not its attention:

```
chunked/mlp.py:119  _swiglu_tiled_residual_modulated_op
torch.mm(ws, w_down.t(), out=out_buf[:n])
RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR ... cublasGemmEx
```

`fallback_na`, `triton_na` and `_na3d_kernel` are gone from the stack entirely.
This is the same error class as the 720p probe (`1280×704`), so it is a second,
independent defect — not a regression of this one. It is worked around by not
using `896×512`, which is why the 16:9 grid moved.

### The shape rule: there isn't one

Measured at 60s. Note that a **larger** grid passes where a **smaller** one
fails, in both directions:

| Grid | Pixels | Result |
|---|---|---|
| 1024×576 | 589,824 | ✅ |
| 896×512 | 458,752 | ❌ |
| 1152×640 | 737,280 | ❌ |
| 768×960 | 737,280 | ❌ |
| 768×768 | 589,824 | ✅ |
| 640×640 | 409,600 | ✅ |

Width, height, area and pixel-count models were each proposed and each refuted
within an hour. **The failing set is a collection of bad shapes with no
predictable rule.** Every grid must therefore be measured, never interpolated —
which is why `adapters/ltx._GRID_CEILINGS` holds measured values only and any
absent grid takes a pessimistic 10s.

### Also fixed alongside

`plan_segments` chunked greedily, so a source whose length was not a clean
multiple of the ceiling produced a sliver final window — `240.03s @ 60` gave
`60, 60, 60, 60, 0.03`, and 0.03s is **one frame**. A full model invocation to
produce a frozen flash on the end of the video. Windows are even now. This hit
music-video, video-to-video and extend, whose lengths come from uploaded files
and are therefore never round numbers.

### Still to verify

Image-to-video, extend-video and video-to-video have **never executed on this
hardware in any configuration**, and they condition on an uploaded file —
conditioning previously lowered the ceiling (§4.2). `scripts/ltx_matrix.sh`
covers them; `scripts/coverage_gaps.sh` covers the chained path and the music
runtime, neither of which the matrix reaches.

---

## 0.1 Why this took two days to find

The information was in the repository the entire time. `pyproject.toml` declared
the dependency, pinned the wheel, and its comment named the exact error. The
optimisation guide stated that the default decode mode falls back to Triton
without NATTEN. Nobody read either, because the failure looked like a hardware
limit — and once something is believed to be a hardware limit, the search moves
to workarounds instead of causes.

Every intermediate conclusion was a workaround, and each was wrong in a way the
next measurement exposed. The one thing never tried was reading what the
dependency file said.

---

## 1. Summary

The LTX video VAE's **fallback neighbourhood-attention path** crashes with a CUDA
error when decoding longer generations on this GPU. Short generations are
unaffected. The failure is in a *fallback* Triton kernel, which suggests the
optimised NATTEN kernel is missing on this box and was present on the RTX 5090
it replaced.

**15 Aug status clarification:** production's per-pass ceiling was subsequently
lowered to 10 seconds and the duration/aspect matrix passed through chaining.
The crash is therefore mitigated, not root-caused. The client then exposed a
second issue: forcing every long result through 10-second passes makes prompt,
audio and continuity defects visible at exactly those boundaries. The kernel
defect and the chaining defects must be tracked separately.

```text
RuntimeError: Triton Error [CUDA]: invalid argument
```

---

## 2. Customer impact

With production's current `LTX_MAX_SECONDS=30`:

| Workflow | Aspect | Duration | State |
|---|---|---|---|
| Text to Video | 16:9 | 5s / 10s / 15s | ✅ works |
| Text to Video | 16:9 | **30s / 60s** | ❌ **fails** |
| Text to Video | 9:16 | 5s / 10s | ✅ works |
| Text to Video | 9:16 | **15s / 30s / 60s** | ❌ **fails** |
| Text to Video | 1:1, 4:5 | all | ❓ **unmeasured** |
| Image to Video / Extend Video | all | 30s+ | ❌ **presumed failing** — same code path, unconfirmed |
| Music Video | all | any | ❌ **fails** — un-routed to `mock` 14 Aug as mitigation |
| Video to Video | — | — | ⚠️ untested since migration; worked on the 5090 |
| Music (audio) | — | — | ✅ unaffected, different runtime |

**Portrait is worse than landscape** — 9:16 breaks from 15s upward, so the
customer-visible surface is larger than "long videos are broken". The client uses
9:16 specifically.

**These are live.** A customer selecting 30s or 60s on zolexai.com gets a failed
generation. This was introduced by the 5090 → PRO 6000 migration and went
unnoticed because the migration parity test used **10s**, which passes.

---

## 3. Reproduction

On the GPU box:

```bash
cd /workspace/zolexai/apps/worker
LTX_REPO_DIR=/workspace/ltx2-benchmark LTX_QUANTIZATION=nvfp4-prequant \
LTX_MAX_SECONDS=30 DURATION=30s ASPECT_RATIO=16:9 \
/workspace/zolexai/.venv-worker/bin/python scripts/ltx_smoke.py a koi pond at dawn
```

Substituting `DURATION=10s` passes. This bypasses the API, database, storage and
routing entirely, so it isolates the model.

---

## 4. Evidence

All measured 14 Aug 2026 on `nvfp4-prequant`, via `scripts/ltx_smoke.py`.

### 4.1 The limit is DIMENSION-driven, not duration-driven

Single-pass ceilings differ per aspect ratio, at **identical pixel counts**:

| Aspect | Frame | 5s | 10s | 15s | 20s | 30s |
|---|---|---|---|---|---|---|
| 16:9 | 896×512 | ✅ | ✅ | ✅ | ✅ | ❌ |
| 9:16 | 512×896 | ✅ | ✅ | ❌ | — | ❌ |
| 1:1 | ? | — | — | — | — | — |
| 4:5 | ? | — | — | — | — | — |

896×512 and 512×896 hold the same number of pixels but have **different
ceilings**. `na3d` derives its launch grid from the spatial dimensions
individually, so the taller frame trips the limit sooner. 1:1 and 4:5 are
offered by the product and **have not been measured**.

This is why `LTX_MAX_SECONDS` — a single global per-pass value — must be set to
satisfy the *worst* aspect ratio, not the one that happens to be tested.

### 4.2 Conditioned passes have a lower ceiling than fresh ones

| Request | Ceiling | Pass plan | Result |
|---|---|---|---|
| 20s single, 16:9 | 20 | one fresh 20s | ✅ |
| 30s, 16:9 | 20 | two × 15s | ✅ |
| 60s, 16:9 | 20 | three × 20s | ❌ **section 2** |
| 60s, 16:9 | 15 | four × 15s | ✅ (59.9s, 148s wall) |

A fresh 20s pass succeeds; a *continuation* 20s pass fails. Chained sections
carry conditioning frames from the previous section, so their input is larger
than a fresh generation of the same length. **Testing a duration standalone does
not prove it works as a chained section.**

### 4.3 Two distinct CUDA errors

```text
Triton Error [CUDA]: invalid argument                    ← fresh pass over the limit
Triton Error [CUDA]: an illegal memory access was encountered  ← conditioned pass (60s §2)
```

Both originate from the same `_na3d_kernel[grid](` launch. Whether they are the
same underlying defect is unconfirmed.

### 4.4 Failing call stack, innermost frames

```text
ltx_core/model/video_vae/transformer/chunked/attn.py:157   _attn_on_chunk_impl
ltx_core/model/video_vae/transformer/fallback_na/__init__.py:128  __call__
ltx_core/model/video_vae/transformer/fallback_na/__init__.py:80   na_attention_triton
ltx_core/model/video_vae/transformer/fallback_na/triton_na.py:153 _na3d_kernel[grid](
triton/backends/nvidia/driver.py:328                       self.launch(gridX, gridY, gridZ, …)
RuntimeError: Triton Error [CUDA]: invalid argument
```

Two things the stack tells us:

- It is in the **video VAE decode**, not the diffusion transformer.
- The module is named **`fallback_na`** — this is the path taken when the
  optimised neighbourhood-attention kernel is unavailable.

`invalid argument` from a Triton grid launch typically means a launch dimension
exceeds a device limit — consistent with "works at small sizes, fails at large".

---

## 5. Music Video — largely explained, one gap left

Music Video failed generating **section 1 of 2** from a 29-second track, a pass
of roughly 15 seconds. That looked anomalous when 15s passed for Text to Video —
but §4.1 resolves most of it: 15s only passes at **16:9**, and it is not
established which dimensions Music Video derives. If it renders portrait or a
non-16:9 grid, a 15s pass failing is exactly consistent with the table above.

Remaining gap: Music Video's output dimensions were never captured from the
failing run. Confirm them before assuming it is fully explained — and note it
also conditions on audio, which may alter the latent shape independently of the
frame size.

---

## 6. Why the 5090 did not show this

The 5090 ran 30s single-pass successfully (recorded: 30s max, 60s hard-OOM). It
therefore never entered this fallback, which implies NATTEN was installed there —
either from the base image or as a side effect of that box's setup history. The
PRO 6000 was built from scratch and only `ltx-kernels` (the NVFP4 extension) was
compiled; **NATTEN was never installed**. See `gpu-worker-runbook.md` §34.2.

---

## 7. Options

### 7.1 Workaround — lower `LTX_MAX_SECONDS` (proven for 16:9, pending for portrait)

`LTX_MAX_SECONDS` is the **per-pass** ceiling; the chaining layer
(`worker/longform/chain.py`) already splits anything longer. One environment
variable on the worker, no API rebuild, **no durations removed from the product**.

Proven: `LTX_MAX_SECONDS=15` gives a working 60s at 16:9 (four passes, 148s wall).

**But it is not sufficient.** 9:16 fails at a 15s pass, so a global 15 leaves
portrait broken at 30s and 60s. The value must satisfy the worst aspect ratio the
product offers, and 1:1 / 4:5 are still unmeasured. If portrait tops out at 10s:

| Request | Passes at ceiling 10 | Seams | Est. wall |
|---|---|---|---|
| 30s | 3 | 2 | ~2 min |
| 60s | 6 | 5 | ~3–4 min |

That is a materially different product from the 5090's single-pass 30s. **The
cost of this workaround scales with how low the ceiling has to go**, which is the
main argument for pursuing 7.2 rather than settling here.

Does **not** fix Music Video (see §5).

### 7.2 Proper fix — install NATTEN (medium confidence)

If NATTEN builds for `sm_120`, the fallback is never used and the bug disappears.
Unknown until attempted: whether the LTX repo exposes it as a dependency group,
and whether it has Blackwell support.

### 7.3 Triton version (lower confidence)

The kernel may fail due to a Triton/driver combination on Blackwell rather than
anything about the shape. Bumping or pinning Triton is cheap to try.

### 7.4 Upstream

Worth reporting to Lightricks if 7.2 and 7.3 both fail — a fallback kernel that
cannot launch on a current-generation card is an upstream defect.

---

## 8. Actions taken

| When | What |
|---|---|
| 14 Aug 2026 | Music Video routed back to `runtime: mock` on the VPS, API rebuilt |
| 14 Aug 2026 | 16:9 ceiling established: 20s fresh passes, 30s fails |
| 14 Aug 2026 | Conditioned-pass effect identified (§4.2) |
| 14 Aug 2026 | 9:16 ceiling established: 10s passes, 15s fails |
| 14 Aug 2026 | `LTX_MAX_SECONDS=15` proven for 60s @ 16:9 — **not applied**, insufficient for portrait |

**Historical open list recorded 14 Aug (superseded by the 15 Aug status clarification above):**

- `LTX_MAX_SECONDS` **not changed** on the running worker — production still
  fails at 30s/60s in every aspect ratio
- **1:1 and 4:5 ceilings unmeasured** — both are offered to customers
- Chaining **not verified at whatever the final ceiling turns out to be**, in the
  worst-case aspect (this is the combination that failed at 20)
- **Image to Video and Extend Video not confirmed** — same code path, presumed
  affected
- **Music Video output dimensions not captured** (§5)
- **NATTEN not attempted** (§7.2)

---

## 9. Lessons

**A parity test at one duration and one aspect ratio proved much less than it
appeared to.** The migration was validated with a single 10s 16:9 clip, and the
box was declared good. The product offers five durations × four aspect ratios;
one cell of a twenty-cell matrix was checked, and it happened to be a passing one.

Worse, each narrowing step found the previous conclusion incomplete:

1. "30s is broken" → actually the whole duration ladder needed testing
2. "`LTX_MAX_SECONDS=20` fixes it" → failed at 60s, because conditioned passes
   have a lower ceiling than fresh ones
3. "`LTX_MAX_SECONDS=15` fixes it" → failed at 9:16, because the ceiling is
   dimension-dependent

**After any hardware change, exercise the product's actual parameter matrix** —
durations *and* aspect ratios — not one representative sample. The failure mode
here is size-dependent by construction, and no single sample can see it.

---

## 10. Client findings after the 10-second mitigation (15 Aug)

The client reported a small stop/reset, scene and identity changes, action and
dialogue replay, and cut-off final speech at roughly 10-second intervals. That
timing matches the mitigation's pass ceiling. This does **not** prove every seam
symptom is caused by FFmpeg or by the Triton kernel; it proves the symptoms are
section-boundary correlated.

### 10.1 Deterministic causes confirmed in the repository

| Finding | Confirmed cause before the local fix | Classification |
|---|---|---|
| Action/dialogue repeats every section | `_command()` received the same `job.prompt` for every chained pass | System bug |
| T2V/I2V audio restarts | Each pass generated its own soundtrack and raw pass files were concatenated | System architecture bug |
| Silent T2V/I2V could complete | Final validation required video but did not require an audio stream | System bug |
| I2V identity drifts after pass one | Only the predecessor final still was supplied; the uploaded identity image was dropped | Conditioning bug |
| Repeated Extend could reuse the wrong source | `?source=` was consumed only by React Hook Form's initial defaults; same-route query changes did not reset it | UI state bug |
| V2V sliders appeared ineffective | Public quality/motion/adherence values reached job parameters but the distilled adapter never read them | Fake-control bug |
| FPS/timebase seam risk | Plain generated sections went directly to concat without per-section normalization | Assembly risk, now removed locally |
| Music Video/V2V source track restart | Not supported by code inspection: both paths already strip model audio and attach the complete source track once after visual assembly | Not the same bug; verify artifacts on RTX |

### 10.2 Still unconfirmed

- Whether the visible pause is one duplicated boundary frame, several static
  settling frames from conditioned generation, or a timestamp discontinuity.
- Whether normalized assembly alone removes it. Capture 09.5-10.5 and
  19.5-20.5 second windows from an RTX result and run frame-difference plus
  `freezedetect`; do not hide it with a long crossfade.
- Whether the fallback kernel can be replaced by NATTEN on `sm_120`.
- Whether the current sparse-still V2V path can produce the client's requested
  strong shot-for-shot restyle. It preserves structure but is not a true
  video-conditioned restyle pipeline.

### 10.3 Local mitigation in this worktree (not deployed)

- deterministic section prompt planner; explicit `Persistent:` and
  `Section N / start-end:` instructions are honoured without paraphrasing user
  fragments;
- later I2V passes receive both temporal predecessor context and the original
  image at a low, non-zero-frame identity strength;
- generated sections are normalized to a common FPS, timebase, dimensions and
  stream layout before concat;
- T2V/I2V require one decodable, non-zero-duration audio stream per pass and in
  the finished file;
- source audio for Music Video/V2V remains attached once; Extend preserves the
  source timeline and appends generated tail audio only when the tail has it;
- worker/API/SSE progress now carries phase and section index/total/start/end;
- unsupported public quality/motion/adherence controls are hidden until a
  runtime actually consumes them.

This does not create a full-duration master dialogue/singing track. The current
distilled LTX entry point cannot emit audio-only output, so T2V/I2V remains
`GENERATED_PER_SECTION_AUDIO`. Section prompts stop deterministic dialogue
replay, but exact continuous dialogue timing needs a separate master-audio
provider/path and RTX validation.
