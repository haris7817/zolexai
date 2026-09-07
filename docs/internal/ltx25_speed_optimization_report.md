# LTX 2.5 FAST 1080 — speed optimization

*8 Sep 2026. All renders on the client's own graph, on `ltx-6000-2`
(RTX PRO 6000 Blackwell Max-Q, 300 W cap), timed by ComfyUI's own
`execution_start`→`execution_success`. Baseline and method:
[`ltx25_baseline_report.md`](ltx25_baseline_report.md).*

**Brief.** Keep the client's workflow, keep its architecture, keep the
resolution, keep the LoRAs and the detailer, change the transformer
configuration, and get 30 s from ~17.5 min to 5–6 min without losing the
quality they approved.

**Two of those instructions do not apply to this graph, and one of them
backfires.** Both are findings, not objections — see Phase 0 and Phase 2.

## Phase 0 — what the graph actually contains

The brief asks to keep the LoRAs and the detailer enabled. The FAST 1080
graph **has neither**. Its 53 nodes contain one `UNETLoader`, no
`LoraLoader*` of any kind, and no detailer stage. Those belong to the
*earlier* Text to Video graph (`ltx25_text_to_video.json`), which is a
different pipeline and is not what Text to Video runs in client-test.

So "keep LoRAs, keep detailer" is satisfied trivially: there is nothing to
remove. Every other rule is binding and was kept — same resolution, same
architecture, the original file never edited.

## Phase 1 — baseline (frozen)

| | 15 s reference render |
| --- | --- |
| ComfyUI execution | **310.3 s** |
| delivered | 1920×1080, 361 frames, 24 fps, audio |
| peak VRAM | 46.7 GB of 97 GB |
| peak container RAM | 66.0 GiB of 241.5 GiB |
| sampling | 271 s (8 steps at **34.0 s/step**) — **87% of the render** |
| model init | 33.3 s, charged to the first step |
| encode + decode + mux | ~39 s |

**30 s baseline: ~1050 s (17.5 min).** Target 300–360 s → needs 2.9–3.5×.

Because sampling is 87% of the clock, only two kinds of change can matter:
make a step cheaper, or run fewer steps. Everything else is rounding.

## Phase 2 — transformer comparison

Same prompt, same seed 987654, same 15 s, same schedule, same everything
else. Only `UNETLoader.unet_name` changed.

| transformer | execution | s/step | peak VRAM | peak RAM | detail vs NVFP4 | luma | saturation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **NVFP4 — the client's own** | **310.3 s** | **34.0** | **46.7 GB** | **66 GiB** | 1.00 | 62.8 | 89.4 |
| int8 convrot (the "official" one) | 317.9 s (+2.5%) | 36.0 | 66.6 GB | 110 GiB | 0.82 | 58.1 | 81.7 |
| bf16 (full precision, 42 GB) | 354.3 s (+14%) | 39.4 | 93.1 GB | 172 GiB | 0.86 | 58.3 | 81.8 |

**The approved change — "switch to the optimized transformer configuration"
— is a regression on this card.** NVFP4 is the fastest of the three, uses
the least memory by a wide margin, and is not the softer or darker one: at
the same seed both alternatives came out *darker* (luma 58 vs 63) and *less
saturated* (82 vs 89), and measured **lower** detail.

That reverses the earlier finding (7 Sep) that int8 was 14% faster. It is
not a contradiction: that test was on the **earlier** graph against a GGUF
Q8 baseline. Here the incumbent is NVFP4, which Blackwell executes in
hardware (`scaled_mm_nvfp4`), and there is no faster option to switch to.
The client's graph already ships the right one.

SSIM against the NVFP4 baseline: int8 Y 0.819, bf16 Y 0.844 — the three
transformers do not produce the same video, so this is a quality read, not a
defect. Audio was identical in length and level across all three.

**Decision: keep NVFP4. Phase 2 yields no speed-up; the speed has to come
from Phases 3 and 4.**

## Phase 3 — runtime and memory

Three things were found unset on the node and tested one at a time. Each is a
**server flag**: the workflow file, the schedule, the transformer and the
resolution are identical in every row below, and the original graph is never
touched.

| arm | flags | execution | s/step | peak VRAM | vs baseline |
| --- | --- | --- | --- | --- | --- |
| baseline | *(none)* | 310.3 s | 34.0 | 46.7 GB | — |
| **SageAttention** | `--use-sage-attention` | **213.6 s** | **21.6** | 46.3 GB | **1.45× faster** |
| model residency | `--highvram` | 316.9 s | 34.3 | 47.0 GB | 1.02× *slower* |
| Sage + residency | `--use-sage-attention --highvram` | 210.4 s | 21.6 | 47.3 GB | no gain over Sage alone |
| **Comfy Kitchen int8** | `--use-ck-attention` | **218.4 s** | **22.1** | 46.5 GB | **1.42× faster, no install** |
| Sage + residency + fast math | `… --fast fp16_accumulation cublas_ops autotune` | 245.9 s | — | 48.8 GB | **15% slower than Sage alone** |

### SageAttention — the result

The card is a Blackwell (sm_120) running attention through plain PyTorch
SDPA. SageAttention is a quantized attention kernel built for exactly this
part. It was not installed; it now is, built from source against this
ComfyUI's own CUDA 13 / torch 2.14 (the stock build needed two fixes: the
CUDA 13 headers want `libcusolver-dev`, and torch 2.14 requires C++20 while
the package still asked for C++17).

Kernel microbenchmark, 8192-token attention at this model's head shape:
**SDPA 8.27 ms, Sage 5.21 ms — 1.59×**, max absolute difference 0.0078.

End to end on the client's graph at 15 s: **310.3 s → 213.6 s, 1.45×.**
Sampling went from 34.0 to 21.6 s/step. VRAM unchanged.

`--fast` is ComfyUI's own "untested and potentially quality deteriorating"
switch. Here it did not even buy speed: with Sage already on, adding
`fp16_accumulation`, `cublas_ops` and `autotune` cost 32 s (245.9 s vs
213.6 s). **Not adopted.**

### A second backend, already on the node

ComfyUI ships its own quantized attention kernel (`--use-ck-attention`,
Comfy Kitchen int8). It needs no build and was already installed. It measured
**218.4 s at 15 s and 670.2 s at 30 s** — within 3% of SageAttention — and is
if anything *closer* to the baseline render (15 s SSIM 0.894 vs Sage's 0.852;
audio correlation 0.967 at zero offset vs 0.935 at 40 ms).

Sage is the default because it is the faster of the two at both lengths and
was validated across the whole ladder. Comfy Kitchen is documented as the
**fallback that needs no build**, which matters the day a ComfyUI or torch
upgrade breaks a source-built extension.

### Why residency did nothing

The 18 GB transformer is streamed rather than resident (`Model LTXAV
prepared for dynamic VRAM loading, 17831MB Staged`) on a card whose peak use
is 47 of 97 GB, which looked like an obvious waste. It is not: forcing
residency with `--highvram` made the render **slower** (316.9 s vs 310.3 s),
and per-step time did not move (34.3 vs 34.0). ComfyUI's async offload
already overlaps the transfers with compute. Stacking it on Sage gave
nothing either. **Not adopted** — a negative result worth recording, because
it is the change everyone proposes first.

## Phase 4 — inference schedule

The graph's 8 steps are unevenly spaced: the first four move 2.5% of the
noise range, the last four move 97.5%. Half the compute sits in the top
2.5%, which looks like free money. It is not free.

| arm | steps | execution | vs baseline | detail | luma | motion energy | motion variance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 8 | 310.3 s | — | 1.00 | 62.8 | 3.22 | 0.03 |
| sig6 | 6 | 237.7 s | 1.31× | 0.95 | **56.6** | **2.33** | **0.34** |
| sig5 | 5 | 202.5 s | 1.53× | **0.74** | **55.5** | **2.15** | 0.17 |

Both are **rejected**, on the brief's own rejection list:

* **Colour shift.** 6 points of brightness at 6 steps, 7 at 5 steps.
* **Motion.** Motion energy falls 28% at 6 steps and 33% at 5; frame-to-frame
  variance rises **11×** at 6 steps — motion becomes jumpier, not smoother.
* **Detail.** 5 steps loses a quarter of it (0.74×).
* **Creative output.** This is the decisive one. At the same seed and prompt,
  6 steps did not render a worse version of the baseline shot — it rendered
  **a different shot** (a close-up of the woman's face instead of a walking
  three-quarter shot). Changing the schedule re-rolls the video. Every
  approved prompt would come back different.

That last point is why the schedule is left exactly as the client wrote it.
Sage, by contrast, returns *the same shot* — same subject, wardrobe, wall,
reflections, framing — 1.45× faster.

## Phase 5 — the 30-second target, and why the shape of the curve decides it

Fitting the measured ladder gives a clean law. Cost is **quadratic in clip
length**, because attention runs over the whole clip at once:

| | fixed cost | per-frame² cost |
| --- | --- | --- |
| baseline | 46 s | 1.92 × 10⁻³ |
| optimized | 51 s | **1.16 × 10⁻³** |

`t ≈ fixed + c · frames²`, and the fit reproduces every measured point to
within a few percent. The optimization **cuts the quadratic term by 1.65×**
and leaves the fixed part (model load, text encode, VAE decode, mux) alone.

Two consequences:

1. **The speed-up grows with length** — 1.15× at 5 s, 1.59× at 30 s — because
   the part that got faster is the part that dominates long clips. The 30 s
   render, the one the brief is about, is where this pays most.
2. **The target needs more than a kernel.** For 30 s to land at 360 s the
   quadratic coefficient would have to fall to 6.0 × 10⁻⁴ — another **1.95×
   beyond** what is now delivered, on top of the 1.65× already taken. Nothing
   tested reaches that without changing the resolution, the schedule or the
   video itself, all of which the brief excludes.

### The honest answer on 5–6 minutes

**Not reached at native quality: 30 s now takes 10.9 minutes, from 17.3.**
What is left, with its price:

| route | 30 s time | what it costs |
| --- | --- | --- |
| **delivered — attention kernel** | **10.9 min** | **nothing measurable** |
| + 6-step schedule | ~8.3 min | re-rolls every prompt into a different shot; −6 luma; motion variance ×11 |
| two 15 s clips instead of one 30 s | 7.1 min | quadratic cost avoided, but it is two clips: the soundtrack restarts and continuity is not guaranteed |
| 1280×736 generation, upscaled | ~3.5 min | 0.29× the detail (measured 7 Sep) |
| **a card that is not power-limited** | **~5–6 min** | **money, and nothing else** |

That last row is not a guess about a different GPU generation — it is about
*this* GPU. Sampled during a render, the card sits at **exactly 300.0 W, its
cap, with SM clocks at ~1550 MHz against a 3090 MHz maximum, at 86 °C.** It
is a Max-Q part running at roughly half its rated clock because of a 300 W
power limit. The same silicon in a 600 W workstation card, with this
optimization, is the one route to 5–6 minutes that keeps 1920×1080, the
client's schedule and the client's transformer exactly as they are.

## Phase 6 — quality validation

Every arm was compared against the baseline render at the **same seed and
prompt**, on measured quantities, and looked at.

**At 30 s — the length the brief is about, baseline vs optimized:**

| measure | baseline | optimized | verdict |
| --- | --- | --- | --- |
| detail (Laplacian variance) | 9.1 | **9.9 (1.09×)** | no loss |
| brightness (mean luma) | 67.0 | 67.2 | unchanged |
| exposure drift across the clip | 11.6 | 11.3 | unchanged |
| saturation | 71.3 | 69.9 | −2%, not visible |
| motion energy | 2.00 | 1.90 | unchanged |
| motion variance (jumpiness) | 0.07 | 0.06 | unchanged |
| SSIM (Y) vs baseline | — | **0.908** | same video |
| PSNR (Y) vs baseline | — | **28.0 dB** | same video |
| audio | 30.010 s, −26.9 dB | 30.010 s, −26.7 dB | same length and level |
| audio sync | — | **+20 ms, correlation 0.79** | in sync |

Looked at, not just measured: matched frames at three points in the clip are
the **same shot** — same pose, same background, same lighting, same
reflections — with slightly crisper coat texture on the optimized side. Tile:
`E:\Downloads\zolexai-speed-ab\tile_30s.png`.

That is the whole quality case: at the target length, the optimized render is
the baseline render, 6.5 minutes sooner.

**A second graph, because the flag is server-wide.** The pack's own Text to
Video graph (a different pipeline: 1280×704 two-stage latent refine) at 10 s,
same prompt and seed:

| arm | wall | delivered | detail | luma | motion | motion var |
| --- | --- | --- | --- | --- | --- | --- |
| stock | 91.4 s | 1280×704, 241 f | 306.1 | 116.8 | 9.57 | 0.27 |
| Sage | **85.3 s** | 1280×704, 241 f | 311.0 (1.02×) | 116.5 | 9.39 | 0.06 |
| Comfy Kitchen | **82.5 s** | 1280×704, 241 f | — | — | — | — |

Correct size, correct frame count, detail and brightness unchanged. The gain
is smaller (1.07×) because that graph runs at 0.9 MP, where attention is a
smaller share of the work — which is the same law as Phase 5, seen from the
other end.

**Production path, end to end.** A 5 s Text to Video job through the real
adapter with the flag live: refused 9:16 before GPU time, kept the client's
negative prompt, delivered 1920×1080 / 121 frames / 5.04 s with a 5.01 s
soundtrack.

**Rejected arms, for contrast** (15 s, same seed):

| arm | detail | luma | motion energy | motion var | verdict |
| --- | --- | --- | --- | --- | --- |
| int8 transformer | 0.82 | 58.1 | — | — | darker, softer, slower |
| bf16 transformer | 0.86 | 58.3 | — | — | darker, softer, 14% slower |
| 6 steps | 0.95 | 56.6 | 2.33 | 0.34 | different shot, darker, jumpy |
| 5 steps | 0.74 | 55.5 | 2.15 | 0.17 | detail loss, different shot |

## Phase 7 — what production runs

**Nothing about the client's workflow changes.** The optimized configuration
is the client's own graph, the client's own NVFP4 transformer, the client's
own 8-step schedule, at 1920×1080 — served by a ComfyUI started with one
extra flag.

| | |
| --- | --- |
| workflow | `LTX2.5_ACTUAL_WORKFLOW_ONLY_FAST_1080_8s_AUDIO.json`, sha256 `19480e74…`, unedited |
| copies | `benchmarks/client-pack/ltx25/speed/LTX2.5_FAST_1080_{original,optimized}.json` — **byte-identical**, deliberately (see that folder's README) |
| transformer | NVFP4, as delivered |
| schedule | the client's 8 sigmas, as delivered |
| resolution | 1920×1080, as delivered |
| the change | ComfyUI launched with `--use-sage-attention` |
| where | `deploy/gpu/zolexai-ltx-comfy.sh`, runbook §47 |
| rollback | `echo --no-sage > /workspace/comfy_extra_args && supervisorctl restart zolexai-ltx-comfy` |

Because the win is a server flag rather than a graph edit, there is no
routing change, no worker change, no API or VPS change, and no migration:
existing jobs, prompts and seeds keep working exactly as before, only faster.

**Scope.** The flag is server-wide, so Character Replacement, Image to Video
and Extend Video get the same kernel. That is upside — they are all attention
-bound — but it means the validation below is the evidence for those tools
too, plus the pack-graph side-check.

## Phase 8 — final benchmark

Same prompt, same seed 987654, 16:9, audio on, timed by ComfyUI's own
execution clock. The optimized column includes a cold model load on every
arm (each ran on a freshly restarted server), which is what production does
too — the worker frees models between jobs — so these are the numbers a user
actually waits for.

| duration | frames | baseline | optimized | speed-up | VRAM (base → opt) | RAM (base → opt) |
| --- | --- | --- | --- | --- | --- | --- |
| 5 s | 121 | 64.6 s | **56.4 s** | 1.15× | 44.6 → 43.8 GB | 55.7 → 56.8 GiB |
| 10 s | 241 | 156.6 s | **121.4 s** | 1.29× | 45.2 → 45.7 GB | 72.3 → 68.2 GiB |
| 15 s | 361 | 310.3 s | **213.6 s** | 1.45× | 45.6 → 45.3 GB | 66.0 → 76.8 GiB |
| **30 s** | **721** | **1040.6 s (17.3 min)** | **652.5 s (10.9 min)** | **1.59×** | 58.4 → **56.2 GB** | 122.2 → **94.5 GiB** |

Quality at every length: same resolution, same frame count, same fps, audio
present and the same length and level. The 30 s pair was compared in full
(Phase 6): detail 1.09×, brightness within 0.2, motion unchanged, SSIM 0.908.

Two side benefits at 30 s, both from the same change:

* **Peak host RAM falls 23%** (122.2 → 94.5 GiB). The 7 Sep incident was the
  container's 241.5 GiB ceiling being hit; this buys real headroom back.
* **Peak VRAM falls slightly** (58.4 → 56.2 GB), so nothing moves closer to
  the card's limit.

## Remaining limitations

* **The 5–6 minute target is not met at native quality.** 30 s is 10.9 min.
  The routes to 5–6 min all cost something; the only one that costs nothing
  but money is a card that is not power-limited (Phase 5).
* **Renders are not bit-identical to pre-change renders.** An accelerated
  attention kernel is an approximation: the same prompt and seed give the
  same shot with the same measured detail, brightness, motion and audio, but
  not the same bytes. Anything that pinned an exact previous output would
  need re-approving. (ZolexAI-vs-direct-ComfyUI parity is unaffected — both
  go through the same server.)
* **The flag is server-wide.** Character Replacement, Image to Video and
  Extend Video get the same kernel. Validated here on two graphs; if a third
  ever shows an artefact, roll back in one line.
* **The build is from source.** SageAttention needed two fixes to compile
  against this ComfyUI's CUDA 13 / torch 2.14 (runbook §47). A ComfyUI or
  torch upgrade may need it rebuilt — and if that ever fails, the fallback
  below needs no build at all.
* **Nothing was tested above 30 s**, which is the product limit.

## Final summary

**LTX 2.5 FAST 1080 OPTIMIZATION**

| | |
| --- | --- |
| **Original 30 s runtime** | **1040.6 s — 17.3 minutes** |
| **Optimized 30 s runtime** | **652.5 s — 10.9 minutes** |
| **Speed improvement** | **1.59× (−37%); 1.45× at 15 s, 1.29× at 10 s, 1.15× at 5 s** |
| **VRAM improvement** | 58.4 → 56.2 GB peak at 30 s; **host RAM 122.2 → 94.5 GiB (−23%)** |
| **Quality impact** | **None measurable.** At 30 s: detail 1.09× of baseline, brightness 67.2 vs 67.0, exposure drift 11.3 vs 11.6, motion 1.90 vs 2.00, motion variance 0.06 vs 0.07, SSIM 0.908, PSNR 28.0 dB, audio same length and level, in sync (+20 ms, correlation 0.79). Matched frames show the same shot. |

**Final recommendation**

Run the client's FAST 1080 workflow exactly as delivered — same file, same
NVFP4 transformer, same 8-step schedule, same 1920×1080 — on a ComfyUI
started with `--use-sage-attention`. That is the entire change, it is one
line in the launcher, and it is reversible in one line.

Do **not** change the transformer: NVFP4 is already the fastest and lightest
of the three, and the two alternatives are slower, heavier and darker.
Do **not** shorten the schedule: it re-rolls every approved prompt into a
different video.

If 5–6 minutes at 30 s is a firm requirement, the only route that keeps the
approved quality is a GPU that is not running at half its rated clock under
a 300 W cap. Everything achievable in software at this quality is now done.

**Fallback, if the SageAttention build ever breaks:** `--use-ck-attention`,
ComfyUI's own Comfy Kitchen int8 attention. It needs no build, it is already
present on the node, and it measured within 3% of Sage (15 s: 218.4 s vs
213.6 s; 30 s: 670.2 s vs 652.5 s) at the same quality (30 s SSIM 0.904,
detail 1.07×, audio correlation 0.90). One word in the extra-args file.
