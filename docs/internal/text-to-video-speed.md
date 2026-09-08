# Text to Video on the FAST 1080 graph — why it is slower, and what buys it back

*7 Sep 2026. Renders on the RTX PRO 6000 node (`ltx-6000-2`), all 15 s,
seed 31337, the harbour prompt, automatic dialogue off. Commit `763f11d`
added the canvas lever; nothing about the default changed.*

## Why 15 seconds went from 106 s to 306 s

Two different graphs, two different delivered sizes.

| | earlier client T2V graph (`ltx_comfy`) | FAST 1080 graph (now Text to Video) |
| --- | --- | --- |
| delivered size | **1280×704** (0.9 MP) | **1920×1080** (2.1 MP) |
| pipeline | 8-step pass at base size → ×2 **latent** upscale → 3-step refine → decode → ×0.5 | one 8-step pass at 1920×1088 → decode → crop |
| 5 s | 42–49 s | 58 s |
| 15 s | 106 s | 304–306 s |
| 30 s | 215 s | ~1050 s |

The FAST graph generates 2.3× the pixels in a single pass, and attention
cost grows faster than pixel count — which is also why 30 s is 3.4× the
cost of 15 s rather than 2×. The "35 s" in memory is the old graph's
5-second cell. No quality toggle is involved; that setting was removed on
5 Sep. This is purely the canvas the client's graph asks for.

## The lever, as built

The graph already ends in `ImageScale` (lanczos, 1920×1080, `crop=center`)
— that is how it turns 1088 rows into 1080. So a smaller generation canvas
needs **one widget pair**: the graph's own final node upscales and
centre-crops, and the soundtrack never passes through it. That is
`Fast1080Edits.canvas` / `LTX_HD_CANVAS`, default `native`.

## Measured: 1280×736 (the 720p proposal)

| arm | wall | delivered | Laplacian variance (12 frames) |
| --- | --- | --- | --- |
| native 1920×1088 | **304.3 s** | 1920×1080, 361 f | mean 445 (245–714) |
| 1280×736 → lanczos ×1.5 | **95.2 s** | 1920×1080, 361 f | mean **127** (96–154) |

**3.2× faster. Detail 0.29× of native**, and the tile agrees with the
number: rope and net texture, gull feathers and wood grain are crisp in
native and smoothed in the upscale. A pixel upscaler cannot restore detail
the model never generated; it interpolates. This does not meet "without
affecting the quality".

Caveat on the comparison: the same seed at a different canvas gives a
different composition, so this is a like-for-like *quality* read, not a
pixel diff. The Laplacian metric rewards edges and can be fooled by
sharpened noise — the tile is the other half of the evidence.

Tile and both clips: `E:\Downloads\zolexai-canvas-ab\`.

## Measured: 1600×896 (the middle canvas)

| arm | wall | delivered | Laplacian variance |
| --- | --- | --- | --- |
| native 1920×1088 | 304.3 s | 1920×1080 | 445 |
| 1600×896 → lanczos ×1.2 | **177.4 s (1.7×)** | 1920×1080 | **217 (0.49×)** |
| 1280×736 → lanczos ×1.5 | 95.2 s (3.2×) | 1920×1080 | 127 (0.29×) |

The curve is smooth and unkind: every second saved by a smaller canvas is
paid for in detail, roughly in proportion. There is no point on it that is
"faster without affecting the quality". 1600×896 is the defensible
compromise if speed matters more than fine texture for a *test* — rigging
and ropes hold, surfaces are softer — but it should be described as that.

**Two measurement traps, so nobody re-learns them.** ComfyUI's per-node
cache returns an identical prompt+seed+canvas instantly (a 1.1 s "wall");
time a real render by changing the seed — wall does not depend on it. And a
job queued behind another on ComfyUI carries the queue wait in the
adapter's clock (the first 1600 arm read 318 s); render on a free card.

## The actual cause, found the same evening: the node was memory-throttled

Every number above was measured on a node that had **hit its container memory
ceiling**. `memory.max` is 241.5 GiB; `memory.peak` equalled it;
`memory.events` showed **592,934 `max` hits** — the kernel reclaiming on
nearly every allocation, which throttles every GPU step — and `oom_kill` had
gone from 4 to **6**. ComfyUI-ltx was SIGKILLed at 17:26 and again at 19:36,
the second time with a client's Character Replacement job in flight; that job
ended `job_abandoned`.

The tell was the pack graph's own default — the 1280×704 delivery that took
**106 s** for 15 s on 5 Sep — taking **247 s** today by ComfyUI's own
execution clock. Power (300 W, the Max-Q part's default, capped since day
one) and clocks had not changed. What had changed was resident RAM: every
transformer loaded that day (GGUF, int8, NVFP4) stayed in memory, plus the
text encoder, plus Character Replacement's ~110 GiB per window.

**Mitigation:** `LTX_COMFY_FREE_AFTER_JOB=true` on the node — models unload
after every ltx_comfy / ltx_hd job (cold reload measured at ~9 s here), the
orphaned prompt interrupted, worker restarted idle. Between jobs the process
now sits at ~11 GiB.

**Confirmed:** the same pack-default 15 s render on the fresh process:
**91.7 s** (vs 247 s) — faster than 5 Sep, because int8 with the LoRAs off
is faster than the pack's GGUF. Same graph, same output, 2.7×.

| render (15 s) | under pressure | fresh process |
| --- | --- | --- |
| pack default, 1280×704 | 247 s | **91.7 s** |
| FAST native, 1920×1080 | 304 s | **314.5 s** |

**Two different stories, then.** The memory ceiling explains the pack
graph's 2.7× and the kills — but the FAST graph is **no faster on a clean
node**. Native 1920×1088 in one 8-step pass over 361 frames is intrinsically
~300 s for 15 s on this card; the pack graph's 1280×704 delivery is
intrinsically ~92 s. The throttling was real and had to be fixed (it cost a
client a job), but it was not the reason the client's graph is slow.

The canvas and two-stage numbers above were measured on the throttled node
and are valid only as *relative* comparisons among themselves. The one path
not yet measured clean is the pack graph's own refine delivered by
*downscale* rather than halved — see the next section.

**A second gap the incident exposed, fixed:** when the platform cancelled
that job, the worker's next progress report was rejected, `LeaseLost` rose
out of the progress callback, and the job's ComfyUI prompt kept the card for
its full length after the worker had walked away. `wait()` now cancels the
prompt on any exception it did not raise itself.

## The two-stage path, measured clean — and a misreading corrected

The pack graph's wiring, read off the compiled prompt rather than assumed:

    ResolutionSelector → EmptyImage → ImageScaleBy(0.5) → GetImageSize → EmptyLTXVLatentVideo
                                                              ↓
              first pass at HALF the selector size → LTXVLatentUpsampler ×2 → 3-step refine at selector size → deliver

So the selector's megapixel budget is the **delivered** size and the
`ImageScaleBy` sizes the **first pass**. I had read that node as a final
scaler and built (and named) a lever on that belief — `final_scale_by`, now
`base_scale`. A render I described as "the refine delivered by downscale"
was in fact a first pass at 0.75 of the selector size. The doc above this
section was written under that misreading and is left as the record of it.

Clean numbers, 15 s at 16:9, same seed as the native reference:

| path | first pass | delivered | wall (ComfyUI exec) | detail vs native |
| --- | --- | --- | --- | --- |
| FAST native (deployed Text to Video) | 1920×1088, 8 steps | 1920×1080 | **~310 s** | 1.00 |
| pack graph, as shipped (0.9 MP, base 0.5) | 640×368 | 1280×736 | **92 s** | — (not 1080p) |
| pack graph, 0.9 MP, base 0.75 | 960×552 | 1920×1080 (cropped) | **240 s** | 0.46 |
| pack graph, 0.52 MP delivery, base 1.0 | 992×544 | 1984×1080 (cropped) | 245 s (throttled node) | 0.31 |
| FAST graph, 1600×896 canvas + lanczos | — | 1920×1080 | 177 s (throttled) | 0.49 |
| FAST graph, 1280×736 canvas + lanczos | — | 1920×1080 | 95 s (throttled) | 0.29 |

**Verdict.** On this card, with this model family, native 1080p costs ~300 s
for 15 s and every faster route measured pays for it in detail — the
two-stage refine included, which at a 1080p delivery recovers roughly half
the native detail for a 1.3× saving. There is no "faster without losing
quality" inside the graph. What there was, was a node running everything
2.7× slow because it had hit its memory ceiling — fixed, and that fix is
the only free speed on the table.

**What the client can be offered honestly:**

1. Native 1080p at ~5 min per 15 s (what is deployed), on a node that no
   longer throttles or drops jobs.
2. A **speed setting** the customer chooses: `LTX_HD_CANVAS=1600x896`
   (~3 min, visibly softer) — presented as a draft quality, not as 1080p.
3. Hardware: this is a 300 W Max-Q part running at ~1.6 GHz under its own
   cap; a full-power RTX PRO 6000 or a second card for parallel jobs is
   the only route that keeps native quality and cuts wall time.

## What NOT to do

* Ship A as the default and call it 1080p. It is a 720p video in a 1080p
  container, and the tile shows it.
* Swap to NVFP4 for speed: measured the same speed as int8 on this stack,
  and darker.
* Cut steps: the graph is already the 8-step distilled schedule.

---

## Superseded in part, 8 Sep 2026 — then reverted the same day

*The measurement below stands; the deployment does not. The client judged
the optimized videos worse and asked for the stock kernel and the ~17 min /
30 s render back, so this document's original conclusion is once again what
production does. Kept because the numbers are real and the option is one
line away.*

### What was measured: a 1.45–1.59× that no metric could fault

This document's conclusion — "every faster path measured costs detail" — was
true of everything tested on 7 Sep, all of which changed **what the model
generates** (a smaller canvas, a different base size, a different graph).

It missed the one lever that changes only **how fast the same maths runs**:
the attention kernel. The node was running plain PyTorch SDPA on a Blackwell
card with no accelerated attention installed. With SageAttention built and
`--use-sage-attention` on, the FAST 1080 graph renders 15 s in **213.6 s
instead of 310.3 s (1.45×)** and 30 s in **652.5 s instead of 1040.6 s
(1.59×)** at 1920×1080 — same graph, same schedule, same NVFP4 transformer,
same seed. At 30 s: detail 1.09× of baseline, brightness 67.2 vs 67.0, motion
and exposure drift unchanged, SSIM 0.908, audio in sync, and the same shot on
screen. The gain grows with length because the quadratic attention term is
what got cheaper.

The canvas and two-stage numbers below stand; they are still the wrong trade.
Full method and the rejected options: `ltx25_speed_optimization_report.md`.

---

## Deployed 8 Sep 2026: the client's 720p-generate / 1080p-deliver plan

After the SageAttention revert, the client asked for their original proposal
instead: generate at the LTX 720p size and let the GPU upscale to 1080p,
keeping the audio. It is on — `LTX_HD_CANVAS=720p` on the GPU worker.

| ratio | generated | delivered |
| --- | --- | --- |
| 16:9 | 1280×704 | 1920×1080 |
| 9:16 | 704×1280 | 1080×1920 |
| 1:1 | 960×960 | 1080×1080 |

The upscale is the graph's own closing `ImageScale` (lanczos, `crop=center`),
so it runs on the GPU inside ComfyUI and costs a second or two. The
soundtrack never passes through that node — a compiled test pins that the
audio reaches `CreateVideo` without meeting it — so "keep the same audio" is
structural, not a promise.

### Measured, same prompt and seed as the native render

| length | native | 720p → 1080p | gain |
| --- | --- | --- | --- |
| 5 s (16:9) | 45.7 s | **21.6 s** | 2.1× |
| 5 s (9:16) | 61.7 s | **37.2 s** | 1.7× |
| 15 s | 312.7 s | **97.6 s** | 3.2× |
| **30 s** | **1040.6 s (17.3 min)** | **254.3 s (4.2 min)** | **4.1×** |

**30 s now beats the 5–6 minute target**, which nothing else tried did.

### What it costs, honestly

The detail penalty is real but **content-dependent**, which the single 0.29×
figure from 7 Sep did not convey:

* On the texture-rich harbour prompt (rope, netting, water), 7 Sep measured
  **0.29× the fine detail** of native — visibly smoothed.
* On a shallow-depth-of-field portrait prompt, the same comparison measures
  **0.99×**: the sharp plane is thin and mostly bokeh, so there is little
  fine texture to lose. Brightness moved a lot (63.2 → 43.2) but that is
  scene-level — at a different generation canvas the model renders a
  different video (SSIM 0.669), not a degraded copy of the same one.

So: portraits, close-ups and shallow-DOF scenes hold up well; wide landscapes
and detailed textures lose the most. Rollback is one setting —
`LTX_HD_CANVAS=native` on the GPU worker, then restart it.
