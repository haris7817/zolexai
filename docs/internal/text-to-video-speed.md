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

## The options, honestly

| option | expected wall (15 s) | detail | cost to build |
| --- | --- | --- | --- |
| A. 1280×736 + lanczos | 95 s (measured) | 0.29× (measured) | done, switch only |
| B. 1600×896 + lanczos | 177 s (measured, 1.7×) | 0.49× (measured) | done, switch only |
| C. two-stage latent: 8 steps at 960×544 → ×2 latent upscaler → 3-step refine at 1920×1088 | est. ~190 s (0.63× native step-cost, better under attention superlinearity) | near native — the refine pass *generates* detail at full size | needs the upscaler + second sampler in the FAST graph |
| D. 30 s as 2×15 s sections | 1050 → ~610 s for 30 s only | seam + audio restart | exists (chain), off |

**C is LTX's designed 1080p path**, and the client already uses exactly it in
their own earlier T2V graph and their MULTI4 character-replacement graph
(`ltx-2.5-latent-spatial-upscaler-x2`, sigmas `0.85, 0.725, 0.4219, 0`).
Adding it to the FAST graph is a redesign of the client's workflow, which is
the one thing the adapter is built not to do on its own. The right move is
to ask the client for the two-stage variant of their FAST graph — they
clearly have the pattern — or to get their explicit go-ahead to add it.

## What NOT to do

* Ship A as the default and call it 1080p. It is a 720p video in a 1080p
  container, and the tile shows it.
* Swap to NVFP4 for speed: measured the same speed as int8 on this stack,
  and darker.
* Cut steps: the graph is already the 8-step distilled schedule.
