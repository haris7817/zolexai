# The client's FAST 1080 workflow — validation (7 Sep 2026)

The client sent a fourth ComfyUI workflow and asked whether it should be used
in ZolexAI: `LTX2.5_ACTUAL_WORKFLOW_ONLY_FAST_1080_8s_AUDIO.json`. It was
evaluated as supplied before any change, then driven through the backend.

**Immutable original:**
`benchmarks/client-pack/ltx25/client_original/LTX2.5_ACTUAL_WORKFLOW_ONLY_FAST_1080_8s_AUDIO.json`
sha256 `19480e7432539bed3e478ef1ee998df906857cf5103e4104daa73989b1f6e712`,
151,017 bytes. Never edited; every test compiles a copy in memory.

## 1. What it is

**Text to Video**, and it is not the character-replacement graph. It carries
an image path, but `use image input` is `false` and the `LoadImage` slot is
empty; that boolean feeds a `ComfyNotNode` whose output is `bypass_i2v`, which
switches `LTXVImgToVideoInplace` off. One boolean turns it into image to
video, so it is text-to-video-as-configured on a hybrid graph. Classified by
tracing the links, not by the presence of the image node.

## 2. The graph, by connection

| | |
|---|---|
| Transformer | `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`, `weight_dtype: default` |
| Quantization | **NVFP4 already** — nothing to switch |
| Video VAE | `ltx-2.5-video-vae-conv-bf16.safetensors` (the *conv* variant, **not** the `ltx-2.5-video-vae-bf16` our pack uses) |
| Audio VAE | `ltx-2.5-audio-vae-bf16.safetensors` |
| Text encoder | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` (loaded twice: encoder and enhancer) |
| **LoRAs** | **none** |
| **Detailer** | **none** |
| Sampler | `SamplerCustomAdvanced`, `euler_ancestral`, `CFGGuider` cfg 1 |
| Steps | **8** — `ManualSigmas` `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0` (9 sigmas, single stage, no refinement pass, no upscaler) |
| Seed | 42 |
| Resolution | generates 1920×1088, `VAEDecodeTiled` (tile 1536, overlap 256, temporal 256/32), `ImageScale` lanczos centre to 1920×1080 |
| FPS | **24** |
| Frames | **193**, computed by the graph: `1 + floor(a*b/8)*8` with fps 24 and duration 8 |
| Duration | 8.0417 s |
| Audio | generated in-graph (`LTXVEmptyLatentAudio` → `LTXVAudioVAEDecode` → `CreateVideo`) |
| Output | `SaveVideo`, prefix `LTX25_SPEED_PROMISE_8s_NATIVE_1080` |

**The 18 is not the frame rate.** It is `img_compression` on `LTXVPreprocess`.
FPS is 24, confirmed four ways: the `fps` primitive, `LTXVConditioning`'s
frame rate, `CreateVideo`'s fps, and the frame arithmetic.

**Dormant paths that still execute.** Two `TextGenerateLTX2Prompt` enhancers
feed a switch whose selector (`Enhance positive prompt`) is false, and two
`GemmaAPITextEncode` nodes are gated by a `StringContains` on an empty API
key, so no external call is made. Both branches of each switch are wired, so
whether the enhancers cost time depends on ComfyUI's lazy evaluation. Not
isolated here; the measured wall time already includes whatever they cost.

## 3. Environment

RTX PRO 6000, 97 GB. ComfyUI at `/workspace/ComfyUI-ltx`. All **34** node
classes the workflow needs are installed, including `TextGenerateLTX2Prompt`,
`GemmaAPITextEncode`, `ComfySwitchNode`, `ComfyNotNode`,
`ComfyMathExpression`, `StringContains` and `ResizeImageMaskNode`.

**One file was missing:** `ltx-2.5-video-vae-conv-bf16.safetensors` was on no
disk on the machine. It is published in `Lightricks/LTX-2.5` (gated; the node
already held credentials for that repo, which is where our other LTX 2.5
weights came from), so the file the workflow names was fetched rather than
substituted — 1,452,269,922 bytes, sha256
`685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d`, placed in
`models/vae/`. Adding a file changes nothing for the existing workflows.

**One thing had to be supplied.** The `LoadImage` slot is empty in the
delivered file and ComfyUI refuses that at validation
(`LoadImage.VALIDATE_INPUTS() missing 1 required positional argument`), even
though `bypass_i2v` discards the picture. A 64×64 grey placeholder
(`input/zolex_placeholder.png`) satisfies the loader. It reaches no sampler.

## 4. A compiler fix was needed first

The workflow was exported by **frontend 1.48.7**, which writes only the
positional `widgets_values` array. The pack's three graphs carry
`widgets_values_named` on all 39 nodes and the compiler read that directly;
this file has it on **zero** nodes, so the compiler emitted nodes with no
inputs and the server rejected every one of them.

`worker/comfy/widget_values.py` now reads the positional form the way the
browser does, against `/object_info`. Three rules, each found by an actual
rejection:

* a seed carries its `control_after_generate` mode and a file picker its
  upload state — neither is an input the server accepts;
* a `COMFY_DYNAMICCOMBO` swallows its chosen option's inputs. Without this,
  `ResizeImageMaskNode` reads `["scale longer dimension", 1536, "lanczos"]`
  as two widgets and quietly takes **1536 as the scaling method**;
* those option inputs are addressed as `resize_type.longer_size`.

A subgraph instance is the same problem a level down, and its own input list
cannot solve it: the `Preprocess` instance omits `img_compression` from its
inputs while its values still carry it, so the widget slots come from the
definition. A layout the resolver cannot account for raises rather than
guessing. The named path is untouched and the pack's graphs compile byte for
byte, asserted with and without a catalogue (`tests/test_widget_values.py`,
10 tests; `tests/test_ltx_graphs.py` 29 unchanged).

## 5. Baseline — the workflow as supplied, direct to ComfyUI

Its own prompt, its own seed 42, nothing changed.

| | |
|---|---|
| Wall | **121.5 s** |
| Peak VRAM | **47,048 MiB** (45.9 GB) — includes the cold model load |
| Output | 1920×1080, 24 fps, **193 frames**, 8.0417 s |
| Audio | AAC 48 kHz stereo, 8.01 s, present and non-silent |
| Errors | none |

Delivered exactly what the graph specifies.

## 6. Quality

Inspected at 0, 25, 50, 75 and 100 % (`scratchpad/client-wf-baseline/`).

The prompt asks for three shots with hard cuts — cabin interior, doorway,
exterior freefall — and all three are present in order. Identity holds across
the cuts: the same woman in the same black helmet, clear goggles and blue
jumpsuit, the same instructor behind her. Anatomy is sound, hands and harness
buckles are clean, the horizon stays level, and skin and equipment hold
detail at 1080. Per-second luminance runs 116 / 95 / 100 / 66 / 124 / 130 /
131 / 130; the dip at 3 s is the doorway shot, bright sky against a dark
interior, not drift. Mean luminance 111.7, saturation 14.3.

No temporal artefacts, no morphing, no duplicated limbs seen at these five
points. This is a demanding multi-shot 8 s sequence and it holds together.

## 7. Through ZolexAI

`compile_fast_1080` (in `worker/comfy/ltx_graphs.py`) drives the graph from a
job's inputs — positive, negative, duration in seconds, seed, output prefix,
and an optional conditioning image with its own switch — and nothing else.
The workflow JSON is not modified; every edit is in memory.

Two bugs surfaced building it, both worth having: the seed and the image
switch are values promoted across a subgraph boundary and live in `inputs`
as literals, so writing the node's widget was silently discarded. A
literal-aware `FlatGraph.set_value` fixes it. A silently ignored seed would
have made every render identical for the wrong reason.

**Phase 7, the same prompt through `LtxComfyService`** — the submit / poll /
collect path the production adapter uses:

| Run | Wall | Peak VRAM | Output |
|---|---|---|---|
| Direct ComfyUI, seed 42 | 121.5 s | 47,048 MiB | 1920×1080, 193 f, 8.0417 s |
| ZolexAI, seed 42 | **6.1 s** | 2,990 MiB | identical probe |
| ZolexAI, fresh seed 987654 | **112.7 s** | 34,476 MiB | 1920×1080, 193 f, 8.0417 s |

The 6.1 s is the finding, not an anomaly: ComfyUI's per-node cache reused
every node upstream of the output, which means **the prompt ZolexAI compiled
was identical to the direct one** everywhere that matters — stronger evidence
than comparing pixels. Only `SaveVideo` re-ran, because the job set its own
output prefix. With a fresh seed the work is real and the wall time matches
the direct run (the direct run's extra ~9 s and higher VRAM are the cold
model load).

And the two files are **pixel-identical**: `psnr` over all 193 frames reports
`mse_avg 0.00` and `psnr_avg inf` on every plane. Not "close" — the same
pixels.

**Conclusion: the adapter does not alter the client's generation behaviour.**

## 8. NVFP4 comparison

Not applicable to this workflow: it already specifies NVFP4, so nothing was
switched. The separate question — NVFP4 versus int8 on *our* Text to Video
graph — was measured the same day and is in
`ltx-t2v-legacy-adapters.md`: at 30 s and a fixed seed, int8 190.3 s against
NVFP4 192.1 s, the same speed, with NVFP4 visibly darker and less saturated.
On that graph int8 is the better of the two. That says nothing about this
workflow, which was built around NVFP4 and renders well on it.

## 9. What changed in the repo

| Change | Why |
|---|---|
| `worker/comfy/widget_values.py` (new) | read positional widget exports |
| `worker/comfy/ltx_graphs.py` | `flatten(catalogue=…)`, `FlatGraph.set_value`, `Fast1080Edits`, `compile_fast_1080` |
| `tests/test_widget_values.py` (new, 10) | pin the resolver and the byte-for-byte guarantee |
| `benchmarks/client-pack/ltx25/client_original/` | the client's file, unedited |

Video to Video is untouched: its workflow, models, parameters, routing and UI
are unchanged, and nothing here is on its path. The three pack graphs compile
byte for byte, asserted. Rollback is `git revert` of the two commits; the
downloaded VAE can stay, since no existing workflow names it.

## 10. Recommendation

**The workflow works, and it is good.** 8 s of 1920×1080 with audio in about
two minutes, three clean shots, identity held. Per megapixel-frame it is
slightly cheaper than our current Text to Video (0.30 s against 0.34), which
is what a single 8-step stage with no LoRAs, no detailer and no second pass
should cost.

It is **not a drop-in replacement** for our Text to Video, and the difference
is a product decision rather than a technical one:

* it is fixed at 1920×1080, against our 1280×704 — 2.3× the pixels;
* the delivered graph is 8 s. Its duration is a plain input and the frame
  formula is the model's own lattice, so other lengths are a setting, but
  every length beyond 8 s needs measuring: at 2.3× the pixels a 30 s render
  is a different proposition, and 45.9 GB peak at 8 s is the number to watch;
* it is single stage, so it trades the second refinement pass for speed.
  Whether that reads as better or worse is the client's judgement, and the
  side-by-side to put in front of them is this against our current output at
  the same prompt.

**Suggested next step:** measure it at 15 s and 30 s and check peak VRAM,
then show the client both outputs at one prompt and let them choose whether
this becomes a new high-resolution tier or replaces the current one.
