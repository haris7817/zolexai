# Text to Video: the LTX 2.3 adapters and the LTX-2 detailer (7 Sep 2026)

The client's ComfyUI operator read our Text to Video graph and reported that
it "uses the LTX 2.5 core files, but mixes them with older LTX 2.3/2.0
LoRAs", that "the active older LoRAs are the biggest problem and could
contribute to darkening, color changes, weak prompt adherence, and
inconsistent motion", and that they may be "why videos are taking longer".
Their instructions: switch off `LTX-2.3-OmniNFT-RL-Lora` and
`ltx2.3-transition` in the Power Lora Loader, bypass the
`ltx-2-19b-ic-lora-detailer`, optionally bypass `ModelPreviewOverrideKJ`,
and generate a 30 s result as connected shorter sections.

## 1. What we actually submit

Measured, not assumed — the compiled API prompt for a 30 s Text to Video:

| Node | Class | What it carries |
|---|---|---|
| `366` | `UnetLoaderGGUF` | `LTX-2.5-Distilled-Q8_0.gguf` |
| `5730` | `Power Lora Loader (rgthree)` | `lora_1` OmniNFT-RL 2.3 @ 0.4 **on**, `lora_2` transition 2.3 @ 0.8 **on** |
| `5747` | `ModelPreviewOverrideKJ` | preview only; the node runs `--preview-method none` |
| `5464:5565` | `LoraLoaderModelOnly` | `ltx-2-19b-ic-lora-detailer` @ 0.3 |

The model chain is `366 → 5730 → 5747 → 5464:5565 → guider`, and `5730`
also feeds `clip` to both text encoders.

**Three corrections to the operator's reading.**

1. **The detailer feeds one guider, not both.** `5464:5748` (the
   full-resolution second stage) takes the model through it; `5464:5750`
   (the 8-step half-resolution first stage) already takes it straight from
   `5747`. Bypassing the detailer makes the two stages agree — a smaller
   change than "connect the model directly to both".
2. **The Power Lora Loader patches the text encoder too**, so the 2.3
   adapters do touch prompt adherence — by a route the operator did not
   name, but their instinct on adherence is defensible.
3. **None of this is in Character Replacement.** Its own rgthree loader
   (node `1371`) is empty; it loads only `LTX25_Ripple_v11` @ 1.35, which
   is the graph's entire purpose. The request applies to Text to Video and
   First/Last Frame (so also image-to-video and extend-video).

## 2. The adapters are not incompatible — they load and apply in full

The "2.5 core files mixed with older LoRAs" framing implies a mechanical
mismatch. There is none. Reading the safetensors headers on the node and
comparing target module names against the LTX 2.5 22B transformer:

| Adapter | Tensors | Target modules | Matched in LTX 2.5 22B |
|---|---|---|---|
| `ltx-2-19b-ic-lora-detailer` | 960 | 480 | **480 (100 %)** |
| `LTX-2.3-OmniNFT-RL-Lora_bf16` | 2688 | 1344 | **1344 (100 %)** |
| `ltx2.3-transition` | 1152 | 576 | **576 (100 %)** |

The detailer covers transformer blocks 0–47; the model has exactly 48. The
ComfyUI log carries no "lora key not loaded" and no shape warning for any
of them. So all three genuinely patch the model, and the question is not
compatibility but taste (`scratchpad/keymatch.py`).

## 3. The LoRAs cannot be why renders are slow

The measured ladder through this graph (RTX PRO 6000, 1280×704, models
resident) is 5 s = 48.8 s, 10 s = 76.3 s, 15 s = 106.3 s, 30 s = 215.2 s —
about 0.28 s per output frame plus a ~20 s fixed cost. **Every one of those
cells already had all three adapters applied**, and a LoRA is a weight patch
paid once at model load, not per frame. The slope is frames.

The candidate with a real mechanism is the one the operator raised almost in
passing: Text to Video loads the **community Q8_0 GGUF**, which is
dequantized on every forward pass, while Character Replacement already loads
Lightricks' **int8 safetensors**. Two different transformers under three
product names, and the node already carries the int8 and nvfp4 builds.

## 4. The A/B on the client's own 30 s boxing prompt

Same prompt (job `aa1118c7`), same seed 424242, same 30 s, 16:9.

| Arm | Render | Y mean | Y drift over the clip | Saturation | Job |
|---|---|---|---|---|---|
| as delivered | **222.1 s** | 62.9 | **+11.0** | 14.5 | `cf3d7b3a` |
| both 2.3 LoRAs off + detailer bypassed | **210.9 s** | 60.3 | **+34.6** | 12.3 | `121cafc2` |

The operator's changes are 5 % faster and **worse on the very thing they
were meant to fix**: exposure drift over the clip tripled, and the clip
starts far darker (Y 34.2 against 49.9). Visually the cleaned arm frames
wider — the boxers sit further from camera and hold noticeably less face and
skin detail (`scratchpad/t2v-compare.png`).

That is consistent with what OmniNFT is: a reward-tuned quality pass over
the whole model. It is part of this pack's look, not a defect sitting on top
of it.

**Verdict: do not flip these on by default.** The switches ship off. One
prompt and one seed is not a proof, but it is enough to refuse a silent
change to every video the client generates, and the direction is against the
hypothesis.

## 5. What shipped

All compile-time, all default off; the frozen JSONs are never written and a
deployment that says nothing renders exactly what it rendered before.

| Setting | `execution` key | Meaning |
|---|---|---|
| `ltx_comfy_disabled_loras` | `disabled_loras` | comma-separated file-name fragments switched off in the Power Lora Loader |
| `ltx_comfy_bypass_detailer` | `bypass_detailer` | bypass the 19B detailer, both guiders then take the model directly |
| `ltx_comfy_transformer` | `transformer` | override the transformer file; the loader class follows the extension |

Turning a Power Lora Loader entry off sets `on: false` and leaves the file
and strength as delivered — rgthree's own gate is
`if value['on'] and (strength_model != 0 or strength_clip != 0)`, read in
`power_lora_loader.py` on the node, so this is the switch the operator would
flick in the browser. The detailer is matched by file name, not node id,
because the id differs between the two graphs. `ltx_comfy_submitted` now
logs the model chain, including a fragment that matched no LoRA, so a render
can be traced to the chain that produced it.

## 6. Sectioned 30 s generation — not yet, and not at 10 s

Today a 30 s Text to Video is one submission; `_run_generation` refuses
anything above the per-pass ceiling rather than chaining, so lowering
`ltx_comfy_max_segment_seconds` would **fail the job**, not section it.

Three reasons to hold:

* **It is slower here, not faster.** 215 s as one pass, against ~227 s for
  2 × 15 s and ~258 s for 3 × 10 s, on the measured ladder.
* **The soundtrack restarts at every section.** Both graphs generate their
  own audio per submission; the joins are butt joins with an 80 ms fade.
* **Every section gets the same prompt today**, which is the recorded cause
  of "the action repeats every 10 seconds" on the older runtime.

A 10 s cap would also put the 15 s rung off the 8k+1 lattice (7.5 s windows
→ 181 frames, which the graph rejects). Legal caps are 30, 20, 15, 6, 5. If
we section, it should be **2 × 15 s**, after per-section prompts and the
audio seam are fixed.

## 7. Open

* The int8 transformer arm (speed) — running.
* `ModelPreviewOverrideKJ` bypass: preview-only and therefore safe, but it
  decodes and JPEG-encodes up to 240 preview frames per run, so there may
  be a saving. Not built; it needs its own flag and its own timing arm.
* If the client wants the operator's changes anyway, they are one
  deployment setting each — but they should see the side-by-side first.
