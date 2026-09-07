# The client's FAST 1080 workflow at 8, 15 and 30 seconds (7 Sep 2026)

A controlled duration benchmark of the client's own high-resolution workflow.
Only the duration was changed; everything else — model, sampler, sigmas, CFG,
resolution, FPS, prompt, seed, conditioning, output encoding — stayed as
delivered, and that is demonstrated per run rather than asserted (§3).

Companion to `client-ltx25-fast-1080-validation.md`, which covers the graph
analysis and the ZolexAI integration.

## 1. Identity

| | |
|---|---|
| File | `LTX2.5_ACTUAL_WORKFLOW_ONLY_FAST_1080_8s_AUDIO.json` |
| sha256 | `19480e7432539bed3e478ef1ee998df906857cf5103e4104daa73989b1f6e712` |
| Bytes | 151,017 |
| Workflow id / revision | `7c9d2167-35ab-4e5f-b51f-0b349b6021de` / 0 |
| Nodes | 12 root + 41 across 5 subgraphs = **53**; 25 root links |
| Frontend | 1.48.7 (`comfy-core` 0.3.30, `comfyui-kjnodes` 1.0.8) |
| Immutable copy | `benchmarks/client-pack/ltx25/client_original/` — never written |

## 2. Configuration (unchanged in every run)

Transformer `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`, **NVFP4**,
`weight_dtype: default`. Video VAE `ltx-2.5-video-vae-conv-bf16.safetensors`,
audio VAE `ltx-2.5-audio-vae-bf16.safetensors`, text encoder
`gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`.
**No LoRAs. No detailer.** `CFGGuider` cfg **1**, `euler_ancestral`,
`ManualSigmas` with 9 values = **8 steps, single stage**, no refinement pass,
no upscaler. Generates **1920×1088**, `VAEDecodeTiled` (1536 / 256 / 256 / 32),
`ImageScale` lanczos centre to **1920×1080**. **24 fps**. Audio generated
in-graph. Seed **42**, the workflow's own skydiving prompt and negative.

Frame count is the graph's own arithmetic, `1 + floor(fps × seconds / 8) × 8`,
so it is computed, not chosen:

| Duration | Frames | Delivered | 8k+1 |
|---|---|---|---|
| 8 s | 193 | 8.0417 s | ✓ |
| 15 s | 361 | 15.0417 s | ✓ |
| 30 s | 721 | 30.0417 s | ✓ |

## 3. Only the duration changed — shown, not claimed

Each run compiles the workflow in memory and diffs the compiled prompt
against the 8 s one before submitting. Both longer runs reported:

```
15 s → nodes differing from the 8 s prompt: 1
   5512 PrimitiveFloat: {'value': 8.0} -> {'value': 15.0}
30 s → nodes differing from the 8 s prompt: 1
   5512 PrimitiveFloat: {'value': 8.0} -> {'value': 30.0}
```

One node, the duration primitive. Every other node byte-identical.

## 4. Results — all three completed

| Metric | 8 s | 15 s | 30 s |
|---|---:|---:|---:|
| Resolution | 1920×1080 | 1920×1080 | 1920×1080 |
| FPS | 24 | 24 | 24 |
| Frames | 193 | 361 | 721 |
| Duration | 8.0417 s | 15.0417 s | 30.0417 s |
| Generation time | **121.5 s** | **306.3 s** | **1050.6 s** |
| Queue wait | 0.0 s | 0.0 s | 0.0 s |
| Peak VRAM | 45.9 GB¹ | 34.3 GB | **56.9 GB** |
| Peak RAM | not sampled | 230 GiB² | **202 GiB** |
| Max GPU util | — | 100 % | 100 % |
| Output size | 9.2 MB | 20.3 MB | 32.0 MB |
| Audio | 48 kHz stereo, 8.01 s | 48 kHz stereo, 15.01 s | 48 kHz stereo, 30.01 s |
| Stability | success | success | success |
| Errors | none | none | none |

¹ The 8 s figure includes a cold model load; the warm re-run measured 34.5 GB.
² The 15 s RAM figure is **not comparable**: ComfyUI's execution cache from
earlier runs was still resident. The cache was cleared before the 30 s run
(container went 136.9 → 67.5 GiB), so 202 GiB is the trustworthy number and
the workflow's own requirement is lower than 230 GiB suggests.

### Cost per unit

| | 8 s | 15 s | 30 s |
|---|---:|---:|---:|
| Seconds per frame | 0.630 | 0.848 | **1.457** |
| Seconds of compute per second of video | 15.1 | 20.4 | **35.0** |
| Frames vs 8 s | ×1.00 | ×1.87 | ×3.74 |
| Time vs 8 s | ×1.00 | ×2.52 | **×8.65** |
| VRAM vs 8 s | ×1.00 | ×0.75 | ×1.24 |

**Time grows far faster than length.** 3.74× the frames costs 8.65× the time.
Per-frame cost more than doubles from 8 s to 30 s, which is the signature of
attention over a longer sequence, not of anything misconfigured.

## 5. ZolexAI parity

The same compiled prompt driven through `LtxComfyService` — the submit /
poll / collect path the production adapter uses.

| | 15 s | 30 s |
|---|---|---|
| Wall through ZolexAI | 9.1 s | 15.2 s |
| PSNR vs direct | y `inf` u `inf` v `inf`, average `inf` | y `inf` u `inf` v `inf`, average `inf` |
| SSIM vs direct | Y 1.000000 U 1.000000 V 1.000000 **All 1.000000** | Y 1.000000 U 1.000000 V 1.000000 **All 1.000000** |
| MSE | 0.00 on every plane | 0.00 on every plane |
| Audio (sha256 of decoded PCM) | `7d4ca825…` = `7d4ca825…` | `77507dcf…` = `77507dcf…` |
| Probe | 1920×1080, 361 f, 15.0417 s | 1920×1080, 721 f, 30.0417 s |

The short wall times are the finding, not an anomaly: ComfyUI's per-node
cache reused everything upstream of the output, which means the prompt
ZolexAI compiled was identical to the direct one. Only `SaveVideo` re-ran,
because the job sets its own prefix — which is also why the files differ by
one byte in length while the pixels and audio are identical.

**ZolexAI does not alter the client's workflow result, at either length.**

## 6. Quality

Inspected at 0, 25, 50, 75 and 100 % of each output; tile at
`scratchpad/bench-highres/quality-8-15-30.png`.

All three are coherent and hold detail at 1080. Identity is consistent within
each run: the same woman, helmet, goggles and jumpsuit, the same instructor.
Anatomy is sound, hands and harness hardware are clean, the horizon stays
level, no flicker, morphing or duplicated limbs at the sampled points. Audio
is present and the right length in all three.

| | 8 s | 15 s | 30 s |
|---|---:|---:|---:|
| Mean luminance | 111.7 | 106.1 | 105.7 |
| Luminance drift, first to last second | +14.5 | +9.7 | **+24.6** |
| Mean saturation | 14.3 | 11.6 | **9.8** |

Two honest observations. **Saturation falls as the clip lengthens**, 14.3 to
9.8, and **drift is largest at 30 s**. Neither is severe and neither reads as
a defect in the frames, but the trend is real and would be the thing to watch
if 30 s were offered. **Pacing stretches rather than gaining shots:** at 30 s
the sequence spends its first three sampled points still inside the aircraft,
where the 8 s run has already jumped. The prompt describes a fixed set of
shots, so a longer duration dwells on them rather than inventing more.

## 7. Hardware

**NVIDIA RTX PRO 6000 Blackwell Max-Q, 97,887 MiB.** Host RAM 503 GB;
container cgroup ceiling **241.5 GiB** (`memory.max`), with 4 historical
`oom_kill` events on this box.

**Nothing here is a hardware limitation.** At the worst point, 30 s used
56.9 GB of 97 GB VRAM (58 %) and 202 GiB of the 241.5 GiB container ceiling
(84 %). Both completed with headroom and the `oom_kill` counter did not move.

The limitation is **compute time**, and it belongs to the workflow's own
design rather than to our hardware or integration: at 2.09 megapixels per
frame with attention over 721 frames, 30 s costs 17.5 minutes. Classify as a
**workflow cost characteristic**, not a hardware or software fault.

Worth noting for capacity: the 84 % RAM figure was reached only after
clearing ComfyUI's cache first. Left uncleared, the 15 s run alone reached
230 of 241.5 GiB. Running long jobs back to back without freeing the cache is
the realistic OOM risk, not any single render.

## 8. Recommendation — **B, a separate high-resolution option**

Not A. It cannot replace the current Text to Video: that renders 1280×704 in
215 s at 30 s, where this needs 1050 s for the same length. Five times the
wait as a silent substitution is not acceptable.

Not C or D. Nothing is experimental or blocked. It ran three times without a
single error, at three lengths, with byte-exact parity through our stack and
comfortable hardware headroom. There is nothing to optimise before it can be
deployed, only a decision to take.

**Offer it as a distinct high-resolution tier, and make the length ladder
match its cost:**

* **8 s at about 2 minutes** — the sweet spot, and what the client built it for.
* **15 s at about 5 minutes** — reasonable.
* **30 s at 17.5 minutes** — technically fine, commercially doubtful. On a
  node that serves one job at a time, one 30 s render blocks the queue for
  the length of eight 8 s renders. Recommend holding it back, or gating it,
  until there is capacity to absorb it.

The quality argument for the tier is strong: 1920×1080 against 1280×704 is
2.3× the pixels, and per megapixel-frame it is *cheaper* than our current
graph (0.30 s against 0.34), because it runs one 8-step stage with no LoRAs,
no detailer and no refinement pass.

## 9. Rollback

Benchmark commit `640b8c1` is the stable point recorded before this work; the
benchmark itself changed no repository file. Everything it produced lives
outside the tree, on the node under `/workspace/holdcheck/bench/` and locally
under `scratchpad/bench-highres/`.

Nothing in production was touched: the three pack graphs, Video to Video,
Character Replacement and Music Video are unmodified, and no deployed
configuration was changed. The client's JSON is byte-identical to the file
they sent. To undo the earlier integration work entirely, revert `a0553ab`
and `640b8c1`; the downloaded conv VAE may stay, since no existing workflow
names it.
