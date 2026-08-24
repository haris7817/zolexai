# Tier 1 — decision-critical H3 measurements

**Date:** 24 August 2026 · Instance `C.48538452` · H3 revision `42ed227e` ·
diffusers `0.40.0.dev0` · torch `2.11.0+cu128` · bf16, provider-native.

**`num_inference_steps = 30` on every run below is PROVISIONAL.** It is
specified nowhere by the provider — not in the blocks, the checkpoint, the
README or the model docs, and the official request omits it. Every number here
carries that caveat. **GPU TUNING REQUIRED.**

Nothing deployed. `auto` still resolves to LTX. Full 406-run pack NOT executed.

---

## 1. What ran, and what it cost

All at the provider-native canvas (1344x768, 768 short edge), 124 frames
(5.167 s), 29 evaluations.

| Test | References | Wall | x real time | Peak VRAM | Peak host | Swap | Result |
|---|---|---:|---:|---:|---:|---:|---|
| **B1** subject image | 1 image | 580.9 s | 112x | 81.7 GB | 71.0 GB | 0.2 GB | **PASS** |
| **B2** video reference | 1 video (120 f) | 1127.9 s | 218x | 88.2 GB | 73.5 GB | 0.2 GB | **PASS** |
| **B3** supplied audio | 1 image + 1 audio | 663.0 s | 128x | 83.3 GB | 71.9 GB | 0.2 GB | **PASS** |
| **B4** continuation | 1 video (120 f) | *in flight* | | | | | *pending* |

**Cost is driven by reference payload, not just output length.** An image
reference costs 112x real time; adding an audio reference takes it to 128x; a
120-frame video reference takes it to **218x**. D-group (person image + source
video) and E-group (singer + song) carry the heaviest payloads, so they sit at
the expensive end — which is exactly where the decision matters.

## 2. Two memory limits, both found the hard way

### 2.1 Length: 14.375 s does not run on this machine

B3 was first attempted at **345 frames (14.375 s)** — inside H3's documented
5–15 s range. The process was **killed silently**: no Python traceback, no CUDA
error, no `ABORT` from the guard (then set at 8 GB, sampling every 5 s). A CUDA
OOM raises a catchable exception; a silent SIGKILL with no trace is
**consistent with the host OOM killer**, though `dmesg` is not readable in this
container so it is not directly confirmed.

Re-run at **124 frames (5.167 s)** it passed comfortably at 71.9 GB host.

> **Consequence for routing.** H3's documented 5–15 s window is **not
> achievable on this machine at provider-native resolution** — only the bottom
> of it is. A 30 s music-video section becomes **6 H3 generations** rather than
> 2, at 128x real time each. That compounds the speed gap rather than offsetting
> it.

This needs one confirmation run at 345 frames under the raised guard before it
is treated as settled.

### 2.2 Memory accumulates across generations in one process

B4 ran third in a single process, after B2 and B3, and the guard fired at
**1 GB available**. The same workflow with the same reference kind had peaked at
73.5 GB as B2 in that same process. Nothing about B4 is heavier than B2.

> **Consequence for the harness.** H3 generations must run **one per process**.
> Batching them in a loop exhausts host RAM regardless of the individual
> workload, and the failure looks like a workflow failure when it is not.

B4 is being re-run alone to confirm.

## 3. B3 — is `fully_copy` real? **Yes, and it is not a mux**

Cross-correlation of the output soundtrack against the supplied window, both
decoded to 32 kHz mono, with the LTX post-mux case as a control:

```text
H3 fully_copy    correlation +0.954   lag 0.0 ms   RMS -15.61 -> -13.84 dBFS
LTX post-mux     correlation +1.000   lag 0.0 ms   RMS -13.88 -> -13.91 dBFS   (control)
```

The control is what makes this readable. A byte-level mux scores **+1.000** and
preserves level exactly. H3 scores **+0.954** and comes out 1.8 dB louder — the
supplied song is carried through, but **reconstructed through the audio VAE**
rather than copied, which is what the architecture describes ("video and audio
come out of the same denoising loop"; an audio reference "is encoded by the
audio VAE alone").

That distinction is the whole point:

- **LTX default**: the track is byte-identical *because the model never saw it*.
- **H3 `fully_copy`**: the track is slightly altered *because the audio was
  inside the generation*.

Only the second can produce mouth movement that follows a vocal.

## 4. Lip-sync evidence — measured, with a control

Per-frame mouth-ROI motion against the vocal envelope at 24 fps:

```text
H3 B3 fully_copy   motion 4.794   corr(vocal, motion) +0.338   best lag +500 ms
LTX post-mux       motion 4.545   corr(vocal, motion) +0.128   best lag -208 ms
```

**H3 tracks the vocal about 2.6x better than the LTX control.** Both mouths move
about equally much; only H3's movement is correlated with the singing.

| | Evidence level |
|---|---|
| **H3 Ref2VA `fully_copy`** | **between A and B** — audio exists and is the supplied song; motion correlates positively but not strongly enough to assert B |
| **LTX default music video** | **A only** — audio exists, mouth motion is uncorrelated with it |

**Level C is not claimable from this probe** for either engine.

Three weaknesses in my own probe, stated so the number is not over-read:

1. The best lag landed at **+12 frames, the edge of the search range**, so part
   of the correlation may be spurious rather than a real 500 ms offset.
2. The ROI is a fixed box; the singer's head moves, so head motion contaminates
   the mouth signal.
3. **The window contains no instrumental gap.** The strongest available test —
   does the mouth stop when the vocal stops — cannot be run on it. The frozen
   song has a break at ~53–55 s, and a window spanning it is the right next
   measurement.

Visually, the B3 output holds the singer's identity, wardrobe and room from the
reference image and shows genuinely varied mouth shapes across the clip — open
wide with teeth visible, open narrow, and fully closed.

## 5. LTX music-video baseline — already solved in our own code

Tracing the path (Part 2, C1) found that **no new code is needed**, and that the
behaviour I measured yesterday was already documented:

```text
execution.audio_conditioning: true  ->  _A2VID = ltx_pipelines.a2vid_two_stage
    transformer_dev + distilled LoRA, quantize=False, offload_cpu=True
    args: --audio-path, --audio-start-time, --audio-max-duration
    the MASTER file seeked, never a re-encoded per-section slice
```

`adapters/ltx.py:62-64` states it plainly: *"The default never shows the model
the music at all, so no amount of prompting can make a performer's mouth follow
a vocal."*

So this was **documented behaviour behind an existing opt-in**, in the same
shape as `v2v_engine: transform` — not an undiscovered bug. What yesterday's
measurement added is the empirical proof that the default is prompt-only plus
post-mux, and the benchmark-validity consequence.

The flag is read at `ltx.py:1904`, commented out at `music-video.yaml:106`, and
is already benchmark-only and default OFF. **C2 and C3 require no change.**

Guided-tier weights are now on the box: `ltx-2.5-22b-dev-transformer-bf16`
(39.13 GB) and `ltx-2.5-22b-distilled-lora-450-bf16` (8.29 GB).

## 6. Preliminary cost projection — **PRELIMINARY, not final**

Measured on this card, provider-native both sides:

```text
LTX 2.5   5.00 s in  28.2 s  =   5.6x real time
H3        5.17 s in 580.9 s  = 112.3x real time   (image ref)
                     663.0 s = 128.3x            (image + audio ref)
                    1127.9 s = 218.2x            (video ref)
```

Against the frozen pack's 4,945 s of H3 video, at the **cheapest** observed
reference load:

```text
H3 side    ~154 GPU-hours   ~$200 at $1.296/hr
at the video-reference rate it is closer to 300 hours
LTX side     ~8.8 GPU-hours   ~$11
```

**Label: PRELIMINARY.** Resolution, steps and runtime may all change, and every
figure carries the provisional step count. The documented 960x544 canvas is
worth about 2.3x per step, but that is a deviation to test deliberately
(`H3_NATIVE_RES` vs `H3_COST_RES`), not to apply silently.

## 7. Execution tiers

The 406-run pack is **unchanged and remains the launch-grade suite**. Only
execution priority changes.

| Tier | Contents | Status |
|---|---|---|
| **1 — decision-critical** | B1–B4, LTX native A2V, D1 LTX vs H3, one short E-group pair | in progress |
| **2 — only if H3 earns it** | 15 s and 30 s cells, reliability repeats, more D/E/B cases | **blocked** on Tier 1 |
| **3 — full pack** | all 41 cases / 100 cells / 406 runs | **not started, do not start** |

No frozen case has been deleted or altered.

## 8. Open

- B4 confirmation in a fresh process.
- One 345-frame run under the raised guard, to settle whether the 14.375 s
  failure is host OOM.
- LTX native A2V smoke on the same window as B3.
- D1 and the E-group pair.
- **FL2VA still not downloaded.** Correct — Ref2VA has not yet earned it.
