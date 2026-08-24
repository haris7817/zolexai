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
| **B4** continuation | 1 video (120 f) | 1130.9 s | 219x | 88.2 GB | 73.1 GB | 0.2 GB | **PASS** (fresh process) |

**Cost is driven by reference payload, not just output length.** An image
reference costs 112x real time; adding an audio reference takes it to 128x; a
120-frame video reference takes it to **218x**. D-group (person image + source
video) and E-group (singer + song) carry the heaviest payloads, so they sit at
the expensive end — which is exactly where the decision matters.

## 2. Two memory limits, both found the hard way

### 2.1 Length — CORRECTED: 14.375 s DOES run

**This section previously said the opposite, and it was wrong.**

The first 345-frame (14.375 s) attempt was killed silently — no traceback, no
CUDA error — and I recorded provider-native 15 s as unavailable on this
machine, with a routing consequence attached. That attempt ran **second in a
shared process, after B2**. Once §2.2 established that memory accumulates per
process, the failure had a second possible cause and the conclusion was not
safe.

Re-run **alone, in a fresh process**, with 0.5 s tracing:

```text
345 frames / 14.375 s   image + audio references
  wall                2429.9 s   = 169x real time
  peak host RAM         77.9 GB  (minimum free 47.2 GB)
  peak VRAM             95.0 GB of 95.6   <- 99.4%, saturated
  peak swap              0.2 GB
  PASS
```

**Provider-native 15 s is available on this configuration.** The earlier failure
was accumulation, not length. The section-count assumptions revert:

```text
30 s music video   H3 = 2 sections   (not 6)
60 s               H3 = 4 sections
```

Two real limits do come out of the corrected run:

- **VRAM is saturated at 99.4%** for 345 frames with an image and an audio
  reference. A *video* reference costs more than an image one (88.2 GB at only
  124 frames), so **345 frames plus a video reference will not fit**. Long
  D-group cells are the ones to watch, not long E-group cells.
- **Cost per second of output rises with length**, as quadratic attention
  implies: 128x real time at 124 frames, **169x at 345**. Longer H3 clips are
  disproportionately expensive, so chaining shorter sections is cheaper per
  second than generating long ones — the opposite of the intuition that fewer,
  longer sections save time.

### 2.2 Memory accumulates across generations in one process

B4 ran third in a single process, after B2 and B3, and the guard fired at
**1 GB available**. The same workflow with the same reference kind had peaked at
73.5 GB as B2 in that same process. Nothing about B4 is heavier than B2.

> **Consequence for the harness.** H3 generations must run **one per process**.
> Batching them in a loop exhausts host RAM regardless of the individual
> workload, and the failure looks like a workflow failure when it is not.

**Confirmed.** Re-run alone, B4 completed at 1130.9 s / 88.2 GB VRAM /
73.1 GB host — within three seconds and 0.4 GB of B2, which is the same
reference kind. B4 was never heavier than B2; the failure was accumulation
alone.

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

### 4.1 B4 — continuation is soft, not seam-exact

B4 passes: the reference is consumed, the scene does not reset, and the output
is coherent — same lake, same dawn light, same shoreline character. But the
camera **jumps**: the shoreline sits much closer and the sun has moved left
between the source's last frame and the continuation's first.

That is the documented behaviour rather than a defect. ref2va references "do
not bind the generated geometry" — they are encoded at their own resolution and
the target canvas defaults to H3's own. So **ref2va continuation is a soft,
scene-level continuation**, and a seam-exact one would need FL2VA's keyframe
(`image`) conditioning, which is the partition deliberately not downloaded.

Consequence for I-group (extend) and long-form: H3's continuation via ref2va
will not hold a seam the way LTX's frame conditioning does. If seam-exact
continuation matters, that is an FL2VA question, not a Ref2VA one.

## 8. Open

- LTX native A2V smoke on the same window as B3.
- D1 and the E-group pair.
- **FL2VA still not downloaded.** Correct — Ref2VA has not yet earned it.

---

## 9. E-group head-to-head — provider-native both sides

Same song window (48.0 s, spanning the measured vocal gap at 50.5–52.25 s),
same semantic intent, each engine prompted in its own provider's style.

| | LTX native a2vid | H3 Ref2VA fully_copy |
|---|---|---|
| Output | 1024x576, 241 frames, 10.042 s | 1344x768, 243 frames, 10.125 s |
| **Wall clock** | **147.9 s = 14.7x real time** | **1462.0 s = 144.4x real time** |
| Audio preservation | **+1.000**, RMS -14.72 -> -14.75 | +0.997, RMS -14.69 -> -13.51 |
| Audio reaches the model | **YES** — `--audio-path` proven in live argv | YES — reference in the denoising loop |
| **Goal-B (mouth settles in the gap)** | **ratio 2.368 — YES** | **ratio 2.822 — YES** |
| Identity conditioning | **none available** on this path | singer image reference |

**Both engines achieve Goal B.** The mouth settles during the instrumental gap
and resumes when the vocal returns, on both. The difference between 2.37 and
2.82 is inside the noise of a crude ROI probe and should not be read as H3
winning.

**H3 costs about 10x more wall clock for comparable presence behaviour.**

### 9.1 A measurement error, and its correction

The first pass reported LTX at ratio **1.086** — "no difference between singing
and silence" — which would have made LTX look structurally incapable of
lip-presence. **That was my probe, not the model.**

One fixed mouth ROI was used for both engines. H3 frames the singer tight and
LTX frames her wide, so the same box captured mostly face on one and a lot of
room on the other. Measured background motion in the LTX clip:

```text
background ROI   VOCAL 3.040    GAP 8.177
```

Background motion in the gap was more than double that during singing, and it
swamped the mouth signal entirely. Re-measured with that baseline subtracted,
LTX lands at 2.368 and clears Goal B comfortably.

The lesson is worth keeping: **a cross-engine visual metric has to be robust to
framing**, or it measures the shot rather than the model. Any Tier-2 scoring
that compares engines on pixels needs the same treatment.

### 9.2 What this does to the music-video question

Before this pair, the working assumption was that H3 was the only engine that
could lip-sync, because LTX's *default* path never receives the audio. That is
still true of the default. It is **not** true of LTX's native A2V path, which
receives the track and behaves comparably at a tenth of the cost.

So the E-group question changes from *"can LTX do this at all"* to *"is H3's
quality advantage — identity conditioning, resolution, articulation detail —
worth 10x the wall clock"*, which is a genuine product judgement rather than a
capability gap.

Level C (phoneme articulation) remains unclaimed for both.
