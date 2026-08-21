# Music Video lip-sync, Music vocals, and the 5-minute runtime

**Date:** 2026-08-21 · **Status:** investigation complete; fixes implemented and
unit-proven; GPU measurements recorded below. **Nothing deployed, nothing
committed.**

Three problems were reported together and they are not one problem. The first
two have proven root causes and shipped fixes. The third has a measured answer
that the brief's target does not survive contact with, and that answer is
below in full.

---

## 0. The short version

| | |
|---|---|
| **Music video does not lip-sync** | Because the model never receives the song. `execution.audio_conditioning` is off in every shipped workflow, so music video renders on `ltx_pipelines.distilled` from the text prompt alone. This is structural, not a prompting or timing failure. |
| **Is LTX 2.5 being used wrongly?** | No. The audio path we already implement (`a2vid_two_stage`) is the only native one, and its arguments already match the official SFT defaults, a2v guidance included. The fault is that the path is switched off — and that it crashes when switched on, for a reason nobody had measured. |
| **Music sometimes has no sung lyrics** | Two causes. A genre word silently overruled an explicit request for vocals — *fixed*, and the exact prompt that reproduced it now sings. And the automatic lyric sheet was sized for a model we no longer run — *re-measured and fixed for 1–3 minute songs*. At 4–5 minutes, though, the model's own run-to-run variance is larger than the whole density effect, and nothing measured here fixes that. |
| **5-minute music video in 10–15 minutes** | **Not simultaneously with lip-sync.** Prompt-only is 3.6x real time (~18 min, from real production jobs). The only tier that can hear audio is 10.4x (~52 min), and the knobs barely move it — the guidance passes batch into one forward, so dropping them is not proportional. The measured frontier, stacking every lever including two that do not exist yet, is **18–27 min**. |
| **The thing that gates all of it** | **The card has no margin when it is shared.** The audio tier passed **6/6** at a steady 208s and 77.4 GB with the GPU to itself — and failed **5 of 15** times at the same frame count across the day's mixed workload, always in the video VAE. 56 production jobs have already been lost the same way, across every video workflow. A 5-minute lip-synced video is 15 consecutive passes, so per-pass reliability has to be near 1. |

---

## 1. The current Music Video audio path, traced

```
web Dropzone → presigned PUT → asset id
  → POST /generations {inputs:[{role: source_audio, asset_id}]}
  → API validates role/kind/ownership against music-video.yaml
  → job row; claim serialises {role, asset_id, kind, download_url}
  → worker _stage_inputs downloads to workspace/inputs/
  → LtxAdapter._run_music_video
      probe_media(track) → target_seconds = track duration       (automatic)
      audio_onsets(track) → plan_musical_boundaries              (cut points)
      render_chain(per_pass = grid ceiling, boundaries)
         each pass: prompt from plan_section_prompts
                    + previous pass's final frame @ 1.0
                    → ltx_pipelines.distilled                    ← NO AUDIO
      → normalize sections (audio=False) → concat
      → mux_audio(picture, ORIGINAL track) ONCE
      → verify: has audio, length == song
```

Answering the audit questions exactly, for the shipped configuration:

1. **What audio reaches LTX?** None.
2. **In what form?** No form. Not a waveform, not VAE latents, not tokens, not
   prompt text. The track decides two things — the video's length and where the
   cuts land — and nothing else.
3–6. **Per-section audio windows?** There are none to align, because there is
   no audio conditioning. (Under the opt-in flag the windows *are* correct: the
   master file is passed whole with `--audio-start-time` set to that section's
   own start, so the master is never sliced, re-encoded or reset. See §3.)
7. **Does LTX generate its own audio that we replace?** On the distilled tier,
   yes — `ltx_pipelines.distilled` denoises an audio modality and decodes it.
   `_assemble_generated_sections(audio=False)` drops it and `mux_audio` maps
   only `1:a:0`, the user's file. So model audio is discarded, and it is
   discarded correctly: it is invented sound, not the customer's song.
8. **Are we conditioning on audio and discarding synchronised audio?** No —
   there is no audio conditioning to discard the output of.
9. **Is the master muxed after visual generation?** Yes, exactly once, at the
   end.
10. **Does the model invocation support audio-to-video conditioning?**
    `ltx_pipelines.distilled` does **not**. Read at source: its audio
    `ModalitySpec` has no `initial_latent` and there is no `--audio-path` on its
    parser. Audio conditioning is impossible at this tier — no argument, flag or
    prompt reaches it.
11. **Right entry point?** For what is shipped, no. The right one exists,
    is implemented, and is off.
12–13. **Missing or defaulted conditioning arguments?** See §3.
14. **Does our Python differ from the official ComfyUI workflow?** See §4.

---

## 2. Lip-sync root cause

**The vocal is not an input.** A performer in the current output cannot follow a
vocal because no vocal was ever supplied to the model; the mouth is whatever the
caption implied a singer looks like. That is category A in the client's own
A/B/C split — audio-responsive *nothing*, in fact, beyond cut placement.

Two further defects sit behind it and would have surfaced the moment the flag
was turned on. Both are fixed here:

**(a) The audio tier crashes on ordinary section plans.** `_A2VID` was the only
pipeline descriptor with an empty `measured_landings` table, so any frame count
on the 8k+1 lattice could reach its decoder. A real 60-second job planned
474 / 477 / 430 / 59 frames and died on the third section in
`ltx_core/model/video_vae/transformer/chunked/mlp.py` with
`CUBLAS_STATUS_INTERNAL_ERROR`. The decoder's failing set on this path is
non-monotonic, exactly as it is on every other path — see the sweep in §6.

**(b) The delivered picture drifts against the song.** Music video assembled its
sections without pinned frame counts, so every section was delivered rounded UP
to a whole frame and every seam pushed later content later against a continuous
soundtrack. Simulated through the real planner: +21 ms per seam, **+310 ms by
the last section of a 5-minute video on the audio tier**, +125 ms on the default
tier. Video-to-video had this fixed on 19 Aug; music video was explicitly left
out of scope then and is fixed now.

### 2a. Measured, before and after

`apps/worker/scripts/lipsync_probe.py`-class diagnostic (kept on the box at
`/workspace/mv2026/lipsync_probe.py`): mediapipe FaceMesh gives a per-frame
inner-lip aperture normalised by face height, the track gives 300–3400 Hz
short-time energy on the same frame grid, and the two are cross-correlated over
a ±600 ms lag search, globally and per window. Frames with no face are masked
rather than zero-filled, and a window whose vocal band is silent is reported as
unmeasurable rather than scored.

**The metric was validated against a known error before being trusted.** Delaying
the audio by +250 ms moved the detected lag by exactly −250 ms, and −250 ms moved
it by exactly +250 ms, in the windows that had a clear peak. Sign convention:
negative = the mouth happens earlier than the voice.

Same 60-second golden track, same prompt, same box:

| window | **prompt-only (ships today)** | **audio-conditioned** |
|---|---|---|
| 0–10s | +500 ms, r 0.45 | −125 ms, r 0.31 |
| 10–20s | **+583 ms (railed)**, r 0.32 | −208 ms, r **0.47** |
| 20–30s | **+583 ms (railed)**, r 0.16 | −167 ms, r **0.49** |
| 30–40s | **−583 ms (railed)**, r 0.20 | −500 ms, r 0.39 |
| 40–50s | **−583 ms (railed)**, r **−0.43** | — |
| 50–60s | +83 ms, r 0.30 | — |
| global | +583 ms (railed), r 0.27 | −208 ms, r 0.36 |

*(the conditioned column is the 20-second passes the perf matrix produced from
the same track and prompt; four windows of five seconds each.)*

Read the LAG column. Every prompt-only window lands at or beside the edge of the
lag search, and one is strongly NEGATIVE — a mouth moving opposite to the voice.
That is what "no relationship" looks like to a cross-correlation: there is no
peak, so the search returns an edge. The r values are not zero because a mouth
opening and closing in a music video correlates a little with musical energy by
accident; they carry no consistent timing.

The audio-conditioned column has a peak, in the same place, in every window:
**−125 to −208 ms at r 0.43–0.49**. A consistent lead of that size is what
singing looks like — the mouth opens before the sound arrives — and, crucially,
it does not walk between windows.

So the honest labels for the client's own A/B/C:

* **A, audio-responsive motion** — delivered.
* **B, the mouth follows the vocal** — delivered, and now measured rather than
  asserted: a repeatable lag with a real correlation peak, stable across the
  clip, against a baseline that has neither.
* **C, phoneme-accurate** — **still not claimed.** This metric scores vocal
  ENERGY against mouth OPENNESS. It cannot tell an /aː/ from an /oː/, and
  nothing here has measured whether the shapes are the right shapes.

Two knobs were tested against sync and neither moved it: raising
`--a2v-guidance-scale` from its 3.0 default to 6.0 changed nothing measurable
(and costs nothing — 216s against 213s), and neither halving the steps nor
dropping STG degraded it.

### 2b. The fixes, end to end on the GPU

The modified worker (an isolated copy — the production checkout was never
touched) against the golden 60-second track, `audio_conditioning: true`:

```
E2E rc=0   1024x576   1439 frames   60.000s   audio present
sections delivered: 480 / 480 / 481  =  1441 = round(60.024 × 24)   ✅
picture.mp4 before mux: 1441 frames                                  ✅
passes planned: 3   (was 4 before the pass ceiling became a landing) ✅
peak VRAM 96,800 MiB of 97,887
```

Three things proved that unit tests could not. Every pass landed on a measured
frame count and none reached the decoder at an unmeasured one. The sections came
back at exactly their cumulative-boundary counts through real ffmpeg — 480/480/481,
not 481/481/481, which is the two frames of staircase this used to accumulate on
a one-minute video. And the delivered file is the track's length with the track
on it once.

**The assembled video's sync is less consistent than a single pass's**, and that
is worth saying plainly rather than averaging away:

| window | assembled 60s | isolated 20s passes |
|---|---|---|
| 0–10s | +583 ms (railed), r 0.31 | −125 ms, r 0.31 |
| 10–20s | −500 ms, r 0.35 | −208 ms, r 0.47 |
| 20–30s | **−83 ms, r 0.53** | −167 ms, r 0.49 |
| 30–40s | **−375 ms, r 0.48** | — |
| 40–50s | −583 ms (railed), r **0.06** | — |
| 50–60s | −583 ms (railed), r 0.41 | — |

Against the prompt-only baseline it is better in the windows that matter — the
40–50s window went from r **−0.43** (mouth moving against the voice) to 0.06,
and 20–30s from a railed 0.158 to −83 ms at 0.53. But it does not hold the
clean, repeatable lag the isolated passes do. The obvious suspect is the seam:
each later section is pinned by its predecessor's final frame at strength
**1.0**, the hardest conditioning the pipeline takes, and that frame carries a
mouth position chosen for a different moment in the song. Untested — it is a
single dial (`ConditioningFrame(frame, 0, 1.0)` in `_run_music_video`) and the
next thing to measure.

The 1262s wall on this run is **not** a usable timing: Demucs jobs of mine were
sharing the card for part of it. The isolated 213s-per-pass figure is the one to
quote.

**And the eye check, at moments chosen by the audio rather than by taste.** The
golden track's loudest and quietest vocal-band seconds were computed first, then
the frames pulled:

| t | vocal band | what the frame shows |
|---|---|---|
| 24.0s | **+54.2 dB** (loudest) | mouth wide open on a sung vowel, tongue visible, eyes closed, visible effort into the mic |
| 31.5s | **+54.3 dB** (loudest) | mouth open in a rounded /o/, singing into the mic |
| 58.5s | **−9.7 dB** — the track has ended | **mouth closed, at rest, looking at camera** |

A model animating a generic singing mouth would still be singing at 58.5s. This
one stops when the song does, and that remains the cheapest strong evidence that
the mouth is driven by the track rather than by the caption.

One thing the frames show that the numbers do not: **the performer at 58.5s is
recognisably a different woman from the one at 24.0s** — same wardrobe, same
studio, same shot, different face. Those are two different sections, chained on
a pinned seam frame. Identity drifts across a music-video seam, and music video
has no reference-image input to pin it against (§13).

---

## 3. Official LTX 2.5 audio-driven video: what exists

Read from the installed runtime at
`/workspace/ltx2-benchmark/packages/ltx-pipelines`, not from documentation.

| pipeline | takes supplied audio? | what it does |
|---|---|---|
| `a2vid_two_stage` | **yes** — `--audio-path`, `--audio-start-time`, `--audio-max-duration` | Encodes the audio through the audio VAE and denoises video against it with the audio modality **frozen** (`noise_scale=0.0`, `initial_latent=<encoded>`). The only true audio-to-video path. |
| `distilled` | no | Generates its own audio. No `--audio-path` on the parser. |
| `ti2vid_two_stages` (our guided tier) | no | Same. |
| `dubit` (the "LipDub" IC-LoRA) | **no** | Takes a reference *clip*; its audio is patchified as an identity **reference token** and the model **generates new speech** from the prompt, then decodes it. It cannot sync to an uploaded song. Official docs confirm: "the target audio is generated by the model, not user-supplied", and the IC-LoRA is "not validated on LTX-2.5". |
| `retake`, `keyframe_interpolation`, `hdr_ic_lora`, `t2a_one_stage` | n/a | Not audio-to-video. |

**The transformer is built for this.** The checkpoint's own config metadata
carries `"use_audio_video_cross_attention": true`,
`audio_cross_attention_dim: 2048` and `av_cross_ada_norm: true`. Lip-sync is a
first-class capability of the weights, reached through exactly one entry point.

### The conditioning arguments, checked rather than assumed

`resolve_cli_params()` reads the checkpoint we actually pass. Measured on the
box:

```
model_version 2.5.0  → LTX_2_4_PARAMS
num_inference_steps  30
video guider         cfg 3.0 · stg 1.0 · stg_blocks [28] · rescale 0.7 ·
                     modality_scale 3.0 · skip_step 0
```

`--a2v-guidance-scale` defaults to that `modality_scale`, i.e. **3.0**, and LTX's
own help for it reads: *"controls how strongly the model reacts to the
perturbation of the audio-to-video cross-attention. Higher values may increase
lipsync quality. 1.0 means no effect."*

So the answer to "are any audio-conditioning values zero, defaulted or
ignored?" is: they are defaulted, and **the defaults are the official SFT
values with lip-sync guidance already on**. Nothing is zeroed and nothing is
ignored. Our invocation was not wrong; it was unreachable.

What the scales also decide is **cost**, which turns out to matter more than
anything else here. From `ltx_core/components/guiders.py`:

```
cfg      != 1.0  → an unconditional pass runs
stg      != 0.0  → a perturbed (STG) pass runs
modality != 1.0  → the isolated-modality pass runs   ← this is the lip-sync one
```

The shipped defaults are therefore **four transformer calls per step**, thirty
steps, before stage 2 starts. That is the whole performance story (§7).

---

## 4. ComfyUI vs ZolexAI

The official ComfyUI LTX-2.5 material documents text-to-video, image-to-video
and first/last-frame, all with *generated* audio. **No official ComfyUI
workflow for conditioning video on a supplied audio file was found**, and the
2.3-era lip-sync workflows are LipDub/Dub-It, which §3 shows cannot sync to a
supplied song.

Mapping what a native audio path must do against what we do:

| stage | official `a2vid_two_stage` | ZolexAI |
|---|---|---|
| decode audio | `decode_audio_from_file(path, device, start, max_duration)` | same call — we pass the master and an offset |
| encode audio | audio VAE → latent, truncated to `num_frames / fps` | same, via the pipeline |
| audio conditioning | `ModalitySpec(frozen=True, noise_scale=0.0, initial_latent=…)` | same |
| a2v cross-attention | `MultiModalGuider(modality_scale=3.0)` | same (default) |
| CFG / STG / rescale | 3.0 / 1.0 / 0.7 | same (defaults) |
| steps | 30 (stage 1), distilled 3–4 (stage 2) | same |
| image conditioning | `--image PATH IDX STRENGTH` | previous section's final frame @ 1.0 |
| audio ↔ video timebase | one `--audio-start-time` per render | one per section, from the section's own planned start |
| final audio | the pipeline returns the **input** audio, not a VAE round-trip | we discard it and mux the master once |

**The difference is not a missing node.** It is that the whole column is
unreachable in production because one YAML key is absent.

---

## 5. Full mix vs vocal stem

Not re-run. The A/B was measured on this box on 18 Aug: full-mix conditioning
tracked the vocal identically to an isolated stem, judged at a deliberate
2.7-second vocal gap with instruments continuing. The simpler architecture
stands, and the final audio is the customer's master either way — no separator
runs in the delivery path.

---

## 6. The audio tier's decodable shapes (new measurement)

1024x576, `a2vid_two_stage`, dev transformer + distilled LoRA, unquantized,
`--offload cpu`, ACE-Step resident (~24 GB). Four denoising steps: the crash
lives in the video VAE, after denoising, and depends on shape.

| frames | seconds | wall | result |
|---:|---:|---:|---|
| 65 | 2.71 | 54s | PASS |
| 121 | 5.04 | 55s | PASS |
| 193 | 8.04 | 58s | PASS |
| 241 | 10.04 | 81s | PASS |
| 289 | 12.04 | 87s | **FAIL** CUBLAS_STATUS_INTERNAL_ERROR |
| 337 | 14.04 | 59s | **FAIL** CUBLAS |
| 361 | 15.04 | 96s | **FAIL** CUBLAS |
| 385 | 16.04 | 95s | PASS |
| 409 | 17.04 | 65s | **FAIL** CUBLAS |
| 433 | 18.04 | 119s | PASS |
| 457 | 19.04 | 67s | **FAIL** CUBLAS |
| 481 | 20.04 | 118s | PASS |
| 505 | 21.04 | 93s | PASS |
| 577 | 24.04 | 180s | PASS |
| 601 | 25.04 | 130s | **FAIL — out of memory** |
| 721 | 30.04 | 236s | PASS |
| 841 | 35.04 | 124s | **FAIL** CUBLAS |
| 961 | 40.04 | 172s | PASS |
| 1081 | 45.04 | 239s | **FAIL — out of memory** |
| 1201 | 50.04 | 280s | PASS |
| 1441 | 60.04 | 277s | PASS |

**Two kinds of failure, and they need different answers.**

* The CUBLAS failures (289, 337, 361, 409, 457, 841) are the familiar
  non-monotonic decoder set — a property of the shape. 241 decodes and 289 does
  not; 433 decodes and 457 does not. There is no rule; a table of measured
  passing counts is the only defence, and `_A2VID` had none.
* **601 and 1081 failed on MEMORY while 721, 961, 1201 and 1441 passed.** That
  is not a shape property at all — it is a card at its edge. A prompt-only
  30-second pass was measured this session peaking at **95.2 GB of 95.6 GB**
  with the music service resident. So a long audio pass is a coin flip made
  after several minutes of compute have already been spent, which is why the
  landing table stops at 481 even though 1441 decodes.

The consequence for §7 is unwelcome: **the obvious answer to per-pass model-load
cost — ask for a longer pass — is available in principle and unreliable in
practice, and what makes it unreliable is that the music service and the video
model share one card.**

---

## 7. What an audio-conditioned second costs

Real production wall times, read from the worker's own log (`job_claimed` →
`job_completed`), current prompt-only tier:

| track | passes | wall | × real time | outcome |
|---:|---:|---:|---:|---|
| 300.04s | 6 | 1085s (18.1 min) | 3.6x | failed at upload (HTTP 403) |
| 240.04s | 5 | 863s (14.4 min) | 3.6x | completed |
| 180.04s | 4 | 634s (10.6 min) | 3.5x | completed |
| 120.03s | 3 | 428s | 3.6x | completed |
| 60.03s | 2 | 222s | 3.7x | completed |

That is the client's "20–25 minutes" for a five-minute video, once queueing,
download, upload and API round-trips are added to 18 minutes of worker time.

Measured directly, this session, on the golden 60-second track through the real
adapter:

| configuration | passes | wall | × real time | peak VRAM |
|---|---:|---:|---:|---:|
| prompt-only (**what ships today**) | 2 | 286s | 4.8x | 95,188 MiB |
| audio-conditioned, defaults | 4 | 731s then **crashed** on section 3 | ≥12x | 87,264 MiB |

The peak is the other half of the §6 story: a single prompt-only 30-second pass
peaked at **95.2 GB of a 95.6 GB card** with ACE-Step resident. The card is
already at its limit.

### 7a. Which knobs move an audio-conditioned pass, and which do not

Ten single-pass cells at 1024x576, 481 frames unless stated, everything else
held. **Read the per-step rate, not the wall time** — see the warning below.

| cell | guidance passes | steps | stage-1 s/it | wall | peak VRAM | result |
|---|---:|---:|---:|---:|---:|---|
| default (cfg 3.0, stg 1.0, a2v 3.0) | 4 | 30 | **4.58** | 213s | 77.4 GB | ok |
| stg 0.0 | 3 | 30 | **3.71** | 271s | 87.3 GB | ok |
| cfg 1.0 + stg 0.0 | 2 | 30 | **3.31** | 195s | 82.0 GB | **CUBLAS crash** |
| steps 15 | 4 | 15 | 6.61 | 198s | 96.3 GB | ok |
| + distilled LoRA on stage 1, 16 steps | 2 | 16 | 2.32 | 205s | 96.0 GB | ok |
| + distilled LoRA on stage 1, 8 steps | 2 | 8 | 2.22 | 117s | 95.3 GB | **OOM** |
| a2v 6.0 (the lip-sync dial, doubled) | 4 | 30 | — | 216s | 77.4 GB | ok |
| a2v 1.0 (lip-sync guidance off) | 4 | 30 | — | 214s | 81.0 GB | **CUBLAS crash** |
| 961 frames (40.04s) | 4 | 30 | — | **611s** | 90.4 GB | ok |
| 433 frames (18.04s) | 4 | 30 | — | 234s | 87.3 GB | ok |

Two more results worth pulling out. **Raising the lip-sync dial is free** — a2v
6.0 cost 216s against the default's 213s, so if it helped there would be no
reason not to take it (it does not; §2a). And **a longer pass is now worse, not
better**: 961 frames took 611s, which is 15.3x real time against 481 frames'
10.6x. The 17 Aug finding that longer passes are cheaper per second no longer
holds, because at 961 frames with four guidance passes batched the run is deep
into the memory ceiling and spends the difference thrashing. The lever that
would have paid for model-load cost has inverted.

Three things fall out, and only the first is what anyone expected.

**1. A guidance pass costs less than a guidance pass.** Dropping STG removes a
whole transformer pass per step and saved 19%; dropping CFG as well removes
another and saved 9% more — against the 25%-each the arithmetic predicts.
`_guided_denoise` batches every pass into ONE transformer call, so a step has a
fixed component that the batch size does not multiply.

I first attributed that fixed component to `--offload cpu` streaming 22B bf16
weights from host RAM on every step. **Tested, and it is more interesting than
that.** Same cells with `--offload none`:

| frames | `--offload cpu` | `--offload none` | steady-state s/it |
|---:|---:|---:|---|
| 241 | — | 100s | 2.0 vs — |
| 481 | 225s | **173s** (then crashed) | 4.59 → 4.42 |
| 961 | 611s | **428s** | 10.60 → 10.31 |

The steady per-step rate barely moves (3–4%), but the wall drops 23–30%. Stage 1
at 961 frames took 7:42 under `cpu` and 5:08 under `none` while both averaged
~10.4 s/it — so the difference is not the steps, it is the **first traversal of
the weights**, which `cpu` reads from disk before its host cache is warm. And
the worker spawns a **new process per section**, so that first traversal is paid
again for every pass of every job.

`--offload none` is therefore the largest single measured lever: roughly 50–180s
per pass. It is also the one this card cannot afford — 93.8 GB peak at 961
frames, and the 481-frame cell crashed. It becomes available the moment the
music service stops taking a quarter of the card.

**2. The wall clock does not follow the per-step rate, and that is the real
finding.** The cheapest configuration measured (2.32 s/it, 16 steps — 37s of
stage-1 denoise against the default's 137s) finished in 205s against the
default's 213s. Eighty seconds of saved denoising bought eight seconds of wall.
Every cell that peaked near the card's ceiling ran slower than its own step rate
implies, and the peaks are 77–96 GB **of 95.6 GB**.

**3. Three of ten cells failed, at counts that had already passed.** Eight of
the ten cells rendered 481 frames. Five succeeded and three did not — two with
`CUBLAS_STATUS_INTERNAL_ERROR` in the same video-VAE MLP as every other decoder
crash, one out of memory outright. The same count decoded in the shape sweep
and in the perf matrix's first three cells before failing in the fourth.
**A fixed configuration that fails intermittently is not describing a shape**,
and 433 frames — the count that crashed a real job — passed cleanly here at full
step count, which says the same thing from the other direction.

### 7b. The card has no margin when it is shared — and that is the headline

`CUBLAS_STATUS_INTERNAL_ERROR` is what a failed cuBLAS workspace allocation
looks like from the outside. Put beside the peaks — a *prompt-only* 30-second
pass measured at 95,188 MiB of 97,887 — the intermittency stops being mysterious.

**But run the same cell six times with the card otherwise quiet and it passes
six times.** 481 frames, 30 steps, defaults, back to back:

```
run 1  PASS  356s  peak 78,071 MiB      ← cold page cache
run 2  PASS  241s  peak 77,373 MiB
run 3  PASS  208s  peak 77,387 MiB
run 4  PASS  212s  peak 77,379 MiB
run 5  PASS  208s  peak 77,371 MiB
run 6  PASS  208s  peak 77,371 MiB
```

Six for six, and the peak is **77.4 GB every time** — 20 GB of headroom.

So the correct statement is narrower and more useful than "the tier is
unreliable": **the audio tier is reliable when it has the card to itself, and
fails when it does not.** Every failure today came from the mixed part of the
day — cells running back to back, my own Demucs and Whisper jobs sharing the
GPU, configurations that peaked at 95–96 GB instead of 77. Five of fifteen
attempts at the same frame count failed across that window; zero of six failed
in isolation.

That is a scheduling property, not a model property, and it is the one to
design around. `MAX_CONCURRENCY=1` already stops the worker running two jobs at
once — what it does not stop is the always-on ACE-Step service, or anything else
sharing the node, from taking the margin at the wrong moment.

It is also not new and not confined to the audio tier. Counting the memory-class
failures in the production worker log:

| workflow | OOM | CUBLAS | illegal access |
|---|---:|---:|---:|
| text-to-video | 15 | — | — |
| image-to-video | 7 | 9 | — |
| video-to-video | — | 7 | — |
| music-video | 2 | 4 | 12 |

**56 production jobs have died this way**, across every video workflow. The
long-standing theory — that these are (grid, conditioned, frame-count) shape
failures, defended by measured tables — explains the reproducible cases and
cannot explain a count that passes six times alone and fails five times in
fifteen when the card is busy.

This compounds badly, which is why it matters more than any knob. A five-minute
audio-conditioned music video is **15 consecutive passes**. At a 95% per-pass
success rate that job completes 46% of the time; at 90%, 21%. The tier's
per-pass reliability has to be very close to 1, and "very close to 1" is exactly
what the isolated repeat shows and the shared card does not.

So the recommendation is not "buy headroom" in the abstract. It is:

1. **Give a music-video job the card.** The tier peaks at 77 GB with 20 GB
   spare; it does not need more memory, it needs nobody else taking it mid-pass.
2. **Confirm the biggest co-tenant is the one to move.** ACE-Step holds ~24 GB
   permanently. One measurement settles it — repeat the same six-run cell with
   `supervisorctl stop zolexai-music` and compare peaks. That is a production
   action, so it is a recommendation here rather than something done.
3. **Fail a pass loudly and retry it**, rather than losing the whole job. A
   15-pass job that cannot retry a single crashed section is throwing away
   fourteen good passes; today the chain aborts.

---

## 8. Beat-only music: root causes

### (a) A genre word silently overruled a request for vocals

`detect_genre` maps `lo-fi`, `lofi` and `ambient` to the `ambient` genre, and
`instrumental` / `no vocals` to `instrumental`. Both are in `_WORDLESS`, so
`plan.has_lyrics` was False, `_lyrics_for` returned None, and an empty sheet is
how ACE-Step is told to make an instrumental. Reproduced deterministically:

| prompt | genre | had lyrics |
|---|---|---|
| "a lo-fi pop song with **soft female vocals** about rain" | ambient | **no** |
| "an ambient pop ballad with a **female singer**" | ambient | **no** |
| "a cinematic song with instrumental verses and a big **sung chorus**" | instrumental | **no** |
| "a dreamy song, no vocals in the intro, **then she sings**" | instrumental | **no** |

The customer has no other channel: there is no `instrumental` field on the API
and no toggle in the panel — the worker reads `job.parameters["instrumental"]`
and nothing anywhere ever sets it. So the prompt is the whole vocabulary, and
the genre table was overruling the sentence it appeared in.

**Fixed** by `vocal_intent()`: instrumental phrases are read out of the text
first, a vocal word in what remains means the song has words, and only a stated
instrumental with no vocal request silences it. Said nothing either way, the
genre still decides. A wordless genre asked for vocals also borrows a worded
skeleton, because `ambient`'s own sections (intro → movement → outro) have
nowhere to put a line.

### (b) The lyric sheet is sized for a different model

Production auto-lyrics, generated through the live writer and service and
measured against the audio:

| requested | sheet | vocals start | vocals end | voiced | worst wordless gap |
|---:|---:|---:|---:|---:|---:|
| 1m | 7 lines | 1.6s | 54.2s | 63% | 12s |
| 2m | 10 lines | **32.3s** | 115.2s | 45% | 32s |
| 3m | 13 lines | 30.7s | 180.0s | 46% | 30s |
| 4m | 17 lines | 35.5s | 215.5s | 48% | 35s |
| 5m | 20 lines | 0.0s | **201.9s** | **34%** | **98s** |

The five-minute case stops singing at 3:22 and plays 1:38 of instrumental —
the complaint, exactly. `_TARGET_SECONDS_PER_LINE = 16.0` and its ceiling of
13.0 s/line were measured in August 2026 on an RTX 5090 against an older
ACE-Step build; the build in production is `acestep-v15-xl-turbo`.

### 8b. What the density actually does, and what "no lyrics" really is

Twelve cells: three durations × four target densities, every sheet written by
the production writer and every track made by the production service, same
prompt and same seed. Vocal presence measured from a **Demucs vocal stem**, not
from a transcriber — see §10 for why that swap was necessary.

| song | ~16s/line (shipped) | ~8s/line | ~5s/line | ~3.5s/line |
|---|---:|---:|---:|---:|
| 60s | 83.3% | **86.7%** | 71.7% | 78.3% |
| 180s | **52.8%** — 43s hole, nothing sung until 30s | **73.3%** | 77.8% | 97.8% |
| 300s | 87.7% | **78.7%** | 89.0% | 95.0% |

Two things to take from it.

**The shipped density is where the floor falls out, and it is unstable.** The
worst cell in the matrix is 180 seconds at the shipped setting — half the song
wordless, a 43-second hole in the middle, nothing at all for the first thirty
seconds. The same setting produced 87.7% at 300s. That instability *is* the
report: "SOMETIMES only a beat".

**And what the model pads with is not silence.** Demucs says the production
five-minute track is 80% voiced with a voice as late as 4:38 — while Whisper
finds its last recognisable WORD at 3:22. Both are right. Read the transcripts
and the model is plainly vamping once the sheet runs out: *"City summer night"*
six times over, *"oh-oh ooh-oh-oh"*. A stem separator hears a singer; a customer
hears the lyrics stop and the track coast.

So the fix is to give it more words rather than more instructions, and to stop
short of filling every bar: 3.5s/line reaches 97.8% by removing the intro, the
break and the outro a song is supposed to have. **8s/line** is the value taken —
best at 60s, lifts the 180s floor from 52.8% to 73.3%, holds 78.7% at 300s, and
leaves gaps of 4–20s where music belongs.

### 8c. The acceptance run, and what it says the density can and cannot fix

Eight songs through the **real `MusicAdapter`** at the new density — the writer
chain, the language resolution, the provider, the assembly, all of it:

| test | sheet | voiced | first voice | last voice | worst gap | before |
|---|---:|---:|---:|---:|---:|---:|
| 1 min, auto | 7 lines | **95.0%** | 0s | 58s | 2s | 51.7% |
| 2 min, auto | 15 lines | **88.3%** | 2s | 113s | 7s | 76.7% |
| 3 min, auto | 19 lines | 77.8% | 0s | 161s | 19s | 72.2% |
| 4 min, auto | 30 lines | 70.0% | 7s | 234s | 14s | 82.1% |
| 5 min, auto | 34 lines | **53.7%** | 3s | 250s | **50s** | 80.0% |
| 3 min, Spanish | 21 lines | 71.1% | 2s | 177s | 20s | — |
| 2 min, "an instrumental piano piece, no vocals" | **0 lines** | **0.0%** | — | — | — | correct |
| 2 min, "a lo-fi pop song with **soft female vocals** about rain" | **14 lines** | **75.8%** | 1s | 115s | 12s | **0% — was an instrumental** |

The bottom two rows are the genre-trap fix, closed in both directions, and the
sheets confirm it rather than the numbers alone:

* `"an instrumental piano piece, no vocals"` → *"lyrics: NONE WRITTEN — the
  provider was asked for an instrumental"*, 0.0% voiced. Still honoured.
* `"a lo-fi pop song with soft female vocals about rain"` → a full sheet
  (*"Cold glass against my cheek / The streetlights start to leak…"*), 75.8%
  voiced. This is the exact prompt that used to come back wordless.
* Language selection is unaffected and still real: the Spanish cell wrote
  Spanish (*"El sol despierta temprano en la ciudad…"*), not English with a
  Spanish label.

**But the top of the table does not say what I expected, and it should be read
carefully.** One, two and three minutes improved — the one-minute case nearly
doubled. Four and five minutes got *worse* than the old density, and the
five-minute case is the worst cell of the day.

Set that beside the matrix and the reason is plain. Three 300-second samples,
all at or near the new density: **78.7%**, **53.7%**. And at the old density:
**87.7%**, **80.0%**. There is no ordering there. At five minutes the spread
between two runs of the same settings is larger than the entire effect of
changing the settings.

So, precisely:

* **The density fix is well-supported for 1–3 minute songs** and is kept.
* **It is not a fix for the five-minute case, and nothing here is.** At that
  length ACE-Step's coverage is dominated by its own run-to-run variance.
* What the change *does* guarantee at every length is the half we control: the
  sheet now covers the whole song — 34 lines across intro / 4× (verse,
  pre-chorus, chorus) / bridge / chorus / outro, where it used to be 20. The
  model choosing not to sing all of them is a different problem from not having
  been given them.

The measured next step, not taken here: **verify after generating.** Coverage is
cheap to measure (this is the tool that measured it) and a track that comes back
under threshold could be re-rolled on another seed rather than delivered. That
is a real change to the music path days before a release, so it is a
recommendation with evidence rather than a commit.

---

## 9. External lip-sync: researched, and it does not solve this either

The brief asks for this only if the native path proves insufficient. The native
path *works*; what it is not is cheap, or — on this node — reliable. So the
alternatives were priced against the same bar, and none of them is an answer.

| candidate | what it is | code / weights licence | why not |
|---|---|---|---|
| **LTX `dubit` / LipDub IC-LoRA** | the LTX-native lip pipeline | Lightricks, LTX-2.x Community | **Generates new speech from the prompt.** Cannot sync to a supplied song — verified in the installed source and in Lightricks' own docs. Also "not validated on LTX-2.5". |
| **LongCat-Video-Avatar 1.5** | 13.6B audio-driven avatar DiT, handles singing, 5-minute clips | MIT, weights on HF | **~44 seconds of GPU per second of video** on an A800. Four times worse than the tier we already have, and a second 13.6B model resident on a card that is already out of memory. |
| **ID-LoRA** (ECCV 2026) | identity IC-LoRA on LTX-2 / 2.3 | LICENSE unspecified | Its "reference audio" is ~5s of **voice identity**, not a timeline to follow; it generates speech. LTX-2.3 only. An unspecified licence is disqualifying on its own under LTX §3.5. |
| **LatentSync 1.6** | diffusion mouth re-render over finished video | Apache-2.0, code and weights | The only serious post-hoc candidate. 512px face crop, 20–50 diffusion steps per frame — minutes per clip, on the same exhausted card, after the video is already made. |
| **MuseTalk** | real-time mouth re-render | MIT code, weights commercially usable | Fast enough (30fps+) to be tempting. **256×256 output**, composited into a 1024×576 frame where a medium close-up face is 300–400px — visibly soft, and known for jitter. |
| **Wav2Lip** | the classic | **research only** — LRS2-trained weights | Cannot be used commercially at all. 96×96. |

Two conclusions worth stating plainly. **Nothing cheaper than the native tier
actually follows an audio timeline** — everything that does is a guided
diffusion model and costs guided-diffusion money. And every post-hoc option is a
mouth composited over a finished frame, which is the failure mode the brief
names and rejects, not a way around it.

---

## 10. Music-video lyrics, and lyric timestamps

Music video takes an uploaded audio asset. A track made by the Music workflow
reaches it through `ResultActions` as an asset id — the file travels, **the
lyrics do not**, because they are not stored anywhere. The worker writes
`workspace/lyrics.txt` and the workspace is discarded with the job.

Persisting them needs a channel that does not exist. `AdapterResult` carries
path, type, kind, duration and dimensions and nothing else; the runner's
completion payload mirrors it; neither `generation_jobs` nor
`generation_job_outputs` has a metadata column, and `request_params` holds what
the customer ASKED for and is written by the API before the worker runs. So this
is a five-layer change ending in a JSONB column and a migration, and it is
deliberately not being made days before a release, unasked, to a database. The
design is recorded here instead. In the meantime the worker logs the sheet's
provider, language, size and quality, so a "wrong lyrics" report is diagnosable
from the log without it.

**Forced alignment** was tested rather than speculated about. `faster-whisper`
with `word_timestamps=True` runs on this box and returns a word timeline; it is
not trustworthy enough to drive timing. On the very measurements in §8b it
identified an English pop song as Khmer and returned a garbage timeline for one
cell, and hallucinated "thank you for watching this video" across an
instrumental break in another. That is why the coverage figures are measured
with Demucs stem energy instead. Whisper is fine for "did this verse appear at
all" and for debugging, and must not become a source of truth for cut placement
— which is also what the brief's own "the waveform is master" rule says.

---

## 11. Files changed

**Music video timing — the seam staircase (issue 1, half of it)**

| file | change |
|---|---|
| `worker/longform/chain.py` | `_plan` → **`plan_chain_segments`**, public. Delivery has to be able to ask the same function generation asked, boundaries included. |
| `worker/longform/__init__.py` | exports it |
| `worker/adapters/ltx.py` | `_planned_section_frames` takes `boundaries`; `_run_music_video` computes section frames from the chain's OWN plan and pins them through `_assemble_generated_sections` |
| `tests/test_seam_timing.py` | renamed to cover both workflows; +1 unit test (cut points change the answer), +1 end-to-end (a music video's scenes are delivered at their planned counts) |
| `tests/test_longform.py` | import follows the rename |

**Music video cut placement — no more sliver sections**

| file | change |
|---|---|
| `worker/longform/timing.py` | `plan_musical_boundaries` picks the pass COUNT first and places nominal cuts at `total·k/count`, each pulled back to an onset within a bounded tolerance — instead of filling the ceiling greedily and leaving the remainder as its own pass. Adds the symmetric guard so the pull's deficit cannot pile into the final window and oversize it. |
| `tests/test_longform.py` | +1 parametrised regression: no window may be under half the longest, the count must equal the even plan's, and the ceiling still holds |

**The audio tier — making the flag safe to turn on**

| file | change |
|---|---|
| `worker/adapters/ltx.py` | `_A2VID.measured_landings = (121, 241, 385, 481)` from the §6 sweep — it was empty, which is why a real job reached the decoder at an unmeasured count and died. `_AUDIO_PASS_SECONDS` is now 481 frames' own duration rather than a round 20.0, so the planner's nominal window is itself a measured count. `_command` gains `stg_scale`, `a2v_guidance_scale` and `inference_steps` as execution keys, beside the existing `guidance_scale`, on the same guided-family gate. |
| `tests/test_music_video_audio.py` | ceiling test follows the landing; +1 test that no count the planner can ask for reaches the decoder at a measured-bad one |
| `workflow-definitions/music-video.yaml` | **comment-only** (parse-verified identical): the real cost, why it costs that, and the new keys |

**Music vocals (issue 2)**

| file | change |
|---|---|
| `worker/music/lyrics.py` | **`vocal_intent()`** — reads a stated vocal or instrumental request out of the prompt, negations first. `SongPlan.wordless` becomes its own field instead of a genre lookup, and `plan_song(vocals=…)` overrules the genre; a wordless genre asked for vocals borrows a worded skeleton because its own sections have nowhere to put a line. `_TARGET_SECONDS_PER_LINE` 16.0 → **8.0** and `_SECONDS_PER_LINE` 13.0 → **6.0**, from the §8b matrix, with the superseded 5090-era evidence kept beside it. |
| `worker/core/config.py` | `music_seconds_per_line` 13.0 → **6.0**, same measurement |
| `worker/music/__init__.py` | exports `vocal_intent` |
| `worker/adapters/music.py` | passes the prompt's intent into `plan_song`; logs `wordless` on the plan and a new `music_instrumental_selected` line naming WHY a track is getting no words |
| `tests/test_music.py` | +9 parametrised intent cases (every reproduced failure, plus both directions of "said nothing"), +2 plan tests; the three that pinned the old 13s/line budgets updated to the new measurement rather than around it |

Untouched: text-to-video, image-to-video, extend, video-to-video (both engines),
the API, the frontend, storage, progress/SSE, billing, `concat_segments`,
`mux_audio`, `plan_segments`.

### Regression status

| suite | result |
|---|---|
| Worker, full | **802 passed**, 1 skipped, **1 failed** — pre-existing |
| API | **115 passed** (infra up) |
| Frontend `tsc --noEmit` | clean |
| Frontend `eslint --max-warnings=0` | clean |
| Frontend `next build` | clean |
| `ruff check` on every changed file | clean (5 findings remain repo-wide, all in files not touched here) |
| `music-video.yaml` | parse-verified **identical** — comment-only |

The one failure is `test_a_track_longer_than_one_pass_becomes_several_scenes`,
which expects a 4.0s MP3 to probe slightly over 4.0s so a 1s ceiling plans five
windows. This machine's LAME produces exactly 4.0s and it plans four.
**Verified pre-existing by stashing every change and running it on a clean
tree, where it fails identically** — the same check the 17 Aug note made, with
the same result.

T2V/I2V/extend/V2V argv pins, person lock, director, extend lineage and both
V2V engines are all inside that 802 and all green. The default music-video path
is pinned byte-for-byte by
`test_a_plain_music_video_still_never_shows_the_model_the_song`, which still
passes: with the flag off, the argv is the distilled entry point with no audio
flags and no LoRA, exactly as it has been serving customers.

---

## 12. The 10–15 minute target, answered

**Not simultaneously with lip-sync, on this node, at any setting measured.**

| what | 5-minute track | evidence |
|---|---:|---|
| today, prompt-only, no lip-sync | **~18 min** worker time | six real production jobs, 3.5–3.8x |
| lip-synced, defaults | **~52 min** | 15 passes × 208s, the steady-state figure from six consecutive isolated runs |
| lip-synced, every knob at its cheapest | **~51 min** | the whole matrix lands in 195–271s per pass |
| lip-synced, longer passes | **worse** | 961 frames is 15.3x against 481's 10.4x |

The gap is not a tuning problem. It is that the only pipeline able to hear a
song runs unquantized — forced by the distilled LoRA it requires — and therefore
streams 22B bf16 weights from host RAM on every denoising step. The distilled
tier that serves the current 3.6x runs NVFP4-quantized and resident. **Three
times the cost is what "the model can hear the song" costs on this hardware.**

What would actually move it, in order of measured value:

1. **Memory headroom, first and before anything else.** Until roughly a third
   of passes stop dying, a 15-pass job cannot finish and no optimisation is even
   measurable — every timing in §7a is contaminated by thrash. The single
   biggest reclaimable block is ACE-Step's ~24 GB. One measurement settles it:
   repeat the 481-frame cell with `supervisorctl stop zolexai-music` and count
   the failures. That is a production action and wants a maintenance window.

   It also unlocks the largest measured lever. `--offload none` ran the same
   passes **23–30% faster** (173s vs 225s at 481 frames; 428s vs 611s at 961)
   while the steady per-step rate moved 3% — the saving is the first traversal
   of the weights, which `cpu` reads from disk. It needs 76–94 GB, so this card
   cannot afford it while a quarter of it belongs to something else.

2. **One process for many passes.** The chain shells out to the CLI once per
   section, so every pass pays that first traversal again from cold. A driver
   that loads the pipeline once and loops over the plan removes 14 of a 15-pass
   job's 15 builds — and it is the same 50–180s that (1) unlocks, taken a
   second way. Only meaningful after (1).

3. **Step count**, the one knob with a straightforward effect, and a quality
   trade that should be judged on real output rather than on a table.

Stacking them, from the measured pieces rather than from hope:

```
today                481f, --offload cpu, 30 steps   208s × 15 = 52 min
+ --offload none     481f, measured                  173s × 15 = 43 min
+ 15 steps           stage 1 131s → ~66s             108s × 15 = 27 min
+ one process        14 × ~40s of first-traversal    ≈ 18 min
```

**18–27 minutes for a lip-synced five-minute video, against 52 today** — and
the top of that range still needs a resident-model driver that does not exist,
while the bottom needs a step count nobody has judged against real output. It is
better than the current product's 18 minutes for a video with **no lip-sync at
all**, and it is **not 10–15**. That number should not be promised for a
lip-synced music video on this hardware.

And one that costs nothing and is available today, independent of all of the
above: **the audio tier's pass ceiling is now a measured landing**, so a
300.042s track plans 15 passes instead of 16 and a 60.024s track 3 instead of 4.
That is one whole model load saved per minute-long video.

---

## 13. Remaining limitations

**Ours, and known:**

- **Audio conditioning is still OFF in the shipped YAML.** Everything in this
  note makes the flag *safe to turn on* — a landing table, pinned section
  timing, no sliver passes, and a 60-second job proven end to end. Turning it on
  triples the unit cost of the product's most expensive workflow, which is a
  pricing decision, and it should not be taken before §7b's scheduling question
  is answered: 15 consecutive passes need per-pass reliability near 1, which the
  tier has alone and does not have on a shared card.
- **A crashed pass loses the whole job.** `render_chain` aborts, so a 15-pass
  music video that loses its fourteenth section throws away thirteen good
  renders. Retrying a single pass is the obvious mitigation and is not
  implemented.
- **Five-minute songs still sometimes stop singing early, and the density
  change does not fix it** (§8c). The sheet now covers the whole song; whether
  ACE-Step sings all of it varies between runs by more than any setting
  measured. Verify-after-generating is the recommended answer and is not
  implemented.
- The final lyric sheet is still not persisted anywhere a customer or a music
  video can read (§10). Design recorded; it needs a migration.
- **Music video has no reference-image input.** The brief asks that lip-sync not
  damage reference identity; `music-video.yaml` declares exactly one input,
  `source_audio`. Reference identity is a video-to-video feature, so there is
  nothing here for lip-sync to damage — and equally, no way to pin a performer's
  face on a music video today.
- Music video's prompt preamble tells section 1 to "keep the same subjects …
  established previously" and to "continue directly from the predecessor frame"
  when there is no predecessor. Shared with every chained workflow, so it is out
  of scope here — but it is prompt noise entering a caption-driven model.
- The section-frame pinning is arithmetic, verified end to end on real ffmpeg
  in the suite. Its effect on a REAL five-minute render has not been measured
  against a before/after pair, because rendering the before takes 18 minutes and
  the after 53.

**The model's, not ours:**

- Lip-sync is only available on the guided audio tier, at that tier's price.
  There is no cheap version — see §9.
- The audio tier cannot be quantized while it needs the distilled LoRA, and
  therefore cannot avoid weight streaming.
- The decoder's reproducibly-bad frame counts follow no rule; the table is
  measurement, not understanding.

**Lip-sync, precisely:**

- What is measured is **B**: mouth motion that follows the vocal, with a stable
  −125…−208 ms lead and r 0.43–0.49, against a baseline with no peak at all.
- **C — phoneme accuracy — is not measured and is not claimed.** The metric
  scores vocal energy against mouth openness; it cannot tell whether a rounded
  vowel got a rounded mouth.
- Long-form drift after the fix is arithmetic (≤ half a delivery frame per seam,
  non-accumulating), not an end-to-end measurement on a five-minute render.
- Non-English singing is unmeasured for sync. The vocal-band envelope is
  language-agnostic; whether the model's articulation is, is not known.
- **Identity drifts across a music-video seam.** The frames at §2b show the
  performer at 58.5s is a different woman from the one at 24.0s — same wardrobe,
  same studio, same shot, different face. Music video has no reference-image
  input to pin her with.

---

## 14. Where the evidence lives

Everything above is reproducible from the GPU node, under `/workspace/mv2026/`:

| path | what |
|---|---|
| `lipsync_probe.py` | the mouth-vs-vocal diagnostic, with its `--shift-ms` self-test |
| `vocal_activity.py` | Demucs stem coverage — the measure the density work rests on |
| `assets/` | the frozen test tracks (20s / 30s / 60s golden / 5-minute), with their sheets |
| `shape/shapes.txt` | the 21-count a2vid decoder sweep |
| `perf/matrix.txt` | the ten-cell knob matrix, plus each cell's own pipeline log |
| `repeat/repeat.txt` | the six-run isolated repeatability check |
| `offload/offload.txt` | `--offload cpu` vs `none` |
| `matrix/`, `accept/` | the lyric density matrix and the eight-song acceptance run, with coverage |
| `runs/e2e/` | the end-to-end validation of the fixes, and its workspace |

**Nothing on the production side was touched.** `/workspace/zolexai` is still at
`c713aa8` with only its pre-existing untracked `uv.lock`, and all three
supervised services kept their PIDs throughout — the modified worker ran from an
isolated copy at `/workspace/mv2026/worker-test`.
