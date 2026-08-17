# V2V, Music Video and lip-sync: research, architecture, implementation

**Date:** 2026-08-17 · **Status:** implemented behind two opt-in flags; both
paths GPU-verified on the RTX PRO 6000. Nothing existing changed behaviour.

---

## 1. What the client's shared repo actually provides

`E:\Downloads\ltx-main` is **not** the upstream Lightricks repository. It is the
client's own product: 24 files, ~21k lines — a Node backend, a React studio, and
a 7.4k-line Python engine (`ltx23_service.py`) that shells out to **LTX-2.3**.
We run **LTX-2.5**. The CLI surfaces differ, but the mechanics transfer, and the
engine is the single most useful document in this investigation because it is a
*working* productisation of exactly the three features being asked for.

What it does that we did not:

| Their endpoint | Mechanism | Our equivalent before today |
|---|---|---|
| `/video-to-video` | `--video-conditioning PATH STRENGTH` (real temporal conditioning) + a LoRA per mode | `--image` stills only |
| `/music-video` | `a2vid` segments, each given the **master** track at `--audio-start-time` | no audio conditioning at all |
| `/audio-to-video` | same, single segment | — |

Three details from their engine that turned out to be load-bearing:

1. **`_snap_frames`: "(frames - 1) % 8 == 0"** — our hard-won 8k+1 discovery is
   their stated invariant. Audio paths use a **ceil** snap because snapping down
   truncates the conditioning audio.
2. **`_effective_quantization` forces quantization to `none` whenever a LoRA is
   loaded** ("LoRA+FP8 fusion can use unsupported Triton fp8e4nv kernels") and
   fits the unquantized model with `--offload cpu`. This is the single most
   valuable line in their codebase for us — see §6.
3. **Their music video is our lip-sync prototype, productised**: per-segment
   seed = seed + index, first segment takes the user image at ≤1.0, every later
   segment pins the previous segment's final frame at 1.0, container padding
   absorbed at concat rather than by rendering an extra segment.

---

## 2. The three client YouTube videos

All three resolve to LTX-2.5 overview/ComfyUI content. **None demonstrates
music-video generation, audio-to-video or lip-sync.**

| ID | Title | Channel | Relevance |
|---|---|---|---|
| `XAhuGRCI2tM` | LTX-2.5 Open Weights Running In ComfyUI With Multi-Shot & Diffusion Fidelity Rendering | Benji's AI Playground | **Real.** Multi-shot and DFR |
| `pgWSDVSgon0` | LTX-2.5 (w4a8 + GGUFs) \| LOW VRAM Workflow | REBEL AI | None — we have 96 GB |
| `d0VFDNO45_U` | LTX 2.5 Is INSANE! The New AI Video Model Is Finally Here | Vantage with AI | None — launch overview |

**Reusable:** "multi-shot" is a *prompting* technique, not a flag — one
chronological paragraph with named transitions, 2–4 cuts in a single generation,
holding character/environment/lighting across cuts. It is directly applicable to
`plan_section_prompts` and to the long-form identity requirement. `dfr_pipeline`
(Diffusion Fidelity Rendering) is installed and unused; it is a quality lever
for later, not a V2V or lip-sync answer.

**Not reusable:** the quantization/GGUF material is for VRAM-constrained cards.

---

## 3. Existing V2V architecture (before)

```
upload → probe_media → target = source.duration_seconds   ← automatic already
       → grid_for_source (source's own aspect)
       → render_chain(per_pass = grid ceiling)
           each pass: extract_frames_at(source window) → --image PATH IDX 0.45
                      + previous final frame at 0.85 (frame 0)
       → normalize → concat → mux source audio ONCE
       → verify duration/dimensions
```

Answering the 16 audit questions directly: duration is read from the probe (1,2)
and a supplied duration is rejected by the API (3); frames come from
`extract_frames_at` at half-step offsets across the pass's own window (4,5);
**only stills are used — no video conditioning ever reached LTX** (6,7);
structure strength 0.45, continuity 0.85, reference 0.3 (8); the prompt competes
with the stills (9); source audio is muxed once at the end (10,11); segmentation
is by measured grid ceiling (12,13); FPS/timebase normalized before concat (14);
duration is verified against the source (15); cuts survive only insofar as the
stills capture them (16).

**Root cause of "restyling too weak", confirmed not guessed:** a photograph of
the source carries its colour, light and material *along with* its geometry. At
strength 0.45 across 3–16 stills per pass, the prompt spends itself fighting the
source's look. No strength value fixes this — lowering it loses the structure,
raising it loses the restyle. The signal is wrong, not the dial.

---

## 4. Existing Music Video architecture (before)

```
upload → probe_media → target = track.duration_seconds    ← automatic already
       → audio_onsets(track) → plan_musical_boundaries    ← cut points only
       → render_chain: prompt-only passes, previous final frame at 1.0
       → normalize (audio=False) → concat
       → mux_audio(picture, ORIGINAL track) ONCE
       → verify: has audio, length == song
```

**How the music influenced the visuals: duration and cut placement. Nothing
else.** Not beat, amplitude, section, lyric, vocal, tempo or embedding. The
model never received the audio in any form.

That is a complete answer to the lip-sync question for the old pipeline: a
singer in that output cannot move their mouth in time with a vocal, because the
vocal was not an input. It is structural, not a prompting failure.

---

## 5. Existing lip-sync capability (before)

None. Nothing in the worker performed, requested or approximated audio-visual
synchronisation of a performer.

---

## 6. Root problems, confirmed

1. **V2V**: still-frame conditioning is the wrong signal for restyling. (§3)
2. **Music video**: the model is never shown the song. (§4)
3. **Both**: the distilled tier has no CFG, no negative prompt and no step
   count, so prompt adherence cannot be pushed at inference.
4. **A latent trap that had been misdiagnosed for days**: every prior
   audio-tier "resolution ceiling" was the **LoRA + quantization clash**, not a
   limit of the card. With a LoRA loaded, quantization must be dropped entirely
   and the unquantized model fitted with `--offload cpu`. 1024x576 — which had
   never once completed quantized — renders clean.

---

## 7. Chosen architecture

Both fixes are **native LTX-2.5 pipelines already installed on the box**. No new
model, no new licence, no third-party lip-sync stage.

### V2V — `execution.v2v_engine: transform`

```
source window → ffmpeg edgedetect → control clip @ pass grid/fps/FRAME COUNT
              → ltx_pipelines.ic_lora
                  --lora  <Union Control 2.3>  1.0
                  --video-conditioning <control.mp4> 1.0
                  --image <previous final frame> 0 0.85     ← seam only
                  (no --quantization, --offload cpu)
              → everything else unchanged: concat, source audio once, verify
```

Why edges: they carry exactly what a restyle must preserve (subject outline,
placement, camera geometry, scene layout) and none of what it must discard
(colour, light, material, time of day). Canny needs no model and no additional
licence — ffmpeg is already a hard requirement. Depth and pose are strictly
better for some jobs, need DepthCrafter/DWPose, and sit behind the same seam
later.

### Music Video — `execution.audio_conditioning: true`

```
master track (never sliced) → ltx_pipelines.a2vid_two_stage per pass
        --audio-path <MASTER> --audio-start-time <section start>
        --audio-max-duration <frames/fps + 1 latent>
        --image <previous final frame> 0 1.0
        (dev transformer + --distilled-lora, no --quantization, --offload cpu)
   → parts assembled SILENT → mux_audio(original master) ONCE  ← unchanged
```

The pipeline seeks, so the master is never cut, re-encoded or re-timed: the
audio the model hears at 2:31 is bit-for-bit the audio the finished video plays
at 2:31. As a side effect this is the **guided** tier, so it also has CFG and a
negative prompt available — the adherence lever the distilled tier lacks.

### Lip-sync: native, and honestly labelled

Separating the client's own A/B/C:

- **A — visuals change with the music**: already delivered (onset-aligned cuts),
  now genuinely audio-driven as well.
- **B — the performer's mouth follows the vocal**: delivered by a2vid's
  audio-video cross-attention. This is what the model is built for.
- **C — verified phoneme-accurate lip-sync**: **not claimed.** It needs a
  sync-offset measurement (SyncNet-class) against real output, which has not
  been run. The product must not say "lip-sync" until it has.

---

## 8. Alternatives rejected

| Option | Why not |
|---|---|
| **`ltx_pipelines.dubit`** (the native "Dub-It" pipeline) | Looks like the lip-sync answer and is not. It has **no `--audio-path`** — the reference clip supplies identity and the model *generates new speech from the prompt*. It cannot sync to an uploaded song. Its LoRA (`LTX-2.3-22b-IC-LoRA-DubIt`) is also not on the box. |
| **A bolt-on lip-sync model** (LatentSync / MuseTalk / Wav2Lip class) | A face-only overlay pass after generation: new licence review, new VRAM resident alongside a 22B transformer, identity/artifact risk at the mouth boundary, and drift to manage over a 4-minute song. Native audio conditioning exists, is installed and is proven to run — adding a third-party model before using it would be backwards. Revisit only if measurement shows B is insufficient. |
| **Raw RGB as `--video-conditioning`** | Accepted by the CLI, but Union Control expects an aligned *control* signal; RGB degrades to a weak style hint. The client's engine states this explicitly. |
| **Tuning the existing restyle's strengths** | The signal is wrong, not the dial (§3). Swept already; it trades drift against style and cannot swap a look. |
| **Padding the grid to a multiple of 128** to satisfy stage 2 | Changes the customer's aspect ratio. Rendering at 2x and stopping at stage 1 gives the exact target grid instead. |
| **Slicing the master per section** | Re-encodes the track N times and puts a codec boundary at every seam of the thing the visuals synchronise to. `--audio-start-time` makes it unnecessary. |
| **Making either path the default now** | Both change cost and behaviour. The transform engine is a different product; audio conditioning is ~4x the compute. Both are one YAML line from being on. |

---

## 9. Files changed

**New**
- `apps/worker/worker/media/control.py` — edge-map control-signal extractor.
- `apps/worker/tests/test_transform.py` — 13 tests.
- `apps/worker/tests/test_music_video_audio.py` — 11 tests.

**Modified**
- `apps/worker/worker/adapters/ltx.py` — `LtxPipeline` descriptors
  (`_DISTILLED`/`_IC_LORA`/`_A2VID`), `ControlConditioning`, `AudioConditioning`,
  `LoraSpec`, `_optional_weight`, `_audio_pass_seconds`, `_audio_window_seconds`,
  `_run_transform`, `_deliver_restyle`, audio-conditioned music video,
  `_command`/`_renderer`/`_launcher` extended.
- `apps/worker/worker/media/probe.py` — `frame_count`, `audio_sample_rate`,
  `audio_channels`.
- `apps/worker/worker/media/__init__.py` — exports.
- `apps/worker/tests/conftest.py`, `tests/test_ltx.py` — launcher stub takes the
  module argument; fake weights include the optional tiers.
- `apps/worker/tests/test_media.py` — probe + control-signal tests.
- `workflow-definitions/video-to-video.yaml`, `music-video.yaml` — documented
  the new keys, their costs and their limits. **Comment-only; parsed YAML is
  unchanged.**

---

## 10. V2V changes

Automatic source duration **was already correct** and is untouched: `duration_mode:
source`, the API rejects a supplied duration, the frontend hides the selector,
the worker reads the probe, and the output is verified against the source's
length. The client's P0 was already met; the tests now cover it on both engines.

What is new is the transformation path (§7), plus a shared `_deliver_restyle` so
"the source's audio survives exactly once" and "the result is the source's
length" cannot hold on one engine and quietly not on the other.

---

## 11. Music Video changes

Automatic duration and single-master-audio **were already correct** and are
untouched. What is new is that the model can now hear the song, per section, by
seeking into the master.

---

## 12. Lip-sync changes

Native (`a2vid_two_stage`), not a separate provider — see §7 and §8. The
provider seam the brief asked for exists as `LtxPipeline` + `AudioConditioning`:
a future `LipSyncProvider` would be a second pipeline descriptor and a second
conditioning type, not a rewrite. Adding one before measuring goal C would be
adding a model because it is popular, which the brief explicitly warned against.

---

## 13. RTX PRO 6000 measurements (17 Aug 2026)

All unquantized with `--offload cpu` (the LoRA rule). Card idles at ~24 GB with
ACE-Step resident; peak during these runs stayed within the 96 GB envelope
alongside it.

**IC-LoRA transform**

| grid | frames | stage | wall |
|---|---|---|---|
| 512x320 (req 1024x640) | 97 | 1 only | **36s** |
| 1024x576 (req 2048x1152) | 193 | 1 only | **62s** |
| 1024x576 | 193 | two-stage | **FAIL** — VAE rearrange, odd latent height |

≈7.7x real time at the production grid — comparable to the existing restyle.

**Audio-conditioned (a2vid, 1024x576)**

| frames | video | wall | wall per second |
|---|---|---|---|
| 193 | 8.04s | 147s | 18.3x |
| 241 | 10.04s | 141s | 14.0x |
| 481 | 20.04s | 211s | **10.5x** |

Longer passes are *cheaper* — each pass reloads a 22B transformer from host RAM
and that fixed cost dominates a short one. Hence a 20s pass ceiling: a 3-minute
song is 9 passes / ~32 min, against 22 passes / ~54 min at 8s. Roughly 4x the
default tier. **This is the strongest available argument for model residency**
(§16), but correctness came first.

---

## 13a. End-to-end music-video test (17 Aug, post-implementation)

A real 40s vocal track generated by the production ACE-Step service (known
lyrics, solo female vocal), then rendered exactly the way `_run_music_video`
renders it with `audio_conditioning: true` — 2 passes of 481 frames, master
seeked per pass, parts assembled silent, master muxed once.

| | |
|---|---|
| Scene 1 (`--audio-start-time 0.0`) | PASS, 213s |
| Scene 2 (`--audio-start-time 20.0`) | PASS, 214s |
| Song / video | 40.032s → 40.000s |
| Total GPU | 427s for 40s of video — **10.7x real time**, matching §13 |

**Lip-sync verdict — goal B is real, and the evidence is the silence.**
Vocal-band energy (300–3400 Hz, 1s windows) against the rendered mouth:

| t | vocal band | mouth |
|---|---|---|
| 3.2s | −17.8 dB | closed (between words) |
| 3.6s | −17.8 dB | open, rounded vowel, eyes closed, visible effort |
| 21.0s | −19.4 dB | open, singing |
| 38.0s | **−78.3 dB** (track has ended) | **closed** |

A model that animated a mouth continuously would still be singing at 38s. This
one stops when the audio stops, which is the cheapest strong evidence that the
mouth is driven by the track rather than by the prompt.

**Identity across the seam held**: the frame 1s after the 20s boundary is the
same performer, wardrobe, headphones and studio. Two passes is a weak test of
that; a 3-minute song is nine.

Still not established: goal **C**. Per-frame phoneme accuracy and drift over a
full song need a sync-offset measurement, not eyeballed stills.

## 14. Tests

| Suite | Result |
|---|---|
| Worker | **509 passed**, 1 skipped, 1 failed (pre-existing — see below) |
| API | **107 passed** (infra up) |
| Worker lint (ruff, changed files) | clean |
| Frontend typecheck (`tsc --noEmit`) | clean |
| Frontend lint (`eslint --max-warnings=0`) | clean |
| Frontend build (`next build`) | clean |

24 new tests. Coverage of the client's named cases: V2V duration on both engines
(tests 1–2), source audio preserved once (3), silent source stays silent (4),
control clips per window preserve timing/cuts (5), transformation strength
verified visually on the GPU (6); music video duration, continuity of the
master, and per-section audio windows (1–3, 5).

**The one failure is pre-existing and environment-dependent.**
`test_a_track_longer_than_one_pass_becomes_several_scenes` expects a 4.0s MP3 to
probe slightly over 4.0s; this machine's LAME produces exactly 4.0s, so 4 windows
are planned instead of 5. **Verified by stashing all of my changes and re-running
it on a clean tree, where it fails identically.** Not caused by and not related
to this work.

Two test-harness signature updates were required and are noted in §9: the
`_launcher` stub now accepts the module argument. No assertion was weakened.

---

## 15. Regressions

None. T2V, I2V, Extend and Music generation are untouched — the default tier's
argv is byte-identical (`test_the_command_carries_every_flag_the_benchmark_needed`
and the whole existing suite still pin it), and both new paths are unreachable
unless a workflow sets the flag. `test_the_default_engine_is_still_the_still_conditioned_restyle`
and `test_a_plain_music_video_still_never_shows_the_model_the_song` assert
exactly that. Storage, progress/SSE, history, cancellation, heartbeat/lease and
billing were not touched.

---

## 16. Remaining limitations

**ZolexAI (ours to fix)**
- Neither new path is enabled in any shipped workflow. Turning them on is one
  YAML line each, and is a product/pricing decision (§7, §8).
- Production workflow definitions are baked into the API image and differ from
  this repo's copies (`runtime: mock` here). Enabling either flag requires an
  API image rebuild — the runbook already documents this.
- Model residency is not implemented. §13 shows it is now the largest available
  win for audio-conditioned jobs.
- The transform engine has been verified on two clips at two grids. It has not
  been run chained across multiple passes on real customer footage, nor on
  footage with hard scene cuts.

**LTX / model limitations (not ours)**
- Stage 2 cannot run at 1024x576 (odd latent height). Worked around by rendering
  at 2x and stopping at stage 1; the workaround is exact, not a compromise.
- The distilled tier has no CFG, negative prompt or step count. Only the guided
  tiers do.
- The decoder's bad-shape set still follows no rule; the measured tables in
  `ltx.py` remain the only defence.
- `dubit` cannot sync to supplied audio at all (§8).

**Lip-sync limitations**
- Goal **C** (phoneme-accurate, drift-free over a full song) is **unverified**.
  What exists is goal B. The next measurement is a sync-offset check at the
  start, middle and end of a 3-minute render — a model that is synchronised for
  5 seconds and drifts by 3 minutes would not be acceptable, and nothing here
  has measured that yet.
- Identity across many chained sections still relies on the pinned final frame
  and will drift over a long song; multi-shot prompting (§2) is the untested
  lever.
