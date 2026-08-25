# Client H3 ComfyUI pack — first reproduction results

**Date:** 25 August 2026 · RTX PRO 6000 Blackwell WS (Israel box) · commit
lineage from `0490b0b` (the materialized graphs).

**Bottom line first: the client's execution architecture is real, and it
changes the H3 economics by an order of magnitude.** The 60-in-6 claim is not
met as delivered, and long-form content coherence with the pack's default
prompts is poor — both stated precisely below.

---

## 1. What was reproduced, exactly

| Component | Pin | Status |
|---|---|---|
| ComfyUI core | tag `v0.33.3` = `4da9e2db` | **exact** |
| `ComfyUI_MiniMax_H3_Extender` | `6a3583d0…9972` | **exact, full SHA** |
| `ComfyUI-Easy-Use` | `4de1ab3b…2720` | **exact, full SHA** |
| Workflow graphs | the client's three JSONs, byte-transcribed and structurally validated | **exact** |
| Weights | **Comfy-Org/MiniMax-H3 (official)** — every file's SHA256 verified against the repo's LFS records | **official** |

Weights, all `ALL OFFICIAL-VERIFIED`:

```text
diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors  9255f52b…  20.97 GB
text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors       bc2ced0f…  27.14 GB
vae/minimax_h3_video_vae_fp16.safetensors                           7c1f1314…   5.21 GB
vae/minimax_h3_audio_vae_fp32.safetensors                           8e505d95…   0.61 GB
loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors    5b9ab5ad…   1.96 GB  (staged, unused)
```

**The encoder mystery is closed.** `qwen3vl_32b_minimax_h3_int8_convrot` was
never a community/uncensored file — it lives in the **official Comfy-Org repo**
under `text_encoders/`; earlier searches missed it because HF search does not
index files inside repos. No abliterated variant was touched. The official
ComfyUI tutorial (`docs.comfy.org/tutorials/video/minimax/minimax-h3`) is the
provenance chain: it names these exact INT8 ConvRot files.

Recorded deviations (Phase-17 format):

| Setting | Client pin | Ours | Reason |
|---|---|---|---|
| Weight source | unstated | Comfy-Org official, SHA-verified | client had no copy; official beats any mirror (Abiray copies differ in SHA) |
| Model paths | `H3\…` backslashes | `H3/…` | Linux; the guide itself says "reselect the same files" |
| Duration index | ships at 4 | set per run | the guide instructs "Set the duration index first" |
| Python/torch | unstated | 3.12 / torch 2.11.0+cu128 | sm_120 requires CUDA ≥ 12.8 wheels |

## 2. Measured performance — the headline

All runs: the client's exact R2V graph, 544x320, 20 steps, `res_multistep`/
`beta`, fixed pack seeds, three reference images (our D1 man as Picture 1, the
rehearsal room as Picture 2, the singer as Picture 3), default pack prompts.

| Preset | Output | Wall | x real time | Peak VRAM | Host free |
|---|---|---:|---:|---:|---:|
| 5 s | 124 f / 5.167 s | **65.3 s** (incl. first model load) | 12.6x | 52.3 GB | ample |
| 15 s | 362 f / 15.083 s | **165.8 s** | **11.0x** | 58.7 GB | 67 GB |
| 60 s | 1,433 f / 59.708 s, 5 clips | **738.6 s** | **12.4x** | 59.4 GB | 67 GB |

Against our provider-native BF16 diffusers baseline on the same card:

```text
15 s class:  BF16 2429.9 s (169x)   →   INT8 pack 165.8 s (11x)    ≈ 14.7x faster
5 s class:   BF16 1686 s D1 (326x)  →   INT8 pack 65.3 s           ≈ 26x faster (different refs/res/steps)
```

**Attribution (Phase 13), honestly decomposed:** pixels 174,080 vs 1,032,192 =
**5.93x**; steps 20 vs 30 = **1.5x**; combined ≈ 8.9x of the ~15x. The
remaining ~1.7x belongs to INT8+pruning and the ComfyUI execution path
(resident models, dynamic VRAM staging). So most of the speed is *resolution*,
some is *steps*, and a real but smaller share is the quantized build. None of
it is magic.

Memory behaviour is the quiet win: **~59 GB VRAM peak and no host-RAM
pressure at all** (67 GB free throughout, vs BF16's 78 GB host peaks and
one-process-per-generation rule). The server stayed resident across all three
runs — no reload between jobs, which is what production wants.

## 3. The client claim (Phase 22)

```text
CLIENT 60s CLAIM
================
Exact client workflow available:      YES (graphs; transcribed + validated)
Exact pinned environment reproduced:  YES (ComfyUI v0.33.3, both node pins full-SHA)
Exact INT8 weights:                   YES (official Comfy-Org, SHA-verified)

5 s:   5.167 s   wall  65.3 s   VRAM 52.3 GB
15 s: 15.083 s   wall 165.8 s   VRAM 58.7 GB
60 s: 59.708 s   wall 738.6 s   VRAM 59.4 GB   segments 5   context 22 f

Claimed ~60 s in ~6 min:  NOT CONFIRMED — 12.3 min at the pack's own 20 steps.
```

The plausible path to ~6 min is the **official** `ref2v_turbo_4step` LoRA
(5x fewer steps), already downloaded and SHA-verified but deliberately not
used — the baseline had to be the pack as delivered. That experiment is next,
and it is a quality question as much as a speed one.

## 4. Quality — mechanics pass, content drifts

Every output was probed and eyeballed, not merely validated.

**5 s: PASS.** Real scene — the singer at the mic, the room from Picture 2
preserved, generated audio at −15.2 dB, clean frames, luma 74.1, interframe
delta 1.25. Node status confirms full semantics: `refs 3 | ref pack 3 linked,
imported Ref 1,2,3 | generated 1`.

**60 s: the seams hold; the story does not.**

- Interframe delta at the four boundaries (frames 311/600/889/1161):
  4.5 / 8.1 / 5.1 / 10.5 against a global mean of 2.7 and an in-segment max of
  69 — elevated blips, **nothing close to a hard cut**. Frame-pair inspection
  agrees: every boundary pair is visually continuous. The 22-frame motion
  context genuinely works.
- **But the content wanders between segments.** Segment 2 drifts to a warm
  close-up with a different shirt; segment 4 collapses into a near-static
  portrait of the Picture-1 man against his reference backdrop; segment 5
  snaps back to the singer scene. Subjects swap, wardrobe changes, the scene
  resets — everything the pack's own prompts say to avoid.

**Why, and why it is not a verdict on the model:** the shipped Prompts 1–5 are
generic placeholders ("the requested subject", "the requested action") that
never re-describe who is on screen, and our three reference images were
deliberately unrelated (a smoke, not a composition). This is precisely the
presence/identity drift our LTX audit fixed with explicit per-section
re-stated constraints — the failure class is universal, not H3-specific. The
pack's own guide says each later prompt "should begin from the prior final
frame"; the defaults do not.

**Consequence:** the ZolexAI-optimized layer over this pack is obvious and is
work we have already done once for LTX — per-segment prompts that re-describe
subject, wardrobe and scene every segment. That is the "better results" the
client system needs, and it composes with the pack rather than replacing it.

## 5. Where this leaves the routing question

| | LTX 2.5 | H3 BF16 diffusers | **H3 INT8 client pack** |
|---|---|---|---|
| 5 s T2V-class | 28.2 s @1024x576 | 581–1686 s @1344x768 | 65 s @544x320 |
| 15 s class | — | 2429.9 s | **165.8 s** |
| 60 s | ~2 chained passes | impractical | **738.6 s, one queue** |
| VRAM | ~28–95 GB by path | 84–95 GB + 78 GB host | **~59 GB, no host pressure** |
| Local H3 production candidate | — | NO (economics) | **CONDITIONAL — yes on speed/stability; quality needs the structured D1 + prompt-discipline pass** |

Still true and unchanged: production untouched, `auto → ltx`, hybrid paused,
H3 API paused, Tier 2 and the 406 pack not started, `P0 LTX CODE FOLLOW-UP`
(person-anchor artifact) still open.

## 6. Next, in order

1. **Structured D1 through this pack** — coherent per-segment prompts, proper
   reference roles: the real quality read, and the LTX-transform vs
   H3-INT8 vs H3-BF16 three-way.
2. **Turbo 4-step A/B** — the 6-minute question, with quality scored.
3. Resolution ladder (`544x320` vs `960x544` vs native) — quality-vs-speed
   curve for the client tiering.
4. FL2VA INT8 (I2V graph) once R2V quality is judged — one more 21 GB file.

---

## 7. Second pass — structured prompts, turbo, resolution, I2V (25 Aug)

All warm-server runs of the client's graphs via the parameterized runner
(`--prompt`/`--drop-ref3` are guide-sanctioned edits; `--turbo`/`--width` are
recorded deviations).

| Cell | Config | Wall | x RT | VRAM | Verdict |
|---|---|---:|---:|---:|---|
| **D1 structured, 544x320** | R2V, P1 rewritten, Picture 3 dropped | **55.2 s** | 10.7x | 54.8 GB | **PASS** — Picture-1 man at the mic in the Picture-2 room, identity held, stable, no artifacts |
| **D1 turbo 4-step** | + official `ref2v_turbo_4step_v0.1` LoRA, steps 4 | **15.1 s** | **2.9x** | 55.0 GB | **QUALITY FAIL** — wrong subject (singer, not Picture 1), oversaturated, light-streak artifacts |
| **D1 960x544** | 3.07x pixels | **170.9 s** | 33.1x | 59.0 GB | **BEST OUTPUT OF THE EVALUATION** — identity unmistakable, full beard, exact coat, clean detail |
| **I2V 5 s, 1280x736** | FL2VA graph, source = the man's portrait | **336.6 s** (incl. FL2VA first load) | 65x | 63.1 GB | **PASS, near-perfect** — first frame is the source photo; photographic identity through the clip |

What these four cells settle:

1. **The 60 s drift was the prompts, not the model.** One rewritten prompt
   turned R2V from subject-wandering into a clean, correct performance shot.
2. **Turbo v0.1 is not a drop-in.** 3.7x faster and quality collapses on
   reference identity. The 6-minute claim stays speed-plausible,
   quality-unacceptable at v0.1 defaults. Re-test on future LoRA versions,
   or with lower strength / hybrid step counts — as experiments, not defaults.
3. **Resolution is the honest quality dial.** Cost scales linearly with pixels
   (3.07x pixels -> 3.10x time). 544x320 = draft tier, 960x544 = delivery
   tier at ~33x RT, both far below BF16 economics.
4. **FL2VA pins identity the way Ref2VA cannot.** For customer-photo
   animation, the I2V graph is the quality path; Ref2VA is for composed
   scenes and re-enactment. This mirrors our B4 diffusers finding — it is the
   architecture, not the runtime.

FL2VA INT8 downloaded and SHA-verified (`e889202c…`); disk still >300 GB free.

## 8. The discipline proof — 60 s with per-segment re-stated prompts (25 Aug)

Same graph, same seeds, same references and duration plan as the drifting
default-prompt 60 s run; the only change is Prompts 1–5 rewritten in the
ZolexAI style — every segment re-describes the subject ("still the man from
<Picture 1>: grey beard, navy wool coat"), the room, and the camera, and names
its handoff pose.

```text
wall 718.4 s (12.0 min) · 5 clips · 1,433 frames · 59.708 s · VRAM 58.4 GB
```

**Sampled mid-segment (frames 150/450/750/1020/1300): the same man, same coat,
same microphone, same room, same framing in all five segments.** The
default-prompt run's failures — the wardrobe change, the portrait collapse,
the scene reset — are all gone. Nothing else changed.

That is the client system, settled:

```text
client pack architecture (pinned graphs, official INT8 weights)
        + ZolexAI per-segment prompt discipline
        = stable 60 s reference video at ~12x real time on one RTX PRO 6000
```

## 9. Final status

| Workflow | Path | Verdict |
|---|---|---|
| Reference video (R2V) | pack + discipline prompts, 544x320 draft / 960x544 delivery | **production candidate** |
| Photo animation (I2V) | pack FL2VA graph, 1280x736 | **production candidate** (identity is photographic) |
| Long form 60 s | R2V index 4 + discipline prompts | **proven stable** |
| Turbo 4-step | official LoRA v0.1 | **rejected at defaults** — speed 3.7x, quality collapse |
| T2V | LTX 2.5 (28.2 s, 5.6x RT) | unchanged — LTX remains the T2V engine |
| Music video | LTX native A2V (14.7x RT, Goal-B) | unchanged |

Unchanged and true: production untouched, `auto → ltx`, hybrid and H3 API
paused, Tier 2 / 406 pack not started, `P0 LTX CODE FOLLOW-UP` open.
