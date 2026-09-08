# Music Video on the client's music-video worker (v1.8.0) — 8 Sep 2026

The client delivered `ZolexAI-Music-Video-Worker-v1.8.0.zip`
(sha256 `5c512aecd3b28bb440f26ad853928fb38592666a4faf6d70d70d2d4f4dc69ead`),
a complete music-video orchestrator, with the note that it now supports
bands of up to five people. It is integrated as the engine behind Music
Video in client-test. This is the account of what it is, what it needed
from us, what was measured, and what is still open.

## What the package is

`zolex_music_worker` (7,200 lines, 33 of its own tests, all passing on the
node) turns a song plus a short direction into a planned render:

1. decodes the song to 48 kHz stereo, aligns it to a 24 fps timeline
   (songs up to 300 s);
2. analyses transients, tempo, energy and a six-part structure; in
   parallel transcribes timestamped lyrics with faster-whisper large-v3
   when the direction asks to follow the lyrics (or lyrics are pasted:
   plain, LRC or SRT);
3. plans 2–5 s shots cut on transients and lyric boundaries, assigns
   shot families (environment, close/medium performance, wide movement,
   detail inserts, seated reflection, firelight, profile), rotates solo
   coverage across the performers and adds duet / full-band scenes, gives
   instrumentalists instrument actions, and writes one prose prompt per
   shot with the active lyric idea in it;
4. renders one anchor still per distinct shot through a deployer-supplied
   image generator — with the performers' real faces from their reference
   pictures;
5. renders each shot independently on the official
   `ltx_pipelines.a2vid_two_stage` (the audio tier: dev transformer +
   distilled LoRA, audio conditioning from the song at the shot's own
   offset, the anchor pinned at frame 0), with technical QA and up to
   three attempts per shot, resumable per shot;
6. assembles the accepted clips at the working size with the aligned
   song, finishes once to 4K with FFmpeg CUDA Lanczos + NVENC (audio
   stream-copied), and validates the master.

It is vendored byte-for-byte at `apps/worker/zolex_music_worker/`
(fingerprint pinned by `tests/test_music_video_worker.py`) and the
delivery's docs, tests and examples are kept under
`benchmarks/client-pack/music-video-worker/v1.8.0/`.

## What the package leaves to the deployer, and what this node supplies

| their contract | ours | where |
| --- | --- | --- |
| anchor-image generator (command) | Qwen-Image-Edit-2509 on the LTX ComfyUI, Lightning 4-step LoRA; up to 3 reference pictures per still (more become one contact sheet); text-to-image for empty scenes and fictional identities | `scripts/mv_anchor.py`, `worker/comfy/qwen_edit.py` |
| render backend | `scripts/mv_render.py`: frees ComfyUI, then the official a2vid CLI with this node's six model files and the platform's audio-tier flags | `scripts/mv_render.py` |
| transcription | faster-whisper large-v3 in-process on the GPU | `worker.musicvideo.prepare_whisper_libraries` |
| 4K finish | FFmpeg `scale_cuda` + `h264_nvenc` (the package's own default) | package |
| lip-sync command | none exists on this platform → `ltx_audio_conditioning_only`, as the package reports | — |
| reference-video style matching | not wired (no product input for it; Qwen2.5-VL not installed) | — |

The platform side is `worker/musicvideo/` (job → their request and config,
their `status.json` → the progress bar) and `worker/adapters/music_video.py`
(runtime `music_video`). The CLI runtime `ltx` still serves `music-video`
wherever a deployment routes it there; that is the rollback.

**Why an image model had to be installed.** The node had no still-image
model. The anchor is what puts the customer's actual performer into a shot
composed for the lyric — a text-only anchor cannot do that, and the
package's fallback (the reference photo fitted to the frame) never leaves
the photo's own setting. Qwen-Image-Edit-2509 is Apache-2.0, ComfyUI runs
it natively, and it keeps the faces in its reference pictures: the first
two-performer test still (rooftop, blue hour, microphone, guitar) is
recognisably both men from their photos. 29 GB of weights, 12.5 s for the
first still including load, 2–8 s per still after.

## Product surface

- `music-video.yaml`: five optional picture inputs `performer_1…5`,
  `settings.performers` (role + description per picture, sent as the
  `performers` parameter, validated against the slots) and
  `settings.lyrics` (pasted lyrics + a language hint for the transcriber).
  The definition's pin in `test_untouched_workflows.py` moved on the
  client's word (this delivery).
- API: `PerformerSpec{slot, role, description}`, at most five; the
  registry refuses performers on workflows that do not declare them and
  refuses slots the definition has no picture input for.
- Web: a role select and a description field beneath each performer
  dropzone; the band travels back on "Reuse settings"; lyric copy reads as
  transcription on video output. A request with no band is byte-identical
  to before.
- Deploy: `vps-local.sh --profile client-test` routes `music-video` to
  `music_video` (production keeps the CLI audio tier). Both api and web
  rebuild. Runbook §49 has the node side.

## Measured, 8 Sep 2026 (RTX PRO 6000 Max-Q, node ltx-6000-2)

40-second song, two performers with pictures, "according to the lyrics",
16:9, the package's default 24 steps. Nine shots planned (109–104 frames
each, every one rendered in the package's minimum 121-frame window).

| stage | wall |
| --- | --- |
| decode, analysis, transcription (9 lyric lines, en 0.74), plan | 2–4 s |
| 9 anchor stills (Qwen, incl. first load) | 51 s |
| 9 shots on a2vid, 24 steps, offload cpu (`run1`) | 120 s each; 1,085 s |
| assembly + 4K (3840×2160, NVENC) + QA | 31 s |
| **total, run1** | **1,183 s (19.7 min) — 30× real time** |
| 9 shots with the transformer resident (`LTX_UNQUANTIZED_OFFLOAD=none`, `run2`) | 96 s each (one outlier at 172 s) |

Where a shot's time goes (offload cpu / none): 24-step stage 1 at 640×352
= 70 s / 53 s (2.9 → 2.25 s per step — weight streaming from host RAM is
the difference); 3-step stage 2 at 1280×704 = 8 s; model load, prompt
encoding and VAE decode ≈ 50 s / 35 s per process. Resident weights peak
at 43.7 GB on the card, which fits beside nothing else — the render
command evicts ComfyUI first for that reason.

Delivery QA passed (960 frames, 24 fps, 3840×2160, AAC); every shot's
technical QA passed. The frames: the two men from the photos, in their
described clothes, across an opening insert, solo close-ups and mediums,
two duet scenes at a night bonfire and a corner store, a rooftop profile —
the lyric ideas ("bottle", "hold the light", "black cab, backstreet") are
visible as shot content.

**What this means for a whole song.** Per shot ≈ 96–120 s, one shot per
~4.4 s of song: a 3-minute song is ~41 shots ≈ 65–80 min; a 5-minute song
≈ 67 shots ≈ 1.8–2.2 h, inside the 2-hour job budget only at the resident
setting. The platform's previous Music Video (the CLI audio tier, 20 s
sections, 15 steps) did 3 minutes in ~22 min. The package buys per-shot
lyric direction, real performer faces, cuts on the music and 4K for about
3–4× the GPU time. The package's own docs say the same: their five-minute
target needs ~12 warm render lanes.

## Levers, in order of what they are worth

1. **Keep the transformer resident** (`LTX_UNQUANTIZED_OFFLOAD=none`,
   measured −20 % per shot). Set on the node.
2. **A warm render service.** Each shot pays ~35 s to start a process and
   load 70 GB of weights; a service holding the pipeline between shots
   would take a shot to ~60 s. `scripts/mv_render.py` is the seam
   (`MUSIC_VIDEO_RENDER_COMMAND` names a replacement); the package's
   `CommandRenderAdapter` already sends `worker_slot`/`gpu_id`. Not built
   yet — it must also unload when idle so Text to Video can have the card.
3. **Steps.** 24 is the package's default; the platform's own audio tier
   ran 15 after the user saw and accepted the trade (27 Aug). At 15 the
   first stage is ~33 s: a shot ≈ 75 s resident. `execution.inference_steps`
   or `MUSIC_VIDEO_INFERENCE_STEPS`.
4. Longer shots would not help: every shot renders 121 frames regardless
   (the package's minimum window), and the counts between 121 and 193 are
   unmeasured decoder landings on this card — hence `shot_max_seconds`
   5.0 rather than their 7.0.

## Open

- Multi-reference identity: the package allows four pictures per person;
  the product takes one per performer (one dropzone per role). Their
  guidance calls one clear front-facing photo the minimum.
- Group anchors with more than three members condition on a contact
  sheet, not three separate pictures; a five-piece wide has not been
  rendered yet.
- The reference-video style input (`reference_video_url`) is not
  exposed; wiring it needs yt-dlp and the Qwen2.5-VL analyser.
- Lip-sync is what the audio tier gives (mouth follows the vocal);
  nothing measures phoneme accuracy.
- Whisper on sung vocals is the package's chosen transcriber; the earlier
  finding stands that it can mislabel a language on a heavy mix. Pasted
  lyrics always win.
