# ZolexAI Cinematic Music-Video Worker

This is the backend worker for turning a complete song plus a short direction such as `make me a video according to the lyrics of the song` into a planned, resumable music-video render. It analyzes the audio, automatically transcribes timestamped lyrics when requested, translates each lyric window into a filmable scene, creates or loads performer identities, builds frame-exact scenes, generates format-specific anchors, renders each scene independently, retries failed scenes, assembles the accepted clips with the song, upscales the picture once, and validates the 4K master.

The source is ready to copy into a Python backend. Model weights are installed directly on the backend by the included authenticated, resumable model installer instead of being duplicated inside every workflow ZIP. A production deployment must also point the worker at an anchor-image generator and either the official LTX audio-to-video pipeline or an existing render command.

## What is implemented

- One-line creative directions with built-in love, heartbreak, luxury, street, party, inspirational, and general cinematic treatments.
- Zero to five performers, with up to four reference images per person. Two performers receive duet coverage; groups of three to five receive rotating individual coverage plus selected full-band wides and climax scenes. With no references, the worker creates canonical fictional adult identities before it creates scene anchors.
- Automatic multilingual lyric transcription with Faster-Whisper `large-v3`, plus supplied plain-text, LRC, SRT, or existing-service command modes.
- Optional reference-video style matching from an uploaded file or pasted HTTPS link: secure allowlisted fetching, automatic cut/pacing, motion and color measurement, plus an included local Qwen2.5-VL-3B analyzer for shot scales, camera angles, generic scene archetypes, lighting and composition.
- Timestamp-aligned lyric scenes, repeated-chorus motifs, a built-in semantic visual interpreter, symbolic handling of sensitive lines, and an optional local-LLM directing hook.
- Fast LTX scene rendering at 1280×704 landscape, 704×1280 portrait, or 1024×1024 square, all divisible by 32.
- Automatic 4K delivery at 3840×2160 landscape, 2160×3840 portrait, or 2160×2160 square, all at 24 FPS.
- A separate FFmpeg CUDA Lanczos/NVENC finishing stage. The song is encoded once in the working master, then stream-copied unchanged during the 4K upscale.
- Audio decode to 48 kHz stereo PCM, decoded-duration measurement, frame alignment, transient analysis, rough tempo detection, energy/vocal heuristics, and six-part song structure.
- Beat-aware 2–7 second shot planning with a 4.5 second target and exact frame coverage.
- Songs up to five minutes (300 seconds), normally divided into about 67 independently recoverable scenes at the default pacing.
- Structured LTX prompts with one action and one camera behavior per shot.
- Official LTX-2.5 `ltx_pipelines.a2vid_two_stage` command integration, including per-shot audio offsets, image conditioning, legal `8k+1` generation windows, and deterministic retry seeds.
- Command adapters for any existing ComfyUI/image service or alternate video service.
- Per-shot persistence, technical QA, up to three attempts, cooperative cancellation, safe resume, an exclusive job lock, and final H.264/AAC assembly.
- Configurable concurrent render workers with deterministic GPU assignment and ordered final assembly.
- A measured five-minute latency target for a three-minute, single-format job, including a capacity calculator, optional preflight enforcement, parallel audio/lyric analysis, and parallel anchor creation.
- CLI and FastAPI entry points.
- A one-command model installer for the exact LTX-2.5, Faster-Whisper and Qwen files, with gated-access preflight, free-space checking, revision recording, remote-size validation, LFS SHA-256 verification, resume support, a credential-free manifest, and generated backend environment settings.

## Production requirements

- Linux and Python 3.11 or newer.
- FFmpeg and FFprobe available on `PATH`.
- The `lyrics` Python extra and a cached Faster-Whisper model when automatic transcription is enabled.
- The `reference-links` Python extra when customers can paste reference-video URLs.
- A CUDA-compatible PyTorch installation plus the `reference-ai` extra when reference videos use the default local semantic analyzer. Pre-cache its model for predictable job latency.
- The `models` Python extra for the included model installer. The developer must personally accept the LTX-2.5 community license before authentication can access its gated files.
- An NVIDIA GPU environment suitable for the selected video model.
- LTX installed in its own environment, with the configured transformer, text encoder, video VAE, audio VAE, spatial upsampler, and distilled LoRA files.
- A production anchor-image command. It may call ComfyUI, Flux, SDXL, or another image service, but it must follow the contract below.
- Optional identity/anatomy QA and dedicated singing lip-sync commands for stricter production delivery.

Exact LTX installation and model-download commands are in `docs/LTX-2.5-SETUP.md`.

The built-in `reference` anchor backend only crops an uploaded reference and is useful for development. The built-in `mock` backends verify orchestration without consuming GPU time. Neither is a substitute for a production image generator.

## Fast rendering and five-minute jobs

The maximum decoded song length is 300 seconds. The worker never sends five minutes to the video model at once. It creates roughly 4.5-second scenes, renders them independently, and assembles them in exact timeline order. A five-minute song normally creates about 67 scenes.

Safe one-GPU configuration:

```text
ZOLEX_RENDER_CONCURRENCY=1
ZOLEX_RENDER_GPU_IDS=0
LTX_NUM_INFERENCE_STEPS=24
```

Four independent GPU workers:

```text
ZOLEX_RENDER_CONCURRENCY=4
ZOLEX_RENDER_GPU_IDS=0,1,2,3
LTX_NUM_INFERENCE_STEPS=24
```

Each worker receives one scene at a time. Results may finish out of order, but the assembler restores the planned order. Failed scenes retry independently and completed scenes are reused after a restart.

For the lowest latency at scale, point `ZOLEX_RENDER_COMMAND_JSON` at already-warm render services—one service per GPU—so model weights stay resident between scenes. The included direct LTX adapter starts the official Python CLI for each scene and is simpler, but model loading adds overhead. The command request contains `worker_slot` and `gpu_id`, and both values are also available as command-template placeholders.

Rendering speed is hardware-dependent. Parallelism reduces total job time only when each worker has a separate GPU or the external rendering service safely schedules capacity. Do not set concurrency above one on a single GPU; concurrent LTX processes can exhaust VRAM.

Scene generation never runs at 4K. Landscape and portrait scenes use 1280×704 and 704×1280 respectively. After editorial assembly, the worker removes the small LTX alignment padding with a centered crop, scales the picture to 3840×2160 or 2160×3840 on CUDA, encodes with NVENC, and stream-copies the working master's AAC track. This is high-quality GPU scaling, not neural super-resolution; connect `ZOLEX_UPSCALE_BACKEND=command` if the backend has a dedicated AI video upscaler. AI super-resolution normally costs more time.

### Five-minute processing target for a three-minute video

This is separate from the five-minute maximum song length. The worker is calibrated to the supplied real baseline: a three-minute video currently completes in about 900 seconds. A 300-second target requires approximately three times the measured render capacity if per-lane performance stays unchanged.

Use the existing warm backend through `ZOLEX_RENDER_BACKEND=command`; do not replace it with the direct per-scene LTX CLI for this latency target. Set `ZOLEX_BASELINE_RENDER_CONCURRENCY` to the number of simultaneous render lanes used by the measured 15-minute run. The worker calculates the required concurrency, writes it to `latency-plan.json`, exposes it in the final result, and can reject an under-provisioned job before rendering when `ZOLEX_ENFORCE_LATENCY_CAPACITY=true`.

Example: if the current 15-minute benchmark used four equivalent render lanes, the five-minute target requires about 12 equivalent lanes. This is a capacity target, not a promise: queue load, renderer speed, retries, storage, and GPU availability can still move the actual completion time. The worker does not automatically reduce quality settings to claim success.

The target assumes one output format. Requesting landscape and portrait in the same job nearly doubles generative work and therefore doubles the calculated capacity requirement. Copy `config/five-minute-target.env.example`, connect its command to the backend's existing warm renderer, and replace the example baseline concurrency with the real measured value.

Audio feature analysis and lyric transcription now run concurrently. Unique scene anchors also generate concurrently according to `ZOLEX_ANCHOR_CONCURRENCY`, while render clips retain fixed GPU lanes and exact editorial order.

Additional tuning and deployment guidance is in `docs/PERFORMANCE-AND-5-MINUTE.md`.

## Install

```bash
unzip zolexai-music-video-worker.zip
cd zolexai-music-video-worker
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[api,lyrics,reference-links,reference-ai,models]'
cp .env.example .env
```

Install and verify all model weights directly into backend model storage:

```bash
# First accept the LTX terms at https://huggingface.co/Lightricks/LTX-2.5
hf auth login
zolex-music-video install-models --models-root /models
```

The installer writes `/models/zolex-models.env`. Load those values through the process manager, keep the existing `LTX_PYTHON` setting for the official LTX environment, then run `zolex-music-video doctor`. Full options and storage details are in `docs/MODEL-INSTALLER.md`.

Export the values from `.env` through the backend's process manager, Docker environment, or secrets system. The package does not silently load `.env` files.

Check the host:

```bash
zolex-music-video doctor
```

## Minimal request

Only the creative direction is short. The backend still supplies a unique job ID and the stored song path.

```json
{
  "job_id": "mv-love-0001",
  "audio": "/srv/zolexai/uploads/song.wav",
  "prompt": "make me a video according to the lyrics of the song",
  "formats": ["16:9", "9:16"]
}
```

That one prompt activates automatic transcription. The worker aligns the detected lyric segments to the song, uses the active lines to direct each scene, keeps repeated chorus imagery recognizable, and fails before rendering if no usable lyrics can be recovered. The final videos are 3840×2160 and 2160×3840 and match the complete decoded audio duration up to five minutes.

To make the new video follow the filmmaking style of another video, add either one optional local path or one approved HTTPS link:

```json
{
  "job_id": "mv-reference-0001",
  "audio": "/srv/zolexai/uploads/customer-song.wav",
  "prompt": "make me a video according to the lyrics of the song",
  "reference_video": "/srv/zolexai/uploads/style-reference.mp4",
  "formats": ["16:9"]
}
```

Webpage link example:

```json
{
  "job_id": "mv-reference-link-0001",
  "audio": "/srv/zolexai/uploads/customer-song.wav",
  "prompt": "make me a video according to the lyrics of the song",
  "reference_video_url": "https://youtu.be/approved-reference",
  "formats": ["16:9"]
}
```

For a pasted link, the backend fetches one temporary local copy, rejects playlists, enforces HTTPS, blocks IP-address URLs, checks the approved-domain list, and limits duration and file size before analysis. The included local vision model then recognizes supported camera-angle families, shot scales, composition, lighting and generic scene types from a contact sheet while the metric pass measures cuts, pacing, motion and color. Those directions enter the treatment, anchor prompts and every LTX scene prompt. The customer's song remains the only delivery audio. The worker does not copy the reference's people, dialogue, lyrics, logos, recognizable locations or exact shot sequence. Details, model setup and the external-service override are in `docs/REFERENCE-VIDEO.md`.

`lyric_mode` defaults to `automatic`: lyric transcription runs when the prompt explicitly mentions following the lyrics. Set it to `always` to transcribe every song or `off` to disable lyric processing. Supplying a `lyrics` string always takes priority over automatic transcription and may contain plain lines, LRC timestamps, or SRT timestamps. `lyrics_language` is optional.

Run it directly:

```bash
zolex-music-video run --request /srv/zolexai/requests/mv-love-0001.json
```

With one real performer:

```json
{
  "job_id": "mv-love-0002",
  "audio": "/srv/zolexai/uploads/song.wav",
  "prompt": "make me a love video",
  "performers": [
    {
      "id": "artist",
      "role": "lead",
      "description": "preserve the white linen outfit and silver chain",
      "reference_images": [
        "/srv/zolexai/uploads/artist-front.png",
        "/srv/zolexai/uploads/artist-three-quarter.png"
      ]
    }
  ],
  "formats": ["16:9"],
  "lip_sync": true
}
```

A single supplied performer produces a solo romantic treatment; the worker does not invent an unidentified partner. Two supplied performers can act as the romantic leads. With no performer images, the configured image generator creates fictional adult identity references first.

## Duets and five-person bands

The backend accepts as many as five separate performers, with up to four reference images for each person. Two performers receive alternating solo coverage plus duet scenes. Groups of three to five receive rotating individual scenes plus selected ensemble medium shots, full-group wides and climax scenes. If `max_people_visible_per_shot` is omitted from a normal JSON request, it defaults to the number of supplied performers up to five.

Every member keeps a separate ID, role, description and reference-image set. Roles such as vocalist, guitarist, bassist, keyboardist and drummer are carried into the image and video prompts so instrumentalists are given instrument-appropriate actions instead of being told to sing. Group prompts require clear spacing and forbid face blending, identity swaps, duplicate members and unassigned people.

A complete five-member request is included at `examples/request.five-person-band.json`. See `docs/DUETS-AND-GROUPS.md` for the input contract and identity guidance.

## HTTP API

Start the included API:

```bash
uvicorn zolex_music_worker.api:app --host 0.0.0.0 --port 8080
```

Create and inspect a job:

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/music-video-jobs \
  -H 'content-type: application/json' \
  --data-binary @request.json

curl -sS http://127.0.0.1:8080/v1/music-video-jobs/mv-love-0001
```

Cancel cooperatively:

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/music-video-jobs/mv-love-0001/cancel
```

FastAPI background tasks are suitable for a single worker process. In a multi-host production system, submit the same request object to the existing queue and call `zolex_music_worker.worker.run_job` inside the queue consumer. Keep `ZOLEX_WORK_ROOT` on durable shared storage if jobs may resume on another host.

## Anchor command contract

Set:

```text
ZOLEX_ANCHOR_BACKEND=command
ZOLEX_ANCHOR_COMMAND_JSON=["/path/generate-anchor","--request","{request_json}","--output","{output}"]
```

The worker writes a JSON request and invokes the configured argument array without a shell. Available placeholders are `{request_json}`, `{output}`, `{width}`, `{height}`, and `{references_json}`. The request includes:

```json
{
  "schema_version": 1,
  "anchor_role": "close_performance",
  "format": "16:9",
  "width": 1280,
  "height": 704,
  "delivery_width": 3840,
  "delivery_height": 2160,
  "performer_ids": ["artist"],
  "references": ["/absolute/reference.png"],
  "prompt": "complete still-image prompt",
  "output": "/absolute/required-output.png"
}
```

The command must exit zero and write an RGB-compatible PNG at exactly the requested working width and height. The delivery dimensions describe the final master; anchors and LTX scene renders use the smaller working size for speed. For `anchor_role=identity_reference`, the command must create exactly one fictional adult identity on a 768×768 canvas. For other roles, it must use all supplied references as identity conditioning, preserve wardrobe and attributes, and compose the lyric-directed shot. Never return a URL; download the image before the command exits.

## Lyric transcription and directing

The default production transcriber is Faster-Whisper `large-v3`. It runs only for a lyric-driven request, so a normal request such as `make me a love video` does not incur transcription time. The exact transcript is stored at `analysis/lyrics.json` with frame-aligned segments, language, confidence, and detected repeated lines.

To use an existing speech-to-text service instead:

```text
ZOLEX_TRANSCRIPTION_BACKEND=command
ZOLEX_TRANSCRIPTION_COMMAND_JSON=["/path/transcribe-lyrics","--request","{request_json}","--output","{output}"]
```

The command receives `{request_json}`, `{audio}`, `{language}`, and `{output}` placeholders. It must write:

```json
{
  "language": "en",
  "language_probability": 0.98,
  "model": "production-asr",
  "segments": [
    {
      "start_seconds": 12.4,
      "end_seconds": 16.8,
      "text": "timestamped lyric line",
      "confidence": 0.94
    }
  ]
}
```

The built-in lyric interpreter always creates a safe, filmable theme and visual beat from each lyric window and places the active lyric idea inside the corresponding LTX prompt. An optional local-LLM command can refine those decisions:

```text
ZOLEX_LYRIC_DIRECTOR_COMMAND_JSON=["/path/direct-lyrics","--request","{request_json}","--output","{output}"]
```

It reads the complete treatment, transcript, timing, assigned performers, and built-in fallback direction. It must return `{"shots":[{"id":"shot-001","lyric_theme":"...","visual_direction":"..."}]}`. Returned shot IDs and lengths are validated; the command cannot alter timing or performer assignment.

The full processing order, operating modes, and production notes are in `docs/LYRIC-TO-SCENE.md`.

## Alternate video-render command

The direct LTX adapter is the normal path. To call an existing render service instead, set:

```text
ZOLEX_RENDER_BACKEND=command
ZOLEX_RENDER_COMMAND_JSON=["/path/render-shot","--request","{request_json}","--output","{output}"]
```

Available placeholders are `{request_json}`, `{output}`, `{anchor}`, `{audio}`, `{prompt}`, `{start_seconds}`, `{raw_frames}`, `{delivered_frames}`, `{fps}`, `{width}`, `{height}`, `{delivery_width}`, `{delivery_height}`, `{seed}`, `{worker_slot}`, `{gpu_id}`, and `{target_job_seconds}`. `width` and `height` are the fast scene-render dimensions; the delivery values are the required 4K master dimensions. The request file is the authoritative contract and also contains the worker assignment, five-minute target, latency priority, and preserve-quality policy. The command must create a decodable scene video at `output`. The worker normalizes and trims it to the exact delivered frame count, assembles all clips at working resolution, and performs one separate GPU upscale.

## Automatic 4K upscale

Production defaults to FFmpeg CUDA scaling and NVENC:

```text
ZOLEX_UPSCALE_BACKEND=cuda
ZOLEX_UPSCALE_GPU_ID=0
ZOLEX_UPSCALE_CQ=18
```

The landscape path center-crops 1280×704 to remove its 14-pixel LTX padding on each side before scaling to 3840×2160. Portrait performs the equivalent top-and-bottom crop before scaling to 2160×3840. This prevents geometric stretching. `working-master.mp4` contains the full aligned song; the 4K command uses `-c:a copy`, so the audio bytes are unchanged during upscaling.

The CUDA FFmpeg build must provide the `hwupload_cuda` and `scale_cuda` filters plus the `h264_nvenc` encoder. The worker also supports `ZOLEX_UPSCALE_BACKEND=cpu` for diagnostics and a shell-free command adapter for a true AI upscaler:

```text
ZOLEX_UPSCALE_BACKEND=command
ZOLEX_UPSCALE_COMMAND_JSON=["/srv/zolexai/bin/ai-upscale","--request","{request_json}","--output","{output}"]
```

Command placeholders are `{request_json}`, `{input}`, `{output}`, `{width}`, `{height}`, `{delivery_width}`, `{delivery_height}`, `{fps}`, `{frames}`, and `{gpu_id}`. The command writes the exact 4K video to `{output}`; the worker then takes the unchanged audio from the working master. GPU Lanczos finishing is normally far faster than generative rendering, but its exact time depends on GPU, decoder, NVENC throughput, storage, song length, and concurrent load. Benchmark it on the deployment host; the code does not promise that every five-minute 4K upscale takes only a few seconds.

## QA and lip-sync hooks

Technical QA is built in: frame count, frame rate, dimensions, decode health, darkening, luminance jumps, final audio presence, and exact master duration. Visual identity, hands, anatomy, and high-confidence mouth synchronization require specialized models, so production may attach these supervised commands:

```text
ZOLEX_QA_COMMAND_JSON=["/path/qa-shot","--clip","{clip}","--shot","{shot_json}","--report","{report}"]
ZOLEX_LIPSYNC_COMMAND_JSON=["/path/lipsync-shot","--request","{request_json}","--output","{output}"]
```

The QA command exits zero to accept the clip and nonzero to trigger the normal retry policy. Its placeholders are `{clip}`, `{report}`, `{shot_json}`, `{format}`, `{width}`, `{height}`, and `{fps}`. The lip-sync command receives a request with the clip, full aligned audio, shot offset, duration, performer IDs, and required output path. It must write the requested video and exit zero; the worker then normalizes and rechecks it.

LTX audio conditioning remains active even without a dedicated lip-sync hook, but the result file explicitly reports `ltx_audio_conditioning_only` and marks the delivery for creative review.

## Job files and recovery

Every job is stored under `ZOLEX_WORK_ROOT/<job_id>/`:

```text
request.json
status.json
events.jsonl
audio/
latency-plan.json
analysis/song-analysis.json
analysis/lyrics.json
analysis/reference-video/contact-sheet.jpg
analysis/reference-video/style-profile.json
treatment.json
identities/identity-package.json
16x9/editorial-plan.json
16x9/shots.json
16x9/anchors/
16x9/shots/shot-001/attempt-001/
16x9/shots/shot-001/accepted.mp4
16x9/final/working-master.mp4
16x9/final/upscale-command.json
16x9/final/output.mp4
16x9/final/delivery-qa.json
result.json
```

Re-run the same request and job ID to resume. Accepted clips are revalidated and reused. Failed or missing shots resume independently. Reusing a job ID with a different request is rejected. `shots.json` keeps the strict portable five-field renderer manifest; `editorial-plan.json` contains the richer internal plan.

## Backend integration checklist

1. Copy this directory into the backend image or install it as a local Python package.
2. Install a CUDA-enabled FFmpeg build with `scale_cuda` and `h264_nvenc`, this package, and the official LTX pipeline in the GPU environment.
3. Install this package with `.[api,lyrics,reference-links]` and allow the configured Faster-Whisper model to download once into the model cache, or attach the existing transcription command.
4. Set the six LTX model paths from `.env.example` to the actual files on the server.
5. Point `ZOLEX_ANCHOR_COMMAND_JSON` at the backend's identity-aware image generator.
6. Store song and performer-image uploads before submitting a job. For the reference, send either a stored server path or `reference_video_url` from the webpage link field.
7. Use a durable queue for multi-worker deployment; a job ID may have only one active worker.
8. Connect `ZOLEX_REFERENCE_ANALYZER_COMMAND_JSON` to a local vision model or existing multimodal service for semantic camera/style analysis; the built-in technical analyzer remains available without it.
9. Enable the optional lyric-director, QA, and lip-sync commands when the production backend provides them.
10. Run `python -m unittest discover -s tests -v`, then run one mock reference-video job, one lyric transcription job, one LTX GPU shot, and one CUDA 4K-upscale smoke job before production traffic.

## Development smoke mode

Mock mode exercises the real decode, analysis, planning, anchors, retry/resume, frame trimming, assembly, and delivery-QA path without loading a generative model:

```bash
export ZOLEX_ANCHOR_BACKEND=mock
export ZOLEX_RENDER_BACKEND=mock
export ZOLEX_UPSCALE_BACKEND=cpu
zolex-music-video run --request examples/request.mock.json --work-root ./runs
```

The resulting video is a moving diagnostic slate, not a creative delivery.

## Security boundaries

- Job IDs are restricted and cannot escape the work root.
- External commands are JSON argument arrays and are never passed through a shell.
- Input files must exist before a job starts; every source is hashed.
- Reference links are restricted to HTTPS, an administrator-controlled hostname allowlist, non-IP hosts, one video rather than a playlist, configured duration and byte limits, and one cached job-local file. Other arbitrary request URLs are never downloaded.
- Only ingest reference videos the customer has permission to use. The backend must authenticate users, authorize storage paths, enforce ownership, rate-limit submissions, scan downloads/uploads, and isolate GPU jobs.
- Model licenses and acceptable-use requirements remain the deployer's responsibility.
