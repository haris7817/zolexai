# Reference-video style workflow

## Customer input

The workflow accepts either one stored server-side video path in `reference_video` or one pasted HTTPS link in `reference_video_url`. Do not send both.

```json
{
  "job_id": "reference-style-0001",
  "audio": "/srv/zolexai/uploads/customer-song.wav",
  "prompt": "make me a video according to the lyrics of the song",
  "reference_video": "/srv/zolexai/uploads/style-reference.mp4",
  "formats": ["16:9"]
}
```

The customer song remains the delivery audio. The reference video's audio, people, dialogue and story are not transferred.

For a webpage link field, submit:

```json
{
  "job_id": "reference-link-0001",
  "audio": "/srv/zolexai/uploads/customer-song.wav",
  "prompt": "make me a video according to the lyrics of the song",
  "reference_video_url": "https://youtu.be/approved-reference",
  "formats": ["16:9"]
}
```

Install `.[reference-links]` for the built-in yt-dlp adapter. The default approved hosts are `youtube.com`, `youtu.be`, and `vimeo.com`; subdomains are accepted. Administrators can replace this list only with domains their service is permitted to ingest:

```text
ZOLEX_REFERENCE_FETCH_BACKEND=yt_dlp
ZOLEX_REFERENCE_ALLOWED_HOSTS=youtube.com,youtu.be,vimeo.com
ZOLEX_REFERENCE_MAX_BYTES=1073741824
ZOLEX_REFERENCE_FETCH_TIMEOUT=1800
```

The URL must use HTTPS and cannot include credentials or an IP-address host. Playlists are disabled. The downloaded file is capped by `ZOLEX_REFERENCE_MAX_BYTES`, cached inside the job directory for safe resume, and then subjected to the same decoded-duration limit as an uploaded reference. A trusted existing fetch service can be connected with `ZOLEX_REFERENCE_FETCH_BACKEND=command` and `ZOLEX_REFERENCE_FETCH_COMMAND_JSON`; its placeholders are `{request_json}`, `{url}`, `{output}`, `{max_bytes}`, and `{max_seconds}`.

## Automatic analysis

When a reference is supplied, the worker automatically:

1. Securely fetches and caches an approved pasted link when present, then validates and hashes the local reference file.
2. Extracts evenly spaced visual samples and creates `contact-sheet.jpg`.
3. Detects scene cuts and estimates the reference's average shot duration.
4. Measures brightness, contrast, saturation, color temperature, dominant mean color and frame-to-frame visual change.
5. Sends the contact sheet to the included Qwen2.5-VL-3B analyzer, which recognizes visible shot scales, supported camera-angle families, generic scene archetypes, lens/composition tendencies, lighting and camera language.
6. Validates the model's structured JSON and combines it with the deterministic measurements. A configured external analyzer command takes priority when present.
7. Unloads the local vision model and empties its CUDA cache before LTX rendering begins.
8. Uses the validated style profile in the treatment, anchor prompts and every LTX scene prompt.
9. Moves the new scene cadence toward the reference's detected average shot length, clamped to the worker's safe 2–7 second scene range.

The complete analysis is saved at `analysis/reference-video/style-profile.json`. This makes the style decision inspectable and reproducible.

## Included local vision AI

The production environment loaded through `WorkerConfig.from_env` defaults to:

```text
ZOLEX_REFERENCE_VISION_BACKEND=qwen
ZOLEX_REFERENCE_VISION_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
ZOLEX_REFERENCE_VISION_DEVICE=cuda:0
ZOLEX_REFERENCE_VISION_MAX_NEW_TOKENS=768
ZOLEX_REFERENCE_VISION_MIN_PIXELS=200704
ZOLEX_REFERENCE_VISION_MAX_PIXELS=1003520
ZOLEX_REFERENCE_VISION_LOCAL_FILES_ONLY=false
```

Install CUDA-enabled PyTorch using the build appropriate for the production GPU, then install the packaged analyzer dependencies:

```bash
python -m pip install '.[reference-ai]'
```

The model weights are not embedded in the ZIP. On the first run, Transformers downloads them from the configured model ID. For production, pre-cache them on every worker, set `ZOLEX_REFERENCE_VISION_LOCAL_FILES_ONLY=true`, or set `ZOLEX_REFERENCE_VISION_MODEL` to an absolute local model directory. A cold download is not part of the five-minute processing target.

The model sees the numbered contact sheet, not the customer song and not the final delivery. The prompt asks it to report only claims supported by the sampled frames. Its response is saved to `analysis/reference-video/reference-vision-ai.json`, then its objects are deleted and CUDA memory is released before LTX is loaded. Set `ZOLEX_REFERENCE_VISION_BACKEND=disabled` to retain only deterministic measurements.

The included analyzer uses `Qwen/Qwen2.5-VL-3B-Instruct`, the smallest instruction-tuned Qwen2.5-VL variant, to reduce load time and VRAM pressure. Its official model card describes image/video understanding and structured-output capabilities: <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>.

## Existing vision-service override

For the lowest latency at scale, an already-warm local vision service can override the in-process analyzer:

```text
ZOLEX_REFERENCE_ANALYZER_COMMAND_JSON=["/srv/zolexai/bin/analyze-reference-video","--request","{request_json}","--contact-sheet","{contact_sheet}","--output","{output}"]
```

Available placeholders are `{request_json}`, `{video}`, `{contact_sheet}`, and `{output}`. The command reads the generated request and writes a JSON object containing any of these string fields:

```json
{
  "style_summary": "grounded handheld hip-hop film with controlled energy",
  "camera_language": "low tracking moves, restrained push-ins and stable performance framing",
  "shot_scale_and_angles": "low-angle environmental wides alternating with eye-level medium performance frames and close-ups",
  "scene_archetypes": "generic urban night exteriors and sparse practical-lit performance interiors",
  "lighting_style": "warm practical key light with deep neutral shadows",
  "color_palette": "warm amber highlights, natural skin and restrained cyan shadows",
  "editing_rhythm": "brisk hard cuts averaging three seconds",
  "composition_style": "alternating environmental wides and centered medium performance frames",
  "motion_style": "natural walking and performance gestures with moderate camera motion"
}
```

Unknown fields are ignored. Accepted fields must be nonempty strings no longer than 2,000 characters. External commands are argument arrays and never run through a shell. When `ZOLEX_REFERENCE_ANALYZER_COMMAND_JSON` is set, it takes priority over `ZOLEX_REFERENCE_VISION_BACKEND`.

## Originality boundary

The reference is a filmmaking-style guide. The worker transfers general pacing, shot-scale/angle families, camera energy, generic scene archetypes, lighting, color and composition while generating new scenes for the uploaded song and lyrics. Prompts explicitly forbid copying the reference's people, dialogue, lyrics, logos, recognizable locations or exact shot sequence. Semantic analysis improves direction but does not guarantee pixel-identical camera geometry or reproduction of every reference shot.
