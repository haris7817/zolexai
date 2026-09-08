# Automatic lyric-to-scene workflow

## User experience

The backend only needs the stored audio path, a job ID, and the user's short direction:

```json
{
  "job_id": "mv-rap-0001",
  "audio": "/srv/zolexai/uploads/song.wav",
  "prompt": "make me a video according to the lyrics of the song",
  "formats": ["16:9"]
}
```

The phrase `according to the lyrics` activates the lyric workflow automatically. The UI does not need a separate lyrics field. The optional `lyrics` field can override transcription when the user already has authoritative lyrics; it accepts plain lines, LRC, or SRT. Use `lyric_mode=always` if the product should transcribe every music-video request regardless of wording.

## Processing order

1. Decode the full uploaded song and establish the exact 24 FPS master timeline.
2. Analyze energy, transients, rough tempo, and structural sections.
3. Transcribe the mixed song with Faster-Whisper `large-v3`, preserving segment timestamps and detected language.
4. Move candidate cuts toward both musical transients and lyric boundaries while retaining 2–7 second scenes.
5. Attach every overlapping lyric segment to its exact scene.
6. Interpret each lyric window into a concrete visual theme and action. Repeated lines receive a recurring chorus motif. Sensitive lines use symbolic, non-graphic imagery.
7. Create identity-conditioned scene anchors and self-contained LTX prompts.
8. Render and retry scenes independently, restore editorial order at the LTX working size, attach the full original audio, GPU-upscale the picture once to 4K while copying the audio, and run delivery QA.

If the user explicitly asks for lyric-based scenes and no usable transcript is produced, the job fails before GPU scene rendering. It does not silently return an unrelated mood video.

## Production configuration

Install:

```bash
python -m pip install '.[api,lyrics]'
```

Configure:

```text
ZOLEX_TRANSCRIPTION_BACKEND=faster_whisper
ZOLEX_TRANSCRIPTION_MODEL=large-v3
ZOLEX_TRANSCRIPTION_DEVICE=cuda
ZOLEX_TRANSCRIPTION_COMPUTE_TYPE=float16
ZOLEX_TRANSCRIPTION_DOWNLOAD_ROOT=/models/faster-whisper
```

For CPU transcription, set `ZOLEX_TRANSCRIPTION_DEVICE=cpu` and use a compatible compute type such as `int8`. GPU transcription is normally faster. The selected model is downloaded by Faster-Whisper on first use unless it already exists in the configured cache or the model setting points to a local directory.

Music vocals can be harder to transcribe than speech, especially under loud instrumentation, effects, ad-libs, or overlapping voices. For releases where every lyric must be exact, pass authoritative lyrics from the artist or connect a production music-transcription service through `ZOLEX_TRANSCRIPTION_COMMAND_JSON`. The scene planner still uses the same validated timestamp contract.

## Optional narrative director

The built-in interpreter is deterministic and works without another model. It recognizes common themes including love, loss, success, street identity, celebration, faith, memory, and escape, while the active lyric text gives the LTX text encoder the specific meaning of each scene.

For deeper metaphors and a stronger whole-song story, connect a local LLM service through `ZOLEX_LYRIC_DIRECTOR_COMMAND_JSON`. The command receives the full transcript, song treatment, sections, scene timing, and assigned performers. It may return a refined `lyric_theme` and `visual_direction` for each known shot ID, but it cannot change the timeline, add performers, or replace the original song.

## 4K delivery

| Format | Scene working size | Final required size |
| --- | ---: | ---: |
| 16:9 | 1280×704 | 3840×2160 |
| 9:16 | 704×1280 | 2160×3840 |
| 1:1 | 1024×1024 | 2160×2160 |

The working sizes preserve scene-render speed and remain divisible by 32. The assembler first creates one working master with the aligned song. A separate CUDA stage center-crops the small LTX alignment padding, scales the picture with Lanczos, encodes it with NVENC, and stream-copies the audio. Final QA verifies resolution, frame count, frame rate, audio presence, and duration.
