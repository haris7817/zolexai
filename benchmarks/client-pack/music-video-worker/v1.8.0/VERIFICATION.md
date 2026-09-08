# Verification record

Validation date: 2026-09-07

## Automated suite

Command:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Result: 33 tests passed.

Coverage includes request validation, path-safe job IDs, short-prompt defaults, automatic lyric-request detection, supplied LRC and SRT parsing, repeated-line detection, timestamp-aligned lyric scene direction, local and URL reference-video aliases, mutual-exclusion and HTTPS validation, allowlisted-host enforcement, IP-address blocking, a complete command-fetched URL-reference job, real FFmpeg reference sampling and cut detection, contact-sheet generation, technical style profiling, local-vision JSON validation and profile integration, shot-scale/angle prompt propagation, reference-guided pacing, originality instructions in every scene prompt, five-performer acceptance and automatic five-person visibility, six-performer rejection, rotating band solos, full-band ensemble scenes, distinct multi-identity prompt protections and role labels, deterministic frame-exact planning, a 7,200-frame five-minute timeline, landscape and portrait LTX/4K plans, shell-free command templating, fixed GPU lanes, parallel analysis and rendering, five-minute capacity calculations, working-master assembly, 3840×2160 CPU finishing, audio stream copy, final delivery QA, resume, the CUDA command contract, exact installer-to-runtime model path mapping, forbidden root-target validation, and a complete simulated authenticated model download with size/SHA verification, manifest creation and credential exclusion.

## Five-minute planning test

- Input timeline: synthetic 300-second music analysis.
- Output timeline: exactly 7,200 frames at 24 FPS.
- Planned scenes: 75 for the synthetic two-second transient grid.
- Coverage: no gaps or overlaps.
- Maximum accepted source duration: 300 seconds.
- Render design: independent scenes assigned to fixed sequential GPU lanes, then restored to editorial order.

## Reference-file planning smoke test

Input: supplied `50877.mp4` reference file.

- Decoded audio duration: 181.7890417 seconds.
- Aligned timeline: 4,363 frames at 24 FPS, or 181.7916667 seconds.
- Planned scenes: 41.
- Average scene: 4.434 seconds.
- Shortest scene: 4.333 seconds.
- Longest scene: 5.125 seconds.
- Timeline coverage: exactly 4,363 frames with no gaps or overlaps.
- Renderer manifest: exact `id`, `start_seconds`, `duration_seconds`, `prompt`, and relative `image` fields.

## Package verification

- Source compiles with Python 3.12.
- Wheel builds without network access.
- Wheel installs in a clean virtual environment.
- Installed `zolex-music-video --version` returns `1.8.0`.
- Wheel metadata contains the `reference-ai` extra for Transformers, Accelerate and Qwen-VL utilities and the `models` extra for Hugging Face Hub/Xet downloads.
- FFmpeg 6.1.1 and FFprobe 6.1.1 passed the mock end-to-end run.
- The mock end-to-end working master was verified at 1280×704 with audio. Its final CPU-smoke delivery was verified at 3840×2160, 24 FPS, with the exact 48-frame timeline and audio present.

## Deployment test still required

The package intentionally does not duplicate approximately 91 GB of model weights inside the ZIP. LTX-2.5 is gated and requires the deployer's own acceptance of its community license and authenticated account before download. Therefore a live official-weight download, real LTX-2.5 GPU shot, mixed-music Faster-Whisper transcription, live Qwen visual inference, third-party reference-link fetch, and CUDA/NVENC 4K pass were not generated in this environment. The complete installer was exercised against a local simulated Hub contract, including all exact LTX filenames, pinned revision handling, file-size checks, LFS SHA-256 checks, manifest generation and credential exclusion. The developer must install `.[api,lyrics,reference-links,reference-ai,models]`, install a compatible CUDA PyTorch build, authenticate after accepting the LTX terms, run `zolex-music-video install-models --models-root /models`, and complete the model/GPU smoke tests.
