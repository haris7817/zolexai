# ZolexAI Music Video Worker v1.8.0 — as delivered (8 Sep 2026)

The client's music-video orchestrator, delivered as
`ZolexAI-Music-Video-Worker-v1.8.0.zip`:

```
sha256  5c512aecd3b28bb440f26ad853928fb38592666a4faf6d70d70d2d4f4dc69ead
```

verified on receipt against the digest the client quoted. Their own note
with the delivery: bands of up to five people, a separate identity and role
per member, up to four reference images per person, rotating solo shots,
duet and smaller group shots, full-band wides and climax scenes,
instrument-specific actions, prompts that prevent face blending / swapping /
duplication, a five-person request example, 33 automated tests.

## What is where

| in the ZIP | in this repository | edited? |
| --- | --- | --- |
| `src/zolex_music_worker/*.py` | `apps/worker/zolex_music_worker/` — the package the worker imports | **never** (pinned byte-for-byte by `apps/worker/tests/test_music_video_worker.py`) |
| `README.md`, `VERIFICATION.md`, `docs/`, `examples/`, `tests/`, `config/`, `scripts/`, `systemd/`, `pyproject.toml`, `.env.example`, `.gitignore` | this directory | no |
| `dist/zolexai_music_video_worker-1.8.0-py3-none-any.whl` | not kept — a build of the same source | — |

Their test suite was run unchanged on the GPU node (Python 3.12.3, FFmpeg
6.1.1) on 8 Sep 2026: `Ran 33 tests in 2.928s — OK`.

## How it is used

The platform runs the package as delivered and supplies what its README
leaves to the deployer (the "Backend integration checklist"): the anchor-
image generator, the render backend, the transcriber and the 4K finisher.
That side lives in `apps/worker/worker/musicvideo/`,
`apps/worker/worker/adapters/music_video.py` and
`apps/worker/scripts/mv_anchor.py` / `mv_render.py`; the full account is
`docs/internal/music-video-worker.md`.
