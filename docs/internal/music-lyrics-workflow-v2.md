# Music Lyrics Workflow v2.0 — lyrics first, validated, then sung (9 Sep 2026)

*The client's specification of the same name, implemented on the Music
tool. Their one-line requirement: at least 90% of the finished song must
carry sung lyrics, verified from the audio, in any language, with strict
rhyme — and no unvalidated draft ever reaches the music model.*

## Where it sits

The Music adapter (`worker/adapters/music.py`) already did plan → write →
sing. What it did not do was hold the sheet to a timeline, judge a rhyme,
refuse a thin draft, or listen to the result. v2 adds exactly those, as a
layer between the plan and the provider, and leaves the provider seam and
the writer chain as they were:

```
plan_song ─► build_blueprint ─► write (Cerebras → template) ─► rhyme + gates
      │              │                                             │
      │         reference.py                               dry run stops here
      ▼              ▼                                             ▼
   ACE-Step ◄── production caption + expanded sheet ◄──── PreparedSong
      │
      ▼
   verify_song (Demucs stem coverage + Whisper recall) ─► retry / deliver / fail
      │
      ▼
   AdapterResult.report ─► API `result` ─► result page
```

Module by module, under `worker/music/`:

| module | what it decides |
|---|---|
| `blueprint.py` | every second assigned; wordless parts squeezed to 10%; a line target per section at **4 s/line**; the chorus written once, placed everywhere |
| `syllables.py` | per-script syllable estimates and filler detection ("oh oh yeah", "(humming)") |
| `rhyme.py` | language-aware rhyme keys (stress rules for es/it/pt, silent finals for fr, hangul decomposition…), scheme labels (AABB/ABAB/AAAA), a verdict per group with a **confidence** |
| `timing.py` | the sheet on the timeline: LRC/SRT/JSON, planned coverage (a line is credited `syllables / slow-delivery-rate` seconds, capped at its slot) |
| `reference.py` | upload or link (yt-dlp): BPM from onset autocorrelation, energy curve, section changes, sung fraction from the stem, cadence from the transcript; the transcript is kept private for the originality check only |
| `gates.py` | the pre-audio checklist and its failure codes |
| `workflow.py` | the orchestrator: options, reference, blueprint, write/repair loop, files |
| `transcribe.py` / `verify.py` | word-level Whisper (the music-video extra's model) aligned line by line; stem coverage; the post-audio verdict |
| `report.py` | the full job report (`lyrics-report.json`, one log record) and the bounded customer `result` |

## The numbers, and where they came from

* **4 s/line.** The 21 Aug matrix (`_SECONDS_PER_LINE` in `lyrics.py`)
  measured ACE-Step sung coverage at 16 / 8 / 5 / 3.5 s per line across
  three durations. Ninety percent arrives only around 3.5–4 s/line; 8 s
  (the v1 target) gives 73–87%. v2 plans at 4 and the model's own
  arrangement does the rest — the retry loop covers the run-to-run
  variance the matrix also showed (52.8% vs 87.7% on the same sheet).
* **Coverage credit at a slow delivery** (`SYLLABLES_PER_SECOND`: pop 1.8,
  rap 3.5, ballad 1.4). A three-word line is not eight seconds of singing
  whatever its slot is; but a singer stretches a short line, so the floor
  rate is deliberately slow. It catches the real defect — two lines over
  thirty seconds — without calling every short lyric "half instrumental".
* **Breaths are not pauses.** Line timing is contiguous. Deducting a
  0.35 s breath per line made 90% unreachable by construction.
* **The rhyme validator is heuristic and says so.** Every report carries
  `method` and `confidence`. A generated song is *failed* over rhyme only
  in strict mode where the confidence is `high` (en, es, it, pt, fr, ko).
  For Arabic script, Hindi, Japanese and Chinese the verdict is recorded
  and warned about, never enforced — refusing a customer's Urdu song on a
  last-two-letters comparison would be a worse product than the one the
  workflow replaces. A customer's own sheet is never rewritten and never
  refused over rhyme or density at all; the report shows what was found.

## Failure codes

Preflight (no audio made): `REFERENCE_UNAVAILABLE`, `REFERENCE_AUDIO_INVALID`,
`BLUEPRINT_VALIDATION_FAILED`, `VOCAL_COVERAGE_BELOW_90`, `FILLER_ABOVE_LIMIT`,
`RHYME_VALIDATION_FAILED`, `LANGUAGE_MISMATCH`, `REFERENCE_SIMILARITY_TOO_HIGH`.
After audio: `VOCAL_COVERAGE_BELOW_90`, `LYRIC_RECALL_TOO_LOW`,
`OUTPUT_DURATION_MISMATCH`. They appear as `[CODE]` at the head of the
job's internal detail and as `failure_code` in the report; the customer
sees copy per code (`MusicAdapter._failure_for`).

## Verification and the retry

After assembly the track is measured twice: sung fraction from the Demucs
stem (`vocal_separator_python`, language-proof) and line recall from a
Whisper transcript aligned to the sheet (`music_verify_transcription`).
Either tool missing → that measure is `unmeasured` and does not fail the
job; the report says so. A failing take is re-recorded with a new seed and
a firmer production brief (the sheet is kept — it already passed), up to
`music_verify_max_retries` (2). Still failing: `music_verify_policy`
decides — `fail` (the specification; default) or `deliver_best`, which
ships the best take with its numbers in the result.

## What the customer gets

On the job's `result` (API `GenerationJobPublic.result`, 64 KB bound):
the lyrics as written, LRC, SRT, timed JSON, planned and measured
coverage, the coverage method, lyric recall, rhyme pass rate and
validator confidence, whether a reference was used, the originality
result, retries and warnings. The result page shows the sheet, the
numbers and Save .txt/.lrc/.srt. On the worker, `<workspace>/lyrics/`
holds every artefact including `lyrics-report.json` (the spec's full
logging list) and `reference-profile.json` (high-level attributes only).

## Request surface

`settings.lyrics_workflow: true` on `music.yaml` unlocks `rhyme_scheme`,
`rhyme_mode`, `point_of_view`, `clean_mode`, `reference_audio_url`,
`dry_run` (same 422 policy as lyrics elsewhere) and an optional
`reference_audio` upload input. `dry_run` is the specification's
`/music-lyrics/plan?dry_run=true`: the writer and the reference tools live
on the worker, so it runs as a job that stops at the anchor and reports
the plan's numbers in its message.

## Deployment

* API: alembic migration `20260909_1200_generation_result` adds
  `generation_jobs.result` (nullable JSONB). **Run `alembic upgrade head`
  before the new API image serves traffic** — the ORM selects the column.
* Worker: `music_lyrics_workflow: v2` is the default; `v1` per node or per
  job (`execution.music_lyrics_workflow`). Recall needs `faster-whisper`
  (the `music-video` extra, already on the node); coverage needs the
  Demucs venv (`VOCAL_SEPARATOR_PYTHON`, already on the node); links need
  yt-dlp in the LTX venv (installed 9 Sep for Music Video).
* Not yet measured on a GPU as of this writing: the first real v2 song's
  measured coverage, the Whisper recall on sung vocals in each offered
  language, and how often the retry fires. `lyrics-report.json` of the
  first client jobs is the thing to read.
