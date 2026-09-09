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

## Measured on the client-test node, 9 Sep 2026 (Spanish, "romantic latin pop")

| length | planned | measured sung (stem) | recall (heard) | rhyme | takes | wall |
|---|---|---|---|---|---|---|
| 1 min | 90% | **76–84%**, vocal enters at ~9–11 s every take | 71% | 100% | 4 | 40–50 s |
| 2 min | 90% | **92%** | 85% | 100% | 2 | 40 s |
| 3 min | 90% | **88%** (takes 87–90%) | 72% | 100% | 4 | 110 s |
| 5 min | 92% | **87–90%** across takes | 58% | 100% | 4 | 177 s |

Three things learned the hard way, all fixed the same day:

* **Cerebras withdrew `gemma-4-31b` for this key** (404 `model_not_found`
  while `/v1/models` still lists it). The node now runs `gpt-oss-120b`
  for lyrics and Director/Auto Dialogue (`CEREBRAS_LYRICS_MODEL`,
  `CEREBRAS_DIRECTOR_MODEL`). That model spent its whole output budget
  reasoning about Spanish rhymes and returned empty messages until
  `reasoning_effort: low` was sent (`cerebras_lyrics_reasoning_effort`)
  and the reserve raised to 2500 tokens.
* **The writer chain hid the repair method**, so no rhyme repair ever ran
  in production; and the model echoed the repair prompt's label back as a
  one-line "[verse] LYRIC SHEET:" that collapsed every verse of a 3-minute
  song. Repairs now delegate through the chain and a rewrite whose shape
  differs from the original is discarded.
* **"auto" rhyme scheme is AABB.** ABAB failed the strict validator after
  three Spanish drafts and repairs; couplets pass every time.

The one-minute song is the honest limit: the model opens with a ~10 s intro
whatever the brief says (a sung `[intro]` section, denser lines and four
takes were all tried), and ten seconds is a sixth of the song. Under the
specification's `fail` policy a 1-minute request will usually be refused
with `VOCAL_COVERAGE_BELOW_90`; `music_verify_policy: deliver_best` ships
the best take with its numbers instead. Two minutes and up pass at the
line, with the retry doing real work.

Recall counts a line heard with some words changed (Whisper on sung
vocals: 40% exact on takes where every line was audibly present); the
exact-match share is in the report. Threshold 0.5.

## 10 Sep 2026 — the client's three corrections, enforced in the backend

Their findings on a delivered 2-minute Spanish song: singing from 0:12 to
1:35, ~25 s instrumental tail, ~70% sung; some lines unnatural; rhymes
inconsistent; and a reference-audio song at 136 BPM A major against a
92 BPM B major reference. Three mandatory fixes, all landed the same day:

**1. Coverage, intro, outro and breaks — cut, not asked.** The model opens
with a ~10 s intro and ends with a tail whatever the brief says, so
`worker/music/trim.py` generates a take ~22 s + 5% longer than the song
and cuts the requested length starting 2.5 s before the first sung note
(stem spans; transcript spans without a separator), fading at the cuts.
`verify.py` gained an `INSTRUMENTAL_BREAK_TOO_LONG` gate
(`music_max_break_seconds`, 3 s). Measured live afterwards: 2-min Spanish
**93.6% sung, intro 2.5 s, longest break 2.4 s, first take**. Overshoot as
a ratio (1.35) made the model stretch the sheet across the longer take and
the cut lost the last chorus — hence the fixed-seconds form.

**2. Reference audio conditions the model.** The dormant `reference_audio`
payload key was never read by the service; the real fields are
`reference_audio_path` (under the system temp dir), `task_type: text2music`
and `audio_cover_strength` (0.3 — style transfer; 1.0 would be a cover).
`worker/music/keys.py` measures BPM (onset + envelope autocorrelation) and
key (Krumhansl–Schmuckler on a numpy chroma) for the reference and for the
finished song; the reference's BPM and key are sent as `bpm`/`key_scale`;
`compare_to_reference` scores tempo (3% tolerance, octave-aware), key,
energy-curve shape and sung density into a similarity, and a mismatch is a
retry reason (`REFERENCE_MISMATCH`). The mandated log record
(`reference_audio_loaded`, `reference_embedding_created`,
`reference_conditioning_applied`, `reference_duration`, `target_bpm`,
`target_key`) is written as `music_reference_conditioning`, and a provider
that cannot condition refuses the job (`REFERENCE_CONDITIONING_FAILED`).
Measured live: reference 71 BPM F major → song **71 BPM**, key heard as F
minor — the parallel-mode ambiguity chroma cannot settle on a dense mix, so
same-tonic mode disagreement scores partial rather than failing.

**3. Lyric quality and flow engine** (`worker/music/quality.py`). Order is
now concept → narrative outline → lyrics → rhyme/meter → judge → music →
transcription. The outline (theme, narrator, addressee, emotion, conflict,
development, final message, hook) goes into the writer's brief; a
native-editor judge scores coherence and grammar and names the lines with
real errors, which go back as targeted repairs (twice), with the client's
floors (0.90 / 0.95) as the rewrite triggers. Refusal only below a hard
floor (0.5): the judge is a language model's opinion and scored 0.15
grammar on plain Spanish before its rubric was narrowed to errors, not
taste. Rhyme groups are also checked for syllable parity (±1) —
reported and repaired, never a refusal, and always after rhyme.
The writer chain now delegates `ask_json` and `rewrite_lines` (without
which the judge was silently unmeasured), and an empty answer at the
length limit retries once with a doubled budget.

Live after all of it, reference-conditioned 2-min Spanish: 93% sung,
intro 2.5 s, no breaks, recall 100%, rhyme 100%, coherence 0.94, grammar
0.89, three takes.
