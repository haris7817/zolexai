# The golden benchmark pack

**Date:** 24 August 2026 · **Branch:** `dual-engine-benchmark-prep`

Frozen inputs so a comparison run in October means the same thing as one run
in August. The pack itself lives at [`benchmarks/`](../../benchmarks/); this
document is what it is for and what state it is in.

The premise is narrow and worth saying plainly: **two runs that used different
inputs are not two results.** A prompt quietly reworded between sessions, or a
source image someone re-exported, produces a table that looks comparable and
is not — and nothing in the output would say which.

---

## 1. What is frozen

**Prompts and cases.** `benchmarks/frozen/cases.json` records every case's
prompt hash against its `prompt_version`, plus its workflow, parameters,
execution keys, strategies and repeat count. The cases live in code
(`worker/providers/benchmark.py`), so git already versions them — but git
tracks that a *line* changed, not that a *benchmark* changed. The frozen file
closes that gap: a changed prompt at an unchanged version fails
`tests/test_golden_pack.py`. Editing a benchmark prompt is legitimate; doing
it silently is not.

**Media.** `benchmarks/assets.manifest.json` declares every asset with its
SHA256, geometry, purpose, the cases that need it, and its provenance. The
media itself is **not** in git — binaries do not belong in this repository,
and a benchmark asset in git history is a licensing problem that outlives the
benchmark. The files travel out of band and sit under `benchmarks/assets/`,
which is ignored.

That is exactly why the hashes matter. The manifest is the only thing proving
the file on the GPU node is the file the last run used.

## 2. Current state — 19 assets, all pending

```text
uv run python apps/worker/scripts/golden_pack.py --status
19 assets · 19 pending acquisition · 0 blocking
41 cases · 100 cells · 406 runs with repeats
```

| Group | Assets | For |
|---|---|---|
| Reference person | `person-closeup`, `person-waistup`, `person-fullbody` | D1–D5, J1 — **one identity across all three framings** |
| Source video | `src-closeup-speaking`, `src-waistup-speaking`, `src-fullbody-movement`, `src-fast-movement`, `src-moving-camera`, `src-hard-cut`, `src-multiple-characters` | C and D groups |
| Image to video | `i2v-portrait`, `i2v-fullbody`, `i2v-landscape`, `i2v-difficult-lighting`, `i2v-multiple-subjects` | B group |
| Music video | `singer-reference`, `benchmark-song`, `benchmark-song-lyrics` | E group, J1 |
| Extend | `extend-source` | I1 |

**Pending is a readiness state, not a failure.** The verifier separates it
from the two states that invalidate a comparison — a file that is *missing*
or *different* — and only those stop a run.

## 3. Acquisition requirements

**Legality first.** No commercial song, no scraped video. Prefer media we
create ourselves or generate with our own models; record the source and the
right to use it in the manifest before marking anything `acquired`. An
unlicensed asset outlives the benchmark it was collected for.

**One identity across the reference-person framings.** The D group measures
identity retention, so the identity has to be the constant. Three framings of
the same person, neutral expression, even lighting, face unobstructed.

**The song is the long pole.** Generate it with our own ACE-Step service: the
rights are then unambiguous *and* the lyric sheet is known rather than
transcribed. Whisper is not a source of truth here — it called an English pop
song Khmer once, and vocal coverage is measured from a Demucs stem. For
lip-sync to be measurable at all the track needs a clear lead vocal, at least
one instrumental gap of two seconds or more, a verse and a chorus, and
identifiable vocal onset and offset moments. Freeze the lyrics, language and
section timings beside the audio.

**Subject consent** for any real person appearing in a source clip, recorded
in the provenance field. (The consent gate for the person-replacement product
feature remains separately open.)

## 4. Using the pack

```bash
uv run python apps/worker/scripts/golden_pack.py --status   # what exists, what drifted
uv run python apps/worker/scripts/golden_pack.py --verify   # GPU-day gate; non-zero = stop
uv run python apps/worker/scripts/golden_pack.py --hash <file>   # after shooting an asset
uv run python apps/worker/scripts/golden_pack.py --freeze   # after a deliberate case change
```

`--verify` is the gate that runs before any comparison. It exits non-zero on a
hash mismatch, a missing acquired file, or case drift; a pending asset is
reported and does not fail it.

## 5. The result manifest

`result_skeleton()` ships **empty** — a skeleton carrying plausible numbers is
the easiest way for a fabricated benchmark to reach a decision. Each
`RunRecord` carries enough to reproduce it months later: case id, strategy,
provider, pipeline, model and runtime revision, GPU, prompt version, the asset
hashes actually used, seed, requested and actual duration, sections, every
timing leg (including both halves of a hybrid and the model switch), peak VRAM
and host RAM, success with a failure class, retries, scores, and lip-sync
level.

**Seeds are per-provider.** Record them for reproducibility within an engine;
never present one as a controlled variable across two engines.

**Failure classes** are the ones that lead to different actions:
`oom`, `cublas`, `model_load`, `decode`, `duration_mismatch`, `audio_mismatch`,
`identity_failure`, `reference_failure`, `seam_failure`, `prompt_failure`,
`corrupt_output`, `timeout`, `unknown`. Treating them as one number hides the
only thing the number is for — and `corrupt_output` is not hypothetical here:
this project has shipped video that passed ffprobe and was solid green.

## 6. What the pack does not cover

- **`expected/` is empty.** Reference outputs are added after the first GPU
  session; there is nothing honest to put there yet.
- Scoring is human. The card is defined and weighted, but somebody watches the
  output — no automated quality metric is claimed.
- Lip-sync level C (phoneme accuracy) has never been demonstrated on any path
  here and is not expected to be scored from these assets without a dedicated
  probe.
