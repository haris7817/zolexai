# The golden benchmark pack

Frozen inputs for the LTX / H3 / hybrid comparison. The point of freezing is
narrow and worth stating plainly: **two runs that used different inputs are
not two results.** A prompt quietly reworded between Tuesday and Friday, or a
source image re-exported at a different quality, produces a table that looks
comparable and is not.

```text
benchmarks/
  README.md              this file
  assets.manifest.json   every media asset: hash, geometry, provenance, cases
  frozen/cases.json      every case: prompt hash, version, params, strategies
  assets/                the media itself — NOT in git
  expected/              reference outputs, added after the first GPU session
```

## What is committed and what is not

The **manifest** is committed; the **media** is not. Binaries do not belong in
this repository, and a benchmark asset that lives in git history forever is a
licensing problem waiting to happen. `assets/` is ignored; the files sit
beside the manifest on whichever machine is running the comparison.

That is exactly why the hashes matter — the media travels out of band, so the
manifest is the only thing proving the file on the GPU node is the file the
last run used.

## Using it

```bash
# what exists, what is pending, whether the cases have drifted
uv run python apps/worker/scripts/golden_pack.py --status

# the GPU-day gate. Non-zero exit means DO NOT start a comparison
uv run python apps/worker/scripts/golden_pack.py --verify

# after shooting an asset: hash it, paste into the manifest, mark acquired
uv run python apps/worker/scripts/golden_pack.py --hash assets/i2v/portrait.png

# after deliberately changing a case (and bumping its prompt_version)
uv run python apps/worker/scripts/golden_pack.py --freeze
```

## Rules

1. **A hash mismatch stops the comparison.** Not a warning. We cannot compare
   LTX from image A against H3 from image A-as-edited and learn anything.
2. **Provenance before acquisition.** An asset may not be marked `acquired`
   without a recorded source and right to use. Prefer media we create
   ourselves or generate with our own models; never a commercial track pulled
   off the internet — an unlicensed song outlives the benchmark it was for.
3. **One identity across the reference-person framings.** The D group measures
   identity retention, so the identity has to be the constant.
4. **Bump `prompt_version` when a prompt changes.** The frozen pack fails the
   test otherwise, which is the intended behaviour: editing a benchmark prompt
   is fine, doing it silently is not.
5. **Do not re-encode an acquired asset.** It changes the hash and silently
   changes the input.
6. **Seeds are per-provider.** Record them for reproducibility within an
   engine; never present one as a controlled variable across two engines.

## The song

`benchmark-song.wav` should be generated with our own ACE-Step service, for
two reasons: the rights are unambiguous, and the lyric sheet is *known* rather
than transcribed. Whisper is not a source of truth here — it called an English
pop song Khmer once, and vocal coverage is measured from a Demucs stem.

Requirements that make lip-sync measurable at all: a clear lead vocal, at
least one instrumental gap of two seconds or more, a verse and a chorus, and
identifiable vocal onset and offset moments. Freeze the lyrics, language and
section timings beside the audio.
