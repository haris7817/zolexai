# Music Video — the customer's prompt, enforced (9 Sep 2026)

*Client report of the same day, in one line: a customer asked for a Beverly
Hills penthouse full of hay bales, a hot-pink Lamborghini towing a tractor
down a country road and a Nashville barn turned neon nightclub — and received
a generic country-pop performance video. Their diagnosis was right.*

## Where the prompt went

Nowhere. `zolex_music_worker.director.expand_direction` reads the customer's
prompt to match a keyword→genre table (`love`, `rap`, `sad`, …) and takes
**every** location, palette, insert and camera move from that profile. The
text is stored as `treatment["original_direction"]` and never read again.
`compile_shot_prompts` then builds each render prompt from tables — family,
location, lighting, action, camera — plus identity and lyric direction. Not
one noun the customer wrote reaches a shot. Their package; their fault; the
one thing about it that is ours is that we ran it.

The "automatic eight-second scene changes" they mention are the same thing
seen from the other side: `music_video_shot_target_seconds: 8.0`, cut on
transients, with no mode that does anything else.

## The seam, not the package

The package is vendored verbatim and never edited (`worker/adapters/
music_video.py`). Its `worker.py` binds the stages it runs by name, so
`worker/musicvideo/enforce.py` rebinds five names on that module to wrappers
that consult a per-job context — and with no context set, every wrapper calls
the original untouched. That is the kill switch (`execution.music_video_enforce:
false`), and it is what every pre-existing test exercises.

| wrapper | what it adds |
|---|---|
| `expand_direction` | the treatment's locations become the customer's, in story order; the brief rides along |
| `build_plan` | single-shot: one family, one place, every boundary `continue` |
| `generate_anchors` | composes and **validates first**, so a plan missing a mandatory element is refused before the first still; a dry run stops here |
| `compile_shot_prompts` | the customer's story leads each prompt: theme, this shot's place and beat(s), its props, their prohibitions; the package's identity / camera / audio / continuity follow unchanged |
| `_render_all_shots` | single-shot: sequential, each window anchored on the previous accepted clip's last frame |

The validate-before-anchors placement matters: the package compiles its
prompts *after* its anchors, so the client's "no expensive rendering until
coverage is 100%" rule could only be honoured by composing early and letting
the package's own later call be idempotent.

## The brief (`worker/musicvideo/brief.py`)

Two extractors, one contract. The hosted writer (the Auto Dialogue chain)
reads the prompt into JSON; behind it a regex extractor treats every sentence
as a beat, "in/at/to a …" phrases as places, and a short noun list as props.
The writer's brief is merged **over** the heuristic's, never instead of it:
anything the heuristic found in the customer's words that the writer dropped
is put back, because a dropped noun is the whole fault.

Concrete things the customer named are **mandatory** (their rule). Coverage is
their exact words, case-insensitive, no fuzzy matching — "a barn" does not
count for "a Nashville barn transformed into a neon nightclub". Measured on
the client's own prompt: 27 mandatory items, 100% coverage over 25 shots,
beats in chronological order.

A bug worth recording from the test suite: with fewer shots than beats the
first draft dropped the trailing beats, coverage failed, and a short song with
a detailed prompt would have been *refused*. Beats are now sliced across
shots — several per shot when there are more beats than shots.

## Lip-sync routing

Their rule — *a visible singer during active vocals must render
audio-conditioned* — was already true of every shot: both render backends
pass the song and the shot's exact start time (`scripts/mv_render.py`,
`worker/comfy/ltx_a2v.py`). There is no text-only fast route here to
mis-route to.

What was **not** true is that the prompt agreed. `apply_lyric_directions`
sets `vocals_present` only for three performance families, so a performer in
a `wide_movement` shot under a sung line was told *"the mouth remains
naturally at rest"* while the track sang. The `compile_shot_prompts` wrapper
now marks any shot with a performer **and** active lyric segments as singing
— their words: never classify a singer-facing shot as B-roll. The remaining
lever they describe, a dedicated per-face lip-sync pass with a mouth-motion
score, needs the lip-sync service this platform does not have.

## Locked single shot

`wants_single_shot` reads "one continuous shot", "no cuts", "static camera",
"same framing" and the like; `execution.music_video_shot_mode: single_shot`
forces it. The song is still rendered in windows — the model renders one at a
time — but each window after the first is anchored on the previous window's
last frame, every shot is the same family in the same place, and the prompt
says so. Unmeasured on a GPU as of this writing.

## Dry run

`execution.music_video_dry_run: true`: analysis, treatment, plan, prompts and
validation run; `prompt-trace.json` is written; the job stops at the anchor
boundary with no GPU time spent and reports as an operator error naming the
trace. (`plan_only` in the package alone stops *after* anchors.)

## Traceability

`<job>/music-video/<job_id>/prompt-trace.json`: the original prompt, the
brief and its source, and per shot the planned prompt, the final prompt, its
SHA-256, the audio window, the performer ids, the route
(`audio_conditioned_a2v`) and whether lip-sync was required. The client asked
to be able to prove the prompt reached every stage; this is that proof.

## Reference-video link

The package already fetches (yt-dlp) and analyses a link
(`reference_fetch.py`, `vision_reference_analyzer.py`, `reference_style.py`)
and folds its camera language and cut rhythm into the treatment — originality
policy: grammar only, never people, places or shot order. Nothing forwarded a
URL to it. Now: `reference_video_url` on the API (HTTPS, YouTube/Vimeo, same
host list as the worker's), `settings.reference_video` on Music Video, a box
in the panel, `build_request` forwards it, `build_config` points the fetcher
at yt-dlp in the LTX venv. With no vision model the analyser uses its
built-in metrics — rhythm, palette, motion — which is what the treatment
consumes; Qwen2.5-VL is opt-in behind `music_video_reference_vision` and its
weights are not on the node.

`yt-dlp 2026.08.19` was installed into `/workspace/ltx2-benchmark/.venv` on
the node the same day (the venv is `uv`-managed: `uv pip install --python …`).

## Extend Video

Separate report, same day: "no coherent option of audio". `continuation.py`
writes a **new** soundtrack per pass from the prompt and edge-fades it onto
the source's at the seam — not the source's music continuing — and
`extend-video.yaml` declared no `sound` setting at all, so a customer had no
way to say "no new audio". `settings.sound: true` now; off drops the audio
stream from the delivered file, the same switch with the same meaning every
other video tool has.

## Measured, same day

The first Music Video ever to complete on this platform — job `98a0435e`,
25 shots, 3:15 song — finished at 3840×2160, 194.67 s, 945 MB, in 24 min
wall on one GPU (~50 s a shot, steady). That was on the *previous* code: the
job that proved the 422 fix, before any of the above ran. The enforcement
layer has not yet rendered end to end; the plan and prompts are proven in the
suite, the picture is not.
