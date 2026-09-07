# Automatic dialogue — the client's pack, and what runs from it

*7 Sep 2026. Source: `LTX25_Dialogue-2.zip` (`ltx25_auto_dialogue_backend`, 6
files, 1,347 lines). Code: `apps/worker/worker/dialogue/`, commits `a5b3458`
and `d3aa99c`.*

## What the client sent

Two ideas in one package.

**The first** is automatic dialogue: when the scene contains somebody who
could speak and the customer wrote no words, write them a line. Their
`auto_dialogue.py` is a policy layer — two pattern sets deciding whether a
prompt already has dialogue or explicitly refuses it, an OpenAI-shaped call to
a language model, and a fail-open posture.

**The second** is coherent long-form: split 10/15/30 seconds into sections,
plan the whole story once, hand each section its predecessor's exact final
frame, render one master voice track for the entire video with a persistent
`voice_id`, concatenate the sections *without* their own audio, and apply
lip synchronisation to the stitched result.

Their own tests pass: 12 of 12, on a clean Python 3.13.

## What of it we can execute

The first idea, in full. The second, not at all — and not for want of trying.

Steps 2 and 7 of their required production order are "render the complete
dialogue timeline using ONE persistent `voice_id`" and "apply lip
synchronisation to the stitched video using the master voice track". This
platform has **no text-to-speech service** and **no lip-sync stage**;
`lip_sync` appears in this repository only as a scoring column in
`providers/benchmark.py`. ACE-Step sings songs and cannot hold a speaker's
identity.

The pack itself refuses a 10/15/30-second spoken job when no `voice_id` is
configured, rather than deliver a video whose voice changes halfway. That
guard is correct. Dropped in as-is, every multi-section talking job would
raise `ContinuityConfigurationError` instead of rendering.

**What we already had, and did not need from them.** Sectioning with exact
final-frame handoff and consecutive seeds is `worker/longform/chain.py`,
GPU-validated and driving five workflows. A single global plan per request —
with character appearance locks, timed dialogue carrying speaker and delivery,
character exits, continuity locks, a speech budget and a maximum-silence rule
— is `worker/director/`, 2,919 lines against their 639.

## What was actually missing

Director mode is wired into the **legacy `ltx` runtime**. The client-test
profile runs Text to Video, Image to Video and Extend on `ltx_comfy`, and the
new HD tool on `ltx_hd`, and neither of those had any dialogue planning at
all — they do deterministic prompt structuring and a soundscape clause and
nothing else. On the graphs the client is testing, nobody was deciding what
anyone says.

That is the hole this fills, and it needs no new service.

## How it works

`worker/longform/language.py` already decides the soundtrack's owner, on one
thing: whether the prompt contains quoted words.

| prompt | what `soundscape_clause` emits |
| --- | --- |
| no quoted words | "No one speaks. The only sounds are the ones the scene itself makes." |
| quoted words | "The only words anyone speaks are the quoted lines above … spoken a single time and not repeated" |

Every measurement behind that still holds — the narration leak, the repeated
line, the described-silence tail. What it never had was a way to get quoted
words into a prompt the customer wrote without any.

So `worker/dialogue/` writes them, and stops. The adapters take one line each
and nothing downstream knows it exists.

## Two departures from the pack, both on our measurements

**It writes a conversation, not one line.** Their code writes a single line
per section — right for the 8-second section it was built against. Nine GPU
renders on 18–19 Aug 2026 (`worker/director/plan.py`) say a 15-second clip
carrying two lines echoes its last word: dead air is not neutral, and the
model fills it by repeating itself or reading the caption aloud. Line count
and word budget therefore come from `target_spoken_lines` and `speech_budget`
— the functions that already carry that measurement — so they cannot drift
from it.

| length | lines asked for | word budget |
| --- | --- | --- |
| 8 s | 2 (max 4) | 11 |
| 15 s | 4 (max 7) | 25 |

**The HD path composes its own anti-repeat rule.** No soundscape clause runs
there, and a short line in a long clip loops without one — a customer got
"Never mess with the family" repeated for a full twenty seconds on 28 Aug.

## Where it applies, and where it refuses

Text to Video, Image to Video, Text to Video HD. Not Extend Video or Video to
Video, whose prompts describe a continuation or a restyle rather than a scene
to invent; not Music Video, which has a song; not Director mode, which plans
its own dialogue with locks this module knows nothing about.

It also stands back when: the feature is off, the prompt asks for silence, the
prompt already has dialogue *or a speech verb*, sound is off on either switch,
or the clip is under five seconds. Every refusal is logged with its reason.

**Failure is open.** A video whose dialogue could not be written is the video
the customer asked for, rendered from the prompt they wrote. This is the one
place the posture differs from Director mode, which fails the job — someone
who asked for a planned dialogue scene and got a silent one was answered with
a different request; someone who asked for a video was not.

## Validation

A written line is not accepted on trust. It must come from a speaker the model
itself declared, carry no nested quotes (which would end the quoted span early
and leave the model speaking half a line and reading the rest as prose), stay
inside the word budget, and not repeat — a repeated line arriving from the
writer rather than the renderer is no better.

## Configuration

| setting | default | notes |
| --- | --- | --- |
| `AUTO_DIALOGUE_ENABLED` | `false` | on for the client-test node |
| `AUTO_DIALOGUE_LOCAL_FALLBACK` | `true` | Gemma behind the hosted writer |
| `AUTO_DIALOGUE_TIMEOUT_SECONDS` | `30` | far under Director's 900 |
| `AUTO_DIALOGUE_MAX_TOKENS` | `1200` | includes reasoning headroom |
| `AUTO_DIALOGUE_TEMPERATURE` | `0.7` | the Director planner's value |

A job's own `auto_dialogue` parameter overrides the default either way. There
is no UI toggle yet; adding one needs the workflow YAML **and**
`catalog.server.ts`, which hand-projects the schema.

The writers are the ones the platform already uses: Cerebras first
(`CEREBRAS_API_KEY`, ~0.5 s), the local `gemma-4-e2b-it` checkpoint behind it
through `scripts/director_plan.py` unchanged — that script takes a system
prompt, a user prompt and a pair of markers, and nothing in it is specific to
a Director plan.

**Node state, 7 Sep 2026.** The GPU node had neither: no key, no checkpoint.
Both were installed — the key in `.env.gpu-worker`, the checkpoint at
`/workspace/ltx2-benchmark/models/gemma-4-e2b-it` (9.6 GB). Before that,
Director mode had no planner on that box either.

## Measured

Hosted writer, on the node, four scenes:

| scene | asked | written | time |
| --- | --- | --- | --- |
| taxi driver and passenger, 15 s | ~4 lines, 25 words | 4 lines, 15 words, 2 speakers | 0.7 s |
| chef and waiter, 8 s | ~2 lines, 11 words | 2 lines, 6 words, 2 speakers | 0.5 s |
| empty snow valley, 15 s | ~4 lines | **declined — no speaker** | 0.2 s |
| two engineers, 15 s | ~4 lines, 25 words | 4 lines, 21 words, 2 speakers | 0.4 s |

The declined landscape is the behaviour that matters most: the prompt is
returned exactly as written, and the video stays silent.

### One real HD render, transcribed

15 s, 1920x1080 with sound, **306.3 s wall** (auto dialogue 0.5 s of it).
Prompt deliberately of the shape that used to get "No one speaks" — two
mechanics in a garage, people present, no quoted words, no speech verb.

Whisper (`whisper-small.en`) on the finished file:

| planned | spoken | at |
| --- | --- | --- |
| "Hold it right there." | "Hold it steady right there" | 0.88 s |
| "Is that the leak?" | "Is it the fuel line yeah looks like a leak" | 3.92 s |
| "Yeah, the seal is shot." | "Yeah, it looks like a leak" | 9.56 s |
| "Got a spare part?" | "Want me to grab the wrench" | 12.68 s |

**Two verdicts, and they differ.**

*Coverage is right.* Four lines across fifteen seconds with no gap over three
seconds, in the right language, the right register and plainly about the scene
on screen. The pacing arithmetic borrowed from `director/plan.py` did its job.

*Adherence is not.* Only the first line survived. The rest are paraphrases,
and `"looks like a leak"` is spoken **twice** — the exact artefact the
anti-repeat sentence exists to prevent, produced anyway.

**The likely cause, untested.** All four lines reach the graph as one run-on
paragraph. Director mode — where this runtime was measured delivering lines
verbatim — does not do that: its compiler distributes lines across timed
events with action beats between them, and that separation is an independently
measured lever (`TARGET_SECONDS_PER_LINE`, the pause cues in `compiler.py`).
The HD path also skips `structure_prompt` entirely. Handed a list of things
said, the model rendered the gist, which is a fair reading of what it got.

The next experiment is therefore separation, not more instruction: interleave
the lines with the beats between them rather than listing them. That is a
prompt-composition change inside `compose`, measurable with one render per
arm, and it should not need anything new from the graph.

*Caveat on the evidence.* Whisper can mishear generated speech. Four lines all
differing plus a repeat is a pattern rather than transcription noise, but the
exact wording above should not be quoted as ground truth.

## Not done

* Sectioned multi-voice dialogue with a master voice track and post-stitch
  lip sync — needs a TTS service and a lip-sync stage this platform does not
  have. Reported rather than substituted, per the client's own rule.
* A UI toggle. Environment switch only for now.
* Their 10-second layout (5 + 5). Our HD ladder is 8 and 15, both single-pass,
  where sectioning would be slower *and* restart the audio at the seam — the
  precise defect their pack exists to prevent.
