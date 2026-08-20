# 60-second continuity, Director-aware Extend, and the I2V language check

**Date:** 2026-08-20 · **Status:** implemented, unit-tested, GPU-measured on an
isolated checkout; NOT committed, NOT deployed (per instruction)
**Scope rule:** 5/15/30s Text-to-Video is the known-good baseline and renders
byte-identically (argv-pinned). Standard Extend, standard I2V, V2V, Music and
Music Video are untouched.

---

## 1. The 60-second root cause — proven, twice over

**Geometry.** `LTX_MAX_SECONDS=60` on the box means a 60s request is ONE
1441-frame pass (production log, job `52783150`: `passes: 1`). The entire
section machinery — the thing that gives each window only its own events and
carries state across seams — is bypassed at exactly the duration where the
problems live. Single-pass 60s dialogue never measured cleaner than 1 repeat
in 22 lines; 30s passes measured 0.

**Presence blindness.** The compiler's constancy block (added 19 Aug against
flicker) asserted every character "present and solid in every single frame" —
including one whose planned departure is in the same caption. Compiled text
for a goodbye scene literally contained both "the man opens the door and walks
out" and "the man stays fully visible from the first frame to the last".

**Reproduced on the GPU (R1, 60s single pass, kitchen-goodbye idea):**
transcript clean (14 lines, once each — the 19 Aug dialogue fixes hold), but
the presence probe (per-second person count via the on-box vision checkpoint,
confirmed by eye) read:

    f001–f038  2 2 2 … 2      both present
    f039–f042  1 1 1 1        man leaves ≈38s, as planned
    f043–f048  2 1 2 1 1 1    ← flicker
    f049–f060  2 2 2 … 2      ← FULL RETURN for the final 12 seconds

— while the soundtrack said "He is finally gone. The house is silent." At 50s
and 58s he stands at the counter in his blue coat. The model obeyed the
caption; the caption was wrong.

## 2. The fix (minimal, two halves)

**Geometry:** `execution.max_segment_seconds: 30` in `text-to-video.yaml` and
`image-to-video.yaml`. 60s renders as two measured-clean 30s sections through
the existing chain; every other menu duration is the same single pass it
always was (pinned by test). Director extensions get the same 30s cap in the
adapter.

**Semantics — departures are state, not prose:**

- `DirectorEvent.exits` — character ids who leave at that event and stay gone.
  The planner is briefed to tag permanent departures; validation refuses an
  exit contradicted later (speaking after leaving, exiting twice, unknown id).
- Presence follows the section BUCKETS: from the section after the one that
  renders the departure, the character is out of the cast sentence, out of
  both constancy sentences, and never named again — on this runtime a name in
  a caption is a request for its owner.
- The resulting state is stated positively the moment the exit completes:
  "From this moment, the woman is alone in the scene." Action language
  ("walks out") appears exactly once, in its own section; later sections carry
  the state, never the action.
- People-count continuity facts ("Two people are in the kitchen") are dropped
  from any plan that carries an exit — restating a count after it stops being
  true is the same ghost in different words.
- The described-silence tail and the constancy verbs now agree with who is
  actually left ("the woman holds the moment", "keeps … face … voice").

The dialogue/event ledger the task asked for already existed by construction:
one global plan, each event bucketed into exactly one section, completed
sections' lines structurally absent from later prompts. What was missing was
the entity ledger; `exits` is it.

## 3. GPU measurements (RTX PRO 6000, isolated checkout, 20 Aug 2026)

| run | shape | result |
|---|---|---|
| R1 60s director (BROKEN baseline) | 1×60 | departure flickers, then departed man fully returns for last 12s |
| F1 60s director (fixed) | 2×30 | **presence 2×34 → 1×26, zero flicker, zero return**; 13 lines once each; no restart at the seam; frames 720+713=1433 = output exactly (no dup/drop at concat); seam frames at 29.5/30.5s identical people, room, bag; **191.4s wall vs 284.8s — 33% faster** |
| F2 30s director + departure | 1×30 | boy leaves at ~24s, stays gone; final frame is the planned end-state (sailor alone mending the net); one "Good night" echo at tail (known residual class) |
| F3 60s standard, continuous car | 2×30 | same car, coast, sun across the seam; motion continuous — TEST 5 holds |

The mid-run single-person frames in F2 are close-up coverage (verified by
eye), not flicker — presence probes count what the CAMERA shows.

## 4. Director-aware Extend — the lineage was already in the rows

`generation_job_outputs` maps every generated asset to the job that made it,
and that job's row has carried `prompt` (the idea, verbatim) and
`request_params` (`prompt_mode`, `dialogue_language`) since the feature
launched. So Director ancestry is **resolved at creation time from existing
data — no migration, no protocol change — and works retroactively for every
Director video ever generated.**

At extend-creation the API walks source asset → producing job and stores
`director_lineage` in the new job's params: mode, language, idea,
`prior_seconds`, and (for I2V ancestors) the ORIGINAL upload's asset id,
which is also attached as a server-injected `identity_image` input. An extend
of an extend inherits its parent's stored lineage and accumulates the
seconds, so the third +10s still knows the story began four videos ago.

The worker plans the CONTINUATION: the ancestor's language (never a silent
fall-back to English), the idea as THE STORY SO FAR that has already fully
happened before second 0, and the **anchored register — the finished video's
last frame owns WHO and WHAT exactly the way an uploaded photograph does**,
so grounding strips any invented appearance that would fight the footage.
Compilation, chaining, seam conditioning: the existing machinery, unchanged.
The identity image rides every extension pass at the same mid-window 0.2
reference I2V chains have always used.

No lineage → the planner is never consulted and the extension is
byte-identical to what has always shipped (pinned by test). Missing/unready
identity image drops the anchor, never the lineage; malformed lineage is
ignored.

## 5. The I2V dialogue-language "gap" was not reproducible

Driven with a real browser against live production: the I2V panel shows the
Prompt Mode toggle and the **Dialogue Language selector**, correctly, on the
deployed build (screenshot taken 20 Aug). The API serves
`prompt_modes: true` for image-to-video; the form layer is generic on it; and
the planned dialogue is genuinely in the selected language, measured on the
box for I2V requests: es "¿Cómo será la escuela mañana?", fr "L'école
changera beaucoup, non ?", de "Wie sieht Bildung morgen aus?". The likely
origin of the report is a browser bundle cached from before the 19 Aug VPS
rebuild. **Portuguese is absent by design** — the five offered languages are
the set the vendor documents as validated for generated speech.

## 6. Honest limits

- **Model-level line echo is reduced, not eliminated** — F2's tail repeated
  "Good night" once. Known residual class (survivors were always the
  opening/closing line); per-pass geometry now caps exposure at 30s.
- **Presence machinery covers CHARACTERS.** Irreversible object state
  (popped balloon, closed drawer) rides on predecessor frames across seams
  and on the model within a pass; no object ledger was built (smallest
  structure that stops the reported failure — people coming back).
- Exit validation can catch a character SPEAKING after their exit (speakers
  are structured); an exited character re-appearing in later action PROSE is
  planner error the code cannot detect.
- Continuation planning knows the ancestor's IDEA, not its exact rendered
  lines: a continuation could phrase a thought similar to one already spoken.
  The register ("everything the story so far implies has already been said")
  is the mitigation; the exact-line ledger would require persisting plans at
  completion, deliberately not built (protocol + storage change).
- Vision/presence probe counts what the camera shows — close-ups read as
  fewer people; sustained runs, not single frames, are the signal.


## Post-deploy: the "suuu" hiss on quiet standard extensions (measured, no fix shipped)

Customer report, 20 Aug: a +10s standard extension of a quiet standard I2V
came back with hiss ("suuu") instead of sound. Reproduced and measured on the
box — 1 source + 2 bare-prompt extensions + 2 extensions with an appended
positive soundscape sentence, extension segments analysed (spectral flatness,
centroid, volume, whisper):

    source's own audio      flatness 0.49   1884 Hz   -36.4 dB   (decent)
    extension, bare  x2     0.53 / 0.56     ~2040 Hz  -45 dB     (one = pure
                                                        phoneme garbage in
                                                        whisper - the "suuu")
    extension + line x2     0.62 / 0.57     ~2230 Hz  -45 dB     (WORSE)

Two findings:

1. **Extension-pass audio of speechless scenes is systematically thinner and
   whiter than the source pass's own audio** — every extension segment, both
   arms. This is the officially documented limitation ("when generating audio
   without speech, the audio may be of lower quality") amplified by the
   extension pass starting its soundtrack cold.
2. **An appended ambience sentence does NOT help — it measured worse.** The
   obvious lever fails its A/B; do not ship it on vibes later.

What actually works: dialogue owns the soundtrack. Director-lineage
extensions plan speech and inherit the measured-clean speech results;
standard extensions can carry sound in the user's own prompt (speech,
humming) but ambience prose alone is not a fix. The structural fix is
audio-conditioned extensions (a2vid hearing the source's tail) — guided
family, ~4x cost, a pricing decision in the same class as music-video's
`audio_conditioning`.
