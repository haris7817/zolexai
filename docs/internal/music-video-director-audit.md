# Music video quality audit — why the lip-sync was good and the picture was not

**Date:** 27 August 2026 · **Engine:** unchanged (LTX 2.5 native A2V) ·
**Commits:** `7f1fe42`, `70817d1`

The client's report was precise: *"lip-sync is good, but all the videos no
matter the prompt come out the same, with people in the back."* A frame they
sent showed a second, worse symptom — garbled banner text and what read as
our own prompt header burned into the picture.

Both were ours. Neither was the model.

---

## 1. What the model was actually being sent

The decisive step was printing the assembled prompt rather than reasoning
about it. For a 20-word customer prompt, a music-video section received:

```text
LONG-FORM CONTINUATION — SECTION 1 OF 3.
Keep the same subjects, identities, faces, clothing, colours, object counts,
vehicles, environment and camera direction established previously.
PERSISTENT USER CONSTRAINTS (verbatim):
A vintage red sports car races through neon-lit rainy streets, no people
NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY (verbatim):
Continue naturally from the preceding section without introducing a new event.
Continue directly from the predecessor frame. Do not replay, restart or
summarise any earlier action or dialogue. Complete this section's assigned
dialogue before the section ends.
```

Plus, appended by the 26 Aug vocal-direction work:

```text
The vocal is active in this passage: she sings the words, her mouth clearly
articulating in time with the voice in the music.
```

Four defects are visible in that text, and together they explain the
complaint exactly:

| # | Defect | Consequence |
|---|---|---|
| 1 | `she sings the words, her mouth clearly articulating` appended to **every** section | A female singer was **invented** in every music video, whatever the customer asked for |
| 2 | Scaffolding is ~90 words against the customer's ~20 — and names "faces, clothing" and "dialogue" ×3 | Generic nouns read as suggestions: people, crowds, talking |
| 3 | Section 1 told to keep what was "established previously" when nothing was | The model fills the void from its own prior — a performance scene with an audience |
| 4 | ALL-CAPS labels, `(verbatim)`, `SECTION 1 ONLY` | **This runtime reads captions as content**: the labels rendered into the frame as on-screen text |

## 2. The structural finding

Music video had **audio intelligence and no visual direction.**

Present and working: onset detection placing cuts on musical boundaries
(`plan_musical_boundaries`), and — since 26 Aug — vocal-activity spans from
stem separation telling the system exactly which seconds are sung.

Absent: any notion of a shot. The Director module (`worker/director/`) that
plans shots for text-to-video is **never called for music video**. Every
section received the same repeated prompt, so a three-minute video was one
composition held for three minutes.

That asymmetry is the whole answer to "why is lip-sync good but the video
weak": the audio path was engineered, the picture path was a caption.

## 3. What was built

`worker/longform/music_video.py` — a shot director, deterministic, no model
in the loop (the same discipline as the rest of the prompt layer).

**Section roles from the audio the worker already analyses:**

| Signal | Source | Used for |
|---|---|---|
| Who is singing | vocal spans (stem separation) | sung vs instrumental |
| How big this moment is | section RMS vs the song's own median | chorus vs verse |
| Where we are in the song | position | intro / outro |

→ `intro · verse · chorus · bridge · outro`, each with its own shot
vocabulary (framing + one clear camera move), and a rule that no section may
repeat the framing of the section before it.

**The rendered prompt is prose.** No headings, no all-caps, no `(verbatim)`,
no newlines — and it closes with an explicit ban: *no text, no logos, no
captions and no watermark anywhere in the picture.*

Before and after, same customer prompt:

```text
BEFORE  LONG-FORM CONTINUATION — SECTION 2 OF 5. Keep the same subjects,
        identities, faces, clothing … NEW ACTION OR DIALOGUE FOR THIS
        SECTION ONLY (verbatim): Continue naturally … she sings the words,
        her mouth clearly articulating …

AFTER   A vintage red sports car races through neon-lit rainy streets at
        night. This continues the same unbroken performance, the same
        subject and the same place as a moment ago. Filmed as a low-angle
        hero shot; the camera rises, looking up. Nothing enters the scene
        that is not described above — no extra people, no crowd, no
        audience, no text, no logos, no captions and no watermark anywhere
        in the picture.
```

## 4. Constraints honoured

LTX is unchanged. The A2V engine, the audio conditioning path, the
conditioning-master padding, the routing and the lip-sync are all untouched.
Only prompt generation and section planning changed.

## 5. Open — not yet done

- **Before/after benchmark renders.** The five test cases the brief asks for
  (performance, dance, story, product, emotional close-up) need GPU time and
  a human verdict. Nothing here claims a measured quality delta yet.
- **External prompting research** (LTX's own enhancer vocabulary, published
  camera-control terms) — in flight; the shot vocabulary above is reasoned,
  not yet corroborated against vendor documentation.
- **`prompt_structuring_v2`** — fixes defect 3 for the LTX paths that still
  use the generic planner. Flag-gated, deployment YAML, not yet enabled.
- **Beat-level cutting.** Cuts land on musical boundaries; shot *changes*
  are per section. Sub-section cutting on the beat is unbuilt.
