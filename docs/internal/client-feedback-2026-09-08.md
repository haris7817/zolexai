# The client's 8 Sep 2026 report — what was true, and what shipped

Five findings arrived together, diagnosed against the client's own ComfyUI
graphs rather than against this platform. Three of them name a real fault
here; two name machinery that has never run here. This is the audit, then the
changes.

## 1. "Auto Dialogue produced no dialogue" — TRUE, and the cause was reach

Their diagnosis was that the backend never calls `prepare_comfy_job()` before
submitting the workflow. There is no such function in this repository, and
the feature is not missing: `apps/worker/worker/dialogue/` writes spoken
lines into a prompt that has none, both single-pass adapters call it, and it
has passed since 7 Sep 2026.

What was missing was every way to ask for it.

* `settings.auto_dialogue` did not exist, so no workflow offered the switch
  and no panel drew one.
* `dialogue_language` was gated on `settings.prompt_modes`, which Text to
  Video has not declared since 5 Sep 2026 — so the client's own request body
  (`{"auto_dialogue": true, "dialogue_language": "English",
  "maximum_speakers": 4}`) came back **422 on two of its three fields**.
* `AUTO_DIALOGUE_ENABLED` is off by default, so with nothing set the worker
  logged `auto_dialogue_skipped reason=disabled` and rendered a silent video.

And when it did run and fail, it failed **open** — by design, and that design
was wrong for a customer who ticked a box. A silent video that looks like a
success is the worst outcome available.

### What shipped

* `settings.auto_dialogue: true` on Text to Video, Image to Video and Text to
  Video HD (`workflow-definitions/*.yaml`), through the API's workflow schema,
  the shared zod contract, and into the panel as a switch with a **Maximum
  speakers** choice under it.
* `dialogue_language` is now accepted from *either* control — prompt modes or
  Auto Dialogue — and compared case-insensitively, so "English" is not a 400.
* `maximum_speakers` reaches the worker's own validator ceiling
  (`native.MAX_SPEAKERS`, 4) and its screenplay system prompt.
* **Fail closed on an explicit request.** `requested_explicitly()` splits the
  two contracts: a deployment default still falls open; a job whose own
  parameters asked for dialogue and could not be given any raises
  `AdapterError` naming what happened. Refusals that are *answers* — the
  prompt already has quoted words, the customer asked for silence, the sound
  is off, the scene has nobody in it — still render, because those videos are
  correct.
* `comfy_prompt_submitted` logs the positive and negative text as the graph
  receives it, plus a quoted-line count. Their ask was to "log the final value
  inserted into node 5508"; this is that value, at both submit points.

## 2. "The negative prompt is empty" — TRUE of their graph, not of ours

The box is empty in the ZIP. Every job this platform submits fills it from
`worker/comfy/ltx_prompts.py`.

What their universal list carries that ours did not is worth having: the
temporal terms (`temporal flicker`, `frame-to-frame inconsistency`,
`texture crawling`), the continuity terms (`unintended color changes`,
`paint color changes`) and the physics terms (`broken physics`,
`floating objects`).

### What shipped

`UNIVERSAL_NEGATIVE`, verbatim, composed **in front of** each workflow's own
list rather than replacing it, with the large overlap between them
deduplicated case-insensitively so no term is spent twice.
`execution.negative_prompt` still replaces the workflow's list, as a
deployment override always has; `execution.universal_negative: false` restores
the exact pre-change text.

Their own caveat stands and is worth repeating to them: negative conditioning
suppresses named failure modes. It does not track an object. The identity
clauses in the positive prompt still do that work.

## 3. "The Simpsons render darkened, and ReActor is wrong for a cartoon" — HALF TRUE

ReActor, RetinaFace, `inswapper_128` and the per-frame SAM mask exist in the
client's `LTX2.5_Character-Replacement_MULTI4_FHD_COMBINED.json` and **nowhere
in this deployment**. Our Character Replacement graph carries no face-swap
branch at all, which is also why our render is not the eleven-minute one they
measured. There is nothing here to switch off, and the two-path plan they
proposed (tracked face crops for humans, reference conditioning for cartoons)
describes a pipeline we do not run.

Their underlying observation is still right, and it lands on machinery we
*do* run:

* the negative prompt lists `cartoon, illustration, anime, CGI` as faults to
  avoid — while the positive prompt asks for exactly that;
* `worker/media/skin.py` keys on the **classic human skin chroma range**, and
  a bright drawn face sits close enough to it for the gate to fire and pull
  the character toward a human skin level. A dark grey face is what that
  looks like from outside.

### What shipped

`CharacterReplacementAdapter.is_stylised()` — `execution.subject_style`
outright, else a named heuristic over the prompt (`STYLISED_PATTERNS`). On a
drawn character:

| stage | photographed | drawn |
|---|---|---|
| style terms in the negative | kept | dropped |
| `chain_skin_clause` | on | off |
| `skin_hold` | on | off |
| `skin_anchor` | on | off |
| exposure lock | `CHARACTER_REPLACEMENT_EXPOSURE` | `..._STYLISED` — the same guarantee in terms a drawing has, naming colours rather than skin |

The photographed path is untouched, and a test asserts that.

## 4. "Upscale to 4K after the video is done" — SHIPPED

Text to Video HD has delivered 4K since 8 Sep 2026. Its `_upscale_4k` body
moved to `worker/media/upscale.py` unchanged, and Character Replacement now
calls it: `execution.delivery: "4k"` (or
`CHARACTER_REPLACEMENT_DELIVERY=4k`), applied **once**, after the windows are
joined and the source audio is laid over — one encode, one audio stream, no
seam. The frame is computed from the delivered canvas (2160 on the short
side, both sides even) rather than looked up by ratio label, because this tool
takes whatever shape the customer's source is.

Off by default, so no deployment starts writing files four times the size
without being told.

## 5. "Never use the previous chunk as the next reference" — NOT ADOPTED

Chained Character Replacement seeds each window with the previous window's
last frame, deliberately, and
`docs/internal/character-replacement-full-length.md` records why: a fresh
photo reference per window restarts the character's pose and lighting at
every seam. The darkening that motivated their advice is fixed at the seed
and per delivered frame instead (`skin_hold`, `_anchor_seed`), and on a drawn
character those stages now stand down entirely — the same outcome by a route
that does not reintroduce visible seams. `execution.chain_reference: photo`
still switches to their scheme for a deployment that wants to measure it.

Their timing arithmetic assumed our render carries their ReActor/SAM cost. It
does not, so the "3 minutes plus upscaling" figure is not a prediction of
anything this pipeline was doing.

## 6. The LTX 2.5 guideline pack — RECEIVED 9 Sep 2026, SHIPPED

The eight files arrived as `ltx-2.5-guideline-pack.zip`. Their own header
calls the text *"an implementation-oriented paraphrase of the official LTX-2.5
prompting guidance"* and cites `ltx.io/blog/ltx-2-5-prompt-guide` and
`docs.ltx.io/models/ltx-2-5` — so it is the client's writing, not the
vendor's, and vendoring it raises no licensing question. It lives unedited
under `apps/worker/worker/prompt/ltx25/guidelines/`.

`worker/prompt/ltx25/` is the routing, assembly and validation around it:

* **`forms.py`** — the client's routing order exactly (Dub-It → screenplay →
  multi-shot → single-shot), plus their two caveats: Dub-It is only for
  replacing speech in a source video, and Image to Video stays single-shot
  unless a cut is *named*. `execution.ltx25_form` pins it.
* **`compose.py`** — `core.txt` plus **one** form guide plus the audio rules,
  because they were explicit that this is a selection, not a concatenation.
  `terminology.json` goes in two halves: the camera and edit vocabularies
  bind (they are what makes the form boundary decidable), everything else is
  an explicitly optional menu with "never add a subject, style, effect or
  sound to the scene merely to use a term from it".
* **`validate.py`** — the half of `validation.txt` a machine can decide, in
  their own ERROR/WARNING/PASS vocabulary. An ERROR buys one corrective round
  naming the broken rule; still failing falls open.

### Three decisions worth knowing

1. **It runs BEFORE Auto Dialogue.** Their screenplay guide tells a rewriter
   that sees injected lines to treat them as authoritative. Running first is
   stronger than instructing for it — the lines are composed onto an already
   finished prompt, so no model is ever positioned to paraphrase them. That
   matters because paraphrase is the *measured* failure: the first real Auto
   Dialogue render delivered four lines as "one verbatim, three paraphrased,
   one phrase twice".
2. **A rewritten job stands `prompt_structuring` down.** `core.txt` §7 and
   `validation.txt` §11 forbid saying the same thing twice, and
   `worker/longform/enhance.py` appends its own continuity block. A job that
   is *not* rewritten keeps it.
3. **`soundscape_clause` still runs.** It is not general craft — it is three
   GPU-measured fixes (the model narrating its caption, a five-word line
   looping for twenty seconds, the described-silence tail) that the pack has
   no equivalent of. Where they overlap they agree.

### Off by default

`LTX25_GUIDELINES_ENABLED` / `execution.ltx25_guidelines`, default **off**.
The prompt text is the product on these workflows, this rewrites all of it
through a language model, and what it replaces carries measurements the pack
does not. It wants an A/B on a real node before it becomes the default —
which the per-job override makes one request rather than one redeploy.

Dub-It is vendored and unreachable: no workflow here replaces speech in a
source video. The constant and the guide exist so the day one does, it is a
routing line rather than a research task.

## Still open

Nothing from their report. For the record, the one question worth putting back
to them is **precedence**: where the pack's generic guidance and our
GPU-measured prompt rules disagree, we have kept ours (decision 3 above). That
is a judgement, not a rule they gave us.

<!-- The original entry, kept for the record:
The **LTX 2.5 prompting-guideline pack** (`core.txt`, `single-shot.txt`,
`multi-shot.txt`, `screenplay.txt`, `audio-dialogue.txt`, `dub-it.txt`,
`terminology.json`, `validation.txt`, and a router that picks between them).
The routing *rules* were described; the guideline **texts** were not sent, and
writing them from a summary would ship invented rules under the vendor's name.

It also overlaps `worker/longform/enhance.py` and `worker/director/`, which
already structure prompts on this runtime and carry their own GPU-measured
reasons for what they say. Where the two disagree is a question for the
client, not a merge.

**Ask them for the eight files.**
-->
