# Diagnosis — subject flash-back (A) and camera direction (B)

**Repo commit:** `3bd80165da35c52d1f0fd7cd16d5daacf0a7032a` · branch `main` · working tree clean
**Companion to:** [ZOLEXAI-CURRENT-STATE-AUDIT.md](ZOLEXAI-CURRENT-STATE-AUDIT.md)
**Phase:** 1 — diagnosis only. **No repository file was modified.** No GPU, no inference, no commit.

**Method.** Every verdict below was produced by executing the repository's own functions
(`structure_prompt`, `_separate`, `plan_section_prompts`, `plan_chain_segments`,
`_per_pass_seconds`, `_audio_pass_seconds`, `_guided_pass_seconds`, `_identity_anchor`,
`system_prompt`, `user_prompt`) from a driver outside the repo, plus targeted greps.
Prompt text pasted below is real output, not paraphrase.

---

## Executive summary

Two confirmed defects, one of which the audit described with the wrong cause.

- **A5 is a real classifier bug**, but not the one [audit §C.7](ZOLEXAI-CURRENT-STATE-AUDIT.md) states. The operative regex is `_DIALOGUE_LINE`, not `_PERSISTENT_LINE`. A continuity rule the *worker itself wrote* is being read as a **user-authored dialogue turn** and allocated to exactly one section.
- **B5 is a real contradiction.** A user who types *"The camera never moves"* gets *"the camera keeps moving through the same environment"* appended to the same prompt, by the worker, unconditionally.

And one finding that changes how A5 should be handled:

> **Fixing A5 in isolation makes issue A worse, not better.** The misclassification is currently the only thing stopping the persistence rule from reaching every section. Correct the classifier and *"every subject present at the start is still present at the end"* goes from **1 section** to **all N**. That is a quality decision you own, so A5's remedy is sequenced behind it rather than shipped alone.

One item hits your do-not-touch list and I have **stopped** on it: **A4** is caused by the interaction of `max_segment_seconds: 30` and `_TWO_IMAGE_SAFE_FRAMES`, both forbidden. Reported, not adjusted.

---

## Part A — subjects disappear and flash back

### A1 — the standing persistence rule · **CONFIRMED**

**The text, verbatim** ([enhance.py:122-132](../apps/worker/worker/longform/enhance.py#L122-L132)):

```python
    lines = [text, "", "CONTINUITY (fixed for the entire video):"]
    lines += [f"- {rule}" for rule in rules]
    lines += [
        "- The same subjects keep the same faces, clothing, colours and count "
        "in every frame.",
        "- One continuous scene: the camera keeps moving through the same "
        "environment, and every subject present at the start is still present "
        "at the end.",
    ]
```

**Two standing rules, not one**, and they behave differently once a job chains (see A5):

| Rule | Reaches |
|---|---|
| `- The same subjects keep the same faces, clothing, colours and **count in every frame**.` | **every section** (classified persistent) |
| `- One continuous scene: … **every subject present at the start is still present at the end**.` | **section 1 only** (misclassified as an action — A5) |

Both are presence assertions. The first is the one that reaches every pass.

**Which workflows get it** — `execution.prompt_structuring`, read at [ltx.py:1061](../apps/worker/worker/adapters/ltx.py#L1061):

| Workflow | `prompt_structuring` | Gets the rules? |
|---|---|---|
| `text-to-video` | `true` ([text-to-video.yaml:89](../workflow-definitions/text-to-video.yaml#L89)) | **yes** |
| `image-to-video` | `true` ([image-to-video.yaml:74](../workflow-definitions/image-to-video.yaml#L74)) | **yes** |
| `extend-video` | `true` ([extend-video.yaml:71](../workflow-definitions/extend-video.yaml#L71)) | **yes** |
| `music-video` | `true` ([music-video.yaml:71](../workflow-definitions/music-video.yaml#L71)) | **yes** |
| `video-to-video` | *absent* | no |
| `music` | *absent* | no (not an LTX workflow) |

**Can anything suppress it once a subject legitimately leaves? — No.** There are exactly three suppression paths and **none is presence-aware**:

1. `execution.prompt_structuring` falsy — [ltx.py:1061](../apps/worker/worker/adapters/ltx.py#L1061)
2. `wants_director(job)` true — same line (Director mode skips structuring entirely)
3. `_ALREADY_STRUCTURED` matches the **user's own text** — [enhance.py:70-72](../apps/worker/worker/longform/enhance.py#L70-L72), pattern `^\s*(persistent|section\s+\d+|continuity)\s*:`

Path 3 is a *shape* test on the user's prompt, not a *content* test. Probed against real prompts:

```
  bail=True  rule_appended=False  'Persistent: a red car'
  bail=True  rule_appended=False  'Section 1: he walks in'
  bail=True  rule_appended=False  'Continuity: the hat is red'
  bail=False rule_appended=True   'A man walks out of the room and never comes back.'
  bail=False rule_appended=True   'The woman leaves. She does not return.'
  bail=False rule_appended=True   'Subjects: two men. One exits at 20s.'
```

> A user who writes **"The woman leaves. She does not return."** receives a prompt that also says **"every subject present at the start is still present at the end"** and **"the same subjects keep the same … count in every frame"**.

`enhance.py` contains no exit/leave/departure vocabulary of any kind; it inspects only counts (`_COUNT_PATTERN`) and colours (`_COLOUR_PATTERN`).

**Mode split:** applies in **STANDARD mode only**. Director mode skips `structure_prompt`.

---

### A2 — exit tracking exists only in Director mode · **CONFIRMED**

`grep -rn 'exits|departed|present_ids|exit_time|has_exits|_PRESENCE_COUNT|_drop_presence_counts'` across `apps/worker/worker`, `apps/api/app`, `workflow-definitions` returns **41 hits, every functional one inside `worker/director/`**:

| File | Hits | Kind |
|---|---|---|
| `worker/director/plan.py` | 24 | `DirectorEvent.exits`, `exit_time`, `present_ids`, `has_exits`, `_parse_exits`, `_check_exit_consistency`, `_drop_presence_counts`, `_PRESENCE_COUNT` |
| `worker/director/compiler.py` | 8 | the `departed` walk, `_after_exit_sentence`, `_remaining_sentence`, the log |
| `worker/director/provider.py` | 3 | the `DEPARTURES` rules in the planning brief |
| `worker/adapters/ltx.py` | 2 | **prose only** — a comment at [:1279](../apps/worker/worker/adapters/ltx.py#L1279), and the unrelated word in *"the render exits with a ValueError"* at [:2778](../apps/worker/worker/adapters/ltx.py#L2778) |
| `workflow-definitions/text-to-video.yaml` | 1 | **prose only** — [:73](../workflow-definitions/text-to-video.yaml#L73) |

**Nothing in `worker/longform/`, `worker/media/`, or the adapter tracks presence.** No equivalent exists in standard mode.

**What standard mode does when a subject leaves — plainly:**

> **Nothing records it, and the prompt asserts the opposite.** There is no field, no flag, no parse, no state. The seam carries a single PNG (pixels, no semantics). Every section repeats *"the same subjects keep the same faces, clothing, colours and count in every frame"*, and section 1 additionally carries *"every subject present at the start is still present at the end"*. The model is told, in text, in every pass, that the cast is constant — including the passes after someone has walked out.

The Director-mode machinery that solves exactly this is documented at [plan.py:163-177](../apps/worker/worker/director/plan.py#L163-L177), and its docstring names the measured failure it was built for (GPU, 20 Aug 2026: a man who *"flickered at 43-48s and stood fully returned for the final twelve seconds"*). **That is your reported symptom A, and the fix for it exists only in Director mode.**

**Mode split:** Director mode has full exit handling. Standard mode has none.

---

### A3 — seam conditioning and overlap · **CONFIRMED (both parts)**

**Single PNG at strength 1.0.** `render_chain` extracts one frame per seam ([chain.py:168-172](../apps/worker/worker/longform/chain.py#L168-L172)) via `ffmpeg -sseof -1 -i <part> -update 1 <png>` ([frames.py:43-46](../apps/worker/worker/media/frames.py#L43-L46)). Strength by workflow: `1.0` for t2v/i2v/extend/music-video ([ltx.py:1133](../apps/worker/worker/adapters/ltx.py#L1133), [:1261](../apps/worker/worker/adapters/ltx.py#L1261), [:1880](../apps/worker/worker/adapters/ltx.py#L1880)); `0.85` for v2v (`v2v_continuity_strength`).

**Overlap is 0.0 everywhere.** `plan_segments`' default is `overlap_seconds=0.0`, and both `chain.py` call sites are identical:

```
  plan_segments default overlap_seconds = 0.0
  chain.py: return plan_segments(total_seconds, max_segment_seconds=per_pass_seconds)
  chain.py: return plan_segments(total_seconds, max_segment_seconds=per_pass_seconds)
```

No caller anywhere passes a non-zero value. `Segment.overlap_seconds`, `source_start_seconds`, `generate_seconds` and `trim_start_seconds` ([segments.py:40-61](../apps/worker/worker/media/segments.py#L40-L61)) are implemented and unreachable.

**Seam counts, computed through `plan_chain_segments`:**

| Workflow / tier | per-pass (s) | 30 s | 60 s | 180 s source |
|---|---|---|---|---|
| text-to-video (`max_segment_seconds: 30`) | 30.0000 | 1 sec / **0 seams** | 2 sec / **1 seam** | 6 sec / **5 seams** |
| image-to-video (`max_segment_seconds: 30`) | 30.0000 | 1 sec / **0 seams** | 2 sec / **1 seam** | 6 sec / **5 seams** |
| extend-video (no key) | 60.0000 | 1 sec / **0 seams** | 1 sec / **0 seams** | 3 sec / **2 seams** |
| music-video, default tier | 60.0000 | 1 sec / **0 seams** | 1 sec / **0 seams** | 3 sec / **2 seams** |
| music-video, audio tier | 20.0417 | 2 sec / **1 seam** | 3 sec / **2 seams** | 9 sec / **8 seams** |
| video-to-video, transform engine | 8.0000 | 4 sec / **3 seams** | 8 sec / **7 seams** | 23 sec / **22 seams** |
| t2v/i2v, guided tier | 5.0000 | 6 sec / **5 seams** | 12 sec / **11 seams** | 36 sec / **35 seams** |

*(t2v/i2v cap at `"60s"` in `supported_durations`; the 180 s column is the arithmetic, reached only via chained extends. v2v/music-video take their length from the upload, so 180 s is a real customer case there.)*

Relevance to symptom A: **your report says "especially in longer videos", and seam count is the only thing that scales with length.** Each seam is a fresh 22B subprocess with no shared latent, KV cache, or RNG state — only one PNG and the section text.

**Mode split:** identical in both modes; the chain is mode-agnostic.

---

### A4 — the source image is dropped past pass 1 · **CONFIRMED — and it is worse than stated**

Dry-run of `_identity_anchor` across every offered image-to-video duration:

```
 duration  per_pass  sections  pass-2 len  frames   anchor   detail
       5s      30.0         1           —     120      n/a    single pass; still is frame 0 @1.0
      10s      30.0         1           —     240      n/a    single pass; still is frame 0 @1.0
      15s      30.0         1           —     360      n/a    single pass; still is frame 0 @1.0
      30s      30.0         1           —     720      n/a    single pass; still is frame 0 @1.0
      60s      30.0         2       30.00     720  DROPPED    720 not in [120, 240, 360]
```

**`"60s"` is the only image-to-video duration that chains at all, and it is exactly the one where the anchor drops.** So at the shipped configuration `_identity_anchor` ([ltx.py:2432-2466](../apps/worker/worker/adapters/ltx.py#L2432-L2466)) **never fires on any duration the product offers.** Its `i2v_reference_strength = 0.2` and its `frames // 3` index are unreachable code in practice.

Pass lengths that *would* keep it, and the cost:

```
   120 frames = 5.0000s  ->  60s job = 12 sections, 11 seams
   240 frames = 10.0000s ->  60s job =  6 sections,  5 seams
   360 frames = 15.0000s ->  60s job =  4 sections,  3 seams
   480 frames = 20.0000s ->  DROPPED
   720 frames = 30.0000s ->  DROPPED   (current)
```

Director-lineage extensions land in the same place: the 30.0 s clamp at [ltx.py:1283](../apps/worker/worker/adapters/ltx.py#L1283) gives 720-frame passes → `DROPPED`.

> ### ⛔ STOP — do-not-touch constants implicated
>
> The drop is produced by the intersection of two values you placed off-limits:
> **`_TWO_IMAGE_SAFE_FRAMES = {120, 240, 360}`** ([ltx.py:363](../apps/worker/worker/adapters/ltx.py#L363)) and
> **`execution.max_segment_seconds: 30`** ([text-to-video.yaml:81](../workflow-definitions/text-to-video.yaml#L81), [image-to-video.yaml:70](../workflow-definitions/image-to-video.yaml#L70)).
>
> Every route to keeping the anchor requires changing one of them. I have not adjusted either and am not proposing a value. Both carry dated production-failure comments — the 720-frame two-image crash of 20 Aug 2026 ([ltx.py:2443-2448](../apps/worker/worker/adapters/ltx.py#L2443-L2448)) and the 60 s story-coherence measurement ([text-to-video.yaml:70-81](../workflow-definitions/text-to-video.yaml#L70-L81)). **This item ends here pending your decision.**

**Mode split:** applies to both modes on image-to-video; Director-lineage extend is affected identically.

---

### A5 — the continuity block is misclassified · **CONFIRMED as an outcome · audit §C.7's stated cause is WRONG**

**Line-by-line regex probe** of `structure_prompt`'s own output:

```
line             _PERSISTENT_LINE  _DIALOGUE_LINE  _SEQUENCE_START  _TIMED_LINE  _SECTION_LINE
header           False             False           False            False        False
bullet:count     False             False           False            False        False
bullet:colour    False             False           False            False        False
bullet:faces     False             False           False            False        False
bullet:scene     False             TRUE            False            False        False
```

**The operative regex is `_DIALOGUE_LINE`, not `_PERSISTENT_LINE`:**

```
  text before the first colon = '- One continuous scene'   (len 22, limit is 48)
  _DIALOGUE_LINE = '^\s*[^:\n]{1,48}:\s+.+$'
```

The bullet contains an internal colon after `- One continuous scene`, so `_separate`'s first loop catches it at [prompts.py:148-150](../apps/worker/worker/longform/prompts.py#L148-L150) — the branch whose purpose is to recognise **`Name: "spoken line"` dialogue turns** — and appends it to `actions`.

The audit's C.7 blamed the header not matching `_PERSISTENT_LINE`. That is *true but not causal*: `_PERSISTENT_LINE` captures only the text after the colon **on its own line**, which for `CONTINUITY (fixed for the entire video):` is empty. The bullets are separate lines and would never have been swept up by a header match. The header's non-match is a second, independent quirk (it also means `_ALREADY_STRUCTURED.search(structure_prompt(...)) → False`, so `structure_prompt` would restructure its own output if ever called twice).

**Real `_separate()` output on the real structured prompt:**

```
PERSISTENT:
   | A rain-soaked neon alley at night.
   | Two cars idle at the kerb, one matte black and one red.
   | Steam rises from a grate.
   | CONTINUITY (fixed for the entire video):
   | - Exactly 2 cars appear, and they remain the only cars on screen for the entire video.
   | - The matte black one stays matte black from the first frame to the last.
   | - The same subjects keep the same faces, clothing, colours and count in every frame.
ACTIONS (each is assigned to exactly ONE section):
   | - One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.
```

**Real compiled prompts, 2 sections** (`plan_section_prompts(structured, 2, total_seconds=60.0)`):

```
--- SECTION 1/2 ---
LONG-FORM CONTINUATION — SECTION 1 OF 2.
Keep the same subjects, identities, faces, clothing, colours, object counts, vehicles, environment and camera direction established previously.
PERSISTENT USER CONSTRAINTS (verbatim):
A rain-soaked neon alley at night.
Two cars idle at the kerb, one matte black and one red.
Steam rises from a grate.
CONTINUITY (fixed for the entire video):
- Exactly 2 cars appear, and they remain the only cars on screen for the entire video.
- The matte black one stays matte black from the first frame to the last.
- The same subjects keep the same faces, clothing, colours and count in every frame.
NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY (verbatim):
- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.
Continue directly from the predecessor frame. Do not replay, restart or summarise any earlier action or dialogue. Complete this section's assigned dialogue before the section ends.

--- SECTION 2/2 ---
... (identical header and PERSISTENT block) ...
NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY (verbatim):
Continue naturally from the preceding section without introducing a new event.
Continue directly from the predecessor frame. Do not replay, restart or summarise any earlier action or dialogue. Complete this section's assigned dialogue before the section ends.
```

At 3 sections the rule still reaches only the first:

```
  section 1/3 NEW-ACTION block: '- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.'
  section 2/3 NEW-ACTION block: 'Continue naturally from the preceding section without introducing a new event.'
  section 3/3 NEW-ACTION block: 'Continue naturally from the preceding section without introducing a new event.'
```

Single-pass jobs are unaffected — the prompt is byte-identical to `structure_prompt`'s output and contains the rule.

**Regex mismatch or intended? — Mismatch.** Three pieces of evidence, all from the repository's own stated contracts:

1. `prompts.py`'s module docstring ([prompts.py:8-13](../apps/worker/worker/longform/prompts.py#L8-L13)): *"Ambiguous prose remains persistent rather than being silently reinterpreted."* A generated continuity rule read as a dialogue turn is precisely a silent reinterpretation.
2. `enhance.py` labels the block *"CONTINUITY (**fixed for the entire video**)"* — a per-section allocation contradicts its own label.
3. The two blocks have opposite semantics by design: `PERSISTENT USER CONSTRAINTS` is *"an unchanging reference shared by all sections"*; `NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY` is followed by *"Complete this section's assigned dialogue before the section ends"* — the model is being asked to **perform** a continuity rule as if it were a line of dialogue.

**Mode split:** STANDARD mode only, and only on multi-section jobs.

> **⚠️ Sequencing warning.** The misclassification is currently the *only* thing limiting the persistence rule to one section. Repair the classifier alone and *"every subject present at the start is still present at the end"* reaches **all N sections** — strengthening exactly the assertion under suspicion in issue A. See the Fix Plan.

---

## Part B — camera direction does not work

### B1 — which camera concepts exist at all · **CONFIRMED (audit §10.1 is correct)**

Grep across `apps/worker/worker`, `apps/api/app`, `apps/web/src`, `packages`, `workflow-definitions`:

| Term | Hits | Verdict |
|---|---|---|
| `low.angle`, `high.angle`, `eye.level`, `top.down`, `dutch`, `birds.eye`, `worms.eye`, `camera.angle`, `\bangle\b` | **0** | **camera angle does not exist anywhere** |
| `focal`, `\blens\b`, `aperture`, `depth.of.field`, `bokeh`, `film.stock`, `rule.of.thirds` | **0** | **lens / optics do not exist** |
| `match.cut`, `\bwipe\b`, `whip.pan`, `smash.cut` | **0** | **named transitions do not exist** |
| `\btruck\b`, `\bcrane\b`, `pedestal`, `\borbit`, `\barc\b`, `handheld`, `steadicam`, `gimbal` | **0** | **these movements do not exist** |
| `overhead` | 2 | *unrelated* — "syscall overhead", "pure overhead" |
| `composition` | 12 | *unrelated* — image compositing (`masks.py`, `ltx.py`), module composition (`fallback.py`, `writer.py`), and **one line in the vision describer's brief** ([vision.py:50](../apps/worker/worker/director/vision.py#L50)) whose output goes to the **planner**, never to the model |
| `framing` | 3 | *prose only* — two comments, one line of the planner brief |
| `dissolve` | 1 | *unrelated* — "dissolved this project's … ceiling" ([ltx.py:69](../apps/worker/worker/adapters/ltx.py#L69)) |
| `\bfade\b` | 20 | *unrelated* — audio crossfade in the music adapter |
| `\btransition` | 64 | *unrelated* — job-status transitions, and `compiler._transition` which is a **narrative** transition (`"A moment later"`), not a camera one |

**Confirmed:** shot size and movement exist only as free text; **angle, lens, composition-as-a-field and every named transition do not exist at any layer.**

---

### B2 — camera exists only as `DirectorEvent.camera` · **CONFIRMED**

Every reference to a camera *value* in the entire product:

```
compiler.py:197   camera = _humanise(event.camera.strip(), plan)
compiler.py:199   sentences.append(_camera_sentence(camera))
compiler.py:511   def _camera_sentence(camera: str) -> str:
compiler.py:521   move = _normalise_move(move)
compiler.py:529   _MOVE_VERBS = re.compile(...)
compiler.py:538   def _normalise_move(move: str) -> str:
compiler.py:547   if not _MOVE_VERBS.match(text):
plan.py:419       camera = str(entry.get("camera") or "").strip()
plan.py:448       camera=camera,
provider.py:156   "camera": "<one of: medium shot | ... >"      (a string inside the planner brief)
provider.py:169   ... In "action" and "camera" text refer to characters by role words
```

**All ten are inside `worker/director/`.** Zero hits in `apps/api/app`, `apps/web/src`, `packages`, or the workflow YAMLs.

**Unvalidated** — [plan.py:419](../apps/worker/worker/director/plan.py#L419) is the entire parse:

```python
camera = str(entry.get("camera") or "").strip()
```

No membership check, no enum, no default, no rejection. `_parse_timeline` validates `start`, `end`, `speaker`, `dialogue`, `action` and `exits`; `camera` is only `str()`-coerced. `{"camera": "a purple flugelhorn"}` parses and reaches the prompt.

**Standard mode has no camera handling at all** — no field, no parse, no compiler branch. Confirmed by the grep above.

---

### B3 — camera state does not cross a seam · **CONFIRMED**

```
compiler.py:194        previous_camera = ""
compiler.py:198        if camera and camera.lower() != previous_camera.lower():
compiler.py:200            previous_camera = camera
```

`previous_camera` is a local initialised inside `_compile_section`, which is called once per section ([compiler.py:96-106](../apps/worker/worker/director/compiler.py#L96-L106)). Consequences:

- The first event of **every** section always emits a camera sentence, even when it repeats the last camera of the previous section.
- No section knows what shot the previous section ended on.
- The only cross-seam camera signal is the prose line *"Keep the same … camera direction established previously"* in the standard-mode header ([prompts.py:112](../apps/worker/worker/longform/prompts.py#L112)) — which Director mode does not emit at all.

---

### B4 — video-to-video never compiles a prompt · **CONFIRMED**

The five `_renderer` call sites:

| Handler | Line | Passes `prompt_for_step`? |
|---|---|---|
| `_run_generation` (t2v, i2v) | [1155-1159](../apps/worker/worker/adapters/ltx.py#L1155-L1159) | **yes** |
| `_run_extension` | [1289-1294](../apps/worker/worker/adapters/ltx.py#L1289-L1294) | **yes** |
| `_run_restyle` | [1468](../apps/worker/worker/adapters/ltx.py#L1468) | **no** |
| `_run_transform` | [1736-1743](../apps/worker/worker/adapters/ltx.py#L1736-L1743) | **no** |
| `_run_music_video` | [1899-1905](../apps/worker/worker/adapters/ltx.py#L1899-L1905) | **yes** |

The restyle call site in full:

```python
render=self._renderer(job, reporter, dimensions=grid, conditioning=conditioning),
```

So `_command` falls back to `job.prompt` at [ltx.py:2723](../apps/worker/worker/adapters/ltx.py#L2723):

```python
"--prompt", job.prompt if prompt is None else prompt,
```

**Every section of a video-to-video job receives byte-identical prompt text.** `video-to-video.yaml` also omits `prompt_structuring`, so no continuity block is appended either. A 180 s source on the transform engine is **23 sections, 22 seams, one unchanging prompt** and no camera handling of any kind.

---

### B5 — the standing rule contradicts an explicit camera request · **CONFIRMED**

**User prompt:**

```
A locked-off static camera on a tripod. The camera never moves. A woman stands at a window in a quiet room.
```

**Real `structure_prompt` output — what a single-pass job sends as `--prompt`, verbatim:**

```
A locked-off static camera on a tripod. The camera never moves. A woman stands at a window in a quiet room.

CONTINUITY (fixed for the entire video):
- The same subjects keep the same faces, clothing, colours and count in every frame.
- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.
```

```
  user asked for      : "The camera never moves."          -> True
  worker appended     : "the camera keeps moving ..."      -> True
```

**Real 2-section output — the contradiction lands in two different blocks:**

```
--- SECTION 1/2 ---
...
PERSISTENT USER CONSTRAINTS (verbatim):
A locked-off static camera on a tripod.
The camera never moves.
A woman stands at a window in a quiet room.
CONTINUITY (fixed for the entire video):
- The same subjects keep the same faces, clothing, colours and count in every frame.
NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY (verbatim):
- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.
...
```

Two aggravating details, both consequences of A5:

- The contradicting clause sits under **"NEW ACTION … FOR THIS SECTION ONLY"**, immediately above *"Complete this section's assigned dialogue before the section ends"* — i.e. presented as something to **perform**.
- It appears in **section 1 only**. Section 2 has no camera contradiction. The two halves of one video are given different camera instructions.

**Relevant context, from the repository's own reasoning** ([enhance.py:19-24](../apps/worker/worker/longform/enhance.py#L19-L24)):

> Negative phrasing. The distilled model has no negation mechanism (no negative prompt, no CFG), so "no cuts, no other cars" reads as "cuts, other cars" with extra steps.

By that same argument, the user's *"The camera never moves"* is a negation this runtime is documented as unable to act on, while the worker's *"the camera keeps moving"* is a positive assertion. **Recorded as fact; whether it dominates in the render is a GPU question.**

**Mode split:** STANDARD mode only, on t2v / i2v / extend / music-video.

---

### B6 — does a typed camera instruction survive? · **PARTIAL — yes in standard, no in Director**

**User prompt used:** `Locked-off static camera, low angle looking up at the subject. A woman stands at a window in a quiet room. She turns to face us.`

**STANDARD mode — survives verbatim, in every section.** Real `_separate` output:

```
PERSISTENT (reaches EVERY section):
   | Locked-off static camera, low angle looking up at the subject.
   | A woman stands at a window in a quiet room.
   | She turns to face us.
   | CONTINUITY (fixed for the entire video):
   | - The same subjects keep the same faces, clothing, colours and count in every frame.
ACTIONS (reaches ONE section only):
   | - One continuous scene: the camera keeps moving through the same environment, ...
```

The user's camera line lands in `PERSISTENT` and is repeated verbatim in every section. **It is never parsed as camera data** — it is opaque text that happens to be preserved. It is also never reconciled with the worker's own contradicting clause (B5).

**DIRECTOR mode — can be dropped entirely.** The user's text reaches only the **planner**, as the `IDEA:` line:

```
IDEA: Locked-off static camera, low angle looking up at the subject. A woman stands at a window in a quiet room. She turns to face us.
DURATION: 30 seconds
DIALOGUE LANGUAGE: the same language the idea is written in
TOTAL_LINES: write 8 spoken lines across the whole video. Not fewer than 7, and never more than 15.
TOTAL_WORDS: keep the spoken words under 55 in total, which is roughly 6 words per line.
```

`job.prompt` is **never** passed to `_command` in Director mode — `structure_prompt` is skipped ([ltx.py:1061](../apps/worker/worker/adapters/ltx.py#L1061)) and the caption is compiled entirely from the plan ([ltx.py:1118-1121](../apps/worker/worker/adapters/ltx.py#L1118-L1121)). So the user's camera words reach the model **only if the planner chose to encode them into `DirectorEvent.camera`**.

Three properties of the brief make that unreliable:

1. **A closed vocabulary that has no angle.** [provider.py:156-157](../apps/worker/worker/director/provider.py#L156-L157) offers `medium shot | medium close-up | close-up | two-shot | over-the-shoulder shot | wide shot; plus 'static' or a subtle move`. **"low angle" is not expressible.**
2. **No rule tells the planner to honour a camera request from the idea.** Verified programmatically — `'camera'` appears 4 times in the brief; **no line contains both "idea" and "camera"**.
3. **An active counter-instruction** ([provider.py:193-194](../apps/worker/worker/director/provider.py#L193-L194)): *"Dialogue needs readable faces: prefer medium shots, close-ups, two-shots and reaction shots, with a static camera or a subtle push-in. No fast camera moves."* A user asking for a sweeping crane move is instructing one system while the brief instructs the other in the opposite direction.

The only text the planner is *contractually* forbidden to drop is **quoted dialogue** (`required_quotes`, [plan.py:266-268](../apps/worker/worker/director/plan.py#L266-L268), enforced at [plan.py:818-842](../apps/worker/worker/director/plan.py#L818-L842)). There is no equivalent protection for camera, action, or setting.

---

## Symptom → cause table

| # | Symptom | Cause | Mode / workflow | Evidence |
|---|---|---|---|---|
| A-1 | A departed subject is asserted still present, in every pass | `- The same subjects keep the same faces, clothing, colours and **count in every frame**` reaches every section; nothing is presence-aware | **Standard**; t2v, i2v, extend, music-video | [enhance.py:125-127](../apps/worker/worker/longform/enhance.py#L125-L127); §A1 probe |
| A-2 | Same, in section 1 only, phrased as an action to perform | `- One continuous scene: … still present at the end` misclassified as a dialogue turn by `_DIALOGUE_LINE` | **Standard**; multi-section only | [prompts.py:63](../apps/worker/worker/longform/prompts.py#L63), [:148-150](../apps/worker/worker/longform/prompts.py#L148-L150); §A5 |
| A-3 | No mechanism exists to say "they left" | Exit tracking is Director-only; standard mode has no presence model | **Standard**; all four workflows | §A2 grep (41 hits, all in `worker/director/`) |
| A-4 | Worse in longer videos | Seams scale with length; each is a fresh subprocess carrying one PNG and no semantic state; overlap is 0.0 | **Both**; all chained workflows | §A3 table; [chain.py:168-172](../apps/worker/worker/longform/chain.py#L168-L172) |
| A-5 | I2V subject drifts after ~30 s | The source still is dropped at pass 2 on the only chaining duration (`720 ∉ {120,240,360}`) | **Both**; image-to-video 60 s, Director-lineage extend | §A4 dry-run — **⛔ blocked constants** |
| A-6 | V2V subject drifts across a 3-minute source | 22 seams, one unchanging prompt, no continuity block at all | **Both**; video-to-video transform | §A3, §B4 |
| B-1 | "Low angle" has no effect | The concept does not exist at any layer — 0 hits repo-wide | **Both**; all workflows | §B1 grep |
| B-2 | Camera has no effect in standard mode | No camera field, parse, or compiler branch exists outside `worker/director/` | **Standard**; all workflows | §B2 |
| B-3 | "Static camera" is undermined | The worker appends *"the camera keeps moving through the same environment"* to the same prompt | **Standard**; t2v, i2v, extend, music-video | §B5 real output |
| B-4 | Shot size drifts across a cut | `previous_camera` resets to `""` per section; no cross-seam camera state | **Director**; chained jobs | [compiler.py:194](../apps/worker/worker/director/compiler.py#L194) |
| B-5 | Camera request vanishes in Idea mode | User text reaches only the planner; closed vocabulary with no angle, no honour-the-idea rule, plus a counter-instruction | **Director**; t2v, i2v | §B6 |
| B-6 | Camera has no effect in V2V | No prompt compilation at all; identical prompt every section | **Both**; video-to-video | §B4 |

---

## Ranking

| Rank | Item | Why this rank |
|---|---|---|
| **P0** | **A5** — continuity rule read as a dialogue turn | A confirmed classifier bug on the default path of four workflows. Cheap to prove, cheap to fix, and it distorts every multi-section prompt. **Entangled with the A design decision — see Fix Plan.** |
| **P0** | **B5** — worker contradicts an explicit user camera request | The system asserts the opposite of what the customer typed, unconditionally, in the same prompt. Directly explains "static camera has no reliable effect". |
| **P1** | **A2 / A1** — standard mode has no presence model | The structural cause of symptom A. Director mode already solves it; standard mode has nothing. Requires a design decision, not a patch. |
| **P1** | **B2 / B6** — camera is Director-only, and droppable there | The structural cause of symptom B. Standard mode has no camera concept; Director mode's vocabulary cannot express an angle and has no rule to honour the user's. |
| **P2** | **A6 / B6(v2v)** — video-to-video sends one prompt to 22 sections | Affects the longest jobs the product accepts. Fixable by passing `prompt_for_step`, but that changes text reaching the model. |
| **P2** | **B4 / B3** — no cross-seam camera state | Real, but secondary to "camera does not exist at all" in standard mode. |
| **P3** | **A4** — identity anchor never fires | ⛔ **Blocked.** Both routes touch forbidden constants. Reported, stopped. |
| **P3** | **A3** — seam density | Not a defect; a consequence of measured ceilings. Listed because it modulates every other A symptom. |

---

## Fix plan — three lists, no work started

### 🟢 SAFE STATIC FIX — a genuine bug, no quality judgement required

**S1 · A5 — the `_DIALOGUE_LINE` misclassification** ([prompts.py:63](../apps/worker/worker/longform/prompts.py#L63), [:148-150](../apps/worker/worker/longform/prompts.py#L148-L150))

A worker-generated continuity rule is being read as a user-authored dialogue turn. This violates `prompts.py`'s own stated contract. The mechanism is unambiguous and the repair is mechanical.

> **Do not apply S1 alone.** The misclassification is currently the only thing limiting *"every subject present at the start is still present at the end"* to one section. Repairing it promotes that rule from **1 section** to **all N**, strengthening the assertion suspected of causing issue A. **S1 must be sequenced after D1.**

**S2 · Correct audit §C.7.** The audit names `_PERSISTENT_LINE`/the header as the cause; the operative regex is `_DIALOGUE_LINE` and the internal colon. Documentation-only, no behaviour change.

*(Nothing else qualifies. B5's contradiction is confirmed, but every remedy edits a continuity rule — see D1.)*

### 🟡 DESIGN DECISION — needs your approval before any edit

**D1 · The wording and scope of the two standing continuity rules** ([enhance.py:122-132](../apps/worker/worker/longform/enhance.py#L122-L132)) — **decide this first; S1 and D2 both depend on it.**

Four independent questions, listed without a recommendation:

1. Should *"every subject present at the start is still present at the end"* reach every section, one section, or none?
2. Should the presence clause and the camera clause stay in one bullet, or separate? They are currently one sentence, so the camera contradiction (B5) and the presence assertion (A1) cannot be tuned apart.
3. Should the camera clause be suppressed when the user's text already asserts a camera constraint?
4. Should the block header change so `_ALREADY_STRUCTURED` matches its own output? (Today `structure_prompt` would restructure its own output if called twice.)

**D2 · A presence model for standard mode.** Director mode's `exits` machinery ([plan.py:163-177](../apps/worker/worker/director/plan.py#L163-L177)) is the repository's own answer to your exact symptom, and it exists only in Director mode. Extending anything equivalent to standard mode is new behaviour on the default path.

**D3 · Whether camera becomes a real concept.** Adding a camera field (request schema, workflow YAML, or a structured field on the standard path), extending the Director vocabulary to include angle, or adding a rule to the brief that binds the planner to a camera request in the idea. All are additive product changes.

**D4 · Whether video-to-video should compile a prompt.** Passing `prompt_for_step` to the two v2v call sites would give each section its own text. That changes what reaches the model on a currently-stable path.

**D5 · Cross-seam camera state.** Carrying `previous_camera` across sections in the Director compiler.

**⛔ D6 · A4 — BLOCKED, not proposed.** Every route touches `_TWO_IMAGE_SAFE_FRAMES` or `max_segment_seconds`. Both are forbidden and both carry dated production-failure comments. **Stopped as instructed.**

### 🔬 GPU VALIDATION — questions only a render can answer

| # | Question | Why it cannot be answered statically |
|---|---|---|
| G1 | Does removing or rewording the presence rule reduce flash-back? | Whether the text causes the symptom, or merely coexists with it, is visual |
| G2 | Does the A5 repair (rule in every section) make flash-back better or worse? | The direction of effect is unknown; this is the risk that makes S1 unsafe alone |
| G3 | With the contradiction present, does the render obey the user or the worker? | Requires rendering the B5 prompt |
| G4 | Does removing the camera clause make "locked-off static" actually hold? | Visual |
| G5 | Does the Director shot vocabulary produce reliable shot sizes at all? | Requires a real planner **and** a render |
| G6 | Does seam density correlate with flash-back? (0 vs 1 vs 5 vs 22 seams) | Visual, and the cheapest experiment: same prompt, several `per_pass` values |
| G7 | Would restoring the I2V identity anchor help — and is a 720-frame two-image pass still fatal? | Both halves need the GPU; **and the second is the forbidden measurement** |
| G8 | Does per-section prompting improve or destabilise video-to-video? | Visual |

---

## Corrections to the existing audit

| Audit item | Correction |
|---|---|
| **§C.7** | The stated cause is wrong. The operative regex is `_DIALOGUE_LINE` (`^\s*[^:\n]{1,48}:\s+.+$`) matching the internal colon in `- One continuous scene:`, not `_PERSISTENT_LINE` failing on the header. The header's non-match is a separate, non-causal quirk. |
| **§2.11 (I2V) and §C.10** | Both say the anchor is dropped on 60 s I2V. Stronger statement: **60 s is the only I2V duration that chains**, so `_identity_anchor` never fires on any offered duration at the shipped configuration. |
| **§10.1** | Verified correct by exhaustive grep. No change. |

---

**PHASE 1 COMPLETE — STOPPED FOR APPROVAL.** No repository file has been modified. Nothing is staged or committed.
