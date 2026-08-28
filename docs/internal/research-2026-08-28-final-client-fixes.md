# The 28 Aug 2026 client verdict — four complaints, four causes, four fixes

The client's message, verbatim and unedited, because the wording is the
evidence:

> Bro video to vodeo still bad no reference image. I put a video and press
> better and give me a whole different video
> Music videos still doing glitches
> Age to video dont follow the age that I put as reference
> Look at the video to vide that I create because I can't download them
> Video to video bro bad in best and standard. I can[']t download anything so
> check generations

Four distinct reports. Three of them have a single mechanical cause each, and
the fourth turns out to be a consequence of the first rather than its own bug.

---

## 1. "Press better and give me a whole different video" — CONFIRMED, by design

Not a malfunction. The Best level was routed to the H3 R2V graph, and that
graph reads **images**. The mapping was the one the proven D1 run used:

| Slot | What it got |
| --- | --- |
| Picture 1 | the customer's reference photo (identity) |
| Picture 2 | **the first frame of the source video** (environment) |
| Picture 3 | disconnected |

Everything after frame 0 of the customer's footage — every movement, every
cut, every thing that made them upload a video rather than a photo — was
never read. The engine then generated a fresh performance. It is a good
engine; it is a good *different product*. The adapter's own docstring said so
plainly: *"the source video's motion is not re-enacted; that remains LTX
transform's job."*

Shipping that behind a control labelled "Best" on a workflow called Video to
Video was the mistake. A customer who uploads footage and presses the better
button is not asking for a second opinion on their idea.

**Fix.** Both quality levels now run LTX transform, the engine that drives
every rendered frame from a canny edge map of the source. The Fast/Best
difference becomes what the engine is asked to do:

* **Fast** — restyle the footage from the prompt. No matting pass.
* **Best** — additionally replace the person with the reference photo:
  BiRefNet mattes the people in each pass and the reference is re-shown
  inside that region, so the face is the photo's and the motion stays the
  footage's. ~27 s per pass extra. GPU-verified 19 Aug 2026.

Three code changes carry it:

* `settings.h3_comfy_video_to_video` (default **False**) — H3's `supports()`
  declines Video to Video. Flip the env var to reach the R2V path again for a
  benchmark.
* `resolve_adapter` gains a safety net: a quality level pointed at an engine
  that cannot serve the workflow degrades to the base runtime with a
  `runtime_by_quality_unsupported` warning, instead of failing a customer's
  job. This matters because the mapping lives in **deployment-local YAML that
  no test in this repository can see**.
* `execution_by_quality` — a per-quality overlay on the execution block, the
  companion to `runtime_by_quality`. Video to Video uses it to give Best
  `v2v_reference_identity: true` and leave Fast a plain restyle.

**Not attempted, and why.** The H3 Extender node does expose `ref_video_1`
(an IMAGE batch) and the provider documents up to three video references, so
a genuinely video-conditioned H3 path is likely possible. It needs a video
loader wired into a graph the pack freezes and forbids restructuring, and it
cannot be validated anywhere but on the GPU box. That is a next-milestone
experiment, not a fix to hand a client the same week.

## 2. "Doesn't follow the age I put as reference" — CONFIRMED, and it is one word

Every path that carries a person across chained sections describes them to
the model in text, because pixel conditioning alone decays over a chain. The
describer's brief asked for the person's **"age group"**, and the measured
production answer on 19 Aug was:

```
Woman: adult, dark hair, black leather jacket
```

"Adult" is not an age. A video model handed a prompt with no age in it casts
from its own prior, and that prior is a photogenic twenty-something,
whatever the photograph showed. The client uploaded an older reference and
got someone younger back — exactly what that sentence asks for.

**Fix**, in `worker/director/vision.py`:

* the brief now demands the sentence **begin** with an age in years ("a man
  of about 55") plus what visibly carries it — grey hair, a receding
  hairline, lines, weathered or youthful skin, a child's proportions — and
  explicitly forbids the words *adult*, *young adult*, *middle-aged*;
* `_pin_age` repairs a caption that fell back to a band anyway, translating
  it to a number rather than discarding a sentence whose hair and clothing
  are still worth having. A caption already carrying a digit is untouched.

## 2b. Image to Video never described its person at all

Found while fixing the above, and it is the more likely home of the client's
report. `_run_generation` builds each section's prompt from
`plan_section_prompts(job.prompt, …)`. The uploaded still conditions frame 0
and rides later passes as a low-strength anchor — **and the prompt text never
mentions the person once.** Over a 20 or 30-second chain the text prior is
unopposed.

Music video solved this on 28 Aug by captioning its anchor and restating who
the person is in every later section. I2V now does the same, gated on
`execution.i2v_describe_identity` (default on), one caption per video, and
every failure path returns `""` so the prompt reads exactly as it did before.

## 3. "Music videos still doing glitches" — CONFIRMED, and it is the prompt

The two 60-second renders in `benchmarks/review/client-e2e/` (27 Aug, same
pipeline, same track, ~12 minutes apart) settle this:

| | prompt | result |
| --- | --- | --- |
| `24_mv_detailed_prompt.mp4` | names the singer, her hair, her shirt, the microphone, the room | one woman, one wardrobe, one room, mouth on the vocal, for the full minute |
| the THIN-prompt run (`25`/`26`) | a mood and a time of day | a man in a purple suit on a rooftop becomes a woman in a pink one; every shot far enough away that no face is ever legible |

Two mechanisms, both now addressed.

**The shot planner reached for distance.** `intro`, `bridge` and `outro` all
had wide and overhead options, and a song that opens instrumentally
deterministically drew `("a wide establishing shot", …)` for section 1. That
costs twice: the viewer never sees a face on a workflow whose promise is a
mouth moving with the vocal, and **section 1's final frame is the identity
anchor every later section inherits** — an anchor with no legible face hands
the next eight sections nothing to hold.

`_SUBJECT_HELD` is the same three roles reframed to end each move *on the
performer*. `plan_shots(hold_subject=…)` selects it; passing None derives it
from the vocal analysis, so an instrumental track is still free to go wide.

**The prompt field asked for nothing.** The placeholder said "Describe the
visual direction…", which is exactly the prompt that fails. It now asks who
is on screen and where, and carries the working example. The worker can
defend a thin prompt; it cannot recover a description that was never given.

## 4. "I can't download anything" — a symptom, plus one real UI bug

The Download button renders on `capabilities.download && output`. A job with
no output has no button, so *"I can't download them"* is what a **failed**
Video to Video job looks like from the customer's side — and two V2V jobs
failed outright on the evening of 27 Aug, refused by the H3 adapter for
having no reference photo. Fixing §1 removes the cause.

There is a genuine bug underneath it, though. `handleDownload` had a
`try/finally` and no `catch`: a failed signing request left the button saying
"Preparing…", returning to "Download", and doing nothing — indistinguishable
from a blocked navigation, and reported exactly the way the client reported
it. It now surfaces the failure.

**Still to check on production, and it needs a shell I do not have:**

```sql
select g.id, g.workflow_id, g.status, g.error_code, g.created_at,
       o.asset_id, a.status as asset_status, a.content_type, a.size_bytes
from generation_jobs g
left join generation_job_outputs o on o.job_id = g.id
left join assets a on a.id = o.asset_id
where g.workflow_id = 'video-to-video'
order by g.created_at desc limit 20;
```

Three readings and what each would mean:

* `status = failed` → §1, already fixed.
* `status = succeeded`, `asset_status <> 'ready'` → the worker's upload
  confirm is failing; only READY assets are given a URL.
* `content_type = 'image/png'` on a video row → the deployed
  `video-to-video.yaml` lost its mock-output deletion in a `git stash pop`
  (runbook §16, §44.1). The API would then be signing PNG uploads for MP4
  output. Verify with:

```bash
docker exec zolexai-prod-api-1 \
  grep -n "output_content_type\|output_kind\|runtime" \
  /workflow-definitions/video-to-video.yaml
```

---

## What is NOT claimed here

None of this has been run on a GPU. Every change is either a routing decision,
a prompt-text change, or a UI error path, and each is written to degrade to
previously shipped behaviour on any failure — but "written to degrade safely"
is not "measured". The GPU checklist is
`docs/internal/client-final-validation-2026-08-28.md`.
