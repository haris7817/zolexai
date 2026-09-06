# Extend Video — first/last frame on extension, and extending again

**Status: implemented and tested without a GPU (6 Sep 2026); the model's
behaviour with a customer's own first/last frame at an extension seam is
GPU validation pending (§10).**

Safety checkpoint: tag `pre-extension-first-last-frame` = commit `c50f6bf`
(the state the client was testing on). Rollback: §9.

## 1. Audit of the extension flow as it stood at `c50f6bf`

### 1.1 The complete flow, traced

| Step | Where | What happens |
|---|---|---|
| Generate (Image to Video) | `workflow-definitions/image-to-video.yaml` → `GenerationSettingsPanel` | Inputs `source_image` (FIRST FRAME, required) and `last_frame` (optional) come from the YAML; the panel renders one `Dropzone` per declared input. |
| Output | `GenerationService.complete` → `generation_job_outputs` | The worker's file is registered as a READY video asset and attached as the job's primary output. |
| Extend (button) | `ResultActions.tsx`, `GenerationDetail.tsx` | Rendered when `workflow.capabilities.extend && output`; a Next `<Link>` to `/app/create/extend-video?source=<asset_id>`. |
| Extension form | `app/create/[workflowId]/page.tsx` → `CreatorWorkspace` → `withSourceAsset` | The page reads `?source=` on the server and passes `initialSourceAssetId`; the form seeds the workflow's only required input (`source_video`) with it. A same-route hand-off (`?source=A` → `?source=B`) is handled by the `lastSourceAssetId` effect, which resets the form with the new id. |
| Extension job | `POST /generations` → `GenerationService.create` | `validate_request` checks duration (5/10/15/30), aspect, and that every input role is declared (`known_roles`) and every required one present; `_validate_inputs` checks ownership, READY status and asset kind per role. For `extend-video` the Director lineage lookup runs (`producing_job_for_asset`) and stores nothing unless the ancestor was a Director job. |
| Worker | `LtxComfyAdapter._run_extension` → `continue_video` | Probes the source, extracts its final frame, renders one pass of the client's First/Last Frame graph per section with that frame as `first_image` and **`last_image=None`**, drops the one overlap frame per seam, stitches the source in front, verifies the promised length, writes `continuation.json`. |
| Output → Extend again | same as above | The extension's output is a normal video asset of a job whose workflow has `extend: true`, so the button renders again and links to `?source=<new asset>`. |

Nothing in the UI, API or worker counts extensions. There is no
`extensionCount` anywhere; the chain is limited only by the source-file
ceiling on the Extend input (512 MB, "up to 30 minutes").

### 1.2 Finding A — why First/Last Frame is not offered on an extension

`workflow-definitions/extend-video.yaml` declares exactly one input,
`source_video`. The settings panel, the API's `known_roles` check and the
worker all follow the definition, so:

* the panel shows only the source video box;
* a request carrying `first_frame`/`last_frame` would be refused with
  "This tool does not accept those inputs";
* the adapter never sets the graph's `Load Image2` on an extension pass
  (`last_image=None` in `_run_extension`), and always conditions pass 0 on
  the frame it extracted from the source.

The compiler already supports both stills (`compile_first_last_frame` takes
`first_image` and an optional `last_image`, and `drop_last_frame` handles
the one-image case), and `render_pass` already accepts a `PassSpec` with
both — the integration layer simply never passes them for an extension.

### 1.3 Finding B — why "Extend" appears not to work after the first extension

Measured, not guessed:

1. **The route renders the hand-off on demand.** `next build` marks
   `/app/create/[workflowId]` as `● (SSG)`. Under the production server
   (`next start` on the pre-change build, API reachable on both loopback
   families), a hard navigation to `?source=A` shows the source box filled
   with A, and pressing the result's own Extend button (a client-side
   `<Link>` to `?source=B`) replaces it with B; a reload keeps B. The
   readiness report's item 7.6 ("does not fill under `next start`") does
   not reproduce; the earlier e2e failure was environmental (the browser's
   `localhost` resolves to `::1` and the API listened on `127.0.0.1` only).
2. **The live deployment already chained twice.** On zolexai.com on 6 Sep:
   Image to Video `c8c44a52` (30 s) → Extend `4132a1d3` (+30 s → 60.06 s) →
   Extend `b69ca03d` whose source is the first extension's own output
   (+30 s → 90.10 s). Both completed. The second extension's source asset id
   is the first extension's output id, so the same-route hand-off worked
   there at least once. No third extension was ever submitted.
3. **What the customer sees on that click is nothing.** A same-route
   hand-off changes only the query string. The canvas keeps showing the
   result they just pressed Extend on, with its Extend button still there;
   the only change on a desktop layout is the file name inside the source
   box in the right-hand panel. On tablet and phone layouts the panel is a
   closed drawer, so **nothing on screen changes at all**. Every other
   route change in the app repaints the title and the canvas; this one
   does not, which reads as a dead button.
4. **A stale-input hazard once stills exist on Extend.** The hand-off keeps
   `form.getValues()` (prompt, duration — wanted) and would therefore also
   keep a previously chosen first/last frame, so the next extension would
   silently end on the same picture again unless the customer noticed and
   removed it.

So B is a feedback problem in the workspace, not a limit in the API, the
worker or the route; the fix is to make the hand-off visible (and to reset
the framing inputs when a new source arrives), not to remove a counter that
does not exist.

### 1.4 State each extension carries today

| Needed (brief, Phase 4) | Where it is at `c50f6bf` |
|---|---|
| source video | `generation_job_inputs` role `source_video` |
| current duration | the source asset's `duration_seconds` (measured by the worker when it was produced) |
| requested extension | `request_params.duration` |
| first / last frame | not accepted (Finding A) |
| output location | `generation_job_outputs` → asset `storage_key` |
| parent job | derivable only: the job whose output the source asset is (`producing_job_for_asset`); stored on the row only for Director lineage |
| metadata | `continuation.json` in the worker's workspace — deleted with the workspace when the job ends |
| audio state | `request_params.sound`; the engine keeps the source's track and the generated track per pass, edge-faded at seams |

### 1.5 Limits that are real

* One pass is at most 30 s (`ltx_comfy_max_segment_seconds`, the graph's
  own slider maximum); a 30 s step is one pass, never more.
* The Extend input accepts up to 512 MB / "up to 30 minutes" of source. The
  engine re-encodes the whole source once per extension, so each step's
  wall clock grows with the source length (the render itself does not).
* GPU: the First/Last Frame graph peaked at 30.7 GB VRAM for a 5 s pass on
  the RTX PRO 6000 (`ltx25-gpu-benchmark.md`); no extension step needs more
  than one such pass.

## 2. Old behaviour → new behaviour

| | Before (`c50f6bf`) | After |
|---|---|---|
| Extend inputs | SOURCE VIDEO only | SOURCE VIDEO, plus optional FIRST FRAME and LAST FRAME (both images) |
| Pass 0 conditioning | always the source's extracted final frame | the customer's first frame when given, else the source's final frame |
| End of the continuation | wherever the model takes it | the customer's last frame when given, else as before |
| Pressing Extend on an extension result | URL changes, screen does not | canvas returns to its empty state, the source box shows the new file, the settings drawer opens on compact layouts |
| Framing inputs across a hand-off | would carry over | cleared; prompt and duration still carry over |
| Chain bookkeeping | parent derivable, not stored | `parameters.extension = {generation, parent_job_id, source_seconds}` stored with every extend-video job |
| Extension count | none | none (nothing added) |

What did NOT change: the client's graphs (`benchmarks/client-pack/ltx25/`,
sha256-pinned), the compiler, the models/LoRAs/samplers/schedulers, the
extension engine's seam arithmetic, the 5/10/15/30 s ladder, Video to
Video, Music Video, Music, Character Replacement, H3 and its routing.

## 3. The extension flow now

```text
Existing video (source_video, required)
   + optional FIRST FRAME (first_frame, image)
   + optional LAST FRAME  (last_frame, image)
   + 5 / 10 / 15 / 30 s
        │  POST /generations  (roles validated against the YAML; kinds checked;
        │  extension record stored)
        ▼
  worker: LtxComfyAdapter._run_extension
        │  probe source · upload last frame (if any) · seed = first frame (if any)
        │                                           else the source's final frame
        ▼
  continue_video (unchanged chain: one pass per ≤30 s section)
        │  pass k: client First/Last Frame graph
        │     Load Image1 = seed (k = 0) or previous pass's final frame
        │     Load Image2 = last frame, on the FINAL pass only
        │  drop 1 overlap frame per seam · stitch source in front · verify length
        ▼
  Extended video = a NEW asset on a NEW job; the source is never rewritten
        │
        ▼  Extend again (same button, same route, ?source=<new asset>)
```

## 4. First frame, last frame, neither

**Neither.** Byte-for-byte the previous behaviour: the source's final frame
is extracted, uploaded as the only image, `num_images = 1`, and the run
proceeds as before. Pinned by
`test_no_stills_means_the_run_before_this_feature`.

**First frame.** The picture the continuation starts on, instead of the
frame the video happened to end on. The engine is seeded with it (the
source's final frame is not extracted at all), it is uploaded as pass 0's
`Load Image1`, and `continuation.json` records `first_frame` and the pass's
`conditioning_frame`. The overlap policy is unchanged: the graph renders the
still itself at index 0 and that one frame (1/24 s) is dropped, so the
delivered length is exactly source + step. Use: an edited final frame (a
prop added, a colour fixed), or a deliberate cut to a new framing.

**Last frame.** The picture the continuation ends on. Uploaded once, up
front (a broken picture fails the job before any GPU time is spent), and
handed to the FINAL pass as `Load Image2`; the graph's own two-image wiring
(`num_images.index_2 = -1`) is what runs. On a chained multi-pass
continuation (not in the current ladder, where every step is one pass)
only the last pass ends on it.

**Both.** Pass 0 starts on the first frame, the final pass ends on the last;
for a one-pass step they bracket the same pass, exactly as Image to Video
with both stills does.

The stills are resized by the graph to its canvas (the ZIP's behaviour); the
canvas is the product ratio closest to the source video's own frame, and
the engine fits every part to the source's dimensions afterwards.

## 5. Unlimited chained extensions

"Unlimited" means unlimited extension *jobs*, each a separate generation
of at most 30 s of new material, never one long inference. `original → +30
→ +30 → +30 → …` is `test_an_extension_can_be_extended_again_without_limit`
(four generations deep in the test; nothing in the code knows the number).

Every extension is its own job with its own output; the source asset and
its producing job are untouched (the API test checks the original still has
exactly its one output after four extensions, and that the original can
still be extended in parallel — a chain is not a lock).

`parameters.extension` on every extend-video job:

| key | meaning |
|---|---|
| `generation` | 1 for the first extension of any video, 2 for an extension of an extension, and so on |
| `parent_job_id` | the user's own completed job whose output the source is; `null` for an uploaded source |
| `source_seconds` | the source's measured length, so the delivered total (source + step) is known at creation |

It is bookkeeping resolved once at creation (`GenerationService._extension_record`),
not a gate: nothing reads it to refuse anything.

In the workspace, pressing Extend on an extension result hands the result
over on the same route; `handOverSource` seeds the source, keeps prompt and
duration, clears any first/last frame from the previous step, the canvas
returns to its empty state and, where the settings panel is a drawer, it
opens. The previous result stays one click away in the job strip.

## 6. Hardware and workflow limits

Nothing here changes resolution, model, LoRA, sampler, scheduler, workflow
or precision. The limits are the ones measured on 5–6 Sep 2026:

* A pass is ≤ 30 s (graph slider maximum). The ladder is 5/10/15/30 s.
* First/Last Frame pass, 5 s: 63.8–66.9 s wall, 30.7 GB VRAM peak; a 30 s
  T2V pass 215 s / 34.5 GB. A 30 s extension step is one such pass plus a
  re-encode of the source and the stitch.
* The source is re-encoded once per step, so a step on a 5-minute source
  costs minutes of ffmpeg on top of the render; the input caps at 512 MB.
* Character Replacement's 10 s ceiling is unrelated and unchanged.

## 7. Failure behaviour

* A still that cannot be decoded fails the job **before any submission**
  (`test_an_unreadable_last_frame_fails_before_any_render`): no upload, no
  render, non-retriable, customer copy "One of the selected images could
  not be read."; the source asset is untouched.
* A pass that dies mid-render is the same retriable failure as before; no
  assembled output exists for the runner to upload.
* A failed or cancelled extension leaves the source exactly as extendable
  as before: the next Extend of the same video is accepted, and the chain
  record counts only completed ancestors
  (`test_a_failed_or_cancelled_extension_does_not_lock_the_chain`).
* The worker's fail-fast for a dead ComfyUI (`1554bc4`) applies to
  extension passes as to every other pass.
* Nothing disables future extensions on failure — there is no state to
  disable.

## 8. Tests performed (6 Sep 2026, no GPU)

New:

* `apps/worker/tests/test_extension_stills.py` — 6 tests against the fake
  ComfyUI with real ffmpeg files: first frame replaces the source's final
  frame; last frame goes to the final pass only (two-section chain);
  both stills on one pass; no stills = the previous run; unreadable still
  fails before any render; failed pass with stills is retriable and leaves
  no output.
* `apps/api/tests/test_extension_chain.py` — 5 tests against real
  PostgreSQL/Redis: the catalogue offers the two optional stills; the
  request contract (source required, stills optional, unknown role
  refused); a four-deep chain with the record counting 1–4 and the original
  untouched and still extendable; stills stored by role and delivered to
  the worker with download URLs, wrong-kind still refused; failed and
  cancelled steps do not lock the chain.
* `apps/web/scripts/qa-e2e.mjs` §8c — the Extend tool shows FIRST FRAME /
  LAST FRAME optional, and pressing a result's Extend replaces the source
  and empties the canvas.

Existing, re-run: see the run log at the end of this file (§11).

Browser checks under the production build (`next build` + `next start`,
Playwright): §11.

## 9. Rollback

```bash
# Everything back to the exact state the client was testing on:
git checkout dual-engine-benchmark-prep
git reset --hard pre-extension-first-last-frame     # = c50f6bf
# (or, without moving the branch: git checkout pre-extension-first-last-frame)
```

Deploy after a rollback exactly as for any other commit
(`vps-deploy-procedure`): fetch as `zolexai`, checkout, `deploy/vps-local.sh
--profile client-test`, rebuild api + web, restart the GPU worker.

Nothing was deleted: the previous behaviour is the "no stills" path, and
the old hand-off (`withSourceAsset`) is still what a fresh visit uses.

## 10. GPU validation pending

Everything above was proven without a model. Pending on the RTX PRO 6000
(steps in `gpu-validation-checklist.md` style, ≈ 15 minutes):

1. ~~Extend a finished result by 5 s with **no stills**~~ — **DONE on the
   live deployment, 6 Sep 14:38 UTC:** Text to Video `c43bba93` (15 s, 9:16)
   → Extend `6d21b6f8` (+5 s, no pictures): pass 65.5 s, job 70 s end to
   end, output 704×1280 · 20.063 s (promised 20.042), worker log
   `extension_framing first_frame=false last_frame=false`, one pass, seam
   at 15.042 s; chain record `{generation: 1, parent_job_id: c43bba93,
   source_seconds: 15.042}`. A second step on that output (`2bca0283`) was
   accepted with `generation: 2` and queued behind the client's own 30 s
   Image to Video, then cancelled by a person while still queued — not a
   failure. The seam itself was not inspected frame by frame.
2. Extend with a **first frame** that is an edited copy of the source's
   final frame — expect the edit visible from the first continuation frame
   and the seam otherwise continuous.
3. Extend with a **last frame** — expect the final frames to converge on the
   still, as Image to Video with a last frame does.
4. Extend with **both**.
5. Extend the result of (4) again with no stills — the chain continues from
   the delivered last frame.

Record wall clock and VRAM for each in `ltx25-gpu-benchmark.md`.

## 11. Run log (6 Sep 2026, dev machine, no GPU)

| Check | Result |
|---|---|
| Web `tsc --noEmit` | clean |
| Web `eslint src --max-warnings=0` | clean |
| Web `next build` | clean; `/app/create/[workflowId]` still `● (SSG)`, six create pages prerendered — the route was not changed |
| Browser, production build (`next start`), desktop 1440×900 | `?source=A` fills the source box; the Extend tool shows FIRST FRAME optional and LAST FRAME optional; pressing the result's own Extend swaps the source to that result's file and empties the canvas |
| Browser, production build, phone 390×844 | same, and the settings drawer opens on the seeded form |
| Browser, live zolexai.com (read-only) | `?source=<first extension's output>` fills the source box on the deployed build; the client's chain 30 → 60 → 90 s is on record |
| API suite (real PostgreSQL/Redis, `-p no:randomly`) | **136 passed, 3 skipped** (was 131 / 3; +5 new); the 3 skips are the Director-mode tests, unreachable since Director left the product |
| Worker targeted (`test_continuation`, `test_ltx_comfy`, `test_first_last_frame_adapter`, `test_untouched_runtimes`, `test_extension_stills`) | **55 passed** |
| Untouched-module guards (`test_untouched_workflows`, `test_untouched_runtimes`, `test_character_replacement`) | pass — V2V, Music, Music Video, Character Replacement YAML and the CLI/H3 adapters are hash-identical |
| `git diff --stat pre-extension-first-last-frame HEAD -- <protected paths>` | empty: no change under `benchmarks/client-pack/`, `worker/comfy/`, `worker/providers/`, the other six YAMLs, `adapters/ltx.py`, `adapters/h3_comfy.py`, `adapters/character_replacement.py`, `deploy/`, `apps/web/src/app/` |
| Worker full suite (`-p no:randomly`, 24 min) | **1067 passed, 10 failed, 1 skipped**; the 10 failures are exactly the pre-existing set recorded in the readiness report §7.7 (3 Director "never touches the planner", 5 H3 video-to-video, 2 music-video) — none involves the extension code |
| Live deployment, Text to Video → Extend (6 Sep 14:38 UTC) | `c43bba93` (15 s) → `6d21b6f8` +5 s, no pictures: completed in 70 s, 20.063 s output, generation 1; the Extend button on a finished Text to Video result hands the file to the Extend tool with both pictures optional (Playwright, read-only) |
| `qa:e2e` harness (built web + local API + mock worker) | **27 / 27 checks pass.** Two harness defects fixed on the way: §8b sampled the source box before the asset lookup resolved (the "two remaining failures" of 5 Sep, wrongly attributed to the route), and expected Generate enabled with an EMPTY prompt on a tool whose prompt is required. §8c's "press the result's Extend" step is skipped by the harness when no Extend result is among the 8 most recent jobs (the mock T2V jobs it just created push it out); that click is proven by the direct Playwright run above on desktop and phone |
