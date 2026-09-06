# Extend Video — first/last frame on extension, and extending again

**Status: IN PROGRESS.** Section 1 is the pre-change audit (Phase 1 of the
6 Sep 2026 brief), written before any code moved. The remaining sections
are filled in as the phases land.

Safety checkpoint: tag `pre-extension-first-last-frame` = commit `c50f6bf`
(the state the client is testing on). Rollback: §9.

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
