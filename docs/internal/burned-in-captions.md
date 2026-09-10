# Burned-in captions (10 Sep 2026)

Client report, on a delivered file (`52032.mp4`): "It has only H.264 video and
AAC audio — there is no removable subtitle track. The captions are burned into
the generated pixels by LTX, which is why `ffmpeg -sn` will not remove them."

That reading is correct, and it is worth being precise about why, because the
obvious fix is the one that cannot work.

## What it is not

**It is not a backend overlay.** Audited 10 Sep across the whole repo:

```
rg -n "drawtext|subtitles=|libass|\.srt|\.ass|burn_subtitles"
```

No `drawtext`, no `subtitles=`, no `libass`, nothing that composites text onto
a frame. The only `.srt` hits are the Music lyrics sidecar file a customer
downloads (`worker/music/workflow.py`), which is written next to the video and
never into it.

**It is not a missing negative prompt.** This is the part that decides the
fix. The client's own graph already carries, at node 5509, a negative prompt
ending:

> …continuity errors, unexplained scene changes, accidental cuts, repeated
> frames, **subtitles, captions, logos, watermarks**, interface elements…

A caption reached the frame anyway. So the negative box is not underspecified;
it is inert. The client's explanation is the right one: the distilled workflow
runs **CFG 1.0**, and at CFG 1.0 there is no conditional/unconditional gap for
a negative prompt to act through. Adding more words to 5509 would change
nothing.

## The fix, and where it sits

A no-text clause in the **positive** prompt — the only text the model reads at
CFG 1.0. `worker/prompt/no_text.py`, the client's wording verbatim, default
on (`LTX_ALLOW_CAPTIONS=false`).

It is composed onto the prompt **last**: after `apply_guidelines`, after
`add_auto_dialogue`, at the compile itself. That is what the client asked for
("after the prompt enhancer, immediately before sending the workflow to
ComfyUI") and it is the only ordering that works:

- the guideline rewriter never sees it, so it cannot paraphrase it away;
- `validate.check` never sees it, so our own validator cannot reject a prompt
  for carrying the clause the customer required.

**This is a deliberate exception to the client's own guideline pack.**
`core.txt` §7 forbids duplicating the negative prompt in the positive, and
`validation.txt` §11 warns specifically against restating it as prohibitions.
This clause is exactly that. It ships because the instruction is explicit,
from the customer, about a defect they have in hand, and their mechanism is
sound — but it is an exception, and the pack has not changed.

A customer who genuinely wants a sign, a shopfront or a title in frame sets
`allow_captions` on the job; the clause forbids those too.

## The A/B this needs before anyone calls it fixed

**Agreed with the client 10 Sep: run this first, and build OCR only if it
fails.** The prompt clause is prevention and costs nothing per job; the OCR +
inpainting pipeline is remediation and costs a second model on a node that has
already been OOM-killed once at its cgroup ceiling (7 Sep,
`gpu-worker-memory-ceiling`).

Run on `ltx-6000-1`, **same seed both arms**, on a prompt that has produced
captions — a dialogue-heavy one, since that is where they appear.

| arm | setting |
| --- | --- |
| A (control) | `LTX_ALLOW_CAPTIONS=true` |
| B (the fix) | `LTX_ALLOW_CAPTIONS=false` — the default |

Check three things on the output of each, not one:

1. **Captions gone?** The question asked. Eyeball the frames where dialogue
   lands; that is where they appeared before.
2. **Is the clause being narrated?** The known failure mode of a trailing
   prompt tail on this model — measured 18 Aug 2026,
   `ltx-director-idea-mode.md`: the model reads uncovered trailing text aloud
   instead of treating it as an instruction. Listen to the audio for the words
   "no visible written language". **If it narrates, the clause is not wrong —
   move it ahead of the description and re-run.** Do not drop it.
3. **Did the picture change otherwise?** The clause is a prohibition block in
   the positive box, which the guideline pack says costs prompt budget. Compare
   framing and subject against arm A.

Read back what was actually sent: the job log's `positive` field is the exact
string the graph received, clause and all.

## If it fails — what building OCR would mean

The client's design: OCR caption detection → mask expansion → temporal
inpainting (ProPainter) → upscale. Before the upscale, which is right — text
removal after enlargement is slower and leaves bigger artifacts.

What it costs us today: the worker has **no OpenCV, no OCR engine and no
inpainting** — checked 10 Sep. It is PaddleOCR or EasyOCR plus ProPainter
weights and inference on the GPU node, a new stage between decode and
upscale, and a second model resident during a job on a node that is already
memory-constrained.

There is a cheaper middle option worth doing first if the A/B is only
partially good: **the OCR gate alone, no inpainting.** Sample N frames after
decode, detect text, and fail-and-regenerate rather than repair. One
dependency instead of two, no inpainting VRAM during the render, and it turns
a silent defect into a retry.

Neither helps `52032.mp4` or any other file already rendered. Pixels that
already contain text can only be repaired by inpainting; no backend change
reaches them.
