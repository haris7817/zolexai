"""No written language in the picture — a positive-prompt clause.

Client report, 10 Sep 2026, on a delivered file (`52032.mp4`): "It has only
H.264 video and AAC audio — there is no removable subtitle track. The
captions are burned into the generated pixels by LTX, which is why
`ffmpeg -sn` will not remove them."

That reading is right, and our own files corroborate the diagnosis rather
than merely permit it. The graph's negative prompt (node 5509, the client's
own text) **already** ends with "subtitles, captions, logos, watermarks", and
a caption still reached the frame. So the negative box is not where this can
be fixed, for the reason the client gives: the distilled workflow runs
CFG 1.0, and at CFG 1.0 there is no conditional/unconditional gap for a
negative prompt to act through. Whatever is going to constrain the picture
has to be in the positive box, because that is the only text the model reads.

## Why this sits outside the guideline pack

The pack the client sent on 9 Sep says the opposite of what they are asking
for here: `core.txt` §7 forbids duplicating the negative prompt in the
positive, and `validation.txt` §11 warns specifically against restating it as
prohibitions. This clause is a prohibition block restating the negative.

The instruction is explicit, from the customer, about a defect they have in
hand, and their mechanism is sound — so it ships. But it is composed onto the
prompt **after** `apply_guidelines` and **after** `add_auto_dialogue`, at the
compile itself, which is exactly what the client asked for ("after the prompt
enhancer, immediately before sending the workflow to ComfyUI") and is also
the only ordering that works here:

- the rewriter never sees it, so it cannot paraphrase it away, and
- `validate.check` never sees it, so our own §7 check cannot reject a prompt
  for carrying the clause the customer required.

## The one risk worth stating

Measured on this model 18 Aug 2026 and written up in
`docs/internal/ltx-director-idea-mode.md`: an uncovered caption tail is
narrated — the model reads trailing prompt text aloud rather than treating it
as an instruction. This clause is a trailing sentence about dialogue and
text. It needs one A/B on the node before anyone calls it settled; if it
narrates, the fix is to move the clause ahead of the description rather than
to drop it.
"""

from __future__ import annotations

#: The client's text, 10 Sep 2026, verbatim. Not paraphrased and not
#: "improved": it is the string they will read back off a prompt trace.
NO_VISIBLE_TEXT = (
    "No visible written language appears anywhere in the video. "
    "No subtitles, captions, closed captions, dialogue text, lower thirds, "
    "labels, signs, logos, or watermarks. Spoken dialogue is heard only "
    "through the audio and is never displayed visually."
)


def without_visible_text(prompt: str, *, allow_captions: bool = False) -> str:
    """`prompt` with the no-text clause appended, unless captions are allowed.

    `allow_captions=True` returns the prompt untouched — the escape hatch for
    a customer who genuinely wants a sign, a shopfront or a title in frame,
    which this clause would otherwise forbid.

    Appending to an empty prompt returns the clause alone rather than a
    leading space, and the clause is never added twice: a job that has already
    been through here (a retry, or a caller that composes in two places) keeps
    one copy.
    """
    text = (prompt or "").strip()
    if allow_captions or NO_VISIBLE_TEXT in text:
        return text
    return f"{text} {NO_VISIBLE_TEXT}".strip()


__all__ = ["NO_VISIBLE_TEXT", "without_visible_text"]
