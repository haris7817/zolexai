# Audio safety rule

**Client instruction, 28 Aug 2026.** Recorded verbatim because it is a
product rule, not an implementation detail:

> Never convert, narrate, read, or speak the visual prompt, system prompt,
> negative prompt, continuity rules, or backend instructions. Only generate
> speech when the user explicitly provides dialogue. If dialogue is empty,
> generate no voices, narration, or spoken words. For music requests, output
> instrumental music and sound effects only.

## One correction to the diagnosis it came with

The instruction was accompanied by "the backend instructions must never be
sent to the audio generator". There is no audio generator to withhold them
from. Both video engines produce picture and sound in a **single pass from a
single prompt** — LTX's command builder takes `--audio-path` (conditioning on
an audio *file*) and has no audio-prompt flag; H3's graph is driven entirely
by its segment prompts. The audio hears whatever the video hears, always, and
no amount of routing changes that.

So the rule cannot be enforced by withholding text. It is enforced by
*stating* it, in the one prompt both halves read.

## How it is enforced

`worker/longform/language.py`, appended to every standard-mode section prompt:

| the customer's prompt | appended |
| --- | --- |
| no quoted dialogue | `No one speaks. The only sounds are the ones the scene itself makes.` |
| quoted dialogue | `The only words anyone speaks are the quoted lines above, spoken in <language> by the people on screen in natural conversational voices.` |

**Quoted text is the only trigger.** A speech verb is not: "a woman explains
the product" describes what she is *doing*, not what she *says*, and a model
handed a verb with no words either invents them or reads the caption. That is
stricter than the speech-verb list it replaced, and it is what the rule asks
for.

## Why "no one speaks", when this file avoids negations everywhere else

The runtime has no negation mechanism, so a negative usually contributes the
noun it was trying to suppress. This one earns its exception on measurement:
Director mode's compiler emits exactly that phrasing, and its narration leak
went away (`research-2026-08-18-director-idea-mode`, TC2c). Evidence beats the
general rule.

The second sentence is not decoration. "Ambience" was the first wording, and
ambience *means* room tone: the Director research measured its
described-silence tail at −45 dBFS, inaudible on a phone, and the client
reported that same evening as "video dont have the sound". Naming the sounds
the scene MAKES points the soundtrack at events instead of at the room.

## Scope, and what is deliberately untouched

Applies to the standard path of text-to-video, image-to-video and
extend-video, on both engines.

- **Director mode is excluded.** Its plan writes real quoted dialogue and its
  compiler already owns its own tail; a second opinion appended there would
  contradict the words beside it.
- **Music video is excluded.** Its soundtrack is the customer's uploaded
  track, muxed once at the end — nothing generated reaches the viewer's ears.
- **The music workflow is excluded.** It generates songs from lyrics the
  customer supplied or explicitly asked the platform to write, which is
  "dialogue explicitly provided" by any reading. The rule's instrumental-only
  clause is about *video* requests growing unrequested vocals, and forcing it
  here would delete a shipped feature.

`execution.soundscape: false` disables the clause per workflow. It exists
because an earlier wording of this sentence measurably damaged the picture,
and a prompt sentence that can hurt the render needs an off switch that is
not a code change.
