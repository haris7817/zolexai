"""`validation.txt`, reduced to the checks a machine can actually make.

The file defines eleven groups at three levels, and most of them ask a reader
to judge a scene ("WARNING when emotion is expressed only as an abstract
label"). Those are guidance for the writer, and they are in the system prompt
where the writer can act on them. What is implemented here is the subset that
can be decided by looking at the returned text — and every one of those is a
rule whose breach would be invisible until a customer watched the video.

Two of them matter more than the rest, because they are the faults this whole
9 Sep 2026 exercise came from:

  * **A7 — dialogue invented with Auto Dialogue off.** The single most likely
    way an LLM rewriter damages this product: the customer asked for a scene,
    the writer decided the scene would be better with a line in it, and the
    video now says words nobody wrote. `worker/longform/language.py` exists
    because of the model-side version of this fault; this is the writer-side
    version, and it is an ERROR.
  * **S3 — settings leaking into prompt text.** `core.txt` §4 lists them and
    `validation.txt` §3 makes it an ERROR. A prompt that says "16:9, 30
    seconds, seed 42" spends real conditioning on tokens the graph already
    holds as parameters.

An ERROR sends the writer one corrective round naming the rules it broke, the
same shape as the dialogue writer's retry. Anything still failing falls open
to the customer's own prompt: this feature improves a prompt, and a prompt it
cannot improve is one it must not damage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from worker.prompt.ltx25.forms import Form, named_edits, quoted_lines

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    rule: str
    level: str
    detail: str


@dataclass
class Report:
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.level == ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.level == WARNING)

    @property
    def result(self) -> str:
        """`validation.txt`'s FINAL RESULT vocabulary, unchanged."""
        if self.errors:
            return "INVALID"
        return "VALID_WITH_WARNINGS" if self.warnings else "VALID"

    def instruction(self) -> str:
        """What to tell the writer so its second attempt is different."""
        return " ".join(f"{f.rule}: {f.detail}" for f in self.errors)


#: `core.txt` §4 and `validation.txt` §3: the settings that belong in the
#: request, never in prompt text. Written as patterns rather than bare words
#: because several are ordinary English on their own — a "steps" in "she
#: steps off the kerb" is not a sampler setting, and a rule that fired on it
#: would reject good prompts forever.
_SETTINGS_PATTERNS = (
    (r"\bltx[-\s]?2\.?5\b", "names the model"),
    (r"\bltxv?[-\s]?\d", "names the model"),
    (r"\b(?:16:9|9:16|4:5|1:1)\b", "states an aspect ratio"),
    (r"\b(?:720p|1080p|1440p|2160p|4k|uhd)\b", "states a resolution"),
    (r"\b\d{3,4}\s*[x×]\s*\d{3,4}\b", "states a pixel canvas"),
    (r"\b\d+\s*fps\b", "states a frame rate"),
    (r"\bframe\s+rate\b", "states a frame rate"),
    (r"\bseed\s*[:=]?\s*\d+", "states a seed"),
    (r"\bcfg\b", "states a guidance scale"),
    (r"\bguidance\s+scale\b", "states a guidance scale"),
    (r"\b\d+\s+steps\b", "states a step count"),
    (r"\bsampler\b", "names a sampler"),
    (r"\blora\b", "names a LoRA"),
    (r"\bupscaler\b", "names an upscaler"),
    (r"\baspect[_\s]ratio\b", "names an API field"),
    (r"\bnegative[_\s]prompt\b", "names an API field"),
    (r"\bmax(?:imum)?[_\s]speakers\b", "names an API field"),
    (r"\bauto[_\s]dialogue\b", "names an API field"),
    (r"\bdialogue[_\s]language\b", "names an API field"),
    (r"\b\d+\s*[-\s]?second\s+(?:clip|video|shot|generation)\b", "states the duration"),
)

#: Audio direction, for the silent-output check (`validation.txt` §7).
#:
#: Only words that can ONLY be about sound: "score" and "track" are absent
#: because a musical score and a score of people are the same string, and a
#: false positive here throws away a good rewrite.
#:
#: This list is deliberately incomplete and cannot be completed. "Rain drums
#: on the roof" is an audio direction and also a physical description, and no
#: word list separates them. The real defence is the instruction — a silent
#: job is told so in place of the audio guide, in words that leave no room
#: ("Write no ambience, music, speech, singing or sound effects of any kind")
#: — and this catches only what survives that.
_AUDIO_PATTERNS = (
    r"\bambien(?:ce|t)\b",
    r"\bsoundscape\b",
    r"\bsound\s+(?:of|effects?)\b",
    r"\broom\s+tone\b",
    r"\bmusic\b",
    r"\bwe\s+hear\b",
    r"\bis\s+heard\b",
    r"\baudible\b",
    r"\bvoice[- ]?over\b",
    r"\bnarrat(?:es|ion|or)\b",
    r"\bsays?\b",
    r"\bspeaks?\b",
    r"\bwhispers?\b",
    r"\bshouts?\b",
    r"\bsings?\b",
    r"\bsinging\b",
    r"\bmutters?\b",
    r"\bhums?\b",
    r"\bsoundtrack\b",
    r"\bdialogue\b",
)


def _hits(text: str, patterns) -> list[str]:
    out = []
    for entry in patterns:
        pattern, label = entry if isinstance(entry, tuple) else (entry, entry)
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            out.append(f"{label} ({match.group(0)!r})")
    return out


def check(
    text: str,
    form: Form,
    *,
    sound_on: bool = True,
    dialogue_allowed: bool = True,
    supplied_lines: tuple[str, ...] = (),
) -> Report:
    """Validate one rewritten prompt.

    `supplied_lines` are the exact spoken lines that legitimately belong in
    it — the customer's own quoted words, or Auto Dialogue's. A quoted line
    that is not one of them is an invention, which is `validation.txt` §7's
    first ERROR.
    """
    findings: list[Finding] = []
    body = text.strip()

    if not body:
        return Report((Finding("C1", ERROR, "the writer returned nothing"),))

    # ── §3 settings separation ──────────────────────────────────────────
    for hit in _hits(body, _SETTINGS_PATTERNS):
        findings.append(
            Finding("S3", ERROR, f"generation settings must stay out of the prompt — it {hit}")
        )

    # ── §1/§4 form selection ────────────────────────────────────────────
    edits = named_edits(body)
    if form is Form.SINGLE_SHOT and edits:
        findings.append(
            Finding(
                "F1",
                ERROR,
                f"a single continuous take must contain no edit, but it names {edits[0]!r}",
            )
        )
    if form is Form.MULTI_SHOT:
        if not edits:
            findings.append(
                Finding(
                    "F2",
                    ERROR,
                    "a multi-shot prompt must name every cut in words "
                    '(for example "a hard cut transitions to")',
                )
            )
        elif len(edits) > 3:
            findings.append(
                Finding(
                    "M5",
                    WARNING,
                    f"{len(edits) + 1} shots; the guide asks for two to four",
                )
            )

    # ── §7 audio and dialogue ───────────────────────────────────────────
    spoken = quoted_lines(body)
    if spoken and not dialogue_allowed:
        findings.append(
            Finding(
                "A7",
                ERROR,
                "nobody asked for dialogue, so the prompt must contain no spoken "
                f"lines, but it quotes {spoken[0]!r}",
            )
        )
    elif spoken and supplied_lines:
        known = {line.strip().strip(".!?").casefold() for line in supplied_lines}
        invented = [
            line for line in spoken if line.strip().strip(".!?").casefold() not in known
        ]
        if invented:
            findings.append(
                Finding(
                    "A7",
                    ERROR,
                    "the spoken lines are fixed and must appear word for word; "
                    f"{invented[0]!r} was not one of them",
                )
            )

    if not sound_on:
        for hit in _hits(body, _AUDIO_PATTERNS):
            findings.append(
                Finding(
                    "A4",
                    ERROR,
                    f"this video is silent, so the prompt must not direct sound — it says {hit}",
                )
            )
            break

    # ── §11 pipeline duplication ────────────────────────────────────────
    # The universal negative is submitted separately; a positive prompt that
    # repeats it as prohibitions spends conditioning saying the same thing
    # twice, in the box where "no" works least well.
    prohibitions = len(re.findall(r"\b(?:no|without|avoid|never)\s+\w+", body, re.IGNORECASE))
    if prohibitions >= 6:
        findings.append(
            Finding(
                "P11",
                WARNING,
                f"{prohibitions} prohibitions read like the negative prompt restated",
            )
        )

    return Report(tuple(findings))


__all__ = ["ERROR", "WARNING", "Finding", "Report", "check"]
