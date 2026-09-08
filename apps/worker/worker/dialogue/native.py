"""The client's native-dialogue format (`native_dialogue.py`, 8 Sep 2026).

Their package's second revision, after reviewing a render from the first one
(test video 50920): eight isolated phrases, ~28 words for 30 seconds, a
different voice label on every line, "After a short pause" spoken aloud, and
gaps of 1.3–3 s between lines because the prompt licensed scene-only sound
between them. Every one of those has a rule here, and the rules are theirs:

  * total spoken words in a per-duration range (`WORD_RANGES`);
  * every line a complete thought of at least three words;
  * no repeated lines, no invented characters — only visible speakers;
  * ONE stable voice description per speaker, stated once, never per line;
  * NO cues and NO timing labels in the prompt: turns are "says" then
    "replies", and pacing comes from one sentence about brief pauses;
  * ambience stated once as continuous UNDER the voices, never as a sound
    that fills space BETWEEN lines.

What LTX gets is one screenplay-style positive prompt, composed exactly as
`compose_native_ltx_prompt` composes it, for one native audio-video pass.
Nothing here synthesises, replaces or muxes a voice.

One departure from the package, deliberate: it rewrites the customer's prompt
into `visual_prompt` (their instruction to the writer: keep the story, strip
duration / resolution / aspect / API labels). That is what fixes "vertical
video (16:9)" and a shot list that stops at 10 s — both were in the
customer's own text. This module keeps that rewrite, and the caller keeps
fail-open: if no plan can be validated, the customer's prompt renders as
written, exactly as before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Total spoken words per clip length, the client's table. Between rows the
#: same slope is used (1.6–2.27 words per second), so a length the table does
#: not list gets the range the table implies rather than a refusal.
WORD_RANGES: dict[int, tuple[int, int]] = {
    8: (12, 18),
    10: (16, 22),
    15: (24, 34),
    30: (48, 68),
}
MAX_SPEAKERS = 4
MIN_WORDS_PER_LINE = 3
MAX_REPLY_PAUSE_MS = 250


class NativeDialogueRejected(ValueError):
    """The writer's script fails one of the client's validation rules."""


@dataclass(frozen=True)
class NativeSpeaker:
    speaker_id: str
    visual_identity: str
    voice_description: str


@dataclass(frozen=True)
class NativeTurn:
    speaker_id: str
    text: str


@dataclass(frozen=True)
class NativePlan:
    visual_prompt: str
    ambience: str
    speakers: tuple[NativeSpeaker, ...]
    turns: tuple[NativeTurn, ...]

    @property
    def total_words(self) -> int:
        return sum(len(turn.text.split()) for turn in self.turns)


def word_range(seconds: float) -> tuple[int, int]:
    """The client's row for this length, or the range their table implies."""
    whole = int(round(seconds))
    if whole in WORD_RANGES:
        return WORD_RANGES[whole]
    low = max(MIN_WORDS_PER_LINE, int(round(seconds * 1.6)))
    high = max(low + 2, int(round(seconds * 2.27)))
    return low, high


def clean_text(value: Any) -> str:
    """One line of plain text with no quotation marks of its own — a quote
    inside a quoted line ends the line early when the prompt is read."""
    text = " ".join(str(value or "").replace("\n", " ").split())
    return text.strip(' "“”').replace('"', "").replace("“", "").replace("”", "")


def _normalised(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def validate_script(raw: dict[str, Any], seconds: float, *, max_speakers: int = MAX_SPEAKERS) -> NativePlan:
    """`validate_script` from the client's package, rule for rule."""
    visual_prompt = clean_text(raw.get("visual_prompt"))
    ambience = clean_text(raw.get("ambience"))
    raw_speakers = raw.get("speakers")
    raw_turns = raw.get("dialogue_turns")

    if not visual_prompt or not ambience:
        raise NativeDialogueRejected("visual_prompt and ambience are required")
    if not isinstance(raw_speakers, list) or not raw_speakers:
        raise NativeDialogueRejected("at least one speaking character is required")
    if len(raw_speakers) > max_speakers:
        raise NativeDialogueRejected("too many speaking characters")

    speakers: list[NativeSpeaker] = []
    for item in raw_speakers:
        if not isinstance(item, dict):
            raise NativeDialogueRejected("every speaker must be a JSON object")
        speaker = NativeSpeaker(
            speaker_id=clean_text(item.get("speaker_id")).lower().replace(" ", "_"),
            visual_identity=clean_text(item.get("visual_identity")),
            voice_description=clean_text(item.get("voice_description")),
        )
        if not all((speaker.speaker_id, speaker.visual_identity, speaker.voice_description)):
            raise NativeDialogueRejected("every speaker needs identity and one voice description")
        speakers.append(speaker)
    ids = {speaker.speaker_id for speaker in speakers}
    if len(ids) != len(speakers):
        raise NativeDialogueRejected("speaker_id values must be unique")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise NativeDialogueRejected("dialogue_turns cannot be empty")

    turns: list[NativeTurn] = []
    seen: set[str] = set()
    for item in raw_turns:
        if not isinstance(item, dict):
            raise NativeDialogueRejected("every dialogue turn must be a JSON object")
        turn = NativeTurn(
            speaker_id=clean_text(item.get("speaker_id")).lower().replace(" ", "_"),
            text=clean_text(item.get("text")),
        )
        if turn.speaker_id not in ids:
            raise NativeDialogueRejected("dialogue turn uses an unknown speaker")
        if len(turn.text.split()) < MIN_WORDS_PER_LINE:
            raise NativeDialogueRejected("dialogue lines must contain at least three words")
        key = _normalised(turn.text)
        if key in seen:
            raise NativeDialogueRejected("repeated dialogue line detected")
        seen.add(key)
        turns.append(turn)

    # A speaker who never speaks is a character the prompt would describe
    # for nothing; the client's Squidward spoke without ever appearing, and
    # the mirror failure is describing someone who never speaks.
    spoken = {turn.speaker_id for turn in turns}
    plan = NativePlan(
        visual_prompt=visual_prompt,
        ambience=ambience,
        speakers=tuple(s for s in speakers if s.speaker_id in spoken),
        turns=tuple(turns),
    )
    low, high = word_range(seconds)
    if not low <= plan.total_words <= high:
        raise NativeDialogueRejected(
            f"dialogue must contain {low}-{high} total words; received {plan.total_words}"
        )
    return plan


def _spoken(text: str) -> str:
    body = text.strip()
    return body if body and body[-1] in ".!?" else body + "."


def compose_native_prompt(plan: NativePlan, language: str = "English") -> str:
    """`compose_native_ltx_prompt` from the client's package, verbatim in
    shape. The voice lock is stated once per speaker; the turns carry no
    manner, no cue and no clock; the only pacing instruction is the one
    sentence about brief pauses; and ambience is continuous UNDER the voices.
    """
    voice_locks = " ".join(
        f"{speaker.speaker_id} is {speaker.visual_identity} and always speaks with "
        f"the same {speaker.voice_description}."
        for speaker in plan.speakers
    )
    turns = " ".join(
        (
            f'{turn.speaker_id} says, "{_spoken(turn.text)}"'
            if index == 0
            else f'{turn.speaker_id} replies, "{_spoken(turn.text)}"'
        )
        for index, turn in enumerate(plan.turns)
    )
    return (
        f"{plan.visual_prompt} {voice_locks} The spoken language is "
        f"{language or 'English'}. {turns} The dialogue is delivered as one fluent "
        "continuous performance in the exact order written, with complete sentences, "
        "natural conversational pacing, consistent voices, accurate synchronized lip "
        "movement, and only brief natural pauses no longer than "
        f"{MAX_REPLY_PAUSE_MS} milliseconds. No line repeats and no "
        f"additional words are spoken. {plan.ambience} The environmental audio remains "
        "continuous underneath the voices. No subtitles, captions, or on-screen text."
    )


def system_prompt(seconds: float, *, max_speakers: int = MAX_SPEAKERS) -> str:
    """The client's writer instruction, with the word range for this length."""
    low, high = word_range(seconds)
    return (
        "Create one coherent screenplay prompt for native audio-video generation. "
        "Return JSON only with visual_prompt, ambience, speakers, and dialogue_turns. "
        "Keep the user's visual story, characters, cuts, wardrobe, lighting, and "
        "camera intentions. Remove written duration, resolution, aspect-ratio, and "
        f"API-setting labels from visual_prompt. Detect one to {max_speakers} visible speaking "
        "characters. Each speaker object needs speaker_id (person_a through person_d), "
        "visual_identity, and one stable voice_description containing age range, "
        "register, timbre, accent, and pace. Do not assign a different emotion or "
        "voice style to every line. Dialogue turns need only speaker_id and text. "
        "Write one connected monologue or natural conversation with complete "
        "sentences. Every response must logically answer or advance the previous "
        "line. No isolated catchphrases, filler, repeated ideas, narration labels, "
        "timing labels, quotation marks inside text, or invented new characters. "
        "Only characters who are visible on screen may speak. "
        f"Total spoken words must be {low}-{high}, and every line must be at least "
        f"{MIN_WORDS_PER_LINE} words. If the scene has no visible human or human-like "
        "character who could plausibly speak, return {\"has_speaker\": false}."
    )


def user_prompt(scene_prompt: str, seconds: float, language: str = "English") -> str:
    return (
        f"Language: {language or 'English'}\n"
        f"Planning duration: {seconds:g} seconds\n\n"
        f"Scene request:\n{scene_prompt.strip()}"
    )


__all__ = [
    "MAX_REPLY_PAUSE_MS",
    "MAX_SPEAKERS",
    "MIN_WORDS_PER_LINE",
    "NativeDialogueRejected",
    "NativePlan",
    "NativeSpeaker",
    "NativeTurn",
    "WORD_RANGES",
    "clean_text",
    "compose_native_prompt",
    "system_prompt",
    "user_prompt",
    "validate_script",
    "word_range",
]
