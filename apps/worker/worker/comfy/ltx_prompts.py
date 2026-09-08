"""House prompt text for the client's LTX 2.5 graphs.

The pack's positive prompts are sample-specific and are replaced by the
customer's text per job. Its negative prompts are half sample-specific
(one boxing match's crowd, one man's navy blazer) and half generic quality
guards; the generic halves are kept here, verbatim from the ZIP, and the
scene-specific halves are dropped. A deployment may override any of them
with `execution.negative_prompt`.

Nothing here is a redesign of the graphs: the text boxes are user inputs,
and these are the values a job puts in them.
"""

from __future__ import annotations

import re

#: Graph 02's negative prompt, verbatim. Short and scene-free, so it is also
#: what Text to Video ships when a deployment says nothing.
FIRST_LAST_FRAME_NEGATIVE = (
    "blurry, low quality, still frame, frames, watermark, overlay, titles, "
    "has blurbox, has subtitles"
)

#: Graph 01's negative prompt minus its boxing-specific lines (crowd,
#: referee, gloves, ring). What remains are the generic quality, anatomy,
#: motion, camera and overlay guards the pack's author used.
TEXT_TO_VIDEO_NEGATIVE = (
    "cartoon, anime, illustration, CGI, videogame look, plastic skin, blurry "
    "image, low resolution, noise, compression artifacts, oversharpening, "
    "identity drift, face changing, body changing, wardrobe flicker, duplicate "
    "person, extra person, missing person, deformed anatomy, extra arms, extra "
    "legs, extra hands, fused fingers, broken wrists, twisted limbs, warped "
    "torso, stretched neck, detached limbs, rubber arms, sliding feet, "
    "floating, teleporting, flying, impossible movement, robotic motion, stiff "
    "movement, repeated action, frozen frames, background morphing, moving "
    "walls, camera shake, crash zoom, whip pan, spinning camera, unreadable "
    "action, excessive motion blur, random cuts, jump cuts, montage, scene "
    "changes, lighting flicker, strobing, overexposure, crushed shadows, color "
    "shifting, subtitles, captions, logos, watermark, text overlays"
)

#: The Ripple LoRA's documented default prompt (graph 03's note, verbatim).
#: Every character-replacement prompt starts with it; the customer's own
#: description of the new character follows.
CHARACTER_REPLACEMENT_LEAD = (
    "Use the reference video for motion, timing, camera movement, composition, "
    "and unchanged scene content, while consistently propagating the visual "
    "edit established in the first frame throughout the video."
)

#: Graph 03's negative prompt minus the sample's source-person description
#: (clean-shaven, navy blazer, yellow wall, Spanish caption). What remains is
#: the pack's list of replacement failure modes: source leakage, identity
#: drift, anatomy, motion, camera, background, lighting and overlay faults.
CHARACTER_REPLACEMENT_NEGATIVE = (
    "source actor identity, source-video face, captions, subtitles, text "
    "overlay, typography, watermark, logo, source background leakage, source "
    "wardrobe leakage, wrong character, different person, additional person, "
    "duplicate person, identity drift, face morphing, changing facial "
    "structure, changing jawline, changing nose, changing lips, changing age, "
    "skin-tone shifts, facial-hair flicker, beard-shape drift, hair flicker, "
    "hairstyle drift, hairline drift, changing hair length, unstable hair "
    "volume, accessory disappearance, accessory duplication, warped eyewear, "
    "floating jewelry, jewelry flicker, wardrobe replacement, clothing "
    "morphing, clothing color shifts, body-proportion drift, changing shoulder "
    "width, changing head size, stretched face, compressed face, detached head, "
    "twisted neck, broken anatomy, extra arms, missing arms, duplicate arms, "
    "extra hands, missing hands, fused hands, malformed hands, extra fingers, "
    "missing fingers, fused fingers, broken wrists, rubber limbs, impossible "
    "joints, hands clipping through clothing, frozen pose, still image, stop "
    "motion, repeated frames, duplicated frames, frame skipping, reverse "
    "motion, looping motion, jerky movement, robotic movement, twitching, "
    "sudden pose jumps, motion drift, incorrect timing, exaggerated gestures, "
    "unnatural body movement, random mouth movement, lip-sync errors, warped "
    "lips, distorted teeth, cross-eyed gaze, asymmetrical eyes, camera shake, "
    "camera drift, sudden zoom, random pan, random tilt, unwanted reframing, "
    "crop changes, subject leaving frame, abrupt perspective changes, "
    "background geometry changes, unstable horizon, lighting flicker, exposure "
    "pumping, brightness shifts, color flicker, white-balance shifts, jumping "
    "shadows, temporal inconsistency, ghosting, double exposure, motion trails, "
    "smearing, edge tearing, halos, warping, melting, glitch, compression "
    "artifacts, pixelation, low resolution, soft focus, excessive blur, "
    "oversharpening, oversaturation, posterization, artificial skin, plastic "
    "skin, waxy skin, CGI appearance, cartoon, illustration, anime, "
    # The client's own additions (7 Sep 2026), and the reason they matter:
    # every exposure term this list already carried was SYMMETRICAL —
    # "brightness shifts", "exposure pumping", "skin-tone shifts" name a
    # change without naming a direction, and the fault we measured only
    # ever goes one way. Naming the direction is what the client's graph
    # added, and it costs nothing to carry.
    "darker face, darker person, darkening skin, underexposure, crushed "
    "blacks, lighter skin, inconsistent face color, inconsistent hand color"
)

# ── The universal negative ─────────────────────────────────────────────────

#: The client's universal negative prompt (8 Sep 2026), verbatim.
#:
#: It arrived with an instruction to paste it into the graph's empty negative
#: box, which is not what this deployment needed — every workflow here already
#: submits a negative prompt derived from the pack's own graphs. What it is
#: worth is what it carries that ours did not: the temporal terms
#: ("temporal flicker", "frame-to-frame inconsistency", "texture crawling"),
#: the continuity terms ("unintended color changes", "paint color changes"),
#: and the physics terms ("broken physics", "floating objects"). So it is
#: composed IN FRONT of each workflow's own list rather than replacing it,
#: and `_dedupe` drops the large overlap so no term is spent twice.
#:
#: What it will not do is track an object. Negative conditioning suppresses
#: named failure modes; it does not give the model a handle on the thing that
#: must stay the same. The identity clauses in the positive prompt still do
#: that work, and this is stacked on top of them.
UNIVERSAL_NEGATIVE = (
    "temporal flicker, frame-to-frame inconsistency, identity drift, face "
    "changes, body changes, clothing changes, unintended color changes, paint "
    "color changes, color flicker, exposure flicker, shape morphing, geometry "
    "warping, duplicated subjects, disappearing subjects, extra people, extra "
    "objects, fused subjects, deformed anatomy, extra limbs, missing limbs, "
    "malformed hands, extra fingers, distorted faces, unstable backgrounds, "
    "background warping, texture crawling, floating objects, broken physics, "
    "unnatural movement, jerky motion, camera jitter, camera teleportation, "
    "abrupt scale changes, inconsistent lighting, inconsistent shadows, "
    "inconsistent reflections, random scene changes, blurry details, "
    "compression artifacts, unwanted text, captions, subtitles, logos, "
    "watermarks"
)

#: Terms in the house negatives that name a STYLE rather than a fault.
#:
#: They belong in a negative prompt for a photoreal subject and are actively
#: wrong for a drawn one: a Character Replacement job whose new character is
#: a cartoon submits "cartoon, illustration, anime" as things to AVOID while
#: the positive prompt asks for exactly that. The client's 8 Sep 2026 render
#: is the case — a Simpsons character came back with a dark grey face — and
#: this is the half of that report that lands in our pipeline, since the
#: ReActor/SAM face lock they also blamed exists only in their own graph and
#: has never run here.
STYLE_TERMS = frozenset(
    {
        "cartoon",
        "anime",
        "illustration",
        "cgi",
        "cgi appearance",
        "videogame look",
        "artificial skin",
        "plastic skin",
        "waxy skin",
    }
)


def _terms(text: str) -> list[str]:
    return [term.strip() for term in text.split(",") if term.strip()]


def _dedupe(terms: list[str]) -> str:
    """First occurrence wins, comparison case-insensitive.

    A term repeated across two lists is not stronger for being repeated; it
    is one more token in a budget the encoder truncates, and the client's
    universal list overlaps ours heavily by design.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for term in terms:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        kept.append(term)
    return ", ".join(kept)


def compose_negative(*blocks: str, drop_style: bool = False) -> str:
    """The blocks joined, deduplicated, style terms optionally removed."""
    terms = [term for block in blocks if block for term in _terms(block)]
    if drop_style:
        terms = [term for term in terms if term.casefold() not in STYLE_TERMS]
    return _dedupe(terms)


DEFAULT_NEGATIVE: dict[str, str] = {
    "text-to-video": TEXT_TO_VIDEO_NEGATIVE,
    "image-to-video": FIRST_LAST_FRAME_NEGATIVE,
    "extend-video": FIRST_LAST_FRAME_NEGATIVE,
    "character-replacement": CHARACTER_REPLACEMENT_NEGATIVE,
}


def negative_for(
    workflow_id: str, execution: dict, *, drop_style: bool = False
) -> str:
    """The negative prompt this job submits.

    `execution.negative_prompt` still replaces the workflow's own list — that
    is what a deployment override is for. What it no longer replaces is the
    universal block, which is composed in front of whichever list wins unless
    `execution.universal_negative` turns it off. A deployment that wants the
    exact pre-8-Sep-2026 text sets that to false.

    `drop_style` removes the terms that name a look rather than a fault, for
    a subject that IS that look. See `STYLE_TERMS`.
    """
    override = execution.get("negative_prompt")
    house = (
        override.strip()
        if isinstance(override, str) and override.strip()
        else DEFAULT_NEGATIVE.get(workflow_id, FIRST_LAST_FRAME_NEGATIVE)
    )
    universal = execution.get("universal_negative")
    if universal is not None and str(universal).strip().lower() in ("false", "no", "off", "0"):
        return compose_negative(house, drop_style=drop_style)
    return compose_negative(UNIVERSAL_NEGATIVE, house, drop_style=drop_style)


#: The hands clause for CHAINED character replacement (7 Sep 2026). Measured
#: on the client's clip: the face holds the photo's skin tone but the hands
#: darken inside every window and each seed carries it on — the photo shows
#: the hands small and at another pose, the video guide re-synthesises them
#: every frame, and nothing in the text spoke about them. This is the one
#: signal that speaks DURING a window. Relational on purpose (no colour is
#: named, so it cannot pull any character's skin the wrong way) and worded
#: "whenever they are in view" so it never asks for hands the source does not
#: show. Sits between the pack's lead sentence and the customer's own words.
CHARACTER_REPLACEMENT_SKIN = (
    "The character established in the first frame is one person from head to "
    "hands: the hands, wrists and any other bare skin have exactly the same "
    "skin tone as the face and are lit the same way as the face whenever they "
    "are in view - gesturing, coming close to the camera, crossing in front of "
    "the face, leaving and re-entering the frame. That skin tone, on the face "
    "and on the hands alike, stays exactly the same from the first frame to "
    "the last frame."
)


#: The exposure lock, from the client's MULTI4 graph (7 Sep 2026). Their
#: positive prompt carries two sentences this one condenses: the source's own
#: light is preserved ("lighting direction, shadows, ... exposure, contrast,
#: saturation, and white balance"), and the character's skin holds what the
#: first frame established ("Faces, necks, arms, and hands retain the
#: brightness and color established in the edited first frame").
#:
#: RELATIONAL, like `CHARACTER_REPLACEMENT_SKIN` beside it and for the same
#: reason: it names no colour and no brightness, only sameness. A clause that
#: said "bright" or "light skin" would pull every character towards one
#: complexion, which is a worse fault than the one being fixed.
#:
#: Directional on purpose in its second half. Everything the negative prompt
#: carried before today named a CHANGE ("brightness shifts", "exposure
#: pumping") where the measured fault only ever goes one way, and a
#: symmetrical word gives an unguided runtime nothing to lean against.
CHARACTER_REPLACEMENT_EXPOSURE = (
    "The light in the scene is the reference video's own: the same lighting "
    "direction, the same shadows, the same exposure, contrast, saturation and "
    "white balance from the first frame to the last. The character's face, "
    "neck, arms and hands keep exactly the brightness and colour they have in "
    "the first frame and never grow darker as the video goes on."
)


# ── Drawn characters ───────────────────────────────────────────────────────

#: Words in a Character Replacement prompt that say the new character is not
#: a photographed human.
#:
#: A heuristic, and named one. It exists because the alternative is asking a
#: customer to classify their own reference picture, and because every switch
#: it flips is one that does nothing useful on a drawn character and
#: something harmful on one: the style terms in the negative prompt fight the
#: look the prompt asked for, and the skin machinery
#: (`worker/media/skin.py`) keys on the classic human skin chroma range —
#: which a bright cartoon face sits close enough to for the gate to fire on
#: it and pull it toward a human skin level.
#:
#: `execution.subject_style` overrides it in either direction, so a wrong
#: guess is one deployment line to fix rather than a code change.
STYLISED_PATTERNS = (
    r"\bcartoons?\b",
    r"\banime\b",
    r"\bmanga\b",
    r"\bcomic(?:\s*book)?\b",
    r"\banimated\b",
    r"\bcel[- ]?shaded\b",
    r"\b2d\s+character\b",
    r"\bclaymation\b",
    r"\bstop[- ]motion\b",
    r"\bpuppet\b",
    r"\bmuppet\b",
    r"\bcaricature\b",
    r"\bsimpsons?\b",
    r"\bpixar\b",
    r"\bdisney\b",
    r"\bdreamworks\b",
    r"\bstudio ghibli\b",
    r"\billustrat(?:ed|ion)\b",
    r"\bhand[- ]drawn\b",
    r"\bdrawn\b",
)


def looks_stylised(prompt: str) -> bool:
    return any(re.search(pattern, prompt, re.IGNORECASE) for pattern in STYLISED_PATTERNS)


#: The exposure lock for a drawn character.
#:
#: `CHARACTER_REPLACEMENT_EXPOSURE` names skin — "face, neck, arms and hands"
#: — because it was written for the fault measured on a photographed person.
#: A drawn character has none of those as the model understands them, and the
#: same guarantee has to be asked for in terms the picture actually has:
#: the character's own colours, flat and unshaded, holding what the first
#: frame set. Relational for the same reason as its twin — it names no colour.
CHARACTER_REPLACEMENT_EXPOSURE_STYLISED = (
    "The light in the scene is the reference video's own: the same lighting "
    "direction, the same shadows, the same exposure, contrast, saturation and "
    "white balance from the first frame to the last. The character keeps "
    "exactly the colours they have in the first frame - the same face colour, "
    "the same hands, the same outfit, flat and evenly lit - and no part of "
    "them ever grows darker, greyer or duller as the video goes on."
)


def character_replacement_prompt(
    description: str,
    *,
    skin: str | None = None,
    exposure: str | None = None,
) -> str:
    """The pack's lead sentence, then the customer's description of the new
    character (which the sample prompt shows is what carries identity).

    With `skin` (the hands clause, chained jobs only) and `exposure` (the
    client's lighting lock) those go between the two, in that order; with
    neither the text is byte for byte what it always was.

    Order is deliberate. The customer's own description goes LAST, because
    the sample prompt shows it is what carries identity, and the constraints
    read as conditions on the character it introduces rather than the other
    way round."""
    description = description.strip()
    parts = [CHARACTER_REPLACEMENT_LEAD]
    if skin and skin.strip():
        parts.append(skin.strip())
    if exposure and exposure.strip():
        parts.append(exposure.strip())
    if description:
        parts.append(description)
    return " ".join(parts)
