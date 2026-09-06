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
    "skin, waxy skin, CGI appearance, cartoon, illustration, anime"
)

DEFAULT_NEGATIVE: dict[str, str] = {
    "text-to-video": TEXT_TO_VIDEO_NEGATIVE,
    "image-to-video": FIRST_LAST_FRAME_NEGATIVE,
    "extend-video": FIRST_LAST_FRAME_NEGATIVE,
    "character-replacement": CHARACTER_REPLACEMENT_NEGATIVE,
}


def negative_for(workflow_id: str, execution: dict) -> str:
    override = execution.get("negative_prompt")
    if isinstance(override, str) and override.strip():
        return override.strip()
    return DEFAULT_NEGATIVE.get(workflow_id, FIRST_LAST_FRAME_NEGATIVE)


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


def character_replacement_prompt(description: str, *, skin: str | None = None) -> str:
    """The pack's lead sentence, then the customer's description of the new
    character (which the sample prompt shows is what carries identity).

    With `skin` (the hands clause, chained jobs only) it goes between the two;
    without it the text is byte for byte what it always was."""
    description = description.strip()
    parts = [CHARACTER_REPLACEMENT_LEAD]
    if skin and skin.strip():
        parts.append(skin.strip())
    if description:
        parts.append(description)
    return " ".join(parts)
