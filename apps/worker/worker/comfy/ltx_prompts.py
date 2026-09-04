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


def character_replacement_prompt(description: str) -> str:
    """The pack's lead sentence, then the customer's description of the new
    character (which the sample prompt shows is what carries identity)."""
    description = description.strip()
    if not description:
        return CHARACTER_REPLACEMENT_LEAD
    return f"{CHARACTER_REPLACEMENT_LEAD} {description}"
