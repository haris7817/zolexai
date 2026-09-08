from __future__ import annotations

from .models import EditorialPlan, MusicVideoRequest, Performer, Shot


CAMERAS = {
    "environment": "glides slowly along the shoreline with a level horizon",
    "ending_environment": "drifts gently backward as the scene settles into stillness",
    "close_performance": "makes a restrained slow push-in and keeps both eyes sharp",
    "medium_performance": "slides gently sideways while maintaining stable portrait framing",
    "wide_movement": "tracks parallel at a matching pace and keeps the complete body visible",
    "profile_portrait": "holds steady with a subtle forward drift and preserves profile separation",
    "seated_reflection": "moves gradually closer at eye level without changing the composition",
    "firelight_performance": "makes a slow controlled arc while preserving the face and firelight",
    "detail_hands": "moves a few inches laterally while keeping the hands anatomically readable",
    "detail_jewelry": "eases forward slightly while keeping the jewelry attached and sharply defined",
    "detail_object": "holds low and stable with a gentle parallax movement",
}


ACTIONS = {
    "environment": "Small waves spread across reflective sand and recede naturally while a light breeze moves nearby foliage.",
    "ending_environment": "A final thin wave crosses the footprints and slowly recedes, leaving a calm reflective surface.",
    "close_performance": "The performer delivers the current vocal phrase with natural mouth articulation, controlled breathing, subtle head movement and believable eye focus.",
    "medium_performance": "The performer sings toward the lens with relaxed shoulders and one restrained open-hand gesture below the chest.",
    "wide_movement": "The performer takes several natural unhurried steps through the environment with balanced posture and realistic foot placement.",
    "profile_portrait": "The performer looks across the environment, breathes calmly, then returns their attention toward the camera without singing.",
    "seated_reflection": "The performer rests naturally, looks down briefly, then raises their gaze with a thoughtful expression and minimal movement.",
    "firelight_performance": "The performer delivers the vocal phrase with relaxed intensity as warm firelight moves subtly across the face and clothing.",
    "detail_hands": "The hands make one small natural gesture while rings, tattoos and skin details remain fixed and unchanged.",
    "detail_jewelry": "The performer touches the necklace once; the chain follows the body with realistic weight and remains unchanged.",
    "detail_object": "The personal object remains grounded as wind, reflections or nearby fabric create subtle natural motion.",
}


GROUP_ACTIONS = {
    "medium_performance": "Each assigned member performs their stated vocal or instrumental role with complementary natural movement, clear separation and believable eye lines.",
    "wide_movement": "The complete assigned band performs each stated vocal or instrumental role together with grounded rhythmic movement, natural spacing and no member blocking another member's face.",
    "firelight_performance": "Each assigned member performs their stated vocal or instrumental role with restrained intensity while maintaining individual poses and clear facial visibility.",
}


GROUP_CAMERAS = {
    "medium_performance": "slides gently sideways while keeping every assigned performer visible in balanced ensemble framing",
    "wide_movement": "tracks the complete ensemble at a matching pace and keeps every assigned person fully visible",
    "firelight_performance": "makes a slow controlled arc while keeping every assigned face separated, visible and consistently lit",
}


LIGHTING = {
    "opening": "soft golden-hour light",
    "introduction": "warm late-afternoon sunlight",
    "development": "clean tropical daylight with soft reflected fill",
    "emotional_middle": "sunset fading into blue-hour light",
    "climax": "deep blue night balanced with warm practical or firelight",
    "ending": "quiet cool dawn light",
}


def _performer_map(request: MusicVideoRequest) -> dict[str, Performer]:
    return {item.id: item for item in request.performers}


def _performer_label(performer_id: str, performers: dict[str, Performer]) -> str:
    performer = performers.get(performer_id)
    if not performer:
        return performer_id
    details = [f"role: {performer.role.strip() or 'performer'}"]
    if performer.description.strip():
        details.append(performer.description.strip())
    return f"{performer_id} ({'; '.join(details)})"


def _solo_role_action(performer: Performer | None, default: str) -> str:
    if not performer:
        return default
    role = f"{performer.role} {performer.description}".casefold()
    if "drum" in role or "percussion" in role:
        return "The performer plays the drum kit with believable stick contact, coordinated hands and feet, natural rebounds and controlled rhythmic energy."
    if "bass" in role:
        return "The performer plays the bass with believable fretting and picking, grounded posture and natural rhythmic movement."
    if "guitar" in role:
        return "The performer plays the guitar with believable fretting and strumming, secure instrument contact and natural rhythmic movement."
    if "keyboard" in role or "piano" in role:
        return "The performer plays the keyboard with readable coordinated hands, natural posture and restrained rhythmic movement."
    if "dj" in role:
        return "The performer works the DJ controls with small precise hand movements while following the rhythm naturally."
    return default


SECTION_ORDER = ("opening", "introduction", "development", "emotional_middle", "climax", "ending")


def _location_for_section(section: str, locations: list[str]) -> str:
    if not locations:
        return "cinematic environment"
    try:
        section_index = SECTION_ORDER.index(section)
    except ValueError:
        section_index = 2
    if len(locations) == 1:
        return locations[0]
    location_index = round(section_index * (len(locations) - 1) / (len(SECTION_ORDER) - 1))
    return locations[location_index]


def anchor_prompt(shot: Shot, request: MusicVideoRequest, treatment: dict, output_format: str) -> str:
    locations = treatment.get("locations", ["cinematic environment"])
    location = _location_for_section(shot.section, locations)
    lighting = LIGHTING.get(shot.section, "coherent cinematic light")
    lyric_beat = (
        f" Narrative meaning: {shot.lyric_visual_direction}."
        if shot.lyric_visual_direction
        else ""
    )
    reference_style = str(treatment.get("reference_style_prompt", "")).strip()
    reference_clause = (
        f" Transfer only this reference video's general filmmaking style: {reference_style}. "
        "Create a new composition and do not copy its people, logos, dialogue, lyrics, locations or exact shot sequence."
        if reference_style
        else ""
    )
    if shot.performer_ids:
        performers = _performer_map(request)
        identities = "; ".join(_performer_label(item, performers) for item in shot.performer_ids)
        subject_label = "performer" if len(shot.performer_ids) == 1 else "distinct assigned performers"
        exact_people = "one assigned performer" if len(shot.performer_ids) == 1 else f"all {len(shot.performer_ids)} assigned performers"
        anatomy = (
            "Use relaxed plausible starting poses with clear anatomy, readable hands, natural spacing and no occlusion between faces."
            if len(shot.performer_ids) > 1
            else "Use a relaxed plausible starting pose with clear anatomy and readable hands."
        )
        return (
            f"Create one photorealistic cinematic starting frame for a music video using {subject_label} {identities} and each person's supplied identity references. "
            f"Composition: {shot.family.replace('_', ' ')} in {location} under {lighting}.{lyric_beat}{reference_clause} Preserve the exact face, natural asymmetry, skin texture, "
            f"hairstyle, body proportions, wardrobe, tattoos, jewelry and accessories of every assigned person; keep their identities separate and never blend or swap faces. {anatomy} "
            f"Compose natively for {output_format} with safe headroom and no cropped anatomy. Show exactly {exact_people}, with no extras, text, logos, borders or collage layout."
        )
    inserts = treatment.get("insert_strategy", ["environment"])
    section_index = SECTION_ORDER.index(shot.section) if shot.section in SECTION_ORDER else 2
    insert = inserts[section_index % len(inserts)]
    return (
        f"Create one empty photorealistic cinematic starting frame of {location} under {lighting}, emphasizing {insert}.{lyric_beat}{reference_clause} "
        f"Compose natively for {output_format} with realistic depth, coherent light and a level horizon. No people, text, logos, borders or collage layout."
    )


def compile_shot_prompts(plan: EditorialPlan, request: MusicVideoRequest, treatment: dict) -> None:
    performers = _performer_map(request)
    locations = treatment.get("locations", ["cinematic environment"])
    palette = treatment.get("color_palette", "coherent cinematic color")
    reference_style = str(treatment.get("reference_style_prompt", "")).strip()
    reference_clause = (
        f"Transfer the reference video's general filmmaking grammar only: {reference_style}. "
        "Keep this scene original; do not reproduce its people, logos, dialogue, lyrics, locations or exact shot sequence. "
        if reference_style
        else ""
    )
    for shot in plan.shots:
        location = _location_for_section(shot.section, locations)
        lighting = LIGHTING.get(shot.section, "coherent cinematic light")
        group_scene = len(shot.performer_ids) > 1
        action = GROUP_ACTIONS.get(shot.family, ACTIONS[shot.family]) if group_scene else ACTIONS[shot.family]
        if len(shot.performer_ids) == 1:
            action = _solo_role_action(performers.get(shot.performer_ids[0]), action)
        if group_scene and not shot.vocals_present:
            action = (
                "Instrumental members perform their stated roles with believable hand-to-instrument contact and coordinated timing. "
                "Members assigned vocal roles follow the rhythm with relaxed mouths, while everyone maintains distinct natural movement and clear facial separation."
            )
        elif not shot.vocals_present and "vocal phrase" in action:
            action = "The performer breathes naturally and follows the rhythm with subtle eye and shoulder movement while the mouth remains naturally at rest."
        if not shot.vocals_present and "sings toward" in action:
            action = "The performer faces the lens with relaxed shoulders and one restrained open-hand gesture while the mouth remains naturally at rest."
        if len(shot.performer_ids) == 1:
            performer_id = shot.performer_ids[0]
            performer = performers.get(performer_id)
            role = f" Role: {performer.role.strip()}." if performer and performer.role.strip() else ""
            optional_description = f" {performer.description.strip()}" if performer and performer.description.strip() else ""
            identity = (
                f"The performer established by the starting image is {performer_id}.{role}{optional_description} "
                "Their facial identity, natural asymmetry, hairstyle, body proportions, wardrobe, tattoos, jewelry and accessories remain unchanged."
            )
        elif shot.performer_ids:
            descriptions = []
            for performer_id in shot.performer_ids:
                performer = performers.get(performer_id)
                description = performer.description.strip() if performer else ""
                role = performer.role.strip() if performer else "performer"
                details = f"role: {role}"
                if description:
                    details += f"; {description}"
                descriptions.append(f"{performer_id} ({details})")
            identity = (
                f"The distinct assigned performers established by the starting image are {'; '.join(descriptions)}. "
                "Preserve each person's separate facial identity, natural asymmetry, hairstyle, body proportions, wardrobe, tattoos, jewelry and accessories. "
                "Never blend, duplicate or swap their faces, bodies, clothing or accessories, and do not add unassigned people."
            )
        else:
            identity = "The environment remains empty with no people entering the frame."
        camera = GROUP_CAMERAS.get(shot.family, CAMERAS[shot.family]) if group_scene else CAMERAS[shot.family]
        lyric_direction = ""
        if shot.lyric_visual_direction:
            if shot.lyric_sensitive:
                lyric_direction = (
                    f"The imagery translates the current lyric's theme of {shot.lyric_theme} symbolically: "
                    f"{shot.lyric_visual_direction}. "
                )
            else:
                excerpt = shot.lyric_excerpt.replace('"', "'")
                lyric_direction = (
                    f"Interpret the current lyric idea, \"{excerpt}\", through this visible story beat: "
                    f"{shot.lyric_visual_direction}. Do not display the lyric as text. "
                )
        shot.prompt = (
            f"Photorealistic cinematic {shot.family.replace('_', ' ')} in {location} under {lighting}. "
            f"{identity} {reference_clause}{lyric_direction}{action} The camera {camera}. Use {palette}, natural motion, realistic materials, stable exposure and consistent lighting throughout the shot. "
            "One continuous take with no internal cuts, duplicate subjects, malformed anatomy, captions, overlays, logos or watermarks. The supplied song continues without interruption."
        )
