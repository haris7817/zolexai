from __future__ import annotations

from typing import Any

from .analysis import SongAnalysis
from .lyrics import LyricTranscript
from .models import MusicVideoRequest


GENRE_PROFILES: tuple[tuple[tuple[str, ...], dict[str, Any]], ...] = (
    (
        ("love", "romance", "romantic", "amor", "románt", "couple"),
        {
            "genre": "romantic cinematic performance",
            "emotion": "love, longing and hopeful connection",
            "locations": [
                "golden-hour shoreline",
                "quiet palm-lined beach",
                "sunset driftwood area",
                "blue-hour ocean overlook",
                "night bonfire",
                "peaceful dawn shoreline",
            ],
            "palette": "warm skin and amber highlights against teal-blue water and deep blue night",
            "inserts": ["waves", "footprints", "hands", "jewelry", "personal keepsake", "fire"],
        },
    ),
    (
        ("rap", "hip hop", "hip-hop", "trap", "rapper", "rapero"),
        {
            "genre": "cinematic hip-hop performance film",
            "emotion": "confidence, pressure, resilience and earned momentum",
            "locations": [
                "neighborhood street at late afternoon",
                "mural-lined performance alley",
                "minimal warehouse with directional light",
                "rooftop over the city at blue hour",
                "wet neon street at night",
                "quiet dawn overlook",
            ],
            "palette": "natural skin, deep neutral shadows, warm sodium highlights and restrained neon accents",
            "inserts": ["hands", "chain", "sneakers", "street texture", "personal keepsake", "skyline"],
        },
    ),
    (
        ("sad", "heartbreak", "lonely", "triste", "dolor"),
        {
            "genre": "reflective cinematic performance",
            "emotion": "heartbreak, solitude and acceptance",
            "locations": [
                "empty overcast shoreline",
                "rain-streaked intimate interior",
                "quiet blue-hour street",
                "dark ocean overlook",
                "soft dawn horizon",
            ],
            "palette": "cool cyan and slate shadows with restrained warm skin tones",
            "inserts": ["rain on glass", "empty chair", "hands", "footprints", "receding waves"],
        },
    ),
    (
        ("luxury", "rich", "yacht", "premium", "fashion"),
        {
            "genre": "premium luxury performance film",
            "emotion": "confidence, success and effortless control",
            "locations": [
                "sunlit luxury yacht",
                "modern penthouse terrace",
                "marina at golden hour",
                "architectural hotel corridor",
                "night skyline rooftop",
            ],
            "palette": "polished neutrals, ocean blue, warm gold and clean specular highlights",
            "inserts": ["watch", "jewelry", "car detail", "glass reflections", "water wake"],
        },
    ),
    (
        ("street", "urban", "ghetto", "barrio", "city"),
        {
            "genre": "grounded urban performance film",
            "emotion": "resilience, pride and direct confidence",
            "locations": [
                "neighborhood street at golden hour",
                "mural-lined alley",
                "rooftop overlooking the city",
                "corner store exterior",
                "wet neon street at night",
            ],
            "palette": "natural skin, concrete neutrals, warm sodium light and restrained neon color",
            "inserts": ["sneakers", "hands", "chain", "street sign", "pavement reflections"],
        },
    ),
    (
        ("party", "club", "dance", "fiesta"),
        {
            "genre": "energetic nightlife performance film",
            "emotion": "celebration, attraction and release",
            "locations": [
                "sunset beach gathering",
                "open-air dance floor",
                "club corridor with practical lights",
                "night poolside performance",
                "late-night rooftop",
            ],
            "palette": "warm skin, saturated practical lights, cyan shadows and magenta accents",
            "inserts": ["dancing feet", "hands", "glasses", "speakers", "reflections"],
        },
    ),
    (
        ("inspirational", "inspiring", "uplifting", "motivation", "triumph"),
        {
            "genre": "uplifting cinematic performance film",
            "emotion": "perseverance, growing confidence and earned hope",
            "locations": [
                "quiet predawn overlook",
                "sunrise city street",
                "open landscape in clean daylight",
                "high ridge in warm afternoon light",
                "bright horizon at sunset",
            ],
            "palette": "cool predawn tones opening gradually into clean daylight and warm gold",
            "inserts": ["first light", "hands", "forward steps", "wind in fabric", "open horizon"],
        },
    ),
)


DEFAULT_PROFILE = {
    "genre": "cinematic narrative performance film",
    "emotion": "focused, expressive and emotionally grounded",
    "locations": [
        "wide atmospheric landscape",
        "natural-light performance location",
        "intimate reflective setting",
        "blue-hour exterior",
        "night performance location",
        "quiet closing landscape",
    ],
    "palette": "natural skin, controlled contrast, cinematic complementary colors and coherent light",
    "inserts": ["environment", "hands", "wardrobe detail", "personal object", "closing landscape"],
}


def _profile_for(prompt: str) -> dict[str, Any]:
    lowered = prompt.casefold()
    for keywords, profile in GENRE_PROFILES:
        if any(keyword in lowered for keyword in keywords):
            return dict(profile)
    return dict(DEFAULT_PROFILE)


def expand_direction(
    request: MusicVideoRequest,
    analysis: SongAnalysis,
    transcript: LyricTranscript | None = None,
    reference_style: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = _profile_for(request.prompt)
    if not request.allow_background_change:
        profile["locations"] = profile["locations"][:1]
    performer_count = len(request.performers)
    romantic = profile["genre"].startswith("romantic")
    if performer_count == 0:
        performer_strategy = "generate_consistent_adult_couple" if romantic else "generate_consistent_adult_lead"
        performer_ids = ["generated-a", "generated-b"] if romantic else ["generated-a"]
    elif performer_count == 1:
        performer_strategy = "solo"
        performer_ids = [request.performers[0].id]
    elif performer_count == 2 and romantic:
        performer_strategy = "two_romantic_leads"
        performer_ids = [item.id for item in request.performers]
    elif performer_count == 2:
        performer_strategy = "duet"
        performer_ids = [item.id for item in request.performers]
    else:
        performer_strategy = "band_or_group_ensemble"
        performer_ids = [item.id for item in request.performers]

    visual_arc = [
        "atmospheric opening",
        "intimate performer introduction",
        "performance and physical movement",
        "reflective emotional middle",
        "higher-energy visual climax",
        "quiet resolved ending",
    ]
    tempo_note = f"measured tempo near {analysis.tempo_bpm:.2f} BPM" if analysis.tempo_bpm else "tempo not confidently measured"
    lyric_driven = bool(transcript and transcript.requested and transcript.segments)
    narrative_summary = (
        f"A {profile['genre']} follows the timestamped lyrics as a coherent visual story, moving from an atmospheric opening through "
        "the verses and recurring chorus imagery to a stronger climax and resolved ending."
        if lyric_driven
        else (
            f"A {profile['genre']} follows the song from an atmospheric opening through intimate "
            "performance, emotional development, a stronger climax and a peaceful visual resolution."
        )
    )
    treatment = {
        "concept_title": request.prompt.strip().rstrip(".")[:80].title(),
        "original_direction": request.prompt,
        "genre": profile["genre"],
        "core_emotion": profile["emotion"],
        "narrative_summary": narrative_summary,
        "lyric_driven": lyric_driven,
        "lyrics_source": transcript.source if transcript else "not_requested",
        "lyrics_language": transcript.language if transcript else request.lyrics_language,
        "lyric_scene_policy": (
            "interpret each timestamped line with filmable imagery; preserve a repeated motif for detected refrains; use symbolic treatment for sensitive lines"
            if lyric_driven
            else "music structure and user direction only"
        ),
        "performer_strategy": performer_strategy,
        "performer_ids": performer_ids,
        "maximum_people_visible_per_shot": request.max_people_visible_per_shot,
        "group_scene_policy": (
            "rotate individual performance coverage and include selected full-ensemble wides and climax frames"
            if performer_count > 1 and request.max_people_visible_per_shot > 1
            else "individual performer coverage"
        ),
        "locations": profile["locations"],
        "visual_arc": visual_arc,
        "color_palette": profile["palette"],
        "wardrobe_plan": "one locked outfit throughout" if not request.allow_wardrobe_change else "one planned change at a section boundary",
        "performance_style": "natural singing, controlled breathing, subtle gestures and believable eye focus",
        "camera_style": "slow push-ins, stable holds, gentle tracking, restrained arcs and environmental glides",
        "insert_strategy": profile["inserts"],
        "cut_style": "frame-exact beat-aware hard cuts",
        "music_note": tempo_note,
        "unrequested_elements_forbidden": [
            "unassigned people",
            "captions",
            "logos",
            "watermarks",
            "random wardrobe changes",
            "unmotivated location changes",
        ],
    }
    if reference_style:
        reference_prompt = "; ".join(
            str(reference_style.get(field, "")).strip()
            for field in (
                "style_summary",
                "camera_language",
                "shot_scale_and_angles",
                "scene_archetypes",
                "lighting_style",
                "color_palette",
                "editing_rhythm",
                "composition_style",
                "motion_style",
            )
            if str(reference_style.get(field, "")).strip()
        )
        treatment.update(
            {
                "reference_video_used": True,
                "reference_style_profile": reference_style,
                "reference_style_prompt": reference_prompt,
                "reference_originality_policy": reference_style["originality_policy"],
                "color_palette": reference_style.get("color_palette", treatment["color_palette"]),
                "camera_style": reference_style.get("camera_language", treatment["camera_style"]),
                "cut_style": reference_style.get("editing_rhythm", treatment["cut_style"]),
            }
        )
    else:
        treatment.update(
            {
                "reference_video_used": False,
                "reference_style_profile": None,
                "reference_style_prompt": "",
                "reference_originality_policy": "not_applicable",
            }
        )
    return treatment
