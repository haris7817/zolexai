"""Video to Video, Music Video and Music are untouched — pinned by hash.

Client rule for the final milestone (5 Sep 2026): DO NOT TOUCH the existing
Video to Video (no workflow, model, routing, UI or parameter change), keep
the existing music pipeline. The three definitions are pinned by sha256 to
the bytes committed before the milestone began (926d2e3), so any edit to
them — deliberate or a stash-pop accident — fails here before it ships.
Updating a hash is a decision, not a fix: it needs the client's word.

Music Video's pin moved once, on the client's word: on 8 Sep 2026 they
delivered their own music-video worker (v1.8.0, "bands/groups of up to
five people", ZIP sha256 5c512aec…) to be integrated as this tool. The
definition gained five optional performer pictures, the band and lyrics
controls, and nothing else; the CLI runtime still serves it unchanged
wherever a deployment routes it there.

**Video to Video's pin moved on 10 Sep 2026**, also on the client's word,
and it is worth recording that they moved it twice in two days. Their
9 Sep package (`zolexai-music-v2v-backend-v2-fixed.zip`) turned the tool
into a one-to-four person cast replacement and made the reference image
mandatory. Their 10 Sep one (`zolexai-v2v-deploy-code.zip`, built 14:56,
seventeen hours later) withdrew all of that — confirmed deliberate — and
asked instead for a delivery ladder: the reference optional again, and
1080p/4K/8K replacing Fast/Best. The cast work is archived on the
`v2v-cast-replacement-archived` branch rather than deleted, because this
guard exists precisely because this tool keeps changing shape.

It moved TWICE on 11 Sep 2026. First on their verdict against the first
real result — 480-class did not hold "fast hands, fingers, clothing edges
and facial features" — which took generation to 704. Then their
`...-540-multiref-4voice` package took it back to 540-class, restored the
one-to-four person cast they had withdrawn the day before, and added four
AI voice slots. Both moves are theirs and the second is the later word.

What the current pin covers: up to four optional person references and an
optional background, four optional AI voice slots where an empty slot keeps
that person's own voice, a quality control that selects the delivered SIZE
only, and a 540-class generation grid (896x512 at 16:9) under all three
levels. Music keeps its original pin.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.core.config import settings as app_settings
from app.services.workflow_registry import load_registry

DEFINITIONS = Path(app_settings.workflow_definitions_dir)

PINNED_SHA256 = {
    "video-to-video.yaml": "4c836fc23b2baf2a96b841fe9fe5d2e8e3d6f34ff79295517947cff137fa57aa",
    # 8 Sep 2026: the client's music-video worker (see the module note).
    # music-video.yaml re-pinned 9 Sep 2026: the client asked for a
    # reference-video link box (`settings.reference_video: true`). Only
    # that line changed; the inputs, ladder and execution block did not.
    "music-video.yaml": "f8605c0f74ee6e76806535f52b54f5ef923fe75370cfd70220caad3af43bfd42",
    # music.yaml re-pinned 9 Sep 2026: the client specified the Music Lyrics
    # Workflow v2.0 for this tool — `settings.lyrics_workflow: true`, an
    # optional `reference_audio` input and execution-comment docs. The
    # ladder, the prompt block and the runtime line did not change.
    "music.yaml": "6857975216d54a6313313441b10f6ae65cffc8608bee30864d350938f4a98889",
}


def _sha256_lf(path: Path) -> str:
    # Normalised to LF so a Windows checkout with autocrlf hashes like Linux.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_the_untouched_definitions_are_byte_identical_to_their_pins() -> None:
    for name, expected in PINNED_SHA256.items():
        assert _sha256_lf(DEFINITIONS / name) == expected, (
            f"{name} changed. The client asked for no change to this workflow; "
            "if the change is deliberate and approved, update the pin."
        )


def test_video_to_video_contract() -> None:
    """The shape the pin above is guarding, stated in full.

    Its purpose has not changed: this workflow must not drift by accident.
    What it asserts moved on 10 and 11 Sep 2026 because the client changed the
    tool deliberately — see the module note. The parts that did NOT move are
    the ones worth reading here: the duration still comes from the source, the
    source video is still the only REQUIRED input, and the committed runtime is
    still the mock the deploy overlay rewrites.
    """
    registry = load_registry(DEFINITIONS)
    public = registry.get_public("video-to-video")
    assert public.name == "Video to Video"
    assert public.duration_mode == "source"
    assert public.supported_aspect_ratios == ["16:9", "9:16"]
    # Delivery sizes since 10 Sep 2026, replacing Fast/Best.
    assert public.supported_quality_levels == ["1080p", "4k", "8k"]
    assert [spec.role for spec in public.inputs] == [
        "source_video",
        "reference_image",
        "reference_image_2",
        "reference_image_3",
        "reference_image_4",
        "background_image",
        "voice_reference",
        "voice_reference_2",
        "voice_reference_3",
        "voice_reference_4",
    ]
    # Everything except the video is optional: an empty form is a prompt-only
    # restyle, which is the behaviour that survived all four reworks.
    assert [spec.role for spec in public.inputs if spec.required] == ["source_video"]
    assert public.settings.quality is True and public.settings.seed is False
    assert public.settings.sound is True
    definition = registry.get("video-to-video")
    extra = definition.execution.model_extra or {}
    assert definition.execution.runtime == "mock"  # the overlay writes `ltx`, never anything else
    assert extra.get("v2v_engine") == "transform"
    # On for every job and inert without a photo, so the quality overlay
    # carries the delivered size and nothing else.
    assert extra.get("v2v_reference_identity") is True
    assert extra.get("execution_by_quality") == {
        "1080p": {"delivery": "1080p"},
        "4k": {"delivery": "4k"},
        "8k": {"delivery": "8k"},
    }
    # 540-class since their 11 Sep package. It resolves to 896x512 at 16:9 —
    # this runtime needs both sides divisible by 64, so a literal 960x540 is
    # not a shape the model accepts.
    assert extra.get("render_proxy") == "540p"
    # Four people and four voices, one voice per person.
    assert extra.get("v2v_max_people") == 4
    assert extra.get("v2v_max_voices") == 4
    assert extra.get("v2v_voice_clone") is True
    assert extra.get("v2v_voice_mapping") == "visual_slot_order"
    assert definition.execution.timeout_seconds == 5400


def test_music_video_and_music_contracts_unchanged() -> None:
    registry = load_registry(DEFINITIONS)
    music_video = registry.get_public("music-video")
    assert music_video.duration_mode == "source"
    # The song is still the one required input; the five performer pictures
    # (client's music-video worker, 8 Sep 2026) are optional additions.
    assert [spec.role for spec in music_video.inputs if spec.required] == ["source_audio"]
    assert [spec.role for spec in music_video.inputs] == [
        "source_audio", "performer_1", "performer_2", "performer_3", "performer_4", "performer_5",
    ]
    assert music_video.supported_aspect_ratios == ["16:9", "9:16", "1:1"]
    assert music_video.capabilities.extend is False
    music = registry.get_public("music")
    assert music.duration_mode == "minutes"
    assert music.supported_durations == ["1m", "2m", "3m", "4m", "5m"]
    assert music.output_type == "audio"
