"""Video to Video, Music Video and Music are untouched — pinned by hash.

Client rule for the final milestone (5 Sep 2026): DO NOT TOUCH the existing
Video to Video (no workflow, model, routing, UI or parameter change), keep
the existing music pipeline. The three definitions are pinned by sha256 to
the bytes committed before the milestone began (926d2e3), so any edit to
them — deliberate or a stash-pop accident — fails here before it ships.
Updating a hash is a decision, not a fix: it needs the client's word.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.core.config import settings as app_settings
from app.services.workflow_registry import load_registry

DEFINITIONS = Path(app_settings.workflow_definitions_dir)

PINNED_SHA256 = {
    "video-to-video.yaml": "9782ffbe4e356e0f6998ba6b59ea843a8cf6b2272f875da456433368196a6b4b",
    "music-video.yaml": "8f6bee1231984f738f7df7dbdc6bdf4e42e8cd9906e9c2b9431aa1650c05d136",
    "music.yaml": "7e1be469c5df332086e88c9088c454497fd83fdfb199f2a6e01b4773fa322f11",
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


def test_video_to_video_contract_unchanged() -> None:
    registry = load_registry(DEFINITIONS)
    public = registry.get_public("video-to-video")
    assert public.name == "Video to Video"
    assert public.duration_mode == "source"
    assert public.supported_aspect_ratios == ["16:9", "9:16"]
    assert public.supported_quality_levels == ["fast", "best"]
    assert [spec.role for spec in public.inputs] == ["source_video", "reference_image"]
    assert public.settings.quality is True and public.settings.seed is False
    definition = registry.get("video-to-video")
    extra = definition.execution.model_extra or {}
    assert definition.execution.runtime == "mock"  # the overlay writes `ltx`, never anything else
    assert extra.get("v2v_engine") == "transform"
    assert extra.get("v2v_reference_identity") is False
    assert extra.get("execution_by_quality") == {"best": {"v2v_reference_identity": True}}
    assert definition.execution.timeout_seconds == 5400


def test_music_video_and_music_contracts_unchanged() -> None:
    registry = load_registry(DEFINITIONS)
    music_video = registry.get_public("music-video")
    assert music_video.duration_mode == "source"
    assert [spec.role for spec in music_video.inputs] == ["source_audio"]
    assert music_video.supported_aspect_ratios == ["16:9", "9:16", "1:1"]
    assert music_video.capabilities.extend is False
    music = registry.get_public("music")
    assert music.duration_mode == "minutes"
    assert music.supported_durations == ["1m", "2m", "3m", "4m", "5m"]
    assert music.output_type == "audio"
