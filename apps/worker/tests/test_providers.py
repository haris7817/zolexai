"""The provider layer: routing, refusals, and two plans for one job.

What this suite is defending is mostly a negative. A second engine must not
be able to reach a customer by accident, must not be able to answer a request
it structurally cannot serve, and must not quietly stand in for the first one
when someone asks for it by name. The positive half is small by comparison:
both engines can describe what they would do, from a laptop, for the same job.

No model runs here, and none can — H3 has no weights on any node and says so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_job, staged_input
from worker.adapters.base import AdapterError
from worker.providers import (
    AUTO,
    MATRIX,
    H3Provider,
    LtxProvider,
    ProviderRefusal,
    ProviderUnavailable,
    Support,
    auto_routes,
    compare,
    get_provider,
    requested_provider,
    resolve,
    structural_winners,
)


def t2v_job(workspace: Path, **overrides):
    defaults = dict(
        parameters={"duration": "60s", "aspect_ratio": "16:9"},
        execution={"runtime": "ltx", "prompt_structuring": True, "max_segment_seconds": 30},
    )
    return make_job(workspace, **{**defaults, **overrides})


# ── Routing ──────────────────────────────────────────────────────────────


def test_auto_routes_every_workflow_to_ltx(workspace: Path) -> None:
    """The audit's finding, encoded: no comparison has been run, so there is
    no evidence to route on. A router that guessed would become the decision
    nobody made."""
    assert set(auto_routes().values()) == {"ltx"}
    for workflow in auto_routes():
        job = make_job(
            workspace, workflow_id=workflow,
            parameters={"duration": "5s", "aspect_ratio": "16:9"},
        )
        assert resolve(job)[0] == "ltx"


def test_the_override_selects_an_engine_by_name(workspace: Path) -> None:
    job = t2v_job(workspace, parameters={
        "duration": "60s", "aspect_ratio": "16:9", "provider": "h3"
    })
    assert requested_provider(job) == "h3"
    assert resolve(job)[0] == "h3"


def test_a_request_override_beats_the_workflow_one(workspace: Path) -> None:
    """QA has to be able to force an engine without editing a node's YAML."""
    job = t2v_job(
        workspace,
        parameters={"duration": "60s", "aspect_ratio": "16:9", "provider": "h3"},
        execution={"runtime": "ltx", "provider": "ltx"},
    )
    assert resolve(job)[0] == "h3"


def test_an_absent_override_is_auto_not_a_guess(workspace: Path) -> None:
    assert requested_provider(t2v_job(workspace)) == AUTO


def test_an_unknown_provider_is_refused_rather_than_substituted() -> None:
    """A silent fallback would make an A/B compare LTX against LTX."""
    with pytest.raises(AdapterError):
        get_provider("sora")


# ── H3 availability ──────────────────────────────────────────────────────


async def test_h3_refuses_to_generate_and_says_why(workspace: Path) -> None:
    with pytest.raises(ProviderUnavailable, match="[Ll]icence|not installed"):
        await H3Provider().generate(t2v_job(workspace), lambda *a, **k: None)


def test_h3_reports_itself_unusable_including_the_licence_gate() -> None:
    usable, reason = H3Provider().health()
    assert usable is False
    assert "Licence" in reason or "licence" in reason


# ── Compilation: the same job, two plans ─────────────────────────────────


def test_both_engines_plan_the_same_sixty_seconds_differently(workspace: Path) -> None:
    """The headline structural difference, in numbers rather than adjectives:
    LTX's ceiling is a measured 30s of story, H3's is a documented 15s of
    output, so the same minute costs one seam against three."""
    plans = compare(t2v_job(workspace))
    ltx, h3 = plans["ltx"], plans["h3"]

    assert ltx["pipeline"] == "ltx_pipelines.distilled"
    assert len(ltx["sections"]) == 2
    assert h3["pipeline"].endswith("FL2VA")
    assert len(h3["sections"]) == 4
    assert all(s["duration_seconds"] <= 15.0 for s in h3["sections"])
    # Same total, both engines. A comparison of different lengths is not one.
    assert ltx["total_seconds"] == h3["total_seconds"] == 60.0


def test_the_ltx_manifest_matches_the_audited_sixty_second_plan(
    workspace: Path,
) -> None:
    """The manifest is only useful if it is the adapter's own arithmetic.
    These are the numbers the LTX-2.5 audit recorded for a 60s T2V."""
    manifest = LtxProvider().compile(t2v_job(workspace))
    first, second = manifest.sections

    assert (manifest.width, manifest.height) == (1024, 576)
    assert manifest.fps == 24.0
    assert manifest.seam_count == 1
    assert (first.frames_requested, first.frames_rendered) == (720, 736)
    assert (second.frames_requested, second.frames_rendered) == (720, 720)
    assert first.references == []
    assert second.references[0].role == "seam"
    assert second.references[0].strength == 1.0
    assert first.seed != second.seed


def test_h3_refuses_what_it_does_not_document(workspace: Path) -> None:
    """Two structural refusals, both from the official model card: a 4:5
    frame it has no ratio for, and a clip under its four-second floor."""
    aspect = make_job(workspace, parameters={"duration": "10s", "aspect_ratio": "4:5"})
    with pytest.raises(ProviderRefusal, match="4:5"):
        H3Provider().compile(aspect)

    short = make_job(workspace, parameters={"duration": "3s", "aspect_ratio": "16:9"})
    with pytest.raises(ProviderRefusal, match="below its floor"):
        H3Provider().compile(short)

    # LTX serves both, which is the point of recording them.
    assert LtxProvider().compile(aspect).section_count == 1
    assert LtxProvider().compile(short).section_count == 1


def test_a_source_length_workflow_will_not_invent_a_duration(workspace: Path) -> None:
    """Video-to-video and music video take their length from the upload. A dry
    run has not probed one, and guessing would make every downstream number
    fiction."""
    job = make_job(workspace, workflow_id="video-to-video", parameters={"aspect_ratio": "16:9"})
    for provider in (LtxProvider(), H3Provider()):
        with pytest.raises(ProviderRefusal, match="dry_run_source_seconds"):
            provider.compile(job)


def test_a_declared_source_length_compiles_on_both(workspace: Path) -> None:
    job = make_job(
        workspace,
        workflow_id="video-to-video",
        parameters={"aspect_ratio": "16:9"},
        execution={
            "runtime": "ltx",
            "v2v_engine": "transform",
            "dry_run_source_seconds": 30.0,
        },
        inputs=[staged_input("source_video", "video", "video/mp4", None)],
    )
    ltx = LtxProvider().compile(job)
    h3 = H3Provider().compile(job)

    # The transform engine's 8s ceiling against H3's 15s: four sections to two.
    assert ltx.pipeline == "ltx_pipelines.ic_lora"
    assert ltx.section_count == 4
    assert h3.pipeline.endswith("Ref2VA")
    assert h3.section_count == 2
    assert any(r.kind == "video" for r in h3.sections[0].references)


def test_the_person_reference_rides_every_h3_section(workspace: Path) -> None:
    """The structural difference behind the reference-V2V comparison: LTX has
    no identity input at all and carries the person as pixels in a frame,
    while H3 takes them as a subject reference in every generation."""
    job = make_job(
        workspace,
        workflow_id="video-to-video",
        parameters={"aspect_ratio": "16:9"},
        execution={
            "runtime": "ltx",
            "v2v_engine": "transform",
            "v2v_reference_identity": True,
            "dry_run_source_seconds": 30.0,
        },
        inputs=[
            staged_input("source_video", "video", "video/mp4", None),
            staged_input("reference_image", "image", "image/png", None),
        ],
    )
    h3 = H3Provider().compile(job)
    for section in h3.sections:
        assert any(r.role == "identity" for r in section.references)


def test_a_music_video_dry_run_carries_an_audio_window_on_both(
    workspace: Path,
) -> None:
    """Both engines can be handed the customer's own track; they carry it by
    different mechanisms, and the manifest names which."""
    job = make_job(
        workspace,
        workflow_id="music-video",
        parameters={"aspect_ratio": "16:9"},
        execution={
            "runtime": "ltx",
            "prompt_structuring": True,
            "audio_conditioning": True,
            "dry_run_source_seconds": 60.0,
        },
        inputs=[staged_input("source_audio", "audio", "audio/mpeg", None)],
    )
    ltx = LtxProvider().compile(job)
    h3 = H3Provider().compile(job)

    assert ltx.pipeline == "ltx_pipelines.a2vid_two_stage"
    assert ltx.sections[0].audio.mode == "frozen_latent"
    assert ltx.sections[0].audio.returns_input_waveform is True
    # Only H3's fully_copy mode syncs to the supplied signal; the timbre mode
    # would generate new speech and discard the customer's song.
    assert h3.sections[0].audio.mode == "fully_copy"
    assert h3.sections[0].audio.returns_input_waveform is True
    # A minute of song: fifteen-second generations against twenty-second ones.
    assert ltx.section_count == 3
    assert h3.section_count == 4


def test_an_h3_manifest_never_pretends_to_know_its_settings(
    workspace: Path,
) -> None:
    """Steps, guidance and quantization are measurements this project has not
    made. The manifest says so rather than carrying a plausible number."""
    settings = H3Provider().compile(t2v_job(workspace)).settings
    for key in ("steps", "guidance", "quantization"):
        assert "UNKNOWN" in str(settings[key])


# ── The capability matrix ────────────────────────────────────────────────


def test_a_structural_winner_is_only_declared_where_one_engine_cannot(
    workspace: Path,
) -> None:
    winners = structural_winners()
    assert winners["single_pass_over_15s"] == "ltx"
    assert winners["multimodal_references"] == "h3"
    assert winners["prompt_timestamps"] == "h3"
    # Everything comparable is left to the benchmark, not decided here.
    for name in ("t2v", "i2v", "music_video", "long_form_60s", "lip_sync_to_supplied_audio"):
        assert name not in winners
        assert MATRIX[name].gpu_test is True


def test_an_unmeasured_row_can_never_produce_a_winner() -> None:
    assert MATRIX["camera_adherence"].ltx is Support.UNKNOWN
    assert MATRIX["camera_adherence"].structural_winner is None


def test_emulated_never_loses_to_native_by_default() -> None:
    """Our long-form chain reaches 60s and so would H3's. Which looks better
    is a measurement; this property must not pre-empt it."""
    assert MATRIX["long_form_60s"].structural_winner is None
    assert MATRIX["extend_continuation"].structural_winner is None
