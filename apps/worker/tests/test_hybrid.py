"""The hybrid strategy: LTX draft, decoded, handed to H3 as a reference.

What this suite defends is the difference between a hybrid and a laundering.
Handing H3 an LTX video and calling the result "identity retention" measures
how faithfully H3 reproduced our own invention. So the customer's assets must
survive into the final pass, the draft must be distinguishable from them at a
glance, and a combination that would drop an original is refused rather than
compiled.

It also defends the boundary itself. The handoff is decoded RGB because the
two models share no latent space — nothing here may quietly grow a tensor
path between them.

No GPU, no model, no inference: every assertion is about a plan.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_job, staged_input
from worker.providers.base import ProviderRefusal
from worker.providers.hybrid import GENERATED, USER, compile_hybrid
from worker.providers.router import auto_routes
from worker.providers.strategy import (
    HYBRID_EXCLUDED,
    HYBRID_RATIONALE,
    GenerationStrategy,
    HandoffForm,
    default_handoff_form,
    hybrid_allowed,
    parse,
)

#: H3 is hidden by default (client decision, 5 Sep 2026); this module proves the
#: kept code still works when it is switched back on.
pytestmark = pytest.mark.usefixtures("h3_enabled")


def i2v_job(workspace: Path, duration: str = "30s", **overrides):
    defaults = dict(
        workflow_id="image-to-video",
        parameters={"duration": duration, "aspect_ratio": "16:9"},
        execution={
            "runtime": "ltx",
            "prompt_structuring": True,
            "max_segment_seconds": 30,
        },
        inputs=[staged_input("source_image", "image", "image/png", None)],
    )
    return make_job(workspace, **{**defaults, **overrides})


def reference_v2v_job(workspace: Path, seconds: float = 10.0):
    return make_job(
        workspace,
        workflow_id="video-to-video",
        parameters={"aspect_ratio": "16:9"},
        execution={
            "runtime": "ltx",
            "v2v_engine": "transform",
            "v2v_reference_identity": True,
            "dry_run_source_seconds": seconds,
        },
        inputs=[
            staged_input("source_video", "video", "video/mp4", None),
            staged_input("reference_image", "image", "image/png", None),
        ],
    )


def music_video_job(workspace: Path, seconds: float = 60.0):
    return make_job(
        workspace,
        workflow_id="music-video",
        parameters={"aspect_ratio": "16:9"},
        execution={
            "runtime": "ltx",
            "prompt_structuring": True,
            "audio_conditioning": True,
            "dry_run_source_seconds": seconds,
        },
        inputs=[staged_input("source_audio", "audio", "audio/mpeg", None)],
    )


# ── Strategy parsing ─────────────────────────────────────────────────────


def test_the_three_strategies_parse_and_name_their_engines() -> None:
    assert parse("ltx_only") is GenerationStrategy.LTX_ONLY
    assert parse("h3_only") is GenerationStrategy.H3_ONLY
    assert parse("ltx_to_h3_reference") is GenerationStrategy.LTX_TO_H3_REFERENCE

    assert GenerationStrategy.LTX_ONLY.providers == ("ltx",)
    assert GenerationStrategy.H3_ONLY.providers == ("h3",)
    # Both engines, in order — the field a cost table has to read so it never
    # compares half a hybrid against a whole single-engine run.
    assert GenerationStrategy.LTX_TO_H3_REFERENCE.providers == ("ltx", "h3")
    assert GenerationStrategy.LTX_TO_H3_REFERENCE.final_provider == "h3"
    assert GenerationStrategy.LTX_TO_H3_REFERENCE.is_hybrid is True
    assert GenerationStrategy.H3_ONLY.is_hybrid is False


def test_an_unknown_strategy_raises_rather_than_falling_back() -> None:
    """A typo that silently ran ltx_only while the table said hybrid would
    poison every conclusion drawn from that row."""
    with pytest.raises(ValueError, match="unknown generation strategy"):
        parse("ltx_then_h3")


def test_hybrid_is_scoped_to_workflows_with_a_stated_rationale() -> None:
    for workflow in ("image-to-video", "video-to-video", "music-video", "text-to-video"):
        assert hybrid_allowed(workflow)
        assert HYBRID_RATIONALE[workflow]
    # Extension is excluded on purpose, with the reason recorded.
    assert not hybrid_allowed("extend-video")
    assert "continuation" in HYBRID_EXCLUDED["extend-video"]


def test_hybrid_never_reaches_customer_routing() -> None:
    """The strategy is a benchmark cell. Production routing must not know it
    exists."""
    assert set(auto_routes().values()) == {"ltx"}
    assert GenerationStrategy.LTX_TO_H3_REFERENCE.value not in auto_routes().values()


# ── The handoff manifest ─────────────────────────────────────────────────


def test_the_handoff_names_both_passes_and_the_form(workspace: Path) -> None:
    plan = compile_hybrid(i2v_job(workspace))

    assert plan.strategy == "ltx_to_h3_reference"
    assert plan.providers == ("ltx", "h3")
    assert plan.draft.provider == "ltx"
    assert plan.final.provider == "h3"
    assert plan.rationale
    assert any("DECODED RGB" in note for note in plan.notes)

    handoff = plan.handoffs[0]
    assert handoff.source_job_id == "00000000-0000-0000-0000-0000000000ff"
    assert handoff.draft_provider == "ltx"
    assert handoff.final_provider == "h3"
    assert handoff.handoff_form == "video_plus_original_image"


def test_the_original_image_survives_into_every_final_section(
    workspace: Path,
) -> None:
    """The whole point of the I2V hypothesis: the photograph owns WHO, the
    draft only owns approximately how it moves."""
    plan = compile_hybrid(i2v_job(workspace, "60s"))
    assert plan.final.section_count == 4  # H3's 15s ceiling

    for section in plan.final.sections:
        origins = {r.origin for r in section.references}
        assert USER in origins, "the customer's own asset must reach every section"
        assert GENERATED in origins, "and so must the draft"

    for handoff in plan.handoffs:
        assert [r.role for r in handoff.original_references] == ["first_frame"]
        assert all(r.origin == USER for r in handoff.original_references)
        assert all(r.origin == GENERATED for r in handoff.generated_references)


def test_a_generated_reference_is_never_mistakable_for_a_user_asset(
    workspace: Path,
) -> None:
    plan = compile_hybrid(i2v_job(workspace))
    generated = [
        r for s in plan.final.sections for r in s.references if r.is_generated
    ]
    assert generated
    for reference in generated:
        assert reference.origin == "generated_intermediate"
        assert "ltx draft" in reference.source


def test_the_reference_person_survives_a_hybrid_restyle(workspace: Path) -> None:
    """The high-priority comparison. LTX has no identity input at all, so if
    the hybrid dropped the reference photo the cell would measure nothing."""
    plan = compile_hybrid(reference_v2v_job(workspace))
    roles = {r.role for r in plan.handoffs[0].original_references}
    assert "identity" in roles
    assert "structure" in roles
    assert plan.handoffs[0].handoff_form == "full_video"


def test_the_customers_own_track_is_the_audio_on_the_final_pass(
    workspace: Path,
) -> None:
    """A music video measured against the draft's invented audio would be
    measuring nothing at all."""
    plan = compile_hybrid(music_video_job(workspace))

    for handoff in plan.handoffs:
        assert handoff.audio_reference == "source_audio"
    assert any("draft's generated audio is discarded" in n for n in plan.notes)
    for section in plan.final.sections:
        assert section.audio is not None
        assert section.audio.mode == "fully_copy"
        assert section.audio.returns_input_waveform is True


# ── Section mapping ──────────────────────────────────────────────────────


def test_long_form_maps_draft_windows_onto_the_finals_own_sections(
    workspace: Path,
) -> None:
    """The two engines section differently — LTX at 30s, H3 at 15s — so each
    final section takes its own window of the decoded draft rather than
    assuming the two plans line up."""
    plan = compile_hybrid(i2v_job(workspace, "60s"))
    assert plan.draft.section_count == 2
    assert plan.final.section_count == 4

    windows = [h.draft_window for h in plan.handoffs]
    assert windows == [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 60.0)]
    for handoff, section in zip(plan.handoffs, plan.final.sections, strict=True):
        assert handoff.section_index == section.index
        assert handoff.duration_seconds == section.duration_seconds


def test_every_handoff_form_produces_a_coherent_plan(workspace: Path) -> None:
    job = i2v_job(workspace, "10s")
    for form in HandoffForm:
        plan = compile_hybrid(job, form=form)
        handoff = plan.handoffs[0]
        assert handoff.handoff_form == form.value
        assert handoff.generated_references, f"{form} handed over nothing"
        if form in (HandoffForm.FULL_VIDEO, HandoffForm.VIDEO_PLUS_ORIGINAL_IMAGE):
            assert handoff.draft_window is not None
            assert not handoff.selected_frames
        else:
            assert handoff.draft_window is None
            assert handoff.selected_frames


def test_the_default_form_follows_the_workflows_own_hypothesis() -> None:
    assert default_handoff_form("image-to-video") is HandoffForm.VIDEO_PLUS_ORIGINAL_IMAGE
    assert default_handoff_form("video-to-video") is HandoffForm.FULL_VIDEO
    assert default_handoff_form("music-video") is HandoffForm.FULL_VIDEO


# ── Refusals ─────────────────────────────────────────────────────────────


def test_an_out_of_scope_workflow_is_refused_with_its_reason(
    workspace: Path,
) -> None:
    job = make_job(
        workspace,
        workflow_id="extend-video",
        parameters={"duration": "10s", "aspect_ratio": "16:9"},
    )
    with pytest.raises(ProviderRefusal, match="continuation"):
        compile_hybrid(job)


def test_a_plain_restyle_hybrid_is_refused(workspace: Path) -> None:
    """Without a reference person, a v2v hybrid would be a restyle of a
    restyle and neither engine's contribution would be attributable."""
    job = make_job(
        workspace,
        workflow_id="video-to-video",
        parameters={"aspect_ratio": "16:9"},
        execution={
            "runtime": "ltx",
            "v2v_engine": "transform",
            "dry_run_source_seconds": 10.0,
        },
        inputs=[staged_input("source_video", "video", "video/mp4", None)],
    )
    with pytest.raises(ProviderRefusal, match="restyle of a restyle"):
        compile_hybrid(job)


def test_a_form_needing_an_image_is_refused_when_there_is_none(
    workspace: Path,
) -> None:
    job = make_job(
        workspace,
        parameters={"duration": "10s", "aspect_ratio": "16:9"},
        execution={"runtime": "ltx", "prompt_structuring": True},
    )
    with pytest.raises(ProviderRefusal, match="needs the customer's own image"):
        compile_hybrid(job, form=HandoffForm.VIDEO_PLUS_ORIGINAL_IMAGE)


def test_a_hybrid_inherits_h3s_own_refusals(workspace: Path) -> None:
    """H3 cannot render 4:5. A hybrid ending in H3 cannot either, and must
    not paper over it."""
    with pytest.raises(ProviderRefusal, match="4:5"):
        compile_hybrid(i2v_job(workspace, parameters={
            "duration": "10s", "aspect_ratio": "4:5",
        }))


def test_a_draft_clip_may_not_exceed_h3s_reference_limit(workspace: Path) -> None:
    """H3 caps a reference clip at 15s. A section longer than that cannot hand
    over its window, and the plan says so instead of truncating silently."""
    plan = compile_hybrid(i2v_job(workspace, "60s"))
    for handoff in plan.handoffs:
        assert handoff.draft_window is not None
        start, end = handoff.draft_window
        assert end - start <= 15.0
