"""The platform's planner in front of the client's music-video package.

What is asserted is what this codebase decided: that the customer's words
reach every shot, that a plan missing one of them is refused before the GPU,
that a visible performer under sung lyrics is told to sing, that a single-shot
request becomes continuous windows anchored on each other, that a dry run
stops at the anchor boundary, and that with no context set the package runs
exactly as vendored. The package's own behaviour is not asserted on -- it is
theirs, and `test_music_video_worker.py` covers the seam it presents.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.conftest import needs_ffmpeg
from worker.musicvideo.brief import (
    CreativeBrief,
    Event,
    coverage,
    extract_brief,
    heuristic_brief,
    wants_single_shot,
)
from worker.musicvideo.enforce import (
    DryRunComplete,
    Enforcement,
    activate,
    compose_shot_prompt,
    deactivate,
    install,
    uninstall,
)

CLIENT_PROMPT = (
    "A high-energy, cinematic country-pop music video blending Beverly Hills "
    "glamour, rural Americana, and neon dance-club energy. The same female singer "
    "remains visually consistent throughout, performing and lip-syncing naturally "
    "to the supplied song. The video opens in a glass Beverly Hills penthouse "
    "strangely filled with hay bales, clover, farm fencing, and a luxurious silk "
    "bed; her fashionable agent excitedly shows her a tablet celebrating 50 million "
    "streams while her traditional country mother watches skeptically, and the "
    "singer pours a tiny gin martini into a mason jar. A hard cut transitions to a "
    "dusty country road where she drives a hot-pink Lamborghini towing a massive "
    "farm tractor, loses service, and is stopped by a local sheriff before "
    "launching into energetic synchronized choreography with dancers wearing "
    "camouflage over luxury streetwear. Another hard cut reveals an old Nashville "
    "barn transformed into a neon electronic nightclub with horse stalls, spinning "
    "disco lights, and a huge crystal disco ball. She performs with an acoustic "
    "guitar, switches to an amplified banjo when the bass drops, and dances beside "
    "a fiddle player whose bow triggers colorful laser effects. The climax shows "
    "her confidently balancing on a leather saddle in the center of the dance "
    "floor and pouring a matcha latte from a moonshine jug, fully embodying a "
    "glamorous disco ball on a gravel road. Natural body movement, accurate "
    "lip-sync, beat-matched cuts, dynamic tracking shots, bold pink-and-neon "
    "lighting, stable identity and clothing, realistic instruments, no dialogue, "
    "and no captions or generated text"
)


@pytest.fixture
def seam():
    """The wrappers installed for the test, the package restored after.

    The vendored package needs its `music-video` extra (numpy, Pillow, ...);
    a venv without it skips these rather than erroring, the way the GPU-only
    suites skip without ffmpeg."""
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    install()
    yield
    uninstall()


def _package():
    import zolex_music_worker.worker as pkg

    return pkg


def _request(prompt: str = CLIENT_PROMPT, performers: int = 1):
    from zolex_music_worker.models import MusicVideoRequest

    return MusicVideoRequest.from_dict(
        {
            "job_id": "mv-enforce",
            "audio": "song.wav",
            "prompt": prompt,
            "formats": ["16:9"],
            "performers": [
                {"id": f"performer-{i + 1}", "role": "lead vocalist", "reference_images": []}
                for i in range(performers)
            ],
        }
    )


def _analysis(seconds: float = 40.0, fps: int = 24):
    total = int(seconds * fps)
    half = total // 2
    sections = (
        SimpleNamespace(id="introduction", start_frame=0, end_frame=half, vocals_present=True,
                        energy=0.5),
        SimpleNamespace(id="climax", start_frame=half, end_frame=total, vocals_present=True,
                        energy=0.9),
    )
    return SimpleNamespace(tempo_bpm=120.0, sections=sections, total_frames=total,
                           transients=())


def _plan(shots: int = 5, *, performer: bool = True, fps: int = 24, seconds: int = 8):
    from zolex_music_worker.models import EditorialPlan, Shot

    frames = seconds * fps
    items = [
        Shot(
            id=f"shot-{i + 1:03d}",
            start_frame=i * frames,
            frame_count=frames,
            section="introduction" if i < shots // 2 else "climax",
            family="wide_movement" if i % 2 else "medium_performance",
            energy=0.5,
            vocals_present=False,
            performer_ids=["performer-1"] if performer else [],
            anchor_role="medium_performance",
            lyric_segment_ids=[f"line-{i}"],
        )
        for i in range(shots)
    ]
    return EditorialPlan(schema_version=1, fps=fps, format="16:9", width=1920, height=1080,
                         total_frames=frames * shots, shots=items,
                         render_width=1280, render_height=704)


def _treatment(request, analysis, **overrides):
    """The package's own treatment for this request -- unwrapped."""
    from zolex_music_worker.director import expand_direction

    treatment = expand_direction(request, analysis, None, None)
    treatment.update(overrides)
    return treatment


# ── The brief ───────────────────────────────────────────────────────────────


def test_the_heuristic_reads_the_customers_places_props_and_beats() -> None:
    """Every place and prop the client named, from their own sentences."""
    brief = heuristic_brief(CLIENT_PROMPT)
    lowered = [loc.casefold() for loc in brief.locations]
    assert any("beverly hills penthouse" in loc for loc in lowered)
    assert any("dusty country road" in loc for loc in lowered)
    assert any("nightclub" in loc for loc in lowered)
    props = [p.casefold() for p in brief.props]
    for needed in ("hot-pink lamborghini", "farm tractor", "acoustic guitar", "amplified banjo",
                   "leather saddle", "mason jar"):
        assert any(needed in p for p in props), needed
    # Story beats, in the order they were written; style sentences are locks,
    # not beats.
    assert brief.events
    assert brief.events[0].action.startswith("The video opens in a glass Beverly Hills")
    assert all("music video blending" not in e.action for e in brief.events)
    assert brief.prohibited == ["dialogue", "captions"]
    assert brief.single_shot is False


def test_a_style_sentence_is_a_lock_on_every_shot_not_a_beat() -> None:
    brief = heuristic_brief(CLIENT_PROMPT)
    assert "remains visually consistent" in brief.look
    assert "bold pink-and-neon lighting" in brief.look


@pytest.mark.parametrize(
    "text",
    [
        "One continuous shot of her singing on the pier",
        "static camera on the band, no cuts",
        "keep the same framing for the whole song",
        "a single take, unbroken, the camera never moves",
    ],
)
def test_the_customer_can_ask_for_one_take(text: str) -> None:
    assert wants_single_shot(text)
    assert heuristic_brief(text).single_shot is True


def test_an_ordinary_prompt_is_not_a_single_shot() -> None:
    assert wants_single_shot(CLIENT_PROMPT) is False


def test_coverage_is_the_customers_own_words_or_nothing() -> None:
    """No fuzzy matching: "a barn" must not count for "a Nashville barn
    transformed into a neon nightclub"."""
    brief = CreativeBrief(
        original_prompt="x",
        locations=["an old Nashville barn transformed into a neon nightclub"],
        props=["hot-pink Lamborghini"],
        events=[Event("she drives a hot-pink Lamborghini", 0)],
    )
    full = coverage(brief, ["Shot in an old Nashville barn transformed into a neon nightclub; "
                            "she drives a hot-pink Lamborghini"])
    assert full.complete and full.fraction == 1.0
    partial = coverage(brief, ["a barn; she drives a car"])
    assert not partial.complete
    assert "hot-pink Lamborghini" in partial.missing


async def test_the_writer_is_merged_over_the_heuristic_never_instead_of_it() -> None:
    """A writer that drops a location is put right by the heuristic. The
    writer only adds cleaner beats; the heuristic is the guarantee."""

    class _Writer:
        name = "fake"

        async def write(self, request: Any) -> dict[str, Any]:
            return {
                "theme": "country-pop glamour",
                "locations": ["a glass Beverly Hills penthouse"],  # dropped the road and barn
                "props": ["hot-pink Lamborghini"],
                "events": [{"action": "she pours a gin martini into a mason jar",
                            "location_index": 0}],
                "prohibited": ["captions"],
                "single_shot": False,
            }

    brief = await extract_brief(CLIENT_PROMPT, providers=[_Writer()])
    assert brief.source == "writer"
    lowered = [loc.casefold() for loc in brief.locations]
    assert any("dusty country road" in loc for loc in lowered)
    assert any("nightclub" in loc for loc in lowered)
    assert any("farm tractor" in p.casefold() for p in brief.props)


async def test_no_writer_means_the_heuristic_and_never_an_error() -> None:
    brief = await extract_brief(CLIENT_PROMPT, providers=[])
    assert brief.source == "heuristic"
    assert brief.locations


# ── Composition ────────────────────────────────────────────────────────────


def test_the_story_leads_and_the_packages_craft_follows() -> None:
    brief = heuristic_brief(CLIENT_PROMPT)
    package = ("Photorealistic cinematic medium performance in golden-hour shoreline under "
               "warm light. The performer established by the starting image is performer-1. "
               "The camera slides gently sideways. The supplied song continues without "
               "interruption.")
    prompt = compose_shot_prompt(brief, package, package_location="golden-hour shoreline",
                                 index=0, total=25)
    assert prompt.startswith("Story: ")
    assert "This shot is set in" in prompt
    assert "What happens in this shot: The video opens in a glass Beverly Hills penthouse" in prompt
    # The package's genre location is replaced by the customer's, not stacked.
    assert "golden-hour shoreline" not in prompt
    assert "The performer established by the starting image is performer-1" in prompt
    assert prompt.rstrip().endswith("Not allowed: dialogue, captions.")


def test_every_beat_and_prop_lands_somewhere_across_the_shots() -> None:
    brief = heuristic_brief(CLIENT_PROMPT)
    package = "Photorealistic cinematic shot in golden-hour shoreline under warm light."
    prompts = [
        compose_shot_prompt(brief, package, package_location="golden-hour shoreline",
                            index=i, total=25)
        for i in range(25)
    ]
    assert coverage(brief, prompts).complete
    # Chronological: the penthouse beat comes before the barn beat.
    first_penthouse = next(i for i, p in enumerate(prompts) if "penthouse" in p)
    first_barn = next(i for i, p in enumerate(prompts) if "Nashville barn" in p)
    assert first_penthouse < first_barn


def test_single_shot_text_replaces_the_per_shot_location() -> None:
    brief = heuristic_brief("One continuous shot of her singing in a red-lit garage.")
    prompt = compose_shot_prompt(brief, "Photorealistic shot in a beach under sun.",
                                 package_location="a beach", index=3, total=8)
    assert "one continuous take in red-lit garage" in prompt
    assert "picks up exactly where the previous segment ended" in prompt
    assert "This shot is set in" not in prompt


# ── The seam ────────────────────────────────────────────────────────────────


def test_with_no_context_the_package_runs_exactly_as_vendored(seam) -> None:
    """The kill switch. Every other test in the suite relies on this."""
    pkg = _package()
    request, analysis = _request(), _analysis()
    treatment = pkg.expand_direction(request, analysis, None, None)
    assert "creative_brief" not in treatment
    assert treatment["locations"] == _treatment(request, analysis)["locations"]
    plan = _plan()
    pkg.compile_shot_prompts(plan, request, treatment)
    assert not plan.shots[0].prompt.startswith("Story: ")


def test_the_treatment_takes_the_customers_locations_in_story_order(seam) -> None:
    pkg = _package()
    request, analysis = _request(), _analysis()
    token = activate(Enforcement(brief=heuristic_brief(CLIENT_PROMPT), job_id="j"))
    try:
        treatment = pkg.expand_direction(request, analysis, None, None)
    finally:
        deactivate(token)
    assert treatment["genre_locations_replaced"] is True
    assert "penthouse" in treatment["locations"][0]
    assert treatment["original_direction"] == CLIENT_PROMPT
    assert treatment["shot_mode"] == "multi_shot"
    assert "captions" in treatment["unrequested_elements_forbidden"]


def test_every_shot_prompt_carries_the_story_and_the_trace_proves_it(seam, tmp_path: Path) -> None:
    pkg = _package()
    request, analysis = _request(), _analysis()
    ctx = Enforcement(brief=heuristic_brief(CLIENT_PROMPT), job_id="j",
                      workflow_version="1", trace_dir=tmp_path)
    token = activate(ctx)
    try:
        treatment = pkg.expand_direction(request, analysis, None, None)
        plan = _plan(shots=6)
        pkg.compile_shot_prompts(plan, request, treatment)
    finally:
        deactivate(token)
    assert all(shot.prompt.startswith("Story: ") for shot in plan.shots)
    assert coverage(ctx.brief, [s.prompt for s in plan.shots]).complete
    trace = json.loads((tmp_path / "prompt-trace.json").read_text(encoding="utf-8"))
    assert trace["original_user_prompt"] == CLIENT_PROMPT
    assert trace["status"] == "validated"
    assert trace["coverage"]["fraction"] == 1.0
    assert len(trace["shots"]) == 6
    first = trace["shots"][0]
    for key in ("planned_shot_prompt", "final_renderer_prompt", "prompt_sha256",
                "audio_start_time", "audio_end_time", "renderer_route",
                "performer_identity_ids", "lip_sync_required"):
        assert key in first, key
    assert first["renderer_route"] == "audio_conditioned_a2v"
    assert first["final_renderer_prompt"] == plan.shots[0].prompt


def test_a_visible_performer_under_sung_lyrics_sings(seam) -> None:
    """The client's lip-sync routing rule at the prompt: the package told a
    performer in a `wide_movement` shot to keep the mouth at rest while the
    track sang. Never classify a singer-facing shot as B-roll."""
    pkg = _package()
    request, analysis = _request(), _analysis()
    token = activate(Enforcement(brief=heuristic_brief(CLIENT_PROMPT), job_id="j"))
    try:
        treatment = pkg.expand_direction(request, analysis, None, None)
        plan = _plan(shots=4)
        wide = [s for s in plan.shots if s.family == "wide_movement"]
        assert wide and all(not s.vocals_present for s in wide)
        pkg.compile_shot_prompts(plan, request, treatment)
    finally:
        deactivate(token)
    assert all(s.vocals_present for s in plan.shots)
    assert all("mouth remains naturally at rest" not in s.prompt for s in plan.shots)


def test_a_plan_missing_a_mandatory_element_is_refused_before_any_still(seam) -> None:
    """The GPU boundary. `generate_anchors` is where the package starts
    spending, and the coverage check runs first."""
    from zolex_music_worker.errors import ValidationError

    pkg = _package()
    request, analysis = _request(), _analysis()
    brief = heuristic_brief(CLIENT_PROMPT)
    # A brief with a location no composition can supply: an empty event list
    # and a prop that never appears in any beat is still injected; force the
    # miss by giving the brief a mandatory item and then a plan of zero shots.
    ctx = Enforcement(brief=brief, job_id="j")
    called = []
    token = activate(ctx)
    try:
        treatment = pkg.expand_direction(request, analysis, None, None)
        empty = replace(_plan(shots=1), shots=[])
        with pytest.raises(ValidationError, match="coverage"):
            pkg.generate_anchors(plan=empty, request=request, treatment=treatment,
                                 output_dir=Path("."), adapter=called, timeout=1)
    finally:
        deactivate(token)


def test_a_dry_run_stops_at_the_anchor_boundary_with_the_trace_written(
    seam, tmp_path: Path
) -> None:
    pkg = _package()
    request, analysis = _request(), _analysis()
    ctx = Enforcement(brief=heuristic_brief(CLIENT_PROMPT), job_id="j", trace_dir=tmp_path,
                      dry_run=True)
    token = activate(ctx)
    try:
        treatment = pkg.expand_direction(request, analysis, None, None)
        plan = _plan(shots=5)
        with pytest.raises(DryRunComplete) as caught:
            pkg.generate_anchors(plan=plan, request=request, treatment=treatment,
                                 output_dir=tmp_path, adapter=None, timeout=1)
    finally:
        deactivate(token)
    assert caught.value.summary["shots"] == 5
    trace = json.loads((tmp_path / "prompt-trace.json").read_text(encoding="utf-8"))
    assert trace["status"] == "dry_run_passed"
    # Nothing was anchored: no still exists anywhere.
    assert not list(tmp_path.glob("*.png"))


def test_a_single_shot_request_becomes_continuous_windows(seam) -> None:
    pkg = _package()
    request = _request("One continuous shot of her singing in a red-lit garage, no cuts.")
    analysis = _analysis()
    brief = heuristic_brief(request.prompt)
    assert brief.single_shot
    token = activate(Enforcement(brief=brief, job_id="j"))
    try:
        treatment = pkg.expand_direction(request, analysis, None, None)
        assert treatment["shot_mode"] == "locked_single_shot"
        assert len(treatment["locations"]) == 1
        import zolex_music_worker.worker as w
        from zolex_music_worker.config import WorkerConfig

        config = WorkerConfig.from_env(work_root=str(Path(".")))
        plan = w.build_plan(request, treatment, analysis, "16:9", config, None)
    finally:
        deactivate(token)
    assert len(plan.shots) >= 2
    assert {s.family for s in plan.shots} == {"medium_performance"}
    assert all(s.transition_out == "continue" for s in plan.shots)
    assert all(s.performer_ids == ["performer-1"] for s in plan.shots)


@needs_ffmpeg
def test_single_shot_windows_are_anchored_on_the_previous_last_frame(
    seam, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continuity is a picture, not a sentence: every window after the first
    starts on the frame the previous one ended on."""
    pkg = _package()
    request = _request("One continuous shot of her singing in a red-lit garage, no cuts.")
    brief = heuristic_brief(request.prompt)
    plan = _plan(shots=3, seconds=1)
    for shot in plan.shots:
        shot.anchor_image = str(tmp_path / "first.png")
    rendered: list[dict[str, str]] = []

    def fake_render_one_shot(*, shot, plan, audio, request, format_dir, config, journal,
                             worker_slot=0, gpu_id=None):
        rendered.append({"shot": shot.id, "anchor": shot.anchor_image})
        out = format_dir / "shots" / shot.id / "accepted.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        # The package renders synchronously in a worker thread; so does this.
        import subprocess

        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=size=64x36:duration=1:rate=24", "-pix_fmt", "yuv420p", str(out)],
            check=True, timeout=60,
        )
        return out

    class _Journal:
        def update(self, *_a, **_k) -> None:
            pass

    monkeypatch.setattr(pkg, "_render_one_shot", fake_render_one_shot)
    token = activate(Enforcement(brief=brief, job_id="j"))
    try:
        clips = pkg._render_all_shots(plan=plan, audio=None, request=request,
                                      format_dir=tmp_path, config=SimpleNamespace(ffmpeg="ffmpeg"),
                                      journal=_Journal())
    finally:
        deactivate(token)
    assert len(clips) == 3
    assert Path(rendered[0]["anchor"]).name == "first.png"
    assert Path(rendered[1]["anchor"]).parts[-2:] == ("shot-002", "continuity-anchor.png")
    assert Path(rendered[2]["anchor"]).parts[-2:] == ("shot-003", "continuity-anchor.png")
    assert Path(rendered[2]["anchor"]).stat().st_size > 0


def test_uninstall_restores_the_package(seam) -> None:
    pkg = _package()
    from zolex_music_worker.director import expand_direction as original

    assert pkg.expand_direction is not original
    uninstall()
    assert pkg.expand_direction is original
    install()  # the fixture's teardown uninstalls again
