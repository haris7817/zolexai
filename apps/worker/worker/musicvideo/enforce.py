"""The platform's planner, installed in front of the client's — at the seam.

`zolex_music_worker` is vendored verbatim and never edited (see
`worker/adapters/music_video.py`). Its `worker.py` binds the stages it runs by
name — `expand_direction`, `build_plan`, `compile_shot_prompts`,
`generate_anchors`, `_render_all_shots` — so this module rebinds those five
names on the package's `worker` module to wrappers that consult a per-job
context. With no context set, every wrapper calls the original untouched;
that is the kill switch, and it is also what every other test in this suite
exercises.

## What the wrappers do, in the package's own order

  1. `expand_direction` — the treatment's locations become the customer's,
     in story order, and the brief rides along in the treatment.
  2. `build_plan` — for a single-shot request, every shot is the same
     family in the same place and marked `continue`.
  3. `generate_anchors` — runs the prompt composition and the coverage check
     FIRST, so a plan missing a mandatory element is refused before the first
     still is generated. That is the client's "no expensive rendering until
     coverage reaches 100%" rule, and it has to live here because the package
     compiles its prompts *after* its anchors. A dry run stops here.
  4. `compile_shot_prompts` — the package's prompt (identity, camera,
     palette, audio, continuity — all still theirs) is led by the customer's
     story: theme, this shot's location and beat, the props that beat needs,
     and their prohibitions. Idempotent, because the package calls it again
     after anchors and must get the same text.
  5. `_render_all_shots` — for a single-shot request, sequential, each shot
     anchored on the previous accepted clip's last frame.

## The lip-sync routing rule

The client's rule — *a visible singer during active vocals must be
audio-conditioned* — is already how every shot renders here: both backends
pass the song and the shot's exact start time (`mv_render.py`). What was NOT
true is that the prompt agreed. `apply_lyric_directions` sets
`vocals_present` only for three performance families, so a performer in a
`wide_movement` shot during a sung line was told "the mouth remains naturally
at rest" while the audio sang. Step 4 fixes that at the prompt: a shot with a
performer AND active lyric segments sings. Never classify a singer-facing
shot as B-roll — their words.

## Traceability

`prompt-trace.json` beside the plan: the original prompt, the brief, and for
every shot the planned prompt, the final prompt, its SHA-256, the audio
window, the performers, the route and whether lip-sync was required. The
client asked to be able to prove the customer's prompt reached every stage;
this is that proof.
"""

from __future__ import annotations

import contextvars
import dataclasses
import hashlib
import json
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from worker.core.logging import get_logger
from worker.musicvideo.brief import CreativeBrief, Event, coverage

logger = get_logger(__name__)

RENDER_ROUTE = "audio_conditioned_a2v"
"""Every shot's route. There is no fast text-only route on this node: both
render backends condition on the song (`scripts/mv_render.py`)."""


@dataclass
class Enforcement:
    """One job's instructions to the wrappers."""

    brief: CreativeBrief
    job_id: str = ""
    workflow_version: str = ""
    trace_dir: Path | None = None
    dry_run: bool = False
    notes: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def single_shot(self) -> bool:
        return self.brief.single_shot


class DryRunComplete(Exception):
    """Raised at the GPU boundary on a dry run. Carries the report path."""

    def __init__(self, report: Path, summary: dict[str, Any]) -> None:
        super().__init__(f"dry run complete: {report}")
        self.report = report
        self.summary = summary


_CONTEXT: contextvars.ContextVar[Enforcement | None] = contextvars.ContextVar(
    "music_video_enforcement", default=None
)
_ORIGINALS: dict[str, Any] = {}
_LOCK = threading.Lock()


def activate(enforcement: Enforcement) -> contextvars.Token:
    return _CONTEXT.set(enforcement)


def deactivate(token: contextvars.Token) -> None:
    _CONTEXT.reset(token)


def install() -> None:
    """Rebind the five names on the package's worker module. Idempotent."""
    with _LOCK:
        if _ORIGINALS:
            return
        import zolex_music_worker.worker as pkg

        _ORIGINALS.update(
            expand_direction=pkg.expand_direction,
            build_plan=pkg.build_plan,
            compile_shot_prompts=pkg.compile_shot_prompts,
            generate_anchors=pkg.generate_anchors,
            _render_all_shots=pkg._render_all_shots,
        )
        pkg.expand_direction = _expand_direction
        pkg.build_plan = _build_plan
        pkg.compile_shot_prompts = _compile_shot_prompts
        pkg.generate_anchors = _generate_anchors
        pkg._render_all_shots = _render_all_shots
        logger.info("music_video_enforcement_installed")


def uninstall() -> None:
    """Tests only: put the package back exactly as imported."""
    with _LOCK:
        if not _ORIGINALS:
            return
        import zolex_music_worker.worker as pkg

        for name, original in _ORIGINALS.items():
            setattr(pkg, name, original)
        _ORIGINALS.clear()


# ── 1. treatment ────────────────────────────────────────────────────────────


def _expand_direction(request, analysis, transcript=None, reference_style=None):
    treatment = _ORIGINALS["expand_direction"](request, analysis, transcript, reference_style)
    ctx = _CONTEXT.get()
    if ctx is None:
        return treatment
    brief = ctx.brief
    treatment["original_direction"] = request.prompt
    treatment["creative_brief"] = brief.to_dict()
    treatment["prompt_enforced"] = True
    if brief.theme:
        treatment["concept_title"] = brief.theme[:80]
    if brief.locations:
        # The customer's places, in the order they told the story. The
        # package's `_location_for_section` walks this list from opening to
        # ending, so story order and song order line up.
        treatment["locations"] = (
            [brief.locations[0]] if brief.single_shot else list(brief.locations)
        )
        treatment["genre_locations_replaced"] = True
    if brief.camera:
        treatment["camera_style"] = brief.camera
    if brief.single_shot:
        treatment["cut_style"] = "one continuous take with no cuts"
        treatment["shot_mode"] = "locked_single_shot"
    else:
        treatment["shot_mode"] = "multi_shot"
    forbidden = list(treatment.get("unrequested_elements_forbidden") or [])
    for item in brief.prohibited:
        if item.casefold() not in {f.casefold() for f in forbidden}:
            forbidden.append(item)
    treatment["unrequested_elements_forbidden"] = forbidden
    return treatment


# ── 2. plan ─────────────────────────────────────────────────────────────────

_PERFORMANCE_FAMILIES = {"close_performance", "medium_performance", "firelight_performance"}


def _build_plan(request, treatment, analysis, output_format, config, transcript=None):
    plan = _ORIGINALS["build_plan"](request, treatment, analysis, output_format, config, transcript)
    ctx = _CONTEXT.get()
    if ctx is None or not ctx.single_shot:
        return plan
    # One take: one family, one place, one set of people, every boundary a
    # continuation rather than a cut. The boundaries themselves stay — the
    # model renders at most one window at a time — but nothing about the
    # picture is allowed to change across them.
    from zolex_music_worker.planner import _section_for

    performers = next((list(s.performer_ids) for s in plan.shots if s.performer_ids), [])
    family = "medium_performance" if performers else "environment"
    for shot in plan.shots:
        shot.family = family
        shot.anchor_role = family
        shot.performer_ids = list(performers)
        shot.transition_out = "continue"
        midpoint = shot.start_frame + shot.frame_count // 2
        section = _section_for(midpoint, analysis.sections)
        shot.vocals_present = bool(
            getattr(section, "vocals_present", False) and family in _PERFORMANCE_FAMILIES
        )
    ctx.notes.append(f"single_shot: {len(plan.shots)} continuous windows of {family}")
    return plan


# ── 4. prompts ──────────────────────────────────────────────────────────────


def _events_for(brief: CreativeBrief, index: int, total: int) -> list[Event]:
    """The story beats this shot carries, chronologically.

    With more shots than beats, consecutive shots share a beat; with more
    beats than shots, a shot carries several. The second case is the one
    that bit: a three-shot song with six beats dropped the last three, the
    coverage check refused the plan, and the customer got an error for
    writing a detailed prompt over a short song.
    """
    events = brief.events
    if not events:
        return []
    total = max(total, 1)
    start = index * len(events) // total
    stop = (index + 1) * len(events) // total
    if stop <= start:
        return [events[min(start, len(events) - 1)]]
    return list(events[start:stop])


def _location_for(brief: CreativeBrief, event: Event | None) -> str:
    if not brief.locations:
        return ""
    if brief.single_shot or event is None:
        return brief.locations[0]
    if 0 <= event.location_index < len(brief.locations):
        return brief.locations[event.location_index]
    return brief.locations[0]


def _share(items: list[str], index: int, total: int) -> list[str]:
    """The items whose turn it is: index-th of every `total`."""
    return [item for offset, item in enumerate(items) if offset % max(total, 1) == index]


def _props_for(brief: CreativeBrief, events: list[Event], index: int, total: int) -> list[str]:
    """Props named in this shot's beats, plus a share of any prop no beat
    names so every one is asked for somewhere."""
    if not brief.props:
        return []
    lowered = " ".join(e.action for e in events).casefold()
    mine = [p for p in brief.props if p.casefold() in lowered]
    orphaned = [
        p for p in brief.props
        if not any(p.casefold() in e.action.casefold() for e in brief.events)
    ]
    for prop in _share(orphaned, index, total):
        if prop not in mine:
            mine.append(prop)
    return mine


def _orphan_locations(brief: CreativeBrief, index: int, total: int) -> list[str]:
    """Places the customer named that no beat points at -- a writer's brief
    can do that -- handed out across the shots so each is asked for once."""
    pointed = {e.location_index for e in brief.events}
    orphaned = [loc for i, loc in enumerate(brief.locations) if i not in pointed]
    return _share(orphaned, index, total)


def compose_shot_prompt(
    brief: CreativeBrief,
    package_prompt: str,
    *,
    package_location: str,
    index: int,
    total: int,
) -> str:
    """The customer's story in front, the package's craft behind, in the
    client's order: style lock, location, action, then identity / camera /
    audio / continuity (theirs, unchanged), then restrictions."""
    events = _events_for(brief, index, total)
    event = events[0] if events else None
    location = _location_for(brief, event)
    props = _props_for(brief, events, index, total)
    extra_places = [] if brief.single_shot else _orphan_locations(brief, index, total)

    lead: list[str] = []
    if brief.theme:
        lead.append(f"Story: {brief.theme.rstrip('.')}.")
    if brief.look:
        lead.append(f"Style throughout: {brief.look.rstrip('.')}.")
    if brief.single_shot:
        where = f" in {location}" if location else ""
        lead.append(
            f"This is one continuous take{where}: the camera keeps the same "
            "position and framing for the whole video, there are no cuts and no "
            "change of place, and this segment picks up exactly where the previous "
            "segment ended with the same people, light and composition."
        )
    elif location:
        lead.append(f"This shot is set in {location}.")
    if events:
        beats = " ".join(f"{e.action.rstrip('.')}." for e in events)
        lead.append(f"What happens in this shot: {beats}")
    if extra_places:
        lead.append("The story also passes through " + "; ".join(extra_places) + ".")
    if props:
        lead.append("Clearly visible: " + ", ".join(props) + ".")

    body = package_prompt
    if location and package_location and package_location != location:
        body = body.replace(f" in {package_location} under", f" in {location} under", 1)

    tail: list[str] = []
    forbidden = [p for p in brief.prohibited if p]
    if forbidden:
        tail.append("Not allowed: " + ", ".join(forbidden) + ".")
    return " ".join([*lead, body, *tail]).strip()


def _compile_shot_prompts(plan, request, treatment) -> None:
    ctx = _CONTEXT.get()
    if ctx is not None:
        # The client's lip-sync routing rule, applied where it bites: a shot
        # with a performer in it and sung lyrics under it SINGS. The package
        # only says so for three families; every other family told the
        # singer to keep her mouth shut while the track carried her voice.
        for shot in plan.shots:
            if shot.performer_ids and shot.lyric_segment_ids and not shot.vocals_present:
                shot.vocals_present = True
                ctx.notes.append(f"{shot.id}: performer under vocals -> sings ({shot.family})")
    _ORIGINALS["compile_shot_prompts"](plan, request, treatment)
    if ctx is None:
        return

    from zolex_music_worker.prompts import _location_for_section

    brief = ctx.brief
    total = len(plan.shots)
    locations = treatment.get("locations", [])
    shots_trace: list[dict[str, Any]] = []
    for index, shot in enumerate(plan.shots):
        planned = shot.prompt
        package_location = _location_for_section(shot.section, locations) if locations else ""
        shot.prompt = compose_shot_prompt(
            brief, planned, package_location=package_location, index=index, total=total
        )
        shots_trace.append(
            {
                "shot_id": shot.id,
                "audio_start_time": round(shot.start_frame / plan.fps, 3),
                "audio_end_time": round((shot.start_frame + shot.frame_count) / plan.fps, 3),
                "family": shot.family,
                "performer_identity_ids": list(shot.performer_ids),
                "vocals_present": shot.vocals_present,
                "lip_sync_required": bool(shot.performer_ids and shot.vocals_present),
                "renderer_route": RENDER_ROUTE,
                "planned_shot_prompt": planned,
                "final_renderer_prompt": shot.prompt,
                "prompt_sha256": hashlib.sha256(shot.prompt.encode("utf-8")).hexdigest(),
            }
        )

    report = coverage(brief, [s.prompt for s in plan.shots])
    ctx.trace = {
        "job_id": ctx.job_id,
        "workflow_version": ctx.workflow_version,
        "shot_mode": "locked_single_shot" if brief.single_shot else "multi_shot",
        "original_user_prompt": brief.original_prompt,
        "creative_brief": brief.to_dict(),
        "brief_source": brief.source,
        "shots": shots_trace,
        "coverage": report.to_dict(),
        "notes": list(ctx.notes),
        "status": "validated" if report.complete else "rejected",
    }
    _write_trace(ctx)
    logger.info(
        "music_video_prompt_enforced",
        extra={
            "job_id": ctx.job_id,
            "shots": total,
            "shot_mode": ctx.trace["shot_mode"],
            "brief_source": brief.source,
            "mandatory": len(brief.mandatory),
            "coverage": report.fraction,
            "missing": report.missing[:5],
        },
    )
    if not report.complete:
        from zolex_music_worker.errors import ValidationError

        raise ValidationError(
            "Creative brief coverage "
            f"{report.fraction:.0%}; the plan lacks: {'; '.join(report.missing[:4])}"
        )


def _write_trace(ctx: Enforcement) -> Path | None:
    if ctx.trace_dir is None:
        return None
    ctx.trace_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.trace_dir / "prompt-trace.json"
    path.write_text(json.dumps(ctx.trace, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ── 3. anchors: the GPU boundary ────────────────────────────────────────────


def _generate_anchors(*, plan, request, treatment, output_dir, adapter, timeout,
                      identity_references=None, concurrency=1):
    ctx = _CONTEXT.get()
    if ctx is None:
        return _ORIGINALS["generate_anchors"](
            plan=plan, request=request, treatment=treatment, output_dir=output_dir,
            adapter=adapter, timeout=timeout, identity_references=identity_references,
            concurrency=concurrency,
        )
    # Compose and validate BEFORE the first still exists. Raises on a plan
    # that lacks a mandatory element, which is the whole point of doing it here.
    _compile_shot_prompts(plan, request, treatment)
    if ctx.dry_run:
        ctx.trace["status"] = "dry_run_passed"
        report = _write_trace(ctx)
        raise DryRunComplete(report or Path("prompt-trace.json"), {
            "shots": len(plan.shots),
            "shot_mode": ctx.trace.get("shot_mode"),
            "coverage": ctx.trace.get("coverage"),
        })
    if ctx.single_shot and plan.shots:
        # One still for the whole take; every later window is anchored on the
        # previous window's last frame at render time. The placeholder keeps
        # the render contract honest (the anchor field must name a file).
        first_only = dataclasses.replace(plan, shots=[plan.shots[0]])
        _ORIGINALS["generate_anchors"](
            plan=first_only, request=request, treatment=treatment, output_dir=output_dir,
            adapter=adapter, timeout=timeout, identity_references=identity_references,
            concurrency=concurrency,
        )
        for shot in plan.shots[1:]:
            shot.anchor_image = plan.shots[0].anchor_image
        return None
    return _ORIGINALS["generate_anchors"](
        plan=plan, request=request, treatment=treatment, output_dir=output_dir,
        adapter=adapter, timeout=timeout, identity_references=identity_references,
        concurrency=concurrency,
    )


# ── 5. render: continuity ───────────────────────────────────────────────────


def last_frame(clip: Path, out: Path, ffmpeg: str = "ffmpeg") -> Path:
    """The final frame of `clip` as a PNG — the next window's first frame."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-sseof", "-0.08",
         "-i", str(clip), "-frames:v", "1", "-update", "1", str(out)],
        check=True, timeout=120,
    )
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(f"no last frame extracted from {clip}")
    return out


def _render_all_shots(*, plan, audio, request, format_dir, config, journal, **extra):
    ctx = _CONTEXT.get()
    if ctx is None or not ctx.single_shot:
        return _ORIGINALS["_render_all_shots"](
            plan=plan, audio=audio, request=request, format_dir=format_dir,
            config=config, journal=journal, **extra,
        )
    import zolex_music_worker.worker as pkg

    clips: list[Path] = []
    total = len(plan.shots)
    for index, shot in enumerate(plan.shots):
        if index > 0:
            anchor = format_dir / "shots" / shot.id / "continuity-anchor.png"
            ffmpeg = getattr(config, "ffmpeg", "ffmpeg")
            shot.anchor_image = str(last_frame(clips[-1], anchor, ffmpeg))
        journal.update(
            "rendering",
            shot_id=shot.id,
            shot_index=index + 1,
            shot_count=total,
            completed_shots=index,
            render_concurrency=1,
        )
        clip = pkg._render_one_shot(
            shot=shot, plan=plan, audio=audio, request=request, format_dir=format_dir,
            config=config, journal=journal, worker_slot=0, gpu_id=None,
        )
        clips.append(clip)
    return clips


__all__ = [
    "RENDER_ROUTE",
    "DryRunComplete",
    "Enforcement",
    "activate",
    "compose_shot_prompt",
    "deactivate",
    "install",
    "last_frame",
    "uninstall",
]
