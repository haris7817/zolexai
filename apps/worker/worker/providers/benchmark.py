"""The LTX/H3 benchmark: cases, scoring, and the shape of a result.

Written before the GPU exists, on purpose. A benchmark designed after the
hardware arrives is a benchmark designed around whatever the first renders
happened to show — and the point of this exercise is to decide routing on
evidence, which means deciding what counts as evidence first.

Three rules the schema enforces rather than requests:

  * **Same semantics, both engines.** A case owns one ZolexAI request. The two
    providers compile it differently (that is the whole design), but a case
    that could only be expressed for one of them is marked as such rather than
    faked into a comparison.
  * **Lip-sync and audio response are scored separately.** They are not folded
    into an overall number, because a music video that looks lovely and does
    not follow the vocal has failed at the thing it was for.
  * **A single run is not a result.** Every case carries a repeat count, and
    the reliability fields are part of the record, not an afterthought — 481
    frames passed six times in isolation and failed five of fifteen on a
    shared card, and only the repeats caught it.

Nothing here runs a model. `scripts/dual_engine_bench.py` drives it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ── Scoring ──────────────────────────────────────────────────────────────

#: The overall score's components. Weights sum to 100; anything that is a
#: judgement rather than a measurement lives here, and everything here is
#: scored 1-10 by a human watching the output.
SCORE_WEIGHTS: dict[str, int] = {
    "visual_quality": 15,
    "prompt_adherence": 15,
    "temporal_consistency": 15,
    "identity_consistency": 15,
    "motion_quality": 10,
    "camera_adherence": 10,
    "reference_fidelity": 10,
    "long_form_continuity": 5,
    "seam_quality": 5,
}

#: Scored separately and never averaged into the above.
SEPARATE_SCORES = ("lip_sync", "audio_response")

#: The lip-sync ladder this project already uses, so a new engine's result is
#: comparable with the measurements already on record. B is where LTX's audio
#: tier measured; C has never been demonstrated on any path here.
LIP_SYNC_LEVELS = {
    "A": "audio exists in the output",
    "B": "mouth timing responds to the vocal",
    "C": "word- and phoneme-level articulation is synchronised",
}


def overall_score(scores: dict[str, float]) -> float | None:
    """The weighted overall, or None if any component is missing.

    Refusing to average a partial card is deliberate: a case that could not be
    scored on identity because the subject left the frame is not a case with a
    slightly lower score, it is an incomplete one.
    """
    if any(key not in scores for key in SCORE_WEIGHTS):
        return None
    total = sum(scores[key] * weight for key, weight in SCORE_WEIGHTS.items())
    return round(total / sum(SCORE_WEIGHTS.values()), 2)


# ── Cases ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    group: str
    title: str
    workflow_id: str
    prompt: str
    parameters: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    inputs: tuple[tuple[str, str], ...] = ()
    """(role, kind) pairs the case needs from the golden asset set."""

    measures: tuple[str, ...] = ()
    """What this case exists to separate. Free text, but it should name
    something the scoring card can actually carry."""

    repeats: int = 3
    """Three is the floor. Launch-critical paths take five."""

    both_engines: bool = True
    """False where no honest equivalent exists — recorded, not forced."""

    note: str = ""


def _case(*args, **kwargs) -> BenchmarkCase:
    return BenchmarkCase(*args, **kwargs)


_16_9 = {"aspect_ratio": "16:9"}


CASES: tuple[BenchmarkCase, ...] = (
    # ── A · text to video ────────────────────────────────────────────────
    _case("A1", "A", "Simple cinematic scene", "text-to-video",
          "A fishing boat crosses a still harbour at dawn, mist on the water.",
          {**_16_9, "duration": "5s"}, measures=("visual_quality", "motion_quality")),
    _case("A2", "A", "Human dialogue", "text-to-video",
          'A woman in a red coat turns to the camera and says, "You came back."',
          {**_16_9, "duration": "10s"},
          measures=("dialogue", "lip_sync", "identity_consistency"), repeats=5),
    _case("A3", "A", "Fast action", "text-to-video",
          "A cyclist swerves between traffic and skids to a stop at a wet kerb.",
          {**_16_9, "duration": "5s"}, measures=("motion_quality", "temporal_consistency")),
    _case("A4", "A", "Multiple characters", "text-to-video",
          "Three cooks work one narrow kitchen line: one plates, one calls orders, "
          "one carries a tray past them both.",
          {**_16_9, "duration": "10s"},
          measures=("identity_consistency", "prompt_adherence")),
    _case("A5", "A", "Camera control", "text-to-video",
          "A low-angle wide shot of a lighthouse; the camera arcs slowly around it.",
          {**_16_9, "duration": "10s"}, measures=("camera_adherence",)),
    _case("A6", "A", "Multi-shot scene", "text-to-video",
          "A wide shot of an empty platform; then a medium shot of a man checking "
          "his watch; then a close-up of the departure board flipping.",
          {**_16_9, "duration": "15s"},
          measures=("prompt_adherence", "seam_quality"),
          note="H3 documents shot timestamps; LTX's own guides forbid them — this "
               "case is where that compilation difference shows or does not"),
    _case("A7", "A", "Exact prompt adherence", "text-to-video",
          "Exactly two matte black cars idle at a kerb under a green neon sign. "
          "No other vehicles are on the street.",
          {**_16_9, "duration": "15s"}, measures=("prompt_adherence",), repeats=5),
    _case("A8", "A", "Thirty-second long form", "text-to-video",
          "A market wakes up: shutters go up, crates are stacked, a woman in a "
          "yellow apron sets out flowers and greets a neighbour.",
          {**_16_9, "duration": "30s"},
          measures=("long_form_continuity", "seam_quality"), repeats=5),
    _case("A9", "A", "Sixty-second long form", "text-to-video",
          "A market wakes up: shutters go up, crates are stacked, a woman in a "
          "yellow apron sets out flowers, greets a neighbour, then carries the "
          "empty crates back inside.",
          {**_16_9, "duration": "60s"},
          measures=("long_form_continuity", "seam_quality", "identity_consistency"),
          repeats=5,
          note="the headline long-form comparison: 2 LTX sections against 4 H3 ones"),

    # ── B · image to video ───────────────────────────────────────────────
    _case("B1", "B", "Portrait close-up", "image-to-video",
          "She lifts her chin and smiles faintly.",
          {**_16_9, "duration": "5s"}, inputs=(("source_image", "image"),),
          measures=("reference_fidelity", "identity_consistency"), repeats=5),
    _case("B2", "B", "Full-body person", "image-to-video",
          "He walks forward and stops, hands in his pockets.",
          {**_16_9, "duration": "5s"}, inputs=(("source_image", "image"),),
          measures=("reference_fidelity", "motion_quality")),
    _case("B3", "B", "Landscape", "image-to-video",
          "Clouds move over the ridge and grass bends in the wind.",
          {**_16_9, "duration": "5s"}, inputs=(("source_image", "image"),),
          measures=("visual_quality", "temporal_consistency")),
    _case("B4", "B", "Action from a still", "image-to-video",
          "The dog leaps off the step and runs out of frame.",
          {**_16_9, "duration": "5s"}, inputs=(("source_image", "image"),),
          measures=("motion_quality",)),
    _case("B5", "B", "Camera motion from a still", "image-to-video",
          "The camera pushes in slowly on the doorway.",
          {**_16_9, "duration": "5s"}, inputs=(("source_image", "image"),),
          measures=("camera_adherence",)),
    _case("B6", "B", "Dialogue from an image", "image-to-video",
          'She looks up and says, "It is later than you think."',
          {**_16_9, "duration": "10s"}, inputs=(("source_image", "image"),),
          measures=("lip_sync", "identity_consistency"), repeats=5),
    _case("B7", "B", "Thirty-second continuation", "image-to-video",
          "She sets down the cup, crosses to the window and opens it.",
          {**_16_9, "duration": "30s"}, inputs=(("source_image", "image"),),
          measures=("identity_consistency", "long_form_continuity")),
    _case("B8", "B", "Sixty-second continuation", "image-to-video",
          "She sets down the cup, crosses to the window, opens it, then returns "
          "to the table and sits.",
          {**_16_9, "duration": "60s"}, inputs=(("source_image", "image"),),
          measures=("identity_consistency", "long_form_continuity", "seam_quality"),
          repeats=5,
          note="LTX drops its identity anchor at this length (720 frames is not a "
               "measured two-image count) — H3 carries the reference every section"),

    # ── C · standard video to video ──────────────────────────────────────
    _case("C1", "C", "Style transformation", "video-to-video",
          "Turn this into a rain-soaked neon street at night.",
          dict(_16_9), inputs=(("source_video", "video"),),
          execution={"dry_run_source_seconds": 10.0},
          measures=("prompt_adherence", "reference_fidelity")),
    _case("C2", "C", "Source-motion preservation", "video-to-video",
          "Repaint it as a charcoal sketch, same movement.",
          dict(_16_9), inputs=(("source_video", "video"),),
          execution={"dry_run_source_seconds": 10.0},
          measures=("reference_fidelity", "motion_quality")),
    _case("C3", "C", "Camera preservation", "video-to-video",
          "Restyle as an oil painting; keep the camera move exactly.",
          dict(_16_9), inputs=(("source_video", "video"),),
          execution={"dry_run_source_seconds": 10.0}, measures=("camera_adherence",)),
    _case("C4", "C", "Dialogue clip", "video-to-video",
          "Restyle as a 1970s film print; the speech stays in time.",
          dict(_16_9), inputs=(("source_video", "video"),),
          execution={"dry_run_source_seconds": 10.0},
          measures=("lip_sync", "audio_response"), repeats=5),
    _case("C5", "C", "Action clip", "video-to-video",
          "Restyle as ink and wash; keep every movement readable.",
          dict(_16_9), inputs=(("source_video", "video"),),
          execution={"dry_run_source_seconds": 10.0}, measures=("motion_quality",)),
    _case("C6", "C", "Thirty-second clip", "video-to-video",
          "Restyle as a rain-soaked neon street.",
          dict(_16_9), inputs=(("source_video", "video"),),
          execution={"dry_run_source_seconds": 30.0},
          measures=("seam_quality", "long_form_continuity"), repeats=5),

    # ── D · reference-person video to video ──────────────────────────────
    _case("D1", "D", "Reference person, close-up", "video-to-video",
          "The same performance, with this person.",
          dict(_16_9),
          inputs=(("source_video", "video"), ("reference_image", "image")),
          execution={"v2v_reference_identity": True, "dry_run_source_seconds": 10.0},
          measures=("identity_consistency", "reference_fidelity"), repeats=5),
    _case("D2", "D", "Reference person, waist-up", "video-to-video",
          "The same performance, with this person.",
          dict(_16_9),
          inputs=(("source_video", "video"), ("reference_image", "image")),
          execution={"v2v_reference_identity": True, "dry_run_source_seconds": 10.0},
          measures=("identity_consistency",), repeats=5),
    _case("D3", "D", "Reference person, full-body", "video-to-video",
          "The same performance, with this person.",
          dict(_16_9),
          inputs=(("source_video", "video"), ("reference_image", "image")),
          execution={"v2v_reference_identity": True, "dry_run_source_seconds": 10.0},
          measures=("identity_consistency", "reference_fidelity"), repeats=5,
          note="the anchor-geometry case: a headshot reference over a full-body "
               "performance is the shape that produced the bust-at-the-feet bug"),
    _case("D4", "D", "Reference person, moving camera", "video-to-video",
          "The same performance, with this person.",
          dict(_16_9),
          inputs=(("source_video", "video"), ("reference_image", "image")),
          execution={"v2v_reference_identity": True, "dry_run_source_seconds": 10.0},
          measures=("identity_consistency", "camera_adherence")),
    _case("D5", "D", "Reference person across hard cuts", "video-to-video",
          "The same performance, with this person.",
          dict(_16_9),
          inputs=(("source_video", "video"), ("reference_image", "image")),
          execution={"v2v_reference_identity": True, "dry_run_source_seconds": 30.0},
          measures=("identity_consistency", "seam_quality"), repeats=5,
          note="identity is measured to drift across cuts on the LTX path"),

    # ── E · music video ──────────────────────────────────────────────────
    _case("E1", "E", "Fifteen-second music video", "music-video",
          "A singer in a neon-lit bar performs to camera.",
          dict(_16_9), inputs=(("source_audio", "audio"),),
          execution={"audio_conditioning": True, "dry_run_source_seconds": 15.0},
          measures=("lip_sync", "audio_response", "identity_consistency"), repeats=5),
    _case("E2", "E", "Thirty-second music video", "music-video",
          "A singer in a neon-lit bar performs to camera.",
          dict(_16_9), inputs=(("source_audio", "audio"),),
          execution={"audio_conditioning": True, "dry_run_source_seconds": 30.0},
          measures=("lip_sync", "seam_quality"), repeats=5),
    _case("E3", "E", "Sixty-second music video", "music-video",
          "A singer in a neon-lit bar performs to camera.",
          dict(_16_9), inputs=(("source_audio", "audio"),),
          execution={"audio_conditioning": True, "dry_run_source_seconds": 60.0},
          measures=("lip_sync", "seam_quality", "identity_consistency"), repeats=5),
    _case("E4", "E", "Two-minute music video", "music-video",
          "A singer in a neon-lit bar performs to camera.",
          dict(_16_9), inputs=(("source_audio", "audio"),),
          execution={"audio_conditioning": True, "dry_run_source_seconds": 120.0},
          measures=("lip_sync", "identity_consistency", "seam_quality"), repeats=3,
          note="run only if E3 is stable on both engines; cost grows fastest here"),

    # ── F · dialogue ─────────────────────────────────────────────────────
    _case("F1", "F", "One speaker", "text-to-video",
          'A man alone at a kitchen table says, "I should have called."',
          {**_16_9, "duration": "10s"},
          measures=("lip_sync", "dialogue"), repeats=5),
    _case("F2", "F", "Two speakers, turn-taking", "text-to-video",
          'A woman says, "You are early." A man answers, "You are not ready."',
          {**_16_9, "duration": "15s"},
          measures=("dialogue", "identity_consistency"), repeats=5),
    _case("F3", "F", "Emotional dialogue", "text-to-video",
          'A nurse steadies her voice and says, "He is going to be fine."',
          {**_16_9, "duration": "10s"}, measures=("dialogue", "visual_quality")),
    _case("F4", "F", "Reaction shot", "text-to-video",
          'A man says, "They sold it." The woman opposite says nothing and looks away.',
          {**_16_9, "duration": "15s"}, measures=("dialogue", "prompt_adherence")),

    # ── G · camera ───────────────────────────────────────────────────────
    # One case, twenty concepts, because the interesting output is a table of
    # which terms each engine honours — not twenty separate verdicts.
    _case("G1", "G", "Camera vocabulary sweep", "text-to-video",
          "A man stands beside a parked car on an empty road. {CAMERA}",
          {**_16_9, "duration": "5s"},
          measures=("camera_adherence",), repeats=3,
          note="run once per term: locked static, wide, medium, close-up, extreme "
               "close-up, low angle, high angle, eye level, overhead, "
               "over-the-shoulder, push-in, pull-out, pan, tilt, tracking, dolly, "
               "orbit, handheld, hard cut, multi-shot. H3's closed vocabulary maps "
               "some of these and not others — record which, per engine"),

    # ── H · long form ────────────────────────────────────────────────────
    _case("H1", "H", "Thirty seconds, one global plan", "text-to-video",
          "Two friends meet on a bridge, argue about a letter, and one walks away.",
          {**_16_9, "duration": "30s"},
          measures=("long_form_continuity", "dialogue", "seam_quality"), repeats=5),
    _case("H2", "H", "Sixty seconds, one global plan", "text-to-video",
          "Two friends meet on a bridge, argue about a letter, one walks away, and "
          "the other stays alone at the rail.",
          {**_16_9, "duration": "60s"},
          measures=("long_form_continuity", "dialogue", "seam_quality",
                    "identity_consistency"),
          repeats=5,
          note="carries a departure: the case that caught a returning character on "
               "LTX. Score exits explicitly"),

    # ── I · extend ───────────────────────────────────────────────────────
    _case("I1", "I", "Continue an existing clip", "extend-video",
          "He finishes the sentence and turns back to the window.",
          {**_16_9, "duration": "10s"}, inputs=(("source_video", "video"),),
          measures=("seam_quality", "identity_consistency", "audio_response"),
          repeats=5,
          note="LTX conditions on the source's final frame (upstream ships no "
               "extension pipeline); H3 documents video continuation as a Ref2VA "
               "task type. Different mechanisms, comparable outcome"),

    # ── J · multimodal references ────────────────────────────────────────
    _case("J1", "J", "Image plus video plus audio in one request", "video-to-video",
          "This person, moving like this, to this track.",
          dict(_16_9),
          inputs=(("reference_image", "image"), ("source_video", "video"),
                  ("source_audio", "audio")),
          execution={"dry_run_source_seconds": 10.0},
          measures=("reference_fidelity", "identity_consistency", "audio_response"),
          both_engines=False,
          note="H3 only. LTX has no request shape that carries all three: audio "
               "lives on a2vid, control video on ic_lora, and they are different "
               "entry points. Recorded as a capability, not scored as a contest"),
)


def cases_for_group(group: str) -> tuple[BenchmarkCase, ...]:
    return tuple(case for case in CASES if case.group == group.upper())


def case_by_id(case_id: str) -> BenchmarkCase | None:
    return next((case for case in CASES if case.id == case_id.upper()), None)


GROUPS: dict[str, str] = {
    "A": "Text to video",
    "B": "Image to video",
    "C": "Standard video to video",
    "D": "Reference-person video to video",
    "E": "Music video",
    "F": "Dialogue",
    "G": "Camera",
    "H": "Long form",
    "I": "Extend",
    "J": "Multimodal references",
}


# ── Results ──────────────────────────────────────────────────────────────


@dataclass
class RunRecord:
    """One generation. Every field is recorded even when it is boring — a
    result without its conditions cannot be compared with anything later."""

    case_id: str
    provider: str
    pipeline: str = ""
    run_index: int = 0
    gpu: str = ""
    cold_start: bool = True
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    frames: int = 0
    steps: int | None = None
    references: int = 0
    audio_input: bool = False
    model_load_seconds: float | None = None
    generation_seconds: float | None = None
    decode_seconds: float | None = None
    total_wall_seconds: float | None = None
    peak_vram_gb: float | None = None
    peak_host_ram_gb: float | None = None
    succeeded: bool = False
    failure_class: str = ""
    """oom | cublas | corrupt_output | duration_mismatch | audio_mismatch |
    identity_failure | other — the classes this project has actually seen."""

    retries: int = 0
    scores: dict[str, float] = field(default_factory=dict)
    lip_sync_level: str = ""
    notes: str = ""

    def overall(self) -> float | None:
        return overall_score(self.scores)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["overall"] = self.overall()
        return data


def result_skeleton() -> dict[str, Any]:
    """The empty result document the GPU session fills in.

    Shipped empty on purpose. A skeleton with plausible numbers in it is the
    single easiest way for a fabricated benchmark to end up in a decision.
    """
    return {
        "generated_on": None,
        "gpu": None,
        "ltx_commit": None,
        "h3_revision": None,
        "notes": "",
        "runs": [],
        "decisions": {
            workflow: {"provider": None, "evidence": {}}
            for workflow in (
                "t2v_short", "t2v_long", "i2v_short", "i2v_long",
                "v2v_standard", "v2v_reference", "music_video", "dialogue",
                "camera_heavy", "extend", "multimodal_reference",
                "fast_mode", "quality_mode",
            )
        },
    }
