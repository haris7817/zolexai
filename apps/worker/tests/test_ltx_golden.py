"""The LTX regression net for dual-engine work.

A second engine is being prepared beside LTX, and the one thing that must not
happen is LTX moving while nobody is looking. Provider abstractions are
exactly the kind of change that "shouldn't" alter behaviour and quietly does:
a helper reused, a default re-derived one layer up, a plan recomputed instead
of asked for.

So the argv LTX would receive is frozen here, byte for byte, for every shape
the product actually renders — the eight cases named in the dual-engine brief
plus the four public text-to-video durations. Paths are placeholdered because
they differ per machine; nothing else is.

Regenerating is deliberate and visible:

    ZOLEX_UPDATE_GOLDEN=1 pytest tests/test_ltx_golden.py

and the diff it produces is the change under review. A green run means the
engine that served production still receives what it received.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.conftest import make_job, staged_input
from worker.adapters.ltx import (
    _A2VID,
    _DISTILLED,
    _GUIDED,
    _IC_LORA,
    AudioConditioning,
    ConditioningFrame,
    ControlConditioning,
    LoraSpec,
    LtxAdapter,
)
from worker.core.config import settings

GOLDEN = Path(__file__).parent / "golden" / "ltx_invocations.json"


def _normalise(argv: list[str], workspace: Path) -> list[str]:
    """Machine-specific absolute paths become stable placeholders."""
    models = str(settings.ltx_models_root)
    work = str(workspace)
    out: list[str] = []
    for token in argv:
        text = str(token)
        text = text.replace(models, "<MODELS>").replace(work, "<WORK>")
        out.append(text.replace("\\", "/"))
    return out


def _cases(workspace: Path) -> dict[str, list[str]]:
    """Every shape, built exactly the way its handler builds it."""
    adapter = LtxAdapter()
    out = workspace / "out.mp4"
    still = workspace / "inputs" / "source_image.png"
    seam = workspace / "segment-condition-0001.png"
    grid = (1024, 576)
    cases: dict[str, list[str]] = {}

    # ── text-to-video, the four public durations ─────────────────────────
    # 5s/15s/30s are single passes; 60s is two 30s sections, so its second
    # section is the seam-conditioned shape.
    for label, seconds, frames in (
        ("t2v-5s", 5.0, 121),
        ("t2v-15s", 15.0, 361),
        ("t2v-30s", 30.0, 736),
    ):
        job = make_job(
            workspace,
            parameters={"duration": f"{int(seconds)}s", "aspect_ratio": "16:9"},
        )
        cases[label] = adapter._command(
            job, seconds, out, dimensions=grid, num_frames=frames,
            seed=adapter._seed_for_step(job, 0),
        )

    job60 = make_job(
        workspace,
        parameters={"duration": "60s", "aspect_ratio": "16:9"},
        execution={"runtime": "ltx", "max_segment_seconds": 30},
    )
    cases["t2v-60s-section-1"] = adapter._command(
        job60, 30.0, out, dimensions=grid, num_frames=736,
        seed=adapter._seed_for_step(job60, 0),
    )
    cases["t2v-60s-section-2"] = adapter._command(
        job60, 30.0, out, dimensions=grid, num_frames=720,
        conditioning=[ConditioningFrame(seam, 0, 1.0)],
        seed=adapter._seed_for_step(job60, 1),
    )

    # ── image-to-video ───────────────────────────────────────────────────
    i2v = make_job(
        workspace,
        workflow_id="image-to-video",
        parameters={"duration": "5s", "aspect_ratio": "16:9"},
        inputs=[staged_input("source_image", "image", "image/png", still)],
    )
    cases["i2v-5s"] = adapter._command(
        i2v, 5.0, out, dimensions=grid, num_frames=120,
        conditioning=[ConditioningFrame(still, 0, 1.0)],
        seed=adapter._seed_for_step(i2v, 0),
    )

    # ── extend-video ─────────────────────────────────────────────────────
    extend = make_job(
        workspace,
        workflow_id="extend-video",
        parameters={"duration": "15s", "aspect_ratio": "16:9"},
    )
    cases["extend-15s"] = adapter._command(
        extend, 15.0, out, dimensions=grid, num_frames=360,
        conditioning=[ConditioningFrame(workspace / "seed-frame.png", 0, 1.0)],
        seed=adapter._seed_for_step(extend, 0),
    )

    # ── video-to-video, both engines ─────────────────────────────────────
    v2v = make_job(
        workspace,
        workflow_id="video-to-video",
        prompt="repaint it as a charcoal sketch",
        parameters={"aspect_ratio": "16:9"},
    )
    keyframes = [
        ConditioningFrame(workspace / "keyframes" / f"pass-0000-{n:04d}.png", idx, 0.45)
        for n, idx in enumerate((0, 72, 216, 360, 503, 647))
    ]
    cases["v2v-restyle"] = adapter._command(
        v2v, 20.0, out, dimensions=grid, num_frames=720,
        conditioning=keyframes, seed=adapter._seed_for_step(v2v, 0),
    )
    cases["v2v-transform-iclora"] = adapter._command(
        v2v, 6.667, out, dimensions=grid, num_frames=193, pipeline=_IC_LORA,
        conditioning=[ConditioningFrame(workspace / "identity-anchor.png", 0, 1.0)],
        loras=(LoraSpec(settings.ltx_models_root / "loras" / "union.safetensors", 1.0),),
        control=ControlConditioning(workspace / "control-0000.mp4", 1.0),
        seed=adapter._seed_for_step(v2v, 0),
    )

    # ── guided tier ──────────────────────────────────────────────────────
    guided = make_job(
        workspace,
        parameters={"duration": "5s", "aspect_ratio": "16:9"},
        execution={"runtime": "ltx", "generation_engine": "guided"},
    )
    cases["guided-t2v"] = adapter._command(
        guided, 5.0, out, dimensions=grid, num_frames=121, pipeline=_GUIDED,
        seed=adapter._seed_for_step(guided, 0),
    )

    # ── audio tier (a2vid) ───────────────────────────────────────────────
    mv = make_job(
        workspace,
        workflow_id="music-video",
        parameters={"aspect_ratio": "16:9"},
        execution={"runtime": "ltx", "audio_conditioning": True},
    )
    cases["a2v-music-video"] = adapter._command(
        mv, 20.0417, out, dimensions=grid, num_frames=481, pipeline=_A2VID,
        audio=AudioConditioning(
            path=workspace / "inputs" / "source_audio.mp3",
            start_seconds=0.0,
            max_duration_seconds=20.0817,
        ),
        seed=adapter._seed_for_step(mv, 0),
    )

    assert _DISTILLED.module == "ltx_pipelines.distilled"  # the default tier
    return {name: _normalise(argv, workspace) for name, argv in cases.items()}


def test_ltx_invocations_are_unchanged(workspace: Path, fake_models: Path) -> None:
    """The eight named shapes, byte for byte.

    If this fails after a provider change, the provider change is wrong —
    not the golden file. LTX is the verified baseline the second engine is
    being measured against, and a baseline that moves measures nothing.
    """
    current = _cases(workspace)

    if os.environ.get("ZOLEX_UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        pytest.skip("golden snapshot rewritten")

    assert GOLDEN.exists(), (
        "no golden snapshot — regenerate with ZOLEX_UPDATE_GOLDEN=1 and review the diff"
    )
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))

    assert sorted(current) == sorted(expected), "the set of pinned shapes changed"
    for name in sorted(expected):
        assert current[name] == expected[name], f"LTX invocation changed for {name}"


def test_every_pinned_shape_still_names_its_pipeline(
    workspace: Path, fake_models: Path
) -> None:
    """A cheap guard on the snapshot itself: a case that silently stopped
    exercising its tier would freeze the wrong thing and still pass."""
    cases = _cases(workspace)
    assert "ltx_pipelines.ic_lora" in cases["v2v-transform-iclora"]
    assert "ltx_pipelines.ti2vid_two_stages" in cases["guided-t2v"]
    assert "ltx_pipelines.a2vid_two_stage" in cases["a2v-music-video"]
    assert "--audio-path" in cases["a2v-music-video"]
    assert "--skip-stage-2" in cases["v2v-transform-iclora"]
    for label in ("t2v-5s", "t2v-15s", "t2v-30s", "t2v-60s-section-1"):
        assert "ltx_pipelines.distilled" in cases[label]
        assert "--image" not in cases[label], "text-to-video's first pass is unconditioned"
