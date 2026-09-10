"""Burned-in captions: OCR detection, temporal inpainting, and a check.

The client reported captions with no subtitle stream to strip, and specified
the pipeline: OCR detection -> expand mask -> temporal inpainting -> upscale.
`scripts/caption_clean.py` is that, with EasyOCR and ProPainter.

## Why this file no longer pins a pile of thresholds

It used to. A first implementation detected captions with hand-tuned `cv2`
heuristics — a brightness floor, a sparseness band, a centredness test — and
these tests pinned each constant with the measurement behind it. It worked on
the clip it was written against and **missed the very next one**: job
6bdf08bf scored `ratio 0.0`, not a single candidate frame, on a clip whose
captions are plainly legible. Its text was fainter and lower than the one the
constants were fitted to.

That is the whole lesson, and it is why the assertions here are about the
SHAPE of the pipeline rather than about numbers describing what a caption
looks like. EasyOCR found the same clip at 6 of 12 frames with confidence up
to 1.00. A detector trained on text does not need constants describing text.

Measured on the client-test node, 10 Sep 2026, job 6bdf08bf:
ratio 0.562, band measured at 408-456, 215 of 361 frames masked, all 361
repainted, **residual_ratio 0.0**, 94 s end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worker.core.config import settings
from worker.media.captions import CaptionReport, remove_captions


def _script_source() -> str:
    return (Path(__file__).resolve().parents[1] / "scripts" / "caption_clean.py").read_text(
        encoding="utf-8"
    )


# ── The pipeline is the client's, not an approximation of it ───────────────


def test_detection_is_ocr_and_not_a_pile_of_thresholds() -> None:
    """The client named PaddleOCR or EasyOCR, and they were right for a
    reason we measured the hard way: hand-fitted constants describe the ONE
    caption style they were fitted to."""
    source = _script_source()
    assert "import easyocr" in source
    assert "reader.readtext" in source and "reader.detect" in source


def test_inpainting_is_temporal_and_not_a_per_frame_smear() -> None:
    """"Temporal video inpainting such as ProPainter" — the client's words.
    Two cheaper things were tried and neither is good enough: `delogo` smears
    the whole box, and a per-frame `cv2.inpaint` leaves visible softening.
    ProPainter reconstructs from neighbouring frames, which is why the grass
    behind a removed caption comes back as grass."""
    source = _script_source()
    assert "run_streaming_propainter" in source
    assert "cv2.inpaint" not in source


def test_the_caption_band_is_measured_from_the_clip_not_assumed() -> None:
    """The failure of the previous version in one assertion. Where a caption
    sits is a property of the clip, and the per-frame mask pass is bounded by
    what THIS clip's own detection pass found — not by a constant."""
    source = _script_source()
    assert '"band"' in source
    assert "np.percentile(tops, 5)" in source
    assert "band_pad" in source


def test_the_mask_is_expanded_as_the_client_asked() -> None:
    """"Expand caption mask" is step two of their pipeline. A box that hugs
    the glyphs leaves the antialiased rim behind."""
    source = _script_source()
    assert "mask-dilation" in source
    assert "cv2.dilate" in source


def test_the_repair_is_checked_afterwards() -> None:
    """The client asked for an OCR quality check: "if text is detected and
    captions are disabled, it should automatically inpaint the affected
    frames or regenerate the video." The output is read again and what
    survived is reported, so a half-fix cannot be mistaken for a clean one."""
    source = _script_source()
    assert "def verify(" in source
    assert "residual_ratio" in source


def test_removal_happens_before_the_upscale() -> None:
    """"Perform the removal before 4K upscaling, because removing text after
    upscale is slower and leaves larger artifacts." In the adapter that means
    between `collect` and `_finish`, on the generation canvas."""
    source = (
        Path(__file__).resolve().parents[1] / "worker/adapters/ltx_hd.py"
    ).read_text(encoding="utf-8")
    removal = source.index("remove_captions(")
    finish = source.index("output = await self._finish(")
    assert removal < finish


# ── What the worker does with the answer ───────────────────────────────────


async def test_a_detector_that_cannot_start_never_fails_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The render is already paid for. A clip with a caption is worth more to
    the customer than no clip, so every failure on this path returns the
    original and says so in the report."""
    monkeypatch.setattr(
        settings, "ltx_caption_command", "definitely-not-a-real-program-xyz", raising=False
    )
    clip = tmp_path / "render.mp4"
    clip.write_bytes(b"not really a video")
    out, report = await remove_captions(clip, timeout=5)
    assert out == clip
    assert report.detected is False and report.cleaned is False
    assert report.detail


async def test_unreadable_output_is_survived_not_parsed_optimistically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The script prints one JSON object last, after whatever `uv`, torch and
    EasyOCR write to the same stream. Anything else means the run is not
    trustworthy and the original clip is returned."""
    import sys

    monkeypatch.setattr(
        settings,
        "ltx_caption_command",
        f'"{sys.executable}" -c "print(\'Downloading detection model...\')"',
        raising=False,
    )
    monkeypatch.setattr(settings, "ltx_repo_dir", tmp_path, raising=False)
    clip = tmp_path / "render.mp4"
    clip.write_bytes(b"x")
    out, report = await remove_captions(clip, timeout=30)
    assert out == clip
    assert report.cleaned is False


async def test_a_clean_clip_is_returned_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`detected: false` must not produce a re-encode. Most jobs are this
    case, and ProPainter on every one of them would double every job's wall
    time for nothing."""
    import sys

    answer = json.dumps({"detected": False, "cleaned": False, "ratio": 0.06,
                         "frames_inspected": 16, "max_confidence": 0.31})
    fake = tmp_path / "fake_detector.py"
    fake.write_text(f"print({answer!r})", encoding="utf-8")
    monkeypatch.setattr(
        settings, "ltx_caption_command", f'"{sys.executable}" "{fake}"', raising=False
    )
    monkeypatch.setattr(settings, "ltx_repo_dir", tmp_path, raising=False)
    clip = tmp_path / "render.mp4"
    clip.write_bytes(b"x")
    out, report = await remove_captions(clip, timeout=30)
    assert out == clip
    assert report.detected is False
    assert report.ratio == 0.06


def test_a_surviving_caption_is_a_warning_not_a_quiet_field() -> None:
    """The quality check only earns its keep if a bad answer is loud. Above
    the alarm the worker logs `captions_survived_repair` at WARNING rather
    than leaving the number on an INFO line that reads like a success."""
    source = (
        Path(__file__).resolve().parents[1] / "worker/media/captions.py"
    ).read_text(encoding="utf-8")
    assert "captions_survived_repair" in source
    assert "ltx_caption_residual_alarm" in source
    assert 0.0 < settings.ltx_caption_residual_alarm < 1.0


def test_the_report_says_enough_to_answer_is_it_still_happening() -> None:
    """These fields land on `ltx_hd_finished`, so one grep over the worker log
    answers the client's question without joining events per job — including
    what the recogniser actually read, which is how a human sees at a glance
    that it is caption nonsense and not a sign in the scene."""
    report = CaptionReport(
        True, True, ratio=0.562, frames_masked=215, frames_painted=361,
        max_confidence=1.0, residual_ratio=0.0, band=(408, 456),
        sample_text=("Nicch weatther a", "Ittlle friend"),
    )
    fields = report.log_fields
    assert fields["captions_detected"] is True
    assert fields["captions_cleaned"] is True
    assert fields["captions_residual_ratio"] == 0.0
    assert fields["captions_band"] == "408-456"
    assert "Nicch weatther a" in fields["captions_text"]


async def test_a_job_that_wants_text_on_screen_is_left_alone(tmp_path: Path) -> None:
    """`allow_captions` is one switch for both halves. A customer who asked
    for a sign or a title in frame must not have OCR hunt it down after the
    prompt clause has already stood aside for them."""
    from worker.adapters.ltx_hd import LtxHdAdapter

    source = (
        Path(__file__).resolve().parents[1] / "worker/adapters/ltx_hd.py"
    ).read_text(encoding="utf-8")
    assert "settings.ltx_caption_removal and not self._allow_captions(job)" in source
    assert LtxHdAdapter()._allow_captions(_job(tmp_path, allow_captions=True)) is True


def _job(workspace: Path, **params):
    from worker.adapters.base import AdapterJob
    from worker.adapters.ltx_hd import WORKFLOW_ID

    return AdapterJob(
        job_id="hd-job",
        workflow_id=WORKFLOW_ID,
        workflow_version="1",
        prompt="a neon shopfront",
        parameters={"duration": "8s", **params},
        execution={"runtime": "ltx_hd"},
        workspace=workspace,
    )
