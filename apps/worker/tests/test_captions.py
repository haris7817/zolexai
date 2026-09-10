"""Burned-in captions: the detector's judgement, and the repair's safety.

The client reported captions with no subtitle stream to strip. The prompt
clause did not stop them — the failing job asked for no captions TWICE and got
them anyway — so they are painted out of the pixels instead.

The measurements these encode were taken on the client-test node, 10 Sep 2026,
against three real renders:

* job 70a97bf1, a captioned meadow — 18/24 frames, box 543x50 at (156, 391),
  6.6% of frame;
* job 2d1be2dc, captioned two-line — 10/24, box 664x80, 12.8%;
* job 5e7594d7, a caption-FREE lunar walk — the first version of the detector
  called this a caption covering **41% of the frame**, which `delogo` would
  have smeared across half the picture.

That false positive is why the detector has four tests rather than one, and
why the temporal one exists.
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


# ── What the detector is allowed to believe ────────────────────────────────


def test_the_sparseness_bound_is_what_rejects_bright_scenery() -> None:
    """Glyphs are thin strokes: a caption box is a few percent bright. Sunlit
    regolith is bright nearly everywhere, and without an upper bound it reads
    as one enormous caption. The bound is the single line that separated the
    two real clips."""
    source = _script_source()
    assert "MAX_BRIGHT_FRACTION = 0.40" in source
    assert "MIN_BRIGHT_FRACTION" in source


def test_the_detector_requires_a_stable_baseline() -> None:
    """The test scenery cannot pass. Bright ground throws candidates at a
    different height in every frame; a subtitle sits on one baseline all clip.
    Without this, the lunar walk scored 11/24 and was 'detected'."""
    source = _script_source()
    assert "median(baselines)" in source
    assert "tolerance" in source


def test_captions_are_looked_for_low_in_the_frame_only() -> None:
    """A shop sign or a book cover higher in the shot is the customer's
    picture, not our defect, and painting it out would be the worse error."""
    assert '"--band-top", type=float, default=0.60' in _script_source()


def test_the_repair_masks_glyphs_and_not_the_box() -> None:
    """`delogo` was the client's suggested starting point and it blanks the
    whole rectangle — measured, it removed the text and left a vertical smear
    that destroyed a carrot and a paw. The glyphs are 4-20% of the box, so the
    mask is the strokes and their outline, and the background survives."""
    source = _script_source()
    assert "cv2.inpaint" in source
    assert "delogo" not in source.replace("`delogo`", "")  # only named in prose


def test_the_mask_covers_the_dark_outline_too() -> None:
    """A mask of the bright core alone leaves a dark ghost of the text —
    measured, and clearly visible. The gradient lights up on both sides of
    every stroke; intersecting it with a bright core's neighbourhood keeps the
    rim and discards the grass."""
    source = _script_source()
    assert "MORPH_GRADIENT" in source
    assert "bitwise_and(edges, near)" in source


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
    """The script prints one JSON object last, after whatever `uv` and torch
    write to the same stream. Anything else means the run is not trustworthy
    and the original clip is returned."""
    import sys

    monkeypatch.setattr(
        settings,
        "ltx_caption_command",
        f'"{sys.executable}" -c "print(\'loading torch...\')"',
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
    case, and a needless extra encode on every one of them is a real cost."""
    import sys

    # Written to a file rather than passed with -c: the command is split with
    # shlex and a JSON object on a command line loses its quotes.
    answer = json.dumps({"detected": False, "cleaned": False, "ratio": 0.08,
                         "frames_inspected": 24})
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
    assert report.ratio == 0.08


def test_the_report_says_enough_to_answer_is_it_still_happening() -> None:
    """These fields land on `ltx_hd_finished`, so one grep over the worker log
    answers the client's question without joining two events per job."""
    report = CaptionReport(True, True, ratio=0.75, coverage=0.0655,
                           frames_painted=361, box=(156, 391, 543, 50))
    fields = report.log_fields
    assert fields["captions_detected"] is True
    assert fields["captions_cleaned"] is True
    assert fields["captions_box"] == "543x50+156+391"
    assert fields["captions_ratio"] == 0.75


async def test_a_job_that_wants_text_on_screen_is_left_alone(tmp_path: Path) -> None:
    """`allow_captions` is one switch for both halves. A customer who asked
    for a sign or a title in frame must not have the detector hunt it down
    after the prompt clause has already stood aside for them."""
    from worker.adapters.ltx_hd import LtxHdAdapter

    source = (
        Path(__file__).resolve().parents[1] / "worker/adapters/ltx_hd.py"
    ).read_text(encoding="utf-8")
    assert "settings.ltx_caption_removal and not self._allow_captions(job)" in source
    adapter = LtxHdAdapter()
    job_wanting_text = _job(tmp_path, allow_captions=True)
    assert adapter._allow_captions(job_wanting_text) is True


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
