"""Burned-in captions, found and painted out before the clip is enlarged.

The client reported (10 Sep 2026) that delivered files carried captions with
no subtitle stream to strip: "the captions are burned into the generated
pixels by LTX, which is why `ffmpeg -sn` will not remove them." Confirmed on
job 70a97bf1 — 18 of 24 sampled frames, white text on a dark outline, bottom
centre.

## Why the prompt could not fix it

The first attempt was a no-text clause in the positive prompt
(`worker/prompt/no_text.py`), which is still there and still worth having.
It did not work, and the reason is visible in the failing render's own prompt
trace: that job asked for no captions **twice** — the guideline rewriter's own
"No subtitles, captions, or on-screen text" plus our clause — and the model
drew them anyway.

What the frames show is why. The text is not the dialogue; it is not words at
all. "What a tasty carırtt you have there." "I'll nibile genntly, thankıes for
sharing." The model is not captioning the speech, it is drawing the SHAPE of
subtitles, because a talking-animal clip in its training data has a white line
across the bottom. That is a visual prior, not an instruction it is choosing
to disobey, and no third prohibition reaches it.

## Where this runs

Between `service.collect` and the finishing pass: on the generation canvas
(864x480), before any enlargement. The client asked for that ordering —
"removal before 4K upscaling, because removing text after upscale is slower
and leaves larger artifacts" — and it is right twice over. The work is 20x
cheaper on the small frame, and a stroke painted out at 864 wide is
interpolated by the same lanczos that carries the rest of the picture, rather
than being repaired at 8K where the repair is what gets magnified.

## What it does not do

It does not touch audio. `clean` writes video only and the caller remuxes the
original soundtrack, so the track the graph wrote survives bit for bit — the
same discipline as the finishing pass.

It is spatial, not temporal: each frame is repaired from its own surroundings.
ProPainter, which the client named, is the temporal answer and stays the next
step if a clip appears where this is not enough. This is first because it
costs one decode pass and no weights, on a node that has already been
OOM-killed once at its memory ceiling.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from worker.core.config import settings
from worker.core.logging import get_logger
from worker.media.ffmpeg import FfmpegError, ffmpeg

logger = get_logger(__name__)


@dataclass(frozen=True)
class CaptionReport:
    """What the detector found, and whether anything was repaired."""

    detected: bool
    cleaned: bool
    ratio: float = 0.0
    frames_inspected: int = 0
    frames_masked: int = 0
    frames_painted: int = 0
    max_confidence: float = 0.0
    #: The client's quality check: how much of the OUTPUT still reads as text.
    #: 0.0 is a clean repair. Anything above `ltx_caption_residual_alarm` is
    #: logged as a warning, because a half-fix that reports success is the
    #: failure this number exists to make impossible. None means not measured.
    residual_ratio: float | None = None
    sample_text: tuple[str, ...] = ()
    band: tuple[int, int] | None = None
    detail: str = ""

    @property
    def log_fields(self) -> dict:
        fields = {
            "captions_detected": self.detected,
            "captions_cleaned": self.cleaned,
            "captions_ratio": self.ratio,
            "captions_frames_masked": self.frames_masked,
            "captions_frames_painted": self.frames_painted,
            "captions_confidence": self.max_confidence,
        }
        if self.residual_ratio is not None:
            fields["captions_residual_ratio"] = self.residual_ratio
        if self.band:
            fields["captions_band"] = f"{self.band[0]}-{self.band[1]}"
        if self.sample_text:
            # What the recogniser actually read. On the log because it is how
            # a human sees at a glance that this is caption-shaped nonsense
            # ("Nicch weatther a hop") and not a sign in the customer's scene.
            fields["captions_text"] = " | ".join(self.sample_text)[:200]
        if self.detail:
            fields["captions_detail"] = self.detail[-300:]
        return fields


async def remove_captions(
    clip: Path, *, timeout: float, log_extra: dict | None = None
) -> tuple[Path, CaptionReport]:
    """`clip` with any burned-in caption painted out, and what was found.

    Returns the ORIGINAL path unchanged whenever nothing was detected, the
    detector could not run, or the repair failed. Every failure here is
    non-fatal by design: the render already cost GPU minutes, and a clip with
    a caption is worth more to the customer than no clip at all. The report
    says which happened, and the caller logs it.
    """
    extra = dict(log_extra or {})
    out = clip.with_name(f"{clip.stem}-nocaption.mp4")
    argv = [*settings.ltx_caption_argv, "--video", str(clip), "--out", str(out)]

    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(settings.ltx_repo_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as exc:
        logger.warning("caption_detector_unavailable", extra={**extra, "detail": str(exc)})
        return clip, CaptionReport(False, False, detail=str(exc))

    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        logger.warning("caption_detector_timeout", extra={**extra, "seconds": timeout})
        return clip, CaptionReport(False, False, detail=f"timed out after {timeout:.0f}s")

    text = (stdout or b"").decode("utf-8", "replace").strip()
    if process.returncode != 0:
        logger.warning(
            "caption_detector_failed",
            extra={**extra, "returncode": process.returncode, "detail": text[-300:]},
        )
        return clip, CaptionReport(False, False, detail=text[-300:])

    try:
        # The script prints one JSON object last; anything before it is noise
        # from the environment (uv, torch import warnings) and is not ours.
        raw = json.loads(text.splitlines()[-1])
    except (ValueError, IndexError):
        logger.warning("caption_detector_unreadable", extra={**extra, "detail": text[-300:]})
        return clip, CaptionReport(False, False, detail=text[-300:])

    band = raw.get("band") or {}
    residual = raw.get("residual_ratio")
    report = CaptionReport(
        detected=bool(raw.get("detected")),
        cleaned=bool(raw.get("cleaned")) and out.exists(),
        ratio=float(raw.get("ratio") or 0.0),
        frames_inspected=int(raw.get("frames_inspected") or 0),
        frames_masked=int(raw.get("frames_masked") or 0),
        frames_painted=int(raw.get("frames_painted") or 0),
        max_confidence=float(raw.get("max_confidence") or 0.0),
        residual_ratio=float(residual) if residual is not None else None,
        sample_text=tuple(str(t) for t in (raw.get("sample_text") or [])[:6]),
        band=(band["top"], band["bottom"]) if band else None,
        detail=str(raw.get("error") or raw.get("detail") or ""),
    )
    if not report.cleaned:
        return clip, report

    # The client's quality check, acted on rather than merely recorded: text
    # still readable in the output means the repair did not finish the job,
    # and that must be visible in the log as a warning, not buried in a field
    # on an INFO line that reads like a success.
    if report.residual_ratio and report.residual_ratio > settings.ltx_caption_residual_alarm:
        logger.warning("captions_survived_repair", extra={**extra, **report.log_fields})

    # The repaired video carries no audio: give it the original's, copied.
    joined = clip.with_name(f"{clip.stem}-nocaption-av.mp4")
    try:
        await ffmpeg(
            ["-i", str(out), "-i", str(clip), "-map", "0:v:0", "-map", "1:a:0?",
             "-c:v", "copy", "-c:a", "copy", "-shortest",
             "-movflags", "+faststart", str(joined)],
            timeout=timeout,
        )
    except FfmpegError as exc:
        logger.warning("caption_remux_failed", extra={**extra, "detail": str(exc)[-300:]})
        return clip, CaptionReport(**{**report.__dict__, "cleaned": False,
                                      "detail": str(exc)[-300:]})
    logger.info("captions_removed", extra={**extra, **report.log_fields})
    return joined, report


__all__ = ["CaptionReport", "remove_captions"]
