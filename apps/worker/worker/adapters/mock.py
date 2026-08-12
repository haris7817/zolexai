"""Mock generation runtime — M1 only.

**No AI runs here.** This adapter walks a job through exactly the lifecycle a
real provider will, at roughly the pace the client approved during PRE-M1, and
produces a placeholder image so the whole path — claim, progress, SSE, upload,
history — is exercised end to end with no GPU and no provider account.

It exists so that M2 is a *substitution*, not an integration: a real adapter
implements the same three methods, is selected by the workflow's `execution`
block, and nothing above it changes.

**What it does not do.** It generates no video and no audio. Every workflow's
`execution` block declares `output_content_type: image/png`, so the placeholder
is labelled honestly all the way through: the asset is an image, the UI renders
it as an image, and nothing anywhere claims a video was produced.
"""

from __future__ import annotations

import asyncio
import hashlib
import struct
import zlib
from dataclasses import dataclass

from worker.adapters.base import AdapterJob, AdapterResult, ProgressCallback


@dataclass(frozen=True)
class Stage:
    status: str
    progress: int
    message: str
    seconds: float


#: Timings carry over from the approved PRE-M1 prototype (~6.6s total) — long
#: enough to read as real work, short enough to demonstrate live.
STAGES: tuple[Stage, ...] = (
    Stage("preparing", 22, "Setting up your generation…", 1.4),
    Stage("generating", 62, "This usually takes a couple of minutes.", 2.6),
    Stage("post_processing", 82, "Polishing and encoding…", 1.0),
    Stage("uploading", 94, "Almost ready…", 0.8),
)

#: 16:9 unless the request says otherwise. Music has no aspect ratio at all, so
#: it falls through to this.
_DIMENSIONS: dict[str, tuple[int, int]] = {
    "16:9": (960, 540),
    "9:16": (540, 960),
    "1:1": (720, 720),
    "4:5": (720, 900),
}


class MockAdapter:
    name = "mock"

    def supports(self, workflow_id: str) -> bool:
        # Runtime selection happens by workflow `execution.runtime`, so the mock
        # deliberately accepts anything routed to it rather than keeping its own
        # list that could drift from the registry.
        return True

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        for stage in STAGES:
            # Honours cancellation between stages like a real adapter, so the
            # mock keeps exercising the same control flow the GPU worker will.
            job.raise_if_cancelled()
            await on_progress(stage.status, stage.progress, stage.message)
            await asyncio.sleep(stage.seconds)

        width, height = _DIMENSIONS.get(str(job.parameters.get("aspect_ratio") or ""), (960, 540))
        content = _placeholder_png(width, height, seed=f"{job.workflow_id}:{job.job_id}")

        destination = job.workspace / "output.png"
        destination.write_bytes(content)

        return AdapterResult(
            path=destination,
            content_type="image/png",
            kind="image",
            width=width,
            height=height,
            duration_seconds=_duration_seconds(job.parameters.get("duration")),
        )


def _duration_seconds(value: object) -> float | None:
    """Turns "10s" into 10.0 so the asset carries the requested length."""
    text = str(value or "").strip().lower().removesuffix("s")
    try:
        return float(text)
    except ValueError:
        return None


# ── Placeholder image ────────────────────────────────────────────────────
#
# Written by hand rather than pulling in Pillow: the worker's dependency list is
# deliberately tiny (see pyproject.toml), and a PNG encoder for a flat gradient
# is a dozen lines of zlib and struct.


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _placeholder_png(width: int, height: int, *, seed: str) -> bytes:
    """A diagonal gradient in the ZolexAI palette.

    The seed makes consecutive results visually distinct, so a client running
    several generations can tell them apart at a glance instead of seeing four
    identical tiles.
    """
    digest = hashlib.sha256(seed.encode()).digest()
    # Dark base with a lime lift — the brand's surface range, never full
    # saturation, so a grid of these does not read as a wall of green.
    top = (18 + digest[0] % 14, 26 + digest[1] % 22, 10 + digest[2] % 10)
    bottom = (8, 10, 8)

    rows = bytearray()
    for y in range(height):
        ratio = y / max(1, height - 1)
        rows.append(0)  # PNG filter type 0 (None) for this scanline
        for x in range(width):
            # Diagonal blend, clamped so neither end can overflow a byte.
            blend = min(1.0, ratio * 0.75 + (x / max(1, width - 1)) * 0.25)
            rows.extend(
                bytes(
                    int(top[i] + (bottom[i] - top[i]) * blend) & 0xFF
                    for i in range(3)
                )
            )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(bytes(rows), 6))
        + _chunk(b"IEND", b"")
    )
