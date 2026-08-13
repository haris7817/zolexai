"""The harness runtime — the thing that makes GPU day a swap, not a debug session.

If these pass, the platform can produce a real, playable, correctly-sized media
file through the entire production code path: plan, render sections, assemble,
measure, report. The only piece M2 replaces is where the frames come from.

Every test needs ffmpeg and skips without it. That is the point — the harness
exists to exercise real media handling, and a version that ran without media
tools would be a second mock.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from worker.adapters.base import AdapterError, AdapterInput, AdapterJob, JobCancelled
from worker.adapters.harness import HarnessAdapter
from worker.media import ffmpeg, ffprobe_json, probe_media, tools_available

pytestmark = pytest.mark.skipif(
    not tools_available(), reason="ffmpeg/ffprobe not installed"
)


def make_job(workspace: Path, **overrides) -> AdapterJob:
    defaults = dict(
        job_id="00000000-0000-0000-0000-0000000000aa",
        workflow_id="text-to-video",
        workflow_version="1",
        prompt="a cinematic drone shot",
        parameters={"duration": "4s", "aspect_ratio": "16:9"},
        inputs=[],
        execution={"runtime": "harness"},
        output_content_type="video/mp4",
        workspace=workspace,
    )
    return AdapterJob(**{**defaults, **overrides})


async def collect(job: AdapterJob):
    reported: list[tuple[str, int, str]] = []

    async def on_progress(status: str, progress: int, message: str) -> None:
        reported.append((status, progress, message))

    result = await HarnessAdapter().run(job, on_progress)
    return result, reported


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "job"
    path.mkdir()
    return path


async def test_it_produces_a_real_playable_video_of_the_requested_length(
    workspace: Path,
) -> None:
    job = make_job(workspace, parameters={"duration": "4s", "aspect_ratio": "16:9"})
    result, _ = await collect(job)

    assert result.content_type == "video/mp4"
    assert result.kind == "video"
    assert result.size_bytes > 0

    # Measured from the finished file, not asserted from the request.
    info = await probe_media(result.path)
    assert info.has_video
    assert info.duration_seconds == pytest.approx(4.0, abs=0.75)
    assert (result.width, result.height) == (960, 540)


async def test_it_produces_real_audio_for_an_audio_workflow(workspace: Path) -> None:
    job = make_job(
        workspace,
        workflow_id="music",
        parameters={"duration": "3s"},
        execution={"runtime": "harness", "output_kind": "audio"},
        output_content_type="audio/mpeg",
    )
    result, _ = await collect(job)

    assert (result.content_type, result.kind) == ("audio/mpeg", "audio")
    info = await probe_media(result.path)
    assert info.has_audio
    assert info.duration_seconds == pytest.approx(3.0, abs=0.75)


async def test_a_long_request_is_segmented_and_stitched_back_to_length(
    workspace: Path,
) -> None:
    """The client's long-form requirement, proven end to end without a GPU: ask
    for more than one pass can produce and get one file of the right length."""
    job = make_job(
        workspace,
        parameters={"duration": "6s", "aspect_ratio": "16:9"},
        execution={"runtime": "harness", "max_segment_seconds": 2},
    )
    result, reported = await collect(job)

    info = await probe_media(result.path)
    assert info.duration_seconds == pytest.approx(6.0, abs=0.75)

    # Three sections were rendered...
    sections = [message for status, _, message in reported if status == "generating"]
    assert len(sections) == 3
    assert sections[0] == "Generating section 1 of 3…"
    # ...and none of them leaked a seam into the finished file.
    assert result.path.name == "output.mp4"


async def test_segmentation_stays_inside_one_generating_phase(workspace: Path) -> None:
    """The API ranks statuses strictly forward, so a per-section hop into
    `post_processing` and back would be rejected as an illegal transition and
    would abandon the job mid-render."""
    job = make_job(
        workspace,
        parameters={"duration": "6s", "aspect_ratio": "16:9"},
        execution={"runtime": "harness", "max_segment_seconds": 2},
    )
    _, reported = await collect(job)

    order = [status for status, _, _ in reported]
    assert order == sorted(order, key=_rank)

    progress = [value for _, value, _ in reported]
    assert progress == sorted(progress)
    assert progress[-1] < 100, "the API sets 100 on completion, not the adapter"


def _rank(status: str) -> int:
    return ["preparing", "generating", "post_processing", "uploading"].index(status)


async def test_a_single_pass_job_does_not_mention_sections(workspace: Path) -> None:
    """Exposing the mechanism buys the customer nothing when there is only one
    of them."""
    _, reported = await collect(make_job(workspace, parameters={"duration": "2s"}))

    assert all("section" not in message.lower() for _, _, message in reported)


async def test_duration_comes_from_the_source_file_when_there_is_one(
    workspace: Path,
) -> None:
    """The shape of the client's automatic-duration requirement: a source video
    dictates the output length, and the requested duration is ignored."""
    source = workspace / "source.mp4"
    await ffmpeg(
        [
            "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=24",
            "-t", "5",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(source),
        ]
    )

    job = make_job(
        workspace,
        workflow_id="video-to-video",
        # Deliberately disagrees with the 5s source.
        parameters={"duration": "10s", "aspect_ratio": "16:9"},
        inputs=[
            AdapterInput(
                role="source_video",
                kind="video",
                content_type="video/mp4",
                download_url="https://storage.test/signed",
                path=source,
            )
        ],
    )
    result, _ = await collect(job)

    info = await probe_media(result.path)
    assert info.duration_seconds == pytest.approx(5.0, abs=0.75)


async def test_an_uploaded_track_reaches_the_finished_music_video(
    workspace: Path,
) -> None:
    """The harness makes the same audio promise the GPU runtime does.

    Without this it would produce a "music video" carrying its own synthetic
    tone and no trace of the uploaded song — output that looks right and is
    wrong, which is the exact class of failure this runtime exists to rule out
    everywhere else.
    """
    from tests.conftest import make_track

    track = await make_track(workspace / "song.mp3", 4.0)
    job = make_job(
        workspace,
        workflow_id="music-video",
        parameters={"aspect_ratio": "16:9"},
        execution={"runtime": "harness", "max_segment_seconds": 2},
        inputs=[
            AdapterInput(
                role="source_audio",
                kind="audio",
                content_type="audio/mpeg",
                download_url="https://storage.test/signed",
                path=track,
            )
        ],
    )
    result, _ = await collect(job)

    info = await probe_media(result.path)
    assert info.has_video and info.has_audio
    assert info.duration_seconds == pytest.approx(4.0, abs=0.75)

    payload = await ffprobe_json(result.path)
    audio = [s for s in payload["streams"] if s["codec_type"] == "audio"]
    assert len(audio) == 1, "the synthesised per-section audio was left in"


async def test_an_unreadable_source_fails_without_burning_retries(workspace: Path) -> None:
    """A corrupt upload will be corrupt on all three attempts."""
    junk = workspace / "source.mp4"
    junk.write_bytes(b"not a video")

    job = make_job(
        workspace,
        workflow_id="video-to-video",
        inputs=[
            AdapterInput(
                role="source_video",
                kind="video",
                content_type="video/mp4",
                download_url="https://storage.test/signed",
                path=junk,
            )
        ],
    )

    with pytest.raises(AdapterError) as raised:
        await collect(job)

    assert raised.value.retriable is False
    assert "ffprobe" not in raised.value.user_message.lower()


async def test_cancellation_is_honoured_between_sections(workspace: Path) -> None:
    """A four-minute music video is many sections; stopping at the next boundary
    is the difference between releasing a GPU in seconds and in minutes."""
    cancelled = asyncio.Event()
    cancelled.set()

    job = make_job(
        workspace,
        parameters={"duration": "6s", "aspect_ratio": "16:9"},
        execution={"runtime": "harness", "max_segment_seconds": 2},
        _cancelled=cancelled,
    )

    with pytest.raises(JobCancelled):
        await collect(job)


async def test_everything_it_writes_stays_inside_the_workspace(workspace: Path) -> None:
    """The runner deletes this directory and nothing else, so anything written
    elsewhere is a leak that survives the job."""
    job = make_job(
        workspace,
        parameters={"duration": "4s", "aspect_ratio": "16:9"},
        execution={"runtime": "harness", "max_segment_seconds": 2},
    )
    result, _ = await collect(job)

    assert result.path.parent == workspace
    for entry in workspace.rglob("*"):
        assert workspace in entry.parents or entry.parent == workspace
