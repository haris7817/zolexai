"""The LTX 2.5 ComfyUI runtime: adapter flow, service, routing, H3 fencing.

No GPU and no ComfyUI: a fake service answers the exact HTTP conversation
the real one would — catalogue, upload, submit, poll, view, cancel — and
serves a real MP4 (ffmpeg-synthesised, one frame longer than the nominal
length, as the graphs render). Everything except the model is exercised:
compile, upload, submit, progress pacing, collection, validation, muting,
cancellation, failure classes, and the routing decisions around the hidden
H3 engine.

STATUS for the model itself: WAITING FOR GPU VALIDATION.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from tests.conftest import make_clip, needs_ffmpeg
from worker.adapters.base import AdapterError, AdapterInput, AdapterJob, JobCancelled
from worker.adapters.h3_comfy import H3ComfyAdapter
from worker.adapters.ltx_comfy import LtxComfyAdapter, PassSpec
from worker.adapters.registry import available_runtimes, get_adapter
from worker.comfy.client import ComfyClient, ComfyError, evict_comfy_vram
from worker.comfy.ltx_graphs import ASPECT_LABELS
from worker.comfy.ltx_prompts import TEXT_TO_VIDEO_NEGATIVE, negative_for
from worker.core.config import settings
from worker.providers.ltx_comfy import LtxComfyService
from worker.providers.router import get_provider
from worker.workflows.resolver import resolve_adapter

FPS = 24


class FakeLtxComfy:
    """A ComfyUI that speaks just enough HTTP for the adapter."""

    def __init__(self, output_file: Path, *, polls_before_done: int = 2) -> None:
        self.output_file = output_file
        self.polls_before_done = polls_before_done
        self.uploads: list[tuple[str, int]] = []
        self.submitted: dict | None = None
        self.polls = 0
        self.interrupts = 0
        self.queue_deletes: list[list[str]] = []
        self.frees = 0
        self.views: list[dict[str, str]] = []
        self.fail_history = False
        self.reject_submit: dict | None = None
        self.running: list[str] = []
        self.hang = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/system_stats":
            return httpx.Response(200, json={"devices": [{"vram_total": 96 * 2**30}]})
        if path == "/object_info":
            return httpx.Response(200, json=_catalogue())
        if path == "/upload/image":
            # Multipart: the filename rides in the content-disposition.
            body = request.content
            marker = b'filename="'
            start = body.index(marker) + len(marker)
            name = body[start : body.index(b'"', start)].decode()
            self.uploads.append((name, len(body)))
            return httpx.Response(200, json={"name": name, "subfolder": "", "type": "input"})
        if path == "/prompt":
            self.submitted = json.loads(request.content)
            if self.reject_submit is not None:
                return httpx.Response(400, json=self.reject_submit)
            self.running = ["p-1"]
            return httpx.Response(200, json={"prompt_id": "p-1", "number": 1, "node_errors": {}})
        if path.startswith("/history/"):
            self.polls += 1
            if self.hang or self.polls < self.polls_before_done:
                return httpx.Response(200, json={})
            self.running = []
            if self.fail_history:
                return httpx.Response(
                    200,
                    json={
                        "p-1": {
                            "status": {
                                "status_str": "error",
                                "messages": [["execution_error", {"exception_message": "boom"}]],
                            }
                        }
                    },
                )
            combine = next(
                nid
                for nid, e in self.submitted["prompt"].items()
                if e["class_type"] == "VHS_VideoCombine"
            )
            prefix = self.submitted["prompt"][combine]["inputs"]["filename_prefix"]
            subfolder, _, stem = prefix.rpartition("/")
            return httpx.Response(
                200,
                json={
                    "p-1": {
                        "status": {"status_str": "success", "messages": []},
                        "outputs": {
                            combine: {
                                "gifs": [
                                    {
                                        "filename": f"{stem}_00001.png",
                                        "subfolder": subfolder,
                                        "type": "output",
                                        "format": "image/png",
                                    },
                                    {
                                        "filename": f"{stem}_00001-audio.mp4",
                                        "subfolder": subfolder,
                                        "type": "output",
                                        "format": "video/h264-mp4",
                                        "frame_rate": FPS,
                                    },
                                ]
                            }
                        },
                    }
                },
            )
        if path == "/view":
            self.views.append(dict(request.url.params))
            return httpx.Response(200, content=self.output_file.read_bytes())
        if path == "/interrupt":
            self.interrupts += 1
            return httpx.Response(200, json={})
        if path == "/free":
            self.frees += 1
            return httpx.Response(200, json={})
        if path == "/queue":
            if request.method == "POST":
                body = json.loads(request.content)
                if "delete" in body:
                    self.queue_deletes.append(list(body["delete"]))
                return httpx.Response(200, json={})
            return httpx.Response(
                200,
                json={"queue_running": [[0, pid, {}] for pid in self.running], "queue_pending": []},
            )
        return httpx.Response(404)


def _catalogue() -> dict:
    """Only what the adapter asks a live server about."""
    return {
        "ResolutionSelector": {
            "input": {
                "required": {
                    "aspect_ratio": [list(ASPECT_LABELS.values())],
                    "megapixels": ["FLOAT"],
                    "multiple": ["INT"],
                }
            }
        }
    }


def _service(fake: FakeLtxComfy) -> LtxComfyService:
    client = ComfyClient(
        "http://ltx-comfy.test", poll_seconds=0.01, transport=httpx.MockTransport(fake.handler)
    )
    return LtxComfyService(client=client)


def _job(
    workspace: Path,
    workflow: str = "text-to-video",
    inputs: list[AdapterInput] | None = None,
    **params,
) -> AdapterJob:
    return AdapterJob(
        job_id="job-ltx-1",
        workflow_id=workflow,
        workflow_version="1",
        prompt="A koi pond at dawn, mist over the water.",
        parameters={"duration": "5s", "aspect_ratio": "16:9", **params},
        inputs=inputs or [],
        execution={"runtime": "ltx_comfy"},
        output_content_type="video/mp4",
        workspace=workspace,
    )


async def _rendered(path: Path, seconds: float) -> Path:
    """What the graph would write: fps·s+1 frames of video with audio."""
    return await make_clip(path, seconds + 1.0 / FPS, audio=True)


def _recorder():
    reports: list[tuple[str, int, str]] = []

    async def on_progress(status: str, progress: int, message: str, details=None) -> None:
        reports.append((status, progress, message))

    return on_progress, reports


# ── Text to Video, end to end against the fake ─────────────────────────────


@needs_ffmpeg
async def test_text_to_video_runs_the_client_graph_end_to_end(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 5.0))
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, reports = _recorder()

    result = await adapter.run(_job(workspace, seed=7), on_progress)

    assert result.kind == "video" and result.content_type == "video/mp4"
    assert result.path.parent == workspace and result.path.exists()
    assert result.duration_seconds and abs(result.duration_seconds - (5 + 1 / FPS)) < 0.2
    assert fake.uploads == []  # text-to-video stages nothing

    prompt = fake.submitted["prompt"]
    assert fake.submitted["client_id"] == "zolex-job-ltx-1"
    texts = {
        e["_meta"]["title"]: e["inputs"]["text"]
        for e in prompt.values()
        if e["class_type"] == "CLIPTextEncode"
    }
    assert texts["CLIP Text Encode (Prompt) positive"].startswith(
        "A koi pond at dawn, mist over the water."
    )
    assert texts["CLIP Text Encode (Prompt) negative"] == TEXT_TO_VIDEO_NEGATIVE
    [slider] = [e for e in prompt.values() if e["class_type"] == "mxSlider"]
    assert slider["inputs"]["Xi"] == 5
    [selector] = [e for e in prompt.values() if e["class_type"] == "ResolutionSelector"]
    assert selector["inputs"]["aspect_ratio"] == "16:9 (Widescreen)"
    [combine] = [e for e in prompt.values() if e["class_type"] == "VHS_VideoCombine"]
    assert combine["inputs"]["filename_prefix"] == "zolexai/job-ltx-1/output"
    for entry in prompt.values():
        if entry["class_type"] == "RandomNoise":
            assert entry["inputs"]["noise_seed"] == 7  # the customer's seed wins
    # The file came back through /view with the history's own coordinates.
    assert fake.views == [
        {"filename": "output_00001-audio.mp4", "subfolder": "zolexai/job-ltx-1", "type": "output"}
    ]
    # Progress moved forward through the house vocabulary and ended uploading.
    statuses = [status for status, _, _ in reports]
    assert statuses[0] == "preparing" and statuses[-1] == "uploading"
    progresses = [p for _, p, _ in reports]
    assert progresses == sorted(progresses)


@needs_ffmpeg
async def test_aspect_ratio_and_duration_reach_the_graph(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 10.0))
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    await adapter.run(_job(workspace, duration="10s", aspect_ratio="9:16"), on_progress)

    prompt = fake.submitted["prompt"]
    [slider] = [e for e in prompt.values() if e["class_type"] == "mxSlider"]
    assert slider["inputs"]["Xi"] == 10 and slider["inputs"]["Xf"] == 10.0
    [selector] = [e for e in prompt.values() if e["class_type"] == "ResolutionSelector"]
    assert selector["inputs"]["aspect_ratio"] == "9:16 (Portrait Widescreen)"


@needs_ffmpeg
async def test_seed_defaults_to_the_job_and_differs_between_jobs(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    render = await _rendered(tmp_path / "render.mp4", 5.0)
    seeds = []
    for job_id in ("job-a", "job-b"):
        fake = FakeLtxComfy(render)
        adapter = LtxComfyAdapter(service=_service(fake))
        job = AdapterJob(
            job_id=job_id,
            workflow_id="text-to-video",
            workflow_version="1",
            prompt="x",
            parameters={"duration": "5s", "aspect_ratio": "16:9"},
            execution={"runtime": "ltx_comfy"},
            output_content_type="video/mp4",
            workspace=workspace,
        )
        on_progress, _ = _recorder()
        await adapter.run(job, on_progress)
        seeds.append(
            [
                e["inputs"]["noise_seed"]
                for e in fake.submitted["prompt"].values()
                if e["class_type"] == "RandomNoise"
            ]
        )
    assert seeds[0] != seeds[1]
    assert LtxComfyAdapter.seed_base(_job(workspace)) == LtxComfyAdapter.seed_base(_job(workspace))


@needs_ffmpeg
async def test_sound_off_drops_the_audio_stream_after_verification(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 5.0))
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    result = await adapter.run(_job(workspace, sound=False), on_progress)

    from worker.media import probe_media

    info = await probe_media(result.path)
    assert info.has_video and not info.has_audio
    assert result.path.name.endswith("_muted.mp4")


# ── Failure classes ─────────────────────────────────────────────────────────


@needs_ffmpeg
async def test_a_rejected_submission_is_not_retried(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 5.0))
    fake.reject_submit = {
        "error": {"message": "Prompt outputs failed validation"},
        "node_errors": {"366": {}},
    }
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    with pytest.raises(AdapterError) as raised:
        await adapter.run(_job(workspace), on_progress)
    assert raised.value.retriable is False
    assert "validation" in raised.value.internal_detail.lower()
    assert "ComfyUI" not in raised.value.user_message


@needs_ffmpeg
async def test_a_failed_execution_is_retriable_and_says_why_internally(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 5.0))
    fake.fail_history = True
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    with pytest.raises(AdapterError) as raised:
        await adapter.run(_job(workspace), on_progress)
    assert raised.value.retriable is True
    assert "boom" in raised.value.internal_detail


@needs_ffmpeg
async def test_a_file_of_the_wrong_length_fails_its_check(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # The graph "rendered" 2 s for a 5 s request: the check must refuse it.
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 2.0))
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    with pytest.raises(AdapterError) as raised:
        await adapter.run(_job(workspace), on_progress)
    assert "differs from planned" in raised.value.internal_detail


async def test_lengths_beyond_one_pass_are_refused_with_the_way_out(tmp_path: Path) -> None:
    adapter = LtxComfyAdapter(service=_service(FakeLtxComfy(tmp_path / "none.mp4")))
    on_progress, _ = _recorder()
    with pytest.raises(AdapterError) as raised:
        await adapter.run(_job(tmp_path, duration="60s"), on_progress)
    assert raised.value.retriable is False
    assert "Extend" in raised.value.user_message


async def test_an_aspect_ratio_the_selector_lacks_is_refused(tmp_path: Path) -> None:
    adapter = LtxComfyAdapter(service=_service(FakeLtxComfy(tmp_path / "none.mp4")))
    on_progress, _ = _recorder()
    with pytest.raises(AdapterError) as raised:
        await adapter.run(_job(tmp_path, aspect_ratio="4:5"), on_progress)
    assert raised.value.retriable is False
    assert "16:9" in raised.value.user_message


async def test_workflows_this_runtime_does_not_serve_are_refused(tmp_path: Path) -> None:
    adapter = LtxComfyAdapter(service=_service(FakeLtxComfy(tmp_path / "none.mp4")))
    assert not adapter.supports("video-to-video")
    assert not adapter.supports("music-video")
    on_progress, _ = _recorder()
    with pytest.raises(AdapterError) as raised:
        await adapter.run(_job(tmp_path, workflow="video-to-video"), on_progress)
    assert raised.value.retriable is False


# ── Cancellation ────────────────────────────────────────────────────────────


@needs_ffmpeg
async def test_cancelling_mid_render_removes_the_prompt_and_interrupts(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 5.0))
    fake.hang = True
    adapter = LtxComfyAdapter(service=_service(fake))
    cancelled = asyncio.Event()
    job = AdapterJob(
        job_id="job-ltx-1",
        workflow_id="text-to-video",
        workflow_version="1",
        prompt="x",
        parameters={"duration": "5s", "aspect_ratio": "16:9"},
        execution={"runtime": "ltx_comfy"},
        output_content_type="video/mp4",
        workspace=workspace,
        _cancelled=cancelled,
    )
    on_progress, _ = _recorder()

    async def cancel_soon() -> None:
        await asyncio.sleep(0.05)
        cancelled.set()

    asyncio.create_task(cancel_soon())
    with pytest.raises(JobCancelled):
        await adapter.run(job, on_progress)

    assert fake.queue_deletes == [["p-1"]]
    assert fake.interrupts == 1  # it was the running prompt


@needs_ffmpeg
async def test_a_time_budget_expiry_also_cancels_the_prompt(tmp_path: Path) -> None:
    import time

    from worker.adapters.base import JobTimedOut

    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 5.0))
    fake.hang = True
    adapter = LtxComfyAdapter(service=_service(fake))
    job = AdapterJob(
        job_id="job-ltx-1",
        workflow_id="text-to-video",
        workflow_version="1",
        prompt="x",
        parameters={"duration": "5s", "aspect_ratio": "16:9"},
        execution={"runtime": "ltx_comfy"},
        output_content_type="video/mp4",
        workspace=workspace,
        _cancelled=asyncio.Event(),
        _deadline_monotonic=time.monotonic() + 0.1,
    )
    on_progress, _ = _recorder()
    with pytest.raises((JobTimedOut, AdapterError)):
        await adapter.run(job, on_progress)
    assert fake.queue_deletes == [["p-1"]]


# ── Service ─────────────────────────────────────────────────────────────────


async def test_health_reports_what_a_live_server_lacks(tmp_path: Path) -> None:
    fake = FakeLtxComfy(tmp_path / "none.mp4")
    ok, detail = await _service(fake).health()
    # The fake catalogue only knows ResolutionSelector: every other node class
    # is reported missing, which is exactly what a bare ComfyUI would show.
    assert ok is False
    assert "node class" in detail and "not installed" in detail
    assert detail.endswith("…")  # more than the twelve it lists


async def test_health_reports_an_unreachable_service() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = ComfyClient("http://ltx-comfy.test", transport=httpx.MockTransport(down))
    ok, detail = await LtxComfyService(client=client).health()
    assert ok is False and "unreachable" in detail


async def test_collect_picks_the_video_entry_and_downloads_it(tmp_path: Path) -> None:
    fake = FakeLtxComfy(tmp_path / "render.mp4")
    (tmp_path / "render.mp4").write_bytes(b"\x00" * 4096)
    service = _service(fake)
    history = {
        "outputs": {
            "188": {
                "gifs": [
                    {
                        "filename": "x_00001.png",
                        "subfolder": "zolexai/j",
                        "type": "output",
                        "format": "image/png",
                    },
                    {
                        "filename": "x_00001-audio.mp4",
                        "subfolder": "zolexai/j",
                        "type": "output",
                        "format": "video/h264-mp4",
                    },
                ]
            }
        }
    }
    dest = await service.collect(history, tmp_path / "out" / "output.mp4")
    assert dest.read_bytes() == b"\x00" * 4096
    assert fake.views[-1]["filename"] == "x_00001-audio.mp4"


async def test_collect_without_a_video_entry_is_an_error(tmp_path: Path) -> None:
    service = _service(FakeLtxComfy(tmp_path / "none.mp4"))
    with pytest.raises(ComfyError):
        await service.collect(
            {"outputs": {"1": {"images": [{"filename": "a.png"}]}}}, tmp_path / "o.mp4"
        )


async def test_upload_goes_through_the_upload_endpoint(tmp_path: Path) -> None:
    fake = FakeLtxComfy(tmp_path / "none.mp4")
    still = tmp_path / "still.png"
    still.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    name = await _service(fake).upload(still, name="zolex_j_first.png")
    assert name == "zolex_j_first.png"
    assert fake.uploads[0][0] == "zolex_j_first.png"


def test_referenced_models_cover_the_whole_pack() -> None:
    models = LtxComfyService().referenced_models()
    flat = {f for files in models.values() for f in files}
    assert "LTX-2.5-Distilled-Q8_0.gguf" in flat
    assert "LTXVideo/v2/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" in flat
    assert "LTX/LTX-2.5/LTX25_Ripple_v11.safetensors" in flat
    assert "ltx2.3-transition.safetensors" in flat


def test_negative_prompt_can_be_overridden_per_deployment() -> None:
    assert negative_for("text-to-video", {}) == TEXT_TO_VIDEO_NEGATIVE
    assert negative_for("text-to-video", {"negative_prompt": "  custom  "}) == "custom"


def test_positive_prompt_keeps_the_customers_words_first(tmp_path: Path) -> None:
    job = _job(tmp_path)
    text = LtxComfyAdapter.positive_prompt(job)
    assert text.startswith(job.prompt)
    structured = LtxComfyAdapter.positive_prompt(
        _job(
            tmp_path,
        )
        if False
        else AdapterJob(
            job_id="j",
            workflow_id="text-to-video",
            workflow_version="1",
            prompt=job.prompt,
            parameters=job.parameters,
            execution={"runtime": "ltx_comfy", "prompt_structuring": True},
            output_content_type="video/mp4",
            workspace=tmp_path,
        )
    )
    assert structured.startswith(job.prompt)
    assert len(structured) >= len(text)


# ── Routing and the hidden engine ───────────────────────────────────────────


def test_the_runtime_is_registered_and_nothing_committed_routes_to_it() -> None:
    assert "ltx_comfy" in available_runtimes()
    assert get_adapter("ltx_comfy").name == "ltx_comfy"
    definitions = Path(__file__).resolve().parents[3] / "workflow-definitions"
    for yaml_path in definitions.glob("*.yaml"):
        assert "runtime: ltx_comfy" not in yaml_path.read_text(encoding="utf-8"), yaml_path.name


def test_h3_is_hidden_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "enable_h3", False)
    adapter = H3ComfyAdapter()
    assert not adapter.supports("image-to-video")
    assert not adapter.supports("text-to-video")
    with pytest.raises(AdapterError) as raised:
        get_provider("h3")
    assert "ENABLE_H3" in raised.value.internal_detail
    monkeypatch.setattr(settings, "runtimes", "ltx,h3_comfy,music")
    assert "h3_comfy" not in settings.runtime_list
    monkeypatch.setattr(settings, "enable_h3", True)
    assert "h3_comfy" in settings.runtime_list
    assert adapter.supports("image-to-video")


async def test_a_hidden_h3_job_is_refused_not_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "enable_h3", False)
    on_progress, _ = _recorder()
    with pytest.raises(AdapterError) as raised:
        await H3ComfyAdapter().run(_job(tmp_path, workflow="image-to-video"), on_progress)
    assert raised.value.retriable is False
    assert "enable_h3=False" in raised.value.internal_detail


def test_a_stale_best_routing_falls_back_to_the_base_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 28 Aug overlay routed Text to Video Best to H3. With the engine
    hidden, such a claim is served by the base runtime and logged — the
    customer's job neither dies nor reaches H3."""
    monkeypatch.setattr(settings, "enable_h3", False)
    job = AdapterJob(
        job_id="j",
        workflow_id="text-to-video",
        workflow_version="1",
        prompt="x",
        parameters={"duration": "5s", "quality": "best"},
        execution={"runtime": "ltx", "runtime_by_quality": {"fast": "ltx", "best": "h3_comfy"}},
    )
    assert resolve_adapter(job).name == "ltx"


def test_the_new_runtime_resolves_from_the_execution_block() -> None:
    job = AdapterJob(
        job_id="j",
        workflow_id="text-to-video",
        workflow_version="1",
        prompt="x",
        parameters={"duration": "5s"},
        execution={"runtime": "ltx_comfy"},
    )
    assert resolve_adapter(job).name == "ltx_comfy"


def test_video_to_video_never_resolves_to_the_new_runtime() -> None:
    job = AdapterJob(
        job_id="j",
        workflow_id="video-to-video",
        workflow_version="1",
        prompt="x",
        parameters={},
        execution={"runtime": "ltx", "runtime_by_quality": {"best": "ltx_comfy"}},
    )
    # The safety net: ltx_comfy declines V2V, so Best stays on the CLI runtime.
    assert resolve_adapter(job).name == "ltx"


async def test_eviction_skips_the_callers_own_service_and_the_hidden_h3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        return httpx.Response(200, json={})

    monkeypatch.setattr(settings, "enable_h3", False)
    monkeypatch.setattr(settings, "ltx_comfy_base_url", "http://ltx.test")
    monkeypatch.setattr(settings, "h3_comfy_base_url", "http://h3.test")
    real_client = httpx.AsyncClient

    def fake_client(**kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real_client(**kw)

    monkeypatch.setattr("worker.comfy.client.httpx.AsyncClient", fake_client)
    await evict_comfy_vram(exclude="http://ltx.test")
    assert hits == []
    await evict_comfy_vram()
    assert hits == ["http://ltx.test/free"]
    monkeypatch.setattr(settings, "enable_h3", True)
    hits.clear()
    await evict_comfy_vram(exclude="http://ltx.test")
    assert hits == ["http://h3.test/free"]


def test_pass_spec_section_copy(tmp_path: Path) -> None:
    adapter = LtxComfyAdapter(service=_service(FakeLtxComfy(tmp_path / "none.mp4")))
    spec = PassSpec(
        seconds=5, positive="p", negative="n", aspect_label="16:9 (Widescreen)", seed_base=1
    )
    sectioned = adapter.with_section(spec, band=(15, 50), section=(1, 2, 0.0, 5.0))
    assert sectioned.band == (15, 50) and sectioned.section == (1, 2, 0.0, 5.0)
