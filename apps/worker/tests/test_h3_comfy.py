"""The H3 ComfyUI runtime: graph compilation, adapter flow, health.

The frozen client graphs in `benchmarks/client-pack/` are used directly — the
same files the GPU runs — so a graph edit that would break the conversion
breaks here first, without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tests.conftest import make_clip, needs_ffmpeg
from worker.adapters.base import AdapterError, AdapterInput, AdapterJob
from worker.adapters.h3_comfy import H3ComfyAdapter, h3_comfy_health
from worker.comfy import ComfyClient, ComfyError, GraphEdits, load_graph, to_api_prompt
from worker.comfy.graph import duration_index_for
from worker.core.config import settings

R2V_GRAPH = settings.h3_comfy_workflows_dir / "minimax_h3_r2v_extender.json"
I2V_GRAPH = settings.h3_comfy_workflows_dir / "minimax_h3_i2v_extender.json"
T2V_GRAPH = settings.h3_comfy_workflows_dir / "minimax_h3_t2v_extender.json"


# ── Graph compilation ───────────────────────────────────────────────────────


def test_frozen_graphs_are_present() -> None:
    assert R2V_GRAPH.is_file()
    assert I2V_GRAPH.is_file()
    assert T2V_GRAPH.is_file()


def test_duration_presets_map_exactly_and_nothing_else() -> None:
    assert duration_index_for(5.0) == 0
    assert duration_index_for(10.0) == 1
    assert duration_index_for(15.0) == 2
    assert duration_index_for(30.0) == 3
    assert duration_index_for(60.0) == 4
    assert duration_index_for(20.0) is None
    assert duration_index_for(7.0) is None


def test_r2v_conversion_applies_only_the_sanctioned_edits() -> None:
    graph = load_graph(R2V_GRAPH)
    api = to_api_prompt(
        graph,
        GraphEdits(
            duration_index=0,
            prompts={1: "STRUCTURED PROMPT ONE"},
            images={"REFERENCE IMAGE 1": "a.png", "REFERENCE IMAGE 2": "b.png"},
            drop_reference_3=True,
            width=960,
            height=544,
            filename_prefix="zolex_job1",
            output_directory="/tmp/ws",
        ),
    )
    # Duration index landed.
    assert api["2"]["inputs"]["value"] == 0
    # Prompt 1 replaced; prompt 2 untouched (still the pack's own text).
    assert api["10"]["inputs"]["value"] == "STRUCTURED PROMPT ONE"
    assert "Continue directly" in api["11"]["inputs"]["value"]
    # Images repointed, Picture 3 fully disconnected.
    assert api["31"]["inputs"]["image"] == "a.png"
    assert api["32"]["inputs"]["image"] == "b.png"
    assert "33" not in api
    assert "ref_3" not in api["34"]["inputs"]
    # Canvas and output routing.
    assert api["3"]["inputs"]["value"] == 960
    assert api["4"]["inputs"]["value"] == 544
    assert api["37"]["inputs"]["filename_prefix"] == "zolex_job1"
    assert api["37"]["inputs"]["output_directory"] == "/tmp/ws"
    # Linux path fix on the model loaders, per the guide's own note.
    assert api["27"]["inputs"]["unet_name"].startswith("H3/")
    assert api["28"]["inputs"]["clip_name"].startswith("H3/")


def test_t2v_conversion_needs_no_images_and_takes_the_canvas() -> None:
    """The 25 Aug client-test experiment: the T2V graph compiles with prompts
    and a canvas alone — no image loaders exist to feed."""
    graph = load_graph(T2V_GRAPH)
    api = to_api_prompt(
        graph,
        GraphEdits(
            duration_index=1,
            prompts={1: "STRUCTURED PROMPT ONE"},
            width=544,
            height=960,
            filename_prefix="zolex_job2",
            output_directory="/tmp/ws",
        ),
    )
    assert api["2"]["inputs"]["value"] == 1
    assert api["10"]["inputs"]["value"] == "STRUCTURED PROMPT ONE"
    # Portrait canvas landed on the width/height primitives.
    assert api["3"]["inputs"]["value"] == 544
    assert api["4"]["inputs"]["value"] == 960
    assert api["32"]["inputs"]["filename_prefix"] == "zolex_job2"
    assert api["32"]["inputs"]["output_directory"] == "/tmp/ws"
    # No LoadImage node anywhere in the compiled prompt.
    assert not any(entry["class_type"] == "LoadImage" for entry in api.values())


def test_conversion_never_touches_sampling() -> None:
    """Without an explicit steps edit, sampling stays exactly the pack's."""
    graph = load_graph(R2V_GRAPH)
    api = to_api_prompt(graph, GraphEdits(duration_index=4))
    extender = api["36"]["inputs"]
    assert extender["steps"] == 20
    assert extender["sampler_name"] == "res_multistep"
    assert extender["scheduler"] == "beta"
    assert extender["context_length"] == "22"


def test_steps_override_reaches_every_graph_at_the_api_layer() -> None:
    """The 26 Aug speed dial. On R2V it must override the widget-mapped 20;
    on T2V — whose list-form widgets never reach the submission at all — it
    must still land, because the API inputs are the only reliable layer.
    Sampler and scheduler stay pinned either way."""
    r2v = to_api_prompt(load_graph(R2V_GRAPH), GraphEdits(duration_index=0, steps=12))
    assert r2v["36"]["inputs"]["steps"] == 12
    assert r2v["36"]["inputs"]["sampler_name"] == "res_multistep"

    t2v = to_api_prompt(load_graph(T2V_GRAPH), GraphEdits(duration_index=0, steps=12))
    extenders = [
        e for e in t2v.values() if e["class_type"] == "MiniMaxH3Extender"
    ]
    assert extenders and all(e["inputs"]["steps"] == 12 for e in extenders)

    # I2V spells its schedule differently: one BasicScheduler node.
    i2v = to_api_prompt(load_graph(I2V_GRAPH), GraphEdits(duration_index=0, steps=12))
    schedulers = [
        e for e in i2v.values() if e["class_type"] == "BasicScheduler"
    ]
    assert schedulers and all(e["inputs"]["steps"] == 12 for e in schedulers)


def test_i2v_conversion_reaches_the_source_image_loader() -> None:
    graph = load_graph(I2V_GRAPH)
    api = to_api_prompt(
        graph,
        GraphEdits(duration_index=0, images={"I2V SOURCE IMAGE": "src.png"}),
    )
    assert api["16"]["inputs"]["image"] == "src.png"
    # The I2V canvas primitives stay at the proven 1280x736.
    assert api["4"]["inputs"]["value"] == 1280
    assert api["5"]["inputs"]["value"] == 736


def test_linked_inputs_become_node_references() -> None:
    graph = load_graph(R2V_GRAPH)
    api = to_api_prompt(graph, GraphEdits(duration_index=0))
    # The Extender's model input is a reference to the UNETLoader.
    assert api["36"]["inputs"]["model"] == ["27", 0]


# ── Adapter flow against a fake ComfyUI ─────────────────────────────────────


class FakeComfy:
    """The minimum of ComfyUI's HTTP surface the adapter touches."""

    def __init__(self, workspace: Path, job_id: str, *, fail_submit: bool = False) -> None:
        self.workspace = workspace
        self.job_id = job_id
        self.fail_submit = fail_submit
        self.submitted: dict | None = None
        self.polls = 0
        self.interrupts = 0
        self.queue_deletes: list[list[str]] = []
        #: what GET /queue reports as currently executing (prompt ids).
        self.running: list[str] = []
        #: when True, /history never reports completion (for timeout tests).
        self.never_finish = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/prompt":
            if self.fail_submit:
                return httpx.Response(400, json={"error": "bad prompt"})
            self.submitted = json.loads(request.content)
            return httpx.Response(200, json={"prompt_id": "p-1", "node_errors": {}})
        if request.url.path == "/history/p-1":
            self.polls += 1
            if self.never_finish or self.polls < 2:
                return httpx.Response(200, json={})
            return httpx.Response(
                200,
                json={"p-1": {"status": {"status_str": "success", "messages": []}}},
            )
        if request.url.path == "/interrupt":
            self.interrupts += 1
            return httpx.Response(200, json={})
        if request.url.path == "/queue":
            if request.method == "POST":
                body = json.loads(request.content)
                if "delete" in body:
                    self.queue_deletes.append(list(body["delete"]))
                return httpx.Response(200, json={})
            return httpx.Response(
                200,
                json={
                    "queue_running": [[0, pid, {}] for pid in self.running],
                    "queue_pending": [],
                },
            )
        return httpx.Response(404)


def _client(fake: FakeComfy) -> ComfyClient:
    return ComfyClient(
        "http://comfy.test",
        poll_seconds=0.01,
        transport=httpx.MockTransport(fake.handler),
    )


async def _progress(status: str, progress: int, message: str, details=None) -> None:
    pass


async def _png(path: Path) -> Path:
    from worker.media.ffmpeg import ffmpeg

    await ffmpeg(
        ["-f", "lavfi", "-i", "color=c=red:s=64x64:d=0.1", "-frames:v", "1", str(path), "-y"]
    )
    return path


def _job(workspace: Path, workflow: str, inputs: list[AdapterInput], **params) -> AdapterJob:
    return AdapterJob(
        job_id="job1",
        workflow_id=workflow,
        workflow_version="1",
        prompt="A man in a navy coat sings at a microphone.",
        parameters={"duration": "5s", **params},
        inputs=inputs,
        execution={"runtime": "h3_comfy"},
        output_content_type="video/mp4",
        workspace=workspace,
    )


def _input(role: str, path: Path, kind: str = "image") -> AdapterInput:
    return AdapterInput(
        role=role,
        kind=kind,
        content_type="image/png" if kind == "image" else "video/mp4",
        download_url="http://unused",
        path=path,
    )


@needs_ffmpeg
async def test_i2v_run_end_to_end_against_fake_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(settings, "h3_comfy_input_dir", tmp_path / "comfy_in")

    source = await _png(tmp_path / "face.png")
    # The Final Decode writes into the workspace; the fake stands in for it.
    await make_clip(workspace / "zolex_job1_00001.mp4", 5.0)

    fake = FakeComfy(workspace, "job1")
    adapter = H3ComfyAdapter(client=_client(fake))
    result = await adapter.run(
        _job(workspace, "image-to-video", [_input("source_image", source)]),
        _progress,
    )

    assert result.kind == "video"
    assert result.duration_seconds and abs(result.duration_seconds - 5.0) < 1.5
    # The submission carried the frozen I2V graph with our edits: the staged
    # image name, the job-scoped output routing, and a disciplined prompt.
    prompt = fake.submitted["prompt"]
    assert prompt["16"]["inputs"]["image"].startswith("zolex_job1_")
    assert prompt["79"]["inputs"]["output_directory"] == str(workspace)
    assert "navy coat" in prompt["11"]["inputs"]["value"]
    # Staged copies are cleaned up.
    assert not list((tmp_path / "comfy_in").glob("zolex_job1_*.png"))


@needs_ffmpeg
async def test_reference_v2v_maps_identity_and_source_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(settings, "h3_comfy_input_dir", tmp_path / "comfy_in")

    reference = await _png(tmp_path / "person.png")
    source = await make_clip(tmp_path / "perf.mp4", 1.0)
    await make_clip(workspace / "zolex_job1_00001.mp4", 5.0)

    fake = FakeComfy(workspace, "job1")
    adapter = H3ComfyAdapter(client=_client(fake))
    result = await adapter.run(
        _job(
            workspace,
            "video-to-video",
            [
                _input("reference_image", reference),
                _input("source_video", source, kind="video"),
            ],
        ),
        _progress,
    )
    assert result.kind == "video"
    prompt = fake.submitted["prompt"]
    # Identity → Picture 1, source first frame → Picture 2, Picture 3 gone —
    # the proven D1 mapping.
    assert prompt["31"]["inputs"]["image"].endswith("_ref1.png")
    assert prompt["32"]["inputs"]["image"].endswith("_ref2.png")
    assert "33" not in prompt
    # Quality tier by default → the measured delivery canvas.
    assert prompt["3"]["inputs"]["value"] == 960
    assert prompt["4"]["inputs"]["value"] == 544


@needs_ffmpeg
async def test_draft_tier_selects_the_pack_canvas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(settings, "h3_comfy_input_dir", tmp_path / "comfy_in")
    reference = await _png(tmp_path / "person.png")
    await make_clip(workspace / "zolex_job1_00001.mp4", 5.0)

    fake = FakeComfy(workspace, "job1")
    adapter = H3ComfyAdapter(client=_client(fake))
    job = AdapterJob(
        job_id="job1",
        workflow_id="video-to-video",
        workflow_version="1",
        prompt="p",
        parameters={"duration": "5s"},
        inputs=[_input("reference_image", reference)],
        execution={"runtime": "h3_comfy", "h3_tier": "draft"},
        workspace=workspace,
    )
    await adapter.run(job, _progress)
    prompt = fake.submitted["prompt"]
    assert prompt["3"]["inputs"]["value"] == 544
    assert prompt["4"]["inputs"]["value"] == 320


@needs_ffmpeg
async def test_h3_steps_flows_into_the_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`execution.h3_steps: 12` — the user's 26 Aug speed decision — must be
    what the server is actually asked to run."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(settings, "h3_comfy_input_dir", tmp_path / "comfy_in")
    source = await _png(tmp_path / "face.png")
    await make_clip(workspace / "zolex_job1_00001.mp4", 5.0)

    fake = FakeComfy(workspace, "job1")
    adapter = H3ComfyAdapter(client=_client(fake))
    job = AdapterJob(
        job_id="job1",
        workflow_id="image-to-video",
        workflow_version="1",
        prompt="p",
        parameters={"duration": "5s"},
        inputs=[_input("source_image", source)],
        execution={"runtime": "h3_comfy", "h3_steps": 12},
        workspace=workspace,
    )
    await adapter.run(job, _progress)
    prompt = fake.submitted["prompt"]
    carriers = [
        e
        for e in prompt.values()
        if e["class_type"] in ("MiniMaxH3Extender", "BasicScheduler")
    ]
    assert carriers and all(e["inputs"]["steps"] == 12 for e in carriers)


async def test_h3_steps_refuses_configuration_typos(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    adapter = H3ComfyAdapter(client=_client(FakeComfy(workspace, "job1")))
    job = AdapterJob(
        job_id="job1",
        workflow_id="text-to-video",
        workflow_version="1",
        prompt="p",
        parameters={"duration": "5s"},
        inputs=[],
        execution={"runtime": "h3_comfy", "h3_steps": "fast"},
        workspace=workspace,
    )
    with pytest.raises(AdapterError) as raised:
        await adapter.run(job, _progress)
    assert raised.value.retriable is False
    assert "h3_steps" in raised.value.internal_detail


async def test_reference_v2v_without_reference_is_refused(tmp_path: Path) -> None:
    adapter = H3ComfyAdapter(client=_client(FakeComfy(tmp_path, "job1")))
    with pytest.raises(AdapterError) as raised:
        await adapter.run(_job(tmp_path, "video-to-video", []), _progress)
    assert "reference photo" in raised.value.user_message
    assert raised.value.retriable is False


async def test_off_lattice_duration_renders_the_next_preset_and_trims(
    tmp_path: Path,
) -> None:
    """The client sells 20s; the pack's lattice has no such plan. The render
    is the next preset up (30s, index 3) and the delivery is an exact cut —
    the 27 Aug round-two decision."""
    job = _job(tmp_path, "image-to-video", [], duration="20s")
    index, trim_to = await H3ComfyAdapter._duration_index(job)
    assert (index, trim_to) == (3, 20.0)

    # Exact lattice lengths still carry no trim.
    exact = _job(tmp_path, "image-to-video", [], duration="15s")
    assert await H3ComfyAdapter._duration_index(exact) == (2, None)


async def test_a_length_beyond_the_ladder_is_refused_with_the_menu(
    tmp_path: Path,
) -> None:
    adapter = H3ComfyAdapter(client=_client(FakeComfy(tmp_path, "job1")))
    job = _job(tmp_path, "image-to-video", [], duration="90s")
    with pytest.raises(AdapterError) as raised:
        await adapter.run(job, _progress)
    assert "5s" in raised.value.user_message
    assert "60s" in raised.value.user_message


@needs_ffmpeg
async def test_submit_failure_is_structured_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(settings, "h3_comfy_input_dir", tmp_path / "comfy_in")
    source = await _png(tmp_path / "face.png")

    fake = FakeComfy(workspace, "job1", fail_submit=True)
    adapter = H3ComfyAdapter(client=_client(fake))
    with pytest.raises(AdapterError) as raised:
        await adapter.run(
            _job(workspace, "image-to-video", [_input("source_image", source)]),
            _progress,
        )
    assert raised.value.user_message == "This request could not be started."
    assert raised.value.retriable is False


async def test_timeout_deletes_a_pending_prompt_and_never_blind_interrupts(
    tmp_path: Path,
) -> None:
    """The 25 Aug production orphan: a budget-expired prompt still PENDING in
    the queue survived /interrupt and rendered for twenty minutes as an
    orphan while its own retry queued behind it. Timeout must delete the
    prompt from the queue — and must NOT send /interrupt when the prompt is
    not the one executing, because that kills an innocent neighbour."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeComfy(workspace, "job1")
    fake.never_finish = True
    fake.running = ["someone-elses-prompt"]
    client = _client(fake)
    with pytest.raises(ComfyError) as raised:
        await client.wait(
            _job(workspace, "image-to-video", []), "p-1", timeout_seconds=0.05
        )
    assert "took too long" in raised.value.user_message
    assert fake.queue_deletes == [["p-1"]]
    assert fake.interrupts == 0


async def test_timeout_interrupts_the_prompt_only_when_it_is_running(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeComfy(workspace, "job1")
    fake.never_finish = True
    fake.running = ["p-1"]
    client = _client(fake)
    with pytest.raises(ComfyError):
        await client.wait(
            _job(workspace, "image-to-video", []), "p-1", timeout_seconds=0.05
        )
    assert fake.queue_deletes == [["p-1"]]
    assert fake.interrupts == 1


def test_supports_exactly_the_approved_workflows() -> None:
    adapter = H3ComfyAdapter()
    assert adapter.supports("image-to-video")
    assert adapter.supports("video-to-video")
    # 25 Aug client-test experiment: T2V is reachable; routing YAML decides.
    assert adapter.supports("text-to-video")
    assert not adapter.supports("music-video")
    assert not adapter.supports("extend-video")


# ── Health ──────────────────────────────────────────────────────────────────


async def test_health_reports_unreachable_service(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = ComfyClient("http://comfy.test", transport=httpx.MockTransport(down))
    ok, detail = await h3_comfy_health(client)
    assert not ok
    assert "unreachable" in detail


async def test_health_names_every_missing_piece(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def up(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/system_stats":
            return httpx.Response(
                200, json={"devices": [{"vram_total": 90 * 2**30}]}
            )
        if request.url.path == "/object_info":
            return httpx.Response(200, json={"MiniMaxH3Extender": {}})
        return httpx.Response(404)

    monkeypatch.setattr(settings, "h3_comfy_models_dir", tmp_path / "models")
    client = ComfyClient("http://comfy.test", transport=httpx.MockTransport(up))
    ok, detail = await h3_comfy_health(client)
    assert not ok
    # Missing node classes and missing weights are both named, not summarised.
    assert "missing node classes" in detail
    assert "weight missing" in detail


def test_audio_context_stays_pack_default_unless_explicitly_set() -> None:
    """The pack ships audio_context_length=0; None must not touch it, and an
    explicit value must reach both the Extender and every MotionContextRAM."""
    graph = load_graph(R2V_GRAPH)
    untouched = to_api_prompt(graph, GraphEdits(duration_index=4))
    assert untouched["36"]["inputs"]["audio_context_length"] == 0

    changed = to_api_prompt(
        graph, GraphEdits(duration_index=4, audio_context_length=22)
    )
    assert changed["36"]["inputs"]["audio_context_length"] == 22

    i2v = load_graph(I2V_GRAPH)
    changed_i2v = to_api_prompt(
        i2v, GraphEdits(duration_index=4, audio_context_length=22)
    )
    ram_nodes = [
        entry
        for entry in changed_i2v.values()
        if entry["class_type"] == "MiniMaxH3MotionContextRAM"
    ]
    assert ram_nodes
    assert all(e["inputs"]["audio_context_length"] == 22 for e in ram_nodes)


@needs_ffmpeg
async def test_v2v_without_duration_follows_the_source_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`duration_mode: source` sends no duration; the nearest preset to the
    source clip's own length is the honest reading of that request."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(settings, "h3_comfy_input_dir", tmp_path / "comfy_in")
    reference = await _png(tmp_path / "person.png")
    source = await make_clip(tmp_path / "perf.mp4", 4.0)  # nearest preset: 5 s
    await make_clip(workspace / "zolex_job1_00001.mp4", 5.0)

    fake = FakeComfy(workspace, "job1")
    adapter = H3ComfyAdapter(client=_client(fake))
    job = AdapterJob(
        job_id="job1",
        workflow_id="video-to-video",
        workflow_version="1",
        prompt="p",
        parameters={},  # no duration, as the source-mode API sends
        inputs=[
            _input("reference_image", reference),
            _input("source_video", source, kind="video"),
        ],
        execution={"runtime": "h3_comfy"},
        workspace=workspace,
    )
    await adapter.run(job, _progress)
    assert fake.submitted["prompt"]["2"]["inputs"]["value"] == 0  # 5 s preset


@needs_ffmpeg
async def test_vram_is_freed_after_a_job_on_cotenanted_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After an H3 job the adapter unloads ComfyUI's models, because 52 GB of
    idle residency plus ACE-Step plus an LTX pass does not fit one card."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(settings, "h3_comfy_input_dir", tmp_path / "comfy_in")
    source = await _png(tmp_path / "face.png")
    await make_clip(workspace / "zolex_job1_00001.mp4", 5.0)

    freed = []

    class FreeTrackingFake(FakeComfy):
        def handler(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/free":
                freed.append(json.loads(request.content))
                return httpx.Response(200, json={})
            return super().handler(request)

    fake = FreeTrackingFake(workspace, "job1")
    adapter = H3ComfyAdapter(client=_client(fake))

    # The lazy default (25 Aug): H3 keeps its model warm between H3 jobs —
    # the LTX/music adapters evict it on their way in instead.
    await adapter.run(
        _job(workspace, "image-to-video", [_input("source_image", source)]),
        _progress,
    )
    assert not freed

    # Eager freeing remains available for nodes where back-to-back H3 is rare.
    monkeypatch.setattr(settings, "h3_comfy_free_after_job", True)
    await make_clip(workspace / "zolex_job1_00002.mp4", 5.0)
    await adapter.run(
        _job(workspace, "image-to-video", [_input("source_image", source)]),
        _progress,
    )
    assert freed and freed[0]["unload_models"] is True


async def test_other_engines_evict_comfy_on_their_way_in(tmp_path: Path) -> None:
    """The other half of the lazy policy: `evict_comfy_vram` is what LTX and
    music call before taking the card, and it asks ComfyUI to unload."""
    from worker.comfy import evict_comfy_vram

    workspace = tmp_path / "ws"
    workspace.mkdir()
    freed = []

    class FreeTrackingFake(FakeComfy):
        def handler(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/free":
                freed.append(json.loads(request.content))
                return httpx.Response(200, json={})
            return super().handler(request)

    fake = FreeTrackingFake(workspace, "job1")
    await evict_comfy_vram(_client(fake))
    assert freed and freed[0]["unload_models"] is True


def test_seed_base_makes_each_job_its_own_video() -> None:
    """The 26 Aug cache discovery: fixed seeds + deterministic model +
    ComfyUI's cache meant 'regenerate' returned the byte-identical file in
    35 s forever. A seed base shifts every segment seed while keeping the
    pack's per-segment distinctness; no base keeps the pack untouched."""
    graph = load_graph(T2V_GRAPH)

    stock = to_api_prompt(graph, GraphEdits(duration_index=4))
    stock_seeds = sorted(
        e["inputs"]["value"]
        for e in stock.values()
        if e["class_type"] == "PrimitiveInt"
        and e["inputs"]["value"] > 1_000_000  # the seed primitives
    )
    assert stock_seeds == [731003101, 731003102, 731003103, 731003104, 731003105]

    seeded = to_api_prompt(graph, GraphEdits(duration_index=4, seed_base=5_000_000))
    new_seeds = sorted(
        e["inputs"]["value"]
        for e in seeded.values()
        if e["class_type"] == "PrimitiveInt" and e["inputs"]["value"] >= 5_000_000
    )
    assert new_seeds == [5_000_101, 5_000_102, 5_000_103, 5_000_104, 5_000_105]
    assert len(set(new_seeds)) == 5  # per-segment distinctness survives

    # I2V: literal RandomNoise seeds shift; linked ones (fed by the shifted
    # SEGMENT SEED primitives) stay as links and inherit through the graph.
    i2v = to_api_prompt(
        load_graph(I2V_GRAPH), GraphEdits(duration_index=0, seed_base=5_000_000)
    )
    literal_noise = [
        e["inputs"]["noise_seed"]
        for e in i2v.values()
        if e["class_type"] == "RandomNoise"
        and isinstance(e["inputs"].get("noise_seed"), int)
    ]
    assert all(5_000_000 <= s < 5_001_000 for s in literal_noise)
    i2v_primitive_seeds = [
        e["inputs"]["value"]
        for e in i2v.values()
        if e["class_type"] == "PrimitiveInt"
        and isinstance(e["inputs"].get("value"), int)
        and e["inputs"]["value"] >= 5_000_000
    ]
    assert len(i2v_primitive_seeds) == 5
    assert len(set(i2v_primitive_seeds)) == 5


def test_seed_base_is_stable_per_job_and_unique_across_jobs(tmp_path: Path) -> None:
    job_a = _job(tmp_path, "text-to-video", [])
    same_a = _job(tmp_path, "text-to-video", [])
    assert H3ComfyAdapter._seed_base(job_a) == H3ComfyAdapter._seed_base(same_a)

    import dataclasses

    job_b = dataclasses.replace(job_a, job_id="another-job")
    assert H3ComfyAdapter._seed_base(job_a) != H3ComfyAdapter._seed_base(job_b)

    # An explicit seed parameter wins.
    job_c = _job(tmp_path, "text-to-video", [], seed=1234)
    assert H3ComfyAdapter._seed_base(job_c) == 1234


async def test_best_tier_does_not_offer_sixty_seconds(tmp_path: Path) -> None:
    """The Fast/Best product decision: Best caps at 30s. The refusal lists
    only what is actually on offer — no 60s in the menu."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    adapter = H3ComfyAdapter(client=_client(FakeComfy(workspace, "job1")))
    job = AdapterJob(
        job_id="job1",
        workflow_id="text-to-video",
        workflow_version="1",
        prompt="p",
        parameters={"duration": "60s"},
        inputs=[],
        execution={"runtime": "h3_comfy", "h3_max_seconds": 30},
        workspace=workspace,
    )
    with pytest.raises(AdapterError) as raised:
        await adapter.run(job, _progress)
    assert "60s" not in raised.value.user_message
    assert "30s" in raised.value.user_message


@needs_ffmpeg
async def test_sound_off_strips_the_audio_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`sound: false` — the customer asked for a silent video; the stream is
    dropped, the picture copied untouched."""
    from worker.media.probe import probe_media

    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(settings, "h3_comfy_input_dir", tmp_path / "comfy_in")
    source = await _png(tmp_path / "face.png")
    await make_clip(workspace / "zolex_job1_00001.mp4", 5.0, audio=True)

    fake = FakeComfy(workspace, "job1")
    adapter = H3ComfyAdapter(client=_client(fake))
    job = AdapterJob(
        job_id="job1",
        workflow_id="image-to-video",
        workflow_version="1",
        prompt="p",
        parameters={"duration": "5s", "sound": "false"},
        inputs=[_input("source_image", source)],
        execution={"runtime": "h3_comfy"},
        workspace=workspace,
    )
    result = await adapter.run(job, _progress)
    info = await probe_media(result.path)
    assert info.has_video and not info.has_audio


def test_context_length_is_left_alone_unless_asked_for() -> None:
    """The pack's 22-frame handoff window is an invariant until measured."""
    graph = load_graph(T2V_GRAPH)
    api = to_api_prompt(graph, GraphEdits(duration_index=3))
    extenders = [e for e in api.values() if e["class_type"] == "MiniMaxH3Extender"]
    assert extenders
    for entry in extenders:
        assert entry["inputs"]["context_length"] == "22"


def test_a_wider_handoff_window_reaches_the_extender() -> None:
    graph = load_graph(T2V_GRAPH)
    api = to_api_prompt(graph, GraphEdits(duration_index=3, context_length=48))
    for entry in api.values():
        if entry["class_type"] == "MiniMaxH3Extender":
            assert entry["inputs"]["context_length"] == "48"
