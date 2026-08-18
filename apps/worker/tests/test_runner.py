"""The job runner: workspace, staging, lease keepalive, timeout, cleanup.

`runner.py` and the workspace helper had no test coverage at all before this
batch, which is uncomfortable given they are the files that decide whether a job
survives. Everything here is a behaviour that used to be absent or wrong and now
has to stay right.

The API is a fake: the runner's contract with it is six JSON calls, and standing
up FastAPI and PostgreSQL to observe them would test the API rather than this.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from worker.adapters.base import (
    AdapterError,
    AdapterJob,
    AdapterResult,
    ProgressCallback,
)
from worker.core.client import ApiUnavailable
from worker.core.config import settings
from worker.jobs import runner as runner_module
from worker.jobs.runner import JobRunner
from worker.jobs.workspace import job_workspace

WORKER_ID = "11111111-1111-1111-1111-111111111111"
JOB_ID = "22222222-2222-2222-2222-222222222222"


class FakeApi:
    """Records what the runner reported, and can refuse a lease on demand."""

    def __init__(self, *, reject_progress_after: int | None = None) -> None:
        self.progress: list[dict[str, Any]] = []
        self.completed: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []
        self._reject_after = reject_progress_after

    async def report_progress(self, job_id: str, **payload: Any) -> dict[str, Any]:
        self.progress.append(payload)
        if self._reject_after is not None and len(self.progress) > self._reject_after:
            return {"accepted": False, "reason": "lease expired"}
        return {"accepted": True}

    async def report_complete(self, job_id: str, **payload: Any) -> dict[str, Any]:
        self.completed.append(payload)
        return {"accepted": True}

    async def report_failure(self, job_id: str, **payload: Any) -> dict[str, Any]:
        self.failures.append(payload)
        return {"accepted": True}


def make_claim(**overrides: Any) -> dict[str, Any]:
    claim = {
        "job_id": JOB_ID,
        "workflow_id": "text-to-video",
        "workflow_version": "1",
        "prompt": "a cinematic drone shot",
        "parameters": {"duration": "10s", "aspect_ratio": "16:9"},
        "inputs": [],
        "execution": {"runtime": "stub"},
        "lease_token": "33333333-3333-3333-3333-333333333333",
        "attempt": 1,
        "max_attempts": 3,
        "output_upload_key": "users/u/generated/j/output.png",
        "output_upload_url": "https://storage.test/put",
        "output_content_type": "image/png",
    }
    claim.update(overrides)
    return claim


class StubAdapter:
    """Stands in for a provider. Records the job it was handed."""

    name = "stub"

    def __init__(self, behaviour=None) -> None:
        self.behaviour = behaviour
        self.seen: AdapterJob | None = None
        self.cleaned_up = False

    def supports(self, workflow_id: str) -> bool:
        return True

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        self.seen = job
        try:
            if self.behaviour is not None:
                await self.behaviour(job, on_progress)
            output = job.workspace / "output.png"
            output.write_bytes(b"rendered")
            return AdapterResult(path=output, content_type="image/png", kind="image")
        finally:
            self.cleaned_up = True


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keeps scratch directories inside the test's own tmp_path."""
    monkeypatch.setattr(settings, "workspace_dir", tmp_path / "workspaces")
    return tmp_path / "workspaces"


@pytest.fixture
def stub_adapter(monkeypatch: pytest.MonkeyPatch):
    """Routes `runtime: stub` at the resolver the runner actually calls."""

    def install(adapter: StubAdapter) -> StubAdapter:
        monkeypatch.setattr(runner_module, "resolve_adapter", lambda _job: adapter)
        return adapter

    return install


@pytest.fixture(autouse=True)
def no_real_uploads(monkeypatch: pytest.MonkeyPatch):
    uploaded: list[tuple[Path, str]] = []

    async def fake_upload(url: str, path: Path, content_type: str) -> int:
        uploaded.append((path, content_type))
        return path.stat().st_size

    monkeypatch.setattr(runner_module, "upload_output_file", fake_upload)
    return uploaded


# ── The happy path ───────────────────────────────────────────────────────


async def test_a_job_runs_uploads_and_completes(stub_adapter, no_real_uploads) -> None:
    api = FakeApi()
    stub_adapter(StubAdapter())

    await JobRunner(api, WORKER_ID).run(make_claim())

    assert len(api.completed) == 1
    assert api.failures == []
    # Size comes from the file on disk, not from a length the adapter asserted.
    assert api.completed[0]["size_bytes"] == len(b"rendered")
    assert no_real_uploads[0][1] == "image/png"


async def test_the_adapter_receives_a_real_writable_workspace(stub_adapter) -> None:
    api = FakeApi()
    adapter = stub_adapter(StubAdapter())

    await JobRunner(api, WORKER_ID).run(make_claim())

    assert adapter.seen is not None
    assert JOB_ID in adapter.seen.workspace.name


# ── Cleanup ──────────────────────────────────────────────────────────────


async def test_the_workspace_is_removed_after_success(stub_adapter, isolated_workspace) -> None:
    api = FakeApi()
    adapter = stub_adapter(StubAdapter())

    await JobRunner(api, WORKER_ID).run(make_claim())

    assert adapter.seen is not None
    assert not adapter.seen.workspace.exists()


async def test_the_workspace_is_removed_after_failure(stub_adapter) -> None:
    """The case that actually fills a disk: jobs that fail leave the most behind
    — a staged source video plus half the segments."""

    async def explode(job: AdapterJob, _on_progress) -> None:
        (job.workspace / "half-written.mp4").write_bytes(b"x" * 1024)
        raise AdapterError("nope", internal_detail="deliberate")

    api = FakeApi()
    adapter = stub_adapter(StubAdapter(behaviour=explode))

    await JobRunner(api, WORKER_ID).run(make_claim())

    assert len(api.failures) == 1
    assert adapter.seen is not None
    assert not adapter.seen.workspace.exists()


async def test_a_retry_does_not_inherit_the_previous_attempt_s_files(stub_adapter) -> None:
    """Attempts share a job id, so a stale file would be indistinguishable from
    one this attempt produced."""
    seen_contents: list[list[str]] = []

    async def record(job: AdapterJob, _on_progress) -> None:
        seen_contents.append(sorted(p.name for p in job.workspace.iterdir()))
        (job.workspace / "leftover.tmp").write_bytes(b"x")

    api = FakeApi()
    stub_adapter(StubAdapter(behaviour=record))

    runner = JobRunner(api, WORKER_ID)
    await runner.run(make_claim())
    await runner.run(make_claim(attempt=2))

    assert seen_contents == [[], []]


def test_a_job_starting_on_a_full_disk_fails_before_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running out of disk mid-render also breaks the *next* job, so the check
    belongs before the expensive part."""
    monkeypatch.setattr(settings, "min_free_disk_mb", 1024 * 1024 * 1024)

    with pytest.raises(AdapterError) as raised:
        with job_workspace("some-job"):
            pytest.fail("the workspace should not have opened")

    assert raised.value.retriable is True
    assert "MB free" in raised.value.internal_detail


# ── Input staging ────────────────────────────────────────────────────────


async def test_inputs_are_staged_to_disk_and_handed_to_the_adapter(
    stub_adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1 downloaded every input purely to check it was reachable and then threw
    the bytes away, so a real adapter would have fetched the same file twice."""
    downloads: list[str] = []

    async def fake_download(url: str, dest: Path, *, role: str) -> Path:
        downloads.append(role)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"staged bytes")
        return dest

    monkeypatch.setattr(runner_module, "download_input_to", fake_download)

    # Inspected from inside `run`, because by the time the runner returns the
    # workspace is gone — which is itself the behaviour two tests above assert.
    observed: dict[str, Any] = {}

    async def inspect(job: AdapterJob, _on_progress) -> None:
        source = job.input_for("source_video")
        reference = job.input_for("reference_image")
        assert source is not None and reference is not None
        observed["source_bytes"] = source.require_path().read_bytes()
        observed["source_suffix"] = source.require_path().suffix
        observed["reference_suffix"] = reference.require_path().suffix

    api = FakeApi()
    stub_adapter(StubAdapter(behaviour=inspect))
    claim = make_claim(
        workflow_id="video-to-video",
        inputs=[
            {
                "role": "source_video",
                "kind": "video",
                "content_type": "video/mp4",
                "download_url": "https://storage.test/a",
            },
            {
                "role": "reference_image",
                "kind": "image",
                "content_type": "image/png",
                "download_url": "https://storage.test/b",
            },
        ],
    )

    await JobRunner(api, WORKER_ID).run(claim)

    assert api.failures == []
    assert downloads == ["source_video", "reference_image"]
    # The adapter reads a local file — it does not re-fetch a URL the runner
    # has already downloaded once, which is what M1 forced it to do.
    assert observed["source_bytes"] == b"staged bytes"
    # Recognisable extensions, because ffmpeg and friends read them.
    assert observed["source_suffix"] == ".mp4"
    assert observed["reference_suffix"] == ".png"


async def test_an_unreachable_input_fails_before_the_adapter_runs(
    stub_adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovering a bad input after minutes of GPU time is the expensive
    ordering; discovering it in the first second is free."""

    async def fake_download(url: str, dest: Path, *, role: str) -> Path:
        raise AdapterError("One of the selected files could not be read.")

    monkeypatch.setattr(runner_module, "download_input_to", fake_download)

    api = FakeApi()
    adapter = stub_adapter(StubAdapter())

    await JobRunner(api, WORKER_ID).run(
        make_claim(
            inputs=[
                {
                    "role": "source_video",
                    "kind": "video",
                    "content_type": "video/mp4",
                    "download_url": "https://storage.test/a",
                }
            ]
        )
    )

    assert adapter.seen is None, "the adapter must not have been started"
    assert len(api.failures) == 1


# ── Lease keepalive ──────────────────────────────────────────────────────


async def test_a_silent_stage_longer_than_the_lease_keeps_the_job(
    stub_adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression that mattered most.

    Only a progress report renews a lease. A real render is silent for far
    longer than one, and M1 had nothing renewing it — so the reaper handed a
    healthy job to a second worker while the first was still rendering it.
    """
    monkeypatch.setattr(settings, "lease_keepalive_seconds", 0.05)

    async def work_silently(_job: AdapterJob, _on_progress) -> None:
        await asyncio.sleep(0.3)

    api = FakeApi()
    stub_adapter(StubAdapter(behaviour=work_silently))

    await JobRunner(api, WORKER_ID).run(make_claim())

    # "preparing" plus several renewals — the adapter itself reported nothing.
    assert len(api.progress) > 2
    assert len(api.completed) == 1


async def test_keepalive_repeats_progress_rather_than_inventing_it(
    stub_adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lease renewal is proof of life, not progress. Inflating the bar while
    nothing happens is how a stalled job looks healthy."""
    monkeypatch.setattr(settings, "lease_keepalive_seconds", 0.05)

    async def report_then_wait(_job: AdapterJob, on_progress) -> None:
        await on_progress("generating", 40, "Working…")
        await asyncio.sleep(0.25)

    api = FakeApi()
    stub_adapter(StubAdapter(behaviour=report_then_wait))

    await JobRunner(api, WORKER_ID).run(make_claim())

    renewals = [p for p in api.progress if p["progress"] == 40]
    assert len(renewals) > 1
    assert all(p["message"] == "Working…" for p in renewals)
    assert max(p["progress"] for p in api.progress) == 40


async def test_a_lost_lease_stops_the_adapter_and_reports_nothing(
    stub_adapter, monkeypatch: pytest.MonkeyPatch, no_real_uploads
) -> None:
    """When the platform takes a job away — a cancellation, a reassignment — the
    worker must release the GPU rather than finish a result nobody wants."""
    monkeypatch.setattr(settings, "lease_keepalive_seconds", 0.05)

    async def long_work(job: AdapterJob, _on_progress) -> None:
        for _ in range(60):
            job.raise_if_cancelled()
            await asyncio.sleep(0.02)

    api = FakeApi(reject_progress_after=1)
    adapter = stub_adapter(StubAdapter(behaviour=long_work))

    await JobRunner(api, WORKER_ID).run(make_claim())

    assert adapter.cleaned_up, "the adapter's finally block must have run"
    assert api.completed == []
    assert api.failures == [], "a job the platform reclaimed must not be re-reported"
    assert no_real_uploads == []


# ── Timeout ──────────────────────────────────────────────────────────────


async def test_an_overrunning_adapter_is_stopped_and_the_job_fails(
    stub_adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a budget a hung provider call holds a concurrency slot forever:
    the job never completes, never fails, and the slot never comes back."""
    monkeypatch.setattr(settings, "job_timeout_seconds", 0.1)

    async def hang(_job: AdapterJob, _on_progress) -> None:
        await asyncio.sleep(30)

    api = FakeApi()
    adapter = stub_adapter(StubAdapter(behaviour=hang))

    await asyncio.wait_for(JobRunner(api, WORKER_ID).run(make_claim()), timeout=5)

    assert len(api.failures) == 1
    assert adapter.cleaned_up
    assert "too long" in api.failures[0]["user_message"].lower()


async def test_a_workflow_may_shorten_its_own_budget(
    stub_adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`execution` is `extra="allow"`, so per-workflow tuning needs no schema
    change — which is how M2 will set real limits per model."""
    monkeypatch.setattr(settings, "job_timeout_seconds", 600)

    async def hang(_job: AdapterJob, _on_progress) -> None:
        await asyncio.sleep(30)

    api = FakeApi()
    stub_adapter(StubAdapter(behaviour=hang))

    claim = make_claim(execution={"runtime": "stub", "timeout_seconds": 0.1})
    await asyncio.wait_for(JobRunner(api, WORKER_ID).run(claim), timeout=5)

    assert len(api.failures) == 1


# ── Failure reporting ────────────────────────────────────────────────────


async def test_provider_detail_never_reaches_the_customer_message(stub_adapter) -> None:
    async def explode(_job: AdapterJob, _on_progress) -> None:
        raise AdapterError(
            "This generation could not be completed. Please try again.",
            internal_detail="torch.cuda.OutOfMemoryError on device 0",
        )

    api = FakeApi()
    stub_adapter(StubAdapter(behaviour=explode))

    await JobRunner(api, WORKER_ID).run(make_claim())

    failure = api.failures[0]
    assert "cuda" not in failure["user_message"].lower()
    assert "cuda" in failure["internal_detail"].lower()


async def test_an_adapter_crash_still_fails_the_job_cleanly(stub_adapter) -> None:
    """A bug in a provider adapter must not strand the job."""

    async def crash(_job: AdapterJob, _on_progress) -> None:
        raise ZeroDivisionError("adapter bug")

    api = FakeApi()
    stub_adapter(StubAdapter(behaviour=crash))

    await JobRunner(api, WORKER_ID).run(make_claim())

    assert len(api.failures) == 1
    assert api.failures[0]["retriable"] is True
    assert "ZeroDivisionError" in api.failures[0]["internal_detail"]


async def test_an_unreachable_api_leaves_the_job_to_the_reaper(stub_adapter) -> None:
    """Nothing can be reported when the only channel is down; the lease expiring
    is the recovery path, and inventing a local failure would be worse."""

    class DeadApi(FakeApi):
        async def report_complete(self, job_id: str, **payload: Any) -> dict[str, Any]:
            raise ApiUnavailable("connection refused")

    api = DeadApi()
    stub_adapter(StubAdapter())

    await JobRunner(api, WORKER_ID).run(make_claim())

    assert api.failures == []


async def test_a_dropped_progress_update_does_not_discard_a_healthy_job(
    stub_adapter,
) -> None:
    """THE BUG, 2026-08-17: six jobs died because a STATUS MESSAGE could not be
    delivered — one a video-to-video that had already rendered seven of its
    eight sections. A progress update is telemetry; the job is the work.

    The transport blip is transient by definition, so the run must carry on and
    still deliver. The lease keeper is separately renewing the lease, and a
    lease that is genuinely gone arrives as a REJECTION, which is what
    `test_a_refused_lease_stops_the_adapter` pins — a different path entirely.
    """

    class FlakyProgressApi(FakeApi):
        """Drops the first update, then behaves. That is the shape of the real
        fault: one dead pooled socket, everything afterwards fine."""

        def __init__(self) -> None:
            super().__init__()
            self.dropped = 0

        async def report_progress(self, job_id: str, **payload: Any) -> dict[str, Any]:
            if self.dropped == 0:
                self.dropped += 1
                raise ApiUnavailable("RemoteProtocolError calling /progress")
            return await super().report_progress(job_id, **payload)

    api = FlakyProgressApi()
    adapter = StubAdapter()
    stub_adapter(adapter)

    await JobRunner(api, WORKER_ID).run(make_claim())

    assert api.dropped > 0, "the test must actually exercise a dropped update"
    assert len(api.completed) == 1, "the job still finished and was reported"
    assert api.failures == []
