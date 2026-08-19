"""Shared scaffolding for the worker suite.

The GPU-backed runtimes have exactly one seam between this codebase and a
model: the argv prefix that launches it. Substituting a plain Python script
there — one that records the command it was given and copies a real MP4 into
place — exercises every line of the platform except the model itself: command
construction, subprocess supervision, conditioning, chaining, cancellation,
assembly, muxing, validation.

`render_stub` is that substitution, and it records the FULL argv of every
invocation as JSON. That is what lets a test assert something as specific as
"the third pass was conditioned on stills lifted from the third window of the
source, at the configured strength, and on the second pass's final frame at
frame zero" without any hardware.

Anything touching real media needs ffmpeg and skips without it; the worker
image ships it, and so does the GPU node, which is where those assertions
actually count.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from worker.adapters.base import AdapterInput, AdapterJob, ProgressCallback
from worker.adapters.ltx import LtxAdapter
from worker.core.config import settings
from worker.media import ffmpeg, tools_available

needs_ffmpeg = pytest.mark.skipif(
    not tools_available(), reason="ffmpeg/ffprobe not installed"
)


# ── Jobs ─────────────────────────────────────────────────────────────────


def make_job(workspace: Path, **overrides) -> AdapterJob:
    defaults = dict(
        job_id="00000000-0000-0000-0000-0000000000ff",
        workflow_id="text-to-video",
        workflow_version="1",
        prompt="a cinematic drone shot over a coastline",
        parameters={"duration": "2s", "aspect_ratio": "16:9", "quality": "High"},
        inputs=[],
        execution={"runtime": "ltx"},
        output_content_type="video/mp4",
        workspace=workspace,
    )
    return AdapterJob(**{**defaults, **overrides})


def staged_input(role: str, kind: str, content_type: str, path: Path | None) -> AdapterInput:
    return AdapterInput(
        role=role,
        kind=kind,
        content_type=content_type,
        download_url="https://storage.test/signed",
        path=path,
    )


async def collect(job: AdapterJob, adapter=None):
    """Runs an adapter and returns (result, every progress report it made)."""
    reported: list[tuple[str, int, str]] = []

    async def on_progress(
        status: str, progress: int, message: str, _details=None
    ) -> None:
        reported.append((status, progress, message))

    result = await (adapter or LtxAdapter()).run(job, on_progress)
    return result, reported


def recorder() -> tuple[ProgressCallback, list[tuple[str, int, str]]]:
    """A progress callback plus the list it appends to."""
    reported: list[tuple[str, int, str]] = []

    async def on_progress(
        status: str, progress: int, message: str, _details=None
    ) -> None:
        reported.append((status, progress, message))

    return on_progress, reported


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_hosted_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches a hosted service, whatever the developer's .env holds.

    `WorkerSettings` reads the repo's `.env`, so a machine with a real
    `CEREBRAS_API_KEY` put one into every test run — and because the Director
    chain prefers the hosted planner, the adapter tests started making live
    calls to api.cerebras.ai. On a box behind a TLS-intercepting proxy each of
    those hangs to its 60-second timeout before falling back, which turned a
    ten-minute suite into one that had not finished in twenty-seven.

    Autouse and unconditional: a suite whose runtime and network behaviour
    depend on which credentials happen to be lying around is not a suite. A
    test that wants the hosted provider constructs it explicitly with its own
    key and a mock transport (`tests/test_cerebras_director.py`).
    """
    monkeypatch.setattr(settings, "cerebras_api_key", "")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "job"
    path.mkdir()
    return path


@pytest.fixture
def fake_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A weights directory that passes the existence check without 85 GB.

    Includes the optional tiers' files: a test exercising the audio or
    control-conditioned path needs them present, and a test asserting they are
    NOT required for text-to-video asserts that against `_MODEL_FILES` directly.
    """
    from worker.adapters.ltx import _MODEL_FILES, _OPTIONAL_MODEL_FILES

    root = tmp_path / "models"
    for relative in (*_MODEL_FILES.values(), *_OPTIONAL_MODEL_FILES.values()):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    monkeypatch.setattr(settings, "ltx_model_dir", root)
    return root


@pytest.fixture
def stub_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory that exists, standing in for the LTX repo."""
    repo = tmp_path / "ltx-repo"
    repo.mkdir()
    monkeypatch.setattr(settings, "ltx_repo_dir", repo)
    return repo


# ── The model substitute ─────────────────────────────────────────────────


def stub_launcher(monkeypatch: pytest.MonkeyPatch, script: Path) -> None:
    """Routes `_launcher()` to a local script; every real flag still lands.

    The module argument is accepted and ignored: one stub stands in for every
    LTX entry point, and which one a path selected is asserted from the recorded
    argv rather than from which binary ran.
    """
    monkeypatch.setattr(
        LtxAdapter, "_launcher", lambda self, module=None: [sys.executable, str(script)]
    )


def render_stub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture: Path,
    *,
    sleep: float = 0.0,
    fail_on_pass: int | None = None,
) -> Path:
    """Installs a fake model and returns the path of its invocation log.

    Each line of the log is the complete argv of one render, as JSON. The stub
    copies `fixture` to whatever `--output-path` it was told to write, so the
    surrounding pipeline gets a genuine decodable MP4 and every assembly,
    measurement and validation step downstream is real.

    `fail_on_pass` makes the Nth invocation (0-based) exit non-zero, which is
    how partial-failure handling is proved: passes before it ran, passes after
    it must not.
    """
    log = tmp_path / "invocations.jsonl"
    script = tmp_path / "render.py"
    script.write_text(
        "import json, pathlib, shutil, sys, time\n"
        "args = sys.argv[1:]\n"
        f"log = pathlib.Path({str(log)!r})\n"
        "index = sum(1 for _ in log.open()) if log.exists() else 0\n"
        "with log.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps(args) + '\\n')\n"
        + (
            f"if index == {fail_on_pass}:\n"
            "    print('torch.OutOfMemoryError: CUDA out of memory', flush=True)\n"
            "    sys.exit(3)\n"
            if fail_on_pass is not None
            else ""
        )
        + "print('INFO:...:Building text encoder from /x', flush=True)\n"
        "print('INFO:...:Running denoising loop (8 steps)', flush=True)\n"
        + (f"time.sleep({sleep})\n" if sleep else "")
        + "out = args[args.index('--output-path') + 1]\n"
        f"shutil.copyfile({str(fixture)!r}, out)\n"
        "print(f'INFO:...:Video saved to {out}', flush=True)\n"
    )
    stub_launcher(monkeypatch, script)
    return log


def invocations(log: Path) -> list[list[str]]:
    """Every recorded command, in the order the passes ran."""
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]


def conditioning_of(argv: list[str]) -> list[tuple[str, int, float]]:
    """The `--image PATH FRAME STRENGTH` triples in one recorded command."""
    found: list[tuple[str, int, float]] = []
    for index, token in enumerate(argv):
        if token == "--image":
            path, frame, strength = argv[index + 1 : index + 4]
            found.append((path, int(frame), float(strength)))
    return found


def value_of(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# ── Real media fixtures, synthesised ─────────────────────────────────────


async def make_clip(
    path: Path,
    seconds: float,
    *,
    audio: bool = False,
    size: str = "160x120",
    rate: int = 24,
) -> Path:
    args = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}"]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100"]
    args += [
        "-t", f"{seconds:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", "24",
    ]
    if audio:
        args += ["-c:a", "aac"]
    args += [str(path)]
    await ffmpeg(args)
    return path


async def make_track(path: Path, seconds: float, *, beats_per_minute: int = 0) -> Path:
    """An audio file. With a BPM, a click track — energy rises on the beat.

    The click matters for the timing tests: a continuous tone has no onsets to
    find, and asserting that a detector finds nothing in silence proves much
    less than asserting it finds the beats in a track that has them.
    """
    if beats_per_minute:
        period = 60.0 / beats_per_minute
        source = (
            f"sine=frequency=440:sample_rate=44100,"
            f"tremolo=f={1 / period:g}:d=0.9"
        )
    else:
        source = "sine=frequency=340:sample_rate=44100"

    await ffmpeg(
        [
            "-f", "lavfi", "-i", source,
            "-t", f"{seconds:.3f}",
            "-c:a", "libmp3lame", "-b:a", "128k",
            str(path),
        ]
    )
    return path
