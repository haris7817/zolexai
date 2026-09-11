"""AI voice replacement for Video to Video (client package, 11 Sep 2026).

Four positional slots, one per mapped person. The invariants are all about
what must NOT change: the words, the timing, the music and ambience, the
people whose slots are empty, and the length of the finished file.

The worker embeds no cloning model. It owns this contract and sends the work
to a private service at `VOICE_CLONE_URL`. So these tests own the contract
too: what is sent, what is refused, and what happens to the answer.

**A hole is an instruction, not a gap.** Cloning Person 3 alone must leave
Persons 1 and 2 with their own voices, so slot order is positional and is
never closed up. That is the opposite of the person IMAGES, where a hole is
refused — see `test_ltx_video_to_video.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tests.conftest import collect, make_clip, make_job, needs_ffmpeg, render_stub, staged_input
from worker.adapters.base import AdapterError
from worker.adapters.ltx import LtxAdapter
from worker.core.config import settings
from worker.media import probe_media
from worker.media.voice_clone import VoiceCloneError, clone_source_voices

ROOT = Path(__file__).resolve().parents[3]
DEFINITION = ROOT / "workflow-definitions" / "video-to-video.yaml"

VOICE_ROLES = (
    "voice_reference",
    "voice_reference_2",
    "voice_reference_3",
    "voice_reference_4",
)


def voice_job(workspace: Path, source: Path | None, voices: dict[int, Path], **overrides):
    inputs = [staged_input("source_video", "video", "video/mp4", source)]
    for slot, path in sorted(voices.items()):
        inputs.append(staged_input(VOICE_ROLES[slot - 1], "audio", "audio/wav", path))
    execution = {
        "runtime": "ltx",
        "v2v_engine": "transform",
        "render_proxy": "540p",
        "delivery": "native",
        "v2v_voice_clone": True,
    }
    execution.update(overrides.pop("execution", {}))
    defaults = dict(
        workflow_id="video-to-video",
        prompt="repaint it as a charcoal sketch",
        parameters={},
        inputs=inputs,
        execution=execution,
    )
    return make_job(workspace, **{**defaults, **overrides})


# ── The slots ────────────────────────────────────────────────────────────


def test_the_definition_offers_four_optional_voices_one_per_person() -> None:
    workflow = yaml.safe_load(DEFINITION.read_text(encoding="utf-8"))
    roles = [item["role"] for item in workflow["inputs"]]
    assert [role for role in roles if role.startswith("voice_")] == list(VOICE_ROLES)

    by_role = {item["role"]: item for item in workflow["inputs"]}
    for role in VOICE_ROLES:
        assert by_role[role]["required"] is False, f"{role} must be optional"
        assert by_role[role]["kind"] == "audio"
        help_text = by_role[role]["help"].lower()
        assert "keep" in help_text and "own voice" in help_text, (
            "the copy must say what leaving the slot empty does"
        )

    execution = workflow["execution"]
    assert execution["v2v_voice_clone"] is True
    assert execution["v2v_max_voices"] == 4
    assert execution["v2v_voice_mapping"] == "visual_slot_order"
    assert execution["v2v_preserve_unmapped_voices"] is True
    assert execution["v2v_preserve_non_speech_audio"] is True


def test_the_offered_audio_types_are_ones_the_platform_accepts() -> None:
    """A form that offers a type the upload rule rejects is a promise the
    platform breaks at the worst moment — after the customer picked a file.
    Their package offered `audio/x-m4a`, which our allowlist does not."""
    accepted = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4", "audio/ogg"}
    workflow = yaml.safe_load(DEFINITION.read_text(encoding="utf-8"))
    for item in workflow["inputs"]:
        if not item["role"].startswith("voice_"):
            continue
        unknown = [mime for mime in item["accept"] if mime not in accepted]
        assert not unknown, f"{item['role']} offers {unknown}, which uploads reject"


def test_a_hole_stays_a_hole(workspace: Path) -> None:
    """Cloning Person 3 alone must not hand Person 1 that voice."""
    third = workspace / "three.wav"
    third.write_bytes(b"x")
    slots = LtxAdapter._v2v_voice_references(voice_job(workspace, None, {3: third}))

    assert slots == [None, None, third, None]
    assert len(slots) == 4, "the slots are positional and always four long"


def test_no_voices_is_the_ordinary_path(workspace: Path) -> None:
    assert LtxAdapter._v2v_voice_references(voice_job(workspace, None, {})) == [None] * 4


# ── What reaches the service ─────────────────────────────────────────────


class _Recorder:
    """Captures one multipart request and answers with real audio."""

    def __init__(self, reply: bytes, status: int = 200) -> None:
        self.reply = reply
        self.status = status
        self.data: dict | None = None
        self.files: list[str] = []
        self.headers: dict | None = None
        self.url: str | None = None

    def install(self, monkeypatch) -> None:
        recorder = self

        class FakeResponse:
            status_code = recorder.status
            content = recorder.reply
            text = ""

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc) -> None:
                return None

            async def post(self, url, data=None, files=None, headers=None):
                recorder.url = url
                recorder.data = data
                recorder.files = sorted(files or {})
                recorder.headers = headers
                return FakeResponse()

        monkeypatch.setattr("worker.media.voice_clone.httpx.AsyncClient", FakeClient)


@needs_ffmpeg
async def test_the_provider_is_told_which_slots_are_live_and_what_to_preserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = await make_clip(tmp_path / "source.mp4", 2.0, audio=True)
    reply = await make_clip(tmp_path / "reply.mp4", 2.0, audio=True)
    one, three = tmp_path / "v1.wav", tmp_path / "v3.wav"
    one.write_bytes(b"a")
    three.write_bytes(b"b")

    recorder = _Recorder(reply.read_bytes())
    recorder.install(monkeypatch)
    monkeypatch.setattr(settings, "voice_clone_url", "http://voice.invalid/v1/convert")
    monkeypatch.setattr(settings, "voice_clone_api_key", "s3cret")

    await clone_source_voices(
        source, [one, None, three, None], tmp_path / "mix.wav", duration_seconds=2.0
    )

    assert recorder.url == "http://voice.invalid/v1/convert"
    # Only the supplied voices are uploaded, under their own slot numbers.
    assert recorder.files == ["source_media", "voice_1", "voice_3"]
    assert json.loads(recorder.data["active_slots"]) == [1, 3]
    assert recorder.data["mapping_mode"] == "visual_slot_order"
    # The four promises the customer is actually buying.
    assert recorder.data["preserve_words"] == "true"
    assert recorder.data["preserve_timing"] == "true"
    assert recorder.data["preserve_unmapped_voices"] == "true"
    assert recorder.data["preserve_non_speech_audio"] == "true"
    assert recorder.data["duration_seconds"].startswith("2.")
    assert recorder.headers["Authorization"] == "Bearer s3cret"


@needs_ffmpeg
async def test_the_returned_mix_is_conformed_to_the_sources_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider that returns a slightly long or short mix must not move the
    dialogue against the picture."""
    source = await make_clip(tmp_path / "source.mp4", 3.0, audio=True)
    overrun = await make_clip(tmp_path / "overrun.mp4", 4.4, audio=True)
    voice = tmp_path / "v1.wav"
    voice.write_bytes(b"a")

    _Recorder(overrun.read_bytes()).install(monkeypatch)
    monkeypatch.setattr(settings, "voice_clone_url", "http://voice.invalid/v1/convert")

    mix = await clone_source_voices(
        source, [voice, None, None, None], tmp_path / "mix.wav", duration_seconds=3.0
    )

    info = await probe_media(mix)
    assert info.has_audio
    assert info.duration_seconds == pytest.approx(3.0, abs=0.12)


@needs_ffmpeg
async def test_an_unconfigured_service_fails_instead_of_returning_the_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point. A customer who paid for a voice change must not be
    handed the original voices with a success message."""
    source = await make_clip(tmp_path / "source.mp4", 1.0, audio=True)
    voice = tmp_path / "v1.wav"
    voice.write_bytes(b"a")
    monkeypatch.setattr(settings, "voice_clone_url", "")

    with pytest.raises(VoiceCloneError, match="VOICE_CLONE_URL"):
        await clone_source_voices(
            source, [voice, None, None, None], tmp_path / "mix.wav", duration_seconds=1.0
        )


@needs_ffmpeg
async def test_a_provider_error_is_surfaced_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = await make_clip(tmp_path / "source.mp4", 1.0, audio=True)
    voice = tmp_path / "v1.wav"
    voice.write_bytes(b"a")
    _Recorder(b"nope", status=503).install(monkeypatch)
    monkeypatch.setattr(settings, "voice_clone_url", "http://voice.invalid/v1/convert")

    with pytest.raises(VoiceCloneError, match="503"):
        await clone_source_voices(
            source, [voice, None, None, None], tmp_path / "mix.wav", duration_seconds=1.0
        )


async def test_asking_to_clone_nothing_is_a_programming_error(tmp_path: Path) -> None:
    with pytest.raises(VoiceCloneError, match="no voice references"):
        await clone_source_voices(
            tmp_path / "source.mp4", [None] * 4, tmp_path / "mix.wav", duration_seconds=1.0
        )


# ── Where it sits in the job ─────────────────────────────────────────────


@needs_ffmpeg
async def test_a_silent_source_refuses_the_voice_rather_than_ignoring_it(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = await make_clip(workspace / "source.mp4", 2.0, audio=False)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))
    voice = workspace / "v1.wav"
    voice.write_bytes(b"a")

    with pytest.raises(AdapterError) as failure:
        await collect(voice_job(workspace, source, {1: voice}))
    assert "no voices to replace" in failure.value.user_message
    assert failure.value.retriable is False


@needs_ffmpeg
async def test_a_workflow_with_cloning_off_refuses_a_voice(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))
    voice = workspace / "v1.wav"
    voice.write_bytes(b"a")

    with pytest.raises(AdapterError) as failure:
        await collect(voice_job(
            workspace, source, {1: voice}, execution={"v2v_voice_clone": False},
        ))
    assert "not enabled" in failure.value.user_message


@needs_ffmpeg
async def test_no_voice_slot_means_the_service_is_never_called(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ordinary restyle must not depend on a voice service being up."""
    called: list[str] = []

    async def explode(*args, **kwargs):
        called.append("clone")
        raise AssertionError("the voice service was called for a job with no voices")

    monkeypatch.setattr("worker.adapters.ltx.clone_source_voices", explode)
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))

    await collect(voice_job(workspace, source, {}))

    assert called == []
    info = await probe_media(workspace / "output.mp4")
    assert info.has_audio, "the source's own soundtrack still comes back"


@needs_ffmpeg
async def test_the_cloned_mix_is_muxed_before_the_upscale(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Order matters for cost, not correctness: converting audio against an
    8K picture buys nothing, and the single delivery encode should copy one
    finished soundtrack rather than re-mux an enlarged file."""
    from worker.media import ffmpeg

    source = await make_clip(workspace / "source.mp4", 2.0, audio=True, size="256x144")
    replacement = tmp_path / "reply.wav"
    await ffmpeg([
        "-f", "lavfi", "-i", "sine=frequency=330:duration=2.0",
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(replacement),
    ])
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))
    voice = workspace / "v1.wav"
    voice.write_bytes(b"a")

    seen: list[tuple[int, int]] = []

    async def fake_clone(source_media, voice_slots, dest, **kwargs):
        info = await probe_media(source_media)
        seen.append((info.width or 0, info.height or 0))
        dest.write_bytes(replacement.read_bytes())
        return dest

    monkeypatch.setattr("worker.adapters.ltx.clone_source_voices", fake_clone)

    result, _ = await collect(voice_job(
        workspace, source, {1: voice},
        parameters={"quality": "1080p"},
        execution={"delivery": "1080p"},
    ))

    assert seen, "the voice service was never reached"
    # It is handed the SOURCE, not an upscaled picture.
    assert seen[0] == (256, 144)
    assert (result.width, result.height) == (1920, 1080), "the upscale still happened after"
    info = await probe_media(workspace / "output.mp4")
    assert info.audio_stream_count == 1, "exactly one soundtrack on the delivered file"
