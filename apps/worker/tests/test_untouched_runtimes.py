"""Video to Video and the music pipeline still run where they always ran.

The final milestone added two runtimes (`ltx_comfy`, `character_replacement`)
beside the CLI LTX runtime. This module pins the boundary: the CLI adapter
still owns Video to Video, Music Video and its restyle/audio machinery; the
new adapters decline those workflows; and the resolver keeps a stale
deployment line from moving them.

The CLI adapter's own behaviour is proven by its existing suites
(test_video_to_video, test_reference_identity, test_transform, test_ltx_golden,
test_music_video*), which the milestone did not modify.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from worker.adapters.base import AdapterJob
from worker.adapters.character_replacement import CharacterReplacementAdapter
from worker.adapters.ltx import LtxAdapter
from worker.adapters.ltx_comfy import LtxComfyAdapter
from worker.adapters.music import MusicAdapter
from worker.workflows.resolver import resolve_adapter

ROOT = Path(__file__).resolve().parents[3]

#: The CLI runtime's source, as committed before the milestone (926d2e3).
#: Only its callers changed shape; the file itself did not.
PINNED_LTX_ADAPTER_SHA256 = "ebdb10ea64fc4590101ca6315fb70a0a816463c6445c93fa959f7d93f673dfa7"


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_the_cli_adapter_still_serves_the_untouched_workflows() -> None:
    adapter = LtxAdapter()
    assert adapter.supports("video-to-video")
    assert adapter.supports("music-video")
    assert adapter.supports("extend-video")  # the rollback runtime for Extend
    assert set(LtxAdapter._SUPPORTED) == {
        "text-to-video",
        "image-to-video",
        "extend-video",
        "video-to-video",
        "music-video",
    }


def test_the_new_runtimes_decline_the_untouched_workflows() -> None:
    for workflow in ("video-to-video", "music-video", "music"):
        assert not LtxComfyAdapter().supports(workflow), workflow
        assert not CharacterReplacementAdapter().supports(workflow), workflow
    assert MusicAdapter().supports("music")


def test_the_resolver_keeps_video_to_video_on_its_runtime() -> None:
    for quality in ("fast", "best", None):
        job = AdapterJob(
            job_id="j",
            workflow_id="video-to-video",
            workflow_version="1",
            prompt="x",
            parameters={"quality": quality} if quality else {},
            execution={
                "runtime": "ltx",
                "runtime_by_quality": {"fast": "ltx", "best": "ltx"},
                "v2v_engine": "transform",
                "execution_by_quality": {"best": {"v2v_reference_identity": True}},
            },
        )
        assert resolve_adapter(job).name == "ltx"
    music_video = AdapterJob(
        job_id="j",
        workflow_id="music-video",
        workflow_version="1",
        prompt="x",
        parameters={},
        execution={"runtime": "ltx", "audio_conditioning": True},
    )
    assert resolve_adapter(music_video).name == "ltx"


def test_the_cli_adapter_source_did_not_change_this_milestone() -> None:
    """A byte-level guard on apps/worker/worker/adapters/ltx.py.

    The pin is the file as of the milestone's starting commit. If it fails,
    either the file changed (the client asked that it not) or the pin needs
    updating after an approved change — decide which before touching it.
    """
    assert _sha256_lf(ROOT / "apps/worker/worker/adapters/ltx.py") == PINNED_LTX_ADAPTER_SHA256
