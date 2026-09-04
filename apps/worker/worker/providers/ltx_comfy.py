"""The LTX 2.5 ComfyUI service — one provider, two adapters.

The client's three graphs run on one long-lived ComfyUI instance the worker
connects to and never manages (the same relationship it has with ACE-Step
and with the H3 ComfyUI). This module is that connection, with the five
operations the integration brief names:

    health()    — reachable, nodes installed, models visible, graphs compile
    generate()  — submit a compiled prompt
    progress()  — poll until done, pacing the customer's bar, honouring cancel
    cancel()    — remove the prompt from the queue / interrupt it
    collect()   — bring the finished file back into the job's workspace

Everything travels over HTTP: inputs go up through `/upload/image`, outputs
come back through `/view`. No shared filesystem is assumed, which is what
lets the worker and the service live in different containers or on different
hosts if that ever becomes useful.

The `text-to-video`/`image-to-video`/`extend-video` adapter and the
character-replacement adapter both use this object. Keeping the service
apart from either adapter is what makes "a completely separate module" for
character replacement true without duplicating the transport.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from worker.adapters.base import AdapterJob
from worker.comfy.client import ComfyClient, ComfyError
from worker.comfy.ltx_graphs import (
    ASPECT_LABELS,
    GenerationEdits,
    ReplacementEdits,
    combo_options,
    compile_character_replacement,
    compile_first_last_frame,
    compile_text_to_video,
    graph_sha256,
    load_graph,
    model_files_referenced,
    verify_against_object_info,
)
from worker.core.config import settings
from worker.core.logging import get_logger

logger = get_logger(__name__)

GraphName = Literal["text_to_video", "first_last_frame", "character_replacement"]

#: Graph files in the frozen pack, and the sha256 each was delivered with.
GRAPH_FILES: dict[GraphName, str] = {
    "text_to_video": "ltx25_text_to_video.json",
    "first_last_frame": "ltx25_first_last_frame.json",
    "character_replacement": "ltx25_character_replacement.json",
}
GRAPH_SHA256: dict[GraphName, str] = {
    "text_to_video": "2dcd9661118c947cc1cae0e5aa59656b519387a8f8e86f8e4c06545bd07b914c",
    "first_last_frame": "1926bd6dd4f897b45eb8f9e20072066f90fd01678107287f6b6459921e4da967",
    "character_replacement": "2ea7547268f8742ba657fcf390800501e39ba7aff5d1736fa6b41ed988b1adc9",
}

#: Weight files the pack loads, by ComfyUI models/ subfolder — the deep
#: health check's manifest. Sizes are recorded on the node at provisioning
#: (`scripts/ltx_comfy_health.py --record`), not guessed here.
REQUIRED_WEIGHTS: dict[str, str] = {
    "diffusion_models/LTX-2.5-Distilled-Q8_0.gguf": "Abiray/LTX-2.5-Distilled-GGUF",
    "diffusion_models/LTXVideo/v2/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors": "Lightricks/LTX-2.5",  # noqa: E501
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors": "Lightricks/LTX-2.5",  # noqa: E501
    "vae/ltx-2.5-video-vae-bf16.safetensors": "Lightricks/LTX-2.5",
    "vae/ltx-2.5-audio-vae-bf16.safetensors": "Lightricks/LTX-2.5",
    "vae/taeltx2_3.safetensors": "madebyollin/taehv",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors": "Lightricks/LTX-2.5",  # noqa: E501
    "loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors": "Kijai/LTX2.3_comfy",
    "loras/ltx2.3-transition.safetensors": "joyfox/LTX-2.3-Transition-LORA",
    "loras/ltx-2-19b-ic-lora-detailer.safetensors": "Lightricks/LTX-2-19b-IC-LoRA-Detailer",
    "loras/LTX/LTX-2.5/LTX25_Ripple_v11.safetensors": "WepeNerd/LTX-Ripple",
}

_MIN_FREE_DISK_BYTES = 30 * 2**30
_MIN_VRAM_BYTES = 40 * 2**30  # UNMEASURED for this pack; the GGUF Q8 alone is 23.6 GB


class LtxComfyService:
    name = "ltx_comfy"

    def __init__(self, client: ComfyClient | None = None) -> None:
        self._client = client
        self._object_info: dict[str, Any] | None = None

    @property
    def client(self) -> ComfyClient:
        if self._client is None:
            self._client = ComfyClient(
                settings.ltx_comfy_base_url,
                request_timeout=settings.ltx_comfy_request_timeout,
                poll_seconds=settings.ltx_comfy_poll_seconds,
            )
        return self._client

    # ── Graphs ────────────────────────────────────────────────────────────

    @staticmethod
    def graph_path(name: GraphName) -> Path:
        return settings.ltx_comfy_workflows_dir / GRAPH_FILES[name]

    def load(self, name: GraphName) -> dict[str, Any]:
        path = self.graph_path(name)
        if not path.is_file():
            raise ComfyError(
                "This tool is temporarily unavailable.",
                internal_detail=f"frozen client graph missing: {path}",
                retriable=False,
            )
        return load_graph(path)

    def graphs_match_the_pack(self) -> list[str]:
        """Which frozen graphs differ from the delivered ZIP, by hash."""
        drift: list[str] = []
        for name, expected in GRAPH_SHA256.items():
            path = self.graph_path(name)
            if not path.is_file():
                drift.append(f"{path.name}: missing")
            elif graph_sha256(path) != expected:
                drift.append(f"{path.name}: sha256 differs from the delivered ZIP")
        return drift

    # ── Live catalogue ────────────────────────────────────────────────────

    async def object_info(self, *, refresh: bool = False) -> dict[str, Any] | None:
        """The server's node catalogue, cached per process. None when unreachable."""
        if self._object_info is None or refresh:
            try:
                self._object_info = await self.client.object_info()
            except Exception as exc:  # noqa: BLE001 - unreachable is a valid answer
                logger.warning("ltx_comfy_object_info_unavailable", extra={"error": str(exc)})
                return None
        return self._object_info

    async def aspect_options(self) -> list[str] | None:
        info = await self.object_info()
        if info is None:
            return None
        return combo_options(info, "ResolutionSelector", "aspect_ratio")

    # ── Health ────────────────────────────────────────────────────────────

    async def health(self, *, deep: bool = False) -> tuple[bool, str]:
        """Can this node run the pack right now, and if not, what is missing.

        Order matters: the cheap, decisive checks first. A miss at any step
        reports unavailable; the resolver's fallback then keeps the job on
        the base runtime rather than failing it.
        """
        problems: list[str] = []
        problems.extend(self.graphs_match_the_pack())

        reachable, detail = await self.client.reachable()
        if not reachable:
            return False, detail

        info = await self.object_info(refresh=True)
        if info is None:
            return False, "ComfyUI reachable but /object_info failed"

        aspect = combo_options(info, "ResolutionSelector", "aspect_ratio") or []
        for ratio in ("16:9", "9:16", "1:1"):
            if not any(o == ratio or o.startswith(ratio + " ") for o in aspect):
                problems.append(f"ResolutionSelector offers no {ratio} option")

        for name, api in self._probe_prompts().items():
            for problem in verify_against_object_info(api, info):
                problems.append(f"{name}: {problem}")

        if deep and settings.ltx_comfy_models_dir is not None:
            root = settings.ltx_comfy_models_dir
            for relative in REQUIRED_WEIGHTS:
                if not (root / relative).is_file():
                    problems.append(f"weight missing on disk: {relative}")
        if deep:
            try:
                usage = shutil.disk_usage(settings.ltx_comfy_models_dir or Path.cwd())
                if usage.free < _MIN_FREE_DISK_BYTES:
                    problems.append(f"free disk {usage.free / 2**30:.1f} GB below the 30 GB floor")
            except OSError:
                pass
            try:
                stats = await self.client.system_stats()
                devices = stats.get("devices") or []
                vram = int(devices[0].get("vram_total", 0)) if devices else 0
                if vram and vram < _MIN_VRAM_BYTES:
                    problems.append(f"vram_total {vram / 2**30:.0f} GB below the 40 GB floor")
            except Exception:  # noqa: BLE001
                problems.append("system_stats unavailable")

        if problems:
            return False, "; ".join(problems[:12]) + (" …" if len(problems) > 12 else "")
        return True, (
            f"LTX ComfyUI up at {settings.ltx_comfy_base_url}; "
            f"{len(GRAPH_FILES)} graphs compile and every node class, combo value "
            "and model file they name is offered by the server"
        )

    def _probe_prompts(self) -> dict[str, dict[str, Any]]:
        """The three graphs compiled with placeholder job inputs."""
        probe = GenerationEdits(
            positive="health check",
            negative="",
            seconds=5,
            aspect_label=ASPECT_LABELS["16:9"],
            seed_base=1,
            filename_prefix="zolexai/health",
        )
        out: dict[str, dict[str, Any]] = {}
        out["text_to_video"] = compile_text_to_video(self.load("text_to_video"), probe)
        out["first_last_frame"] = compile_first_last_frame(
            self.load("first_last_frame"),
            GenerationEdits(
                positive=probe.positive,
                negative=probe.negative,
                seconds=probe.seconds,
                aspect_label=probe.aspect_label,
                seed_base=probe.seed_base,
                filename_prefix=probe.filename_prefix,
                first_image="health.png",
                last_image="health.png",
            ),
        )
        out["character_replacement"] = compile_character_replacement(
            self.load("character_replacement"),
            ReplacementEdits(
                positive=probe.positive,
                negative=probe.negative,
                video="health.mp4",
                image="health.png",
                seconds=5,
                width=736,
                height=1280,
                seed_base=1,
                filename_prefix=probe.filename_prefix,
            ),
        )
        return out

    def referenced_models(self) -> dict[str, list[str]]:
        """Every weight the three compiled prompts load, for documentation."""
        merged: dict[str, list[str]] = {}
        for api in self._probe_prompts().values():
            for key, files in model_files_referenced(api).items():
                bucket = merged.setdefault(key, [])
                for file in files:
                    if file not in bucket:
                        bucket.append(file)
        return merged

    # ── The five operations ───────────────────────────────────────────────

    async def upload(self, path: Path, *, name: str) -> str:
        return await self.client.upload_input(
            path, name=name, timeout=settings.ltx_comfy_transfer_timeout
        )

    async def generate(self, api_prompt: dict[str, Any], *, client_id: str) -> str:
        return await self.client.submit(api_prompt, client_id=client_id)

    async def progress(
        self,
        job: AdapterJob,
        prompt_id: str,
        *,
        timeout_seconds: float,
        on_tick: Callable[[float], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        return await self.client.wait(
            job, prompt_id, timeout_seconds=timeout_seconds, on_tick=on_tick
        )

    async def cancel(self, prompt_id: str) -> None:
        await self.client.cancel(prompt_id)

    async def collect(self, history: dict[str, Any], dest: Path) -> Path:
        """Downloads the finished MP4 the graph's output node reported.

        VideoHelperSuite reports its files under `outputs[<node>]["gifs"]`
        (the key is historical; the entries are videos). The final combined
        file is the one whose format is a video and whose type is `output`.
        """
        outputs = history.get("outputs") or {}
        candidates: list[dict[str, Any]] = []
        for entries in outputs.values():
            for key in ("gifs", "videos", "images"):
                for item in entries.get(key) or []:
                    filename = str(item.get("filename") or "")
                    fmt = str(item.get("format") or "")
                    if filename.lower().endswith(".mp4") or fmt.startswith("video/"):
                        candidates.append(item)
        if not candidates:
            raise ComfyError(
                "The finished video could not be found.",
                internal_detail=f"no video entry in history outputs: {list(outputs)[:10]}",
            )
        # `output` over `temp`, and the last one written over any earlier.
        candidates.sort(key=lambda item: (str(item.get("type")) == "output",))
        chosen = candidates[-1]
        return await self.client.download_output(
            filename=str(chosen["filename"]),
            subfolder=str(chosen.get("subfolder") or ""),
            output_type=str(chosen.get("type") or "output"),
            dest=dest,
            timeout=settings.ltx_comfy_transfer_timeout,
        )

    async def free_memory(self) -> None:
        await self.client.free_memory()


def wall_clock() -> float:
    return time.monotonic()
