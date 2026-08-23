"""LTX as a provider — a reader of the shipped adapter, never a rewrite of it.

Every number this module reports is produced by calling `LtxAdapter`'s own
planning helpers. That is deliberate and it is the entire safety property:
if this file computed section counts or frame counts itself, the manifest
would be a second opinion that could drift from what the GPU is actually sent,
and the regression snapshot built on it would be worthless. Ask the adapter,
or do not report it.

Generation delegates straight to the adapter. Nothing here is on the path of a
running job.
"""

from __future__ import annotations

from worker.adapters.base import AdapterJob, AdapterResult, ProgressCallback
from worker.adapters.ltx import _A2VID as _A2VID_PIPELINE
from worker.adapters.ltx import (
    _AUDIO_WINDOW_PAD_SECONDS,
    _DISTILLED,
    _GUIDED,
    _IC_LORA,
    _TWO_IMAGE_SAFE_FRAMES,
    LtxAdapter,
    conforming_frames,
    safe_frame_count,
)
from worker.core.config import settings
from worker.director import wants_director
from worker.longform import plan_chain_segments, plan_section_prompts, structure_prompt
from worker.providers.base import ProviderRefusal
from worker.providers.capabilities import MATRIX, Capability
from worker.providers.manifest import (
    AudioWindow,
    GenerationManifest,
    ReferenceSpec,
    SectionPlan,
)


class LtxProvider:
    name = "ltx"

    def __init__(self, adapter: LtxAdapter | None = None) -> None:
        self._adapter = adapter or LtxAdapter()

    def capabilities(self) -> dict[str, Capability]:
        return MATRIX

    def health(self) -> tuple[bool, str]:
        root = settings.ltx_models_root
        if not settings.ltx_repo_dir.exists():
            return False, f"LTX repo directory {settings.ltx_repo_dir} is absent"
        if not root.exists():
            return False, f"LTX weights root {root} is absent"
        return True, ""

    def validate(self, job: AdapterJob) -> list[str]:
        if not self._adapter.supports(job.workflow_id):
            return [f"LTX does not serve workflow {job.workflow_id}"]
        return []

    # ── Compilation ──────────────────────────────────────────────────────

    def compile(self, job: AdapterJob) -> GenerationManifest:
        """The plan LTX would execute, read out of the adapter itself."""
        refusals = self.validate(job)
        if refusals:
            raise ProviderRefusal(refusals[0], capability=job.workflow_id)

        adapter = self._adapter
        notes: list[str] = []

        if job.workflow_id in ("video-to-video", "music-video"):
            # Length comes from the upload, which a dry run has not probed.
            # `dry_run_source_seconds` is how a caller states it.
            total = job.execution_float("dry_run_source_seconds", 0.0)
            if total <= 0:
                raise ProviderRefusal(
                    f"{job.workflow_id} takes its length from the uploaded file; "
                    "supply execution.dry_run_source_seconds to compile a dry run",
                    capability="duration",
                )
            notes.append(
                "duration taken from execution.dry_run_source_seconds (a real job "
                "probes the upload)"
            )
        else:
            total = adapter._requested_seconds(job)

        pipeline, grid, per_pass = self._tier(job, total)
        if job.execution.get("prompt_structuring") and not wants_director(job):
            prompt_text = structure_prompt(
                job.prompt, v2=bool(job.execution.get("prompt_structuring_v2"))
            )
        else:
            prompt_text = job.prompt

        segments = plan_chain_segments(total, per_pass)
        section_prompts = self._section_prompts(job, prompt_text, len(segments), total)

        sections: list[SectionPlan] = []
        for segment in segments:
            requested = adapter._frame_count(segment.duration_seconds)
            references, conditioned = self._references(job, segment.index, requested)
            audio = self._audio(job, segment.start_seconds, pipeline)
            if audio is not None:
                conditioned = True
            rendered = self._rendered_frames(pipeline, grid, requested, conditioned)
            if rendered != requested:
                notes.append(
                    f"section {segment.index}: {requested} frames is not a measured "
                    f"landing on this decode path, rendering {rendered} and trimming back"
                )
            if audio is not None:
                audio = AudioWindow(
                    start_seconds=audio.start_seconds,
                    duration_seconds=(
                        rendered / float(settings.ltx_frame_rate)
                        + _AUDIO_WINDOW_PAD_SECONDS
                    ),
                    mode=audio.mode,
                    returns_input_waveform=audio.returns_input_waveform,
                )
            sections.append(
                SectionPlan(
                    index=segment.index,
                    start_seconds=round(segment.start_seconds, 4),
                    duration_seconds=round(segment.duration_seconds, 4),
                    frames_requested=requested,
                    frames_rendered=rendered,
                    seed=adapter._seed_for_step(job, segment.index),
                    prompt=section_prompts[segment.index],
                    references=references,
                    audio=audio,
                )
            )

        notes.extend(self._unconsumed_inputs(job, pipeline))

        return GenerationManifest(
            provider=self.name,
            workflow_id=job.workflow_id,
            pipeline=pipeline.module,
            total_seconds=round(total, 4),
            width=grid[0],
            height=grid[1],
            fps=float(settings.ltx_frame_rate),
            sections=sections,
            settings=self._settings(job, pipeline, per_pass),
            notes=notes,
        )

    # ── The pieces, each asking the adapter rather than deciding ─────────

    def _tier(self, job: AdapterJob, total: float):
        """(pipeline, grid, per-pass ceiling) exactly as the handler picks them."""
        adapter = self._adapter
        grid = adapter._requested_dimensions(job)

        if job.workflow_id == "video-to-video":
            if str(job.execution.get("v2v_engine") or "").strip() == "transform":
                per_pass = min(
                    adapter._per_pass_seconds(job, grid),
                    max(1.0, job.execution_float("transform_pass_seconds", 8.0)),
                )
                return _IC_LORA, grid, per_pass
            return _DISTILLED, grid, adapter._per_pass_seconds(job, grid)

        if job.workflow_id == "music-video":
            if job.execution.get("audio_conditioning"):
                return _A2VID_PIPELINE, grid, adapter._audio_pass_seconds(job)
            return _DISTILLED, grid, adapter._per_pass_seconds(job, grid)

        if job.execution.get("generation_engine") == "guided":
            return _GUIDED, grid, adapter._guided_pass_seconds(job)

        return _DISTILLED, grid, adapter._per_pass_seconds(job, grid)

    def _section_prompts(
        self, job: AdapterJob, prompt_text: str, count: int, total: float
    ) -> list[str]:
        if job.workflow_id == "video-to-video" and not job.execution.get(
            "v2v_section_prompts"
        ):
            # The shipped default: every section receives identical text.
            return [prompt_text] * count
        if wants_director(job):
            # A real Director plan needs a planner; the manifest records the
            # shape rather than inventing dialogue.
            return [f"<director caption, section {i + 1} of {count}>" for i in range(count)]
        return plan_section_prompts(
            prompt_text,
            count,
            total_seconds=total,
            v2=bool(job.execution.get("prompt_structuring_v2")),
        )

    def _references(
        self, job: AdapterJob, index: int, frames: int
    ) -> tuple[list[ReferenceSpec], bool]:
        items: list[ReferenceSpec] = []
        still = job.input_for("source_image")
        reference = job.input_for("reference_image")

        if job.workflow_id == "video-to-video":
            strength = 0.85 if index else job.execution_float(
                "v2v_reference_strength", 0.3
            )
            if index or reference is not None:
                items.append(
                    ReferenceSpec(
                        role="seam" if index else "identity",
                        kind="image",
                        native="--image PATH 0 STRENGTH",
                        frame_index=0,
                        strength=strength,
                        source="previous section final frame" if index else "reference_image",
                    )
                )
            if str(job.execution.get("v2v_engine") or "").strip() == "transform":
                items.append(
                    ReferenceSpec(
                        role="structure",
                        kind="video",
                        native="--video-conditioning PATH STRENGTH",
                        strength=job.execution_float("v2v_control_strength", 1.0),
                        source="canny edge map of the source window",
                    )
                )
            else:
                items.append(
                    ReferenceSpec(
                        role="structure",
                        kind="image",
                        native="--image PATH FRAME STRENGTH (keyframes)",
                        strength=job.execution_float("v2v_structure_strength", 0.45),
                        source="stills lifted from this section's source window",
                    )
                )
            return items, True

        if index == 0 and still is not None:
            items.append(
                ReferenceSpec(
                    role="first_frame",
                    kind="image",
                    native="--image PATH 0 1.0",
                    frame_index=0,
                    strength=1.0,
                    source="source_image",
                )
            )
        elif index > 0 or job.workflow_id == "extend-video":
            items.append(
                ReferenceSpec(
                    role="seam",
                    kind="image",
                    native="--image PATH 0 1.0",
                    frame_index=0,
                    strength=1.0,
                    source=(
                        "source final frame"
                        if index == 0
                        else "previous section final frame"
                    ),
                )
            )
        if index > 0 and still is not None:
            if frames in _TWO_IMAGE_SAFE_FRAMES:
                items.append(
                    ReferenceSpec(
                        role="identity",
                        kind="image",
                        native="--image PATH FRAME STRENGTH",
                        frame_index=min(frames - 1, max(1, frames // 3)),
                        strength=job.execution_float("i2v_reference_strength", 0.2),
                        source="source_image",
                    )
                )
        return items, bool(items)

    def _audio(self, job: AdapterJob, start: float, pipeline) -> AudioWindow | None:
        if pipeline is not _A2VID_PIPELINE:
            return None
        return AudioWindow(
            start_seconds=round(start, 4),
            duration_seconds=0.0,  # filled once the rendered frame count is known
            mode="frozen_latent",
            returns_input_waveform=True,
        )

    def _rendered_frames(self, pipeline, grid, requested: int, conditioned: bool) -> int:
        if pipeline.conforming_only:
            frames = conforming_frames(requested)
            landing = next(
                (c for c in pipeline.measured_landings if c >= frames), None
            )
            return landing if landing is not None else frames
        return safe_frame_count(grid, requested, conditioned=conditioned)

    def _unconsumed_inputs(self, job: AdapterJob, pipeline) -> list[str]:
        """Inputs the customer supplied that never reach the model on this path.

        Without this a multimodal request compiles into a perfectly ordinary
        looking LTX plan that has quietly dropped two of its three references,
        and a benchmark would score that plan as though it had been given the
        same material as the other engine. Say it out loud instead.
        """
        notes: list[str] = []
        if job.input_for("source_audio") is not None and pipeline is not _A2VID_PIPELINE:
            notes.append(
                "source_audio never reaches the model on this path — it is muxed "
                "onto the finished picture. Only a2vid_two_stage conditions on "
                "audio, and it serves music-video with audio_conditioning on"
            )
        if (
            job.input_for("reference_image") is not None
            and job.workflow_id == "video-to-video"
            and not job.execution.get("v2v_reference_identity")
        ):
            notes.append(
                "reference_image is used only as a low-strength look anchor on "
                "the first pass; identity replacement needs "
                "v2v_reference_identity with the transform engine"
            )
        if job.input_for("source_video") is not None and job.workflow_id not in (
            "video-to-video",
            "extend-video",
        ):
            notes.append(
                f"source_video is not an input {job.workflow_id} consumes on LTX"
            )
        return notes

    def _settings(self, job: AdapterJob, pipeline, per_pass: float) -> dict:
        return {
            "per_pass_seconds": round(per_pass, 4),
            "quantization": settings.ltx_quantization if pipeline.quantize else None,
            "offload": "cpu" if pipeline.offload_cpu else None,
            "distilled_lora": pipeline.distilled_lora,
            "skip_stage_2": pipeline.stage_1_only,
            "conforming_only": pipeline.conforming_only,
            "measured_landings": list(pipeline.measured_landings),
            "guidance": (
                "pipeline defaults (cfg 3.0 / stg 1.0 blocks [28] / 30 steps)"
                if pipeline.distilled_lora
                else "not applicable — this entry point has no guiders"
            ),
            "prompt_structuring_v2": bool(job.execution.get("prompt_structuring_v2")),
        }

    # ── Running ──────────────────────────────────────────────────────────

    async def generate(
        self, job: AdapterJob, on_progress: ProgressCallback
    ) -> AdapterResult:
        return await self._adapter.run(job, on_progress)
