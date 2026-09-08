# Performance and five-minute operation

## Duration behavior

`ZOLEX_MAX_SOURCE_SECONDS` defaults to `300`. Duration is measured after FFmpeg decodes the uploaded audio, not from user metadata. A source longer than 300 seconds plus one frame of tolerance is rejected before creative rendering begins.

The aligned output duration is:

```text
round(decoded_seconds × 24) / 24
```

At the default 4.5-second scene target, a 300-second song normally produces about 67 scenes. Every scene has an exact start frame and delivered frame count. The final master contains 7,200 video frames at 24 FPS, with the aligned original song encoded into the working master and stream-copied during 4K finishing.

Landscape scenes render at 1280×704 and finish at 3840×2160. Portrait scenes render at 704×1280 and finish at 2160×3840; square uses 1024×1024 and 2160×2160. The worker assembles at working size, then performs one CUDA Lanczos/NVENC finishing pass. This keeps 4K delivery from multiplying the cost of every generative scene. Final QA rejects any master that does not have the required dimensions, audio, frame rate, or frame count.

When the prompt asks to follow the lyrics, transcription happens once before scene rendering. Faster-Whisper model startup and transcription add a small fixed stage to the job; they do not change the frame-exact duration. Keep its model cache on persistent storage so workers do not download or initialize from a cold cache unnecessarily.

When `reference_video` is supplied, frame sampling, cut detection and visual analysis run alongside song analysis and lyric transcription. Built-in analysis is lightweight compared with generative rendering. A configured external vision model can add latency, so keep it warm and include it in the complete five-minute production benchmark.

## Five-minute processing target

The configured processing target is 300 seconds for one 180-second 4K output. This is distinct from the 300-second maximum source-audio duration. The target covers the whole job, including 4K finishing.

The capacity calculator uses the backend's measured baseline:

```text
required concurrency = ceil(
  baseline concurrency × baseline job seconds × requested workload ratio
  / target job seconds
)
```

For the measured 900-second baseline and a 300-second target, the required capacity multiplier is 3×. If the 15-minute job used four warm render lanes, configure approximately 12 equivalent lanes. Two requested output formats count as twice the workload.

Set:

```text
ZOLEX_TARGET_JOB_SECONDS=300
ZOLEX_BASELINE_AUDIO_SECONDS=180
ZOLEX_BASELINE_JOB_SECONDS=900
ZOLEX_BASELINE_RENDER_CONCURRENCY=<lanes used by the measured run>
ZOLEX_RENDER_CONCURRENCY=<calculated target lanes>
ZOLEX_RENDER_BACKEND=command
ZOLEX_ENFORCE_LATENCY_CAPACITY=true
```

The command backend is required for target enforcement because it preserves the existing warm renderer. Direct LTX mode launches a new CLI process for each scene and is therefore not treated as target-ready even when many GPU IDs are supplied.

The worker writes `latency-plan.json` before creative work and sends `job_latency_target_seconds`, `latency_priority`, and `quality_policy` to the command-render request. It never reduces inference steps or scene quality automatically. Target readiness only means the configured render capacity matches the measured baseline calculation; it is not a deadline guarantee. Rebenchmark the complete v1.5 pipeline because URL fetching, 4K finishing and optional reference analysis are fixed stages outside scene generation.

To reduce orchestration overhead, audio analysis and lyric transcription run at the same time, and unique anchor images use `ZOLEX_ANCHOR_CONCURRENCY`. Render lanes stay sequential per assigned GPU so parallelism cannot accidentally place two local LTX jobs on one device.

## Concurrency model

`ZOLEX_RENDER_CONCURRENCY` controls how many independent scenes the orchestrator submits simultaneously. `ZOLEX_RENDER_GPU_IDS` assigns one visible CUDA device to each local direct-LTX worker.

Examples:

```text
# One GPU
ZOLEX_RENDER_CONCURRENCY=1
ZOLEX_RENDER_GPU_IDS=0

# Four GPUs
ZOLEX_RENDER_CONCURRENCY=4
ZOLEX_RENDER_GPU_IDS=0,1,2,3
```

The worker validates that the GPU-ID list contains at least as many entries as the concurrency setting. Direct LTX subprocesses receive `CUDA_VISIBLE_DEVICES`. Alternate command renderers receive `worker_slot` and `gpu_id` in both the request JSON and template placeholders.

Concurrency is per music-video job. The backend's global queue must still prevent different jobs from oversubscribing the same GPUs.

## Recommended production topology

For repeated jobs, use one warm model service per GPU and select `ZOLEX_RENDER_BACKEND=command`. The worker can then submit several scenes concurrently without reloading model components for every shot. The renderer service should keep LTX resident, serialize work on its assigned GPU, write the requested output path, and return only when the file is complete.

The direct `ltx` adapter is useful for a simple deployment and GPU smoke tests. It launches the official `ltx_pipelines.a2vid_two_stage` CLI for each scene, so process and model-loading overhead may dominate short scenes.

## Speed-quality controls

The default `LTX_NUM_INFERENCE_STEPS=24` is a speed-oriented starting point. Increase it for more refinement or reduce it only after visual testing. The official repository notes that 20–30 steps can maintain quality when its gradient-estimation optimization is configured; follow the exact LTX revision installed on the deployment host.

Optional official CLI flags can be passed as a JSON argument array:

```text
LTX_EXTRA_ARGS_JSON=["--quantization","fp8-cast"]
```

Worker-owned arguments such as prompt, image, audio, frame count, dimensions, seed, and output path cannot be overridden through this field. That prevents a tuning option from breaking timeline or output guarantees.

Install the official Linux/CUDA `natten` extra when supported. FP8 compatibility and performance depend on the GPU generation, CUDA build, PyTorch build, and checkpoint type; validate it on the deployment host.

## Estimating completion time

A useful approximation is:

```text
job_seconds ≈ sum(scene_render_seconds) / active_gpu_workers + song/lyrics/reference analysis + anchors + QA + working assembly + 4K upscale
```

This is not a guaranteed deadline. Identity-anchor generation, retries, lip-sync correction, output resolution, inference steps, quantization, model loading, storage throughput, GPU type, decode speed, and NVENC availability all affect total time. The FFmpeg CUDA scaler is not neural super-resolution; a command-connected AI upscaler can recover or invent more detail but normally takes longer.

## Storage and recovery

Five-minute jobs create many independent raw attempts and accepted clips. Keep the job root on fast local NVMe while rendering, then upload only final masters and audit files to permanent object storage. Retain accepted clips until delivery QA passes. A resumed job revalidates and reuses them instead of starting over.
