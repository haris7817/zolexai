# ZolexAI Current-State Audit

**Generated:** 21 August 2026
**Repo commit:** `3bd80165da35c52d1f0fd7cd16d5daacf0a7032a`
**Branch:** `main`
**Working tree:** clean (`git status --porcelain` empty, `git diff --stat` empty)
**Audit type:** READ-ONLY extraction. No file in the repository was modified. No GPU was started, no weights downloaded, no inference run, no production system contacted.

> **How to read this report.** It records what the repository *contains*, verbatim, with a `file:line` citation for every value. It makes no judgement about whether any value is correct. Anything that merely looked unusual is recorded without conclusion in **Appendix C**.

---

## 0. Summary of What Exists

### EXTRACTION COVERAGE

```
EXTRACTION COVERAGE
──────────────────────────────────
Workflows traced           6 of 6
Settings extracted         150
Settings with file:line    150 of 150
Effective values resolved  46 of 46
Invocation snapshots       13 of 13
Camera fields documented   1 structured, 32 concepts assessed
Prompt examples captured   7 of 7
Unresolved (Appendix B)    9
```

Counts explained, so the reviewer can check them:

- **Settings extracted 150** — the data rows of the seven tables in §3.1–§3.7. Every one carries a `file:line`, so `150 of 150`.
- **Effective values resolved 46 of 46** — the rows of the §4.1 precedence table. Each resolves to a concrete effective value, including the rows whose effective value is "flag not emitted". Separately, the LTX pipeline's *own* internal defaults behind those un-emitted flags (cfg, stg, a2v, steps) are **UNKNOWN** and are Appendix B.1 — they live in the external LTX repository, not in this checkout.
- **Invocation snapshots 13 of 13** — every command in §5.2–§5.10, fully resolved. §5.1 ("fast preset") is `— not present`, since no preset system exists.
- **Camera fields documented** — exactly one structured field exists in the entire repository (`DirectorEvent.camera`, a free-text `str`). §10.1 assesses 32 named camera concepts against it.
- **Prompt examples captured 7 of 7** — §12.1 through §12.7, covering 5 s T2V, 60 s T2V (both sections), 60 s Director T2V (both sections), 5 s Director, I2V anchored, Music Video (three sections), and Video-to-Video.
- **Unresolved 9** — Appendix B, which is not empty.

### 0.1 Workflows found

Six YAML definitions in [workflow-definitions/](workflow-definitions/), all validated at API startup by [workflow_registry.py:315](apps/api/app/services/workflow_registry.py#L315) `load_registry`:

| id | name | category | output_type | duration_mode | committed `execution.runtime` |
|---|---|---|---|---|---|
| `text-to-video` | Text to Video | video | video | `fixed` | `mock` ([text-to-video.yaml:68](workflow-definitions/text-to-video.yaml#L68)) |
| `image-to-video` | Image to Video | video | video | `fixed` | `mock` ([image-to-video.yaml:62](workflow-definitions/image-to-video.yaml#L62)) |
| `video-to-video` | Video to Video | video | video | `source` | `mock` ([video-to-video.yaml:77](workflow-definitions/video-to-video.yaml#L77)) |
| `extend-video` | Extend Video | video | video | `fixed` | `mock` ([extend-video.yaml:67](workflow-definitions/extend-video.yaml#L67)) |
| `music-video` | Music Video | audio | video | `source` | `mock` ([music-video.yaml:67](workflow-definitions/music-video.yaml#L67)) |
| `music` | Music | audio | audio | `minutes` | `mock` ([music.yaml:60](workflow-definitions/music.yaml#L60)) |

Five of the six are declared supported by the LTX adapter ([ltx.py:1038-1040](apps/worker/worker/adapters/ltx.py#L1038-L1040)):

```python
_SUPPORTED = frozenset(
    {"text-to-video", "image-to-video", "extend-video", "video-to-video", "music-video"}
)
```

`music` is excluded and routes to `adapters/music.py` (ACE-Step over HTTP) — outside this audit's LTX scope except where noted.

**Additional generation modes that are not separate workflows** (they are branches inside the workflows above, selected by keys in the private `execution` block):

| Mode | Selector | Base workflow(s) | Enabled in committed YAML? |
|---|---|---|---|
| Distilled (default tier) | *(absence of any selector)* | all five video workflows | yes — the default |
| Guided tier | `generation_engine: guided` | text-to-video, image-to-video | **no** — commented out at [text-to-video.yaml:93](workflow-definitions/text-to-video.yaml#L93) and [image-to-video.yaml:78](workflow-definitions/image-to-video.yaml#L78) |
| Transform engine (IC-LoRA) | `v2v_engine: transform` | video-to-video | **yes** — [video-to-video.yaml:119](workflow-definitions/video-to-video.yaml#L119) |
| Person lock | `v2v_person_lock: true` | video-to-video | **no** — commented out at [video-to-video.yaml:135](workflow-definitions/video-to-video.yaml#L135) |
| Reference identity | `v2v_reference_identity: true` | video-to-video | **yes** — [video-to-video.yaml:173](workflow-definitions/video-to-video.yaml#L173) |
| Audio-conditioned | `audio_conditioning: true` | music-video | **no** — commented out at [music-video.yaml:98](workflow-definitions/music-video.yaml#L98) |
| Director (Idea) prompt mode | request parameter `prompt_mode: director` | text-to-video, image-to-video | yes — `settings.prompt_modes: true` on both |
| Director continuation | automatic, via `parameters.director_lineage` | extend-video | yes — injected by the API |

### 0.2 Preset names found

**`— not present`.** There is no preset system in this repository.

- Every workflow declares `supported_quality_levels: []` (t2v:39, i2v:34, v2v:54, extend:43, music-video:45, music:30).
- Every video workflow declares `settings.quality: false`, `settings.motion_strength: false`, `settings.prompt_adherence: false`.
- Grepping for `fast`, `balanced`, `quality`, `pro`, `sft` as preset identifiers returns no preset table anywhere in `apps/worker/`, `apps/api/`, `packages/` or `workflow-definitions/`.

The nearest analogue is the **pipeline tier** (`LtxPipeline` instances at [ltx.py:889-995](apps/worker/worker/adapters/ltx.py#L889-L995)) selected by the `execution` keys in §0.1. Section 3.1 documents these in the place a preset block would occupy.

### 0.3 Pipeline invocation method

**Subprocess CLI.** Not a ComfyUI graph, not a Python API, not an HTTP call.

[ltx.py:2643-2650](apps/worker/worker/adapters/ltx.py#L2643-L2650):

```python
def _launcher(self, module: str = _DISTILLED.module) -> list[str]:
    """The argv prefix that reaches the LTX environment.
    ...
    """
    return ["uv", "run", "python", "-m", module]
```

Launched at [ltx.py:2924-2936](apps/worker/worker/adapters/ltx.py#L2924-L2936) via `asyncio.create_subprocess_exec(*cmd, cwd=str(settings.ltx_repo_dir), start_new_session=True)`, stdout+stderr merged and streamed for progress markers.

The worker process itself never imports torch — stated at [ltx.py:5-7](apps/worker/worker/adapters/ltx.py#L5-L7) and [config.py:186-188](apps/worker/worker/core/config.py#L186-L188).

**Four entry-point modules are reachable:**

| Module string | Constant | Defined at |
|---|---|---|
| `ltx_pipelines.distilled` | `_DISTILLED` | [ltx.py:889](apps/worker/worker/adapters/ltx.py#L889) |
| `ltx_pipelines.ic_lora` | `_IC_LORA` | [ltx.py:894](apps/worker/worker/adapters/ltx.py#L894) |
| `ltx_pipelines.a2vid_two_stage` | `_A2VID` | [ltx.py:915](apps/worker/worker/adapters/ltx.py#L915) |
| `ltx_pipelines.ti2vid_two_stages` | `_GUIDED` | [ltx.py:979](apps/worker/worker/adapters/ltx.py#L979) |

Three further subprocess seams exist in the same LTX environment (`cwd=settings.ltx_repo_dir`), all shipped as scripts in this checkout:

| Purpose | Default argv | Defined at |
|---|---|---|
| Person matting | `uv run python <repo>/apps/worker/scripts/person_matte.py` | [config.py:607-620](apps/worker/worker/core/config.py#L607-L620) |
| Identity anchor compositing | `uv run python <repo>/apps/worker/scripts/person_anchor.py` | [config.py:622-628](apps/worker/worker/core/config.py#L622-L628) |
| Director scene planner | `uv run python <repo>/apps/worker/scripts/director_plan.py` | [config.py:634-642](apps/worker/worker/core/config.py#L634-L642) |
| Image-facts describer | `uv run python <repo>/apps/worker/scripts/director_image_facts.py` | [config.py:644-650](apps/worker/worker/core/config.py#L644-L650) |

### 0.4 LTX version strings found anywhere in the repo (verbatim)

Every occurrence of an LTX version token in non-doc, non-test source:

| String | Where |
|---|---|
| `LTX-2.5` | [ltx.py:1](apps/worker/worker/adapters/ltx.py#L1) (module docstring), [registry.py:26](apps/worker/worker/adapters/registry.py#L26), [config.py:300](apps/worker/worker/core/config.py#L300) |
| `LTX 2.5` | [config.py:229](apps/worker/worker/core/config.py#L229), [director/provider.py:4](apps/worker/worker/director/provider.py#L4), [director/compiler.py:3](apps/worker/worker/director/compiler.py#L3) |
| `ltx-2.5` (path segment) | [config.py:192](apps/worker/worker/core/config.py#L192), [config.py:275](apps/worker/worker/core/config.py#L275), [config.py:604](apps/worker/worker/core/config.py#L604) |
| `ltx-2.5` (in checkpoint filenames) | [ltx.py:170-179](apps/worker/worker/adapters/ltx.py#L170-L179), [ltx.py:189-190](apps/worker/worker/adapters/ltx.py#L189-L190) |
| `ltx-2.3` | [ltx.py:196](apps/worker/worker/adapters/ltx.py#L196), [video-to-video.yaml:129](workflow-definitions/video-to-video.yaml#L129) |
| `ltx2-benchmark` (default repo dir) | [config.py:183](apps/worker/worker/core/config.py#L183), [.env.example:79](.env.example#L79) |
| `LTX-2.5's native rate` | [config.py:300](apps/worker/worker/core/config.py#L300) |
| `ACE-Step 1.5 XL` | [registry.py:31](apps/worker/worker/adapters/registry.py#L31), [.env.example:82](.env.example#L82) |

**No semantic version number for the LTX *runtime code* (as opposed to the model) appears anywhere in the repository.** There is no pinned commit, tag, submodule, or lockfile for the LTX pipelines repo. See Appendix B.

### 0.5 Checkpoint filenames referenced (verbatim)

Relative to `settings.ltx_models_root`, which is `ltx_model_dir` or (default) `<ltx_repo_dir>/models/ltx-2.5` ([config.py:602-604](apps/worker/worker/core/config.py#L602-L604)).

**Required for every render** — `_MODEL_FILES`, [ltx.py:169-179](apps/worker/worker/adapters/ltx.py#L169-L179):

```python
_MODEL_FILES: dict[str, str] = {
    "transformer_nvfp4": "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
    "transformer_bf16": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "duration_head": "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    "spatial_upsampler": (
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    ),
}
```

**Required only by non-default tiers** — `_OPTIONAL_MODEL_FILES`, [ltx.py:186-197](apps/worker/worker/adapters/ltx.py#L186-L197):

```python
_OPTIONAL_MODEL_FILES: dict[str, str] = {
    "transformer_dev": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "distilled_lora": "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
    "union_control_lora": "loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
}
```

**Non-LTX checkpoints referenced:**

| Path | Purpose | Defined at |
|---|---|---|
| `<ltx_repo_dir>/models/gemma-4-e2b-it` | Director planner AND LTX prompt-enhancer root (`--prompt-enhancer-gemma-root`) | [config.py:630-632](apps/worker/worker/core/config.py#L630-L632); used at [ltx.py:2783](apps/worker/worker/adapters/ltx.py#L2783) |
| `gemma-4-31b` (hosted, Cerebras) | Lyrics writer and Director planner | [config.py:456-459](apps/worker/worker/core/config.py#L456-L459), [config.py:520-523](apps/worker/worker/core/config.py#L520-L523) |
| BiRefNet (named in comments only, no filename) | Person matting | [video-to-video.yaml:146](workflow-definitions/video-to-video.yaml#L146), [video-to-video.yaml:185](workflow-definitions/video-to-video.yaml#L185) |

### 0.6 The single most consequential precedence fact

Every committed workflow YAML sets `execution.runtime: mock` and `execution.output_content_type: image/png` / `execution.output_kind: image`. [registry.py:26-29](apps/worker/worker/adapters/registry.py#L26-L29) records the same:

```python
# LTX-2.5 on a GPU node (NVFP4, RTX 5090). Registered but not yet routed:
# no shipped workflow says `runtime: ltx`, and only a worker started with
# RUNTIMES including "ltx" will ever claim such a job. See adapters/ltx.py.
"ltx": LtxAdapter(),
```

An API test asserts the consequence directly — [test_worker_protocol.py:553](apps/api/tests/test_worker_protocol.py#L553):

```python
assert await claim(client, worker_headers, worker_id, runtimes=["ltx"]) is None
```

**Therefore: at commit `3bd8016`, no LTX code path is reachable by any job.** Section 4 records this and the machinery that would apply if `runtime` were `ltx`; §4.5 records the divergence.

---

## 1. LTX Inventory

`Component` = worker / API / config / schema / test / docs / frontend / infra.
Rows are the files that participate in the LTX flow. Files whose only LTX content is prose are marked `docs`.

### 1.1 Worker — source

| File | Component | LTX Role | Pipeline(s) | Settings Defined Here | Notes |
|---|---|---|---|---|---|
| [apps/worker/worker/adapters/ltx.py](apps/worker/worker/adapters/ltx.py) | worker | The entire LTX runtime: weight tables, grid/frame tables, four pipeline definitions, five workflow handlers, argv builder, subprocess supervision | distilled, ic_lora, a2vid_two_stage, ti2vid_two_stages | `_MODEL_FILES`, `_OPTIONAL_MODEL_FILES`, `_DIMENSIONS`, `_DEFAULT_DIMENSIONS`, `_GRID_CEILINGS`, `_UNMEASURED_CEILING`, `_CONDITIONED_BANDS`, `_BAD_FRAME_BANDS`, `_MEASURED_SAFE_CONDITIONED`, `_TWO_IMAGE_SAFE_FRAMES`, `_FRAME_LATTICE`, `_PIXEL_BUDGET`, `_MAX_OUTPUT_LONG_SIDE`, `_MAX_OUTPUT_SHORT_SIDE`, `_AUDIO_LANDING_FRAMES`, `_AUDIO_PASS_SECONDS`, `_AUDIO_WINDOW_PAD_SECONDS`, `_TRANSFORM_PASS_SECONDS`, `_GUIDED_PASS_SECONDS`, `_V2V_KEYFRAME_SECONDS`, `_V2V_KEYFRAME_BOUNDS`, `_V2V_STRUCTURE_STRENGTH`, `_V2V_CONTINUITY_STRENGTH`, `_V2V_REFERENCE_STRENGTH`, `_V2V_IDENTITY_ANCHOR_STRENGTH`, `_V2V_IDENTITY_RAW_ANCHOR_STRENGTH`, `_V2V_IDENTITY_REFRESH_STRENGTH`, `_V2V_IDENTITY_SUBJECT_ATTENTION`, `_V2V_CONTROL_STRENGTH`, `_V2V_LORA_STRENGTH`, `_MARKERS`, `_CANCEL_POLL_SECONDS`, `_OUTPUT_TAIL_LINES`, `_DETERMINISTIC_FAILURES`, `_KILL_GRACE_SECONDS` | 3128 lines. The only file that builds an LTX argv. |
| [apps/worker/worker/core/config.py](apps/worker/worker/core/config.py) | config | Every environment-overridable LTX runtime value, plus the four subprocess-seam argv properties | all | `ltx_repo_dir`, `ltx_model_dir`, `ltx_quantization`, `ltx_max_seconds`, `ltx_frame_rate`, `ltx_max_source_seconds`, `ltx_max_extend_source_seconds`, `max_segment_seconds`, `person_matte_command`, `person_anchor_command`, `director_planner_command`, `director_gemma_dir`, `director_planner_timeout_seconds`, `director_vision_enabled`, `director_vision_command`, `director_vision_timeout_seconds`, `job_timeout_seconds`, `ffmpeg_path`, `ffprobe_path`, `runtime`, `runtimes` | Pydantic-settings; `.env` at repo root is optional |
| [apps/worker/worker/adapters/registry.py](apps/worker/worker/adapters/registry.py) | worker | Maps `execution.runtime` → adapter instance. `"ltx"` → `LtxAdapter()` | — | `_ADAPTERS` | Unknown runtime = hard, non-retriable error |
| [apps/worker/worker/adapters/base.py](apps/worker/worker/adapters/base.py) | worker | `AdapterJob` (carries `execution` dict), `execution_int`, `execution_float`, `parse_duration_seconds` | — | — | `execution_int/float` are the *only* readers of tuning keys |
| [apps/worker/worker/workflows/resolver.py](apps/worker/worker/workflows/resolver.py) | worker | Builds `AdapterJob` from the API's claim payload; `resolve_adapter` reads `execution.runtime` | — | — | Worker holds no copy of the registry |
| [apps/worker/worker/jobs/runner.py](apps/worker/worker/jobs/runner.py) | worker | Applies `execution.timeout_seconds` as the adapter deadline ([runner.py:128](apps/worker/worker/jobs/runner.py#L128)); stages inputs to disk before `run` | — | — | `claim.get("execution", {}).get("timeout_seconds") or settings.job_timeout_seconds` |
| [apps/worker/worker/longform/chain.py](apps/worker/worker/longform/chain.py) | worker | `render_chain` — the single long-form mechanism for all five workflows; `plan_chain_segments` | all | — | Pass ceiling is a *parameter*, never a constant here |
| [apps/worker/worker/longform/prompts.py](apps/worker/worker/longform/prompts.py) | worker | `plan_section_prompts` — standard-mode per-section prompt splitting | all | `_SECTION_LINE`, `_TIMED_LINE`, `_PERSISTENT_LINE`, `_DIALOGUE_LINE`, `_SEQUENCE_START`, `_INLINE_SEQUENCE`, `_SENTENCE_BOUNDARY` | Deterministic, non-LLM |
| [apps/worker/worker/longform/enhance.py](apps/worker/worker/longform/enhance.py) | worker | `structure_prompt` — appends derived CONTINUITY rules when `execution.prompt_structuring` | all | `_COLOURS`, `_NUMBER_WORDS`, `_COLOUR_PATTERN`, `_COUNT_PATTERN`, `_NOT_NOUNS`, `_ALREADY_STRUCTURED` | Never rewrites the user's text |
| [apps/worker/worker/longform/timing.py](apps/worker/worker/longform/timing.py) | worker | `plan_musical_boundaries` — music-video cut points from onsets | distilled, a2vid | `_MAX_PULL_FRACTION` = 0.2, `_MIN_WINDOW_SECONDS` = 2.0 | Only ever pulls a cut *earlier* |
| [apps/worker/worker/longform/progress.py](apps/worker/worker/longform/progress.py) | worker | `StageReporter`, `band_for` — progress banding across chained passes | all | `GENERATE_FROM`, `GENERATE_TO` | |
| [apps/worker/worker/media/control.py](apps/worker/worker/media/control.py) | worker | `extract_edge_control` — canny control clip for IC-LoRA | ic_lora | `DEFAULT_EDGE_LOW` = 0.1, `DEFAULT_EDGE_HIGH` = 0.4 | ffmpeg `edgedetect` |
| [apps/worker/worker/media/masks.py](apps/worker/worker/media/masks.py) | worker | `build_person_matte`, `build_identity_anchor`, `build_attention_mask`, `build_hybrid_control`, `extract_source_window` — all subprocesses into `ltx_repo_dir` | ic_lora | `DILATION_PASSES` = 2, `FEATHER_RADIUS` = 12, `BACKGROUND_ATTENTION` = 0.5 | |
| [apps/worker/worker/media/frames.py](apps/worker/worker/media/frames.py) | worker | `extract_final_frame` (seam frames), `extract_frames_at` (v2v keyframes), `normalize_clip` | all | `_VIDEO_ARGS`, `_AUDIO_ARGS` | |
| [apps/worker/worker/media/segments.py](apps/worker/worker/media/segments.py) | worker | `plan_segments` (the duration→sections maths), `concat_segments`, `verify_duration` | all | — | Even windows, never greedy |
| [apps/worker/worker/media/audio.py](apps/worker/worker/media/audio.py) | worker | `AudioMode` enum, `mux_audio` (final soundtrack), `audio_onsets`, `detect_onsets` | all | `_AAC_ARGS`, `_PAD_TOLERANCE_SECONDS` = 0.05, `_MAX_PAD_SECONDS` = 3.0 | |
| [apps/worker/worker/media/probe.py](apps/worker/worker/media/probe.py) | worker | `MediaInfo`, `probe_media` — the source duration that drives `duration_mode: source` | all | — | |
| [apps/worker/worker/media/validate.py](apps/worker/worker/media/validate.py) | worker | `OutputExpectation`, `verify_output`, `duration_tolerance` | all | — | Single exit gate on every workflow |
| [apps/worker/worker/media/ffmpeg.py](apps/worker/worker/media/ffmpeg.py) | worker | `ffmpeg`, `ffprobe_json` subprocess wrappers | all | — | |
| [apps/worker/worker/director/provider.py](apps/worker/worker/director/provider.py) | worker | Director planning brief (system + user prompt), `GemmaDirectorProvider`, `create_director_plan`, `wants_director`, `continuation_lineage` | all (prompt-side only) | `DIALOGUE_LANGUAGES`, `_SYSTEM_PROMPT`, `_ANCHORED_RULES`, `_CONTINUATION_RULES`, `_DIRECTOR_WORKFLOWS`, `_PLAN_BEGIN`, `_PLAN_END` | Runs the local Gemma via `ltx_repo_dir` |
| [apps/worker/worker/director/plan.py](apps/worker/worker/director/plan.py) | worker | `DirectorPlan` schema + every deterministic validation rule | — | `WORDS_PER_SECOND` = 2.0, `ESTABLISH_SECONDS` = 2.5, `_BUDGET_SLACK` = 1.15, `TARGET_SECONDS_PER_LINE` = 4.0, `_MINIMUM_LINES` = 2, `MAX_SILENT_GAP` = 6.0, `MAX_CHARACTERS` = 4, `_MAX_EVENTS` = 24, `ANCHORED_SCENE`, `_MIN_DISTINCTIVE` = 4 | The camera field lives here |
| [apps/worker/worker/director/compiler.py](apps/worker/worker/director/compiler.py) | worker | `compile_section_prompts` — DirectorPlan → the caption text sent as `--prompt` | all | `_TRANSITIONS`, `_PAUSE_TRANSITIONS`, `_MOVE_VERBS`, `_QUANTIFIER` | The camera compiler |
| [apps/worker/worker/director/vision.py](apps/worker/worker/director/vision.py) | worker | `source_image_facts` (I2V grounding), `reference_person_facts` (v2v identity caption) — both subprocesses into `ltx_repo_dir` | — | `_SYSTEM_PROMPT`, `_USER_PROMPT`, `_IDENTITY_SYSTEM_PROMPT`, `_IDENTITY_USER_PROMPT`, `_MAX_FACTS_CHARS` = 900, `_MAX_IDENTITY_CHARS` = 350 | |
| [apps/worker/worker/director/cerebras.py](apps/worker/worker/director/cerebras.py) | worker | Hosted Director planner (tried before local Gemma) | — | — | Not LTX; shares the brief |
| [apps/worker/worker/director/__init__.py](apps/worker/worker/director/__init__.py) | worker | Public surface of the Director package | — | — | |
| [apps/worker/worker/music/acestep.py](apps/worker/worker/music/acestep.py) | worker | Mentions LTX only to contrast model-load cost | — | — | `docs`-grade reference |

### 1.2 Worker — scripts (all run inside the LTX environment)

| File | Component | LTX Role | Pipeline(s) | Settings Defined Here | Notes |
|---|---|---|---|---|---|
| [apps/worker/scripts/ltx_smoke.py](apps/worker/scripts/ltx_smoke.py) | infra | One-shot GPU smoke render | distilled | — | Not called by the adapter |
| [apps/worker/scripts/ltx_matrix.sh](apps/worker/scripts/ltx_matrix.sh) | infra | Grid × frame-count measurement sweep that produced `_GRID_CEILINGS` | distilled | — | |
| [apps/worker/scripts/ltx_fixtures.sh](apps/worker/scripts/ltx_fixtures.sh) | infra | Renders fixture clips | distilled | — | |
| [apps/worker/scripts/frame_probe.py](apps/worker/scripts/frame_probe.py) | infra | Probes individual frame counts for decoder survival | distilled | — | Named in [ltx.py:307](apps/worker/worker/adapters/ltx.py#L307) as the way to narrow `_CONDITIONED_BANDS` |
| [apps/worker/scripts/v2v_sweep.sh](apps/worker/scripts/v2v_sweep.sh) | infra | Sweeps `v2v_structure_strength` | distilled | — | Named at [ltx.py:577](apps/worker/worker/adapters/ltx.py#L577) |
| [apps/worker/scripts/v2v_identity_matrix.sh](apps/worker/scripts/v2v_identity_matrix.sh) | infra | Sweeps the identity strengths | ic_lora | — | Named at [video-to-video.yaml:220](workflow-definitions/video-to-video.yaml#L220) |
| [apps/worker/scripts/person_matte.py](apps/worker/scripts/person_matte.py) | worker | The matting CLI invoked by `build_person_matte` | ic_lora | — | Default `person_matte_argv` target |
| [apps/worker/scripts/person_anchor.py](apps/worker/scripts/person_anchor.py) | worker | The composited-anchor CLI invoked by `build_identity_anchor` | ic_lora | — | Default `person_anchor_argv` target |
| [apps/worker/scripts/director_plan.py](apps/worker/scripts/director_plan.py) | worker | The local Gemma planning CLI | — | — | Default `director_planner_argv` target |
| [apps/worker/scripts/director_image_facts.py](apps/worker/scripts/director_image_facts.py) | worker | The image-describer CLI | — | — | Default `director_vision_argv` target |
| [apps/worker/scripts/coverage_gaps.sh](apps/worker/scripts/coverage_gaps.sh) | infra | Test-coverage helper naming LTX modules | — | — | |
| [apps/worker/scripts/av_offset_probe.py](apps/worker/scripts/av_offset_probe.py) | infra | Measures A/V offset in a rendered file | — | — | |
| [apps/worker/scripts/drift_check.sh](apps/worker/scripts/drift_check.sh) | infra | Drift helper | — | — | |

### 1.3 Workflow definitions

| File | Component | LTX Role | Pipeline(s) | Settings Defined Here | Notes |
|---|---|---|---|---|---|
| [workflow-definitions/text-to-video.yaml](workflow-definitions/text-to-video.yaml) | config | T2V product contract + private execution block | distilled (guided commented out) | `runtime`, `max_segment_seconds`, `prompt_structuring`, `output_content_type`, `output_kind`; commented: `generation_engine`, `guided_pass_seconds` | |
| [workflow-definitions/image-to-video.yaml](workflow-definitions/image-to-video.yaml) | config | I2V | distilled (guided commented out) | `runtime`, `max_segment_seconds`, `prompt_structuring`, `output_content_type`, `output_kind`; commented: `generation_engine`, `guided_pass_seconds` | |
| [workflow-definitions/video-to-video.yaml](workflow-definitions/video-to-video.yaml) | config | V2V, the only YAML that enables a non-default engine | ic_lora | `runtime`, `timeout_seconds`, `v2v_engine`, `v2v_reference_identity`, `output_content_type`, `output_kind`; commented: `v2v_keyframe_seconds`, `v2v_keyframes`, `v2v_structure_strength`, `v2v_continuity_strength`, `v2v_reference_strength`, `v2v_control_strength`, `v2v_lora_strength`, `v2v_edge_low`, `v2v_edge_high`, `v2v_person_lock`, `v2v_background_attention`, `v2v_identity_describe_reference`, `v2v_identity_anchor_strength`, `v2v_identity_refresh_strength`, `v2v_identity_subject_attention` | |
| [workflow-definitions/extend-video.yaml](workflow-definitions/extend-video.yaml) | config | Extend | distilled | `runtime`, `prompt_structuring`, `timeout_seconds`, `output_content_type`, `output_kind` | No `max_segment_seconds` |
| [workflow-definitions/music-video.yaml](workflow-definitions/music-video.yaml) | config | Music Video | distilled (a2vid commented out) | `runtime`, `prompt_structuring`, `timeout_seconds`, `align_cuts_to_audio`, `output_content_type`, `output_kind`; commented: `audio_conditioning`, `audio_pass_seconds`, `inference_steps`, `a2v_guidance_scale`, `guidance_scale`, `stg_scale` | |
| [workflow-definitions/music.yaml](workflow-definitions/music.yaml) | config | Music (non-LTX) | — | `runtime`, `timeout_seconds`, `output_content_type`, `output_kind` | LTX adapter refuses it |

### 1.4 API

| File | Component | LTX Role | Pipeline(s) | Settings Defined Here | Notes |
|---|---|---|---|---|---|
| [apps/api/app/schemas/workflow.py](apps/api/app/schemas/workflow.py) | schema | `ExecutionSpec` (`extra="allow"`), `WorkflowDefinition`, `WorkflowPublic` projection | — | `runtime` default `"mock"` ([:138](apps/api/app/schemas/workflow.py#L138)), `output_content_type`, `output_kind`, `timeout_seconds` | Every other `execution` key is untyped passthrough |
| [apps/api/app/services/workflow_registry.py](apps/api/app/services/workflow_registry.py) | API | Loads/validates YAML, `validate_request`, `ids_for_runtimes` | — | `PROMPT_MODES`, `DIALOGUE_LANGUAGES` | Rejects parameters a workflow does not declare |
| [apps/api/app/services/generation.py](apps/api/app/services/generation.py) | API | Creates jobs; builds `director_lineage` and injects `identity_image` input | — | — | [:151-158](apps/api/app/services/generation.py#L151-L158) |
| [apps/api/app/schemas/generation.py](apps/api/app/schemas/generation.py) | schema | `GenerationParameters` — the complete public parameter surface | — | `duration`, `aspect_ratio`, `quality`, `motion_strength`, `prompt_adherence`, `seed`, `lyrics`, `lyrics_language`, `prompt_mode`, `dialogue_language` | `extra="forbid"` |
| [apps/api/app/schemas/internal.py](apps/api/app/schemas/internal.py) | schema | The claim payload — the one place `execution` crosses a process boundary | — | `execution: dict[str, Any]` ([:104](apps/api/app/schemas/internal.py#L104)) | |
| [apps/api/app/services/queue.py](apps/api/app/services/queue.py) | API | Claim-time runtime intersection | — | — | |
| [apps/api/app/core/config.py](apps/api/app/core/config.py) | config | API-side settings (`JOB_LEASE_SECONDS`, `JOB_MAX_ATTEMPTS`) | — | — | No LTX values |

### 1.5 Tests

| File | Component | LTX Role | Notes |
|---|---|---|---|
| [apps/worker/tests/test_ltx.py](apps/worker/tests/test_ltx.py) | test | 63 tests: argv pinning, prompt verbatim, seeds, grids, frame tables, chaining, cancellation, kill semantics | The primary settings pin |
| [apps/worker/tests/test_guided.py](apps/worker/tests/test_guided.py) | test | 12 tests: guided tier argv and landings | |
| [apps/worker/tests/test_transform.py](apps/worker/tests/test_transform.py) | test | 15 tests: IC-LoRA argv, control, `--skip-stage-2` | |
| [apps/worker/tests/test_music_video_audio.py](apps/worker/tests/test_music_video_audio.py) | test | 12 tests: a2vid argv and audio windows | |
| [apps/worker/tests/test_music_video.py](apps/worker/tests/test_music_video.py) | test | 13 tests: default music-video path, muxing | |
| [apps/worker/tests/test_video_to_video.py](apps/worker/tests/test_video_to_video.py) | test | 19 tests: restyle conditioning, source audio | |
| [apps/worker/tests/test_reference_identity.py](apps/worker/tests/test_reference_identity.py) | test | 16 tests: identity anchor/refresh/attention | |
| [apps/worker/tests/test_person_lock.py](apps/worker/tests/test_person_lock.py) | test | 10 tests | |
| [apps/worker/tests/test_person_anchor.py](apps/worker/tests/test_person_anchor.py) | test | 12 tests | |
| [apps/worker/tests/test_longform.py](apps/worker/tests/test_longform.py) | test | 24 tests: `plan_segments`, `render_chain` | |
| [apps/worker/tests/test_seam_timing.py](apps/worker/tests/test_seam_timing.py) | test | 5 tests: `_planned_section_frames` drift | |
| [apps/worker/tests/test_prompt_structuring.py](apps/worker/tests/test_prompt_structuring.py) | test | 15 tests: `structure_prompt` | |
| [apps/worker/tests/test_director*.py](apps/worker/tests/) | test | 90 tests across 4 files | |
| [apps/worker/tests/test_media*.py](apps/worker/tests/) | test | 46 tests | |
| [apps/worker/tests/conftest.py](apps/worker/tests/conftest.py) | test | `fake_models` fixture creating the weight tree | |
| [apps/api/tests/test_workflows.py](apps/api/tests/test_workflows.py) | test | Asserts `"ltx"` never leaks into a public response ([:58](apps/api/tests/test_workflows.py#L58)) | |
| [apps/api/tests/test_worker_protocol.py](apps/api/tests/test_worker_protocol.py) | test | Asserts an `ltx`-only worker claims nothing ([:553](apps/api/tests/test_worker_protocol.py#L553)) | |

### 1.6 Frontend

| File | Component | LTX Role | Notes |
|---|---|---|---|
| [apps/web/scripts/qa-catalog-parity.mjs](apps/web/scripts/qa-catalog-parity.mjs) | frontend | Leak check: asserts `"ltx"` absent from the catalog ([:125](apps/web/scripts/qa-catalog-parity.mjs#L125)) | |
| [apps/web/scripts/qa-e2e.mjs](apps/web/scripts/qa-e2e.mjs) | frontend | Leak check ([:228](apps/web/scripts/qa-e2e.mjs#L228)) | |

**`grep -rn 'ltx' apps/api/app apps/web/src packages` returns zero matches.** The frontend and the API application code contain no LTX reference at all, and set no generation parameter beyond the ten fields of `GenerationParameters` (§1.4).

### 1.7 Config / infra

| File | Component | LTX Role | Settings Defined Here |
|---|---|---|---|
| [.env.example](.env.example) | config | Documents the GPU-node variables, all commented out | `RUNTIMES=ltx`, `LTX_REPO_DIR=/workspace/ltx2-benchmark`, `LTX_MAX_SECONDS=30`, `MUSIC_*`, `CEREBRAS_*` ([.env.example:71-114](.env.example#L71-L114)) |
| `.env` (local, untracked) | config | Sets **no** LTX variable. Variable names present: `API_BASE_URL API_PORT API_URL APP_ENV APP_URL CEREBRAS_AI_MODEL CEREBRAS_API_KEY CORS_ORIGINS DATABASE_URL JOB_LEASE_SECONDS JOB_MAX_ATTEMPTS LOG_FORMAT LOG_LEVEL MINIO_CONSOLE_PORT MINIO_PORT NEXT_PUBLIC_API_URL POSTGRES_DB POSTGRES_PASSWORD POSTGRES_PORT POSTGRES_USER REDIS_PORT REDIS_URL STORAGE_ACCESS_KEY STORAGE_BUCKET STORAGE_ENDPOINT STORAGE_PROVIDER STORAGE_PUBLIC_ENDPOINT STORAGE_REGION STORAGE_SECRET_KEY WEB_PORT WORKER_API_TOKEN WORKER_NAME` | Values not recorded (secrets policy). Only `WORKER_NAME=mock-worker-1` is non-secret and LTX-adjacent. |
| [apps/worker/Dockerfile](apps/worker/Dockerfile) | infra | Worker image | — |
| [infrastructure/compose/](infrastructure/compose/) | infra | Compose stack | — |

### 1.8 Docs (prose only, no executable setting)

[docs/M1-REPORT.md](docs/M1-REPORT.md), [docs/PRE-M1-HANDOFF.md](docs/PRE-M1-HANDOFF.md), [docs/delivery-tracker.md](docs/delivery-tracker.md), [docs/generation-limits.md](docs/generation-limits.md), [docs/milestones.md](docs/milestones.md), [docs/internal/README.md](docs/internal/README.md), [docs/internal/architecture-audit-2026-08-16.md](docs/internal/architecture-audit-2026-08-16.md), [docs/internal/gpu-worker-runbook.md](docs/internal/gpu-worker-runbook.md), [docs/internal/issue-triton-na-kernel.md](docs/internal/issue-triton-na-kernel.md), [docs/internal/ltx-2.5-licensing-review.md](docs/internal/ltx-2.5-licensing-review.md), [docs/internal/next-steps-2026-08-15.md](docs/internal/next-steps-2026-08-15.md), [docs/internal/production-runbook.md](docs/internal/production-runbook.md), and seven `research-2026-08-*` notes.
---

## 2. Workflow Traces

All five video workflows share one entry path down to the adapter. The shared prefix is stated once, then each workflow's divergence follows.

### 2.0 The shared prefix (identical for every workflow)

```
POST /api/v1/generations
  ↓  apps/api/app/api/v1/generations.py  (route)
GenerationCreateRequest  (extra="forbid")
  ↓  apps/api/app/schemas/generation.py:59
WorkflowRegistry.validate_request(...)
  ↓  apps/api/app/services/workflow_registry.py:101
GenerationService.create(...)          ← writes the job row, stores parameters
  ↓  apps/api/app/services/generation.py:100
[queue]  PostgreSQL job table + optional Redis wakeup
  ↓
POST /api/v1/internal/jobs/claim       ← worker claims, sending its `runtimes`
  ↓  apps/api/app/services/queue.py  (intersects workflow.execution.runtime
     with the worker's declared runtimes)
Claim payload  (JobClaim, apps/api/app/schemas/internal.py:104)
   carries: job_id, workflow_id, workflow_version, prompt, parameters,
            inputs[{role,kind,content_type,download_url}], execution{...},
            output_content_type
  ↓
build_adapter_job(claim)               ← apps/worker/worker/workflows/resolver.py:20
  ↓
resolve_adapter(job) → _ADAPTERS[execution["runtime"]]
  ↓  apps/worker/worker/workflows/resolver.py:50
JobRunner._execute                     ← apps/worker/worker/jobs/runner.py
   deadline = execution["timeout_seconds"] or settings.job_timeout_seconds  (:128)
   _stage_inputs → downloads each presigned input to job.workspace  (:185)
  ↓
LtxAdapter.run(job, on_progress)       ← apps/worker/worker/adapters/ltx.py:1045
```

**Inside `LtxAdapter.run`** ([ltx.py:1045-1073](apps/worker/worker/adapters/ltx.py#L1045-L1073)):

| Stage | File | Class/Function | Line | Responsibility |
|---|---|---|---|---|
| Reporter | `worker/longform/progress.py` | `StageReporter.__init__` | 45 | Owns customer-facing progress copy |
| Weight preflight | `worker/adapters/ltx.py` | `LtxAdapter._require_models` | 2253 | Refuses non-retriably if any of the five shared weights + the quantization-selected transformer is absent |
| Prompt structuring | `worker/longform/enhance.py` | `structure_prompt` | 82 | Only when `execution.prompt_structuring` is truthy **and** `wants_director(job)` is false ([ltx.py:1061-1062](apps/worker/worker/adapters/ltx.py#L1061-L1062)) |
| Dispatch | `worker/adapters/ltx.py` | `LtxAdapter.run` handler dict | 1067-1073 | Keyed on `job.workflow_id`, never on which inputs are present |

```python
handlers = {
    "extend-video": self._run_extension,
    "video-to-video": self._run_restyle,
    "music-video": self._run_music_video,
}
handler = handlers.get(job.workflow_id, self._run_generation)
```

`text-to-video` and `image-to-video` both fall through to `_run_generation`.

---

### 2.1 Text-to-Video

```
Entry point (API route)          POST /api/v1/generations
  ↓
Request schema + validation      GenerationParameters / validate_request
  ↓
Workflow selection logic         handlers.get("text-to-video") → _run_generation
  ↓
Preset resolution                — not present  (no preset system, §0.2)
  ↓
Prompt enhancer                  structure_prompt(job.prompt)   [deterministic, not an LLM]
                                 LTX's own --enhance-prompt: present but OFF
                                 (no YAML sets execution.enhance_prompt)
  ↓
Director                         create_director_plan(job, seconds)   [only if prompt_mode=director]
  ↓
Global plan                      DirectorPlan  (one, before any section renders)
  ↓
Section planner                  plan_chain_segments(seconds, per_pass, None)
  ↓
Camera / scene planner           — not present as a separate stage.
                                 Camera exists only as DirectorEvent.camera inside the plan.
  ↓
Prompt compiler                  compile_section_prompts (Director) OR plan_section_prompts (standard)
  ↓
Pipeline selection               _GUIDED if execution.generation_engine == "guided" else _DISTILLED
  ↓
Argument builder                 LtxAdapter._command
  ↓
Model invocation                 LtxAdapter._execute → asyncio.create_subprocess_exec
  ↓
Output handling                  _trim_to (if nudged) → verify_output(expect_audio=True)
  ↓
Continuation state               extract_final_frame → next step.previous_frame
  ↓
Stitching                        _assemble_generated_sections → normalize_clip → concat_segments
  ↓
Final asset                      _assemble → verify_output → _video_result (video/mp4)
```

| Stage | File | Class/Function | Line | Responsibility |
|---|---|---|---|---|
| Handler | `worker/adapters/ltx.py` | `LtxAdapter._run_generation` | 1077 | Whole workflow |
| Shape guard | `worker/adapters/ltx.py` | `_require_generation_shape` | 2102 | Refuses `output_kind: audio`; refuses any input role other than `source_image` |
| Duration | `worker/adapters/ltx.py` | `_requested_seconds` | 2265 | `parse_duration_seconds(job.parameters["duration"])` |
| Conditioning image | `worker/adapters/ltx.py` | `_conditioning_image(job, "source_image")` | 2217 | Returns `None` for T2V |
| Grid | `worker/adapters/ltx.py` | `_requested_dimensions` | 2286 | `_DIMENSIONS[aspect_ratio]`, default `(1024, 576)` |
| Engine switch | `worker/adapters/ltx.py` | `_run_generation` (`guided = ...`) | 1090 | `job.execution.get("generation_engine") == "guided"` |
| Audio mode log | `worker/adapters/ltx.py` | `_record_audio_mode` | 2094 | `GENERATED_PER_SECTION_AUDIO` |
| Director | `worker/director/provider.py` | `wants_director` / `create_director_plan` | 424 / 486 | |
| Per-pass ceiling | `worker/adapters/ltx.py` | `_guided_pass_seconds` / `_per_pass_seconds` | 2405 / 2291 | |
| Chain | `worker/longform/chain.py` | `render_chain` | 106 | |
| Section plan | `worker/longform/chain.py` | `plan_chain_segments` → `plan_segments` | 176 / `segments.py:64` | |
| Prompt per section | `worker/longform/prompts.py` | `plan_section_prompts` | 75 | |
| Renderer | `worker/adapters/ltx.py` | `LtxAdapter._renderer.render` | 2500 | Frame-count substitution, logging, trim |
| Argv | `worker/adapters/ltx.py` | `LtxAdapter._command` | 2652 | |
| Subprocess | `worker/adapters/ltx.py` | `LtxAdapter._execute` | 2898 | |
| Seam frame | `worker/longform/chain.py` | `_final_frame` → `extract_final_frame` | 218 / `frames.py:33` | |
| Stitch | `worker/adapters/ltx.py` | `_assemble_generated_sections` | 2011 | |
| Verify | `worker/adapters/ltx.py` | `_assemble` → `verify_output` | 1986 | `expect_video=True, expect_audio=True, expected_seconds=seconds` |
| Result | `worker/adapters/ltx.py` | `_video_result` | 3120 | `content_type="video/mp4", kind="video"` |

**Conditioning built for T2V** ([ltx.py:1127-1145](apps/worker/worker/adapters/ltx.py#L1127-L1145)):

```python
def conditioning(step: ChainStep) -> list[ConditioningFrame]:
    if step.is_first:
        return [ConditioningFrame(still, 0, 1.0)] if still else []
    items: list[ConditioningFrame] = []
    if step.previous_frame:
        items.append(ConditioningFrame(step.previous_frame, 0, 1.0))
    if still:
        anchor = self._identity_anchor(job, still, step.seconds)
        if anchor is not None:
            items.append(anchor)
    return items
```

For T2V `still is None`, so section 1 gets **no `--image`** and later sections get exactly one: the predecessor's final frame at index 0, strength `1.0`.

---

### 2.2 Image-to-Video

Identical chain to §2.1 — the *same function*, `_run_generation`. The three differences:

1. `still = await self._conditioning_image(job, "source_image")` returns a real path ([ltx.py:1083](apps/worker/worker/adapters/ltx.py#L1083)).
2. Section 1 carries `--image <still> 0 1.0`.
3. Sections 2+ carry the seam frame at index 0 strength 1.0 **plus**, conditionally, the original upload as a second image (`_identity_anchor`).
4. Director mode is `source_anchored=True` ([provider.py:440-446](apps/worker/worker/director/provider.py#L440-L446)), which appends `_ANCHORED_RULES` to the planning brief and enables `_ground_visual_claims`.

`_identity_anchor` ([ltx.py:2432-2466](apps/worker/worker/adapters/ltx.py#L2432-L2466)) verbatim:

```python
def _identity_anchor(
    self, job: AdapterJob, still: Path, pass_seconds: float
) -> ConditioningFrame | None:
    frames = self._frame_count(pass_seconds)
    reference_frame = min(frames - 1, max(1, frames // 3))
    strength = job.execution_float("i2v_reference_strength", 0.2)
    if strength <= 0 or reference_frame <= 0:
        return None
    if frames not in _TWO_IMAGE_SAFE_FRAMES:
        logger.info(
            "identity_anchor_skipped",
            extra={
                "workflow_id": job.workflow_id,
                "frames": frames,
                "reason": "count not measured safe for a second "
                "conditioning image",
            },
        )
        return None
    return ConditioningFrame(still, reference_frame, strength)
```

---

### 2.3 Video-to-Video

```
Entry point                      POST /api/v1/generations   (duration REJECTED — duration_mode: source)
  ↓
Request schema + validation      validate_request rejects a supplied `duration`
                                 (workflow_registry.py:141-156)
  ↓
Workflow selection logic         handlers["video-to-video"] → _run_restyle
  ↓
Preset resolution                — not present
  ↓
Prompt enhancer                  structure_prompt — NOT applied
                                 (video-to-video.yaml has no `prompt_structuring` key)
  ↓
Director                         — not present  (video-to-video is not in _DIRECTOR_WORKFLOWS)
  ↓
Global plan                      — not present
  ↓
Section planner                  plan_chain_segments(source_duration, per_pass, None)
  ↓
Camera / scene planner           — not present
  ↓
Prompt compiler                  — not present. prompt_for_step is NOT passed to _renderer,
                                 so _command falls back to `job.prompt` verbatim
                                 (ltx.py:2723) for EVERY section.
  ↓
Pipeline selection               _IC_LORA   (execution.v2v_engine == "transform")
                                 else _DISTILLED (still-conditioned restyle)
  ↓
Argument builder                 _command(..., pipeline=_IC_LORA, loras=(lora,),
                                          control=..., mask=...)
  ↓
Model invocation                 _execute
  ↓
Output handling                  _trim_to → (no per-section verify_output; require_audio=False)
  ↓
Continuation state               previous pass's final frame at strength v2v_continuity_strength
  ↓
Stitching                        _deliver_restyle → _assemble_generated_sections(audio=False)
  ↓
Final asset                      mux_audio(picture, staged_source) → verify_output
```

| Stage | File | Class/Function | Line | Responsibility |
|---|---|---|---|---|
| Handler | `worker/adapters/ltx.py` | `_run_restyle` | 1359 | Probes source, chooses engine |
| Source staging + probe | `worker/adapters/ltx.py` | `_staged_source(role="source_video", kind="video")` | 2131 | Enforces `settings.ltx_max_source_seconds` |
| Reference image | `worker/adapters/ltx.py` | `_conditioning_image(job, "reference_image")` | 2217 | Optional |
| Grid | `worker/adapters/ltx.py` | `grid_for_source(source.width, source.height)` | 658 | Source aspect, not requested aspect |
| Engine branch | `worker/adapters/ltx.py` | `_run_restyle` | 1382-1385 | `str(job.execution.get("v2v_engine") or "").strip() == "transform"` |
| Refusal | `worker/adapters/ltx.py` | `_run_restyle` | 1387-1401 | `v2v_reference_identity` on a non-transform engine → non-retriable `AdapterError` |
| Transform handler | `worker/adapters/ltx.py` | `_run_transform` | 1477 | |
| Mutual exclusion | `worker/adapters/ltx.py` | `_run_transform` | 1522-1531 | `v2v_person_lock` + `v2v_reference_identity` → refusal |
| Reference caption | `worker/director/vision.py` | `reference_person_facts` | 111 | Appended after the user's prompt ([ltx.py:1552-1561](apps/worker/worker/adapters/ltx.py#L1552-L1561)) |
| Composited anchor | `worker/media/masks.py` | `build_identity_anchor` | 146 | Subprocess to `scripts/person_anchor.py` |
| Control clip | `worker/media/control.py` | `extract_edge_control` | 49 | ffmpeg `edgedetect` |
| Attention mask | `worker/media/masks.py` | `build_person_matte` → `build_attention_mask` | 70 / — | |
| Per-pass ceiling | `worker/adapters/ltx.py` | `_run_transform` | 1725-1730 | `min(_per_pass_seconds(grid), transform_pass_seconds)` |
| Delivery | `worker/adapters/ltx.py` | `_deliver_restyle` | 1752 | |
| Section frame pinning | `worker/adapters/ltx.py` | `_planned_section_frames` | 2049 | |
| Audio restore | `worker/media/audio.py` | `mux_audio` | 59 | The source's own track |

---

### 2.4 Extend / continuation

```
Entry point                      POST /api/v1/generations  (duration = the EXTENSION's length)
  ↓
Request schema + validation      validate_request; the API additionally walks
                                 source asset → producing job and injects
                                 parameters["director_lineage"] and, when present,
                                 inputs["identity_image"]
                                 (generation.py:151-158, _director_lineage at :201)
  ↓
Workflow selection logic         handlers["extend-video"] → _run_extension
  ↓
Preset resolution                — not present
  ↓
Prompt enhancer                  structure_prompt  (extend-video.yaml:71 prompt_structuring: true)
  ↓
Director                         create_director_plan(job, extension_seconds, lineage=lineage)
                                 ONLY when continuation_lineage(job) is not None
  ↓
Global plan                      DirectorPlan with prior_idea / prior_seconds set
  ↓
Section planner                  plan_chain_segments(extension_seconds, per_pass, None)
  ↓
Camera / scene planner           — not present
  ↓
Prompt compiler                  compile_section_prompts (lineage) OR plan_section_prompts
  ↓
Pipeline selection               _DISTILLED  (always — no engine switch on this workflow)
  ↓
Argument builder                 _command
  ↓
Model invocation                 _execute
  ↓
Output handling                  _trim_to
  ↓
Continuation state               seed_frame = source's final frame; then each pass's own final frame
  ↓
Stitching                        normalize_clip(source) + continuation → concat_segments
  ↓
Final asset                      verify_output(expected = source_duration + extension_seconds)
```

| Stage | File | Class/Function | Line | Responsibility |
|---|---|---|---|---|
| Handler | `worker/adapters/ltx.py` | `_run_extension` | 1185 | |
| Source ceiling | `worker/adapters/ltx.py` | `_staged_source(..., limit_seconds=settings.ltx_max_extend_source_seconds)` | 1198-1203 | 1800.0 s |
| Lineage | `worker/director/provider.py` | `continuation_lineage` | 467 | Reads `parameters["director_lineage"]` |
| Seed frame | `worker/adapters/ltx.py` | `_final_frame_of` | 2241 | |
| Identity image | `worker/adapters/ltx.py` | `_conditioning_image(job, "identity_image")` | 1257 | Only on Director-lineage extensions of I2V |
| Grid | `worker/adapters/ltx.py` | `grid_for_source(source.width, source.height)` | 1273 | |
| Director pass clamp | `worker/adapters/ltx.py` | `_run_extension` | 1276-1283 | `per_pass = min(per_pass, 30.0)` when a continuation plan exists |
| Delivery fps | `worker/adapters/ltx.py` | `_delivery_fps` | 3109 | `min(60, max(10, source.fps or 24))` |
| Audio mode | `worker/adapters/ltx.py` | `_record_audio_mode` | 1305-1314 | `SOURCE_AUDIO` / `GENERATED_PER_SECTION_AUDIO` / `NO_AUDIO` |

---

### 2.5 Music Video

```
Entry point                      POST /api/v1/generations  (duration REJECTED — duration_mode: source)
  ↓
Request schema + validation      validate_request
  ↓
Workflow selection logic         handlers["music-video"] → _run_music_video
  ↓
Preset resolution                — not present
  ↓
Prompt enhancer                  structure_prompt  (music-video.yaml:71 prompt_structuring: true)
  ↓
Director                         — not present  (music-video not in _DIRECTOR_WORKFLOWS)
  ↓
Global plan                      — not present
  ↓
Section planner                  _musical_boundaries → plan_musical_boundaries → plan_chain_segments
  ↓
Camera / scene planner           — not present
  ↓
Prompt compiler                  plan_section_prompts(job.prompt, total, total_seconds=track_seconds)
  ↓
Pipeline selection               _A2VID if execution.audio_conditioning else _DISTILLED
  ↓
Argument builder                 _command(..., audio=AudioConditioning(...) or None)
  ↓
Model invocation                 _execute
  ↓
Output handling                  _trim_to
  ↓
Continuation state               previous pass's final frame at strength 1.0
  ↓
Stitching                        _assemble_generated_sections(audio=False, section_frames=...)
  ↓
Final asset                      mux_audio(picture, the whole uploaded track) → verify_output
```

| Stage | File | Class/Function | Line | Responsibility |
|---|---|---|---|---|
| Handler | `worker/adapters/ltx.py` | `_run_music_video` | 1832 | |
| Track staging + probe | `worker/adapters/ltx.py` | `_staged_source(role="source_audio", kind="audio")` | 1852 | Enforces `ltx_max_source_seconds` = 330 s |
| Grid | `worker/adapters/ltx.py` | `_requested_dimensions` | 1857 | The REQUESTED aspect, unlike v2v/extend |
| Tier switch | `worker/adapters/ltx.py` | `_run_music_video` | 1858 | `bool(job.execution.get("audio_conditioning"))` |
| Pass ceiling | `worker/adapters/ltx.py` | `_audio_pass_seconds` / `_per_pass_seconds` | 2388 / 2291 | |
| Cut points | `worker/adapters/ltx.py` | `_musical_boundaries` | 1964 | Guarded by `execution.align_cuts_to_audio` (default `True`) |
| Onsets | `worker/media/audio.py` | `audio_onsets` → `detect_onsets` | 280 / 234 | |
| Boundary maths | `worker/longform/timing.py` | `plan_musical_boundaries` | 45 | |
| Audio window | `worker/adapters/ltx.py` | `_run_music_video.audio_window` | 1882-1893 | `AudioConditioning(staged, step.segment.start_seconds, _audio_window_seconds(frames))` |
| Window pad | `worker/adapters/ltx.py` | `_audio_window_seconds` | 2417 | `frames / 24 + 0.04` |
| Section pinning | `worker/adapters/ltx.py` | `_planned_section_frames(..., boundaries=boundaries)` | 1926-1932 | |
| Mux | `worker/media/audio.py` | `mux_audio` | 59 | The whole original track, once |

---

### 2.6 Audio-conditioned video

**Not a separate workflow.** It is the `_A2VID` branch of `_run_music_video` (§2.5), reached only by `execution.audio_conditioning: true`. No other workflow can select it — `audio=` is passed to `_renderer` at exactly one call site, [ltx.py:1904](apps/worker/worker/adapters/ltx.py#L1904).

### 2.7 Director T2V

Not a separate workflow. It is a branch inside `_run_generation` gated by `wants_director(job)`:

```python
_DIRECTOR_WORKFLOWS = frozenset({"text-to-video", "image-to-video"})

def wants_director(job: AdapterJob) -> bool:
    return (
        job.workflow_id in _DIRECTOR_WORKFLOWS
        and str(job.parameters.get("prompt_mode") or "").strip().lower() == "director"
    )
```
— [provider.py:421-437](apps/worker/worker/director/provider.py#L421-L437)

Chain divergence from §2.1:
- `structure_prompt` is **skipped** ([ltx.py:1061](apps/worker/worker/adapters/ltx.py#L1061)).
- `create_director_plan` runs once, before any section ([ltx.py:1099-1110](apps/worker/worker/adapters/ltx.py#L1099-L1110)).
- `compile_section_prompts(director_plan, step.total, total_seconds=seconds)` replaces `plan_section_prompts` ([ltx.py:1118-1121](apps/worker/worker/adapters/ltx.py#L1118-L1121)).
- A `DirectorFailure` becomes a **non-retriable** `AdapterError` — Director mode never silently falls back to the bare idea.

### 2.8 Director I2V

Same as §2.7 plus `source_anchored=True`, which:
- appends `_ANCHORED_RULES` to the planning brief ([provider.py:337-338](apps/worker/worker/director/provider.py#L337-L338));
- calls `source_image_facts(job)` when `settings.director_vision_enabled` ([provider.py:526](apps/worker/worker/director/provider.py#L526)) — default `False`;
- runs `_ground_visual_claims(plan, f"{idea}\n{grounding}")` ([plan.py:336-337](apps/worker/worker/director/plan.py#L336-L337)), which strips any character `appearance`, `continuity` fact, or `scene` sentence whose distinctive words are not present in the supplied text;
- makes the compiler emit `_anchored_cast_sentence` and the frame-anchored constancy sentence ([compiler.py:403-421](apps/worker/worker/director/compiler.py#L403-L421), [compiler.py:338-352](apps/worker/worker/director/compiler.py#L338-L352)).

### 2.9 Director Extend

Reached automatically — no request parameter. `continuation_lineage(job)` is non-`None` when the API injected `parameters["director_lineage"]` with `prompt_mode == "director"` and a non-empty `idea`. The plan is then built with `prior_idea`, `prior_seconds`, `anchored=True`, and the ancestor's `dialogue_language` ([provider.py:508-518](apps/worker/worker/director/provider.py#L508-L518)), and `_CONTINUATION_RULES` is appended to the brief.

### 2.10 Other video workflows discovered

None. `_ADAPTERS` also registers `"mock"` ([adapters/mock.py](apps/worker/worker/adapters/mock.py)), `"harness"` (real ffmpeg media, no model — [adapters/harness.py](apps/worker/worker/adapters/harness.py)), and `"music"` ([adapters/music.py](apps/worker/worker/adapters/music.py)). None of the three touches LTX.

---

## 2.11 Workflow-specific extraction

### Image-to-Video

| Question | Answer | Evidence |
|---|---|---|
| How is the image passed to the model? | **A file path**, as one `--image PATH FRAME_IDX STRENGTH` triple. Not a tensor, not base64, not a latent. | [ltx.py:722-726](apps/worker/worker/adapters/ltx.py#L722-L726) `ConditioningFrame.as_args`; emitted at [ltx.py:2785-2789](apps/worker/worker/adapters/ltx.py#L2785-L2789) |
| What conditioning strength is used, and where is it set? | Frame-0 still: **`1.0`**, hardcoded at the call site — [ltx.py:1129](apps/worker/worker/adapters/ltx.py#L1129) `return [ConditioningFrame(still, 0, 1.0)] if still else []`. Mid-window identity anchor: **`0.2`**, overridable via `execution.i2v_reference_strength` — [ltx.py:2452](apps/worker/worker/adapters/ltx.py#L2452). |
| Which frame index is the image pinned to? | **0** on the first pass. On later passes the still moves to `min(frames - 1, max(1, frames // 3))` — [ltx.py:2451](apps/worker/worker/adapters/ltx.py#L2451) |
| Is the image re-injected at any later stage? | **Yes, conditionally.** Every pass after the first attempts a second `--image` with the original upload. It is **dropped** unless the pass's frame count is in `_TWO_IMAGE_SAFE_FRAMES = frozenset({120, 240, 360})` — [ltx.py:363](apps/worker/worker/adapters/ltx.py#L363), [ltx.py:2455-2465](apps/worker/worker/adapters/ltx.py#L2455-L2465). At the shipped 30-second pass ceiling a pass is 720 frames, so **on a 60s I2V job the anchor is always dropped** (verified by dry-run, §5.6). |
| Does behaviour differ between stage 1 and stage 2? | **No stage distinction is expressed by this adapter** for the distilled or guided tiers — it passes `--num-frames/--height/--width` once and the pipeline runs its own stages. The one stage-aware behaviour is `LtxPipeline.stage_1_only` ([ltx.py:871-885](apps/worker/worker/adapters/ltx.py#L871-L885)), used **only** by `_IC_LORA`, which doubles the grid and appends `--skip-stage-2`. |
| Any long-form re-conditioning across sections? | Yes: seam frame at index 0 strength `1.0` every pass, plus the conditional identity anchor above. |

### Video-to-Video

| Question | Answer | Evidence |
|---|---|---|
| How is the source video ingested? | Downloaded to `job.workspace` by the runner, then `probe_media`. The staged path is used directly by every subsequent ffmpeg/subprocess call. | [runner.py:185-212](apps/worker/worker/jobs/runner.py#L185-L212), [ltx.py:2131-2215](apps/worker/worker/adapters/ltx.py#L2131-L2215) |
| Is frame extraction performed? Where? At what fps? | **Restyle engine (`_DISTILLED`):** yes — `extract_frames_at(staged, timestamps, ...)` pulls stills at specific timestamps, no fps involved ([ltx.py:1448-1456](apps/worker/worker/adapters/ltx.py#L1448-L1456)). **Transform engine (`_IC_LORA`):** no stills; instead a whole edge-map clip is built at `fps=float(settings.ltx_frame_rate)` = **24** and `frames=<the pass's rendered count>` ([ltx.py:1641-1656](apps/worker/worker/adapters/ltx.py#L1641-L1656), [control.py:88-116](apps/worker/worker/media/control.py#L88-L116)). |
| How is source duration handled if it mismatches requested duration? | It cannot mismatch: the API **rejects** a supplied duration (`duration_mode: source`, [workflow_registry.py:141-156](apps/api/app/services/workflow_registry.py#L141-L156)) and the adapter uses `target_seconds = source.duration_seconds or 0.0` ([ltx.py:1378](apps/worker/worker/adapters/ltx.py#L1378)). A source over `settings.ltx_max_source_seconds` (330.0 s) is refused before any compute ([ltx.py:2196-2214](apps/worker/worker/adapters/ltx.py#L2196-L2214)). |
| What strength/conditioning value is applied? | Restyle: stills at `v2v_structure_strength` default **`0.45`**; seam at `v2v_continuity_strength` default **`0.85`**; reference image at `v2v_reference_strength` default **`0.3`**. Transform: control clip at `v2v_control_strength` default **`1.0`**; LoRA at `v2v_lora_strength` default **`1.0`**; attention mask always at strength **`1.0`** with per-region values inside. Identity: anchor `1.0` (composited) or `0.65` (raw fallback), refresh **`0.0`**, subject attention **`0.5`**. | [ltx.py:544-655](apps/worker/worker/adapters/ltx.py#L544-L655) |
| Is source audio preserved, discarded, or replaced? | **Preserved**, muxed back whole at the end — `mux_audio(picture, staged, output)` ([ltx.py:1809-1810](apps/worker/worker/adapters/ltx.py#L1809-L1810)). If the source has no audio the result has none (`AudioMode.NO_AUDIO`). |
| What happens to model-generated audio? | **Discarded.** The sections are normalised with `audio=False` before concat ([ltx.py:1798-1805](apps/worker/worker/adapters/ltx.py#L1798-L1805)), and `normalize_clip(audio=False)` passes `-an` to ffmpeg ([frames.py:172](apps/worker/worker/media/frames.py#L172)). |

### Extend / continuation

| Question | Answer | Evidence |
|---|---|---|
| How is the source asset located? | As a normal workflow **input** (`role: source_video`), downloaded by presigned URL. Director lineage is separately resolved by the API from the DB (`source asset → producing job`) and injected into `parameters`. | [extend-video.yaml:18-31](workflow-definitions/extend-video.yaml#L18-L31); [generation.py:201-257](apps/api/app/services/generation.py#L201-L257) |
| How many previous frames are fed back, and how? | **Exactly one** — the last decodable frame, as a PNG, passed as `--image <path> 0 1.0`. Extracted with `ffmpeg -sseof -1 -i <src> -update 1 <dest.png>`. | [ltx.py:1252](apps/worker/worker/adapters/ltx.py#L1252), [ltx.py:1259-1266](apps/worker/worker/adapters/ltx.py#L1259-L1266), [frames.py:33-53](apps/worker/worker/media/frames.py#L33-L53) |
| Is it last-frame conditioning, latent continuation, or something else? | **Last-frame conditioning.** No latent is carried; `Segment.overlap_seconds` exists ([segments.py:40-47](apps/worker/worker/media/segments.py#L40-L47)) but `plan_segments` is always called with the default `overlap_seconds=0.0` — no caller in the repository passes a non-zero overlap. |
| How is the prompt carried or modified? | Standard: `structure_prompt` then `plan_section_prompts(job.prompt, total, total_seconds=extension_seconds)` — timestamps are relative to the **extension**, not the combined video ([ltx.py:1243-1248](apps/worker/worker/adapters/ltx.py#L1243-L1248)). Director-lineage: `compile_section_prompts(continuation_plan, ...)`. |
| Is any state persisted between calls? | **Yes, in the database.** `parameters["director_lineage"]` — a dict carrying `prompt_mode`, `dialogue_language`, `idea`, `prior_seconds`, and optionally `identity_image_asset_id` — is written at job-creation time and read by the worker ([generation.py:239-256](apps/api/app/services/generation.py#L239-L256), [provider.py:467-483](apps/worker/worker/director/provider.py#L467-L483)). It is inherited transitively: `parent_params.get("director_lineage")` at [generation.py:227](apps/api/app/services/generation.py#L227). Nothing else persists between calls. |

### Music Video

| Question | Answer | Evidence |
|---|---|---|
| Which pipeline is invoked? | `ltx_pipelines.distilled` by default; `ltx_pipelines.a2vid_two_stage` when `execution.audio_conditioning` is truthy. | [ltx.py:1903](apps/worker/worker/adapters/ltx.py#L1903) `pipeline=_A2VID if audio_conditioned else _DISTILLED` |
| **Is the actual song file passed to the model?** | **NO** in the committed configuration. `music-video.yaml` does not set `audio_conditioning`, so `audio_conditioned` is `False`, `audio=None` is passed to `_renderer` ([ltx.py:1904](apps/worker/worker/adapters/ltx.py#L1904)), and `_command` emits no audio flag ([ltx.py:2796-2797](apps/worker/worker/adapters/ltx.py#L2796-L2797)). **YES** if `audio_conditioning: true` is set. |
| If yes: how, at what point, with what parameter name? | As a **file path to the whole master track**, never a slice, with a seek offset. Three flags together: `--audio-path <path>`, `--audio-start-time <s.sss>`, `--audio-max-duration <s.sss>` — [ltx.py:786-791](apps/worker/worker/adapters/ltx.py#L786-L791). Built per pass by `audio_window(step, frames)` at [ltx.py:1882-1893](apps/worker/worker/adapters/ltx.py#L1882-L1893); `start_seconds = step.segment.start_seconds`, `max_duration_seconds = frames / 24 + 0.04`. |
| If no: what is passed instead? | Nothing audio-related. The picture is generated from the text prompt alone. The track contributes only (a) the total duration, and (b) the cut points via `plan_musical_boundaries`. Stated at [music-video.yaml:86-90](workflow-definitions/music-video.yaml#L86-L90) and [ltx.py:1843-1850](apps/worker/worker/adapters/ltx.py#L1843-L1850). |
| Is any audio conditioning flag enabled or disabled? | `audio_conditioning` — **disabled** (commented out at [music-video.yaml:98](workflow-definitions/music-video.yaml#L98)). `align_cuts_to_audio: true` — **enabled** ([music-video.yaml:82](workflow-definitions/music-video.yaml#L82)); it controls only where the seams land. |
| What happens to model-generated audio? | **Discarded.** Sections are normalised with `audio=False` ([ltx.py:1940](apps/worker/worker/adapters/ltx.py#L1940)) → ffmpeg `-an`. |
| Is the master track re-attached in post? Where? | **Yes, exactly once, at the end.** `mux_audio(picture, staged, output)` inside `assemble()` at [ltx.py:1943-1944](apps/worker/worker/adapters/ltx.py#L1943-L1944). `mux_audio` maps `0:v:0` + `1:a:0`, drops any video-side audio, and uses `-shortest` so the track's end ends the file ([audio.py:88-108](apps/worker/worker/media/audio.py#L88-L108)). |
| Which settings in this path are currently disabled/commented out? | `audio_conditioning`, `audio_pass_seconds`, `inference_steps`, `a2v_guidance_scale`, `guidance_scale`, `stg_scale` — all inside the comment block [music-video.yaml:84-140](workflow-definitions/music-video.yaml#L84-L140). |

### Director

**Where Director logic ends and model configuration begins — the explicit boundary:**

> **`worker/director/compiler.py::compile_section_prompts`** ([compiler.py:61](apps/worker/worker/director/compiler.py#L61)) is the boundary.
>
> Everything above it (`provider.py`, `plan.py`, `vision.py`, `cerebras.py`) produces a `DirectorPlan` — a pure data structure that no model ever sees. Everything below it is the ordinary render path. The compiler's return value is a `list[str]` of caption text, and the *only* way any of it reaches the model is as the value of `--prompt` via `prompt_for_step(step)` → `_command(..., prompt=...)` → `cmd += ["--prompt", ...]` ([ltx.py:2723](apps/worker/worker/adapters/ltx.py#L2723)).
>
> **No Director output configures the model.** Not the duration, not the camera, not the character count, not the language. The `DirectorPlan` cannot change `--num-frames`, `--width`, `--height`, `--seed`, any guidance scale, or which pipeline runs. The one exception is indirect and is a *pass-length clamp*, not a model setting: on `extend-video`, the presence of a continuation plan lowers the per-pass ceiling to 30 s ([ltx.py:1276-1283](apps/worker/worker/adapters/ltx.py#L1276-L1283)).

**Which Director outputs reach the model, and which are internal only:**

| DirectorPlan field | Type | Reaches the model? | How |
|---|---|---|---|
| `scene` | `str` | **Yes** | First sentence of every section caption — [compiler.py:159](apps/worker/worker/director/compiler.py#L159) |
| `tone` | `str` | **No — internal only.** Parsed, stored, never referenced by the compiler. | Parsed at [plan.py:308](apps/worker/worker/director/plan.py#L308); `grep 'plan.tone'` in compiler.py → no hits |
| `language` | `str` | **No — indirect only.** Used to instruct the planner; the dialogue text it produces is what reaches the model. Logged at [compiler.py:114](apps/worker/worker/director/compiler.py#L114). | |
| `duration_seconds` | `float` | **No.** Used for the word/line budgets and bucket maths. | [plan.py:53-115](apps/worker/worker/director/plan.py#L53-L115) |
| `ambience` | `str` | **Yes** | `_ambience_sentence` — [compiler.py:553-558](apps/worker/worker/director/compiler.py#L553-L558) |
| `characters[].id` | `str` | **No** — deliberately stripped by `_humanise` and replaced with the role word | [compiler.py:561-626](apps/worker/worker/director/compiler.py#L561-L626) |
| `characters[].role` | `str` | **Yes** — "the detective" | `_subject` [compiler.py:636-639](apps/worker/worker/director/compiler.py#L636-L639) |
| `characters[].appearance` | `str` | **Yes** — first mention per section | `_full_subject` [compiler.py:647-663](apps/worker/worker/director/compiler.py#L647-L663) |
| `characters[].voice` | `str` | **Yes** — first time they speak in a section, when the event has no `delivery` | `_speech_verb` [compiler.py:468-483](apps/worker/worker/director/compiler.py#L468-L483) |
| `timeline[].start` / `.end` | `float` | **No — timestamps never reach the prompt.** Used only to bucket events into sections and to decide the settle sentence. Stated explicitly at [plan.py:16-17](apps/worker/worker/director/plan.py#L16-L17) and [compiler.py:6-13](apps/worker/worker/director/compiler.py#L6-L13). | |
| `timeline[].action` | `str` | **Yes** | `_event_sentence` [compiler.py:424-454](apps/worker/worker/director/compiler.py#L424-L454) |
| `timeline[].camera` | `str` | **Yes** — rewritten into prose | `_camera_sentence` [compiler.py:511-522](apps/worker/worker/director/compiler.py#L511-L522) |
| `timeline[].speaker` | `str \| None` | **Yes, indirectly** — resolves to the role word that owns the quote | [compiler.py:431](apps/worker/worker/director/compiler.py#L431) |
| `timeline[].dialogue` | `str \| None` | **Yes, verbatim, in double quotes** | [compiler.py:450-452](apps/worker/worker/director/compiler.py#L450-L452) |
| `timeline[].delivery` | `str \| None` | **Yes** — "says in a low and accusing voice" | [compiler.py:475](apps/worker/worker/director/compiler.py#L475) |
| `timeline[].exits` | `tuple[str, ...]` | **Yes, indirectly** — changes which characters appear in cast/constancy sentences, and adds `_after_exit_sentence` / `_remaining_sentence` | [compiler.py:87-94](apps/worker/worker/director/compiler.py#L87-L94), [compiler.py:216-225](apps/worker/worker/director/compiler.py#L216-L225) |
| `source_anchored` | `bool` | **Yes, indirectly** — switches the cast and constancy sentence forms | [compiler.py:162-165](apps/worker/worker/director/compiler.py#L162-L165), [compiler.py:338-352](apps/worker/worker/director/compiler.py#L338-L352) |
| `continuity[]` | `tuple[str, ...]` | **Yes** — every fact restated at the end of every section | [compiler.py:370-371](apps/worker/worker/director/compiler.py#L370-L371) |

**What Director receives as input:** `job.prompt` (the idea, verbatim), `duration_seconds`, the resolved dialogue language, a deterministic seed `zlib.crc32(f"{job_id}:director")`, `source_anchored`, and optionally `image_facts` (I2V, off by default) or `prior_idea`/`prior_seconds` (extension). [provider.py:537-548](apps/worker/worker/director/provider.py#L537-L548).

**Which model Director itself uses:** `CerebrasDirectorProvider` first if `cerebras_api_key` is set and `cerebras_director_enabled` (model `gemma-4-31b`, temperature `0.7`, timeout `60.0` s), then `GemmaDirectorProvider` — the local `gemma-4-e2b-it` checkpoint run as a subprocess in `ltx_repo_dir` with `max_new_tokens: 1600`, timeout `900.0` s. [provider.py:454-464](apps/worker/worker/director/provider.py#L454-L464), [provider.py:352-364](apps/worker/worker/director/provider.py#L352-L364). The complete prompt template is pasted in §11.
---

## 3. Complete Settings Extraction

`Overridable?` column key:
- **env** — a `WorkerSettings` field, settable by environment variable (uppercase field name) or repo-root `.env`.
- **execution** — readable from the workflow YAML's private `execution` block via `job.execution.get` / `execution_int` / `execution_float`.
- **request** — settable by the customer through `GenerationParameters`.
- **no** — a module constant with no read path from configuration.

### 3.1 Sampling

| Setting | Value | Type | Defined In (file:line) | Applies To | Overridable? |
|---|---|---|---|---|---|
| `steps` / `--num-inference-steps` | **not emitted** unless `execution.inference_steps` is set; no YAML sets it | int | emitted at [ltx.py:2764-2768](apps/worker/worker/adapters/ltx.py#L2764-L2768) | `_A2VID`, `_GUIDED` only (guarded by `pipeline.distilled_lora`) | execution |
| LTX's own stage-1 step default | `30` — recorded in a **comment only**, not in code | int | [music-video.yaml:131](workflow-definitions/music-video.yaml#L131) "(pipeline default 30)" | a2vid | — |
| `cfg` / `--video-cfg-guidance-scale` | **not emitted** unless `execution.guidance_scale` is set; no YAML sets it | float | [ltx.py:2756-2763](apps/worker/worker/adapters/ltx.py#L2756-L2763) | `_A2VID`, `_GUIDED` only | execution |
| LTX's own cfg default | `3.0` — comment only | float | [ltx.py:2746](apps/worker/worker/adapters/ltx.py#L2746) "The shipped defaults (3.0 / 1.0 / 3.0)"; also [ltx.py:968](apps/worker/worker/adapters/ltx.py#L968) "CFG (official default 3.0)"; [music-video.yaml:133](workflow-definitions/music-video.yaml#L133) | a2vid, guided | — |
| `stg` / `--video-stg-guidance-scale` | **not emitted** unless `execution.stg_scale` is set; no YAML sets it | float | [ltx.py:2758](apps/worker/worker/adapters/ltx.py#L2758) | `_A2VID`, `_GUIDED` only | execution |
| LTX's own STG default | `1.0` — comment only | float | [ltx.py:2746](apps/worker/worker/adapters/ltx.py#L2746) | a2vid, guided | — |
| `a2v_guidance_scale` / `--a2v-guidance-scale` | **not emitted** unless `execution.a2v_guidance_scale` is set; no YAML sets it | float | [ltx.py:2759](apps/worker/worker/adapters/ltx.py#L2759) | `_A2VID`, `_GUIDED` | execution |
| LTX's own a2v/modality default | `3.0`; `1.0` disables the pass entirely — comment only | float | [ltx.py:2746](apps/worker/worker/adapters/ltx.py#L2746), [ltx.py:2755](apps/worker/worker/adapters/ltx.py#L2755), [music-video.yaml:132-136](workflow-definitions/music-video.yaml#L132-L136) | a2vid | — |
| `audio_stg_scale` | **— not present.** No flag, no key, no reference anywhere in the repo. | — | — | — | — |
| `modality_scale` | Present only as the *description* of `a2v_guidance_scale`: "`a2v_guidance_scale` is the modality one" | — | [ltx.py:2750](apps/worker/worker/adapters/ltx.py#L2750) | — | — |
| `audio_modality_scale` | **— not present.** | — | — | — | — |
| `sigmas` / sigma schedule | **— not present** as a settable value. Referenced once in prose: "run a short sigma schedule against the full checkpoint" | — | [ltx.py:830-831](apps/worker/worker/adapters/ltx.py#L830-L831) | — | — |
| `scheduler` | **— not present.** No flag emitted, no key read. | — | — | — | — |
| `sampler` | **— not present.** | — | — | — | — |
| `seed` / `--seed` | Always emitted. Per pass: `zlib.crc32(f"{job_id}:{index}")` when the user supplied none; otherwise `(user_seed + index) % 2**31` | int | [ltx.py:2886-2894](apps/worker/worker/adapters/ltx.py#L2886-L2894); emitted [ltx.py:2728](apps/worker/worker/adapters/ltx.py#L2728) | all pipelines | request (`parameters.seed`, 0..2³¹−1, [generation.py:35](apps/api/app/schemas/generation.py#L35)) |
| seed fallback inside `_command` | `zlib.crc32(job.job_id.encode())` when `seed is None` | int | [ltx.py:2678-2683](apps/worker/worker/adapters/ltx.py#L2678-L2683) | all | no |
| seed strategy | Distinct per pass, deterministic per job. Comment: "Distinct per pass or every chained render replays the same noise; still deterministic so a retry reproduces." | — | [ltx.py:2588-2589](apps/worker/worker/adapters/ltx.py#L2588-L2589) | all | no |
| `negative_prompt` / `--negative-prompt` | **not emitted** unless `execution.negative_prompt` is a non-empty string after `.strip()`; no YAML sets it | str | [ltx.py:2738-2740](apps/worker/worker/adapters/ltx.py#L2738-L2740) | `_A2VID`, `_GUIDED` only | execution |
| `--enhance-prompt` | **not emitted** unless `execution.enhance_prompt` is truthy; no YAML sets it. When set, emits `--enhance-prompt --prompt-enhancer-gemma-root <settings.director_gemma_root>` | bool | [ltx.py:2769-2784](apps/worker/worker/adapters/ltx.py#L2769-L2784) | all pipelines | execution |

**The guard that makes all guidance flags pipeline-specific** — [ltx.py:2731-2737](apps/worker/worker/adapters/ltx.py#L2731-L2737), verbatim:

```python
if pipeline.distilled_lora:
    # The guided family (dev transformer + distilled LoRA) is the ONLY
    # place these exist — the distilled and ic_lora entry points have
    # no guiders, and sending the flags there is a crash, not a hint.
    # Both stay unset by default: the pipeline's own defaults are the
    # measured baseline, and a guidance change is a quality/cost
    # judgement that belongs in a workflow's execution block.
```

`pipeline.distilled_lora` is `True` only for `_A2VID` and `_GUIDED`.

### 3.2 Geometry

| Setting | Value | Type | Defined In (file:line) | Applies To | Overridable? |
|---|---|---|---|---|---|
| `width` / `height` — aspect table | `"16:9": (1024, 576)`, `"9:16": (576, 1024)`, `"1:1": (768, 768)`, `"4:5": (512, 640)` | dict | [ltx.py:206-213](apps/worker/worker/adapters/ltx.py#L206-L213) | t2v, i2v, music-video | request (`parameters.aspect_ratio`) |
| default grid | `(1024, 576)` | tuple | [ltx.py:214](apps/worker/worker/adapters/ltx.py#L214) | all | no |
| grid for source-derived workflows | `grid_for_source(w, h)` — searches every `(w, h)` on a /64 lattice from 256 to 1024 with `w*h <= _PIXEL_BUDGET`; prefers the largest grid within aspect error `0.08`, else the minimum-error grid | function | [ltx.py:658-690](apps/worker/worker/adapters/ltx.py#L658-L690) | v2v, extend | no |
| `_PIXEL_BUDGET` | `1024 * 576` = `589824` | int | [ltx.py:460](apps/worker/worker/adapters/ltx.py#L460) | `grid_for_source` | no |
| `num_frames` | `max(1, round(seconds * settings.ltx_frame_rate))`, then substituted by `safe_frame_count` or `conforming_frames` + `measured_landings` | int | `_frame_count` [ltx.py:2429-2430](apps/worker/worker/adapters/ltx.py#L2429-L2430); substitution [ltx.py:2523-2537](apps/worker/worker/adapters/ltx.py#L2523-L2537) | all | no |
| `fps` / `--frame-rate` | `24` | int | `settings.ltx_frame_rate` [config.py:299](apps/worker/worker/core/config.py#L299); emitted [ltx.py:2727](apps/worker/worker/adapters/ltx.py#L2727) | all | env (`LTX_FRAME_RATE`) |
| aspect ratio — allowed values | t2v: `["16:9","9:16","1:1","4:5"]`; i2v: `["16:9","9:16","1:1"]`; v2v: `["16:9","9:16"]`; extend: `["16:9","9:16"]`; music-video: `["16:9","9:16","1:1"]`; music: `[]` | list | [text-to-video.yaml:38](workflow-definitions/text-to-video.yaml#L38), [image-to-video.yaml:33](workflow-definitions/image-to-video.yaml#L33), [video-to-video.yaml:53](workflow-definitions/video-to-video.yaml#L53), [extend-video.yaml:42](workflow-definitions/extend-video.yaml#L42), [music-video.yaml:44](workflow-definitions/music-video.yaml#L44), [music.yaml:29](workflow-definitions/music.yaml#L29) | — | request |
| frame divisibility rule | `8k + 1` — `conforming_frames(f) = max(1, f + ((1 - f) % 8))` | function | `_FRAME_LATTICE = 8` [ltx.py:377](apps/worker/worker/adapters/ltx.py#L377); [ltx.py:380-389](apps/worker/worker/adapters/ltx.py#L380-L389) | all | no |
| resolution divisibility rule | Each side divisible by **64** | — | [ltx.py:199-200](apps/worker/worker/adapters/ltx.py#L199-L200); enforced by construction in `_DIMENSIONS` and `grid_for_source`'s `range(256, 1025, 64)` | all | no |
| stage-1 resolution | **— not expressed by this adapter** for distilled/guided/a2vid. For `_IC_LORA` only: `width, height = width * 2, height * 2` then `--skip-stage-2`, so stage 1's native half-size output *is* the target grid | — | [ltx.py:2690-2695](apps/worker/worker/adapters/ltx.py#L2690-L2695) | ic_lora | no |
| stage-2 resolution | **— not expressed.** No flag. | — | — | — | — |
| upscale behaviour | `--spatial-upsampler-path` is always passed; nothing else controls it. `--skip-stage-2` is emitted only for `_IC_LORA` | — | [ltx.py:2704](apps/worker/worker/adapters/ltx.py#L2704), [ltx.py:2798-2799](apps/worker/worker/adapters/ltx.py#L2798-L2799) | all / ic_lora | no |
| output long-side cap | `1920` | int | [ltx.py:465](apps/worker/worker/adapters/ltx.py#L465) | delivery (v2v, extend) | no |
| output short-side cap | `1080` | int | [ltx.py:466](apps/worker/worker/adapters/ltx.py#L466) | delivery | no |
| delivery dimensions | Source's own, scaled to the caps, forced even: `max(2, int(value * scale) // 2 * 2)` | function | `output_dimensions` [ltx.py:693-707](apps/worker/worker/adapters/ltx.py#L693-L707) | v2v, extend | no |
| delivery fps | `min(60.0, max(10.0, source.fps or 24.0))` | float | `_delivery_fps` [ltx.py:3109-3117](apps/worker/worker/adapters/ltx.py#L3109-L3117) | v2v, extend | no |

### 3.3 Conditioning

| Setting | Value | Type | Defined In (file:line) | Applies To | Overridable? |
|---|---|---|---|---|---|
| I2V first-frame strength | `1.0` (literal at the call site) | float | [ltx.py:1129](apps/worker/worker/adapters/ltx.py#L1129) | i2v | no |
| seam (continuity) strength, t2v/i2v/music-video/extend | `1.0` (literal) | float | [ltx.py:1133](apps/worker/worker/adapters/ltx.py#L1133), [ltx.py:1261](apps/worker/worker/adapters/ltx.py#L1261), [ltx.py:1880](apps/worker/worker/adapters/ltx.py#L1880) | t2v, i2v, extend, music-video | no |
| `i2v_reference_strength` (mid-window identity anchor) | `0.2` | float | [ltx.py:2452](apps/worker/worker/adapters/ltx.py#L2452) | i2v, Director-lineage extend | execution |
| identity anchor frame index | `min(frames - 1, max(1, frames // 3))` | int | [ltx.py:2451](apps/worker/worker/adapters/ltx.py#L2451) | i2v, extend | no |
| `_TWO_IMAGE_SAFE_FRAMES` | `frozenset({120, 240, 360})` | frozenset | [ltx.py:363](apps/worker/worker/adapters/ltx.py#L363) | i2v, extend | no |
| `v2v_keyframe_seconds` | `4.0` (a density; clamped `max(0.5, …)`) | float | `_V2V_KEYFRAME_SECONDS` [ltx.py:544](apps/worker/worker/adapters/ltx.py#L544); read [ltx.py:1407-1409](apps/worker/worker/adapters/ltx.py#L1407-L1409) | v2v restyle | execution |
| `v2v_keyframes` | *(unset)* — when set, `max(1, min(16, int(v)))` overrides the density | int | [ltx.py:1406](apps/worker/worker/adapters/ltx.py#L1406), [ltx.py:1413-1414](apps/worker/worker/adapters/ltx.py#L1413-L1414) | v2v restyle | execution |
| `_V2V_KEYFRAME_BOUNDS` | `(3, 16)` | tuple | [ltx.py:559](apps/worker/worker/adapters/ltx.py#L559) | v2v restyle | no |
| `v2v_structure_strength` | `0.45` | float | `_V2V_STRUCTURE_STRENGTH` [ltx.py:564](apps/worker/worker/adapters/ltx.py#L564); read [ltx.py:1417](apps/worker/worker/adapters/ltx.py#L1417) | v2v restyle | execution |
| `v2v_continuity_strength` | `0.85` | float | `_V2V_CONTINUITY_STRENGTH` [ltx.py:581](apps/worker/worker/adapters/ltx.py#L581); read [ltx.py:1418](apps/worker/worker/adapters/ltx.py#L1418) and [ltx.py:1564](apps/worker/worker/adapters/ltx.py#L1564) | v2v both engines | execution |
| `v2v_reference_strength` | `0.3` | float | `_V2V_REFERENCE_STRENGTH` [ltx.py:586](apps/worker/worker/adapters/ltx.py#L586); read [ltx.py:1419-1421](apps/worker/worker/adapters/ltx.py#L1419-L1421), [ltx.py:1565-1567](apps/worker/worker/adapters/ltx.py#L1565-L1567) | v2v both engines | execution |
| `v2v_control_strength` | `1.0` | float | `_V2V_CONTROL_STRENGTH` [ltx.py:643](apps/worker/worker/adapters/ltx.py#L643); read [ltx.py:1562](apps/worker/worker/adapters/ltx.py#L1562) | v2v transform | execution |
| `v2v_lora_strength` | `1.0` | float | `_V2V_LORA_STRENGTH` [ltx.py:652](apps/worker/worker/adapters/ltx.py#L652); read [ltx.py:1563](apps/worker/worker/adapters/ltx.py#L1563) | v2v transform | execution |
| `v2v_edge_low` | `0.1` | float | `DEFAULT_EDGE_LOW` [control.py:45](apps/worker/worker/media/control.py#L45); read [ltx.py:1603](apps/worker/worker/adapters/ltx.py#L1603) | v2v transform | execution |
| `v2v_edge_high` | `0.4` | float | `DEFAULT_EDGE_HIGH` [control.py:46](apps/worker/worker/media/control.py#L46); read [ltx.py:1604](apps/worker/worker/adapters/ltx.py#L1604) | v2v transform | execution |
| `v2v_identity_anchor_strength` | `1.0` | float | `_V2V_IDENTITY_ANCHOR_STRENGTH` [ltx.py:592](apps/worker/worker/adapters/ltx.py#L592); read [ltx.py:1568-1570](apps/worker/worker/adapters/ltx.py#L1568-L1570) | v2v identity | execution |
| raw-anchor cap (fallback) | `0.65` — applied via `min(anchor_strength, 0.65)` when the composited anchor cannot be built | float | `_V2V_IDENTITY_RAW_ANCHOR_STRENGTH` [ltx.py:603](apps/worker/worker/adapters/ltx.py#L603); applied [ltx.py:1592-1596](apps/worker/worker/adapters/ltx.py#L1592-L1596) | v2v identity | no |
| `v2v_identity_refresh_strength` | **`0.0`** (OFF) | float | `_V2V_IDENTITY_REFRESH_STRENGTH` [ltx.py:610](apps/worker/worker/adapters/ltx.py#L610); read [ltx.py:1597-1599](apps/worker/worker/adapters/ltx.py#L1597-L1599) | v2v identity | execution |
| `v2v_identity_subject_attention` | `0.5` | float | `_V2V_IDENTITY_SUBJECT_ATTENTION` [ltx.py:630](apps/worker/worker/adapters/ltx.py#L630); read [ltx.py:1600-1602](apps/worker/worker/adapters/ltx.py#L1600-L1602) | v2v identity | execution |
| `v2v_identity_composited_anchor` | `True` (default when key absent) | bool | [ltx.py:1577](apps/worker/worker/adapters/ltx.py#L1577) | v2v identity | execution |
| `v2v_identity_describe_reference` | `True` (default when key absent) | bool | [ltx.py:1533](apps/worker/worker/adapters/ltx.py#L1533) | v2v identity | execution |
| `v2v_background_attention` | `0.5` | float | `BACKGROUND_ATTENTION` [masks.py:67](apps/worker/worker/media/masks.py#L67); read [ltx.py:1714-1716](apps/worker/worker/adapters/ltx.py#L1714-L1716) | v2v person-lock | execution |
| attention-mask conditioning strength | `1.0`, always | float | [ltx.py:1702](apps/worker/worker/adapters/ltx.py#L1702), [ltx.py:1719](apps/worker/worker/adapters/ltx.py#L1719) | v2v transform | no |
| matte dilation passes | `2` | int | `DILATION_PASSES` [masks.py:51](apps/worker/worker/media/masks.py#L51) | v2v | no |
| matte feather radius | `12` px | int | `FEATHER_RADIUS` [masks.py:56](apps/worker/worker/media/masks.py#L56) | v2v | no |
| `audio_conditioning` | `False` (absent from every YAML) | bool | read [ltx.py:1858](apps/worker/worker/adapters/ltx.py#L1858) | music-video | execution |
| audio window duration | `frames / 24 + 0.04` | float | `_audio_window_seconds` [ltx.py:2417-2427](apps/worker/worker/adapters/ltx.py#L2417-L2427); `_AUDIO_WINDOW_PAD_SECONDS` [ltx.py:510](apps/worker/worker/adapters/ltx.py#L510) | a2vid | no |
| audio start offset | `step.segment.start_seconds`, formatted `%.3f` | float | [ltx.py:1891](apps/worker/worker/adapters/ltx.py#L1891), [ltx.py:789](apps/worker/worker/adapters/ltx.py#L789) | a2vid | no |
| `align_cuts_to_audio` | `True` (default when key absent); set explicitly `true` in music-video.yaml | bool | read [ltx.py:1973](apps/worker/worker/adapters/ltx.py#L1973); set [music-video.yaml:82](workflow-definitions/music-video.yaml#L82) | music-video | execution |

**Conditioning argument shapes** — all four dataclasses, verbatim ([ltx.py:710-802](apps/worker/worker/adapters/ltx.py#L710-L802)):

```python
class ConditioningFrame:      # --image PATH FRAME_IDX STRENGTH
    path: Path; frame_index: int; strength: float
    def as_args(self): return ["--image", str(self.path), str(self.frame_index),
                               str(round(self.strength, 3))]

class ControlConditioning:    # --video-conditioning PATH STRENGTH
    path: Path; strength: float
    def as_args(self): return ["--video-conditioning", str(self.path),
                               str(round(self.strength, 3))]

class MaskConditioning:       # --conditioning-attention-mask PATH STRENGTH
    path: Path; strength: float
    def as_args(self): return ["--conditioning-attention-mask", str(self.path),
                               str(round(self.strength, 3))]

class AudioConditioning:      # --audio-path / --audio-start-time / --audio-max-duration
    path: Path; start_seconds: float; max_duration_seconds: float
    def as_args(self): return ["--audio-path", str(self.path),
                               "--audio-start-time", f"{self.start_seconds:.3f}",
                               "--audio-max-duration", f"{self.max_duration_seconds:.3f}"]

class LoraSpec:               # --lora PATH STRENGTH
    path: Path; strength: float = 1.0
    def as_args(self): return ["--lora", str(self.path), str(round(self.strength, 3))]
```

Argument ordering in `_command` is fixed ([ltx.py:2785-2799](apps/worker/worker/adapters/ltx.py#L2785-L2799)): all `--image` triples (in ascending frame order), then `--lora`, then `--video-conditioning`, then `--conditioning-attention-mask`, then the audio triple, then `--skip-stage-2`.

### 3.4 Runtime

| Setting | Value | Type | Defined In (file:line) | Applies To | Overridable? |
|---|---|---|---|---|---|
| precision / dtype | **— not a flag.** Implied by the checkpoint filename (`-nvfp4` vs `-bf16`) | — | [ltx.py:2639-2641](apps/worker/worker/adapters/ltx.py#L2639-L2641) | all | via `ltx_quantization` |
| `--quantization` | `"nvfp4-prequant"` — emitted **only** when `pipeline.quantize` is `True`, i.e. only for `_DISTILLED` | str | `settings.ltx_quantization` [config.py:271](apps/worker/worker/core/config.py#L271); emitted [ltx.py:2706-2707](apps/worker/worker/adapters/ltx.py#L2706-L2707) | distilled | env (`LTX_QUANTIZATION`) |
| transformer selection by quantization | `"transformer_nvfp4" if "nvfp4" in settings.ltx_quantization else "transformer_bf16"` | str | [ltx.py:2639-2641](apps/worker/worker/adapters/ltx.py#L2639-L2641) | distilled (`transformer_key=None`) | env |
| `--offload cpu` | Emitted when `pipeline.offload_cpu` — `True` for `_IC_LORA`, `_A2VID`, `_GUIDED`; `False` for `_DISTILLED` | — | [ltx.py:2708-2711](apps/worker/worker/adapters/ltx.py#L2708-L2711) | ic_lora, a2vid, guided | no |
| `--distilled-lora <path> 1.0` | Emitted when `pipeline.distilled_lora` — `True` for `_A2VID`, `_GUIDED`. Strength is the literal string `"1.0"` | — | [ltx.py:2712-2717](apps/worker/worker/adapters/ltx.py#L2712-L2717) | a2vid, guided | no |
| VAE tiling | **— not present.** No flag, no key, no reference. | — | — | — | — |
| VAE decode options | **— not present.** | — | — | — | — |
| model residency / caching | **— not present.** A fresh `uv run python -m …` subprocess per pass; the module docstring notes "each pass reloads a 22B transformer from host RAM" | — | [ltx.py:480-481](apps/worker/worker/adapters/ltx.py#L480-L481) | all | no |
| compile options (`torch.compile`) | **— not present.** | — | — | — | — |
| attention backend | **— not a flag.** NATTEN is named as a dependency in prose only | — | [ltx.py:203-204](apps/worker/worker/adapters/ltx.py#L203-L204), [ltx.py:225-226](apps/worker/worker/adapters/ltx.py#L225-L226) | — | — |
| device placement | **— not a flag.** `cwd=settings.ltx_repo_dir` and `start_new_session=True` are the only process controls | — | [ltx.py:2924-2936](apps/worker/worker/adapters/ltx.py#L2924-L2936) | all | no |
| batch size | **— not present.** One pass per subprocess. | — | — | — | — |
| timeout (adapter deadline) | `execution.timeout_seconds` or `settings.job_timeout_seconds` = `1800` | int | [config.py:123](apps/worker/worker/core/config.py#L123); applied [runner.py:128](apps/worker/worker/jobs/runner.py#L128) | all | execution + env |
| per-workflow `timeout_seconds` | v2v `5400`, extend `7200`, music-video `7200`, music `3600`; t2v and i2v **unset** (→ 1800) | int | [video-to-video.yaml:84](workflow-definitions/video-to-video.yaml#L84), [extend-video.yaml:77](workflow-definitions/extend-video.yaml#L77), [music-video.yaml:76](workflow-definitions/music-video.yaml#L76), [music.yaml:64](workflow-definitions/music.yaml#L64) | per workflow | execution |
| retry policy | Retriable unless the pipeline's output tail contains a deterministic-failure needle. `JOB_MAX_ATTEMPTS` default `3` | — | [ltx.py:2989](apps/worker/worker/adapters/ltx.py#L2989), [ltx.py:3007-3013](apps/worker/worker/adapters/ltx.py#L3007-L3013), [.env.example:69](.env.example#L69) | all | env (API side) |
| `_DETERMINISTIC_FAILURES` | `("CUBLAS_STATUS_INTERNAL_ERROR", "CUBLAS_STATUS_NOT_SUPPORTED", "CUBLAS_STATUS_INVALID_VALUE", "no kernel image is available", "CUDA error: invalid argument")` — note OUT OF MEMORY is deliberately absent | tuple | [ltx.py:3007-3013](apps/worker/worker/adapters/ltx.py#L3007-L3013) | all | no |
| cancellation poll | `2.0` s of stdout silence | float | `_CANCEL_POLL_SECONDS` [ltx.py:1014](apps/worker/worker/adapters/ltx.py#L1014) | all | no |
| kill grace | `30.0` s, applied twice (reap the handle, then poll `killpg(pgid, 0)`) | float | `_KILL_GRACE_SECONDS` [ltx.py:3038](apps/worker/worker/adapters/ltx.py#L3038) | all | no |
| diagnostics tail | `40` lines | int | `_OUTPUT_TAIL_LINES` [ltx.py:1017](apps/worker/worker/adapters/ltx.py#L1017) | all | no |
| `ltx_repo_dir` (subprocess cwd) | `Path("/workspace/ltx2-benchmark")` | Path | [config.py:183](apps/worker/worker/core/config.py#L183) | all | env (`LTX_REPO_DIR`) |
| `ltx_model_dir` | `None` → `<ltx_repo_dir>/models/ltx-2.5` | Path\|None | [config.py:191](apps/worker/worker/core/config.py#L191), [config.py:602-604](apps/worker/worker/core/config.py#L602-L604) | all | env (`LTX_MODEL_DIR`) |
| `ltx_max_seconds` | `60` | int | [config.py:279](apps/worker/worker/core/config.py#L279) | all pass ceilings | env (`LTX_MAX_SECONDS`) |
| `ltx_frame_rate` | `24` | int | [config.py:299](apps/worker/worker/core/config.py#L299) | all | env |
| `ltx_max_source_seconds` | `330.0` | float | [config.py:302](apps/worker/worker/core/config.py#L302) | v2v, music-video | env |
| `ltx_max_extend_source_seconds` | `1800.0` | float | [config.py:320](apps/worker/worker/core/config.py#L320) | extend | env |
| `max_segment_seconds` (worker default) | `10` | int | [config.py:174](apps/worker/worker/core/config.py#L174) | fallback only — see §4 | env |
| `max_concurrency` | `2` | int | [config.py:91](apps/worker/worker/core/config.py#L91) | node | env |
| `min_free_disk_mb` | `2048` | int | [config.py:157](apps/worker/worker/core/config.py#L157) | node | env |
| `lease_keepalive_seconds` | `45` | int | [config.py:109](apps/worker/worker/core/config.py#L109) | node | env |
| `shutdown_drain_seconds` | `300` | int | [config.py:132](apps/worker/worker/core/config.py#L132) | node | env |
| `download_timeout_seconds` | `300.0` | float | [config.py:144](apps/worker/worker/core/config.py#L144) | node | env |
| `upload_timeout_seconds` | `900.0` | float | [config.py:148](apps/worker/worker/core/config.py#L148) | node | env |
| `director_planner_timeout_seconds` | `900.0` | float | [config.py:233](apps/worker/worker/core/config.py#L233) | Director | env |
| `director_vision_enabled` | `False` | bool | [config.py:240](apps/worker/worker/core/config.py#L240) | Director I2V | env |
| `director_vision_timeout_seconds` | `300.0` | float | [config.py:263](apps/worker/worker/core/config.py#L263) | Director I2V | env |
| `cerebras_director_model` | `"gemma-4-31b"` (aliases `CEREBRAS_DIRECTOR_MODEL`, `CEREBRAS_AI_MODEL`) | str | [config.py:520-523](apps/worker/worker/core/config.py#L520-L523) | Director | env |
| `cerebras_director_enabled` | `True` | bool | [config.py:534](apps/worker/worker/core/config.py#L534) | Director | env |
| `cerebras_director_timeout_seconds` | `60.0` | float | [config.py:543](apps/worker/worker/core/config.py#L543) | Director | env |
| `cerebras_director_temperature` | `0.7` | float | [config.py:552](apps/worker/worker/core/config.py#L552) | Director | env |
| person-matte subprocess timeout | `1800.0` s | float | [masks.py:80](apps/worker/worker/media/masks.py#L80) | v2v | no |
| identity-anchor subprocess timeout | `600.0` s | float | [masks.py:154](apps/worker/worker/media/masks.py#L154) | v2v | no |
| control-clip ffmpeg timeout | `900.0` s | float | [control.py:61](apps/worker/worker/media/control.py#L61) | v2v | no |
| concat timeout | `900.0` s | float | [segments.py:133](apps/worker/worker/media/segments.py#L133) | all | no |
| `mux_audio` timeout | `1800.0` s | float | [audio.py:63](apps/worker/worker/media/audio.py#L63) | v2v, music-video | no |
| `normalize_clip` timeout | `900.0` s | float | [frames.py:114](apps/worker/worker/media/frames.py#L114) | all | no |

### 3.5 Model paths (exact strings)

Every path emitted is `settings.ltx_models_root / <relative>`. With defaults that root is `/workspace/ltx2-benchmark/models/ltx-2.5`.

| Flag | Relative filename | Chosen by | Defined at |
|---|---|---|---|
| `--transformer-path` | `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | `_DISTILLED` with `"nvfp4" in ltx_quantization` | [ltx.py:170](apps/worker/worker/adapters/ltx.py#L170) |
| `--transformer-path` | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | `_DISTILLED` with a non-nvfp4 quantization; **and** `_IC_LORA` unconditionally (`transformer_key="transformer_bf16"`) | [ltx.py:171](apps/worker/worker/adapters/ltx.py#L171), [ltx.py:896](apps/worker/worker/adapters/ltx.py#L896) |
| `--transformer-path` | `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | `_A2VID` and `_GUIDED` (`transformer_key="transformer_dev"`) | [ltx.py:189](apps/worker/worker/adapters/ltx.py#L189), [ltx.py:917](apps/worker/worker/adapters/ltx.py#L917), [ltx.py:981](apps/worker/worker/adapters/ltx.py#L981) |
| `--text-encoder-path` | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | always | [ltx.py:172](apps/worker/worker/adapters/ltx.py#L172), [ltx.py:2700](apps/worker/worker/adapters/ltx.py#L2700) |
| `--video-vae-path` | `vae/ltx-2.5-video-vae-bf16.safetensors` | always | [ltx.py:173](apps/worker/worker/adapters/ltx.py#L173), [ltx.py:2701](apps/worker/worker/adapters/ltx.py#L2701) |
| `--audio-vae-path` | `vae/ltx-2.5-audio-vae-bf16.safetensors` | always | [ltx.py:174](apps/worker/worker/adapters/ltx.py#L174), [ltx.py:2702](apps/worker/worker/adapters/ltx.py#L2702) |
| `--duration-head-path` | `model_patches/ltx-2.5-duration-head-bf16.safetensors` | always | [ltx.py:175](apps/worker/worker/adapters/ltx.py#L175), [ltx.py:2703](apps/worker/worker/adapters/ltx.py#L2703) |
| `--spatial-upsampler-path` | `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | always | [ltx.py:176-178](apps/worker/worker/adapters/ltx.py#L176-L178), [ltx.py:2704](apps/worker/worker/adapters/ltx.py#L2704) |
| `--distilled-lora` | `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (strength literal `"1.0"`) | `_A2VID`, `_GUIDED` | [ltx.py:190](apps/worker/worker/adapters/ltx.py#L190), [ltx.py:2712-2717](apps/worker/worker/adapters/ltx.py#L2712-L2717) |
| `--lora` | `loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` (strength `v2v_lora_strength`, default `1.0`) | `_IC_LORA` | [ltx.py:196](apps/worker/worker/adapters/ltx.py#L196), [ltx.py:1606-1609](apps/worker/worker/adapters/ltx.py#L1606-L1609) |
| `--prompt-enhancer-gemma-root` | `<ltx_repo_dir>/models/gemma-4-e2b-it` | only with `execution.enhance_prompt` | [config.py:632](apps/worker/worker/core/config.py#L632), [ltx.py:2783](apps/worker/worker/adapters/ltx.py#L2783) |

**Weight-existence enforcement:**
- `_require_models` ([ltx.py:2253-2263](apps/worker/worker/adapters/ltx.py#L2253-L2263)) runs on **every** job and demands the quantization-selected transformer plus `text_encoder, video_vae, audio_vae, duration_head, spatial_upsampler`. Missing → non-retriable "This tool is temporarily unavailable."
- `_optional_weight` ([ltx.py:2802-2825](apps/worker/worker/adapters/ltx.py#L2802-L2825)) checks a file only on the path that selected it.
- `LtxPipeline.extra_weights`: `_IC_LORA` → `("union_control_lora",)`; `_A2VID` and `_GUIDED` → `("transformer_dev", "distilled_lora")`; `_DISTILLED` → `()`.

### 3.6 Prompt & product-surface settings

| Setting | Value | Type | Defined In (file:line) | Applies To | Overridable? |
|---|---|---|---|---|---|
| `prompt.max_length` | `20000` on all six workflows | int | t2v:26, i2v:16, v2v:16, extend:16, music-video:16, music:16 | all | — |
| `prompt_structuring` | `true` on t2v, i2v, extend, music-video; **absent** on v2v and music | bool | [text-to-video.yaml:89](workflow-definitions/text-to-video.yaml#L89), [image-to-video.yaml:74](workflow-definitions/image-to-video.yaml#L74), [extend-video.yaml:71](workflow-definitions/extend-video.yaml#L71), [music-video.yaml:71](workflow-definitions/music-video.yaml#L71) | those four | execution |
| `settings.seed` (UI) | `true` on t2v, i2v, music; `false` on v2v, extend, music-video | bool | t2v:47, i2v:40, v2v:63, extend:51, music-video:51, music:36 | — | — |
| `settings.prompt_modes` | `true` on t2v, i2v only | bool | [text-to-video.yaml:54](workflow-definitions/text-to-video.yaml#L54), [image-to-video.yaml:48](workflow-definitions/image-to-video.yaml#L48) | — | — |
| `settings.quality` / `motion_strength` / `prompt_adherence` | `false` on every workflow (except `music.prompt_adherence: true`) | bool | see per-file lines in §3.6 rows above | — | — |
| `supported_durations` | t2v `["5s","10s","15s","30s","60s"]`; i2v same; extend `["5s","10s","15s","30s","60s","2m","5m"]`; v2v `[]`; music-video `[]`; music `["1m","2m","3m","4m","5m"]` | list | [text-to-video.yaml:37](workflow-definitions/text-to-video.yaml#L37), [image-to-video.yaml:32](workflow-definitions/image-to-video.yaml#L32), [extend-video.yaml:41](workflow-definitions/extend-video.yaml#L41), [video-to-video.yaml:52](workflow-definitions/video-to-video.yaml#L52), [music-video.yaml:43](workflow-definitions/music-video.yaml#L43), [music.yaml:26](workflow-definitions/music.yaml#L26) | — | request |
| `max_size_mb` per input | i2v image 25; v2v video 512, reference image 25; extend video 512; music-video audio 200 | int | i2v:25, v2v:29/45, extend:31, music-video:37 | — | — |
| Director `WORDS_PER_SECOND` | `2.0` | float | [plan.py:36](apps/worker/worker/director/plan.py#L36) | Director | no |
| Director `ESTABLISH_SECONDS` | `2.5` | float | [plan.py:46](apps/worker/worker/director/plan.py#L46) | Director | no |
| Director `_BUDGET_SLACK` | `1.15` | float | [plan.py:50](apps/worker/worker/director/plan.py#L50) | Director | no |
| Director `TARGET_SECONDS_PER_LINE` | `4.0` | float | [plan.py:76](apps/worker/worker/director/plan.py#L76) | Director | no |
| Director `_MINIMUM_LINES` | `2` | int | [plan.py:79](apps/worker/worker/director/plan.py#L79) | Director | no |
| Director `MAX_SILENT_GAP` | `6.0` | float | [plan.py:87](apps/worker/worker/director/plan.py#L87) | Director | no |
| Director `MAX_CHARACTERS` | `4` | int | [plan.py:120](apps/worker/worker/director/plan.py#L120) | Director | no |
| Director `_MAX_EVENTS` | `24` | int | [plan.py:122](apps/worker/worker/director/plan.py#L122) | Director | no |
| Director continuity cap | first `6` facts | int | [plan.py:708](apps/worker/worker/director/plan.py#L708) `seen[:6]` | Director | no |
| Director `max_new_tokens` | `1600` | int | [provider.py:360](apps/worker/worker/director/provider.py#L360) | local Gemma | no |
| Director attempts | 2 per provider (`for sample in (False, True)`) | — | [provider.py:534](apps/worker/worker/director/provider.py#L534) | Director | no |
| `DIALOGUE_LANGUAGES` | `("auto", "english", "spanish", "french", "german", "russian")` | tuple | [provider.py:63](apps/worker/worker/director/provider.py#L63) | Director | request |
| `_MAX_FACTS_CHARS` | `900` | int | [vision.py:62](apps/worker/worker/director/vision.py#L62) | Director I2V | no |
| `_MAX_IDENTITY_CHARS` | `350` | int | [vision.py:89](apps/worker/worker/director/vision.py#L89) | v2v identity | no |
| Progress band | `GENERATE_FROM = 15`, `GENERATE_TO = 85` | int | [progress.py:41-42](apps/worker/worker/longform/progress.py#L41-L42) | all | no |
| `_MARKERS` | see §3.8 | list | [ltx.py:1002-1009](apps/worker/worker/adapters/ltx.py#L1002-L1009) | all | no |
| `_MAX_PULL_FRACTION` | `0.2` | float | [timing.py:32](apps/worker/worker/longform/timing.py#L32) | music-video | argument |
| `_MIN_WINDOW_SECONDS` | `2.0`, capped at `per_pass / 2` | float | [timing.py:38](apps/worker/worker/longform/timing.py#L38), [timing.py:41-42](apps/worker/worker/longform/timing.py#L41-L42) | music-video | no |

### 3.7 Pass-ceiling constants

| Setting | Value | Type | Defined In (file:line) | Applies To | Overridable? |
|---|---|---|---|---|---|
| `_AUDIO_LANDING_FRAMES` | `481` | int | [ltx.py:505](apps/worker/worker/adapters/ltx.py#L505) | a2vid | no |
| `_AUDIO_PASS_SECONDS` | `481 / 24.0` = `20.041666666666668` | float | [ltx.py:506](apps/worker/worker/adapters/ltx.py#L506) | a2vid | `execution.audio_pass_seconds` |
| `_TRANSFORM_PASS_SECONDS` | `8.0` | float | [ltx.py:524](apps/worker/worker/adapters/ltx.py#L524) | ic_lora | `execution.transform_pass_seconds` |
| `_GUIDED_PASS_SECONDS` | `5.0` | float | [ltx.py:538](apps/worker/worker/adapters/ltx.py#L538) | guided | `execution.guided_pass_seconds` |
| `_UNMEASURED_CEILING` | `10.0` | float | [ltx.py:242](apps/worker/worker/adapters/ltx.py#L242) | distilled, any grid absent from `_GRID_CEILINGS` | no |

### 3.8 Progress markers (verbatim)

[ltx.py:1002-1009](apps/worker/worker/adapters/ltx.py#L1002-L1009):

```python
_MARKERS: list[tuple[str, int, str]] = [
    ("Building text encoder", 20, "Understanding your prompt…"),
    ("Running denoising loop", 40, "Generating your video…"),
    ("Building video encoder + spatial upsampler", 55, "Adding detail…"),
    ("Running denoising loop", 70, "Refining your video…"),
    ("Building video decoder", 80, "Rendering the final video…"),
    ("saved to", 85, "Almost done…"),
]
```

These strings are matched against the pipeline's stdout, forward-only ([ltx.py:1020-1030](apps/worker/worker/adapters/ltx.py#L1020-L1030)). They are the repository's only record of the LTX pipeline's own log vocabulary.

---

## 3.1 (spec numbering) Preset definitions

```
Preset name:            — NOT PRESENT
Defined in:             — no preset table exists anywhere in the repository
Applies to workflows:   —
Full setting block:     —
```

Confirmed absent by: `supported_quality_levels: []` in all six YAMLs; `settings.quality: false` in all five video YAMLs; no dict, enum, or table keyed on preset names in `apps/worker/`, `apps/api/`, `packages/`, or `workflow-definitions/`.

**The four pipeline tiers occupy the role a preset block would.** Each is one frozen `LtxPipeline` dataclass. Pasted verbatim below with every field resolved.

The dataclass, verbatim ([ltx.py:805-885](apps/worker/worker/adapters/ltx.py#L805-L885), field declarations only):

```python
@dataclass(frozen=True)
class LtxPipeline:
    module: str
    transformer_key: str | None
    distilled_lora: bool = False
    quantize: bool = True
    offload_cpu: bool = False
    extra_weights: tuple[str, ...] = ()
    conforming_only: bool = False
    measured_landings: tuple[int, ...] = ()
    stage_1_only: bool = False
```

**Tier: `_DISTILLED`** — [ltx.py:889](apps/worker/worker/adapters/ltx.py#L889)
Applies to: text-to-video, image-to-video, extend-video, music-video (default), video-to-video (when `v2v_engine` ≠ `transform`)
```python
_DISTILLED = LtxPipeline(module="ltx_pipelines.distilled", transformer_key=None)
```
Resolved: `module="ltx_pipelines.distilled"`, `transformer_key=None` (→ nvfp4 file), `distilled_lora=False`, `quantize=True`, `offload_cpu=False`, `extra_weights=()`, `conforming_only=False`, `measured_landings=()`, `stage_1_only=False`.
Frame-count rule: full `safe_frame_count` tables (§7).

**Tier: `_IC_LORA`** — [ltx.py:893-909](apps/worker/worker/adapters/ltx.py#L893-L909)
Applies to: video-to-video with `v2v_engine: transform` (the committed configuration)
```python
_IC_LORA = LtxPipeline(
    module="ltx_pipelines.ic_lora",
    transformer_key="transformer_bf16",
    quantize=False,
    offload_cpu=True,
    extra_weights=("union_control_lora",),
    stage_1_only=True,
    conforming_only=True,
    measured_landings=(193,),
)
```
Effects: no `--quantization`; `--offload cpu`; grid doubled and `--skip-stage-2` appended; every pass renders exactly **193 frames** for any request ≤193, then is trimmed back.

**Tier: `_A2VID`** — [ltx.py:914-965](apps/worker/worker/adapters/ltx.py#L914-L965)
Applies to: music-video with `audio_conditioning: true` (currently commented out)
```python
_A2VID = LtxPipeline(
    module="ltx_pipelines.a2vid_two_stage",
    transformer_key="transformer_dev",
    distilled_lora=True,
    quantize=False,
    offload_cpu=True,
    extra_weights=("transformer_dev", "distilled_lora"),
    conforming_only=True,
    measured_landings=(121, 241, 385, _AUDIO_LANDING_FRAMES),
)
```
`measured_landings` resolves to `(121, 241, 385, 481)`.
The sweep recorded in the constant's comment, verbatim ([ltx.py:935-937](apps/worker/worker/adapters/ltx.py#L935-L937)):
```
#   PASS  65 · 121 · 193 · 241 · 385 · 433 · 481 · 505 · 577 · 721 · 961 · 1201
#   FAIL  289 · 337 · 361 · 409 · 457 · 841   (CUBLAS — a shape property)
#   FAIL  601 · 1081                          (out of memory — not one)
```

**Tier: `_GUIDED`** — [ltx.py:978-995](apps/worker/worker/adapters/ltx.py#L978-L995)
Applies to: text-to-video / image-to-video with `generation_engine: guided` (currently commented out)
```python
_GUIDED = LtxPipeline(
    module="ltx_pipelines.ti2vid_two_stages",
    transformer_key="transformer_dev",
    distilled_lora=True,
    quantize=False,
    offload_cpu=True,
    extra_weights=("transformer_dev", "distilled_lora"),
    conforming_only=True,
    measured_landings=(121,),
)
```

**Comparison of the four tiers, fully resolved:**

| Field | `_DISTILLED` | `_IC_LORA` | `_A2VID` | `_GUIDED` |
|---|---|---|---|---|
| `module` | `ltx_pipelines.distilled` | `ltx_pipelines.ic_lora` | `ltx_pipelines.a2vid_two_stage` | `ltx_pipelines.ti2vid_two_stages` |
| transformer file | `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | `ltx-2.5-22b-distilled-transformer-bf16.safetensors` | `ltx-2.5-22b-dev-transformer-bf16.safetensors` | `ltx-2.5-22b-dev-transformer-bf16.safetensors` |
| `--quantization` emitted | **yes**, `nvfp4-prequant` | no | no | no |
| `--offload cpu` | no | **yes** | **yes** | **yes** |
| `--distilled-lora` | no | no | **yes**, strength `1.0` | **yes**, strength `1.0` |
| `--lora` | no | **yes**, Union Control | no | no |
| guidance flags available | **no** | **no** | **yes** (unset by default) | **yes** (unset by default) |
| `--skip-stage-2` | no | **yes** | no | no |
| grid doubling | no | **yes** (`w*2, h*2`) | no | no |
| frame rule | `safe_frame_count` tables | `8k+1` → landings `(193,)` | `8k+1` → landings `(121,241,385,481)` | `8k+1` → landings `(121,)` |
| default pass ceiling | grid ceiling, ≤ `ltx_max_seconds` 60 s | `min(grid ceiling, 8.0 s)` | `20.041666…` s | `5.0` s |

---

## 3.2 (spec numbering) Distilled vs dev/full logic

**Where the code chooses between checkpoints or engines — four decision points.**

**(1) Which transformer, inside a pipeline that has no fixed one** — [ltx.py:2639-2641](apps/worker/worker/adapters/ltx.py#L2639-L2641), verbatim:

```python
def _transformer_file(self) -> str:
    key = "transformer_nvfp4" if "nvfp4" in settings.ltx_quantization else "transformer_bf16"
    return _MODEL_FILES[key]
```

Applied at [ltx.py:2685-2689](apps/worker/worker/adapters/ltx.py#L2685-L2689):

```python
transformer = (
    self._transformer_file()
    if pipeline.transformer_key is None
    else self._optional_weight(pipeline.transformer_key)
)
```

`pipeline.transformer_key is None` is true **only for `_DISTILLED`**. Every other tier names its transformer explicitly.

**(2) Guided vs distilled on T2V/I2V** — [ltx.py:1090](apps/worker/worker/adapters/ltx.py#L1090) and [ltx.py:1159](apps/worker/worker/adapters/ltx.py#L1159), verbatim:

```python
guided = job.execution.get("generation_engine") == "guided"
...
pipeline=_GUIDED if guided else _DISTILLED
```

Note this is a strict `==` against the literal string `"guided"` — no strip, no case-fold.

**(3) Transform vs restyle on V2V** — [ltx.py:1382](apps/worker/worker/adapters/ltx.py#L1382), verbatim:

```python
if str(job.execution.get("v2v_engine") or "").strip() == "transform":
    return await self._run_transform(
        job, reporter, staged, source, target_seconds, reference, grid
    )
```

(This one *does* `.strip()` and coerce to `str`.)

**(4) Audio tier vs distilled on Music Video** — [ltx.py:1858](apps/worker/worker/adapters/ltx.py#L1858) and [ltx.py:1903-1904](apps/worker/worker/adapters/ltx.py#L1903-L1904), verbatim:

```python
audio_conditioned = bool(job.execution.get("audio_conditioning"))
...
pipeline=_A2VID if audio_conditioned else _DISTILLED,
audio=audio_window if audio_conditioned else None,
```

**Which settings change between branches:** see the four-column table at the end of §3.1 above. Summarised as a diff of emitted argv: `_DISTILLED` emits `--quantization nvfp4-prequant`; every other tier drops it and adds `--offload cpu`. `_A2VID`/`_GUIDED` add `--distilled-lora <path> 1.0` and unlock four conditional flags. `_IC_LORA` adds `--lora <path> <strength>` and `--skip-stage-2` and doubles `--width`/`--height`.

**The stated rule linking LoRA to quantization** — [ltx.py:814-822](apps/worker/worker/adapters/ltx.py#L814-L822), verbatim:

> **The quantization rule is not a preference.** Whenever a LoRA is loaded, quantization is dropped entirely and the unquantized model is fitted with `--offload cpu` instead. LoRA + FP8/NVFP4 fusion reaches Triton kernels that do not exist for these shapes; the client's own reference engine forces quantization to none on exactly this condition, and every unexplained "resolution ceiling" this project chased in the audio tier turned out to be this clash rather than a limit of the card. Measured 17 Aug 2026: 1024x576 audio-conditioned, which had never once completed quantized, renders clean.

**Exact internal terminology strings used as identifiers** (not prose):

| String | Meaning | Where it must match |
|---|---|---|
| `"guided"` | value of `execution.generation_engine` | [ltx.py:1090](apps/worker/worker/adapters/ltx.py#L1090) |
| `"transform"` | value of `execution.v2v_engine` | [ltx.py:1382](apps/worker/worker/adapters/ltx.py#L1382), [video-to-video.yaml:119](workflow-definitions/video-to-video.yaml#L119) |
| `"nvfp4-prequant"` | value of `LTX_QUANTIZATION`; the substring `"nvfp4"` selects the transformer | [config.py:271](apps/worker/worker/core/config.py#L271), [ltx.py:2640](apps/worker/worker/adapters/ltx.py#L2640) |
| `"ltx"` | value of `execution.runtime`, and a member of `RUNTIMES` | [registry.py:29](apps/worker/worker/adapters/registry.py#L29) |
| `"mock"`, `"harness"`, `"music"` | the other three registered runtimes | [registry.py:21-36](apps/worker/worker/adapters/registry.py#L21-L36) |
| `"director"` | value of `parameters.prompt_mode` | [provider.py:436](apps/worker/worker/director/provider.py#L436) |
| `"standard"` | the other prompt mode | [workflow_registry.py:286](apps/api/app/services/workflow_registry.py#L286) |
| `"distilled"`, `"dev"` | appear **only** inside checkpoint filenames and dict keys `transformer_nvfp4` / `transformer_bf16` / `transformer_dev` / `distilled_lora` — never as a user- or config-facing tier name | [ltx.py:169-197](apps/worker/worker/adapters/ltx.py#L169-L197) |
| `"sft"`, `"fast"`, `"pro"`, `"balanced"`, `"quality"` | **— not present** as identifiers anywhere in the repository | — |
---

## 4. Configuration Precedence

### 4.0 The four layers, in order of application

```
LAYER 1  Python module constants        apps/worker/worker/adapters/ltx.py (and media/*.py)
                                        — the LAST-RESORT default of every tuning dial

LAYER 2  WorkerSettings (pydantic)      apps/worker/worker/core/config.py
                                        — read from process env, then repo-root .env
                                        — field name uppercased is the env var name

LAYER 3  Workflow YAML `execution:`     workflow-definitions/*.yaml
                                        — parsed by ExecutionSpec (extra="allow")
                                        — travels verbatim in the claim payload
                                        — read ONLY via job.execution.get / execution_int / execution_float

LAYER 4  Request parameters             GenerationParameters (extra="forbid")
                                        — duration, aspect_ratio, seed, prompt_mode,
                                          dialogue_language, lyrics, lyrics_language,
                                          quality, motion_strength, prompt_adherence
```

**There is no "CLI builder" layer.** `_command` is not configurable — it reads Layers 1–4 through the objects above and emits argv. There is no argv override file, no `extra_args` key, no environment variable that appends flags.

**Layers do not simply cascade.** Each dial has its own combining rule, written at its own read site. Three distinct shapes exist:

| Shape | Rule | Example |
|---|---|---|
| **Fallback** | `job.execution_float(key, MODULE_CONSTANT)` — YAML wins if present and coercible, else the constant | `v2v_structure_strength` |
| **Clamp chain** | `max(1.0, min(yaml_or_env, env, table))` — the *lowest* wins | `_per_pass_seconds` |
| **Presence gate** | the flag is emitted only if the key exists at all | `guidance_scale`, `negative_prompt`, `enhance_prompt` |

`execution_int` / `execution_float` ([base.py:190-208](apps/worker/worker/adapters/base.py#L190-L208)) swallow `KeyError`, `TypeError` **and `ValueError`** — a malformed YAML value silently produces the default rather than an error.

---

### 4.1 Precedence table

| Setting | YAML | Python default | Env var | CLI builder | Hardcoded | **EFFECTIVE VALUE** | Why it wins |
|---|---|---|---|---|---|---|---|
| `execution.runtime` | `mock` (all six) | `"mock"` ([workflow.py:138](apps/api/app/schemas/workflow.py#L138)) | `RUNTIME` sets the *worker node's* runtime, not the workflow's | — | — | **`mock`** | YAML is the only source; the worker reads `job.execution.get("runtime") or "mock"` ([resolver.py:57](apps/worker/worker/workflows/resolver.py#L57)). Consequence: `LtxAdapter` is never invoked at this commit. |
| `RUNTIMES` (node capability) | — | `""` ([config.py:63](apps/worker/worker/core/config.py#L63)) | `RUNTIMES` | — | — | **`["mock"]`** on this checkout (`.env` sets neither `RUNTIME` nor `RUNTIMES`) | `runtime_list` prepends `settings.runtime` (default `"mock"`) ([config.py:591-596](apps/worker/worker/core/config.py#L591-L596)) |
| `ltx_quantization` | — | `"nvfp4-prequant"` ([config.py:271](apps/worker/worker/core/config.py#L271)) | `LTX_QUANTIZATION` (unset in `.env`) | — | — | **`nvfp4-prequant`** | Only source. Also selects `transformer_nvfp4` via the substring test at [ltx.py:2640](apps/worker/worker/adapters/ltx.py#L2640). Emitted only when `pipeline.quantize` — i.e. `_DISTILLED` only. |
| `ltx_frame_rate` → `--frame-rate` | — | `24` ([config.py:299](apps/worker/worker/core/config.py#L299)) | `LTX_FRAME_RATE` (unset) | — | — | **`24`** | Only source. Also the divisor in `_frame_count`, `_audio_window_seconds`, overshoot maths, and the fps of every control clip and matte. |
| `ltx_repo_dir` (subprocess cwd) | — | `/workspace/ltx2-benchmark` ([config.py:183](apps/worker/worker/core/config.py#L183)) | `LTX_REPO_DIR` (unset; `.env.example:79` shows the same value) | — | — | **`/workspace/ltx2-benchmark`** | Only source |
| `ltx_models_root` | — | `<ltx_repo_dir>/models/ltx-2.5` ([config.py:602-604](apps/worker/worker/core/config.py#L602-L604)) | `LTX_MODEL_DIR` (unset) | — | — | **`/workspace/ltx2-benchmark/models/ltx-2.5`** | `ltx_model_dir or ltx_repo_dir / "models" / "ltx-2.5"` |
| `ltx_max_seconds` | — | `60` ([config.py:279](apps/worker/worker/core/config.py#L279)) | `LTX_MAX_SECONDS` (unset here; `.env.example:80` **suggests `30`**) | — | — | **`60`** | The `.env.example` line is commented out and `.env` does not set it |
| **per-pass seconds, distilled, t2v/i2v** | `max_segment_seconds: 30` | `settings.ltx_max_seconds` = 60 | `LTX_MAX_SECONDS` | — | `_GRID_CEILINGS[(1024,576)] = 60.0` | **`30.0`** | `max(1.0, min(30, 60, 60))` — see the trace below |
| **per-pass seconds, distilled, extend / music-video / v2v-restyle** | *(no key)* | `settings.ltx_max_seconds` = 60 | `LTX_MAX_SECONDS` | — | grid ceiling | **`60.0`** at 16:9/9:16/1:1/4:5; **`30.0`** at 896×512; **`10.0`** at any grid absent from `_GRID_CEILINGS` | `min(60, 60, measured)` |
| **per-pass seconds, transform** | *(no `transform_pass_seconds` key)* | `_TRANSFORM_PASS_SECONDS = 8.0` ([ltx.py:524](apps/worker/worker/adapters/ltx.py#L524)) | — | — | grid ceiling via `_per_pass_seconds` | **`8.0`** | `min(_per_pass_seconds(job, grid), max(1.0, 8.0))` ([ltx.py:1725-1730](apps/worker/worker/adapters/ltx.py#L1725-L1730)) |
| **per-pass seconds, audio tier** | *(commented out)* | `_AUDIO_PASS_SECONDS = 481/24` ([ltx.py:506](apps/worker/worker/adapters/ltx.py#L506)) | `LTX_MAX_SECONDS` caps it | — | — | **`20.041666666666668`** | `max(1.0, min(20.0417, 60))` ([ltx.py:2402-2403](apps/worker/worker/adapters/ltx.py#L2402-L2403)). **The grid ceiling is NOT consulted.** |
| **per-pass seconds, guided tier** | *(commented out)* | `_GUIDED_PASS_SECONDS = 5.0` ([ltx.py:538](apps/worker/worker/adapters/ltx.py#L538)) | `LTX_MAX_SECONDS` caps it | — | — | **`5.0`** | `max(1.0, min(5.0, 60))` ([ltx.py:2414-2415](apps/worker/worker/adapters/ltx.py#L2414-L2415)). Grid ceiling not consulted. |
| `settings.max_segment_seconds` | — | `10` ([config.py:174](apps/worker/worker/core/config.py#L174)) | `MAX_SEGMENT_SECONDS` | — | — | **never applied on any LTX path** | The LTX adapter's fallback is `settings.ltx_max_seconds`, not this. Only `adapters/harness.py:92-93` reads it. See §4.3. |
| `ltx_max_source_seconds` | — | `330.0` ([config.py:302](apps/worker/worker/core/config.py#L302)) | `LTX_MAX_SOURCE_SECONDS` (unset) | — | — | **`330.0`** | v2v and music-video only |
| `ltx_max_extend_source_seconds` | — | `1800.0` ([config.py:320](apps/worker/worker/core/config.py#L320)) | (unset) | — | — | **`1800.0`** | Passed explicitly as `limit_seconds` at [ltx.py:1202](apps/worker/worker/adapters/ltx.py#L1202), overriding the 330.0 default inside `_staged_source` |
| `timeout_seconds` (adapter deadline) | v2v 5400 / extend 7200 / music-video 7200 / music 3600; t2v & i2v absent | `job_timeout_seconds = 1800` ([config.py:123](apps/worker/worker/core/config.py#L123)) | `JOB_TIMEOUT_SECONDS` | — | — | **t2v/i2v `1800`; v2v `5400`; extend `7200`; music-video `7200`** | `claim.get("execution", {}).get("timeout_seconds") or settings.job_timeout_seconds` ([runner.py:128](apps/worker/worker/jobs/runner.py#L128)) — note `or`, so a YAML `0` would fall through |
| `prompt_structuring` | `true` on t2v/i2v/extend/music-video; absent on v2v/music | — | — | — | — | **on for four workflows**, but additionally gated: `and not wants_director(job)` ([ltx.py:1061](apps/worker/worker/adapters/ltx.py#L1061)) | |
| `align_cuts_to_audio` | `true` (music-video) | `True` (the `.get(key, True)` default) | — | — | — | **`True`** | `job.execution.get("align_cuts_to_audio", True)` ([ltx.py:1973](apps/worker/worker/adapters/ltx.py#L1973)) |
| `generation_engine` | *(commented out)* | absent → `_DISTILLED` | — | — | — | **`_DISTILLED`** | `job.execution.get("generation_engine") == "guided"` is `False` |
| `v2v_engine` | **`transform`** | absent → still-conditioned restyle | — | — | — | **`transform`** (`_IC_LORA`) | The one non-default engine actually enabled |
| `v2v_reference_identity` | **`true`** | absent → `False` | — | — | — | **`True`**, but *inert* unless a `reference_image` input is present: `identity = bool(...) and reference is not None` ([ltx.py:1521](apps/worker/worker/adapters/ltx.py#L1521)) | |
| `v2v_person_lock` | *(commented out)* | absent → `False` | — | — | — | **`False`** | |
| `audio_conditioning` | *(commented out)* | absent → `False` | — | — | — | **`False`** — the model is not given the song | |
| `v2v_structure_strength` | *(commented out)* | `0.45` | — | — | — | **`0.45`** — but see §4.4: **unreachable** while `v2v_engine: transform` is set | `execution_float("v2v_structure_strength", 0.45)` at [ltx.py:1417](apps/worker/worker/adapters/ltx.py#L1417), inside `_run_restyle` after the transform early-return |
| `v2v_keyframe_seconds` / `v2v_keyframes` | *(commented out)* | `4.0` / unset | — | — | `(3, 16)` bounds | **`4.0`** — same unreachability as above | |
| `v2v_continuity_strength` | *(commented out)* | `0.85` | — | — | — | **`0.85`** | Read on **both** engines ([ltx.py:1418](apps/worker/worker/adapters/ltx.py#L1418), [ltx.py:1564](apps/worker/worker/adapters/ltx.py#L1564)) |
| `v2v_reference_strength` | *(commented out)* | `0.3` | — | — | — | **`0.3`** — but on the transform+identity path the `identity` branch returns before it is ever used ([ltx.py:1612-1629](apps/worker/worker/adapters/ltx.py#L1612-L1629)) | |
| `v2v_control_strength` | *(commented out)* | `1.0` | — | — | — | **`1.0`** | |
| `v2v_lora_strength` | *(commented out)* | `1.0` | — | — | — | **`1.0`** | |
| `v2v_edge_low` / `v2v_edge_high` | *(commented out)* | `0.1` / `0.4` ([control.py:45-46](apps/worker/worker/media/control.py#L45-L46)) | — | — | — | **`0.1` / `0.4`** | |
| `v2v_identity_anchor_strength` | *(commented out)* | `1.0` | — | — | `min(x, 0.65)` on fallback | **`1.0`** when the composited anchor builds; **`0.65`** when it does not | [ltx.py:1577-1596](apps/worker/worker/adapters/ltx.py#L1577-L1596) |
| `v2v_identity_refresh_strength` | *(commented out — the YAML comment at [:206](workflow-definitions/video-to-video.yaml#L206) says "default 0.2")* | **`0.0`** ([ltx.py:610](apps/worker/worker/adapters/ltx.py#L610)) | — | — | — | **`0.0`** — no reference re-anchor on later passes | The code constant wins; see Appendix C for the doc/code difference |
| `v2v_identity_subject_attention` | *(commented out)* | `0.5` | — | — | — | **`0.5`** | |
| `v2v_identity_describe_reference` | *(commented out)* | `True` via `.get(key, True)` | — | — | — | **`True`** — a reference-person caption is appended to the prompt | [ltx.py:1533](apps/worker/worker/adapters/ltx.py#L1533) |
| `v2v_identity_composited_anchor` | *(not mentioned in the YAML at all)* | `True` via `.get(key, True)` | — | — | — | **`True`** | [ltx.py:1577](apps/worker/worker/adapters/ltx.py#L1577) |
| `v2v_background_attention` | *(commented out)* | `0.5` ([masks.py:67](apps/worker/worker/media/masks.py#L67)) | — | — | — | **`0.5`**, but unreachable — the `person_lock` branch is dead while `v2v_person_lock` is unset | |
| `i2v_reference_strength` | *(not mentioned in any YAML)* | `0.2` ([ltx.py:2452](apps/worker/worker/adapters/ltx.py#L2452)) | — | — | `_TWO_IMAGE_SAFE_FRAMES` gate | **`0.2`**, but the anchor is dropped entirely unless `frames ∈ {120, 240, 360}` | |
| `guidance_scale`, `stg_scale`, `a2v_guidance_scale`, `inference_steps`, `negative_prompt` | *(all commented out)* | **no default in this repo** | — | — | — | **flags not emitted → the LTX pipeline's own defaults apply** | Presence gate at [ltx.py:2761-2768](apps/worker/worker/adapters/ltx.py#L2761-L2768). Values UNKNOWN — see Appendix B. |
| `enhance_prompt` | *(no YAML sets it)* | absent → falsy | — | — | — | **off** — `--enhance-prompt` never emitted | [ltx.py:2769](apps/worker/worker/adapters/ltx.py#L2769) |
| `--width` / `--height` | via `parameters.aspect_ratio` → `_DIMENSIONS` | `_DEFAULT_DIMENSIONS = (1024, 576)` | — | `dimensions or self._requested_dimensions(job)` ([ltx.py:2673](apps/worker/worker/adapters/ltx.py#L2673)) | `grid_for_source` for v2v/extend | **the explicit `dimensions=` argument, always** | Every `_renderer` call site passes `dimensions=`, so the `_requested_dimensions` fallback inside `_command` is unreachable in production |
| `--seed` | `parameters.seed` | `zlib.crc32(job_id:index)` | — | `zlib.crc32(job_id)` fallback at [ltx.py:2683](apps/worker/worker/adapters/ltx.py#L2683) | — | **`_seed_for_step(job, step.index)`** | The renderer always passes `seed=` ([ltx.py:2589](apps/worker/worker/adapters/ltx.py#L2589)), so `_command`'s own fallback is unreachable in production |
| `--num-frames` | — | `self._frame_count(seconds)` fallback at [ltx.py:2677](apps/worker/worker/adapters/ltx.py#L2677) | — | `num_frames=` argument | frame tables | **the `num_frames=` argument from `_renderer`** | The renderer always passes it ([ltx.py:2590](apps/worker/worker/adapters/ltx.py#L2590)) |
| `--prompt` | `job.prompt` fallback at [ltx.py:2723](apps/worker/worker/adapters/ltx.py#L2723) | — | — | `prompt=` argument | — | **T2V/I2V/extend/music-video: the `prompt=` argument. V2V (both engines): `job.prompt`, because `prompt_for_step` is never passed** | [ltx.py:1468](apps/worker/worker/adapters/ltx.py#L1468) and [ltx.py:1736-1742](apps/worker/worker/adapters/ltx.py#L1736-L1742) omit `prompt_for_step` |
| `execution.output_content_type` | `image/png` (all six) | `_OUTPUT_CONTENT_TYPE[output_type]` map | — | — | `AdapterResult.content_type = "video/mp4"` ([ltx.py:3123](apps/worker/worker/adapters/ltx.py#L3123)) | **the upload is *signed and PUT* as `image/png`; the asset row is *recorded* as whatever `result.content_type` says** | Signing: [internal.py:225](apps/api/app/api/v1/internal.py#L225). PUT header: [runner.py:165](apps/worker/worker/jobs/runner.py#L165) uses `claim["output_content_type"]`. Completion report: [runner.py:174](apps/worker/worker/jobs/runner.py#L174) uses `result.content_type`. Consistent for the mock adapter (which returns `image/png`, [mock.py:86-87](apps/worker/worker/adapters/mock.py#L86-L87)); see Appendix C. |
| `execution.output_kind` | `image` (all six) | — | — | — | — | **`image`**, and it is *read by the LTX adapter*: `job.execution.get("output_kind") == "audio"` is the audio-refusal test ([ltx.py:2109](apps/worker/worker/adapters/ltx.py#L2109)) | |
| `parameters.quality` / `motion_strength` / `prompt_adherence` | — | `None` / `60` / `75` ([generation.py:31-34](apps/api/app/schemas/generation.py#L31-L34)) | — | — | — | **never read by any LTX code path** | `grep -rn 'motion_strength\|prompt_adherence' apps/worker/worker` → 0 hits. Stated in the YAML at [video-to-video.yaml:57-59](workflow-definitions/video-to-video.yaml#L57-L59): "These public controls did reach the job, but the distilled adapter never read them." |

---

### 4.2 Traced reasoning for the clamp chains

**`_per_pass_seconds`** — the single most consequential chain. Verbatim, [ltx.py:2291-2317](apps/worker/worker/adapters/ltx.py#L2291-L2317):

```python
def _per_pass_seconds(
    self, job: AdapterJob, dimensions: tuple[int, int] | None = None
) -> float:
    grid = dimensions or self._requested_dimensions(job)
    measured = _GRID_CEILINGS.get(grid, _UNMEASURED_CEILING)
    requested = float(job.execution_int("max_segment_seconds", settings.ltx_max_seconds))
    # Three clamps, all lowering: the workflow's own override, the
    # operational brake, and what this grid was actually measured at.
    # `settings.ltx_max_seconds` is kept in the chain so one environment
    # variable can still pull every shape down mid-incident without a
    # deploy — that lever saved the product on 14 Aug and must not be
    # quietly removed by making the ceiling per-shape.
    return max(1.0, min(requested, float(settings.ltx_max_seconds), measured))
```

Evaluation, text-to-video at 16:9 with the committed YAML:
1. `grid = (1024, 576)` (explicit, from `_requested_dimensions`).
2. `measured = _GRID_CEILINGS[(1024, 576)] = 60.0`.
3. `requested = float(execution_int("max_segment_seconds", 60)) = float(30) = 30.0` — the YAML key exists and is `30`.
4. `min(30.0, 60.0, 60.0) = 30.0`; `max(1.0, 30.0) = **30.0**`.

Evaluation, extend-video at a 16:9 source:
1. `grid = grid_for_source(1920, 1080) = (1024, 576)`.
2. `measured = 60.0`.
3. `requested = float(execution_int("max_segment_seconds", 60)) = 60.0` — the key is **absent**, so the default `settings.ltx_max_seconds` applies.
4. `min(60.0, 60.0, 60.0) = **60.0**`.

Evaluation, any source whose aspect selects an unmeasured grid (e.g. a 4:3 upload → a grid absent from `_GRID_CEILINGS`):
`measured = _UNMEASURED_CEILING = 10.0` → the pass ceiling is **`10.0`**, regardless of YAML or env.

**`_audio_pass_seconds`** — verbatim, [ltx.py:2402-2403](apps/worker/worker/adapters/ltx.py#L2402-L2403):

```python
requested = job.execution_float("audio_pass_seconds", _AUDIO_PASS_SECONDS)
return max(1.0, min(requested, float(settings.ltx_max_seconds)))
```
→ `max(1.0, min(20.041666666666668, 60.0))` = **`20.041666666666668`**.
Note: `_GRID_CEILINGS` and `execution.max_segment_seconds` are **not** consulted on this path.

**`_guided_pass_seconds`** — [ltx.py:2414-2415](apps/worker/worker/adapters/ltx.py#L2414-L2415), same shape → **`5.0`**.

**Transform pass ceiling** — verbatim, [ltx.py:1725-1730](apps/worker/worker/adapters/ltx.py#L1725-L1730):

```python
per_pass = min(
    self._per_pass_seconds(job, grid),
    max(1.0, job.execution_float(
        "transform_pass_seconds", _TRANSFORM_PASS_SECONDS
    )),
)
```
→ `min(60.0, max(1.0, 8.0))` = **`8.0`** on a measured grid; `min(10.0, 8.0)` = **`8.0`** on an unmeasured one.

**Director extension clamp** — [ltx.py:1276-1283](apps/worker/worker/adapters/ltx.py#L1276-L1283):

```python
per_pass = self._per_pass_seconds(job, grid)
if continuation_plan is not None:
    per_pass = min(per_pass, 30.0)
```
→ a Director-lineage extension gets **`30.0`**; a plain extension keeps **`60.0`**.

**Frame count** — the final arbiter of `--num-frames`. Verbatim, [ltx.py:2507-2537](apps/worker/worker/adapters/ltx.py#L2507-L2537):

```python
requested_frames = self._frame_count(step.seconds)
conditioned = bool(items) or control is not None or audio is not None
if pipeline.conforming_only:
    frames = conforming_frames(requested_frames)
    landing = next(
        (c for c in pipeline.measured_landings if c >= frames), None
    )
    if landing is not None:
        frames = landing
else:
    frames = safe_frame_count(
        dimensions, requested_frames, conditioned=conditioned
    )
```

`conditioned` is `True` if the pass carries **any** of: an `--image`, a `--video-conditioning`, or an `--audio-path`. Recorded at [ltx.py:2511-2521](apps/worker/worker/adapters/ltx.py#L2511-L2521) as having been the cause of a production failure when it was computed as `bool(items)` alone.

`measured_landings` is scanned **ascending** and takes the first entry `>= conforming_frames(requested)`. A request above the largest landing falls through to the plain lattice value ([ltx.py:866-869](apps/worker/worker/adapters/ltx.py#L866-L869)).

**Prompt** — the full chain and where each layer can be overwritten:

```
1. job.prompt                             ← exactly what the user typed
2. if execution.prompt_structuring and not wants_director(job):
       job = replace(job, prompt=structure_prompt(job.prompt))      ltx.py:1061-1062
   (structure_prompt returns the input UNCHANGED when the text already
    matches ^\s*(persistent|section \d+|continuity)\s*:            enhance.py:70-72, 91-92)
3. V2V identity only:
       job = replace(job, prompt=f"{job.prompt}\n\nThe person is …")  ltx.py:1553-1561
4. per section:
       Director:   compile_section_prompts(plan, total, total_seconds)   ltx.py:1118-1121
       standard:   plan_section_prompts(job.prompt, total, total_seconds) ltx.py:1122-1124
       V2V:        — neither; prompt_for_step is not supplied
5. _command:
       "--prompt", job.prompt if prompt is None else prompt            ltx.py:2723
```

`plan_section_prompts` returns `[master_prompt]` unchanged when `section_total <= 1` ([prompts.py:97-98](apps/worker/worker/longform/prompts.py#L97-L98)), so a single-pass job's prompt is byte-identical to step 2's output.

---

### 4.3 Settings parsed but never read

| Setting | Parsed at | Read by | Note |
|---|---|---|---|
| `settings.max_segment_seconds` (default `10`) | [config.py:174](apps/worker/worker/core/config.py#L174) | **only** [harness.py:92-93](apps/worker/worker/adapters/harness.py#L92-L93) | The LTX adapter's fallback is `settings.ltx_max_seconds` (60), not this. Its docstring at [config.py:176-179](apps/worker/worker/core/config.py#L176-L179) describes it as the general default. |
| `parameters.motion_strength` (default `60`) | [generation.py:33](apps/api/app/schemas/generation.py#L33) | nothing in `apps/worker/worker/` | `grep -rn motion_strength apps/worker/worker` → 0 hits |
| `parameters.prompt_adherence` (default `75`) | [generation.py:34](apps/api/app/schemas/generation.py#L34) | nothing in `apps/worker/worker/` | 0 hits |
| `parameters.quality` | [generation.py:31](apps/api/app/schemas/generation.py#L31) | nothing in the LTX adapter | `supported_quality_levels` is `[]` everywhere, so `validate_request` rejects any value ([workflow_registry.py:182-190](apps/api/app/services/workflow_registry.py#L182-L190)) |
| `DirectorPlan.tone` | [plan.py:308](apps/worker/worker/director/plan.py#L308) | nothing in `compiler.py` | Parsed, validated, logged nowhere, never compiled into a caption |
| `Segment.overlap_seconds` / `source_start_seconds` / `generate_seconds` / `trim_start_seconds` | [segments.py:40-61](apps/worker/worker/media/segments.py#L40-L61) | `plan_segments` computes it, but **no caller passes a non-zero `overlap_seconds`** | `grep -rn 'overlap_seconds=' apps/worker/worker` → only the parameter declaration and `overlap_seconds=0.0 if index == 0 else overlap`. `render_chain` never supplies one. |
| `AudioMode.GENERATED_MASTER_AUDIO` | [audio.py:41](apps/worker/worker/media/audio.py#L41) | not used by `adapters/ltx.py` | The LTX adapter records only `SOURCE_AUDIO`, `GENERATED_PER_SECTION_AUDIO`, `NO_AUDIO` |
| `_MODEL_FILES["transformer_bf16"]` via `_transformer_file()` | [ltx.py:171](apps/worker/worker/adapters/ltx.py#L171) | unreachable via `_transformer_file` while `ltx_quantization` contains `"nvfp4"` | Still reachable via `_IC_LORA.transformer_key = "transformer_bf16"` → `_optional_weight` |

### 4.4 Settings declared but overwritten, or in dead code paths

| Setting | Situation | Evidence |
|---|---|---|
| `v2v_structure_strength`, `v2v_keyframe_seconds`, `v2v_keyframes` | **Unreachable while `v2v_engine: transform` is set.** `_run_restyle` returns from the transform branch at [ltx.py:1382-1385](apps/worker/worker/adapters/ltx.py#L1382-L1385) *before* these are read at [ltx.py:1406-1417](apps/worker/worker/adapters/ltx.py#L1406-L1417). | Removing `v2v_engine` from the YAML restores them |
| `v2v_reference_strength` on the transform path | Read at [ltx.py:1565-1567](apps/worker/worker/adapters/ltx.py#L1565-L1567), but the `conditioning` closure's `identity` branch returns before reaching it ([ltx.py:1612-1629](apps/worker/worker/adapters/ltx.py#L1612-L1629)). Used only when `v2v_reference_identity` is off **and** a reference image was supplied. | |
| `v2v_background_attention` | Reachable only inside `if not person_lock: return None` … i.e. only when `v2v_person_lock` is true, which no YAML sets | [ltx.py:1703-1719](apps/worker/worker/adapters/ltx.py#L1703-L1719) |
| `v2v_identity_refresh_strength` | Read, but `if refresh_strength > 0:` is `False` at the default `0.0`, so the second `--image` is never appended | [ltx.py:1622-1628](apps/worker/worker/adapters/ltx.py#L1622-L1628) |
| `_command`'s `seed is None` fallback | `zlib.crc32(job.job_id.encode())` at [ltx.py:2683](apps/worker/worker/adapters/ltx.py#L2683) — the renderer always passes `seed=self._seed_for_step(job, step.index)` | [ltx.py:2589](apps/worker/worker/adapters/ltx.py#L2589) |
| `_command`'s `num_frames is None` fallback | `self._frame_count(seconds)` at [ltx.py:2677](apps/worker/worker/adapters/ltx.py#L2677) — the renderer always passes `num_frames=frames` | [ltx.py:2590](apps/worker/worker/adapters/ltx.py#L2590) |
| `_command`'s `dimensions is None` fallback | `self._requested_dimensions(job)` at [ltx.py:2673](apps/worker/worker/adapters/ltx.py#L2673) — every `_renderer` call passes `dimensions=` | ltx.py:1155, 1290, 1468, 1737, 1900 |
| `_command`'s `skip_stage_2` parameter | `_renderer` is never called with `skip_stage_2=True` by any handler; the flag is set only indirectly by `pipeline.stage_1_only` inside `_command` | [ltx.py:2690-2695](apps/worker/worker/adapters/ltx.py#L2690-L2695); `grep 'skip_stage_2=True'` → only that assignment |
| `_GRID_CEILINGS[(896, 512)] = 30.0` and the other "previous grids" | Reachable only through `grid_for_source`, i.e. only on v2v/extend with a source whose aspect resolves there | [ltx.py:233-236](apps/worker/worker/adapters/ltx.py#L233-L236) |
| `_BAD_FRAME_BANDS[False][(768, 768)]` etc. | Consulted only when `conditioned=False`, which for chained workflows is true only on pass 1 | [ltx.py:318-331](apps/worker/worker/adapters/ltx.py#L318-L331) |
| Every `_MARKERS` entry | Matched against the LTX pipeline's stdout. If the installed pipeline's log lines differ, progress silently never advances (no error path) | [ltx.py:2961-2963](apps/worker/worker/adapters/ltx.py#L2961-L2963) |

### 4.5 Settings whose defaults are never applied because a caller always passes a value

- `_command(dimensions=…)`, `_command(seed=…)`, `_command(num_frames=…)` — as above.
- `_staged_source(limit_seconds=…)`: the default is `settings.ltx_max_source_seconds` (330.0), but `_run_extension` always passes `limit_seconds=float(settings.ltx_max_extend_source_seconds)` (1800.0) ([ltx.py:1198-1203](apps/worker/worker/adapters/ltx.py#L1198-L1203)). So `_staged_source`'s own default applies only to v2v and music-video.
- `_per_pass_seconds(dimensions=None)`: every call site in `ltx.py` passes an explicit grid (lines 1153, 1275, 1463, 1726, 1862), so the `_requested_dimensions` fallback never runs.
- `_launcher(module=_DISTILLED.module)`: `_command` always passes `pipeline.module` ([ltx.py:2698](apps/worker/worker/adapters/ltx.py#L2698)).
- `plan_segments(overlap_seconds=0.0)`: no caller supplies one.

### 4.6 The committed-state divergence (recorded as fact, not as a problem)

Three independent statements in the repository describe the LTX runtime as reachable:

- [ltx.py:1](apps/worker/worker/adapters/ltx.py#L1): *"LTX-2.5 runtime — real GPU generation for every video workflow."*
- [video-to-video.yaml:115](workflow-definitions/video-to-video.yaml#L115): *"ENABLED 17 Aug 2026."*
- [video-to-video.yaml:168](workflow-definitions/video-to-video.yaml#L168): *"GPU-verified the day it was built (research-2026-08-19)."*

And three describe it as not reachable:

- [registry.py:26-28](apps/worker/worker/adapters/registry.py#L26-L28): *"Registered but not yet routed: no shipped workflow says `runtime: ltx`."*
- Every YAML's `runtime: mock` plus the `output_content_type: image/png` block whose comment says *"M1 runs the mock runtime … Remove both when M2 wires up a real provider."*
- [test_worker_protocol.py:553](apps/api/tests/test_worker_protocol.py#L553), which asserts an `ltx`-only worker claims nothing.

**Effective value at commit `3bd8016`: `runtime: mock` for all six workflows.** The LTX adapter is registered, fully tested, and unreachable. Everything Sections 5–12 describe is what *would* execute if `execution.runtime` were `ltx` and the weights were present. This is stated once here and not repeated.
---

## 5. Exact Invocation Snapshots

### 5.0 Method, and how to read these

**Method used: dry-run of the argument builder.** `LtxAdapter._command` was imported and called directly with fully-resolved arguments, in-process, on the worker's own virtualenv. No repository file was modified and no instrumentation was added to the repository — the driver script lives entirely outside the repo, in the session scratchpad. The frame-count logic of `_renderer` ([ltx.py:2523-2537](apps/worker/worker/adapters/ltx.py#L2523-L2537)) was reproduced line-for-line in the driver rather than mocked, because `_renderer.render` is an `async` closure that would otherwise require spawning the pipeline. Every other value — grid, pass ceiling, section plan, per-section prompt, seed, conditioning triples — comes from calling the repository's own functions (`_requested_dimensions`, `_per_pass_seconds`, `_audio_pass_seconds`, `_guided_pass_seconds`, `plan_chain_segments`, `plan_section_prompts`, `structure_prompt`, `_seed_for_step`, `_identity_anchor`, `_audio_window_seconds`, `safe_frame_count`, `conforming_frames`, `grid_for_source`).

**Two substitutions were necessary and are declared:**

1. `pathlib.Path.exists` was made unconditionally `True` for the duration of the dry-run, so `_optional_weight` ([ltx.py:2816](apps/worker/worker/adapters/ltx.py#L2816)) would report the **production** paths rather than refuse for absent local weights. No file was created.
2. The dry-run ran on Windows, so `pathlib` rendered `/` as `\`. **Every path below has been transcribed back to forward slashes.** Nothing else was altered.

**Fixed inputs used for every snapshot** (so the snapshots are comparable):

| Input | Value |
|---|---|
| `job_id` | `7f3a1c22-0000-4000-8000-abcdef012345` |
| `workspace` | `/workspace/job` |
| `parameters.seed` | *(not supplied — so seeds are `crc32(f"{job_id}:{index}")`)* |
| resulting seed, pass 0 | `1148858713` |
| resulting seed, pass 1 | `863830479` |
| `settings.ltx_repo_dir` | `/workspace/ltx2-benchmark` (default) |
| `settings.ltx_models_root` | `/workspace/ltx2-benchmark/models/ltx-2.5` (default) |
| `settings.ltx_quantization` | `nvfp4-prequant` (default) |
| `settings.ltx_frame_rate` | `24` (default) |
| `settings.ltx_max_seconds` | `60` (default) |
| T2V/I2V prompt (pre-structuring) | `A rain-soaked neon alley at night. Two cars idle at the kerb, one matte black and one red. Steam rises from a grate.` |
| V2V source | `1920×1080` (→ `grid_for_source` = `(1024, 576)`) |
| Music-video track | `180.0` s |

`MODELS` below abbreviates `/workspace/ltx2-benchmark/models/ltx-2.5`. **The abbreviation is only in this report** — the real argv carries the full absolute path in every one of those six flags.

**Values injected at runtime (not static) — common to every snapshot:**
- `--seed` — `crc32(job_id:index)` or `(user_seed + index) % 2**31`
- `--output-path` — `<job.workspace>/<prefix>-<index:04d>.mp4`
- every `--image` / `--video-conditioning` / `--conditioning-attention-mask` / `--audio-path` path — files inside `job.workspace`, or the staged upload
- `--prompt` — depends on the user's text and, in Director mode, on a language-model plan
- `--num-frames` — depends on the section length, which for `duration_mode: source` depends on the uploaded file's probed duration
- `--width` / `--height` — for v2v and extend, on the source's own aspect

---

### 5.1 T2V — "fast" preset

**`— not present.`** There is no `fast` preset (§0.2, §3.1). The nearest thing is the default tier, which is §5.2.

### 5.2 T2V — "balanced" preset → the DEFAULT tier (distilled), 5 s

```
Invocation method: subprocess CLI
Built by:          apps/worker/worker/adapters/ltx.py:2652  (LtxAdapter._command)
Launched by:       apps/worker/worker/adapters/ltx.py:2924  (asyncio.create_subprocess_exec, cwd=/workspace/ltx2-benchmark)
Pipeline:          _DISTILLED  (ltx.py:889)
Resolved plan:     grid=(1024,576)  per_pass=30.0s  sections=1  section lengths=[5.0]
Frames:            requested 120 → rendered 121   (conditioned=False; conforming_frames(120)=121)
```

```
uv run python -m ltx_pipelines.distilled \
  --transformer-path MODELS/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
  --text-encoder-path MODELS/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path MODELS/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path MODELS/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --duration-head-path MODELS/model_patches/ltx-2.5-duration-head-bf16.safetensors \
  --spatial-upsampler-path MODELS/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --quantization nvfp4-prequant \
  --prompt 'A rain-soaked neon alley at night. Two cars idle at the kerb, one matte black and one red. Steam rises from a grate.

CONTINUITY (fixed for the entire video):
- Exactly 2 cars appear, and they remain the only cars on screen for the entire video.
- The matte black one stays matte black from the first frame to the last.
- The same subjects keep the same faces, clothing, colours and count in every frame.
- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.' \
  --num-frames 121 \
  --height 576 \
  --width 1024 \
  --frame-rate 24 \
  --seed 1148858713 \
  --output-path /workspace/job/segment-0000.mp4
```

Flags **absent** and why: no `--offload` (`pipeline.offload_cpu` is `False`), no `--distilled-lora`, no `--lora`, no `--negative-prompt` / `--video-cfg-guidance-scale` / `--video-stg-guidance-scale` / `--a2v-guidance-scale` / `--num-inference-steps` (the whole block is gated on `pipeline.distilled_lora`, which is `False`), no `--enhance-prompt`, no `--image` (T2V pass 1 has no conditioning), no `--skip-stage-2`.

After the render, because `121 != 120`: `_trim_to(job, output, 5.0, tolerance_seconds=(121-120)/24 = 0.0417)` re-encodes to exactly 5.000 s with `-c:v libx264 -preset medium -crf 17 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart` ([ltx.py:2865-2884](apps/worker/worker/adapters/ltx.py#L2865-L2884)). Then `verify_output(expect_video=True, expect_audio=True, expected_seconds=5.0)`.

### 5.3 T2V — "quality"/guided preset → `generation_engine: guided`, 5 s

```
Invocation method: subprocess CLI
Built by:          apps/worker/worker/adapters/ltx.py:2652
Pipeline:          _GUIDED  (ltx.py:978)
Resolved plan:     grid=(1024,576)  per_pass=5.0s  sections=1  section lengths=[5.0]
Frames:            requested 120 → rendered 121   (conforming_only; measured_landings=(121,))
Reachable only if a workflow sets `execution.generation_engine: guided`, which no
committed YAML does (commented out at text-to-video.yaml:93).
```

```
uv run python -m ltx_pipelines.ti2vid_two_stages \
  --transformer-path MODELS/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  --text-encoder-path MODELS/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path MODELS/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path MODELS/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --duration-head-path MODELS/model_patches/ltx-2.5-duration-head-bf16.safetensors \
  --spatial-upsampler-path MODELS/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --offload cpu \
  --distilled-lora MODELS/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors 1.0 \
  --prompt '<identical structured prompt to §5.2>' \
  --num-frames 121 \
  --height 576 \
  --width 1024 \
  --frame-rate 24 \
  --seed 1148858713 \
  --output-path /workspace/job/segment-0000.mp4
```

**Note the two differences from §5.2 and nothing else:** `--quantization nvfp4-prequant` is gone; `--offload cpu` and `--distilled-lora … 1.0` are added, and the module changed. The guidance flags this tier makes *available* (`--video-cfg-guidance-scale`, `--video-stg-guidance-scale`, `--a2v-guidance-scale`, `--num-inference-steps`, `--negative-prompt`) are still **not emitted**, because no `execution` key supplies a value ([ltx.py:2761-2768](apps/worker/worker/adapters/ltx.py#L2761-L2768) — each is `if value is not None`).

A 15 s request on this tier becomes **3 sections** (`per_pass = 5.0`); a 60 s request becomes **12 sections with 11 seams** (§6).

### 5.4 I2V — default preset, 5 s

```
Invocation method: subprocess CLI
Built by:          apps/worker/worker/adapters/ltx.py:2652
Pipeline:          _DISTILLED
Resolved plan:     grid=(1024,576)  per_pass=30.0s  sections=1
Frames:            requested 120 → rendered 120  (conditioned=True; 120 ∈ _MEASURED_SAFE_CONDITIONED → passes through untouched)
```

```
uv run python -m ltx_pipelines.distilled \
  --transformer-path MODELS/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
  --text-encoder-path MODELS/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path MODELS/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path MODELS/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --duration-head-path MODELS/model_patches/ltx-2.5-duration-head-bf16.safetensors \
  --spatial-upsampler-path MODELS/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --quantization nvfp4-prequant \
  --prompt '<identical structured prompt to §5.2>' \
  --num-frames 120 \
  --height 576 \
  --width 1024 \
  --frame-rate 24 \
  --seed 1148858713 \
  --output-path /workspace/job/segment-0000.mp4 \
  --image /workspace/job/inputs/source_image.png 0 1.0
```

Note `120` here vs `121` in §5.2 — the *only* difference is the presence of the `--image`, which flips `conditioned` to `True` and routes the count through `_MEASURED_SAFE_CONDITIONED` rather than the lattice.

The input filename comes from `_stage_inputs` ([runner.py:200](apps/worker/worker/jobs/runner.py#L200)): `job.workspace / "inputs" / f"{item.role}{_suffix_for(item.content_type)}"`.

### 5.5 V2V — default preset (the committed configuration: `v2v_engine: transform` + `v2v_reference_identity: true`)

Modelled on a 20.0-second 1920×1080 source with a reference image supplied.

```
Invocation method: subprocess CLI
Built by:          apps/worker/worker/adapters/ltx.py:2652
Pipeline:          _IC_LORA  (ltx.py:893)
Resolved plan:     grid=grid_for_source(1920,1080)=(1024,576)
                   per_pass = min(_per_pass_seconds=60.0, transform_pass_seconds=8.0) = 8.0
                   sections = 3, section lengths = [6.6667, 6.6667, 6.6667]
Frames:            requested 160 → conforming 161 → measured_landings=(193,) → rendered 193
Grid doubling:     stage_1_only=True → --width 2048 --height 1152, plus --skip-stage-2
```

**Pass 1:**

```
uv run python -m ltx_pipelines.ic_lora \
  --transformer-path MODELS/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --text-encoder-path MODELS/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path MODELS/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path MODELS/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --duration-head-path MODELS/model_patches/ltx-2.5-duration-head-bf16.safetensors \
  --spatial-upsampler-path MODELS/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --offload cpu \
  --prompt 'Turn this into a rain-soaked neon street.

The person is <one sentence from worker/director/vision.py:reference_person_facts>. The same person, with the same face, hair and clothing, stays on screen for the whole video.' \
  --num-frames 193 \
  --height 1152 \
  --width 2048 \
  --frame-rate 24 \
  --seed 1148858713 \
  --output-path /workspace/job/transformed-0000.mp4 \
  --image /workspace/job/identity-anchor.png 0 1.0 \
  --lora MODELS/loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors 1.0 \
  --video-conditioning /workspace/job/control-0000.mp4 1.0 \
  --conditioning-attention-mask /workspace/job/attention-0000.mp4 1.0 \
  --skip-stage-2
```

**Pass 2 (the seam):**

```
uv run python -m ltx_pipelines.ic_lora \
  --transformer-path MODELS/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --text-encoder-path MODELS/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path MODELS/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path MODELS/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --duration-head-path MODELS/model_patches/ltx-2.5-duration-head-bf16.safetensors \
  --spatial-upsampler-path MODELS/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --offload cpu \
  --prompt '<the SAME text as pass 1 — v2v never supplies prompt_for_step>' \
  --num-frames 193 \
  --height 1152 \
  --width 2048 \
  --frame-rate 24 \
  --seed 863830479 \
  --output-path /workspace/job/transformed-0001.mp4 \
  --image /workspace/job/transformed-condition-0001.png 0 0.85 \
  --lora MODELS/loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors 1.0 \
  --video-conditioning /workspace/job/control-0001.mp4 1.0 \
  --conditioning-attention-mask /workspace/job/attention-0001.mp4 1.0 \
  --skip-stage-2
```

The reference photo is **not** re-anchored on pass 2, because `v2v_identity_refresh_strength` is `0.0` and the code guards `if refresh_strength > 0:` ([ltx.py:1622](apps/worker/worker/adapters/ltx.py#L1622)).

**The two auxiliary commands each pass runs before the render:**

`extract_edge_control` ([control.py:99-116](apps/worker/worker/media/control.py#L99-L116)), fully resolved for pass 1:
```
ffmpeg -ss 0.000 -t 6.667 -i /workspace/job/inputs/source_video.mp4 \
  -filter:v 'scale=1024:576:force_original_aspect_ratio=increase,crop=1024:576,fps=24,edgedetect=low=0.1:high=0.4,tpad=stop_mode=clone:stop_duration=8.042,format=yuv420p' \
  -frames:v 193 -fps_mode cfr -an \
  -c:v libx264 -preset veryfast -crf 18 \
  /workspace/job/control-0000.mp4
```
Note: the control clip is built at the **un-doubled** grid `(1024, 576)` (`width=grid[0], height=grid[1]` at [ltx.py:1649-1650](apps/worker/worker/adapters/ltx.py#L1649-L1650)) while the render is asked for `2048×1152`.

`build_person_matte` ([masks.py:98-110](apps/worker/worker/media/masks.py#L98-L110)), fully resolved:
```
uv run python /<repo>/apps/worker/scripts/person_matte.py \
  --source /workspace/job/inputs/source_video.mp4 \
  --dest /workspace/job/matte-0000.mp4 \
  --start-seconds 0.000 --duration-seconds 6.667 \
  --width 1024 --height 576 --fps 24 --frames 193 \
  --dilation-passes 2 --feather-radius 12
```
(cwd `/workspace/ltx2-benchmark`.)

`build_identity_anchor` ([masks.py:174-182](apps/worker/worker/media/masks.py#L174-L182)), once per job:
```
uv run python /<repo>/apps/worker/scripts/person_anchor.py \
  --source /workspace/job/inputs/source_video.mp4 \
  --reference /workspace/job/inputs/reference_image.png \
  --dest /workspace/job/identity-anchor.png \
  --start-seconds 0.000 --width 1024 --height 576
```

### 5.5b V2V — the still-conditioned restyle (reached by deleting `v2v_engine` from the YAML)

Same 20.0-second 1920×1080 source, no reference image.

```
Pipeline:          _DISTILLED
Resolved plan:     grid=(1024,576)  per_pass=60.0  sections=1  section lengths=[20.0]
Keyframes:         keyframes_for(20.0) = max(3, min(16, ceil(20.0/4.0))) = 5
Offsets:           [0.0] + [(i+0.5)/5 for i in 0..4] = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
Frames:            requested 480 → conditioned=True → 480 ∉ _MEASURED_SAFE_CONDITIONED,
                   no measured landing in [480, conforming(480)=481], conforming→481,
                   band (361,719,720) → rendered 720
```

```
uv run python -m ltx_pipelines.distilled \
  --transformer-path MODELS/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
  --text-encoder-path MODELS/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path MODELS/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path MODELS/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --duration-head-path MODELS/model_patches/ltx-2.5-duration-head-bf16.safetensors \
  --spatial-upsampler-path MODELS/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --quantization nvfp4-prequant \
  --prompt 'Turn this into an oil painting.' \
  --num-frames 720 \
  --height 576 \
  --width 1024 \
  --frame-rate 24 \
  --seed 1148858713 \
  --output-path /workspace/job/restyled-0000.mp4 \
  --image /workspace/job/keyframes/pass-0000-0000.png 0 0.45 \
  --image /workspace/job/keyframes/pass-0000-0001.png 72 0.45 \
  --image /workspace/job/keyframes/pass-0000-0002.png 216 0.45 \
  --image /workspace/job/keyframes/pass-0000-0003.png 360 0.45 \
  --image /workspace/job/keyframes/pass-0000-0004.png 503 0.45 \
  --image /workspace/job/keyframes/pass-0000-0005.png 647 0.45
```

Six `--image` triples for five keyframes: the leading `0.0` offset is inserted because no continuity or reference frame owns frame 0 ([ltx.py:1443-1444](apps/worker/worker/adapters/ltx.py#L1443-L1444)). Indices are `min(frames - 1, round(offset * (frames - 1)))` against the **rendered** count 720, not the requested 480.

### 5.6 Extend — one section continuation

15-second extension of a 1920×1080 source with no Director lineage.

```
Invocation method: subprocess CLI
Pipeline:          _DISTILLED
Resolved plan:     grid=grid_for_source(1920,1080)=(1024,576)  per_pass=60.0  sections=1
Frames:            requested 360 → conditioned=True → 360 ∈ _MEASURED_SAFE_CONDITIONED → rendered 360 (untouched)
```

```
uv run python -m ltx_pipelines.distilled \
  --transformer-path MODELS/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
  --text-encoder-path MODELS/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path MODELS/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path MODELS/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --duration-head-path MODELS/model_patches/ltx-2.5-duration-head-bf16.safetensors \
  --spatial-upsampler-path MODELS/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --quantization nvfp4-prequant \
  --prompt 'The car pulls away into the rain.

CONTINUITY (fixed for the entire video):
- The same subjects keep the same faces, clothing, colours and count in every frame.
- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.' \
  --num-frames 360 \
  --height 576 \
  --width 1024 \
  --frame-rate 24 \
  --seed 1148858713 \
  --output-path /workspace/job/continuation-0000.mp4 \
  --image /workspace/job/seed-frame.png 0 1.0
```

`seed-frame.png` is produced by `ffmpeg -sseof -1 -i <staged source> -update 1 /workspace/job/seed-frame.png` ([frames.py:43-46](apps/worker/worker/media/frames.py#L43-L46)).

Assembly ([ltx.py:1318-1337](apps/worker/worker/adapters/ltx.py#L1318-L1337)): the continuation is normalised, the **source** is normalised to `part-0000.mp4` at `output_dimensions(1920,1080) = (1920, 1080)` and `_delivery_fps(source)`, and the two are concatenated. Final expectation is `source.duration_seconds + 15.0`, with `tolerance_seconds = duration_tolerance(expected, floor=1.5)`.

### 5.7 Music Video — current configuration (`audio_conditioning` absent → distilled tier)

180-second track, 16:9, `align_cuts_to_audio: true`.

```
Invocation method: subprocess CLI
Pipeline:          _DISTILLED
Resolved plan:     grid=(1024,576)  per_pass=60.0  sections=3  section lengths=[60.0, 60.0, 60.0]
                   (boundaries from plan_musical_boundaries; with synthetic onsets every 3.7 s
                    the pull produced [60.0, 120.0], i.e. the same windows as the even plan)
Frames:            requested 1440 → conditioned=False → conforming 1441 → no band → rendered 1441
```

```
uv run python -m ltx_pipelines.distilled \
  --transformer-path MODELS/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
  --text-encoder-path MODELS/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path MODELS/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path MODELS/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --duration-head-path MODELS/model_patches/ltx-2.5-duration-head-bf16.safetensors \
  --spatial-upsampler-path MODELS/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --quantization nvfp4-prequant \
  --prompt 'LONG-FORM CONTINUATION — SECTION 1 OF 3.
Keep the same subjects, identities, faces, clothing, colours, object counts, vehicles, environment and camera direction established previously.
PERSISTENT USER CONSTRAINTS (verbatim):
A dancer in a neon warehouse.
CONTINUITY (fixed for the entire video):
- The same subjects keep the same faces, clothing, colours and count in every frame.
NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY (verbatim):
- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.
Continue directly from the predecessor frame. Do not replay, restart or summarise any earlier action or dialogue. Complete this section'"'"'s assigned dialogue before the section ends.' \
  --num-frames 1441 \
  --height 576 \
  --width 1024 \
  --frame-rate 24 \
  --seed 1148858713 \
  --output-path /workspace/job/scene-0000.mp4
```

**There is no `--audio-path`, no `--audio-start-time`, no `--audio-max-duration`, and no audio flag of any kind on this command.** The uploaded track reached only `probe_media` (for the total duration) and `audio_onsets` (for the cut points). It is attached at the very end by:

```
ffmpeg -i /workspace/job/picture.mp4 -i /workspace/job/inputs/source_audio.mp3 \
  -map 0:v:0 -map 1:a:0 -c:v copy \
  -c:a aac -b:a 192k -ar 48000 -ac 2 -shortest -movflags +faststart \
  /workspace/job/output.mp4
```
([audio.py:88-108](apps/worker/worker/media/audio.py#L88-L108); the `-c:v copy` branch applies when the picture already covers the track within 0.05 s.)

### 5.8 Audio-conditioned video (`audio_conditioning: true`)

Same 180-second track. **Reachable only by uncommenting [music-video.yaml:98](workflow-definitions/music-video.yaml#L98).**

```
Invocation method: subprocess CLI
Pipeline:          _A2VID  (ltx.py:914)
Resolved plan:     grid=(1024,576)  per_pass=20.041666666666668  sections=9
                   section lengths=[20.0] × 9
Frames:            requested 480 → conforming 481 → measured_landings first ≥481 is 481 → rendered 481
Audio window:      _audio_window_seconds(481) = 481/24 + 0.04 = 20.082 s
```

**Section 1:**

```
uv run python -m ltx_pipelines.a2vid_two_stage \
  --transformer-path MODELS/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  --text-encoder-path MODELS/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path MODELS/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path MODELS/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --duration-head-path MODELS/model_patches/ltx-2.5-duration-head-bf16.safetensors \
  --spatial-upsampler-path MODELS/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --offload cpu \
  --distilled-lora MODELS/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors 1.0 \
  --prompt 'LONG-FORM CONTINUATION — SECTION 1 OF 9.
Keep the same subjects, identities, faces, clothing, colours, object counts, vehicles, environment and camera direction established previously.
PERSISTENT USER CONSTRAINTS (verbatim):
A dancer in a neon warehouse.
CONTINUITY (fixed for the entire video):
- The same subjects keep the same faces, clothing, colours and count in every frame.
NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY (verbatim):
- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.
Continue directly from the predecessor frame. Do not replay, restart or summarise any earlier action or dialogue. Complete this section'"'"'s assigned dialogue before the section ends.' \
  --num-frames 481 \
  --height 576 \
  --width 1024 \
  --frame-rate 24 \
  --seed 1148858713 \
  --output-path /workspace/job/scene-0000.mp4 \
  --audio-path /workspace/job/inputs/source_audio.mp3 \
  --audio-start-time 0.000 \
  --audio-max-duration 20.082
```

**Section 2 (the seam):**

```
uv run python -m ltx_pipelines.a2vid_two_stage \
  --transformer-path MODELS/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  --text-encoder-path MODELS/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path MODELS/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path MODELS/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --duration-head-path MODELS/model_patches/ltx-2.5-duration-head-bf16.safetensors \
  --spatial-upsampler-path MODELS/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --offload cpu \
  --distilled-lora MODELS/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors 1.0 \
  --prompt 'LONG-FORM CONTINUATION — SECTION 2 OF 9.
Keep the same subjects, identities, faces, clothing, colours, object counts, vehicles, environment and camera direction established previously.
PERSISTENT USER CONSTRAINTS (verbatim):
A dancer in a neon warehouse.
CONTINUITY (fixed for the entire video):
- The same subjects keep the same faces, clothing, colours and count in every frame.
NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY (verbatim):
Continue naturally from the preceding section without introducing a new event.
Continue directly from the predecessor frame. Do not replay, restart or summarise any earlier action or dialogue. Complete this section'"'"'s assigned dialogue before the section ends.' \
  --num-frames 481 \
  --height 576 \
  --width 1024 \
  --frame-rate 24 \
  --seed 863830479 \
  --output-path /workspace/job/scene-0001.mp4 \
  --image /workspace/job/scene-condition-0001.png 0 1.0 \
  --audio-path /workspace/job/inputs/source_audio.mp3 \
  --audio-start-time 20.000 \
  --audio-max-duration 20.082
```

The `--audio-path` is the **whole master file**, identical on every pass; only `--audio-start-time` moves. The track is never sliced or re-encoded ([ltx.py:766-780](apps/worker/worker/adapters/ltx.py#L766-L780), [ltx.py:1883-1888](apps/worker/worker/adapters/ltx.py#L1883-L1888)).

The generated audio from these 9 passes is then **discarded** (`audio=False` at [ltx.py:1940](apps/worker/worker/adapters/ltx.py#L1940)) and the original file is muxed over the whole result, exactly as in §5.7.

### 5.9 60-second T2V — section 1 AND section 2, separately

```
Resolved plan:     grid=(1024,576)  per_pass=30.0  sections=2  section lengths=[30.0, 30.0]
                   seams=1, windows 0.0–30.0 and 30.0–60.0
Section 1 frames:  requested 720 → conditioned=False → conforming 720+((1-720)%8)=721?  NO —
                   conforming_frames(720)=721, then _BAD_FRAME_BANDS[False][(1024,576)]
                   band (714, 735, 736) matches 721 → rendered 736
Section 2 frames:  requested 720 → conditioned=True (seam --image) → 720 ∈ _MEASURED_SAFE_CONDITIONED
                   → rendered 720, untouched
```

**Section 1 of 2:**

```
uv run python -m ltx_pipelines.distilled \
  --transformer-path MODELS/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
  --text-encoder-path MODELS/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path MODELS/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path MODELS/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --duration-head-path MODELS/model_patches/ltx-2.5-duration-head-bf16.safetensors \
  --spatial-upsampler-path MODELS/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --quantization nvfp4-prequant \
  --prompt 'LONG-FORM CONTINUATION — SECTION 1 OF 2.
Keep the same subjects, identities, faces, clothing, colours, object counts, vehicles, environment and camera direction established previously.
PERSISTENT USER CONSTRAINTS (verbatim):
A rain-soaked neon alley at night.
Two cars idle at the kerb, one matte black and one red.
Steam rises from a grate.
CONTINUITY (fixed for the entire video):
- Exactly 2 cars appear, and they remain the only cars on screen for the entire video.
- The matte black one stays matte black from the first frame to the last.
- The same subjects keep the same faces, clothing, colours and count in every frame.
NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY (verbatim):
- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.
Continue directly from the predecessor frame. Do not replay, restart or summarise any earlier action or dialogue. Complete this section'"'"'s assigned dialogue before the section ends.' \
  --num-frames 736 \
  --height 576 \
  --width 1024 \
  --frame-rate 24 \
  --seed 1148858713 \
  --output-path /workspace/job/segment-0000.mp4
```

After the render: `736 - 720 = 16` frames of overshoot → `_trim_to(output, 30.0, tolerance_seconds=16/24 = 0.6667)`, then `verify_output(expect_audio=True, expected_seconds=30.0)`, then `extract_final_frame(segment-0000.mp4, segment-condition-0001.png)`.

**Section 2 of 2:**

```
uv run python -m ltx_pipelines.distilled \
  --transformer-path MODELS/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
  --text-encoder-path MODELS/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path MODELS/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path MODELS/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --duration-head-path MODELS/model_patches/ltx-2.5-duration-head-bf16.safetensors \
  --spatial-upsampler-path MODELS/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --quantization nvfp4-prequant \
  --prompt 'LONG-FORM CONTINUATION — SECTION 2 OF 2.
Keep the same subjects, identities, faces, clothing, colours, object counts, vehicles, environment and camera direction established previously.
PERSISTENT USER CONSTRAINTS (verbatim):
A rain-soaked neon alley at night.
Two cars idle at the kerb, one matte black and one red.
Steam rises from a grate.
CONTINUITY (fixed for the entire video):
- Exactly 2 cars appear, and they remain the only cars on screen for the entire video.
- The matte black one stays matte black from the first frame to the last.
- The same subjects keep the same faces, clothing, colours and count in every frame.
NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY (verbatim):
Continue naturally from the preceding section without introducing a new event.
Continue directly from the predecessor frame. Do not replay, restart or summarise any earlier action or dialogue. Complete this section'"'"'s assigned dialogue before the section ends.' \
  --num-frames 720 \
  --height 576 \
  --width 1024 \
  --frame-rate 24 \
  --seed 863830479 \
  --output-path /workspace/job/segment-0001.mp4 \
  --image /workspace/job/segment-condition-0001.png 0 1.0
```

No trim on section 2 (`720 == 720`).

**Then:** each section is `normalize_clip(width=1024, height=576, fps=24, audio=True, frames=None)` → `concat_segments` → `verify_output(expect_video=True, expect_audio=True, expected_seconds=60.0)`.

Note `section_frames` is **not** used on this path — `_run_generation` calls `_assemble_generated_sections` without it ([ltx.py:1170-1172](apps/worker/worker/adapters/ltx.py#L1170-L1172)) because each section carries its own audio. `section_frames` is used only by `_deliver_restyle` and `_run_music_video`.

### 5.10 60-second I2V — section 2 (the two-image case)

```
Resolved plan:     per_pass=30.0  sections=2  section lengths=[30.0, 30.0]
Section 2 frames:  requested 720
_identity_anchor:  reference_frame = min(719, max(1, 720//3)) = 240
                   strength = 0.2
                   720 ∉ _TWO_IMAGE_SAFE_FRAMES {120, 240, 360}  →  RETURNS None
                   → log line "identity_anchor_skipped"
```

```
uv run python -m ltx_pipelines.distilled \
  --transformer-path MODELS/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
  --text-encoder-path MODELS/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path MODELS/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path MODELS/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --duration-head-path MODELS/model_patches/ltx-2.5-duration-head-bf16.safetensors \
  --spatial-upsampler-path MODELS/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --quantization nvfp4-prequant \
  --prompt '<identical to §5.9 section 2>' \
  --num-frames 720 \
  --height 576 \
  --width 1024 \
  --frame-rate 24 \
  --seed 863830479 \
  --output-path /workspace/job/segment-0001.mp4 \
  --image /workspace/job/segment-condition-0001.png 0 1.0
```

**Byte-identical to the 60 s T2V section 2.** At the shipped 30-second pass ceiling, a 60-second Image-to-Video job never carries the uploaded still past pass 1 — the pass count is 720 and the two-image gate admits only 120/240/360. Recorded as fact; the code comment at [ltx.py:2443-2448](apps/worker/worker/adapters/ltx.py#L2443-L2448) states the intent.

### 5.11 Complete flag inventory across all snapshots

Every distinct flag this repository can emit to an LTX pipeline, with the exact condition:

| Flag | Emitted when | Value source |
|---|---|---|
| `--transformer-path` | always | `_transformer_file()` or `pipeline.transformer_key` |
| `--text-encoder-path` | always | `_MODEL_FILES["text_encoder"]` |
| `--video-vae-path` | always | `_MODEL_FILES["video_vae"]` |
| `--audio-vae-path` | always | `_MODEL_FILES["audio_vae"]` |
| `--duration-head-path` | always | `_MODEL_FILES["duration_head"]` |
| `--spatial-upsampler-path` | always | `_MODEL_FILES["spatial_upsampler"]` |
| `--quantization` | `pipeline.quantize` | `settings.ltx_quantization` |
| `--offload cpu` | `pipeline.offload_cpu` | literal `"cpu"` |
| `--distilled-lora PATH 1.0` | `pipeline.distilled_lora` | strength is the literal `"1.0"` |
| `--prompt` | always | `prompt` arg or `job.prompt` |
| `--num-frames` | always | `num_frames` arg |
| `--height` / `--width` | always | `dimensions`, doubled if `stage_1_only` |
| `--frame-rate` | always | `settings.ltx_frame_rate` |
| `--seed` | always | `_seed_for_step` |
| `--output-path` | always | `step.output` |
| `--negative-prompt` | `pipeline.distilled_lora` **and** `execution.negative_prompt` non-empty after strip | `execution.negative_prompt` |
| `--video-cfg-guidance-scale` | `pipeline.distilled_lora` **and** `execution.guidance_scale is not None` | `round(float(v), 3)` |
| `--video-stg-guidance-scale` | `pipeline.distilled_lora` **and** `execution.stg_scale is not None` | `round(float(v), 3)` |
| `--a2v-guidance-scale` | `pipeline.distilled_lora` **and** `execution.a2v_guidance_scale is not None` | `round(float(v), 3)` |
| `--num-inference-steps` | `pipeline.distilled_lora` **and** `execution.inference_steps is not None` | `max(1, int(v))` |
| `--enhance-prompt` | `execution.enhance_prompt` truthy | — |
| `--prompt-enhancer-gemma-root` | same condition as above | `settings.director_gemma_root` |
| `--image PATH IDX STR` | once per `ConditioningFrame`, ascending index | see §3.3 |
| `--lora PATH STR` | once per `LoraSpec` — only `_IC_LORA` supplies one | `v2v_lora_strength` |
| `--video-conditioning PATH STR` | `control` callable supplied — only `_run_transform` | `v2v_control_strength` |
| `--conditioning-attention-mask PATH STR` | `mask` callable returns non-`None` — only `_run_transform` with identity or person-lock | always `1.0` |
| `--audio-path` / `--audio-start-time` / `--audio-max-duration` | `audio` callable supplied — only `_run_music_video` with `audio_conditioning` | see §3.3 |
| `--skip-stage-2` | `pipeline.stage_1_only` (only `_IC_LORA`) or an explicit `skip_stage_2=True` (no caller passes one) | — |

**No other flag is ever emitted.** `_command` is 149 lines and contains no other `cmd +=`.
---

## 6. Long-Form Segmentation

### 6.1 Where the ceiling is defined

There is **no single `max_segment_seconds`**. Five independent ceilings exist, one per tier, each with its own read site.

| Ceiling | Constant / key | Defined at | Value | Read at |
|---|---|---|---|---|
| Distilled, per grid | `_GRID_CEILINGS` | [ltx.py:227-237](apps/worker/worker/adapters/ltx.py#L227-L237) | 60.0 / 30.0 (see table) | [ltx.py:2309](apps/worker/worker/adapters/ltx.py#L2309) |
| Distilled, unmeasured grid | `_UNMEASURED_CEILING` | [ltx.py:242](apps/worker/worker/adapters/ltx.py#L242) | `10.0` | [ltx.py:2309](apps/worker/worker/adapters/ltx.py#L2309) |
| Workflow override | `execution.max_segment_seconds` | [text-to-video.yaml:81](workflow-definitions/text-to-video.yaml#L81), [image-to-video.yaml:70](workflow-definitions/image-to-video.yaml#L70) | `30` on t2v and i2v; **absent everywhere else** | [ltx.py:2310](apps/worker/worker/adapters/ltx.py#L2310) |
| Global brake | `settings.ltx_max_seconds` | [config.py:279](apps/worker/worker/core/config.py#L279) | `60` | [ltx.py:2310](apps/worker/worker/adapters/ltx.py#L2310), [ltx.py:2317](apps/worker/worker/adapters/ltx.py#L2317) |
| Transform tier | `_TRANSFORM_PASS_SECONDS` / `execution.transform_pass_seconds` | [ltx.py:524](apps/worker/worker/adapters/ltx.py#L524) | `8.0` | [ltx.py:1725-1730](apps/worker/worker/adapters/ltx.py#L1725-L1730) |
| Audio tier | `_AUDIO_PASS_SECONDS` / `execution.audio_pass_seconds` | [ltx.py:506](apps/worker/worker/adapters/ltx.py#L506) | `481 / 24.0` = `20.041666666666668` | [ltx.py:2402](apps/worker/worker/adapters/ltx.py#L2402) |
| Guided tier | `_GUIDED_PASS_SECONDS` / `execution.guided_pass_seconds` | [ltx.py:538](apps/worker/worker/adapters/ltx.py#L538) | `5.0` | [ltx.py:2414](apps/worker/worker/adapters/ltx.py#L2414) |
| Director extension | literal `30.0` | [ltx.py:1283](apps/worker/worker/adapters/ltx.py#L1283) | `30.0` | same line |
| Music (non-LTX) | `settings.acestep_max_seconds` | [config.py:368](apps/worker/worker/core/config.py#L368) | `600` | [music.py:529](apps/worker/worker/adapters/music.py#L529) |

`_GRID_CEILINGS` in full, verbatim ([ltx.py:227-237](apps/worker/worker/adapters/ltx.py#L227-L237)):

```python
_GRID_CEILINGS: dict[tuple[int, int], float] = {
    # current product grids — all proven to 60s, the longest length offered
    (1024, 576): 60.0,
    (576, 1024): 60.0,
    (768, 768): 60.0,
    (512, 640): 60.0,
    # previous grids, kept because a source's own aspect can still select them
    (896, 512): 30.0,  # 60s FAILS: CUBLAS_STATUS_INTERNAL_ERROR
    (512, 896): 60.0,
    (640, 640): 60.0,
}
```

### 6.2 Per-workflow effective ceiling

| Workflow | `execution.max_segment_seconds` | Grid source | Effective per-pass seconds (at a product grid) |
|---|---|---|---|
| text-to-video | `30` | `parameters.aspect_ratio` | **30.0** |
| image-to-video | `30` | `parameters.aspect_ratio` | **30.0** |
| extend-video | *absent* | `grid_for_source` | **60.0** (30.0 with a Director lineage) |
| video-to-video, transform engine | *absent* | `grid_for_source` | **8.0** |
| video-to-video, restyle engine | *absent* | `grid_for_source` | **60.0** |
| music-video, default tier | *absent* | `parameters.aspect_ratio` | **60.0** |
| music-video, audio tier | *absent* | `parameters.aspect_ratio` | **20.041666666666668** |
| t2v/i2v, guided tier | `30` (ignored — guided has its own path) | `parameters.aspect_ratio` | **5.0** |

### 6.3 The duration → sections → frames calculation, verbatim

**Step 1 — sections.** `plan_segments`, [segments.py:64-130](apps/worker/worker/media/segments.py#L64-L130):

```python
def plan_segments(
    total_seconds: float,
    *,
    max_segment_seconds: float,
    overlap_seconds: float = 0.0,
) -> list[Segment]:
    if total_seconds <= 0:
        raise ValueError("total_seconds must be positive")
    if max_segment_seconds <= 0:
        raise ValueError("max_segment_seconds must be positive")

    if total_seconds <= max_segment_seconds:
        return [Segment(index=0, start_seconds=0.0, duration_seconds=total_seconds)]

    overlap = max(0.0, min(overlap_seconds, max_segment_seconds / 2))

    count = math.ceil(total_seconds / max_segment_seconds - 1e-9)
    duration = total_seconds / count

    segments = [
        Segment(
            index=index,
            start_seconds=total_seconds * index / count,
            duration_seconds=(
                total_seconds - total_seconds * index / count
                if index == count - 1
                else duration
            ),
            overlap_seconds=0.0 if index == 0 else overlap,
        )
        for index in range(count)
    ]
    return segments
```

**Step 2 — the boundary variant.** `plan_chain_segments`, [chain.py:176-215](apps/worker/worker/longform/chain.py#L176-L215):

```python
def plan_chain_segments(
    total_seconds: float,
    per_pass_seconds: float,
    boundaries: list[float] | None = None,
) -> list[Segment]:
    if not boundaries:
        return plan_segments(total_seconds, max_segment_seconds=per_pass_seconds)

    cuts = [0.0, *sorted(boundaries), total_seconds]
    segments: list[Segment] = []
    for start, end in zip(cuts, cuts[1:], strict=False):
        duration = end - start
        if duration <= 1e-6:
            continue
        if duration > per_pass_seconds + 1e-6:
            raise ValueError(
                f"boundary window {duration:.2f}s exceeds the {per_pass_seconds:.2f}s "
                "pass ceiling"
            )
        segments.append(
            Segment(index=len(segments), start_seconds=start, duration_seconds=duration)
        )
    if not segments:
        return plan_segments(total_seconds, max_segment_seconds=per_pass_seconds)
    return segments
```

**Step 3 — frames.** `_frame_count` then the substitution, [ltx.py:2429-2430](apps/worker/worker/adapters/ltx.py#L2429-L2430) and [ltx.py:2523-2537](apps/worker/worker/adapters/ltx.py#L2523-L2537):

```python
def _frame_count(self, seconds: float) -> int:
    return max(1, round(seconds * settings.ltx_frame_rate))
```

```python
requested_frames = self._frame_count(step.seconds)
conditioned = bool(items) or control is not None or audio is not None
if pipeline.conforming_only:
    frames = conforming_frames(requested_frames)
    landing = next(
        (c for c in pipeline.measured_landings if c >= frames), None
    )
    if landing is not None:
        frames = landing
else:
    frames = safe_frame_count(
        dimensions, requested_frames, conditioned=conditioned
    )
```

### 6.4 Rounding / landing logic

`plan_segments` is deliberately **even, not greedy**. The reason is written in the source, [segments.py:96-110](apps/worker/worker/media/segments.py#L96-L110), quoted verbatim:

> EVEN windows, not greedy ones.
>
> Filling `max_segment_seconds` repeatedly and letting the remainder be its own segment produces a degenerate tail whenever the total is not a clean multiple — and it never is, because the workflows that chain longest take their length from an uploaded file. A four-minute MP3 probes at 240.03s, not 240.0, so a 60s ceiling gave `60, 60, 60, 60, 0.03`. That last window is one frame after `round(0.03 * 24)`: a full model invocation, a real cost, and a frozen flash concatenated onto the end of the customer's video. Observed at test scale 16 Aug 2026 (4.03s at a 1s ceiling → five passes, the last of them 0.03s).
>
> `ceil` then divide gives the same pass count, every window under the ceiling, no degenerate tail, and a progress bar that advances evenly because the passes actually take similar time.

Two further rounding rules:

- `start_seconds` is computed as `total_seconds * index / count`, **never accumulated**, "so the floating point error cannot walk" ([segments.py:118-119](apps/worker/worker/media/segments.py#L118-L119)).
- Delivery frame counts are allocated from the **cumulative** boundary, not per section — `_planned_section_frames`, [ltx.py:2087-2092](apps/worker/worker/adapters/ltx.py#L2087-L2092):

```python
counts: list[int] = []
for segment in segments:
    start = round(segment.start_seconds * fps)
    end = round((segment.start_seconds + segment.duration_seconds) * fps)
    counts.append(max(1, end - start))
return counts
```

### 6.5 Section overlap

**There is none.** `Segment.overlap_seconds` exists ([segments.py:40-47](apps/worker/worker/media/segments.py#L40-L47)) and `plan_segments` accepts an `overlap_seconds` parameter, but **no caller in the repository passes a non-zero value**. `render_chain` calls `plan_chain_segments(total_seconds, per_pass_seconds, boundaries)`, which calls `plan_segments(total_seconds, max_segment_seconds=per_pass_seconds)` — the default `overlap_seconds=0.0` applies. The docstring records the intent ([segments.py:44-47](apps/worker/worker/media/segments.py#L44-L47)):

> Continuity insurance: models drift at a cold start, so a segment that begins by re-generating the tail of its predecessor has something to match against. Set to zero when the provider supports a real continuation signal instead.

So: **overlap 0.0 s on every workflow, every tier.**

### 6.6 How the next section is conditioned

**Last-frame conditioning, always.** `render_chain`, [chain.py:168-172](apps/worker/worker/longform/chain.py#L168-L172):

```python
if chain_frames and segment.index + 1 < total:
    previous_frame = await _final_frame(
        job, step.output, segment.index + 1, prefix
    )
```

`_final_frame` runs `extract_final_frame`, which is `ffmpeg -sseof -1 -i <part> -update 1 <prefix>-condition-<n>.png` ([frames.py:43-46](apps/worker/worker/media/frames.py#L43-L46)). `chain_frames` defaults to `True` and **no caller passes `False`**.

The strength applied to that frame differs per workflow:

| Workflow / engine | Seam frame index | Seam strength | Extra conditioning on later passes |
|---|---|---|---|
| text-to-video | 0 | `1.0` | none |
| image-to-video | 0 | `1.0` | original still at `frames//3`, strength `0.2` — **only if frames ∈ {120, 240, 360}** |
| extend-video | 0 | `1.0` | `identity_image` at `frames//3`, strength `0.2`, same gate — only on Director-lineage extensions of I2V |
| music-video (both tiers) | 0 | `1.0` | audio window (audio tier only) |
| video-to-video restyle | 0 | `0.85` | 3–16 source stills at `0.45` |
| video-to-video transform | 0 | `0.85` | edge-map control clip at `1.0`, attention mask at `1.0`; reference re-anchor is off (`refresh_strength = 0.0`) |

### 6.7 What state carries across a section boundary

| Carried | Mechanism | Evidence |
|---|---|---|
| The last picture | `--image <PNG> 0 <strength>` | [chain.py:168-172](apps/worker/worker/longform/chain.py#L168-L172) |
| The section index and total | `ChainStep.total`, used by the prompt planner and the progress reporter | [chain.py:46-98](apps/worker/worker/longform/chain.py#L46-L98) |
| The window's absolute position | `Segment.start_seconds` — drives the v2v keyframe timestamps, the edge-map `-ss`, the matte `--start-seconds`, and the audio `--audio-start-time` | [ltx.py:1446](apps/worker/worker/adapters/ltx.py#L1446), [ltx.py:1648](apps/worker/worker/adapters/ltx.py#L1648), [ltx.py:1891](apps/worker/worker/adapters/ltx.py#L1891) |
| The global prompt plan | `prompt_plan` is computed **once**, on the first call to `prompt_for_step`, and cached in a `nonlocal` | [ltx.py:1112-1125](apps/worker/worker/adapters/ltx.py#L1112-L1125) |
| The Director plan | Built once before any section; `compile_section_prompts` buckets its events | [ltx.py:1098-1110](apps/worker/worker/adapters/ltx.py#L1098-L1110) |
| Character presence / exits | `departed` set walked across buckets inside the compiler | [compiler.py:87-94](apps/worker/worker/director/compiler.py#L87-L94) |
| The seed | Derived per index, not carried | [ltx.py:2886-2894](apps/worker/worker/adapters/ltx.py#L2886-L2894) |
| **Not carried** | latents, KV cache, model residency, RNG state, VAE state — a fresh subprocess per pass | [ltx.py:2924](apps/worker/worker/adapters/ltx.py#L2924) |

### 6.8 The segmentation table

Computed by executing the repository's own `_per_pass_seconds` / `_audio_pass_seconds` / `_guided_pass_seconds`, `plan_chain_segments`, `_frame_count`, `safe_frame_count` and `conforming_frames` — not derived by hand.

**text-to-video / image-to-video** (`execution.max_segment_seconds: 30`, grid `(1024, 576)`, per-pass `30.0`):

| Requested | Sections | Section Lengths (s) | Frames Per Section (requested → rendered) | Seams | Overlap | Conditioning Method |
|---|---|---|---|---|---|---|
| 5s | 1 | `[5.0]` | `[120 → 121]` | 0 | 0.0 | T2V: none. I2V: still @0 str 1.0 |
| 15s | 1 | `[15.0]` | `[360 → 361]` | 0 | 0.0 | same |
| 30s | 1 | `[30.0]` | `[720 → 736]` | 0 | 0.0 | same |
| 45s | 2 | `[22.5, 22.5]` | `[540 → 545, 540 → 720]` | 1 | 0.0 | pass 1 as above; pass 2 = predecessor frame @0 str 1.0 |
| 60s | 2 | `[30.0, 30.0]` | `[720 → 736, 720 → 720]` | 1 | 0.0 | same |
| 90s | 3 | `[30.0, 30.0, 30.0]` | `[720 → 736, 720 → 720, 720 → 720]` | 2 | 0.0 | same |

*(90 s is not an offered duration on either workflow — `supported_durations` tops out at `"60s"`. The row is computed for completeness.)*

**extend-video** (no `max_segment_seconds`, per-pass `60.0`, every pass conditioned by the seed frame):

| Requested | Sections | Section Lengths (s) | Frames Per Section (requested → rendered) | Seams | Overlap | Conditioning Method |
|---|---|---|---|---|---|---|
| 5s | 1 | `[5.0]` | `[120 → 120]` | 0 | 0.0 | source's final frame @0 str 1.0 |
| 15s | 1 | `[15.0]` | `[360 → 360]` | 0 | 0.0 | same |
| 30s | 1 | `[30.0]` | `[720 → 720]` | 0 | 0.0 | same |
| 45s | 1 | `[45.0]` | `[1080 → 1289]` | 0 | 0.0 | same |
| 60s | 1 | `[60.0]` | `[1440 → 1441]` | 0 | 0.0 | same |
| 90s | 2 | `[45.0, 45.0]` | `[1080 → 1289, 1080 → 1289]` | 1 | 0.0 | pass 2 = predecessor frame @0 str 1.0 |

*(A Director-lineage extension clamps per-pass to `30.0`, so its rows match the t2v table above.)*

**music-video, default tier** (per-pass `60.0`; pass 1 unconditioned, later passes conditioned):

| Requested (= track length) | Sections | Section Lengths (s) | Frames Per Section (requested → rendered) | Seams | Overlap | Conditioning Method |
|---|---|---|---|---|---|---|
| 5s | 1 | `[5.0]` | `[120 → 121]` | 0 | 0.0 | none |
| 15s | 1 | `[15.0]` | `[360 → 361]` | 0 | 0.0 | none |
| 30s | 1 | `[30.0]` | `[720 → 736]` | 0 | 0.0 | none |
| 45s | 1 | `[45.0]` | `[1080 → 1081]` | 0 | 0.0 | none |
| 60s | 1 | `[60.0]` | `[1440 → 1441]` | 0 | 0.0 | none |
| 90s | 2 | `[45.0, 45.0]` | `[1080 → 1081, 1080 → 1289]` | 1 | 0.0 | pass 2 = predecessor frame @0 str 1.0 |

*(Real windows shift when `plan_musical_boundaries` finds onsets — see §6.10.)*

**music-video, audio tier** (`audio_conditioning: true`, per-pass `20.041666666666668`; every pass conditioned because `audio is not None`):

| Requested | Sections | Section Lengths (s) | Frames Per Section (requested → rendered) | Seams | Overlap | Conditioning Method |
|---|---|---|---|---|---|---|
| 5s | 1 | `[5.0]` | `[120 → 121]` | 0 | 0.0 | audio window 0.000 s, 5.082 s |
| 15s | 1 | `[15.0]` | `[360 → 385]` | 0 | 0.0 | audio window 0.000 s, 16.082 s |
| 30s | 2 | `[15.0, 15.0]` | `[360 → 385, 360 → 385]` | 1 | 0.0 | + predecessor frame @0 str 1.0 |
| 45s | 3 | `[15.0, 15.0, 15.0]` | `[360 → 385] × 3` | 2 | 0.0 | same |
| 60s | 3 | `[20.0, 20.0, 20.0]` | `[480 → 481] × 3` | 2 | 0.0 | same |
| 90s | 5 | `[18.0] × 5` | `[432 → 481] × 5` | 4 | 0.0 | same |

**video-to-video, transform engine** (per-pass `8.0`, `_IC_LORA.measured_landings = (193,)`):

| Requested (= source length) | Sections | Section Lengths (s) | Frames Per Section (requested → rendered) | Seams | Overlap | Conditioning Method |
|---|---|---|---|---|---|---|
| 5s | 1 | `[5.0]` | `[120 → 193]` | 0 | 0.0 | anchor/reference @0 + control clip + mask |
| 15s | 2 | `[7.5, 7.5]` | `[180 → 193] × 2` | 1 | 0.0 | pass 2+: predecessor frame @0 str 0.85 + control + mask |
| 30s | 4 | `[7.5] × 4` | `[180 → 193] × 4` | 3 | 0.0 | same |
| 45s | 6 | `[7.5] × 6` | `[180 → 193] × 6` | 5 | 0.0 | same |
| 60s | 8 | `[7.5] × 8` | `[180 → 193] × 8` | 7 | 0.0 | same |
| 90s | 12 | `[7.5] × 12` | `[180 → 193] × 12` | 11 | 0.0 | same |

**t2v/i2v, guided tier** (`generation_engine: guided`, per-pass `5.0`, `_GUIDED.measured_landings = (121,)`):

| Requested | Sections | Section Lengths (s) | Frames Per Section (requested → rendered) | Seams | Overlap | Conditioning Method |
|---|---|---|---|---|---|---|
| 5s | 1 | `[5.0]` | `[120 → 121]` | 0 | 0.0 | as t2v/i2v |
| 15s | 3 | `[5.0] × 3` | `[120 → 121] × 3` | 2 | 0.0 | same |
| 30s | 6 | `[5.0] × 6` | `[120 → 121] × 6` | 5 | 0.0 | same |
| 45s | 9 | `[5.0] × 9` | `[120 → 121] × 9` | 8 | 0.0 | same |
| 60s | 12 | `[5.0] × 12` | `[120 → 121] × 12` | 11 | 0.0 | same |
| 90s | 18 | `[5.0] × 18` | `[120 → 121] × 18` | 17 | 0.0 | same |

### 6.9 Does 60 s produce 2 × 30 s with 1 seam?

**Stated explicitly, as computed:**

- **text-to-video at 60 s: YES — 2 sections of exactly 30.0 s, 1 seam, 0 overlap.** `per_pass = max(1.0, min(30, 60, 60)) = 30.0`; `count = ceil(60.0 / 30.0 - 1e-9) = 2`; `duration = 30.0`. Section 1 renders **736** frames (unconditioned → lattice 721 → band `(714, 735, 736)`) and is trimmed to 30.000 s. Section 2 renders **720** frames (conditioned → `720 ∈ _MEASURED_SAFE_CONDITIONED`, untouched).
- **image-to-video at 60 s: YES — same 2 × 30.0 s, 1 seam.** Additionally: the uploaded still is **not** carried into section 2, because 720 ∉ `_TWO_IMAGE_SAFE_FRAMES`.
- **extend-video at 60 s: NO — 1 section of 60.0 s, 0 seams** (per-pass is 60.0, not 30.0). Unless a Director lineage is present, in which case it is 2 × 30.0 s with 1 seam.
- **music-video with a 60 s track, default tier: NO — 1 section of 60.0 s, 0 seams.**
- **music-video with a 60 s track, audio tier: NO — 3 sections of 20.0 s, 2 seams.**
- **video-to-video with a 60 s source, transform engine: NO — 8 sections of 7.5 s, 7 seams.**
- **t2v/i2v at 60 s on the guided tier: NO — 12 sections of 5.0 s, 11 seams.**

### 6.10 Musical boundaries — worked example

`plan_musical_boundaries` ([timing.py:45-127](apps/worker/worker/longform/timing.py#L45-L127)) run against a 180.0 s track, `per_pass_seconds=60.0`, and synthetic onsets at every 3.7 s:

```
count    = ceil(180.0 / 60.0 - 1e-9) = 3
nominal  = 60.0
pull     = max(0.0, min(0.2, 0.5)) * 60.0 = 12.0
minimum  = min(2.0, 60.0/2) = 2.0

boundaries → [60.0, 120.0]
windows    → [(0.0, 60.0), (60.0, 60.0), (120.0, 60.0)]
frames req → [1440, 1440, 1440]
frames rnd → [1441, 1441, 1441]
```

The pull produced no movement here because the `earliest` guard `total_seconds - per_pass_seconds * (count - index)` pins `earliest = latest` when the track exactly fills its passes — recorded in the source at [timing.py:97-102](apps/worker/worker/longform/timing.py#L97-L102):

> …and never so EARLY that what is left cannot fit in the passes that remain. Without this the deficit from each backward pull has nowhere to go but the final window, which then quietly exceeds the ceiling — the exact oversized request the ceiling exists to prevent. A track that fills its passes exactly has no slack and gets no pull, which is the honest answer: moving a cut there would mean buying a pass.

The invariant `plan_chain_segments` then enforces: a boundary window longer than `per_pass_seconds + 1e-6` raises `ValueError` ([chain.py:205-209](apps/worker/worker/longform/chain.py#L205-L209)).

---

## 7. Frame & Resolution Rules

| Rule | Expression in code | File:line | Applies to | Comment in code (verbatim) |
|---|---|---|---|---|
| Frame count from seconds | `max(1, round(seconds * settings.ltx_frame_rate))` | [ltx.py:2430](apps/worker/worker/adapters/ltx.py#L2430) | all | *(none)* |
| **8k+1 lattice** | `max(1, frames + ((1 - frames) % 8))` | [ltx.py:389](apps/worker/worker/adapters/ltx.py#L389) | all | "Overshoot is at most 7 frames — under a third of a second — and the caller trims back to the exact requested duration, so this is invisible in the delivered video." |
| Lattice constant | `_FRAME_LATTICE = 8` | [ltx.py:377](apps/worker/worker/adapters/ltx.py#L377) | all | "The model's native frame convention: counts of the form 8k+1. Stated in three places in the pipeline source — `retake.py` REJECTS other counts outright (\"must satisfy 8k+1 (e.g. 97, 193)\"), the dubbing pipeline \"silently snaps to the nearest 8k+1\", and the trainer's dataset loader checks `num_frames % 8 != 1`. The entry point this adapter drives does none of that: it accepts whatever it is given and handles the remainder on a path where the decoder's batched GEMM casts its dimensions to int32 and dies." |
| Measured-safe passthrough | `if frames in _MEASURED_SAFE_CONDITIONED: return frames` | [ltx.py:419-420](apps/worker/worker/adapters/ltx.py#L419-L420) | conditioned passes, distilled | "This rule exists because its absence broke production: the lattice snap turned the matrix-proven 720 into 721, and 721-conditioned crashes. Evidence first, always." |
| Reach-down to a measured landing | `min((safe for safe in _MEASURED_SAFE_CONDITIONED if frames <= safe <= conforming_frames(frames)), default=None)` | [ltx.py:427-436](apps/worker/worker/adapters/ltx.py#L427-L436) | conditioned passes, distilled | "Without this, a source 24 milliseconds short of a whole frame — 14.976s, which is 359 frames — snaps PAST the proven 360 to 361, and 361 conditioned crashes the decoder. Three video-to-video jobs died on exactly that on 16 Aug 2026, all three from one upload; the 15.018s file next to it asked for 360, passed through, and worked." |
| Band landing | `for lo, hi, landing in bands: if lo <= frames <= hi: return landing` | [ltx.py:445-451](apps/worker/worker/adapters/ltx.py#L445-L451) | distilled | "The landing is a MEASUREMENT and is used exactly. Snapping it would replace evidence with theory — 1528 is measured-pass and not on the lattice, and \"1529 must be fine, it conforms\" is precisely the reasoning that produced 721." |
| Unlisted grid, conditioned | `bands = _CONDITIONED_BANDS if conditioned else ()` | [ltx.py:438-444](apps/worker/worker/adapters/ltx.py#L438-L444) | distilled | "An unlisted grid still gets the conditioned bands — those are keyed on the count, not the shape… The unconditioned bands are genuinely per-grid measurements, so an unlisted shape gets none of them and relies on the lattice." |
| Non-default tiers: lattice + landings only | `if pipeline.conforming_only: frames = conforming_frames(...); landing = next((c for c in pipeline.measured_landings if c >= frames), None)` | [ltx.py:2523-2533](apps/worker/worker/adapters/ltx.py#L2523-L2533) | ic_lora, a2vid, guided | "Applying them anyway broke production twice in one day, 17 Aug 2026, in opposite ways: the \"measured-safe\" 360 crashed ic_lora's decoder on a portrait source (the measurement was distilled-tier), and the (361,719,720) band would push an audio pass from its MEASURED 481 up to an unmeasured 720." |
| Two-image gate | `if frames not in _TWO_IMAGE_SAFE_FRAMES: return None` | [ltx.py:2455-2465](apps/worker/worker/adapters/ltx.py#L2455-L2465) | i2v, extend | "A pass carrying this is a TWO-image pass, and the decoder's two-image failure set is its own lottery: 720 frames decodes fine with one image and crashed a production job with two, 736 fails identically (so the render-extra-and-trim dodge is useless here), while 120/240/360 pass." |
| `conditioned` definition | `bool(items) or control is not None or audio is not None` | [ltx.py:2522](apps/worker/worker/adapters/ltx.py#L2522) | all | "This cost a production job. The transform engine deliberately drops the source stills the old restyle passed, so its first pass — the common case, with no reference image — carried no `--image` at all. `conditioned` flipped to False, the lattice snapped a 14.976s source's 359 frames up to 361 instead of landing on the measured-safe 360, and 361 is a count this decoder is already documented to die on." |
| Grid /64 divisibility | `_DIMENSIONS` literals; `range(256, 1025, 64)` in `grid_for_source` | [ltx.py:206-213](apps/worker/worker/adapters/ltx.py#L206-L213), [ltx.py:674-676](apps/worker/worker/adapters/ltx.py#L674-L676) | all | "LTX's two-stage pipeline requires dimensions divisible by 64 — 480x848 was rejected outright in benchmarking." |
| Grid pixel budget | `if w * h <= _PIXEL_BUDGET` where `_PIXEL_BUDGET = 1024 * 576` | [ltx.py:460](apps/worker/worker/adapters/ltx.py#L460), [ltx.py:677](apps/worker/worker/adapters/ltx.py#L677) | v2v, extend | "The largest frame measured on this card (1024x576 == 768x768 == 589,824 px)." |
| Aspect-error tolerance | `close = [grid for grid in grids if error(grid) <= 0.08]` then `max(close, key=lambda g: (g[0]*g[1], -error(g)))` | [ltx.py:687-689](apps/worker/worker/adapters/ltx.py#L687-L689) | v2v, extend | "Within a small aspect error the crop is invisible and more pixels win — otherwise 576x320 (1.80) would beat 1024x576 (1.78) for a 16:9 source on a 1% aspect technicality while shrinking the frame." |
| 4:5 exception | `"4:5": (512, 640)` | [ltx.py:212](apps/worker/worker/adapters/ltx.py#L212) | t2v | "4:5 is the odd one out. Exact 4:5 on a /64 lattice is only 512x640, then 768x960, then 1024x1280 — there is no intermediate, and 768x960 fails." |
| Stage-1-only grid doubling | `width, height = width * 2, height * 2; skip_stage_2 = True` | [ltx.py:2690-2695](apps/worker/worker/adapters/ltx.py#L2690-L2695) | ic_lora | "Stage 2 upscales in latent space by halving a latent grid, so it needs both latent dimensions to be even. The product's 16:9 grid is 1024x576, and 576/64 = 9 is odd: the two-stage IC-LoRA path dies in a VAE `rearrange` (\"can't divide axis of length 9 in chunks of 2\"), measured 17 Aug 2026. Padding the grid to a multiple of 128 would change the customer's aspect ratio, which is worse than the problem." |
| Delivery even dimensions | `max(2, int(value * scale) // 2 * 2)` | [ltx.py:704-705](apps/worker/worker/adapters/ltx.py#L704-L705) | v2v, extend | "Even dimensions because yuv420p subsamples chroma 2x2 — libx264 refuses odd sizes outright." |
| Delivery caps | `min(1.0, 1920 / long_side, 1080 / short_side)` | [ltx.py:702](apps/worker/worker/adapters/ltx.py#L702) | v2v, extend | "Delivery is at the source's own resolution (a user's 1080p clip must not come back as 512p), capped at full HD — beyond that the normalization re-encode cost stops being worth invisible extra pixels." |
| Delivery fps clamp | `min(60.0, max(10.0, source.fps or 24.0))` | [ltx.py:3117](apps/worker/worker/adapters/ltx.py#L3117) | v2v, extend | "Clamped because ffprobe reports nonsense for some variable-rate phone recordings, and normalizing to a nonsense rate produces a file that plays at the wrong speed." |
| Pass ceiling clamp | `max(1.0, min(requested, float(settings.ltx_max_seconds), measured))` | [ltx.py:2317](apps/worker/worker/adapters/ltx.py#L2317) | distilled | "Three clamps, all lowering: the workflow's own override, the operational brake, and what this grid was actually measured at." |
| Trim allowance | `overshoot = (frames - requested_frames) / 24.0`; refuses if `actual > seconds + max(0.5, overshoot * 3)` | [ltx.py:2612](apps/worker/worker/adapters/ltx.py#L2612), [ltx.py:2849-2864](apps/worker/worker/adapters/ltx.py#L2849-L2864) | all | "Bounded by what WE added, never more… Trimming an arbitrary amount would make this a length fixer — and a render that came back seconds too long is a fault the verification below exists to catch, not something to quietly cut down to size." |
| Control-clip frame pin | `-frames:v <frames>` plus `tpad=stop_mode=clone:stop_duration=<frames/fps>` | [control.py:86-116](apps/worker/worker/media/control.py#L86-L116) | v2v transform | "`-frames:v` pins the count rather than trusting `-t` to land on it. A window whose duration lands a rounding error short of a frame boundary otherwise yields one frame fewer than the pass will render, and the model is handed a control track that runs out before the shot does." |
| Normalize-clip frame pin | `tpad=stop_mode=clone:stop_duration=<2/fps>` + `-frames:v <n> -fps_mode cfr` | [frames.py:160-176](apps/worker/worker/media/frames.py#L160-L176) | v2v, music-video | "The pad that backs the pin is deliberately TWO FRAMES, not open-ended. Rounding is the only legitimate shortfall and it is under one frame; a clip materially shorter than its pin is a faulty render, and cloning its last picture up to length would hide exactly the wrong-length fault the output verification exists to catch." |
| Mux pad ceiling | `_PAD_TOLERANCE_SECONDS = 0.05`, `_MAX_PAD_SECONDS = 3.0` | [audio.py:51-57](apps/worker/worker/media/audio.py#L51-L57) | v2v, music-video | "A larger gap than this is a planning bug, not a rounding artefact, and padding it would hide the bug behind a frozen picture." |
| Min / max frames | Min: `max(1, …)` in `_frame_count` and `conforming_frames`. **Max: — not present** as an explicit bound; bounded indirectly by the pass ceiling × 24 | [ltx.py:2430](apps/worker/worker/adapters/ltx.py#L2430), [ltx.py:389](apps/worker/worker/adapters/ltx.py#L389) | all | — |
| Min / max resolution | Min: `256` (the `range` floor in `grid_for_source`). Max: `1024` per side, and `589824` px total | [ltx.py:674-677](apps/worker/worker/adapters/ltx.py#L674-L677) | v2v, extend | — |

### 7.1 The landing tables, in full

**`_CONDITIONED_BANDS`** — verbatim, [ltx.py:295-316](apps/worker/worker/adapters/ltx.py#L295-L316):

```python
_CONDITIONED_BANDS: list[tuple[int, int, int]] = [
    # 361..719 → 720. 361 is the second production-proven conditioned failure,
    # found the same way as 721: three video-to-video jobs on one 14.976s
    # upload, which is 359 frames — snapped to 361 and crashed, three times, on
    # 16 Aug 2026. The lattice points between 361 and 719 are UNMEASURED, and
    # 361 being proven bad is exactly why they cannot be assumed good.
    #
    # THIS COSTS TIME: a conditioned 20s pass (480 frames) renders as 720 and is
    # trimmed back, roughly 1.5x the compute it needs. The trade is deliberate —
    # a crash wastes the whole render AND the job, and the duration MENU values
    # (360 at 15s, 720 at 30s) land on measured-safe counts and never enter this
    # band, so only durations taken from an uploaded file land here. Narrow it by
    # probing 369..713 conditioned with `scripts/frame_probe.py`.
    (361, 719, 720),
    # 721..1288 → 1289: the only counts here in practice are the 30.0s menu edge
    # (which passes through as 720 before the snap) and music-video beat
    # windows, which run near the 60s ceiling.
    (721, 1288, 1289),
    (1290, 1384, 1385),
    (1386, 1440, 1441),
    (1442, 1527, 1528),
]
```

**`_BAD_FRAME_BANDS`** — verbatim, [ltx.py:318-331](apps/worker/worker/adapters/ltx.py#L318-L331):

```python
_BAD_FRAME_BANDS: dict[bool, dict[tuple[int, int], list[tuple[int, int, int]]]] = {
    # unconditioned: (band_lo, band_hi, safe_landing)
    False: {
        (1024, 576): [(233, 247, 248), (714, 735, 736)],
        (576, 1024): [(233, 247, 248), (714, 735, 736)],
        (768, 768): [(714, 735, 736)],  # 240 passes on 1:1 — measured
    },
    # conditioned (any --image): grid-independent, so every shape gets the same
    # list. Spelled out per grid rather than left empty because an empty table
    # here is precisely what broke production on 16 Aug.
    True: dict.fromkeys(
        ((1024, 576), (576, 1024), (768, 768), (512, 640)), _CONDITIONED_BANDS
    ),
}
```

**`_MEASURED_SAFE_CONDITIONED`** — [ltx.py:342](apps/worker/worker/adapters/ltx.py#L342):

```python
_MEASURED_SAFE_CONDITIONED = frozenset({120, 240, 360, 720, 1289, 1385, 1441, 1528})
```

**`_TWO_IMAGE_SAFE_FRAMES`** — [ltx.py:363](apps/worker/worker/adapters/ltx.py#L363):

```python
_TWO_IMAGE_SAFE_FRAMES = frozenset({120, 240, 360})
```

**Per-pipeline `measured_landings`:**

```
_DISTILLED.measured_landings = ()                        # uses the tables above instead
_IC_LORA.measured_landings   = (193,)                    # ltx.py:908
_A2VID.measured_landings     = (121, 241, 385, 481)      # ltx.py:964
_GUIDED.measured_landings    = (121,)                    # ltx.py:994
```

### 7.2 Quoted origin stories (why a rule exists)

These are the code comments that explain *why* a rule exists rather than what it does. They are the material that distinguishes a model requirement from a hard-won workaround, so they are reproduced verbatim.

**Why `_GRID_CEILINGS` is per-grid and not one number** — [ltx.py:216-226](apps/worker/worker/adapters/ltx.py#L216-L226):

> Single-pass ceilings **measured per grid**, in seconds. Not derived.
>
> The VAE fails on a *set of bad shapes*, not above a size threshold, and the set follows no rule anyone has been able to predict. Measured on this card at 60s: 1024x576 passes, 896x512 fails, 1152x640 fails, 768x960 fails. A larger grid passing where a smaller one fails rules out every "budget" model, so nothing here may be interpolated or extrapolated — a grid that is not in this table has not been run, and gets `_UNMEASURED_CEILING`.
>
> Before NATTEN this whole table was effectively 10s, because one global value had to satisfy the worst aspect ratio. That cost every 60s render five seams.

**Why the bad-frame tables exist at all** — [ltx.py:244-257](apps/worker/worker/adapters/ltx.py#L244-L257):

> Frame counts the VAE decoder cannot decode, and where to land instead.
>
> The decoder dies in a cuBLAS batched GEMM (`CUBLAS_STATUS_INTERNAL_ERROR` from `cublasGemmStridedBatchedEx`, all dims cast to int32) at specific (grid, conditioned, frame-count) triples. The failing set follows no rule anyone has produced: at 1024x576 unconditioned, 240 fails while 232, 248 and 1440 pass; WITH a conditioning image the same 1440 fails and 240 passes. Every entry below is a measurement from 16 Aug 2026 — nothing interpolated.

**Why the conditioned bands are grid-independent** — [ltx.py:271-294](apps/worker/worker/adapters/ltx.py#L271-L294):

> On 16 Aug the lattice theory looked complete: 1381/1437/1440 FAIL, 1289/1385/1441/1528 PASS, all failures non-conforming. The conditioned bands were emptied on that theory — and within two hours seven customer image-to-video jobs died, because the snap had turned the MATRIX-PROVEN 720 into 721, and 721-conditioned crashes. 720 passes and 721 fails, one frame apart, the mirror image of 1440/1441. There is no rule.
>
> measured PASS (conditioned): 81, 120, 121, 240, 360, 720 (matrix, i2v and production cells), 1289, 1385, 1441, 1528 (probed 16 Aug)
> measured FAIL (conditioned): 361 (production, 3 v2v jobs), 721 (production, 7 jobs), 1381, 1437, 1440, 1464

**Why the audio tier's ceiling is 481 frames and not longer** — [ltx.py:485-497](apps/worker/worker/adapters/ltx.py#L485-L497):

> **Swept 21 Aug 2026 at 1024x576, and longer is not reliably available.** Counts up to 1201 frames (50s) do decode — but 601 and 1081 both died with OUT OF MEMORY while 721, 961 and 1201 in the same sweep passed. That is not a shape property; it is a card running near its edge, with the music service holding ~24 GB of it permanently and a prompt-only pass already measured peaking at 95.2 GB of 95.6. A long pass here is a coin flip, and the coin is flipped after several minutes of compute have already been spent.

**Why the audio ceiling is a landing rather than a round number** — [ltx.py:499-504](apps/worker/worker/adapters/ltx.py#L499-L504):

> Expressed as a landing rather than a round number, deliberately. A ceiling of 20.0 asks for 480 frames, which conforms up to 481 anyway; stating 481's own duration instead means the plan's nominal window IS a measured count, and the pass count falls out one lower on real tracks. A 300.042s track goes from 16 passes to 15, and a 60.024s one from 4 to 3 — a whole model load saved on a one-minute video, for an arithmetic change.

**Why `_A2VID.measured_landings` is a screen and not a guarantee** — [ltx.py:954-963](apps/worker/worker/adapters/ltx.py#L954-L963):

> **This table is a screen, not a guarantee, and the difference matters.** 481 decoded in the sweep and in three consecutive benchmark cells, then failed in a fourth at the same count with the same cuBLAS error in the same video-VAE MLP. A fixed configuration that fails intermittently is not describing a shape; `CUBLAS_STATUS_INTERNAL_ERROR` is what a failed cuBLAS workspace allocation looks like, and this card was measured at 95.2 GB of 95.6 GB during an ordinary pass. So what the table buys is staying away from the counts that fail REPRODUCIBLY and from the large ones that need the most memory. The residual intermittency is a headroom problem, and headroom is not something a frame count can fix.

**Why the transform tier's ceiling is 8.0 s** — [ltx.py:512-523](apps/worker/worker/adapters/ltx.py#L512-L523):

> `ic_lora` decodes on its own path and the distilled tier's 60s grid ceilings carry no evidence about it — the first production transform job proved that at a cost: a single 15s portrait pass crashed the decoder at 360 AND 361 frames, two counts the distilled tables call safe or landable. What IS measured on this path, 17 Aug 2026, unquantized + cpu offload + Union Control: 97 frames (512x320), 193 frames landscape (1024x576, 62s), and 193 frames portrait (576x1024, 67s — probed the same evening the production job died). 8.0s keeps every pass at or under the 193-frame cell on either orientation.

**Why the guided tier's ceiling is 5.0 s** — [ltx.py:526-537](apps/worker/worker/adapters/ltx.py#L526-L537):

> `ti2vid_two_stages` decodes on its own path, and neither the distilled tier's grid ceilings nor the audio tier's 481-frame cell transfer to it — proven the day this tier was added, 17 Aug 2026: 241 frames at 1024x576, a count the audio tier renders happily at this grid, died in this pipeline's decoder with an illegal memory access, while 121 passed clean. What IS measured on this path (dev transformer + distilled LoRA, unquantized, `--offload cpu`): 121 frames at 1024x576 — 146s, ~29x real time, 39.3 GB peak alongside the resident music service.
---

## 8. Audio Paths

| Question | Answer | Evidence (file:line) |
|---|---|---|
| Does T2V receive any audio input? | **No.** `_run_generation` never constructs an `AudioConditioning` and never passes `audio=` to `_renderer`. `_require_generation_shape` additionally refuses any input role other than `source_image`, so an audio asset cannot reach it. | [ltx.py:1155-1159](apps/worker/worker/adapters/ltx.py#L1155-L1159) (no `audio=`); [ltx.py:2119-2129](apps/worker/worker/adapters/ltx.py#L2119-L2129) |
| | T2V **produces** audio: `require_audio=True` makes every section verify `expect_audio=True`, and the final assembly is `audio=True`. So the model's own generated soundtrack is what ships. | [ltx.py:1158](apps/worker/worker/adapters/ltx.py#L1158), [ltx.py:2616-2633](apps/worker/worker/adapters/ltx.py#L2616-L2633), [ltx.py:1170-1177](apps/worker/worker/adapters/ltx.py#L1170-L1177) |
| Which pipeline accepts supplied audio? | **`ltx_pipelines.a2vid_two_stage` only** (`_A2VID`). The `--audio-path` flag reaches `_command` only via the `audio` callable, which is supplied at exactly one call site. | [ltx.py:914](apps/worker/worker/adapters/ltx.py#L914), [ltx.py:1904](apps/worker/worker/adapters/ltx.py#L1904), [ltx.py:2796-2797](apps/worker/worker/adapters/ltx.py#L2796-L2797) |
| Does any path pass a real audio file to the model? | **Only music-video with `execution.audio_conditioning: true`, which no committed YAML sets.** When set, the path passed is the staged upload itself — the whole master file, not a slice: `AudioConditioning(path=staged, start_seconds=…, max_duration_seconds=…)`. | [ltx.py:1882-1893](apps/worker/worker/adapters/ltx.py#L1882-L1893); the key is commented out at [music-video.yaml:98](workflow-definitions/music-video.yaml#L98) |
| Is Dub-It / LipDub used? Which version/checkpoint? | **No.** `grep -rni 'dubit\|dub-it\|lipdub\|lip-dub\|lipsync\|lip_sync' apps/ workflow-definitions/` returns matches **only** in prose: [provider.py:60](apps/worker/worker/director/provider.py#L60) ("the set Lightricks documents as validated for speech (Dub-It)"), [ltx.py:853](apps/worker/worker/adapters/ltx.py#L853) ("`dubit` silently snaps to 8k+1"), [ltx.py:370](apps/worker/worker/adapters/ltx.py#L370) ("the dubbing pipeline"), and [music-video.yaml:124](workflow-definitions/music-video.yaml#L124). **No Dub-It module is invoked, no Dub-It checkpoint filename appears in `_MODEL_FILES` or `_OPTIONAL_MODEL_FILES`, and no `--dub`-style flag is emitted.** |
| What happens to model-generated audio? | **Three different fates, by workflow:** |
| — text-to-video, image-to-video | **Kept and delivered.** Sections normalise with `audio=True`, and the final `verify_output` demands `expect_audio=True`. `AudioMode.GENERATED_PER_SECTION_AUDIO`. | [ltx.py:1091](apps/worker/worker/adapters/ltx.py#L1091), [ltx.py:1170-1177](apps/worker/worker/adapters/ltx.py#L1170-L1177) |
| — extend-video | **Conditional.** `keep_audio = source.has_audio or any(info.has_audio for info in continuation_infos)`. If the source is silent but the model produced audio, the generated audio is kept; if both are silent, `AudioMode.NO_AUDIO`. | [ltx.py:1303-1314](apps/worker/worker/adapters/ltx.py#L1303-L1314) |
| — video-to-video (both engines) | **Discarded.** `_assemble_generated_sections(..., audio=False)`, which passes `-an` to ffmpeg. The comment at [ltx.py:1794-1797](apps/worker/worker/adapters/ltx.py#L1794-L1797): *"`audio=False` on purpose: the model generates its own soundtrack, and a restyle that replaced the user's audio with an invented one would be a bug nobody asked for."* | [ltx.py:1798-1805](apps/worker/worker/adapters/ltx.py#L1798-L1805), [frames.py:172](apps/worker/worker/media/frames.py#L172) |
| — music-video (both tiers) | **Discarded.** Same `audio=False`. | [ltx.py:1935-1941](apps/worker/worker/adapters/ltx.py#L1935-L1941) |
| Where is final audio muxed? | **`worker/media/audio.py::mux_audio`, called from exactly two places.** V2V: [ltx.py:1810](apps/worker/worker/adapters/ltx.py#L1810) `return await mux_audio(picture, staged, output)`. Music-video: [ltx.py:1944](apps/worker/worker/adapters/ltx.py#L1944) `return await mux_audio(picture, staged, output)`. In both cases the second argument is the **staged upload**, not a generated file. T2V/I2V/extend never call it — their audio rides inside the concatenated sections. | [audio.py:59-108](apps/worker/worker/media/audio.py#L59-L108) |
| Is source audio preserved in V2V? | **Yes, exactly once, whole, unmixed.** `mux_audio` maps `0:v:0` and `1:a:0` only, so any audio the picture carried is dropped rather than mixed. When `source.has_audio` is false, `picture.replace(output)` runs instead and the result is silent (`AudioMode.NO_AUDIO`). | [ltx.py:1784-1810](apps/worker/worker/adapters/ltx.py#L1784-L1810), [audio.py:88](apps/worker/worker/media/audio.py#L88) |

### 8.1 The exact mux command

`mux_audio` ([audio.py:85-108](apps/worker/worker/media/audio.py#L85-L108)) resolves to one of two ffmpeg invocations.

Common path (picture already covers the track within `0.05` s) — a **stream copy**:
```
ffmpeg -i <picture.mp4> -i <staged source> -map 0:v:0 -map 1:a:0 \
  -c:v copy -c:a aac -b:a 192k -ar 48000 -ac 2 \
  -shortest -movflags +faststart <output.mp4>
```

Pad path (picture is between `0.05` s and `3.0` s short) — a **re-encode**:
```
ffmpeg -i <picture.mp4> -i <staged source> -map 0:v:0 -map 1:a:0 \
  -filter:v 'tpad=stop_mode=clone:stop_duration=<shortfall+0.5>' \
  -c:v libx264 -preset veryfast -pix_fmt yuv420p \
  -c:a aac -b:a 192k -ar 48000 -ac 2 \
  -shortest -movflags +faststart <output.mp4>
```

A shortfall over `3.0` s raises `FfmpegError(f"{video.name} is {shortfall:.2f}s shorter than {audio.name}; that is a planning failure, not frame rounding")` ([audio.py:92-96](apps/worker/worker/media/audio.py#L92-L96)).

### 8.2 `AudioMode` — the four declared ownership models

[audio.py:37-43](apps/worker/worker/media/audio.py#L37-L43):

```python
class AudioMode(StrEnum):
    """The four supported ownership models for a finished soundtrack."""

    SOURCE_AUDIO = "SOURCE_AUDIO"
    GENERATED_MASTER_AUDIO = "GENERATED_MASTER_AUDIO"
    GENERATED_PER_SECTION_AUDIO = "GENERATED_PER_SECTION_AUDIO"
    NO_AUDIO = "NO_AUDIO"
```

Which the LTX adapter records, per workflow (log-only; `_record_audio_mode` emits a log line and nothing else — [ltx.py:2094-2098](apps/worker/worker/adapters/ltx.py#L2094-L2098)):

| Workflow | Mode | Set at |
|---|---|---|
| text-to-video, image-to-video | `GENERATED_PER_SECTION_AUDIO` | [ltx.py:1091](apps/worker/worker/adapters/ltx.py#L1091) |
| extend-video | `SOURCE_AUDIO` if the source has audio; else `GENERATED_PER_SECTION_AUDIO` if any continuation part does; else `NO_AUDIO` | [ltx.py:1305-1314](apps/worker/worker/adapters/ltx.py#L1305-L1314) |
| video-to-video | `SOURCE_AUDIO` if the source has audio, else `NO_AUDIO` | [ltx.py:1785-1787](apps/worker/worker/adapters/ltx.py#L1785-L1787) |
| music-video | `SOURCE_AUDIO`, unconditionally | [ltx.py:1856](apps/worker/worker/adapters/ltx.py#L1856) |
| *(any)* | `GENERATED_MASTER_AUDIO` — **never set by `adapters/ltx.py`** | — |

### 8.3 Onset detection (music-video cut points only)

`audio_onsets` ([audio.py:280](apps/worker/worker/media/audio.py#L280)) → `audio_envelope` ([audio.py:201](apps/worker/worker/media/audio.py#L201)) → `detect_onsets` ([audio.py:234](apps/worker/worker/media/audio.py#L234)). This is the **only** analysis performed on the uploaded track besides `probe_media`. Its output feeds `plan_musical_boundaries` and nothing else. The module states its own limits at [timing.py:14-18](apps/worker/worker/longform/timing.py#L14-L18), verbatim:

> **What this is not.** `detect_onsets` finds energy rises. It does not find beats, downbeats, bars, phrases, or where the chorus starts, and this module therefore aligns cuts to *events*, not to musical structure. The honest description of the result is "cuts tend to land on hits rather than mid-note". Anything stronger would be a claim the implementation does not support.

A failure here is absorbed, never fatal ([ltx.py:1975-1979](apps/worker/worker/adapters/ltx.py#L1975-L1979)):
```python
try:
    onsets = await cancellable(job, audio_onsets(track))
except FfmpegError as exc:
    logger.info("onset_analysis_skipped", extra={"detail": str(exc)})
    return []
```

---

## 9. IC-LoRA / Control

| Adapter | Used? | File referenced | Strength | Which workflow | Loaded how |
|---|---|---|---|---|---|
| **IC-LoRA Union Control** | **Yes** — the committed V2V configuration | `loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | `v2v_lora_strength`, default **`1.0`** | video-to-video, when `execution.v2v_engine: transform` | `--lora <abs path> 1.0`, built as `LoraSpec(settings.ltx_models_root / self._optional_weight("union_control_lora"), lora_strength)` at [ltx.py:1606-1609](apps/worker/worker/adapters/ltx.py#L1606-L1609); emitted at [ltx.py:2790-2791](apps/worker/worker/adapters/ltx.py#L2790-L2791) |
| **Distilled LoRA** | **Configured but not reachable** — both tiers that use it are commented out | `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | literal string **`"1.0"`** — not configurable | `_A2VID` (music-video + `audio_conditioning`), `_GUIDED` (t2v/i2v + `generation_engine: guided`) | `--distilled-lora <abs path> 1.0` at [ltx.py:2712-2717](apps/worker/worker/adapters/ltx.py#L2712-L2717). Note this is **a different flag** from `--lora` |
| Any other LoRA | **— not present.** `_OPTIONAL_MODEL_FILES` has exactly three entries and `loras=` is passed at exactly one call site. | — | — | — | `grep -n 'loras=' apps/worker/worker/adapters/ltx.py` → [ltx.py:1740](apps/worker/worker/adapters/ltx.py#L1740) only |

**Version note, recorded as fact:** the Union Control adapter's filename carries `ltx-2.3`, while every other checkpoint carries `ltx-2.5`. The source comment at [ltx.py:192-196](apps/worker/worker/adapters/ltx.py#L192-L196) states this deliberately:

> IC-LoRA Union Control. Trained for 2.3 and loads against the 2.5 distilled transformer — verified on the RTX PRO 6000 on 17 Aug 2026, a 97-frame render in 36s, which is what unblocked structure-conditioned video-to-video. It consumes canny / depth / pose control videos, NOT ordinary RGB (see `worker.media.control`).

### 9.1 The control signal it consumes

| Property | Value | Evidence |
|---|---|---|
| Kind | **Canny edge map**, produced by ffmpeg's `edgedetect` filter | [control.py:92](apps/worker/worker/media/control.py#L92) |
| Depth / pose | **— not present.** The docstring names them as things the adapter *can* consume ([control.py:4-6](apps/worker/worker/media/control.py#L4-L6)), but only `edgedetect` is implemented. | |
| Thresholds | `low=0.1`, `high=0.4` (ffmpeg's 0..1 scale) | [control.py:45-46](apps/worker/worker/media/control.py#L45-L46) |
| Filter chain, in order | `scale → crop → fps → edgedetect → tpad → format=yuv420p` | [control.py:88-95](apps/worker/worker/media/control.py#L88-L95) |
| Grid | `grid[0] × grid[1]` — the **un-doubled** grid, while the render asks for `2× ` | [ltx.py:1649-1650](apps/worker/worker/adapters/ltx.py#L1649-L1650) vs [ltx.py:2694](apps/worker/worker/adapters/ltx.py#L2694) |
| fps | `float(settings.ltx_frame_rate)` = `24` | [ltx.py:1651](apps/worker/worker/adapters/ltx.py#L1651) |
| Frame count | The pass's **rendered** count, not its requested one | [ltx.py:2565](apps/worker/worker/adapters/ltx.py#L2565) `control(step, frames)` |
| Encoding | `libx264 -preset veryfast -crf 18`, `-an`, `-fps_mode cfr` | [control.py:110-113](apps/worker/worker/media/control.py#L110-L113) |
| Flag | `--video-conditioning <path> <strength>` | [ltx.py:742-743](apps/worker/worker/adapters/ltx.py#L742-L743) |

The type exists specifically to make the RGB mistake visible at the call site — [ltx.py:729-737](apps/worker/worker/adapters/ltx.py#L729-L737), verbatim:

> The path is a CONTROL SIGNAL (see `worker.media.control`), not the source footage. Passing raw RGB here is accepted by the CLI and produces a weak style hint rather than structural control, which is the failure mode this type exists to make unmistakable at the call site.

### 9.2 The attention mask

| Property | Value | Evidence |
|---|---|---|
| Flag | `--conditioning-attention-mask <path> <strength>` | [ltx.py:758-763](apps/worker/worker/adapters/ltx.py#L758-L763) |
| Conditioning strength | always **`1.0`** — the per-region weight lives inside the clip, not in the flag | [ltx.py:1702](apps/worker/worker/adapters/ltx.py#L1702), [ltx.py:1719](apps/worker/worker/adapters/ltx.py#L1719) |
| Emitted when | `v2v_reference_identity` with a reference image, **or** `v2v_person_lock` with an existing matte. Otherwise the `mask` callable returns `None` and no flag is emitted. | [ltx.py:1664-1719](apps/worker/worker/adapters/ltx.py#L1664-L1719) |
| Identity mode values | `background=1.0`, `subject=v2v_identity_subject_attention` (default `0.5`) | [ltx.py:1698-1699](apps/worker/worker/adapters/ltx.py#L1698-L1699) |
| Person-lock mode values | `background=v2v_background_attention` (default `0.5`), subject implicitly white | [ltx.py:1713-1716](apps/worker/worker/adapters/ltx.py#L1713-L1716) |
| Semantics | "white follows it fully, black ignores it" | [ltx.py:750-751](apps/worker/worker/adapters/ltx.py#L750-L751), [masks.py:60-62](apps/worker/worker/media/masks.py#L60-L62) |
| Failure posture | Under identity replacement a matting failure **fails the pass** — it is not absorbed | [ltx.py:1674-1676](apps/worker/worker/adapters/ltx.py#L1674-L1676) |

### 9.3 Referenced in config but not actually loaded

| Item | Status |
|---|---|
| `v2v_person_lock` and its `v2v_background_attention` | Documented across [video-to-video.yaml:133-164](workflow-definitions/video-to-video.yaml#L133-L164), fully implemented in `_person_locked_control` ([ltx.py:2319-2386](apps/worker/worker/adapters/ltx.py#L2319-L2386)) and `build_hybrid_control`, but the key is **commented out** so the branch never runs |
| BiRefNet | Named in the YAML comments ([video-to-video.yaml:146](workflow-definitions/video-to-video.yaml#L146), [:185](workflow-definitions/video-to-video.yaml#L185)) as the matting model, and as "MIT-licensed, run in the GPU environment". **No BiRefNet filename, checkpoint path, or import appears anywhere in the repository** — it lives behind `scripts/person_matte.py`, whose contents are the CLI contract only |
| `transformer_bf16` via `_transformer_file()` | Unreachable while `ltx_quantization` contains `"nvfp4"`; still reachable via `_IC_LORA.transformer_key` |
| Depth and pose control videos | Named at [control.py:5-6](apps/worker/worker/media/control.py#L5-L6) and [ltx.py:194-195](apps/worker/worker/adapters/ltx.py#L194-L195); only canny is implemented |

---

## 10. Camera System

Search performed across `apps/worker/worker/`, `apps/api/app/`, `apps/web/src/`, `packages/`, and `workflow-definitions/` for the full term list in the spec. **69 total matches**, concentrated in six files. `apps/web/src/` returned **zero** genuine matches (the only hits were the HTML token `<span>` matching the substring "pan").

### 10.1 Current capability table

| Camera Concept | Present? | Representation (type/enum) | Defined In | Reaches Final Prompt? | Example Value |
|---|---|---|---|---|---|
| **SHOT SIZE** |
| wide shot | Yes | free-text inside `DirectorEvent.camera: str` | [plan.py:156](apps/worker/worker/director/plan.py#L156) (planner brief), [plan.py:156](apps/worker/worker/director/plan.py#L156) field decl at [plan.py:156](apps/worker/worker/director/plan.py#L156) | **Yes** → `"A wide shot frames the moment, and the camera remains static."` | `"wide shot, static"` |
| medium shot | Yes | same | [provider.py:156](apps/worker/worker/director/provider.py#L156) | Yes | `"medium shot, subtle push-in"` |
| medium close-up | Yes | same | [provider.py:156](apps/worker/worker/director/provider.py#L156) | Yes | `"medium close-up, static"` |
| close-up | Yes | same | [provider.py:156](apps/worker/worker/director/provider.py#L156) | Yes | `"close-up, static"` |
| two-shot | Yes | same | [provider.py:156](apps/worker/worker/director/provider.py#L156) | Yes | `"two-shot, static"` |
| over-the-shoulder shot | Yes | same | [provider.py:156](apps/worker/worker/director/provider.py#L156) | Yes | `"over-the-shoulder shot, static"` |
| extreme close-up, long shot, establishing shot, insert | **No** — not in the offered vocabulary. Would still pass through as free text if a planner emitted one. | — | — | — | — |
| **CAMERA ANGLE** |
| low angle, high angle, eye level, top-down, overhead, dutch, bird's-eye, worm's-eye | **— not present.** No field, no enum member, no vocabulary entry, no compiler branch. The strings do not appear in any source file. | — | — | — | — |
| **CAMERA MOVEMENT** |
| static | Yes | free text; the compiler special-cases it | [compiler.py:519-520](apps/worker/worker/director/compiler.py#L519-L520) | Yes → `"…and the camera remains static."` | `"static"` |
| push-in | Yes | free text, recognised as a **noun phrase** and prefixed with "makes a" | [compiler.py:547-549](apps/worker/worker/director/compiler.py#L547-L549) | Yes → `"…as the camera makes a subtle push-in."` | `"subtle push-in"` |
| pan | Yes | free text, recognised as a **verb** | `_MOVE_VERBS` [compiler.py:529-535](apps/worker/worker/director/compiler.py#L529-L535) | Yes → `"…as the camera slowly pans."` | `"slowly pans"` |
| tilt | Yes | verb in `_MOVE_VERBS` | [compiler.py:532](apps/worker/worker/director/compiler.py#L532) | Yes | `"tilts up"` |
| track / tracking | Yes | verb in `_MOVE_VERBS` | [compiler.py:532](apps/worker/worker/director/compiler.py#L532) | Yes | `"tracking shot, follows them"` |
| dolly | Yes | verb `dollies` in `_MOVE_VERBS` | [compiler.py:532](apps/worker/worker/director/compiler.py#L532) | Yes | `"dollies in"` |
| zoom | Yes | verb in `_MOVE_VERBS` | [compiler.py:532](apps/worker/worker/director/compiler.py#L532) | Yes | `"zooms out"` |
| push / pull | Yes | verbs `pushes` / `pulls` | [compiler.py:531](apps/worker/worker/director/compiler.py#L531) | Yes | `"pushes in"` |
| drift, glide, circle, rise, descend, follow, hold, remain, stay, settle, reframe | Yes | verbs in `_MOVE_VERBS` | [compiler.py:531-533](apps/worker/worker/director/compiler.py#L531-L533) | Yes | `"circles the table"` |
| truck, crane, pedestal, orbit, arc, handheld, steadicam, gimbal, whip pan | **— not present** in `_MOVE_VERBS`. A planner emitting one would be treated as a noun phrase → `"as the camera makes a crane"`. | — | — | — | — |
| **COMPOSITION** |
| Any composition field | **— not present** in the DirectorPlan. The word "COMPOSITION" appears only in the *vision* describer's brief, whose output goes to the **planner**, never to the model. | [vision.py:50](apps/worker/worker/director/vision.py#L50) | No | `"COMPOSITION: where the subjects are in the frame and how they relate"` |
| Framing implied by the source | Yes, on the anchored path — as a prose instruction, not a field | [provider.py:242-245](apps/worker/worker/director/provider.py#L242-L245) | Yes (in the brief, so indirectly) | *"the video opens on the photograph's own framing"* |
| **LENS / LOOK** |
| focal length, lens, aperture, depth of field, bokeh, film stock, grain | **— not present.** No field, no enum, no vocabulary, no compiler branch. | — | — | — | — |
| **TRANSITION** |
| cut | **Only as a movement verb** (`cuts?` in `_MOVE_VERBS`, [compiler.py:533](apps/worker/worker/director/compiler.py#L533)) and in the *planner brief's* prose ("cuts between the people already in frame", [provider.py:243](apps/worker/worker/director/provider.py#L243)). There is **no transition field**. | — | as prose | `"then cuts to a medium shot"` |
| match cut, dissolve, fade, wipe, whip pan, smash cut | **— not present** anywhere in the repository. | — | — | — | — |
| Section seam | Not a "transition" in the camera sense; it is a hard concatenation of two independently rendered clips, joined by `concat_segments`. No crossfade, no dissolve. | [segments.py:133-169](apps/worker/worker/media/segments.py#L133-L169) | — | — | — |
| **NEGATIVE CONSTRAINTS** |
| "No fast camera moves" | Yes — a **prose rule in the planner brief**, not a machine constraint | [provider.py:193-194](apps/worker/worker/director/provider.py#L193-L194) | reaches the *planner*, not the model | *"prefer medium shots, close-ups, two-shots and reaction shots, with a static camera or a subtle push-in. No fast camera moves."* |

**Camera fields documented: 3.** They are `DirectorEvent.camera` (the only structured camera field in the entire repository), plus two derived prose channels the compiler produces from it (`_camera_sentence`'s "shot" half and its "move" half). There is no camera field on any request schema, workflow YAML, preset, or model argument.

### 10.2 Data model

**Camera is stored as a single free-text string.** Not an enum, not structured data, not separate fields.

The complete type definition, verbatim ([plan.py:151-177](apps/worker/worker/director/plan.py#L151-L177)):

```python
@dataclass(frozen=True)
class DirectorEvent:
    start: float
    end: float
    action: str
    camera: str = ""
    speaker: str | None = None
    dialogue: str | None = None
    delivery: str | None = None
    """Audible manner of the line ("low and accusing") — becomes ".. says in a
    low and accusing voice" in the compiled caption."""

    exits: tuple[str, ...] = ()
    """Character ids who leave the scene at this event and STAY gone.
    ...
    """
```

Parsing, verbatim ([plan.py:419](apps/worker/worker/director/plan.py#L419)):

```python
camera = str(entry.get("camera") or "").strip()
```

**There is no validation of the camera string at all.** No membership check, no enum coercion, no default, no rejection. `_parse_timeline` validates `start`, `end`, `speaker`, `dialogue`, `action` and `exits`; `camera` is only `str()`-coerced and stripped. An event with `"camera": "a purple flugelhorn"` parses successfully and reaches the prompt.

**Are shot size, angle, and movement stored in separate fields or one combined field?**

> **Plainly: one combined field.** A single `camera: str`. The split into "shot" and "move" happens only at *compile* time and is purely lexical — `re.split(r"[,;]", camera)`, first fragment is the shot, the rest is the move ([compiler.py:513-515](apps/worker/worker/director/compiler.py#L513-L515)). There is no angle concept at any layer.

**Every allowed value for every camera field, verbatim.** There is no allow-list in code. The only enumeration anywhere is the *suggestion* inside the planner's JSON-shape brief ([provider.py:156-157](apps/worker/worker/director/provider.py#L156-L157)):

```
"camera": "<one of: medium shot | medium close-up | close-up | two-shot |
  over-the-shoulder shot | wide shot; plus 'static' or a subtle move>",
```

Six shot sizes, `static`, and an unbounded "a subtle move". This is a prompt string sent to a language model — nothing enforces it.

The one machine-readable camera vocabulary is `_MOVE_VERBS`, and it decides only *grammar*, not legality ([compiler.py:529-535](apps/worker/worker/director/compiler.py#L529-L535)):

```python
_MOVE_VERBS = re.compile(
    r"^(?:(?:and\s+|then\s+|slowly\s+|briefly\s+|before\s+)*)"
    r"(?:pans?|pushes|pulls?|tilts?|tracks?|dollies|zooms?|moves?|drifts?|"
    r"glides?|circles?|rises?|descends?|follows?|holds?|remains?|stays?|"
    r"cuts?|switches|shifts?|settles?|reframes?)\b",
    re.IGNORECASE,
)
```

Verbs, verbatim: `pan(s)`, `pushes`, `pull(s)`, `tilt(s)`, `track(s)`, `dollies`, `zoom(s)`, `move(s)`, `drift(s)`, `glide(s)`, `circle(s)`, `rise(s)`, `descend(s)`, `follow(s)`, `hold(s)`, `remain(s)`, `stay(s)`, `cut(s)`, `switches`, `shift(s)`, `settle(s)`, `reframe(s)`. Optional leading adverbs: `and`, `then`, `slowly`, `briefly`, `before`.

### 10.3 Camera flow

**Where camera is decided:** in the Director planner — a language model — and nowhere else.

| Source | Can set camera? | Evidence |
|---|---|---|
| User input (`GenerationParameters`) | **No.** `extra="forbid"`; there is no camera field. | [generation.py:27-56](apps/api/app/schemas/generation.py#L27-L56) |
| Web UI | **No.** Zero camera concepts in `apps/web/src/`. | grep, §10 preamble |
| Workflow YAML | **No.** No `execution` key is read for camera. | grep `camera` in `workflow-definitions/` → only [video-to-video.yaml:159-160](workflow-definitions/video-to-video.yaml#L159-L160), a prose comment about losing the camera at low background attention |
| Preset | **No.** No presets exist. | §3.1 |
| Default | **`""`** — the dataclass default | [plan.py:156](apps/worker/worker/director/plan.py#L156) |
| Director planner | **Yes — the only source.** | [provider.py:156-157](apps/worker/worker/director/provider.py#L156-L157) |
| The user's prompt text (standard mode) | **Indirectly and verbatim.** `plan_section_prompts` copies user-authored fragments through unchanged, so "wide shot" typed by a user reaches the model as their own words. It is never parsed as camera data. | [prompts.py:8-13](apps/worker/worker/longform/prompts.py#L8-L13) |

**Precedence order in code:** there is none to trace, because only one source exists. The resolution logic is one line ([compiler.py:197](apps/worker/worker/director/compiler.py#L197)):

```python
camera = _humanise(event.camera.strip(), plan)
```

**Does an explicit user camera request override Director?**

**No — and the reverse also does not happen.** In Director mode, `structure_prompt` is skipped and `plan_section_prompts` is not called ([ltx.py:1061](apps/worker/worker/adapters/ltx.py#L1061), [ltx.py:1118-1121](apps/worker/worker/adapters/ltx.py#L1118-L1121)); the caption is compiled **entirely** from the plan. The user's literal text reaches the planner as `IDEA:` ([provider.py:278](apps/worker/worker/director/provider.py#L278)) and reaches the model only insofar as the planner chose to carry it. There is no code path by which a user-typed camera instruction is preserved verbatim into a Director-mode prompt. In standard mode the opposite holds: the user's text is verbatim and there is no Director to override.

**Is camera state carried across section boundaries?**

**No.** Two independent mechanisms both reset it:

1. `previous_camera` is a local initialised to `""` inside `_compile_section`, once per section ([compiler.py:194](apps/worker/worker/director/compiler.py#L194)). It is compared only within that section:
```python
previous_camera = ""
...
if camera and camera.lower() != previous_camera.lower():
    sentences.append(_camera_sentence(camera))
    previous_camera = camera
```
So the first event of every section always emits a camera sentence, even if it repeats the last camera of the previous section.

2. `plan_section_prompts` (standard mode) puts the string *"Keep the same … camera direction established previously."* in every section header ([prompts.py:112](apps/worker/worker/longform/prompts.py#L112)) — a prose instruction, not carried state.

**Is there a global camera plan, or per-section decisions?**

**Global by construction, per-event in content.** The `DirectorPlan.timeline` is produced **once**, before any section renders, and every event carries its own `camera` string. Sections do not re-plan — they bucket the pre-existing events by midpoint ([compiler.py:74-79](apps/worker/worker/director/compiler.py#L74-L79)). So there is one global plan; there is **no global camera plan** as a distinct artifact — no field describing the shot progression, no continuity rule about camera, no "opening shot / closing shot" concept.

### 10.4 Camera compiler

The complete transformation, verbatim ([compiler.py:511-550](apps/worker/worker/director/compiler.py#L511-L550)):

```python
def _camera_sentence(camera: str) -> str:
    """ "A medium close-up frames the moment, and the camera remains static." """
    parts = [part.strip() for part in re.split(r"[,;]", camera) if part.strip()]
    shot = parts[0] if parts else camera.strip()
    move = ", ".join(parts[1:])
    shot = _ARTICLE.sub("", shot)
    shot = shot[0].lower() + shot[1:] if shot else shot
    article = "An" if shot[:1].lower() in "aeiou" else "A"
    if not move or "static" in move.lower():
        return f"{article} {shot} frames the moment, and the camera remains static."
    move = _normalise_move(move)
    return f"{article} {shot} frames the moment as the camera {move}."


_MOVE_VERBS = re.compile(
    r"^(?:(?:and\s+|then\s+|slowly\s+|briefly\s+|before\s+)*)"
    r"(?:pans?|pushes|pulls?|tilts?|tracks?|dollies|zooms?|moves?|drifts?|"
    r"glides?|circles?|rises?|descends?|follows?|holds?|remains?|stays?|"
    r"cuts?|switches|shifts?|settles?|reframes?)\b",
    re.IGNORECASE,
)


def _normalise_move(move: str) -> str:
    text = move.strip().rstrip(".")
    text = re.sub(r"^(?:the\s+)?camera\s+", "", text, flags=re.IGNORECASE)
    if not text:
        return "remains steady"
    text = text[0].lower() + text[1:]
    # Planners write moves as noun phrases ("subtle push-in") as often as verb
    # phrases ("pushes in slowly"); a noun phrase needs a verb to sit after
    # "as the camera …".
    if not _MOVE_VERBS.match(text):
        article = "an" if text[:1] in "aeiou" else "a"
        return f"makes {article} {text}"
    return text
```

Two supporting regexes ([compiler.py:51-58](apps/worker/worker/director/compiler.py#L51-L58)):
```python
_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
```

**Three real input → output examples**, produced by running the compiler on the reconstructed plan of §12:

| Input `event.camera` | Output sentence |
|---|---|
| `"wide shot, static"` | `A wide shot frames the moment, and the camera remains static.` |
| `"medium shot, subtle push-in"` | `A medium shot frames the moment as the camera makes a subtle push-in.` |
| `"tracking shot, follows them"` | `A tracking shot frames the moment as the camera follows them.` |

A fourth, from the same run, showing the `slowly pans` verb path:

| Input | Output |
|---|---|
| `"wide shot, slowly pans"` | `A wide shot frames the moment as the camera slowly pans.` |

And the `over-the-shoulder shot` vowel branch (constructed to exercise `article = "An"`):

| Input | Output |
|---|---|
| `"over-the-shoulder shot, static"` | `An over-the-shoulder shot frames the moment, and the camera remains static.` |

**Ordering:** the camera sentence is emitted *before* the event sentence it describes, and only when the camera string differs (case-insensitively) from the previous event's within the same section ([compiler.py:197-200](apps/worker/worker/director/compiler.py#L197-L200)).

**The camera string is also run through `_humanise`** ([compiler.py:197](apps/worker/worker/director/compiler.py#L197)) before compilation, which replaces any character id leaking into it with that character's role word — e.g. `"close-up on boss_marcos"` → `"close-up on the police chief"` ([compiler.py:561-626](apps/worker/worker/director/compiler.py#L561-L626)).

---

## 11. Director

### 11.1 The complete Director output schema, verbatim

Three frozen dataclasses in [apps/worker/worker/director/plan.py](apps/worker/worker/director/plan.py). Reproduced in full, docstrings included.

```python
@dataclass(frozen=True)
class DirectorCharacter:
    id: str
    role: str
    """Short noun phrase the prose refers to them by: "detective", "police chief"."""
    appearance: str
    """Concrete visible description, held constant across every section."""
    voice: str = ""
    """Audible description ("low, gravelly"), woven in when they first speak."""


@dataclass(frozen=True)
class DirectorEvent:
    start: float
    end: float
    action: str
    camera: str = ""
    speaker: str | None = None
    dialogue: str | None = None
    delivery: str | None = None
    """Audible manner of the line ("low and accusing") — becomes ".. says in a
    low and accusing voice" in the compiled caption."""

    exits: tuple[str, ...] = ()
    """Character ids who leave the scene at this event and STAY gone.

    This is the semantic state the 60-second measurement proved a caption
    cannot live without (GPU, 20 Aug 2026): a plan had the man walk out at
    ~38s, and because nothing recorded that he was gone, the standing
    constancy sentence — "present and solid in every single frame" — summoned
    him straight back. He flickered at 43-48s and stood fully returned for the
    final twelve seconds while the soundtrack said "He is finally gone."

    An exit is the one irreversible state a plan can express, and the compiler
    treats it as one: after this event the character is out of every cast
    sentence, out of every constancy sentence, and the scene is restated in
    terms of who REMAINS — never in terms of who left, because on this
    runtime naming the departed is inviting them back."""


@dataclass(frozen=True)
class DirectorPlan:
    scene: str
    tone: str
    language: str
    duration_seconds: float
    ambience: str
    characters: tuple[DirectorCharacter, ...]
    timeline: tuple[DirectorEvent, ...]
    source_anchored: bool = False
    """True when the video starts from an uploaded image (Image to Video).
    ..."""

    continuity: tuple[str, ...] = ()
    """Facts that must look the same in every frame — a red hat that stays the
    same red hat after it is taken off and put back on, a jacket that does not
    change colour when someone turns around.
    ..."""
```

Plus five derived members ([plan.py:215-263](apps/worker/worker/director/plan.py#L215-L263)): `character(id)`, `spoken_words`, `spoken_lines`, `seconds_per_spoken_line`, `exit_time(id)`, `present_ids(at_seconds)`, `has_exits`.

### 11.2 Every field, its type, and whether it reaches the model

*(Full table given in §2.11 under "Director". Summarised counts: 17 leaf fields; **11 reach the model**, 4 are internal-only, 2 reach it indirectly by changing sentence form.)*

The fields that do **not** reach the model: `tone`, `duration_seconds`, `timeline[].start`, `timeline[].end`, `characters[].id`. The last is actively stripped ([compiler.py:561-626](apps/worker/worker/director/compiler.py#L561-L626)).

### 11.3 What Director receives as input

Built at [provider.py:537-548](apps/worker/worker/director/provider.py#L537-L548):

```python
request = DirectorRequest(
    idea=job.prompt,
    duration_seconds=duration_seconds,
    language=language,
    seed=seed + attempt,
    sample=sample,
    source_anchored=anchored,
    prior_idea=prior_idea,
    prior_seconds=prior_seconds,
    image_facts=facts,
    notes=notes,
)
```

- `idea` — `job.prompt`, verbatim, **before** any structuring (Director mode skips `structure_prompt`).
- `duration_seconds` — the requested length (t2v/i2v) or the extension length (extend).
- `language` — `parameters.dialogue_language` if in `DIALOGUE_LANGUAGES`, else `"auto"`; on an extension, the **ancestor's** language from the lineage.
- `seed` — `zlib.crc32(f"{job.job_id}:director".encode()) + attempt` ([provider.py:507](apps/worker/worker/director/provider.py#L507)).
- `sample` — `False` on the first attempt, `True` on the retry.
- `source_anchored` — `job.workflow_id == "image-to-video"`, or `True` on any lineage extension.
- `prior_idea` / `prior_seconds` — from `parameters["director_lineage"]`.
- `image_facts` — `""` unless `settings.director_vision_enabled` (default `False`).
- `notes` — the previous attempt's problems, verbatim.

### 11.4 Which model Director uses, and its complete prompt template

**Provider chain** ([provider.py:454-464](apps/worker/worker/director/provider.py#L454-L464)):

```python
def default_providers() -> list[DirectorProvider]:
    hosted = CerebrasDirectorProvider()
    return [hosted, GemmaDirectorProvider()] if hosted.available else [GemmaDirectorProvider()]
```

| Provider | Model | Config | Timeout |
|---|---|---|---|
| `CerebrasDirectorProvider` | `gemma-4-31b` (`cerebras_director_model`) | `cerebras_director_enabled=True`, `temperature=0.7`, base `https://api.cerebras.ai` | `60.0` s |
| `GemmaDirectorProvider` | local `gemma-4-e2b-it` checkpoint, run as `uv run python <repo>/apps/worker/scripts/director_plan.py` with `cwd=/workspace/ltx2-benchmark` | `max_new_tokens: 1600`, markers `===DIRECTOR_PLAN_BEGIN===` / `===DIRECTOR_PLAN_END===` | `900.0` s |

Two attempts per provider, greedy then sampled ([provider.py:534](apps/worker/worker/director/provider.py#L534)).

**The complete system prompt, verbatim** ([provider.py:137-215](apps/worker/worker/director/provider.py#L137-L215), with `MAX_CHARACTERS`=4 and `MAX_SILENT_GAP`=6 already interpolated as they are at runtime):

```
You are a video director planning a short generated video with spoken dialogue.
Given an IDEA, a DURATION in seconds, and a DIALOGUE LANGUAGE, produce a JSON plan.

Output ONLY a JSON object, no markdown fences, no commentary, with this exact shape:
{
  "scene": "<one sentence: the single location and lighting, concrete and visual>",
  "tone": "<two or three words>",
  "ambience": "<the quiet background sounds of this place, a short phrase>",
  "characters": [
    {"id": "<short_snake_case>", "role": "<short noun phrase, e.g. detective>",
     "appearance": "<concrete visible description: age group, build, hair, clothing with colours>",
     "voice": "<audible voice description: pitch, pace, texture>"}
  ],
  "continuity": [
    "<a fact that must look identical in every frame: a prop's exact colour and
      shape, what each person is wearing, how many people are in the scene>"
  ],
  "timeline": [
    {"start": <seconds>, "end": <seconds>, "action": "<what is visibly happening>",
     "camera": "<one of: medium shot | medium close-up | close-up | two-shot |
       over-the-shoulder shot | wide shot; plus 'static' or a subtle move>",
     "speaker": "<character id or null>", "dialogue": "<the exact spoken words, or null>",
     "delivery": "<audible manner, e.g. 'low and accusing', or null>",
     "exits": ["<character ids who leave the scene at this event and stay gone;
       usually an empty list>"]}
  ]
}

Hard rules:
- Use ONLY the characters the idea implies, at most 4. Do not invent extra
  people. Keep every fact the idea states (counts, colours, named things) exactly as stated.
- "role" is the plain noun phrase prose would call them: "detective", "police chief",
  "woman", "robot". In "action" and "camera" text refer to characters by those role words
  only — NEVER by their id.
- If the idea already contains quoted dialogue or lines like 'Name: "..."', copy those
  spoken words VERBATIM into dialogue events. Never rewrite or drop them.
- All dialogue must be written in the DIALOGUE LANGUAGE.
- If the idea implies nobody would speak, leave every "dialogue" null rather than
  forcing a line into the scene.
- LINE COUNT: write the number of spoken lines given as TOTAL_LINES below. That is
  a target to hit, not a maximum to stay under. Fewer lines than that leaves the
  video silent for long stretches, which is worse than too many.
- THE CONVERSATION RUNS THE WHOLE VIDEO. Spread the lines from the first seconds
  to the last so someone is speaking or reacting throughout. Never leave more than
  6 seconds of the timeline with nobody speaking. The last spoken
  line lands near the end, not in the middle.
- Open on a SHORT establishing beat — one or two seconds of action nobody speaks
  over — then start the dialogue. Do not open with a long silence.
- Never put two spoken lines back to back without a reaction, action or pause
  between them — that is what makes two lines run together as one.
- Keep individual lines SHORT (a handful of words) and stay within the TOTAL_WORDS
  figure below. Many short lines beat a few long ones: short lines are what let the
  conversation cover the whole video without anyone rushing.
- The timeline covers 0 to DURATION seconds in 2-6 second events, in order, no overlaps.
- The conversation must progress: no line repeats an earlier line, and the last event
  resolves or lands the exchange.
- Dialogue needs readable faces: prefer medium shots, close-ups, two-shots and reaction
  shots, with a static camera or a subtle push-in. No fast camera moves.
- The ambience stays quiet under the voices. No background music unless the idea asks.
- Characters keep exactly the same appearance for the whole video.
- VOCABULARY: every line uses different words. If one line says "excellent", no other
  line may say "excellent" — pick another word. Reusing a distinctive word across lines
  is the single thing that makes generated dialogue sound generated.
- DEPARTURES: when the idea has someone leave the scene for good, give the event
  where they go an "exits" list with their character id. A departure is permanent:
  after that event the character never speaks, never acts and never appears again
  in any later event. If they would come back, they never left — use exits only
  for a real goodbye.
- CONTINUITY: list 2-5 facts that must look identical in every single frame. Always
  include what each person is wearing and how many people are present. If any prop is
  picked up, taken off, put down or handled during the scene, describe it there in
  concrete detail (exact colour, material, shape) — a thing that leaves the frame and
  comes back is where the picture drifts.
- Every continuity fact must stay true for the WHOLE video. If someone leaves
  partway through, the number of people changes — leave people-counts out of
  continuity and let the timeline carry the departure.
- Write continuity facts as things that STAY, never as things to avoid: "the red felt
  hat stays the same red felt hat every time it appears", not "the hat does not change".
```

**Appended for Image-to-Video only** — `_ANCHORED_RULES`, verbatim ([provider.py:223-245](apps/worker/worker/director/provider.py#L223-L245)):

```

SOURCE IMAGE MODE — this video starts from a photograph the user uploaded:
- The photograph is the video's exact first frame and its visual truth. The
  photograph decides WHO and WHAT is in the scene; the IDEA decides what happens
  next. You cannot see the photograph unless a PHOTOGRAPH FACTS block is given.
- Cast exactly the people and things the idea (and the PHOTOGRAPH FACTS block,
  when given) says are in the photograph. Never add or remove anyone.
- NEVER invent visible details. "appearance" may carry ONLY visual facts stated
  by the idea or the PHOTOGRAPH FACTS block; when neither states any for a
  character, set "appearance" to "" — their identity then comes from the
  photograph itself. The same rule applies to "scene": name the setting as
  stated, and add no imagined visual detail.
- CONTINUITY on this path: always include how many people are present, plus
  every visual fact the idea or the PHOTOGRAPH FACTS block states. Do not
  describe clothing or props neither of them mentions.
- The action moves FORWARD from the photographed moment. Do not re-stage or
  restart what the photograph already shows; the first event begins exactly
  where the photograph leaves off.
- Camera: the video opens on the photograph's own framing. Keep every shot
  inside the space the photograph establishes — a static camera, a subtle
  push-in, or cuts between the people already in frame. Never call for a
  reveal of anything the photograph does not show.
```

**Appended for a Director-lineage extension** — `_CONTINUATION_RULES`, verbatim ([provider.py:252-266](apps/worker/worker/director/provider.py#L252-L266)):

```

CONTINUATION MODE — this plan extends a video that is already finished:
- THE STORY SO FAR (given below) has ALREADY happened, completely, on screen,
  before second 0 of your timeline. Every question it implies has been asked,
  every answer given, every farewell said. None of it is re-staged, re-asked
  or re-answered.
- Second 0 of your timeline is the exact moment the finished video ends. The
  IDEA describes what happens next; plan only that, moving the same people in
  the same place FORWARD into new ground — new lines, new beats, the next
  stage of the same story.
- Keep the same characters, the same relationships and the same language the
  finished video established. Anyone the story so far sent away stays away.
- The photograph rules above apply: the opening frame is the finished video's
  last moment, and it decides who and what is in the scene.
```

**The user prompt template** ([provider.py:269-321](apps/worker/worker/director/provider.py#L269-L321)) — a real rendering for a 60-second T2V job appears in §12.

### 11.5 Global story plan, or per-section re-planning? — proved from code

**One global plan. Proven three ways.**

**(1) `create_director_plan` is called exactly once per job, before the chain starts.** [ltx.py:1098-1110](apps/worker/worker/adapters/ltx.py#L1098-L1110) sits *above* `render_chain` at [ltx.py:1147](apps/worker/worker/adapters/ltx.py#L1147). `grep -n 'create_director_plan' apps/worker/worker/adapters/ltx.py` → lines 1104 and 1224 only, both before their respective `render_chain` calls.

**(2) The compiled prompt list is memoised in a `nonlocal` and built once.** [ltx.py:1112-1125](apps/worker/worker/adapters/ltx.py#L1112-L1125), verbatim:

```python
prompt_plan: list[str] | None = None

def prompt_for_step(step: ChainStep) -> str:
    nonlocal prompt_plan
    if prompt_plan is None:
        if director_plan is not None:
            prompt_plan = compile_section_prompts(
                director_plan, step.total, total_seconds=seconds
            )
        else:
            prompt_plan = plan_section_prompts(
                job.prompt, step.total, total_seconds=seconds
            )
    return prompt_plan[step.index]
```

The `if prompt_plan is None` guard means passes 2..N return a slice of the list built during pass 1. No section can re-plan.

**(3) `compile_section_prompts` takes the whole plan and returns the whole list in one call** — it buckets, it does not generate. [compiler.py:73-106](apps/worker/worker/director/compiler.py#L73-L106):

```python
section_total = max(1, section_total)
window = total_seconds / section_total
buckets: list[list[DirectorEvent]] = [[] for _ in range(section_total)]
for event in plan.timeline:
    midpoint = (event.start + event.end) / 2
    index = min(section_total - 1, max(0, int(midpoint / window)))
    buckets[index].append(event)
...
captions = [
    _compile_section(plan, events, first=index == 0, cast=casts[index],
                     survivors=survivor_sets[index], window_end=(index + 1) * window)
    for index, events in enumerate(buckets)
]
```

The design intent is stated at [compiler.py:11-17](apps/worker/worker/director/compiler.py#L11-L17):

> Timing therefore lives entirely on this side of the prompt: each timeline event is bucketed into the generation window containing its midpoint … and each section's caption carries ONLY its own events. The global plan is written once, before any section renders — a section can never re-invent the conversation, which is precisely the long-form dialogue-restart failure this design exists to prevent.

**Contrast with standard mode:** `plan_section_prompts` is also called once and memoised the same way, so standard mode is equally global. The difference is only that it splits the *user's own text* rather than a generated plan.

### 11.6 State tracked across sections

| State | Tracked? | Mechanism | Evidence |
|---|---|---|---|
| **Entities (characters)** | Yes | `plan.characters` is global; each section re-introduces every present character's full appearance via `_full_subject`, because `introduced` is a fresh `set()` per section | [compiler.py:158](apps/worker/worker/director/compiler.py#L158), [compiler.py:647-663](apps/worker/worker/director/compiler.py#L647-L663) |
| **Events** | Yes | Bucketed by midpoint; each section's caption carries only its own | [compiler.py:74-79](apps/worker/worker/director/compiler.py#L74-L79) |
| **Dialogue** | Yes | Each line belongs to exactly one bucket, so it is spoken in exactly one section | same |
| **Presence / departures** | **Yes — the one genuinely stateful mechanism.** A `departed: set[str]` is walked across buckets *before* any caption is built | [compiler.py:87-94](apps/worker/worker/director/compiler.py#L87-L94) |
| **Emotion** | **No.** There is no emotion field. `delivery` is per-event and audible, `tone` is global and never compiled. | [plan.py:157-161](apps/worker/worker/director/plan.py#L157-L161) |
| **Camera** | **No** — reset per section (§10.3) | [compiler.py:194](apps/worker/worker/director/compiler.py#L194) |
| **Continuity facts** | Yes — restated at the end of **every** section | [compiler.py:370-371](apps/worker/worker/director/compiler.py#L370-L371) |
| **Voice** | Per section — `f"voice:{speaker.id}"` is added to the per-section `introduced` set, so a character's standing voice is restated the first time they speak in each section | [compiler.py:476-478](apps/worker/worker/director/compiler.py#L476-L478) |
| **Language** | Global — carried in `plan.language`, used for planning and logged; the dialogue text itself is the carrier | [compiler.py:114](apps/worker/worker/director/compiler.py#L114) |

The presence walk, verbatim ([compiler.py:87-94](apps/worker/worker/director/compiler.py#L87-L94)):

```python
departed: set[str] = set()
casts: list[list[DirectorCharacter]] = []
survivor_sets: list[list[DirectorCharacter]] = []
for events in buckets:
    casts.append([c for c in plan.characters if c.id not in departed])
    for event in events:
        departed.update(event.exits)
    survivor_sets.append([c for c in plan.characters if c.id not in departed])
```

### 11.7 Validation rules enforced after parsing (all deterministic)

| Rule | Raises? | Where |
|---|---|---|
| Not a JSON object | yes | [plan.py:304-305](apps/worker/worker/director/plan.py#L304-L305) |
| No `scene` | yes | [plan.py:310-311](apps/worker/worker/director/plan.py#L310-L311) |
| No characters, or > 4 | yes | [plan.py:349-357](apps/worker/worker/director/plan.py#L349-L357) |
| Duplicate character id | yes | [plan.py:372-374](apps/worker/worker/director/plan.py#L372-L374) |
| Missing role, or missing appearance (unless `source_anchored`) | yes | [plan.py:375-377](apps/worker/worker/director/plan.py#L375-L377) |
| No timeline, or > 24 events | yes | [plan.py:391-396](apps/worker/worker/director/plan.py#L391-L396) |
| Non-numeric or invalid time range | yes | [plan.py:404-412](apps/worker/worker/director/plan.py#L404-L412) |
| Event starting beyond `duration + 0.5` | yes | [plan.py:413-415](apps/worker/worker/director/plan.py#L413-L415) |
| Dialogue with no speaker | yes | [plan.py:434-436](apps/worker/worker/director/plan.py#L434-L436) |
| Unknown speaker | yes | [plan.py:437-439](apps/worker/worker/director/plan.py#L437-L439) |
| Event with neither action nor dialogue | yes | [plan.py:440-442](apps/worker/worker/director/plan.py#L440-L442) |
| Exit naming an unknown character | yes | [plan.py:474-477](apps/worker/worker/director/plan.py#L474-L477) |
| A character exiting twice | yes | [plan.py:495-499](apps/worker/worker/director/plan.py#L495-L499) |
| A departed character speaking later | yes | [plan.py:499-508](apps/worker/worker/director/plan.py#L499-L508) |
| A user-quoted line dropped or rewritten | yes | [plan.py:838-842](apps/worker/worker/director/plan.py#L838-L842) |
| Speech over budget | **no** — silently trims planner-invented dialogue from the end | [plan.py:516-557](apps/worker/worker/director/plan.py#L516-L557) |
| Presence counts in a plan with exits | **no** — silently dropped | [plan.py:577-590](apps/worker/worker/director/plan.py#L577-L590) |
| Ungrounded visual claims (anchored only) | **no** — silently dropped | [plan.py:593-636](apps/worker/worker/director/plan.py#L593-L636) |
| Pacing problems (too few lines, long silences) | **no** — reported back as retry notes; accepted on the last attempt | [plan.py:766-815](apps/worker/worker/director/plan.py#L766-L815), [provider.py:585-603](apps/worker/worker/director/provider.py#L585-L603) |
| Repeated vocabulary across lines | **no** — same posture as pacing | [plan.py:728-763](apps/worker/worker/director/plan.py#L728-L763) |
| `camera` string | **not validated at all** | [plan.py:419](apps/worker/worker/director/plan.py#L419) |
| `tone` string | not validated | [plan.py:308](apps/worker/worker/director/plan.py#L308) |
| `ambience` string | not validated | [plan.py:309](apps/worker/worker/director/plan.py#L309) |
---

## 12. Prompt Compilation Examples

### 12.0 Method

Every stage below except the Director *plan itself* was produced by **executing the repository's own functions** — `structure_prompt`, `plan_section_prompts`, `plan_chain_segments`, `system_prompt`, `user_prompt`, `compile_section_prompts` — on the worker's virtualenv, from a driver script outside the repository. Output is pasted exactly as printed.

**The DirectorPlan JSON is marked `RECONSTRUCTED`.** Producing a real one requires either the Cerebras API (a network call with a credential) or the local `gemma-4-e2b-it` checkpoint on a GPU node. Both are out of scope. The plan below was hand-authored to satisfy every rule `parse_plan` enforces, and was then passed through the **real** `compile_section_prompts`, so the *final prompts* are genuine compiler output.

**Test prompt used throughout:**

> *A detective enters an abandoned warehouse, finds a mysterious suitcase, hears footsteps upstairs, discovers a frightened woman, and they escape when fire begins spreading.*

---

### 12.1 5-second T2V — standard mode

```
USER PROMPT
  ↓
A detective enters an abandoned warehouse, finds a mysterious suitcase, hears
footsteps upstairs, discovers a frightened woman, and they escape when fire
begins spreading.
```

```
  ↓  structure_prompt()   [execution.prompt_structuring: true]
```

Output, verbatim (`\n` shown as real newlines):

```
A detective enters an abandoned warehouse, finds a mysterious suitcase, hears footsteps upstairs, discovers a frightened woman, and they escape when fire begins spreading.

CONTINUITY (fixed for the entire video):
- The same subjects keep the same faces, clothing, colours and count in every frame.
- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.
```

No count or colour rules were derived: `_COUNT_PATTERN` found no `<number> <plural noun>` pair and `_COLOUR_PATTERN` found no colour word. Only the two standing rules were appended.

```
  ↓  DIRECTOR PLAN        — not present (standard mode)
  ↓  GLOBAL CAMERA PLAN   — not present (no such artifact exists, §10.3)
  ↓  plan_chain_segments(5.0, per_pass=30.0, None) → 1 section, [5.0]
  ↓  plan_section_prompts(structured, 1, total_seconds=5.0)
```

**SECTION 1 FINAL PROMPT (verbatim, exactly as sent as the `--prompt` argv element):**

```
A detective enters an abandoned warehouse, finds a mysterious suitcase, hears footsteps upstairs, discovers a frightened woman, and they escape when fire begins spreading.

CONTINUITY (fixed for the entire video):
- The same subjects keep the same faces, clothing, colours and count in every frame.
- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.
```

Byte-identical to `structure_prompt`'s output, because `plan_section_prompts` returns `[master_prompt]` unchanged when `section_total <= 1` ([prompts.py:97-98](apps/worker/worker/longform/prompts.py#L97-L98)).

```
  ↓  SEAM STATE           — none; one pass, no seam
```

---

### 12.2 60-second T2V — standard mode, both sections

```
plan_chain_segments(60.0, per_pass=30.0, None) → 2 sections, [30.0, 30.0]
plan_section_prompts(structured, 2, total_seconds=60.0)
```

**SECTION 1 FINAL PROMPT (verbatim):**

```
LONG-FORM CONTINUATION — SECTION 1 OF 2.
Keep the same subjects, identities, faces, clothing, colours, object counts, vehicles, environment and camera direction established previously.
PERSISTENT USER CONSTRAINTS (verbatim):
A detective enters an abandoned warehouse, finds a mysterious suitcase, hears footsteps upstairs, discovers a frightened woman, and they escape when fire begins spreading.
CONTINUITY (fixed for the entire video):
- The same subjects keep the same faces, clothing, colours and count in every frame.
NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY (verbatim):
- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.
Continue directly from the predecessor frame. Do not replay, restart or summarise any earlier action or dialogue. Complete this section's assigned dialogue before the section ends.
```

**SEAM STATE — what carries forward:**

| Carried | Value |
|---|---|
| Picture | `segment-condition-0001.png` — the last decodable frame of `segment-0000.mp4`, extracted by `ffmpeg -sseof -1 … -update 1` |
| Passed as | `--image /workspace/job/segment-condition-0001.png 0 1.0` |
| Text | Only the standing header + the `PERSISTENT USER CONSTRAINTS` block; the previous section's assigned actions are **not** repeated |
| Seed | Independent: `crc32("…:1")` = `863830479`, not `crc32("…:0")` = `1148858713` |
| Nothing else | No latent, no KV cache, no model residency — a fresh subprocess |

**SECTION 2 FINAL PROMPT (verbatim):**

```
LONG-FORM CONTINUATION — SECTION 2 OF 2.
Keep the same subjects, identities, faces, clothing, colours, object counts, vehicles, environment and camera direction established previously.
PERSISTENT USER CONSTRAINTS (verbatim):
A detective enters an abandoned warehouse, finds a mysterious suitcase, hears footsteps upstairs, discovers a frightened woman, and they escape when fire begins spreading.
CONTINUITY (fixed for the entire video):
- The same subjects keep the same faces, clothing, colours and count in every frame.
NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY (verbatim):
Continue naturally from the preceding section without introducing a new event.
Continue directly from the predecessor frame. Do not replay, restart or summarise any earlier action or dialogue. Complete this section's assigned dialogue before the section ends.
```

The `NEW ACTION` fallback line is the literal at [prompts.py:117](apps/worker/worker/longform/prompts.py#L117).

---

### 12.3 60-second T2V — Director (Idea) mode

```
USER PROMPT (the IDEA)
  ↓  structure_prompt   — SKIPPED (wants_director(job) is True; ltx.py:1061)
  ↓  system_prompt(request) + user_prompt(request)  →  the planning model
```

**DIRECTOR USER PROMPT (verbatim, real output of `user_prompt`):**

```
IDEA: A detective enters an abandoned warehouse, finds a mysterious suitcase, hears footsteps upstairs, discovers a frightened woman, and they escape when fire begins spreading.
DURATION: 60 seconds
DIALOGUE LANGUAGE: the same language the idea is written in
TOTAL_LINES: write 15 spoken lines across the whole video. Not fewer than 14, and never more than 30.
TOTAL_WORDS: keep the spoken words under 115 in total, which is roughly 7 words per line.
```

Derivation of those two computed lines:
- `target_spoken_lines(60.0)` = `min(spoken_line_budget(60.0), max(2, ceil(60.0/4.0)))` = `min(30, 15)` = **15**
- floor = `max(2, 15 - 1)` = **14**; ceiling = `int(60.0 // 2)` = **30**
- `speech_budget(60.0)` = `int((60.0 - 2.5) * 2.0)` = **115**
- words per line = `max(3, 115 // 15)` = **7**

The system prompt is the 79-line brief pasted in full at §11.4.

```
  ↓  DIRECTOR PLAN (full JSON)   —  RECONSTRUCTED
```

```json
{
  "scene": "A derelict warehouse at night, lit by a single hanging bulb and shafts of moonlight",
  "tone": "tense, urgent",
  "ambience": "dripping water and distant traffic",
  "characters": [
    {"id": "detective", "role": "detective",
     "appearance": "a man in his forties, heavy build, short grey hair, a charcoal overcoat over a white shirt",
     "voice": "low and gravelled"},
    {"id": "woman", "role": "frightened woman",
     "appearance": "a young woman, slim, long dark hair, a torn red jacket",
     "voice": "thin and breathless"}
  ],
  "continuity": [
    "the battered brown leather suitcase keeps the same brass clasps every time it appears",
    "the detective's charcoal overcoat stays charcoal for the whole video",
    "the woman's torn red jacket stays the same torn red jacket"
  ],
  "timeline": [
    {"start": 0.0,  "end": 5.0,  "action": "The detective pushes the steel door open and steps inside",
     "camera": "wide shot, static",             "speaker": null,        "dialogue": null,                              "delivery": null,               "exits": []},
    {"start": 5.0,  "end": 10.0, "action": "The detective crouches beside a battered case",
     "camera": "medium shot, subtle push-in",   "speaker": "detective", "dialogue": "Somebody left this in a hurry.",   "delivery": "low and wary",     "exits": []},
    {"start": 12.0, "end": 17.0, "action": "The detective looks up at the ceiling",
     "camera": "close-up, static",              "speaker": "detective", "dialogue": "Footsteps. Upstairs.",             "delivery": "hushed",           "exits": []},
    {"start": 20.0, "end": 26.0, "action": "The frightened woman edges out from behind a pillar",
     "camera": "two-shot, static",              "speaker": "woman",     "dialogue": "Please. Don't let him find me.",   "delivery": "thin and shaking", "exits": []},
    {"start": 30.0, "end": 36.0, "action": "The detective holds out a steady hand",
     "camera": "medium close-up, static",       "speaker": "detective", "dialogue": "You're with me now.",              "delivery": "steady",           "exits": []},
    {"start": 40.0, "end": 46.0, "action": "Flame climbs a stack of pallets behind them",
     "camera": "wide shot, slowly pans",        "speaker": "woman",     "dialogue": "The whole floor is burning.",      "delivery": "rising and urgent","exits": []},
    {"start": 50.0, "end": 58.0, "action": "They run together toward the loading bay",
     "camera": "tracking shot, follows them",   "speaker": "detective", "dialogue": "Straight through. Do not stop.",   "delivery": "sharp",            "exits": []}
  ]
}
```

```
  ↓  GLOBAL CAMERA PLAN   — not present. Camera lives per-event inside the
                            timeline above; there is no separate artifact. (§10.3)
  ↓  SECTION 1 PLAN       — buckets: window = 60.0/2 = 30.0
                            midpoints 2.5, 7.5, 14.5, 23.0 → bucket 0
                            midpoints 33.0, 43.0, 54.0     → bucket 1
                            cast[0] = [detective, woman]; survivors[0] = same (no exits)
  ↓  compile_section_prompts(plan, 2, total_seconds=60.0)     [REAL compiler]
```

**SECTION 1 FINAL PROMPT (verbatim, real compiler output):**

```
A derelict warehouse at night, lit by a single hanging bulb and shafts of moonlight. The detective, a man in his forties, heavy build, short grey hair, a charcoal overcoat over a white shirt, and the frightened woman, a young woman, slim, long dark hair, a torn red jacket, are here from the first frame. A wide shot frames the moment, and the camera remains static. Initially, the detective pushes the steel door open and steps inside. A medium shot frames the moment as the camera makes a subtle push-in. A moment later, the detective crouches beside a battered case, and says in a low and wary voice, "Somebody left this in a hurry." A close-up frames the moment, and the camera remains static. A beat of silence passes, and then the detective looks up at the ceiling, and says in a hushed voice, "Footsteps. Upstairs." A two-shot frames the moment, and the camera remains static. The room holds still for a moment, and then the frightened woman edges out from behind a pillar, and says in a thin and shaking voice, "Please. Don't let him find me." For the remaining seconds the exchange settles: they hold each other's gaze with small natural movements, and the room's ambience is the only sound. Each line of dialogue is spoken a single time, and the exchange moves forward to the next speaker as soon as it lands. Under the voices, dripping water and distant traffic, with no background music. The detective and the frightened woman keep exactly the same faces, clothing and voices for the entire video. The detective and the frightened woman stay fully visible in the frame from the first frame to the last, present and solid in every single frame. The battered brown leather suitcase keeps the same brass clasps every time it appears. The detective's charcoal overcoat stays charcoal for the whole video. The frightened woman's torn red jacket stays the same torn red jacket.
```

Note the `_settle_sentence` at the end of the events — emitted because `window_end - events[-1].end = 30.0 - 26.0 = 4.0 > 2.5` ([compiler.py:230-231](apps/worker/worker/director/compiler.py#L230-L231)).

```
  ↓  SEAM STATE
```

| Carried | Value |
|---|---|
| Picture | `segment-condition-0001.png` at `--image … 0 1.0` |
| `departed` set | `set()` — this plan has no exits |
| `cast[1]` | `[detective, woman]` |
| `introduced` set | **Reset** — a fresh `set()` per section, so both appearances are restated in full |
| `previous_camera` | **Reset** to `""` |
| Prompt-plan cache | Already built; section 2 is `prompt_plan[1]` |

**SECTION 2 FINAL PROMPT (verbatim, real compiler output):**

```
A derelict warehouse at night, lit by a single hanging bulb and shafts of moonlight. The detective, a man in his forties, heavy build, short grey hair, a charcoal overcoat over a white shirt, and the frightened woman, a young woman, slim, long dark hair, a torn red jacket, continue mid-scene, identical to before in face, clothing and voice, without repeating any earlier action or line. A medium close-up frames the moment, and the camera remains static. The detective holds out a steady hand, and says in a steady voice, "You're with me now." A wide shot frames the moment as the camera slowly pans. After a short pause, the frightened woman flame climbs a stack of pallets behind them, and says in a rising and urgent voice, "The whole floor is burning." A tracking shot frames the moment as the camera follows them. A beat of silence passes, and then the detective run together toward the loading bay, and says in a sharp voice, "Straight through. Do not stop." Each line of dialogue is spoken a single time, and the exchange moves forward to the next speaker as soon as it lands. Under the voices, dripping water and distant traffic, with no background music. The detective and the frightened woman keep exactly the same faces, clothing and voices for the entire video. The detective and the frightened woman stay fully visible in the frame from the first frame to the last, present and solid in every single frame. The battered brown leather suitcase keeps the same brass clasps every time it appears. The detective's charcoal overcoat stays charcoal for the whole video. The frightened woman's torn red jacket stays the same torn red jacket.
```

Differences from section 1, all produced by the compiler: `_cast_sentence` → the "continue mid-scene" form ([compiler.py:166-176](apps/worker/worker/director/compiler.py#L166-L176)); `_transition(0, …, first_section=False)` returns `""` so the first event has no "Initially"; no settle sentence (`60.0 - 58.0 = 2.0 ≤ 2.5`).

Two artifacts of `_action_clause` visible in this output are recorded in **Appendix C** without conclusion.

---

### 12.4 5-second Director T2V

```
plan_chain_segments(5.0, per_pass=30.0, None) → 1 section
compile_section_prompts(plan, 1, total_seconds=5.0)   — ALL seven events land in the one bucket
```

**SECTION 1 FINAL PROMPT (verbatim, real compiler output):**

```
A derelict warehouse at night, lit by a single hanging bulb and shafts of moonlight. The detective, a man in his forties, heavy build, short grey hair, a charcoal overcoat over a white shirt, and the frightened woman, a young woman, slim, long dark hair, a torn red jacket, are here from the first frame. A wide shot frames the moment, and the camera remains static. Initially, the detective pushes the steel door open and steps inside. A medium shot frames the moment as the camera makes a subtle push-in. A moment later, the detective crouches beside a battered case, and says in a low and wary voice, "Somebody left this in a hurry." A close-up frames the moment, and the camera remains static. A beat of silence passes, and then the detective looks up at the ceiling, and says in a hushed voice, "Footsteps. Upstairs." A two-shot frames the moment, and the camera remains static. The room holds still for a moment, and then the frightened woman edges out from behind a pillar, and says in a thin and shaking voice, "Please. Don't let him find me." A medium close-up frames the moment, and the camera remains static. After a short pause, the detective holds out a steady hand, and says in a steady voice, "You're with me now." A wide shot frames the moment as the camera slowly pans. A beat of silence passes, and then the frightened woman flame climbs a stack of pallets behind them, and says in a rising and urgent voice, "The whole floor is burning." A tracking shot frames the moment as the camera follows them. The room holds still for a moment, and then the detective run together toward the loading bay, and says in a sharp voice, "Straight through. Do not stop." Each line of dialogue is spoken a single time, and the exchange moves forward to the next speaker as soon as it lands. Under the voices, dripping water and distant traffic, with no background music. The detective and the frightened woman keep exactly the same faces, clothing and voices for the entire video. The detective and the frightened woman stay fully visible in the frame from the first frame to the last, present and solid in every single frame. The battered brown leather suitcase keeps the same brass clasps every time it appears. The detective's charcoal overcoat stays charcoal for the whole video. The frightened woman's torn red jacket stays the same torn red jacket.
```

*(A real 5-second job would carry a plan built for 5 seconds — `target_spoken_lines(5.0) = 2`, `speech_budget(5.0) = 5` words. This snapshot reuses the 60-second plan so the two captions are directly comparable; the bucketing behaviour it demonstrates — every event into one caption — is genuine.)*

For completeness, the **real** Director user prompt a 5-second I2V job produces:

```
IDEA: A detective enters an abandoned warehouse, finds a mysterious suitcase, hears footsteps upstairs, discovers a frightened woman, and they escape when fire begins spreading.
DURATION: 5 seconds
DIALOGUE LANGUAGE: the same language the idea is written in
TOTAL_LINES: write 2 spoken lines across the whole video. Not fewer than 2, and never more than 2.
TOTAL_WORDS: keep the spoken words under 5 in total, which is roughly 3 words per line.
```

---

### 12.5 Image-to-Video — Director, source-anchored

The anchored planning brief adds `_ANCHORED_RULES` (pasted at §11.4). After `parse_plan`, `_ground_visual_claims` strips every visual claim not supported by the idea text — with `director_vision_enabled: False` there are no PHOTOGRAPH FACTS, so the vocabulary is the idea alone.

Applying the real `_ground_visual_claims` logic to the plan above: `"a man in his forties, heavy build, short grey hair, a charcoal overcoat over a white shirt"` contains distinctive words (`forties`, `heavy`, `build`, `grey`, `hair`, `charcoal`, `overcoat`, `white`, `shirt`) that the idea does not supply, so `appearance` becomes `""`. The same for the woman. `scene` is replaced by `ANCHORED_SCENE`. `continuity` empties.

**I2V SECTION 1 FINAL PROMPT (verbatim, real compiler output on the grounded plan):**

```
The scene continues exactly as the opening frame shows it. The detective and the frightened woman are already present in the opening frame, and they keep exactly the appearance that frame shows for the whole video. A wide shot frames the moment, and the camera remains static. Initially, the detective pushes the steel door open and steps inside. A medium shot frames the moment as the camera makes a subtle push-in. A moment later, the detective crouches beside a battered case, and says in a low and wary voice, "Somebody left this in a hurry." Each line of dialogue is spoken a single time, and the exchange moves forward to the next speaker as soon as it lands. Under the voices, dripping water and distant traffic, with no background music. The detective and the frightened woman keep exactly the same faces, clothing, hair, colours and voices they have in the first frame, for the entire video. The detective and the frightened woman stay fully visible in the frame from the first frame to the last, present and solid in every single frame.
```

Three compiler differences from the unanchored form, all driven by `source_anchored=True`:
1. `_anchored_cast_sentence` ("are already present in the opening frame") instead of `_cast_sentence` ("are here from the first frame") — [compiler.py:403-421](apps/worker/worker/director/compiler.py#L403-L421)
2. The constancy sentence names `hair` and `colours` and anchors to "the first frame" — [compiler.py:338-352](apps/worker/worker/director/compiler.py#L338-L352)
3. No `continuity` sentences, because grounding emptied the list

**I2V — standard mode** produces exactly the §12.1 / §12.2 prompts. There is no I2V-specific text on the standard path; the difference is entirely in the conditioning (`--image <still> 0 1.0`).

---

### 12.6 Music Video

Music-video prompts are timestamped by convention, so a realistic input is used:

```
A dancer in a neon warehouse.
0:00-0:20 wide shot, the dancer alone
0:20-0:40 close-up on their face
1:00-1:20 the crowd floods in
```

```
  ↓  structure_prompt()   [music-video.yaml:71 prompt_structuring: true]
```

**Output, verbatim:**

```
A dancer in a neon warehouse.
0:00-0:20 wide shot, the dancer alone
0:20-0:40 close-up on their face
1:00-1:20 the crowd floods in

CONTINUITY (fixed for the entire video):
- Exactly 20 crowd floods appear, and they remain the only floods on screen for the entire video.
- The same subjects keep the same faces, clothing, colours and count in every frame.
- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.
```

*(The first derived rule is a real, reproduced output of `_COUNT_PATTERN` against the substring `"20 the crowd floods"`. Recorded without conclusion in Appendix C.)*

```
  ↓  DIRECTOR              — not present (music-video is not in _DIRECTOR_WORKFLOWS)
  ↓  GLOBAL CAMERA PLAN    — not present
  ↓  _musical_boundaries → plan_chain_segments(180.0, 60.0, boundaries) → 3 sections
  ↓  plan_section_prompts(structured, 3, total_seconds=180.0)
       _TIMED_LINE matched all three shot lines → _distribute_timed
       window = 180.0/3 = 60.0
       midpoints: 10.0 → bucket 0 ; 30.0 → bucket 0 ; 70.0 → bucket 1 ; bucket 2 empty
```

**MV SECTION 1 FINAL PROMPT (verbatim):**

```
LONG-FORM CONTINUATION — SECTION 1 OF 3.
Keep the same subjects, identities, faces, clothing, colours, object counts, vehicles, environment and camera direction established previously.
PERSISTENT USER CONSTRAINTS (verbatim):
A dancer in a neon warehouse.
CONTINUITY (fixed for the entire video):
- Exactly 20 crowd floods appear, and they remain the only floods on screen for the entire video.
- The same subjects keep the same faces, clothing, colours and count in every frame.
- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.
NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY (verbatim):
wide shot, the dancer alone
close-up on their face
Continue directly from the predecessor frame. Do not replay, restart or summarise any earlier action or dialogue. Complete this section's assigned dialogue before the section ends.
```

**SEAM STATE:** `scene-condition-0001.png` at `--image … 0 1.0`. On the audio tier, additionally `--audio-start-time` advances to the section's `start_seconds`. Nothing else.

**MV SECTION 2 FINAL PROMPT (verbatim):**

```
LONG-FORM CONTINUATION — SECTION 2 OF 3.
Keep the same subjects, identities, faces, clothing, colours, object counts, vehicles, environment and camera direction established previously.
PERSISTENT USER CONSTRAINTS (verbatim):
A dancer in a neon warehouse.
CONTINUITY (fixed for the entire video):
- Exactly 20 crowd floods appear, and they remain the only floods on screen for the entire video.
- The same subjects keep the same faces, clothing, colours and count in every frame.
- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.
NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY (verbatim):
the crowd floods in
Continue directly from the predecessor frame. Do not replay, restart or summarise any earlier action or dialogue. Complete this section's assigned dialogue before the section ends.
```

**MV SECTION 3 FINAL PROMPT (verbatim):**

```
LONG-FORM CONTINUATION — SECTION 3 OF 3.
Keep the same subjects, identities, faces, clothing, colours, object counts, vehicles, environment and camera direction established previously.
PERSISTENT USER CONSTRAINTS (verbatim):
A dancer in a neon warehouse.
CONTINUITY (fixed for the entire video):
- Exactly 20 crowd floods appear, and they remain the only floods on screen for the entire video.
- The same subjects keep the same faces, clothing, colours and count in every frame.
- One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end.
NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY (verbatim):
Continue naturally from the preceding section without introducing a new event.
Continue directly from the predecessor frame. Do not replay, restart or summarise any earlier action or dialogue. Complete this section's assigned dialogue before the section ends.
```

---

### 12.7 Video-to-Video

**`— no compilation stage exists.`** `_run_restyle` and `_run_transform` never pass `prompt_for_step` to `_renderer`, so `_command` falls through to `job.prompt` at [ltx.py:2723](apps/worker/worker/adapters/ltx.py#L2723) for **every** section. `prompt_structuring` is not declared on `video-to-video.yaml`, so `structure_prompt` does not run either.

The **only** transformation applied is the reference-person caption under identity replacement ([ltx.py:1544-1561](apps/worker/worker/adapters/ltx.py#L1544-L1561)):

```python
facts = await cancellable(job, reference_person_facts(reference))
if facts:
    described = " ".join(facts.split()).rstrip(".")
    job = replace(
        job,
        prompt=(
            f"{job.prompt}\n\n"
            f"The person is {described}. The same person, with the "
            "same face, hair and clothing, stays on screen for the "
            "whole video."
        ),
    )
```

So for a user prompt of `Turn this into a rain-soaked neon street.` with identity replacement active, every pass receives:

```
Turn this into a rain-soaked neon street.

The person is <one sentence from the gemma vision describer>. The same person, with the same face, hair and clothing, stays on screen for the whole video.
```

The describer's brief is at [vision.py:72-81](apps/worker/worker/director/vision.py#L72-L81) and is capped at 350 characters. A failure returns `""` and the prompt goes through unchanged.

---

## 13. Tests

**Run performed:** `python -m pytest -p no:randomly -q` in `apps/worker/`, on the audit machine, 21 Aug 2026.
**Result: `15 failed, 788 passed, 1 skipped in 1406.30s (0:23:26)`.**

The API suite (`apps/api/`) was **not run** — it requires a live PostgreSQL and Redis, which would mean starting infrastructure. Recorded in Appendix B.

### 13.1 Test inventory

| Test file | What it covers | LTX settings asserted | Passing? |
|---|---|---|---|
| [tests/test_ltx.py](apps/worker/tests/test_ltx.py) (63 tests) | The whole adapter: argv shape, prompt verbatimness, seeds, grids, the frame tables, chaining, weight preflight, cancellation, process-group kill, marker parsing | **The heaviest pin in the repo.** See §13.2 | ✅ all pass |
| [tests/test_guided.py](apps/worker/tests/test_guided.py) (12) | `_GUIDED` tier: module name, dev transformer, `--offload cpu`, `--distilled-lora`, guidance-flag emission, `measured_landings=(121,)` | pipeline identity, landings, guidance flags | ✅ |
| [tests/test_transform.py](apps/worker/tests/test_transform.py) (15) | `_IC_LORA`: `--lora`, `--video-conditioning`, `--skip-stage-2`, grid doubling, `measured_landings=(193,)` | pipeline identity, control conditioning, landings | ✅ |
| [tests/test_music_video_audio.py](apps/worker/tests/test_music_video_audio.py) (12) | `_A2VID`: `--audio-path`/`--audio-start-time`/`--audio-max-duration`, window padding, landings | audio triple, `_audio_window_seconds` | ✅ |
| [tests/test_music_video.py](apps/worker/tests/test_music_video.py) (13) | Default music-video path: section count, one continuous track, muxing, no audio flag | pass count, mux behaviour | ❌ **1 of 13 fails** — see §13.3 |
| [tests/test_video_to_video.py](apps/worker/tests/test_video_to_video.py) (19) | Restyle conditioning triples, keyframe density, source-audio restoration, source-duration matching | `v2v_structure_strength`, `v2v_continuity_strength`, keyframe counts | ✅ |
| [tests/test_reference_identity.py](apps/worker/tests/test_reference_identity.py) (16) | Identity anchor strength, refresh gating, subject attention, mutual exclusion with person lock, matting-failure posture | all `v2v_identity_*` values | ✅ |
| [tests/test_person_lock.py](apps/worker/tests/test_person_lock.py) (10) | Hybrid control, `BACKGROUND_ATTENTION` | `v2v_background_attention` | ✅ |
| [tests/test_person_anchor.py](apps/worker/tests/test_person_anchor.py) (12) | Composited-anchor CLI contract and fallback | raw-anchor cap `0.65` | ✅ |
| [tests/test_longform.py](apps/worker/tests/test_longform.py) (24) | `plan_segments` invariants, `render_chain`, seam frames, even-window rule | segmentation arithmetic | ✅ |
| [tests/test_seam_timing.py](apps/worker/tests/test_seam_timing.py) (5) | `_planned_section_frames` cumulative-boundary allocation | frame allocation | ✅ |
| [tests/test_prompt_structuring.py](apps/worker/tests/test_prompt_structuring.py) (15) | `structure_prompt` rules, already-structured bail-out | the CONTINUITY block text | ✅ |
| [tests/test_director.py](apps/worker/tests/test_director.py) (35) | Plan validation, speech budget, exits, verbatim quotes | `WORDS_PER_SECOND`, `MAX_CHARACTERS`, `MAX_SILENT_GAP` | ✅ |
| [tests/test_director_i2v.py](apps/worker/tests/test_director_i2v.py) (18) | Anchored register, `_ground_visual_claims` | anchored sentence forms | ✅ |
| [tests/test_director_extend.py](apps/worker/tests/test_director_extend.py) (7) | Lineage-driven continuation | continuation register | ✅ |
| [tests/test_director_presence.py](apps/worker/tests/test_director_presence.py) (16) | Exits, cast/survivor scoping | presence walk | ✅ |
| [tests/test_cerebras_director.py](apps/worker/tests/test_cerebras_director.py) (14) | Hosted planner availability and fallback | model name, temperature | ✅ |
| [tests/test_media.py](apps/worker/tests/test_media.py) (30) | `plan_segments`, `concat_segments`, `normalize_clip`, `verify_output` | encoder settings | ✅ |
| [tests/test_media_audio.py](apps/worker/tests/test_media_audio.py) (16) | `mux_audio`, onsets, crossfade | `_AAC_ARGS`, pad tolerances | ✅ |
| [tests/test_runner.py](apps/worker/tests/test_runner.py) (17) | `JobRunner`: staging, leases, timeouts, cleanup | `execution.timeout_seconds` handling | ❌ **14 of 17 fail** — see §13.3 |
| [tests/test_worker.py](apps/worker/tests/test_worker.py) (17) | Registration, runtime list | `RUNTIMES` semantics | ✅ |
| [tests/test_client.py](apps/worker/tests/test_client.py) (14) | API client | — | ✅ |
| [tests/test_fleet.py](apps/worker/tests/test_fleet.py) (6) | Multi-runtime routing | `runtime_list` | ✅ |
| [tests/test_harness.py](apps/worker/tests/test_harness.py) (10) | The ffmpeg-only adapter | `settings.max_segment_seconds` | ✅ |
| [tests/test_deployment_layout.py](apps/worker/tests/test_deployment_layout.py) (4) | Image/file layout | — | ✅ |
| [tests/test_transfer.py](apps/worker/tests/test_transfer.py) (7) | Presigned upload/download | — | ✅ |
| [tests/test_music.py](apps/worker/tests/test_music.py) (44), [test_acestep_provider.py](apps/worker/tests/test_acestep_provider.py) (19), [test_cerebras_lyrics.py](apps/worker/tests/test_cerebras_lyrics.py) (34), [test_lyrics_writer.py](apps/worker/tests/test_lyrics_writer.py) (18), [test_music_language.py](apps/worker/tests/test_music_language.py) (18), [test_lyrics_language_detection.py](apps/worker/tests/test_lyrics_language_detection.py) (9) | The non-LTX music runtime | — | ✅ |

### 13.2 Tests that pin generation parameters

These are the assertions that would break if a setting changed. Named individually, because they are the change-detection surface for the later comparison.

| Test | Pins |
|---|---|
| `test_the_command_carries_every_flag_the_benchmark_needed` ([test_ltx.py:70](apps/worker/tests/test_ltx.py#L70)) | The complete distilled argv: all six weight paths, `--quantization`, `--num-frames`, `--height`, `--width`, `--frame-rate`, `--seed`, `--output-path` |
| `test_the_users_prompt_reaches_the_model_verbatim` ([test_ltx.py:108](apps/worker/tests/test_ltx.py#L108)) | That `--prompt` is byte-identical to the input, parametrised over several prompts |
| `test_the_prompt_is_never_rewritten_between_the_claim_and_the_command` ([test_ltx.py:128](apps/worker/tests/test_ltx.py#L128)) | Same, end to end from the claim payload |
| `test_prompt_enhancement_is_off_unless_a_workflow_asks_for_it` ([test_ltx.py:154](apps/worker/tests/test_ltx.py#L154)) | `--enhance-prompt` absent by default |
| `test_every_dimension_is_divisible_by_64` ([test_ltx.py:179](apps/worker/tests/test_ltx.py#L179)) | The `_DIMENSIONS` table |
| `test_seeds_differ_between_jobs_and_repeat_within_one` ([test_ltx.py:190](apps/worker/tests/test_ltx.py#L190)) | The `crc32` seed strategy |
| `test_a_user_seed_reaches_each_section_deterministically` ([test_ltx.py:203](apps/worker/tests/test_ltx.py#L203)) | `(base + index) % 2**31` |
| `test_no_public_duration_can_become_an_oversized_gpu_pass` ([test_ltx.py:272](apps/worker/tests/test_ltx.py#L272)) | That every value in every `supported_durations` list stays within its grid ceiling |
| `test_a_workflow_cannot_raise_the_per_pass_ceiling_above_the_benchmark` ([test_ltx.py:291](apps/worker/tests/test_ltx.py#L291)) | The clamp chain in `_per_pass_seconds` |
| `test_the_pass_ceiling_is_a_property_of_the_grid_not_the_product` ([test_ltx.py:313](apps/worker/tests/test_ltx.py#L313)) | `_GRID_CEILINGS` |
| `TestSafeFrameCount` — 14 tests ([test_ltx.py:346-447](apps/worker/tests/test_ltx.py#L346-L447)) | Every entry of `_MEASURED_SAFE_CONDITIONED`, `_CONDITIONED_BANDS`, `_BAD_FRAME_BANDS`, and the properties "never below the request", "overshoot ≤ ⅓ s", "361 and 721 never emitted" |
| `test_an_unmeasured_grid_gets_the_pessimistic_ceiling` ([test_ltx.py:449](apps/worker/tests/test_ltx.py#L449)) | `_UNMEASURED_CEILING = 10.0` |
| `test_the_command_pins_the_still_as_frame_zero_at_full_strength` ([test_ltx.py:626](apps/worker/tests/test_ltx.py#L626)) | `--image <still> 0 1.0` |
| `test_a_text_to_video_command_carries_no_image_flag` ([test_ltx.py:642](apps/worker/tests/test_ltx.py#L642)) | T2V pass 1 conditioning |
| `test_the_generation_grid_follows_the_source_aspect` ([test_ltx.py:792](apps/worker/tests/test_ltx.py#L792)) | `grid_for_source` outputs, parametrised |
| `test_every_reachable_grid_is_legal_for_the_model` ([test_ltx.py:819](apps/worker/tests/test_ltx.py#L819)) | /64 divisibility across the whole search space |
| `test_output_resolution_is_the_sources_capped_and_even` ([test_ltx.py:835](apps/worker/tests/test_ltx.py#L835)) | `output_dimensions`, 1920/1080 caps |
| `test_every_public_duration_is_a_single_pass_on_a_measured_grid` ([test_ltx.py:842](apps/worker/tests/test_ltx.py#L842)) | The duration menus against the ceilings |
| `test_the_extension_command_pins_dimensions_and_seed` ([test_ltx.py:866](apps/worker/tests/test_ltx.py#L866)) | Extension argv |
| `test_an_extension_source_is_not_held_to_the_render_source_ceiling` ([test_ltx.py:908](apps/worker/tests/test_ltx.py#L908)) | `ltx_max_extend_source_seconds` |
| `test_markers_walk_forward_through_the_real_log_sequence` ([test_ltx.py:1069](apps/worker/tests/test_ltx.py#L1069)) | `_MARKERS` strings |
| `test_a_shape_crash_is_not_retried` ([test_ltx.py:1143](apps/worker/tests/test_ltx.py#L1143)) | `_DETERMINISTIC_FAILURES` |
| `test_missing_weights_fail_before_any_subprocess` ([test_ltx.py:571](apps/worker/tests/test_ltx.py#L571)) | `_MODEL_FILES` completeness |

### 13.3 Failure status — 15 failures, both causes environmental

**Reported as status only. Nothing was fixed.**

**Cause 1 — 14 failures in `tests/test_runner.py`.** Every one is a test that drives `JobRunner.run(...)` end to end. The cause was captured by attaching a log handler to `worker.jobs.runner` outside the repository:

```
INTERNAL_DETAIL: workspace has 389MB free, below MIN_FREE_DISK_MB=2048
RETRIABLE: True
```

The guard is [workspace.py:39-45](apps/worker/worker/jobs/workspace.py#L39-L45):

```python
free_mb = shutil.disk_usage(root).free // _BYTES_PER_MB
if free_mb < settings.min_free_disk_mb:
    ...
    f"workspace has {free_mb}MB free, "
    f"below MIN_FREE_DISK_MB={settings.min_free_disk_mb}"
```

`df` on the audit machine reports `C: 146G total, 144G used, 1.7G available (99%)`. The default `min_free_disk_mb` is `2048` ([config.py:157](apps/worker/worker/core/config.py#L157)). **This is a property of the machine the audit ran on, not of the code.** The 14 tests are:

```
test_a_job_runs_uploads_and_completes
test_the_adapter_receives_a_real_writable_workspace
test_the_workspace_is_removed_after_success
test_the_workspace_is_removed_after_failure
test_a_retry_does_not_inherit_the_previous_attempt_s_files
test_inputs_are_staged_to_disk_and_handed_to_the_adapter
test_a_silent_stage_longer_than_the_lease_keeps_the_job
test_keepalive_repeats_progress_rather_than_inventing_it
test_a_lost_lease_stops_the_adapter_and_reports_nothing
test_an_overrunning_adapter_is_stopped_and_the_job_fails
test_provider_detail_never_reaches_the_customer_message
test_an_adapter_crash_still_fails_the_job_cleanly
test_an_unreachable_api_leaves_the_job_to_the_reaper
test_a_dropped_progress_update_does_not_discard_a_healthy_job
```

**Cause 2 — 1 failure in `tests/test_music_video.py`.**

```
tests\test_music_video.py:114: in test_a_track_longer_than_one_pass_becomes_several_scenes
    assert len(invocations(log)) == 5
E   AssertionError: assert 4 == 5
```

The test's own comment ([test_music_video.py:109-112](apps/worker/tests/test_music_video.py#L109-L112)) states the assumption it rests on:

> Five, not four: the MP3 probes a little over 4.0s because the encoder pads, so a 1s ceiling needs five windows.

The test generates a 4.0-second MP3 with the local ffmpeg and expects the encoder's padding to push the probed duration above 4.0 s, yielding `ceil(4.03 / 1.0) = 5` windows. On the audit machine the encoder produced a file probing at or below 4.0 s, so `plan_segments` returned 4 windows. **This is a property of the local ffmpeg/LAME build, not of the segmentation code** — the second assertion in the same test, `all(s.duration_seconds > 0.1 for s in plan_segments(4.03, max_segment_seconds=1.0))`, uses a literal `4.03` and passes.

**Whether these are pre-existing:** they were not introduced by this audit — no repository file was modified, and `git status` is clean at `3bd8016` both before and after. Whether they pass on the project's normal CI or developer machines is **UNKNOWN** — this audit ran the suite once, on one machine, and has no historical run to compare against. See Appendix B.
---

## Appendix A — Full config file dumps (verbatim)

The six workflow definitions are the complete public+private configuration surface. Their `execution:` blocks are reproduced in full, comments included, because the comments carry measurement provenance the comparison phase needs. Full files are in [audit/appendix-a-workflow-yaml.md](appendix-a-workflow-yaml.md).

### A.1 `settings.ltx_*` and the subprocess seams, verbatim

From [apps/worker/worker/core/config.py](apps/worker/worker/core/config.py):

```python
    # ── LTX runtime (M2, GPU nodes only) ─────────────────────────────────

    ltx_repo_dir: Path = Path("/workspace/ltx2-benchmark")
    """
    Where the LTX repository and its `uv` environment live on a GPU node. The
    adapter shells out to `uv run` with this as the working directory, so the
    model's Python environment stays completely separate from the worker's —
    the worker itself never imports torch.
    """

    ltx_model_dir: Path | None = None
    """Model weights root. Defaults to `<ltx_repo_dir>/models/ltx-2.5`."""

    person_matte_command: str = ""
    person_anchor_command: str = ""
    director_planner_command: str = ""
    director_gemma_dir: Path | None = None
    director_planner_timeout_seconds: float = 900.0
    director_vision_enabled: bool = False
    director_vision_command: str = ""
    director_vision_timeout_seconds: float = 300.0

    ltx_quantization: str = "nvfp4-prequant"
    """
    NVFP4 is the only mode that fits the client's RTX 5090: the BF16
    transformer alone is ~40 GB against 32 GB of VRAM. Verified by benchmark
    on 2026-08-12 (docs/internal/ltx-2.5-licensing-review.md is the licensing
    side; the VRAM ceiling is an engineering fact independent of it).
    """

    ltx_max_seconds: int = 60
    """
    Operational brake on single-pass length, not the real ceiling.

    The real ceiling is per-grid and lives in `adapters/ltx._GRID_CEILINGS`,
    because the VAE fails on particular shapes rather than above a size: on the
    RTX PRO 6000, 1024x576 sustains 60s while 896x512 — fewer pixels — does not.
    A single global number can only ever encode the worst shape, which is how
    every 60s render came to be six passes with five seams.

    This value is the emergency lever: lowering it via the environment pulls
    every shape down immediately, with no deploy. That is exactly what
    contained the 14 Aug incident, so it stays in the clamp chain.

    60 because that is the longest length the product offers and every current
    grid was measured at it (16 Aug 2026, after NATTEN replaced the failing
    Triton fallback kernel). Raising it above 60 does nothing on its own — a
    grid still cannot exceed its measured entry.
    """

    ltx_frame_rate: int = 24
    """LTX-2.5's native rate; num_frames = seconds x this."""

    ltx_max_source_seconds: float = 330.0
    ltx_max_extend_source_seconds: float = 1800.0
```

Resolver properties ([config.py:598-650](apps/worker/worker/core/config.py#L598-L650)):

```python
    @property
    def workspace_root(self) -> Path:
        return self.workspace_dir or Path(tempfile.gettempdir()) / "zolexai-worker"

    @property
    def ltx_models_root(self) -> Path:
        return self.ltx_model_dir or self.ltx_repo_dir / "models" / "ltx-2.5"

    @property
    def person_matte_argv(self) -> list[str]:
        if self.person_matte_command:
            return shlex.split(self.person_matte_command)
        script = Path(__file__).resolve().parents[2] / "scripts" / "person_matte.py"
        return ["uv", "run", "python", str(script)]

    @property
    def person_anchor_argv(self) -> list[str]:
        if self.person_anchor_command:
            return shlex.split(self.person_anchor_command)
        script = Path(__file__).resolve().parents[2] / "scripts" / "person_anchor.py"
        return ["uv", "run", "python", str(script)]

    @property
    def director_gemma_root(self) -> Path:
        return self.director_gemma_dir or self.ltx_repo_dir / "models" / "gemma-4-e2b-it"

    @property
    def director_planner_argv(self) -> list[str]:
        if self.director_planner_command:
            return shlex.split(self.director_planner_command)
        script = Path(__file__).resolve().parents[2] / "scripts" / "director_plan.py"
        return ["uv", "run", "python", str(script)]

    @property
    def director_vision_argv(self) -> list[str]:
        if self.director_vision_command:
            return shlex.split(self.director_vision_command)
        script = Path(__file__).resolve().parents[2] / "scripts" / "director_image_facts.py"
        return ["uv", "run", "python", str(script)]
```

### A.2 `.env.example` — the GPU-node block, verbatim

[.env.example:71-114](.env.example#L71-L114):

```
# ── Generation runtimes (GPU nodes only) ─────────────────────────────────
# Unset everywhere else. A node without these runs the mock runtime and can
# only claim mock-routed workflows, so leaving them blank is safe rather than
# broken.
#
# Video (LTX-2.5). RUNTIMES is what lets a node claim GPU-routed work at all;
# the rest have working defaults in apps/worker/worker/core/config.py.
# RUNTIMES=ltx
# LTX_REPO_DIR=/workspace/ltx2-benchmark
# LTX_MAX_SECONDS=30      # measured single-pass ceiling — longer is segmented
#
# Music (ACE-Step 1.5 XL). Unlike LTX, the model is NOT launched per job — it
# is a long-lived service holding ~24 GB that answers requests in seconds, so
# the worker connects to it the way it would to a database. Start the service
# separately, then point the worker at it.
#
# NOTE: music (~24 GB) and video (~28 GB peak) cannot both be resident on a
# 32 GB card. One GPU serves one workload at a time; running both concurrently
# needs a second GPU.
# RUNTIMES=music              # or "ltx,music" once the queue serialises them
# MUSIC_PROVIDER=acestep
# ACESTEP_BASE_URL=http://127.0.0.1:8001
# ACESTEP_API_KEY=            # only if the service was started with one
# ACESTEP_MAX_SECONDS=600     # measured single-generation ceiling
# MUSIC_SECONDS_PER_LINE=18   # lyric density; see worker/music/lyrics.py
```

**Every LTX line in this file is commented out.** No `.env` variable in the local checkout sets any `LTX_*` value (§1.7).

### A.3 `ExecutionSpec` — the schema every `execution` block is parsed by

[apps/api/app/schemas/workflow.py:128-165](apps/api/app/schemas/workflow.py#L128-L165), verbatim:

```python
class ExecutionSpec(BaseModel):
    """PRIVATE. Never projected into a public response.

    M2 fills this with runner, graph file, model reference and hardware
    requirements. Kept permissive (`extra="allow"`) so adding a field then does
    not require touching this class.
    """

    model_config = ConfigDict(extra="allow")

    runtime: str = "mock"

    output_content_type: str | None = None
    """..."""
    output_kind: Literal["video", "image", "audio"] | None = None
    """Asset kind for what the runtime produces. Pairs with the field above."""

    timeout_seconds: int | None = None
    """
    Wall-clock ceiling for one render of this workflow, enforced by the worker
    (`worker.jobs.runner._execute`). None means the worker's own default.

    Declared here — rather than left to `extra="allow"` — because the API reads
    it too: the output upload URL is signed at claim time and used only after the
    render finishes, so its validity is derived from this number. A workflow that
    raises its render ceiling must not silently outlive its own upload window.
    """
```

**Four typed fields; every other `execution` key in every YAML is untyped `extra="allow"` passthrough.** That includes all fourteen `v2v_*` keys, `max_segment_seconds`, `prompt_structuring`, `align_cuts_to_audio`, `generation_engine`, `audio_conditioning`, `audio_pass_seconds`, `guided_pass_seconds`, `transform_pass_seconds`, `inference_steps`, `guidance_scale`, `stg_scale`, `a2v_guidance_scale`, `negative_prompt`, `enhance_prompt`, and `i2v_reference_strength`. **None of them is validated, type-checked, or spell-checked anywhere.** A misspelled key is silently ignored and the module default applies (`execution_float` / `execution_int` swallow `KeyError`).

### A.4 The complete `GenerationParameters` surface

[apps/api/app/schemas/generation.py:16-56](apps/api/app/schemas/generation.py#L16-L56) — reproduced in §1.4. Ten fields, `extra="forbid"`. **No field controls any model setting except `duration`, `aspect_ratio`, `seed`, `prompt_mode` and `dialogue_language`.**

---

## Appendix B — Coverage gaps: what could NOT be determined, and why

**This appendix is not empty. Nine items.**

**B.1 — The LTX pipelines' own default values for guidance and steps.**
`--video-cfg-guidance-scale`, `--video-stg-guidance-scale`, `--a2v-guidance-scale` and `--num-inference-steps` are **not emitted** by this repository under any committed configuration, so the pipeline's own defaults apply. Those defaults live in the LTX repository at `settings.ltx_repo_dir` (`/workspace/ltx2-benchmark`), which is **not part of this checkout**. The repository records them only in prose:
- `3.0 / 1.0 / 3.0` for cfg / stg / a2v, at [ltx.py:2746](apps/worker/worker/adapters/ltx.py#L2746)
- `30` stage-1 steps, at [music-video.yaml:131](workflow-definitions/music-video.yaml#L131)
- "CFG (official default 3.0)" at [ltx.py:968](apps/worker/worker/adapters/ltx.py#L968)

**These are comments, not code, and were not verified against the pipeline source.** Recorded as `UNKNOWN — defined in the external LTX repository at settings.ltx_repo_dir, which is not in this checkout; the repo's own comments claim 3.0 / 1.0 / 3.0 / 30`.

**B.2 — Which LTX pipelines version / commit is installed.**
There is no submodule, no lockfile, no pinned commit, no version constant, no `requirements` entry, and no `.gitmodules` referencing the LTX repository. `ltx_repo_dir` is a bare path. Recorded as `UNKNOWN — no version pin of any kind for the LTX runtime code exists in this repository`. The *model* version is knowable from the checkpoint filenames (§0.5); the *code* version is not.

**B.3 — Whether the flag names this adapter emits match the installed pipelines' CLI.**
`_command` emits 26 distinct flags. Nothing in this repository validates them against the pipeline's argument parser — the only check is a non-zero exit code at runtime, surfaced as an output tail. The `_MARKERS` strings (§3.8) are the sole indirect evidence of the pipeline's actual behaviour, and they are matched loosely (`in line`). Recorded as `UNKNOWN — flag compatibility is only verifiable by running the pipeline, which this audit does not do`.

**B.4 — The contents of `scripts/person_matte.py`, `person_anchor.py`, `director_plan.py`, `director_image_facts.py`.**
These were inventoried (§1.2) and their **CLI contracts** were extracted from the call sites in `masks.py`, `provider.py` and `vision.py` — argument names, values and cwd are all recorded. Their internals (which matting model, which weights path, which inference settings) were **not** extracted, because they are model-side code that runs in the LTX environment and is not part of the LTX invocation this audit was asked to photograph. Recorded as `PARTIAL — CLI contracts extracted; script internals not read`.

**B.5 — A real DirectorPlan.**
Producing one requires either a Cerebras API call with a credential or the local `gemma-4-e2b-it` checkpoint on a GPU node. Both are excluded by the audit's hard rules. §12's plan is marked `RECONSTRUCTED`; the compiler output derived from it is genuine. Recorded as `UNKNOWN — a real planner output could not be obtained without a model call`.

**B.6 — Real `reference_person_facts` / `source_image_facts` output.**
Same reason as B.5. The prompts that produce them are pasted verbatim (§11.4, [vision.py:42-89](apps/worker/worker/director/vision.py#L42-L89)); the outputs are shown as placeholders in §5.5 and §12.7.

**B.7 — The API test suite was not run.**
`apps/api/tests/` requires a live PostgreSQL and Redis. Starting infrastructure is outside a read-only audit. Recorded as `NOT RUN`. The two API tests that reference LTX were read statically and are cited in §1.5 and §0.6.

**B.8 — Whether the 15 worker-test failures reproduce elsewhere.**
Both causes were identified as properties of the audit machine (§13.3): 1.7 GB free disk against a 2048 MB floor, and a local ffmpeg/LAME build whose MP3 padding differs from the test's assumption. Whether the suite is green on the project's normal machines is **UNKNOWN** — this audit has one run, on one machine, with no historical baseline to compare against.

**B.9 — What the deployed GPU node's YAML actually contains.**
The committed YAML says `runtime: mock` for all six workflows (§0.6, §4.6), while several comments in the same files describe features as "ENABLED" and "GPU-verified". The audit's hard rules forbid SSH to production, so the deployed configuration could not be read. Recorded as `UNKNOWN — production configuration not inspected; everything in this report describes commit 3bd8016 as committed`.

**Everything else in Sections 1–13 was determined from the repository.** Every table row carries a `file:line`, and no value in this report was inferred, interpolated, or assumed.

---

## Appendix C — Observations

**Recorded as facts. No conclusions are drawn, and nothing here is called a defect.** The comparison phase decides what, if anything, any of these means.

**C.1 — The committed `runtime` is `mock` on all six workflows, while several comments in the same files describe LTX features as enabled and GPU-verified.** Cited in full at §4.6.

**C.2 — `output_content_type: image/png` is on all six workflows, and the upload is signed and PUT with that type, while `LtxAdapter` reports `content_type="video/mp4"` at completion.** The upload URL is signed for `execution.output_content_type` ([internal.py:225](apps/api/app/api/v1/internal.py#L225)); the worker PUTs with `claim["output_content_type"]` ([runner.py:165](apps/worker/worker/jobs/runner.py#L165)); the asset row is created with `result.content_type` ([runner.py:174](apps/worker/worker/jobs/runner.py#L174) → [generation.py:527-529](apps/api/app/services/generation.py#L527-L529)). These agree for the mock adapter (which returns `image/png`) and would differ for the LTX adapter (which returns `video/mp4`).

**C.3 — `.env.example` suggests `LTX_MAX_SECONDS=30`; the code default is `60`.** [.env.example:80](.env.example#L80) vs [config.py:279](apps/worker/worker/core/config.py#L279).

**C.4 — `video-to-video.yaml` documents `v2v_identity_refresh_strength` as "default 0.2"; the code constant is `0.0`.** [video-to-video.yaml:205-208](workflow-definitions/video-to-video.yaml#L205-L208) says *"reference re-shown per later pass (default 0.2 — 0.35 flashed the reference PHOTO into a customer video …)"*, while [ltx.py:610](apps/worker/worker/adapters/ltx.py#L610) is `_V2V_IDENTITY_REFRESH_STRENGTH = 0.0` with a docstring beginning *"OFF, on the strength of two production failures in one evening"*.

**C.5 — `.env.example` suggests `MUSIC_SECONDS_PER_LINE=18`; the code default is `6.0`.** [.env.example:95](.env.example#L95) vs [config.py:394](apps/worker/worker/core/config.py#L394), whose docstring records a re-measurement on 2026-08-21 replacing a previous `13.0`.

**C.6 — `settings.max_segment_seconds` (default `10`) is documented as the general segmentation default but is read only by the harness adapter.** [config.py:174-179](apps/worker/worker/core/config.py#L174-L179) vs [harness.py:92-93](apps/worker/worker/adapters/harness.py#L92-L93). The LTX adapter's fallback is `settings.ltx_max_seconds` (`60`).

**C.7 — `structure_prompt`'s output is re-parsed by `plan_section_prompts` on multi-section jobs, and the last CONTINUITY bullet lands in the `NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY` block.** Reproduced in every multi-section prompt in §5.9, §5.7, §5.8 and §12.2. The `_ALREADY_STRUCTURED` bail-out in `structure_prompt` matches `^\s*(persistent|section \d+|continuity)\s*:` ([enhance.py:70-72](apps/worker/worker/longform/enhance.py#L70-L72)); the block `structure_prompt` emits is `CONTINUITY (fixed for the entire video):`, whose parenthetical means `_PERSISTENT_LINE` in `prompts.py` (`^\s*(?:persistent|continuity|subject(?:s)?|scene|constraints?)\s*:\s*(.+)$`) does not match it, so its bullets are classified by the fallback rules instead.

**C.8 — `_COUNT_PATTERN` matched a timestamp fragment in a realistic music-video prompt.** Input line `0:20-0:40 close-up on their face` / `1:00-1:20 the crowd floods in`; output rule `Exactly 20 crowd floods appear, and they remain the only floods on screen for the entire video.` Reproduced in §12.6. The pattern is `\b(?:exactly\s+)?(\d{1,2}|one|two|…)\s+((?:[a-z]+\s+){0,2}?[a-z]+s)\b` ([enhance.py:54-58](apps/worker/worker/longform/enhance.py#L54-L58)); `_NOT_NOUNS` ([enhance.py:62-65](apps/worker/worker/longform/enhance.py#L62-L65)) does not contain `floods`.

**C.9 — `_action_clause` produced two subject/verb disagreements in the §12 compiler output.** `"They run together toward the loading bay"` with speaker `detective` → `the detective run together toward the loading bay`; `"Flame climbs a stack of pallets behind them"` with speaker `woman` → `the frightened woman flame climbs a stack of pallets behind them`. The function strips a leading article/pronoun and rebases the action onto the speaker ([compiler.py:486-508](apps/worker/worker/director/compiler.py#L486-L508)); its docstring notes *"An action about someone ELSE is left alone and precedes the line as its own sentence fragment"*, and the events above name a subject that is not the speaker's role word.

**C.10 — At the shipped 30-second pass ceiling, a 60-second I2V job's second section is byte-identical to a 60-second T2V job's second section.** `_identity_anchor` returns `None` because 720 ∉ `_TWO_IMAGE_SAFE_FRAMES = {120, 240, 360}` ([ltx.py:2455-2465](apps/worker/worker/adapters/ltx.py#L2455-L2465)). Verified by dry-run, §5.10. A `identity_anchor_skipped` log line is emitted.

**C.11 — The `--video-conditioning` control clip is built at the un-doubled grid while `_IC_LORA` asks the model for twice that grid.** Control: `width=grid[0], height=grid[1]` ([ltx.py:1649-1650](apps/worker/worker/adapters/ltx.py#L1649-L1650)) → `1024×576`. Render: `width, height = width * 2, height * 2` ([ltx.py:2694](apps/worker/worker/adapters/ltx.py#L2694)) → `--width 2048 --height 1152`. The same applies to the matte and the attention mask.

**C.12 — Video-to-Video sends the same `--prompt` to every section.** `prompt_for_step` is not supplied by `_run_restyle` or `_run_transform` ([ltx.py:1468](apps/worker/worker/adapters/ltx.py#L1468), [ltx.py:1736-1742](apps/worker/worker/adapters/ltx.py#L1736-L1742)), so `_command` falls back to `job.prompt` ([ltx.py:2723](apps/worker/worker/adapters/ltx.py#L2723)). A 60-second source on the transform engine is 8 sections all carrying identical prompt text. `video-to-video.yaml` also does not declare `prompt_structuring`.

**C.13 — The transform engine's per-pass ceiling of 8.0 s produces one section every 7.5 s.** A 5-minute source (`ltx_max_source_seconds = 330.0`) plans 44 sections with 43 seams, each rendering 193 frames to deliver 180. Computed through `plan_chain_segments(330.0, 8.0, None)`.

**C.14 — `execution` keys are entirely unvalidated.** `ExecutionSpec` is `extra="allow"` with four typed fields; `execution_int`/`execution_float` swallow `KeyError`, `TypeError` and `ValueError` ([base.py:190-208](apps/worker/worker/adapters/base.py#L190-L208)). A misspelled key (`v2v_structure_strenght`) or a malformed value (`"0.45s"`) silently yields the module default with no log line.

**C.15 — `generation_engine` is compared with a bare `==` while `v2v_engine` is `.strip()`ed.** [ltx.py:1090](apps/worker/worker/adapters/ltx.py#L1090) `job.execution.get("generation_engine") == "guided"` vs [ltx.py:1382](apps/worker/worker/adapters/ltx.py#L1382) `str(job.execution.get("v2v_engine") or "").strip() == "transform"`.

**C.16 — `DirectorEvent.camera` is the only structured camera field in the repository, is free text, and is never validated.** [plan.py:419](apps/worker/worker/director/plan.py#L419) is `camera = str(entry.get("camera") or "").strip()` and nothing else. `DirectorPlan.tone` is likewise parsed and never used.

**C.17 — No camera angle, lens, or transition concept exists at any layer.** Searched across worker, API, web, packages and YAML for the full spec term list; §10.1 records the result per concept.

**C.18 — `Segment.overlap_seconds` is implemented end to end but no caller ever supplies a non-zero value.** [segments.py:40-61](apps/worker/worker/media/segments.py#L40-L61), [segments.py:92-94](apps/worker/worker/media/segments.py#L92-L94). Section overlap is therefore 0.0 on every workflow.

**C.19 — `AudioMode.GENERATED_MASTER_AUDIO` is declared but never set by `adapters/ltx.py`.** [audio.py:41](apps/worker/worker/media/audio.py#L41).

**C.20 — `_MODEL_FILES["transformer_bf16"]` is unreachable through `_transformer_file()` while `ltx_quantization` contains `"nvfp4"`,** but is reachable through `_IC_LORA.transformer_key`. [ltx.py:2640](apps/worker/worker/adapters/ltx.py#L2640), [ltx.py:896](apps/worker/worker/adapters/ltx.py#L896).

**C.21 — The Union Control LoRA filename carries version `ltx-2.3` while every other checkpoint carries `ltx-2.5`.** [ltx.py:196](apps/worker/worker/adapters/ltx.py#L196). The source comment states this is deliberate and GPU-verified on 17 Aug 2026.

**C.22 — `_A2VID.measured_landings` includes 481, and the constant's own comment records that 481 failed one benchmark cell out of four with the same cuBLAS error.** [ltx.py:954-963](apps/worker/worker/adapters/ltx.py#L954-L963), quoted at §7.2.

**C.23 — `_GRID_CEILINGS` carries `(896, 512): 30.0` with the inline comment `# 60s FAILS: CUBLAS_STATUS_INTERNAL_ERROR`, and this grid is reachable only through `grid_for_source`,** i.e. only on v2v and extend with a source whose aspect resolves to it. [ltx.py:234](apps/worker/worker/adapters/ltx.py#L234).

**C.24 — Progress marker matching has no failure path.** If the installed pipeline's log lines differ from `_MARKERS`, `match_marker` returns `None` for every line and the progress bar never advances past the band's low bound; no warning is emitted. [ltx.py:2961-2963](apps/worker/worker/adapters/ltx.py#L2961-L2963).

**C.25 — `_run_generation` verifies `expect_audio=True` on every section and on the final file, so a T2V/I2V pass that produced no audio stream fails the job.** [ltx.py:1158](apps/worker/worker/adapters/ltx.py#L1158), [ltx.py:2616-2633](apps/worker/worker/adapters/ltx.py#L2616-L2633), [ltx.py:1174-1176](apps/worker/worker/adapters/ltx.py#L1174-L1176). V2V and music-video do not (`require_audio` defaults to `False`).

**C.26 — `parameters.motion_strength` and `parameters.prompt_adherence` have non-`None` defaults (`60` and `75`) and are stored on every job, but no worker code reads them.** [generation.py:33-34](apps/api/app/schemas/generation.py#L33-L34); the YAMLs declare `motion_strength: false` / `prompt_adherence: false` so the UI hides them, and [video-to-video.yaml:57-59](workflow-definitions/video-to-video.yaml#L57-L59) records that they "did reach the job, but the distilled adapter never read them".

**C.27 — Two workflows offer a `"60s"` duration whose single-pass ceiling differs by a factor of two.** `text-to-video` and `image-to-video` set `max_segment_seconds: 30` → 2 sections; `extend-video` sets none → 1 section of 60.0 s at 1441 frames. Both offer `"60s"` in `supported_durations`.

**C.28 — The audit machine had 1.7 GB free disk against `min_free_disk_mb = 2048`, which is what failed 14 of the 15 tests.** Recorded here because it is a property of the audit environment that a reader comparing test results needs to know. §13.3.

---

*End of report.*
