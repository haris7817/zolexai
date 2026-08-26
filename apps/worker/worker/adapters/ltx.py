"""LTX-2.5 runtime — real GPU generation for every video workflow.

## How it runs

The worker process never imports torch. It shells out to the LTX repository's
own `uv` environment (`settings.ltx_repo_dir`) exactly the way the benchmark
did, which keeps CUDA, model weights and their 40-GB dependency tree entirely
outside this codebase. The subprocess is supervised the same way the media
tools are: killed on cancellation, on timeout, and on any exception path, so a
dead job can never leave a render burning VRAM.

## The measurements this is built on

RTX 5090, 2026-08-12 — facts, not guesses:

  * NVFP4 works end to end; BF16's transformer alone is ~40 GB against 32 GB
    of VRAM and cannot load.
  * A 30s pass completes; a 60s pass hard-OOMs at 29.6/31.4 GiB mid-denoise.
    **Nothing here ever asks the GPU for more than one pass can survive** —
    longer durations become several passes, chained, and the ceiling is read
    from `settings.ltx_max_seconds` rather than written down anywhere.
  * The `distilled` entry point cannot emit audio-only output (it tries to
    attach an H.264 stream to an MP3 container and dies), so music generation
    is a different runtime entirely — see `adapters/music.py`.
  * Conditioning uses the pipeline's `--image PATH FRAME_IDX STRENGTH` input.
    A still pinned at frame 0 at full strength becomes the first frame; the
    same argument at other indices and lower strengths is how a restyle keeps
    the source's composition.

## One chain, five workflows

`worker.longform.render_chain` is the whole long-form mechanism, and every
workflow is a way of choosing what conditions its passes and what happens to
the parts afterwards:

  | workflow       | total length      | conditioning per pass          | assembled as            |
  |----------------|-------------------|--------------------------------|-------------------------|
  | text-to-video  | requested         | previous pass's final frame    | parts                   |
  | image-to-video | requested         | the still, then final frames   | parts                   |
  | extend-video   | requested         | source's final frame, then …   | source + parts          |
  | video-to-video | the SOURCE's own  | source keyframes + continuity  | parts + source audio    |
  | music-video    | the SONG's own    | previous pass's final frame    | parts + the whole song  |

Video-to-video and music video take their length from the uploaded file, never
from a request field — that is the client's automatic-duration requirement, and
it is enforced by reading the probe rather than by trusting a parameter.

## Three entry points, two of them opt-in

The table above describes the DEFAULT tier, `ltx_pipelines.distilled`, which is
the only one that has served production. Two others are reachable, each behind
a private `execution` key, because each is a different quality/cost trade rather
than a strictly better version of the default:

  * **`v2v_engine: transform`** — `ltx_pipelines.ic_lora` plus the Union Control
    IC-LoRA, conditioned on an EDGE MAP of the source rather than on stills of
    it. Stills carry the source's colour and light along with its geometry,
    which is why the default restyle preserves strongly and transforms weakly —
    the client's actual complaint. An edge map carries geometry alone, so the
    prompt owns the look outright.
  * **`audio_conditioning: true`** on music video — `ltx_pipelines.a2vid_two_stage`,
    which is handed the master track seeked to each pass's own moment in the
    song. The default never shows the model the music at all, so no amount of
    prompting can make a performer's mouth follow a vocal.

Both load a LoRA, and **a LoRA means quantization is dropped entirely and the
unquantized model is fitted with `--offload cpu` instead** — see `LtxPipeline`.
That is not a tuning preference; it is the difference between a render and a
crash, and it is what dissolved this project's supposed audio-tier resolution
ceiling on 17 Aug 2026.

## The prompt

The user's prompt reaches the model **verbatim**. It is passed as a single
argv element, so nothing quotes, escapes, truncates or reflows it, and this
module never rewrites, prefixes or "improves" it — a generation that does not
match what someone typed is a bad enough experience without the system having
silently typed something else. `tests/test_ltx.py` pins that byte-for-byte.

The distilled entry point exposes no guidance scale, step count or negative
prompt (its non-distilled sibling does), so there is no adherence dial to turn
from here. LTX's own prompt enhancer is the one available lever and it *does*
rewrite the prompt, so it is opt-in per workflow via `execution.enhance_prompt`
and off everywhere by default.

## Testability

Everything except the model itself is provable without a GPU: `_command()` is
pure, `_execute()` accepts any argv (tests substitute a stub script that writes
a real MP4), and progress parsing is a pure function over output lines. The
only thing the GPU-node run adds is the model.

Each refusal is `retriable=False` with the real reason in `internal_detail` — a
mis-routed job should fail once with a clear log line, not burn three attempts.
"""

from __future__ import annotations

import asyncio
import math
import os
import signal
import zlib
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from worker.adapters.base import (
    AdapterError,
    AdapterJob,
    AdapterResult,
    ProgressCallback,
    cancellable,
    parse_duration_seconds,
)
from worker.comfy import evict_comfy_vram
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.director import (
    DirectorFailure,
    compile_section_prompts,
    continuation_lineage,
    create_director_plan,
    reference_person_facts,
    wants_director,
)
from worker.longform import (
    GENERATE_FROM,
    GENERATE_TO,
    ChainStep,
    RenderStep,
    StageReporter,
    plan_chain_segments,
    plan_musical_boundaries,
    plan_section_prompts,
    render_chain,
    structure_prompt,
)
from worker.media import (
    BACKGROUND_ATTENTION,
    DEFAULT_EDGE_HIGH,
    DEFAULT_EDGE_LOW,
    AudioMode,
    FfmpegError,
    MediaInfo,
    OutputExpectation,
    audio_onsets,
    build_attention_mask,
    build_hybrid_control,
    build_identity_anchor,
    build_person_matte,
    concat_segments,
    duration_tolerance,
    extract_edge_control,
    extract_final_frame,
    extract_frames_at,
    extract_source_window,
    ffmpeg,
    mux_audio,
    normalize_clip,
    probe_media,
    verify_output,
)
from worker.media.vocals import vocal_activity, vocal_fraction

logger = get_logger(__name__)

#: Weight files relative to `settings.ltx_models_root`, exactly as the
#: benchmark laid them out. The transformer is chosen by quantization mode.
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

#: Weights only the non-default entry points need. Kept OUT of `_MODEL_FILES`
#: because that table is what `_require_models` insists on before any render:
#: a node that only ever serves text-to-video must not be told it is broken
#: because it lacks the audio tier's dev checkpoint. Each optional pipeline
#: checks for its own files, when it is actually selected.
_OPTIONAL_MODEL_FILES: dict[str, str] = {
    # The full (non-distilled) transformer. `a2vid_two_stage` is a GUIDED tier
    # and needs it, plus the distilled LoRA below to keep the step count sane.
    "transformer_dev": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "distilled_lora": "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
    # IC-LoRA Union Control. Trained for 2.3 and loads against the 2.5
    # distilled transformer — verified on the RTX PRO 6000 on 17 Aug 2026, a
    # 97-frame render in 36s, which is what unblocked structure-conditioned
    # video-to-video. It consumes canny / depth / pose control videos, NOT
    # ordinary RGB (see `worker.media.control`).
    "union_control_lora": "loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
}

#: LTX's two-stage pipeline requires dimensions divisible by 64 — 480x848 was
#: rejected outright in benchmarking.
#:
#: These grids were re-measured on the RTX PRO 6000 on 16 Aug 2026, after NATTEN
#: replaced the failing Triton fallback kernel (see `_GRID_CEILINGS`). The
#: previous set was inherited from the RTX 5090's 32 GB and was roughly a third
#: smaller than this card sustains.
_DIMENSIONS: dict[str, tuple[int, int]] = {
    "16:9": (1024, 576),
    "9:16": (576, 1024),
    "1:1": (768, 768),
    # 4:5 is the odd one out. Exact 4:5 on a /64 lattice is only 512x640, then
    # 768x960, then 1024x1280 — there is no intermediate, and 768x960 fails.
    "4:5": (512, 640),
}
_DEFAULT_DIMENSIONS = (1024, 576)

#: Single-pass ceilings **measured per grid**, in seconds. Not derived.
#:
#: The VAE fails on a *set of bad shapes*, not above a size threshold, and the
#: set follows no rule anyone has been able to predict. Measured on this card at
#: 60s: 1024x576 passes, 896x512 fails, 1152x640 fails, 768x960 fails. A larger
#: grid passing where a smaller one fails rules out every "budget" model, so
#: nothing here may be interpolated or extrapolated — a grid that is not in this
#: table has not been run, and gets `_UNMEASURED_CEILING`.
#:
#: Before NATTEN this whole table was effectively 10s, because one global value
#: had to satisfy the worst aspect ratio. That cost every 60s render five seams.
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

#: What an unmeasured grid is allowed. Deliberately pessimistic: 10s was the
#: last value proven safe across every shape the product offered, and a grid
#: absent from `_GRID_CEILINGS` is by definition one nobody has run.
_UNMEASURED_CEILING = 10.0

#: Frame counts the VAE decoder cannot decode, and where to land instead.
#:
#: The decoder dies in a cuBLAS batched GEMM (`CUBLAS_STATUS_INTERNAL_ERROR`
#: from `cublasGemmStridedBatchedEx`, all dims cast to int32) at specific
#: (grid, conditioned, frame-count) triples. The failing set follows no rule
#: anyone has produced: at 1024x576 unconditioned, 240 fails while 232, 248 and
#: 1440 pass; WITH a conditioning image the same 1440 fails and 240 passes.
#: Every entry below is a measurement from 16 Aug 2026 — nothing interpolated.
#:
#: Structure: bad band (inclusive) → first measured-safe landing at or above
#: the band. Rendering a few extra frames costs seconds; the output is trimmed
#: back to the exact requested duration afterwards, so the customer's video is
#: bit-for-bit the length they asked for. This removes the failure without a
#: ceiling, without chaining and without seams.
#:
#: Landings marked with their evidence:
#:   248  measured ✅ at 1024x576 and 576x1024
#:   736  measured ✅ at 1024x576, 576x1024 and 768x768
#:   1528 measured ✅ at 1024x576 conditioned; applied to the other grids
#:        because they fail identically at 1440 and are re-verified by the
#:        matrix before any deploy
#:
#: The two tables below share that structure: `_CONDITIONED_BANDS` for passes
#: carrying an `--image`, and `_BAD_FRAME_BANDS` for the per-grid unconditioned
#: measurements.


#: Conditioned bands, applied to EVERY grid — measured or not.
#:
#: On 16 Aug the lattice theory looked complete: 1381/1437/1440 FAIL,
#: 1289/1385/1441/1528 PASS, all failures non-conforming. The conditioned bands
#: were emptied on that theory — and within two hours seven customer
#: image-to-video jobs died, because the snap had turned the MATRIX-PROVEN 720
#: into 721, and 721-conditioned crashes. 720 passes and 721 fails, one frame
#: apart, the mirror image of 1440/1441. There is no rule. These bands map every
#: unmeasured conditioned lattice point to the next MEASURED-PASS count at or
#: above it:
#:
#:   measured PASS (conditioned): 81, 120, 121, 240, 360, 720 (matrix, i2v and
#:                                production cells), 1289, 1385, 1441, 1528
#:                                (probed 16 Aug)
#:   measured FAIL (conditioned): 361 (production, 3 v2v jobs), 721 (production,
#:                                7 jobs), 1381, 1437, 1440, 1464
#:
#: NOT keyed on the grid, unlike the unconditioned bands. Every conditioned
#: failure found so far reproduced on more than one shape — 721 on both 1024x576
#: and 576x1024, 1440 across the matrix — so the frame count is the property
#: that matters here. Keying them per-grid left an unlisted shape able to emit
#: 361 and 721 exactly: unreachable today only because an unmeasured grid takes
#: `_UNMEASURED_CEILING`, and a latent crash guarded by an unrelated constant is
#: how the last two of these were found.
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

#: Conditioned counts proven by the full matrix or by production jobs. These
#: pass through EXACTLY as requested — never snapped, never banded — because
#: every one of them is evidence, and 720→721 is how evidence got replaced by
#: a theory and broke image-to-video for a night.
#:
#: EVERY CELL IN THIS TABLE IS A SINGLE-IMAGE MEASUREMENT. A second
#: conditioning image moves the decoder to a different, mostly unmeasured
#: configuration — see `_TWO_IMAGE_SAFE_FRAMES`, learned from a production
#: crash the night 60s image-to-video first chained.
_MEASURED_SAFE_CONDITIONED = frozenset({120, 240, 360, 720, 1289, 1385, 1441, 1528})

#: Frame counts MEASURED to decode with TWO conditioning images — the seam
#: frame at 0 plus the identity reference mid-window, which is the shape of
#: every I2V chain pass after the first and of Director-lineage extensions.
#:
#: Measured 20 Aug 2026 at 1024x576 (probe: `frame_probe2.py`, the adapter's
#: own command builder), the night the first production 60s I2V chained:
#:
#:   PASS: 120 (production-shape test render, 19 Aug), 240, 360
#:   FAIL: 720 (the production crash, reproduced deterministically),
#:         736 (so the render-extra-and-trim dodge does NOT work here —
#:         this cell family fails by size, not by lattice point)
#:
#: A pass whose count is not in this set carries the seam frame ALONE and the
#: identity anchor is dropped, with a log line. That is the measured-good
#: single-image shape (720 single-image passes everywhere), and identity
#: across such seams is carried by the predecessor frame plus the compiled
#: captions — the exact configuration the 60s two-section validation ran
#: clean. Growing this set is a `frame_probe2.py` measurement, not an
#: opinion.
_TWO_IMAGE_SAFE_FRAMES = frozenset({120, 240, 360})

#: The model's native frame convention: counts of the form 8k+1.
#:
#: Stated in three places in the pipeline source — `retake.py` REJECTS other
#: counts outright ("must satisfy 8k+1 (e.g. 97, 193)"), the dubbing pipeline
#: "silently snaps to the nearest 8k+1", and the trainer's dataset loader
#: checks `num_frames % 8 != 1`. The entry point this adapter drives does none
#: of that: it accepts whatever it is given and handles the remainder on a path
#: where the decoder's batched GEMM casts its dimensions to int32 and dies.
#:
#: Which is the whole two-day bug. `round(seconds * 24)` produces 1440 for a
#: 60s pass and 1381 for one of music video's beat-aligned 57.54s passes;
#: neither is 8k+1, both crash, and 1441 and 1385 do not. Measured 2026-08-16.
_FRAME_LATTICE = 8


def conforming_frames(frames: int) -> int:
    """The smallest 8k+1 count at or above `frames`.

    Overshoot is at most 7 frames — under a third of a second — and the caller
    trims back to the exact requested duration, so this is invisible in the
    delivered video. Compare the alternative it replaced: a measured table of
    poisoned counts, and a band wide enough to cost 40% extra compute on every
    conditioned pass.
    """
    return max(1, frames + ((1 - frames) % _FRAME_LATTICE))


def safe_frame_count(
    dimensions: tuple[int, int], frames: int, *, conditioned: bool
) -> int:
    """The frame count actually sent to the pipeline for this shape.

    Measurement outranks theory here, in this exact order:

    1. A conditioned count that is MEASURED SAFE passes through untouched.
       This rule exists because its absence broke production: the lattice
       snap turned the matrix-proven 720 into 721, and 721-conditioned
       crashes. Evidence first, always.
    2. A conditioned count the lattice would carry PAST a measured-safe count
       lands on the measurement instead. Same principle as (1), one step
       further: reaching over proven ground to land on an unproven lattice
       point is how 361 happened.
    3. Snap to the model's native 8k+1 lattice (its sibling entry points all
       do; every unconditioned crash on record was a non-conforming count,
       and 737 is verified by a live production job).
    4. Apply the measured bands: conditioned lattice points that are not
       measured land on the next measured-pass count above them, and the
       unconditioned low-count bands from the matrix era still apply.

    The caller renders the substitute and trims back to the requested
    duration, so the delivered video is exactly the length asked for, in one
    pass, with no seam.
    """
    if conditioned:
        if frames in _MEASURED_SAFE_CONDITIONED:
            return frames
        # A measured-safe count between the request and the lattice point wins.
        # Without this, a source 24 milliseconds short of a whole frame —
        # 14.976s, which is 359 frames — snaps PAST the proven 360 to 361, and
        # 361 conditioned crashes the decoder. Three video-to-video jobs died on
        # exactly that on 16 Aug 2026, all three from one upload; the 15.018s
        # file next to it asked for 360, passed through, and worked.
        landing = min(
            (
                safe
                for safe in _MEASURED_SAFE_CONDITIONED
                if frames <= safe <= conforming_frames(frames)
            ),
            default=None,
        )
        if landing is not None:
            return landing
    frames = conforming_frames(frames)
    bands = _BAD_FRAME_BANDS[conditioned].get(dimensions)
    if bands is None:
        # An unlisted grid still gets the conditioned bands — those are keyed on
        # the count, not the shape (see `_CONDITIONED_BANDS`). The unconditioned
        # bands are genuinely per-grid measurements, so an unlisted shape gets
        # none of them and relies on the lattice.
        bands = _CONDITIONED_BANDS if conditioned else ()
    for lo, hi, landing in bands:
        if lo <= frames <= hi:
            # The landing is a MEASUREMENT and is used exactly. Snapping it
            # would replace evidence with theory — 1528 is measured-pass and
            # not on the lattice, and "1529 must be fine, it conforms" is
            # precisely the reasoning that produced 721.
            return landing
    return frames

#: The largest frame measured on this card (1024x576 == 768x768 == 589,824 px).
#: A source's own aspect may still synthesise a grid that is not in
#: `_GRID_CEILINGS` — a 4:3 upload has no measured grid, and forcing it to 16:9
#: would make the model keep the style and replace the subject, which is worse
#: than a shorter pass. Such a grid renders at its true aspect and takes
#: `_UNMEASURED_CEILING`, so it is chained rather than gambled on.
_PIXEL_BUDGET = 1024 * 576

#: Delivery is at the source's own resolution (a user's 1080p clip must not
#: come back as 512p), capped at full HD — beyond that the normalization
#: re-encode cost stops being worth invisible extra pixels.
_MAX_OUTPUT_LONG_SIDE = 1920
_MAX_OUTPUT_SHORT_SIDE = 1080

#: The audio tier's measured single-pass length, in seconds.
#:
#: Measured on the RTX PRO 6000, 17 Aug 2026, at 1024x576 — dev transformer +
#: distilled LoRA, unquantized, `--offload cpu`, image-conditioned:
#:
#:   | frames | video  | wall | wall per second of video |
#:   |--------|--------|------|--------------------------|
#:   | 193    | 8.04s  | 147s | 18.3x                    |
#:   | 241    | 10.04s | 141s | 14.0x                    |
#:   | 481    | 20.04s | 211s | 10.5x                    |
#:
#: Longer passes are CHEAPER per second, not dearer: each pass reloads a 22B
#: transformer from host RAM, and that fixed cost is most of a short pass. So
#: the ceiling is set at the longest length actually measured rather than at the
#: shortest one that works — a 3-minute song is 9 passes and ~32 minutes here,
#: against 22 passes and ~54 minutes at 8s.
#:
#: **Swept 21 Aug 2026 at 1024x576, and longer is not reliably available.**
#: Counts up to 1201 frames (50s) do decode — but 601 and 1081 both died with
#: OUT OF MEMORY while 721, 961 and 1201 in the same sweep passed. That is not
#: a shape property; it is a card running near its edge, with the music service
#: holding ~24 GB of it permanently and a prompt-only pass already measured
#: peaking at 95.2 GB of 95.6. A long pass here is a coin flip, and the coin is
#: flipped after several minutes of compute have already been spent.
#:
#: So the ceiling stays at the length with the most evidence behind it rather
#: than the longest one observed to work once. Raising it is a real lever
#: against per-pass model-load cost and it needs its own measurement — ideally
#: with the music service stopped, which is the change that would actually make
#: the headroom exist.
#:
#: Expressed as a landing rather than a round number, deliberately. A ceiling of
#: 20.0 asks for 480 frames, which conforms up to 481 anyway; stating 481's own
#: duration instead means the plan's nominal window IS a measured count, and the
#: pass count falls out one lower on real tracks. A 300.042s track goes from 16
#: passes to 15, and a 60.024s one from 4 to 3 — a whole model load saved on a
#: one-minute video, for an arithmetic change.
_AUDIO_LANDING_FRAMES = 481
_AUDIO_PASS_SECONDS = _AUDIO_LANDING_FRAMES / 24.0

#: One audio latent of headroom on every conditioning window. See
#: `_audio_window_seconds`.
_AUDIO_WINDOW_PAD_SECONDS = 0.04

#: The transform engine's measured single-pass length, in seconds.
#:
#: `ic_lora` decodes on its own path and the distilled tier's 60s grid ceilings
#: carry no evidence about it — the first production transform job proved that
#: at a cost: a single 15s portrait pass crashed the decoder at 360 AND 361
#: frames, two counts the distilled tables call safe or landable. What IS
#: measured on this path, 17 Aug 2026, unquantized + cpu offload + Union
#: Control: 97 frames (512x320), 193 frames landscape (1024x576, 62s), and 193
#: frames portrait (576x1024, 67s — probed the same evening the production job
#: died). 8.0s keeps every pass at or under the 193-frame cell on either
#: orientation. Raising this is a measurement, not an opinion;
#: `execution.transform_pass_seconds` is where the measurement lands.
_TRANSFORM_PASS_SECONDS = 8.0

#: The guided tier's measured single-pass length, in seconds.
#:
#: `ti2vid_two_stages` decodes on its own path, and neither the distilled
#: tier's grid ceilings nor the audio tier's 481-frame cell transfer to it —
#: proven the day this tier was added, 17 Aug 2026: 241 frames at 1024x576, a
#: count the audio tier renders happily at this grid, died in this pipeline's
#: decoder with an illegal memory access, while 121 passed clean. What IS
#: measured on this path (dev transformer + distilled LoRA, unquantized,
#: `--offload cpu`): 121 frames at 1024x576 — 146s, ~29x real time, 39.3 GB
#: peak alongside the resident music service. Raising this is a measurement,
#: not an opinion; `execution.guided_pass_seconds` is where the measurement
#: lands, together with a new entry in `_GUIDED.measured_landings`.
_GUIDED_PASS_SECONDS = 5.0

#: Video-to-video conditioning defaults. Every one is overridable per workflow
#: through the private `execution` block, because the right values are a
#: quality judgement made against real footage on a real GPU, and baking them
#: in would make that judgement a code change.
_V2V_KEYFRAME_SECONDS = 4.0
"""How much output one source still is asked to anchor.

This is a DENSITY, not a count, and that distinction is the whole fix. A fixed
count spreads itself across whatever the pass happens to be: three stills over
a 30-second pass leaves the model generating ten continuous seconds with
nothing tying it to the customer's footage, and the gap is where a restyle
stops being a restyle — subjects drift out of frame, then out of the video.
Reported by the client on a 30s clip ("after 20 secs it changes, then there is
no woman present"), and the arithmetic above is exactly that complaint.

Four seconds is chosen to sit under the interval at which drift became visible
in that footage, not from theory. Conditioning is `--image PATH IDX STRENGTH`
triples, so a denser leash costs nothing at generation time."""

_V2V_KEYFRAME_BOUNDS = (3, 16)
"""Floor and ceiling on stills per pass. The floor keeps short passes from
being anchored only at their ends; the ceiling stops a long pass from becoming
a slideshow of the original with the prompt doing nothing."""

_V2V_STRUCTURE_STRENGTH = 0.45
"""How hard those stills pull. At 1.0 the source frame IS the output frame and
the prompt does nothing; near 0 the prompt wins and the source is a suggestion.
This is the dial between "restyled" and "unrelated".

It moved with the density above and cannot be read apart from it. At three
stills per pass, 0.7 held the look and lost the subject. At one still every
four seconds — roughly 2.7x as many anchors on a 30-second clip — 0.7 held the
subject and lost the look: the first A/B against the client's own footage came
back with the requested oil painting nowhere in it. Total pull is what
changed, so the per-anchor figure came down to compensate.

0.45 is an ESTIMATE, in proportion to the density change and nothing more.
`scripts/v2v_sweep.sh` measures 0.7/0.55/0.4/0.25 against one clip and is what
should set this number; the lowest strength that still holds the subject is
the answer."""

_V2V_CONTINUITY_STRENGTH = 0.85
"""Frame 0 of every pass after the first, taken from the previous pass's last
frame. High, because this is the seam: the two sides of it must be the same
video."""

_V2V_REFERENCE_STRENGTH = 0.3
"""The optional reference image, first pass only, at frame 0. Low on purpose —
the contract the customer reads says it "guides the look", and a strength that
made it the opening frame would be replacing the source's intent with it.
Setting this to 0 in a workflow drops reference conditioning entirely."""

_V2V_IDENTITY_ANCHOR_STRENGTH = 1.0
"""The COMPOSITED anchor at frame 0 of the first pass, under
`v2v_reference_identity`. Full strength, and safe at full strength for the
same reason the reference engine ships it there: the anchor is built from
the footage's own opening frame (`scripts/person_anchor.py` — source person
removed, reference person composited into their box), so its composition IS
the shot and there is nothing for the anchor to hijack. The measured
alternative is what this replaced: a raw portrait anchored against a
full-body opening frame lands its face on forty pixels, and the model
invents the person — the first full-body production jobs, twice."""

_V2V_IDENTITY_RAW_ANCHOR_STRENGTH = 0.65
"""The FALLBACK strength when the composited anchor cannot be built (no
person visible at frame 0, matting unavailable) and the raw photo anchors
instead. Capped below the composited value because a raw photo at 1.0
replaces the shot, not the person — the composition-hijack failure the
brief warned about. 0.65 is the reference engine's own raw-anchor value."""

_V2V_IDENTITY_REFRESH_STRENGTH = 0.0
"""The reference image re-shown at an interior frame of later passes.

OFF, on the strength of two production failures in one evening, 19 Aug 2026,
on the same customer job at the same timestamp: the anchor for pass two
lands at requested_frames // 3 = 10.17s of that upload, and the output
snapped to the reference PHOTOGRAPH itself for a beat — a posed studio
portrait cut into a dance video — at 0.35, and again after the "safe" I2V
value of 0.2 was deployed. The borrowed mechanism does not transfer: in I2V
the reference IS the video's opening frame, so re-showing it reinforces a
composition the render already has, while here the photo's composition is
ALIEN to the footage and ANY strength that helps identity also invites the
shot.

What holds identity across passes instead, both measured on the same
footage: the continuity frame — which carries the REPLACED person forward,
not the source person — and the describer's caption re-stating the person in
words on every pass. The knob remains for footage where those two provably
drift, to be raised only with the 10.17s flash in mind."""

_V2V_IDENTITY_SUBJECT_ATTENTION = 0.5
"""How hard the edge map still steers the PERSON under identity replacement.

The mirror of person lock's `BACKGROUND_ATTENTION`, and the load-bearing half
of the mode: at 1.0 the canny edges re-impose the source person's facial
geometry — jawline, hairline, features — in every pass, and no reference
strength can win against a control signal that redraws the original face
outline 24 times a second. Lowered, the person's placement and pose still
track the footage while their appearance is freed for the reference to own.
Toward 0 the person stops following the source's motion, which defeats the
workflow. The scene around them stays at full strength — camera and layout
are not what this mode is asked to change."""

_V2V_CONTROL_STRENGTH = 1.0
"""How hard the edge map pulls, on the `transform` engine.

Full strength, unlike the restyle's 0.45 stills, and for the opposite reason.
A still at full strength would pin the source's LOOK; an edge map at full
strength pins only its GEOMETRY, and geometry is the entire thing this engine
is asked to preserve. Lowering it does not make the transformation stronger —
it makes the result stop tracking the footage."""

_V2V_LORA_STRENGTH = 1.0
"""Union Control adapter strength. The value the adapter was trained at and the
value the reference engine ships; below it the control signal is progressively
ignored rather than progressively softened."""


def grid_for_source(width: int | None, height: int | None) -> tuple[int, int]:
    """The /64 generation grid closest to the source's aspect, within budget.

    The I2V benchmark made this non-negotiable: conditioning survives only
    when the render's aspect matches the conditioning image's. Grids are the
    model's constraint (each side divisible by 64), the budget is the GPU's.
    Ties in aspect error go to the larger frame.
    """
    if not width or not height:
        return _DEFAULT_DIMENSIONS
    aspect = math.log(width / height)

    def error(grid: tuple[int, int]) -> float:
        return abs(math.log(grid[0] / grid[1]) - aspect)

    grids = [
        (w, h)
        for h in range(256, 1025, 64)
        for w in range(256, 1025, 64)
        if w * h <= _PIXEL_BUDGET
    ]
    # Within a small aspect error the crop is invisible and more pixels win —
    # otherwise 576x320 (1.80) would beat 1024x576 (1.78) for a 16:9 source on
    # a 1% aspect technicality while shrinking the frame.
    #
    # Grids reached this way may have no entry in `_GRID_CEILINGS`. That is
    # deliberate: matching the source's aspect matters more than a long pass,
    # and an unmeasured grid is chained at `_UNMEASURED_CEILING` rather than
    # run at a length nobody has proven for that shape.
    close = [grid for grid in grids if error(grid) <= 0.08]
    if close:
        return max(close, key=lambda grid: (grid[0] * grid[1], -error(grid)))
    return min(grids, key=error)


def output_dimensions(width: int | None, height: int | None) -> tuple[int, int]:
    """The stitched file's resolution: the source's own, capped, made even.

    Even dimensions because yuv420p subsamples chroma 2x2 — libx264 refuses
    odd sizes outright.
    """
    if not width or not height:
        return _DEFAULT_DIMENSIONS
    long_side, short_side = max(width, height), min(width, height)
    scale = min(1.0, _MAX_OUTPUT_LONG_SIDE / long_side, _MAX_OUTPUT_SHORT_SIDE / short_side)

    def even(value: float) -> int:
        return max(2, int(value * scale) // 2 * 2)

    return even(width), even(height)


@dataclass(frozen=True)
class ConditioningFrame:
    """One `--image PATH FRAME_IDX STRENGTH` triple.

    A list of these is the entire conditioning vocabulary of the distilled
    entry point, and every workflow's identity is which list it builds.
    """

    path: Path
    frame_index: int
    strength: float

    def as_args(self) -> list[str]:
        # `round` then `str` rather than a %g format: the pipeline's own
        # documentation writes full strength as "1.0", and "1" is a different
        # token to argument parsers that type-check positionally.
        return ["--image", str(self.path), str(self.frame_index), str(round(self.strength, 3))]


@dataclass(frozen=True)
class ControlConditioning:
    """One `--video-conditioning PATH STRENGTH` pair, for the IC-LoRA tier.

    The path is a CONTROL SIGNAL (see `worker.media.control`), not the source
    footage. Passing raw RGB here is accepted by the CLI and produces a weak
    style hint rather than structural control, which is the failure mode this
    type exists to make unmistakable at the call site.
    """

    path: Path
    strength: float

    def as_args(self) -> list[str]:
        return ["--video-conditioning", str(self.path), str(round(self.strength, 3))]


@dataclass(frozen=True)
class MaskConditioning:
    """One `--conditioning-attention-mask PATH STRENGTH` pair.

    A per-region weight on the control signal: white follows it fully, black
    ignores it. This is how a restyle is told to reproduce the subject from
    their own pixels while leaving the scene free — see `worker.media.masks`.
    """

    path: Path
    strength: float

    def as_args(self) -> list[str]:
        return [
            "--conditioning-attention-mask",
            str(self.path),
            str(round(self.strength, 3)),
        ]


@dataclass(frozen=True)
class AudioConditioning:
    """A window of a track the model generates *against*, not a soundtrack.

    `path` is the WHOLE master file and `start_seconds` selects this pass's
    window inside it. That is the entire reason the model can be given section
    seventeen of a song without the song being cut into seventeen files: the
    pipeline seeks, so the master is never sliced, re-encoded or re-timed, and
    the audio the model hears at 2:31 is bit-for-bit the audio the finished
    video plays at 2:31.

    Nothing here reaches the delivered soundtrack. The generated parts are
    assembled silent and the user's original file is muxed over the result
    once, at the end — see `_run_music_video`.
    """

    path: Path
    start_seconds: float
    max_duration_seconds: float

    def as_args(self) -> list[str]:
        return [
            "--audio-path", str(self.path),
            "--audio-start-time", f"{self.start_seconds:.3f}",
            "--audio-max-duration", f"{self.max_duration_seconds:.3f}",
        ]


@dataclass(frozen=True)
class LoraSpec:
    """One `--lora PATH STRENGTH` pair."""

    path: Path
    strength: float = 1.0

    def as_args(self) -> list[str]:
        return ["--lora", str(self.path), str(round(self.strength, 3))]


@dataclass(frozen=True)
class LtxPipeline:
    """An LTX entry point plus the weights and runtime rules it requires.

    Three entry points are reachable from this adapter and they differ in more
    than a module name — a wrong transformer or a stray `--quantization` is a
    crash, not a quality difference — so the combination is named once here
    rather than assembled at each call site.

    **The quantization rule is not a preference.** Whenever a LoRA is loaded,
    quantization is dropped entirely and the unquantized model is fitted with
    `--offload cpu` instead. LoRA + FP8/NVFP4 fusion reaches Triton kernels that
    do not exist for these shapes; the client's own reference engine forces
    quantization to none on exactly this condition, and every unexplained
    "resolution ceiling" this project chased in the audio tier turned out to be
    this clash rather than a limit of the card. Measured 17 Aug 2026: 1024x576
    audio-conditioned, which had never once completed quantized, renders clean.
    """

    module: str
    transformer_key: str | None
    """Weights key for `--transformer-path`. `None` means "whichever the
    configured quantization selects", which is the default tier's behaviour."""

    distilled_lora: bool = False
    """Pass `--distilled-lora`. The guided tiers need it to run a short sigma
    schedule against the full checkpoint."""

    quantize: bool = True
    offload_cpu: bool = False
    extra_weights: tuple[str, ...] = ()
    """Optional weights keys this pipeline cannot run without."""

    conforming_only: bool = False
    """Snap frame counts to the model's 8k+1 lattice and go no further.

    `safe_frame_count`'s measured-safe counts and bad-frame bands are
    measurements OF THE DISTILLED TIER'S DECODE PATH, one (grid, count, crash)
    cell at a time. The other entry points decode elsewhere — `ic_lora` decodes
    stage-1 latents at half the requested grid with reference tokens appended,
    `a2vid` decodes with an audio stream attached — and the tables carry no
    evidence about those paths in either direction.

    Applying them anyway broke production twice in one day, 17 Aug 2026, in
    opposite ways: the "measured-safe" 360 crashed ic_lora's decoder on a
    portrait source (the measurement was distilled-tier), and the (361,719,720)
    band would push an audio pass from its MEASURED 481 up to an unmeasured
    720. So the non-default tiers take the one rule the pipeline family states
    about itself — `retake` rejects non-8k+1 counts outright, `dubit` silently
    snaps to 8k+1 — and their pass CEILINGS stay at lengths actually measured
    on their own path, which is what keeps the counts small enough to have
    been measured at all."""

    measured_landings: tuple[int, ...] = ()
    """Frame counts PROVEN on this pipeline's own decode path, ascending.

    When non-empty, every pass lands on the smallest of these at or above its
    conforming count and is trimmed back after the render — so an unmeasured
    count can never reach the decoder at all. This is stricter than the
    lattice, deliberately: the chain's overlap arithmetic emits counts like 185
    that conform perfectly and have never been run, and the decoder's failure
    set is non-monotonic — 181 and 193 both passing says NOTHING about 185.
    A request above the largest landing falls back to the plain lattice (it can
    only happen when a workflow override raised the pass ceiling, which is
    itself a claim of new measurement)."""

    stage_1_only: bool = False
    """Render stage 1 only, at DOUBLE the requested grid.

    Stage 2 upscales in latent space by halving a latent grid, so it needs both
    latent dimensions to be even. The product's 16:9 grid is 1024x576, and
    576/64 = 9 is odd: the two-stage IC-LoRA path dies in a VAE `rearrange`
    ("can't divide axis of length 9 in chunks of 2"), measured 17 Aug 2026.
    Padding the grid to a multiple of 128 would change the customer's aspect
    ratio, which is worse than the problem.

    Stage 1 already renders at half the requested size, so asking for 2x and
    stopping there delivers the target grid exactly — with the control adapter
    active in the final-resolution output rather than in a half-size draft that
    stage 2 would then re-imagine. This is what the client's own reference
    engine does for the same reason. Measured: 1024x576, 193 frames, 62s."""


#: The default tier. Fastest, and the only one that has ever served production.
_DISTILLED = LtxPipeline(module="ltx_pipelines.distilled", transformer_key=None)

#: Structure-conditioned video-to-video. Distilled checkpoint (the pipeline
#: refuses anything else) plus an IC-LoRA, hence unquantized + CPU offload.
_IC_LORA = LtxPipeline(
    module="ltx_pipelines.ic_lora",
    transformer_key="transformer_bf16",
    quantize=False,
    offload_cpu=True,
    extra_weights=("union_control_lora",),
    stage_1_only=True,
    conforming_only=True,
    # 193 is the ONE count measured on this path at BOTH product orientations —
    # 1024x576 landscape (62s, twice) and 576x1024 portrait (67s, probed the
    # evening the first production job died there). Other passing cells exist
    # (97 at 512x320, 181 portrait) but each was run at one grid only, and
    # putting a count measured at one grid into a table consulted for another
    # is precisely the mistake this field exists to end. 361 portrait is a
    # measured FAIL — the production crash — and is unreachable from here.
    measured_landings=(193,),
)

#: Audio-conditioned generation: the model hears the track while it draws.
#: Guided tier — dev transformer + distilled LoRA — so unlike the default it
#: also has CFG and a negative prompt available.
_A2VID = LtxPipeline(
    module="ltx_pipelines.a2vid_two_stage",
    transformer_key="transformer_dev",
    distilled_lora=True,
    quantize=False,
    offload_cpu=True,
    extra_weights=("transformer_dev", "distilled_lora"),
    conforming_only=True,
    # This table is why the audio tier can be switched on at all.
    #
    # It was empty until 21 Aug 2026, on the assumption that conforming to the
    # 8k+1 lattice was enough for a path whose bad shapes nobody had looked
    # for. It is not. A real 60-second job planned 474/477/430/59 frames, and
    # its third section died in the video VAE
    # (`chunked/mlp.py`, CUBLAS_STATUS_INTERNAL_ERROR) — the same decoder
    # failure class the distilled tier's tables exist for, on a pipeline that
    # had no table.
    #
    # Swept at 1024x576, four denoising steps (the crash is after denoising and
    # depends on shape), music service resident:
    #
    #   PASS  65 · 121 · 193 · 241 · 385 · 433 · 481 · 505 · 577 · 721 · 961 · 1201
    #   FAIL  289 · 337 · 361 · 409 · 457 · 841   (CUBLAS — a shape property)
    #   FAIL  601 · 1081                          (out of memory — not one)
    #
    # Non-monotonic, as always: 241 and 385 and 433 decode while 289, 337, 361,
    # 409 and 457 do not. Four of the passing counts are listed rather than all
    # of them — every entry is a count this adapter promises the decoder can
    # take, so each one is a claim to keep verified, and these four cover the
    # range the planner actually asks for with little waste. 385 earns its
    # place on the two-pass case: a 30-second track plans 15-second windows,
    # 361 is a measured FAIL, and without 385 both passes would render twenty
    # seconds to deliver fifteen. **Nothing between 241 and 385 decodes at
    # all** — that gap is the model's, not a hole in the sweep.
    #
    # The large counts are left out for a different reason. 601 and 1081 failed
    # on MEMORY while 721, 961 and 1201 passed, so nothing up there is reliably
    # available while the music service holds a quarter of the card — see
    # `_AUDIO_PASS_SECONDS`.
    #
    # **This table is a screen, not a guarantee, and the difference matters.**
    # 481 decoded in the sweep and in three consecutive benchmark cells, then
    # failed in a fourth at the same count with the same cuBLAS error in the
    # same video-VAE MLP. A fixed configuration that fails intermittently is
    # not describing a shape; `CUBLAS_STATUS_INTERNAL_ERROR` is what a failed
    # cuBLAS workspace allocation looks like, and this card was measured at
    # 95.2 GB of 95.6 GB during an ordinary pass. So what the table buys is
    # staying away from the counts that fail REPRODUCIBLY and from the large
    # ones that need the most memory. The residual intermittency is a headroom
    # problem, and headroom is not something a frame count can fix.
    measured_landings=(121, 241, 385, _AUDIO_LANDING_FRAMES),
)

#: The quality tier for text- and image-to-video: the guided two-stage
#: pipeline, which is what "follows the prompt" means upstream — CFG (official
#: default 3.0), spatio-temporal guidance and a negative prompt are all active
#: here, and none of them exist on the distilled entry point. The client's own
#: reference engine measures adherence against exactly this pipeline family,
#: never against distilled. Dev transformer plus the distilled LoRA — the
#: pipeline requires both — so the LoRA/quantization rule applies: unquantized,
#: CPU offload. Measured 17 Aug 2026 at 1024x576, 121 frames: 146s against the
#: distilled tier's 34s for the same clip — ~4.3x, not the ~10x the step-count
#: arithmetic predicted, because the distilled LoRA keeps the short schedule
#: while the guiders run against the dev weights.
_GUIDED = LtxPipeline(
    module="ltx_pipelines.ti2vid_two_stages",
    transformer_key="transformer_dev",
    distilled_lora=True,
    quantize=False,
    offload_cpu=True,
    extra_weights=("transformer_dev", "distilled_lora"),
    conforming_only=True,
    # 121 is the ONE count measured on this pipeline's own decode path, and it
    # was measured at ALL FOUR product grids on 17 Aug 2026: 1024x576 (146s,
    # video AND audio streams verified in the output), 576x1024 (138s),
    # 768x768 (149s), 512x640 (145s). 241 at 1024x576 — safe on distilled,
    # measured on a2vid — is a reproduced FAIL here (illegal memory access in
    # the decoder, confirmed by an isolated retry), which is the whole
    # argument for per-pipeline landings restated by the newest pipeline on
    # the box.
    measured_landings=(121,),
)


#: Ordered milestones matched against the pipeline's log output. Matching is
#: forward-only (an index walks down this list), which is what makes the twice-
#: occurring "Running denoising loop" line map to two different steps. The
#: messages are customer copy: no model, provider or tensor vocabulary.
_MARKERS: list[tuple[str, int, str]] = [
    ("Building text encoder", 20, "Understanding your prompt…"),
    ("Running denoising loop", 40, "Generating your video…"),
    ("Building video encoder + spatial upsampler", 55, "Adding detail…"),
    ("Running denoising loop", 70, "Refining your video…"),
    ("Building video decoder", 80, "Rendering the final video…"),
    ("saved to", 85, "Almost done…"),
]

#: How long the output may be silent before we recheck cancellation. Denoising
#: prints nothing for stretches; without this poll a cancelled job would not
#: stop until the next log line happened to arrive.
_CANCEL_POLL_SECONDS = 2.0

#: Kept for diagnostics when the pipeline fails — the whole log would be huge.
_OUTPUT_TAIL_LINES = 40


def match_marker(line: str, start: int) -> int | None:
    """Index of the first milestone at or after `start` that `line` announces.

    Forward-only so repeated phrases advance rather than repeat, and so a
    skipped milestone (variant pipelines log different stages) cannot wedge
    the whole sequence.
    """
    for index in range(start, len(_MARKERS)):
        if _MARKERS[index][0] in line:
            return index
    return None


class LtxAdapter:
    name = "ltx"

    #: Every workflow whose output is video. Music is a different runtime; the
    #: benchmark proved this entry point cannot write an audio-only file.
    _SUPPORTED = frozenset(
        {"text-to-video", "image-to-video", "extend-video", "video-to-video", "music-video"}
    )

    def supports(self, workflow_id: str) -> bool:
        return workflow_id in self._SUPPORTED

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        reporter = StageReporter(on_progress)
        await reporter.preparing()
        self._require_models()

        # Lazy co-tenancy eviction (25 Aug 2026): H3's ComfyUI keeps its
        # ~52 GB warm between H3 jobs so they stop paying a 40-60 s reload;
        # the engine that actually needs the card clears it on the way in.
        # Milliseconds when nothing is loaded or no ComfyUI exists.
        await evict_comfy_vram()

        # Deterministic prompt structuring, per workflow via the private
        # execution block. The user's text survives verbatim as the first
        # block of the structured prompt — `structure_prompt` appends derived
        # continuity rules, it never rewrites — and the stored job still
        # carries exactly what the user typed, because this happens here in
        # the worker rather than anywhere the prompt is persisted.
        #
        # A Director-mode job skips it: there the user's text is an IDEA, not
        # a prompt — the director compiler owns the entire prompt text, and
        # stacking a second structure on top of it would be the contradiction
        # `structure_prompt`'s own bail-out rule exists to avoid.
        if job.execution.get("prompt_structuring") and not wants_director(job):
            # `prompt_structuring_v2` opts a workflow into the revised
            # continuity block (split bullets, static-camera and departure
            # awareness, idempotent header) and the matching section-planner
            # rules. Off, both layers are byte-identical to what has always
            # shipped — the flag exists so the change can be A/B'd on the GPU
            # before it becomes anyone's default.
            job = replace(
                job,
                prompt=structure_prompt(
                    job.prompt, v2=bool(job.execution.get("prompt_structuring_v2"))
                ),
            )

        # Dispatch on the WORKFLOW, never on which inputs happen to be present:
        # a mis-routed job must fail loudly rather than being quietly treated as
        # whichever workflow its inputs resemble.
        handlers = {
            "extend-video": self._run_extension,
            "video-to-video": self._run_restyle,
            "music-video": self._run_music_video,
        }
        handler = handlers.get(job.workflow_id, self._run_generation)
        return await handler(job, reporter)

    # ── text-to-video / image-to-video ───────────────────────────────────

    async def _run_generation(
        self, job: AdapterJob, reporter: StageReporter
    ) -> AdapterResult:
        """A requested duration, optionally starting from an uploaded still."""
        self._require_generation_shape(job)
        seconds = self._requested_seconds(job)
        still = await self._conditioning_image(job, "source_image")
        dimensions = self._requested_dimensions(job)
        # `execution.generation_engine: guided` swaps the distilled entry point
        # for the guided two-stage pipeline — CFG, STG and a negative prompt
        # active, ~4.3x the render time. Off unless a workflow says so, exactly
        # like the transform engine and audio conditioning before it: a cost
        # and behaviour change is a product decision, never a silent default.
        guided = job.execution.get("generation_engine") == "guided"
        self._record_audio_mode(job, AudioMode.GENERATED_PER_SECTION_AUDIO)

        # Director mode: the idea becomes a validated global plan BEFORE any
        # section renders, and each section's prompt is compiled from that one
        # plan — never re-planned per section, which is what previously made
        # long-form dialogue restart at every seam. The standard path below is
        # untouched when the mode is off.
        director_plan = None
        if wants_director(job):
            await reporter.report(
                "preparing", 14, "Directing your scene…", {"phase": "preparing"}
            )
            try:
                director_plan = await create_director_plan(job, seconds)
            except DirectorFailure as failure:
                raise AdapterError(
                    failure.user_message,
                    internal_detail=failure.internal_detail,
                    retriable=False,
                ) from failure

        prompt_plan: list[str] | None = None

        def prompt_for_step(step: ChainStep) -> str:
            nonlocal prompt_plan
            if prompt_plan is None:
                if director_plan is not None:
                    prompt_plan = compile_section_prompts(
                        director_plan,
                        step.total,
                        total_seconds=seconds,
                        camera_continuity=bool(
                            job.execution.get("director_camera_continuity")
                        ),
                    )
                else:
                    prompt_plan = plan_section_prompts(
                        job.prompt,
                        step.total,
                        total_seconds=seconds,
                        v2=bool(job.execution.get("prompt_structuring_v2")),
                    )
            return prompt_plan[step.index]

        def conditioning(step: ChainStep) -> list[ConditioningFrame]:
            if step.is_first:
                return [ConditioningFrame(still, 0, 1.0)] if still else []

            items: list[ConditioningFrame] = []
            if step.previous_frame:
                items.append(ConditioningFrame(step.previous_frame, 0, 1.0))
            # A predecessor frame carries temporal state but is a weak identity
            # anchor. Keep the original upload in every I2V pass at low strength
            # and away from frame zero, so it guides identity without resetting
            # the section to the composition of the first image — where the
            # decoder is measured to survive the second image at all
            # (`_TWO_IMAGE_SAFE_FRAMES`; a 720-frame two-image pass is the
            # 20 Aug production crash).
            if still:
                anchor = self._identity_anchor(job, still, step.seconds)
                if anchor is not None:
                    items.append(anchor)
            return items

        rendered = await render_chain(
            job,
            seconds,
            per_pass_seconds=(
                self._guided_pass_seconds(job)
                if guided
                else self._per_pass_seconds(job, dimensions)
            ),
            render=self._renderer(job, reporter, dimensions=dimensions,
                                  conditioning=conditioning,
                                  prompt_for_step=prompt_for_step,
                                  require_audio=True,
                                  pipeline=_GUIDED if guided else _DISTILLED),
            reporter=reporter,
        )

        await reporter.stitching()
        output = job.workspace / "output.mp4"
        info = await self._assemble(
            job,
            # Normalize every pass to one FPS/timebase/stream layout before
            # concat. Equal requested dimensions do not guarantee equal media
            # timestamps, and a mismatch becomes a visible seam.
            lambda: self._assemble_generated_sections(
                job, rendered, output, dimensions=dimensions, audio=True
            ),
            output,
            OutputExpectation(
                expect_video=True, expect_audio=True, expected_seconds=seconds
            ),
            reporter,
        )

        await reporter.uploading()
        return _video_result(output, info)

    # ── extend-video ─────────────────────────────────────────────────────

    async def _run_extension(
        self, job: AdapterJob, reporter: StageReporter
    ) -> AdapterResult:
        """source → final frame → continuation → normalize → stitch.

        The finished file is the untouched-in-content source plus the
        continuation, both normalized to one set of encoder parameters at the
        source's own (capped) resolution.
        """
        # Extension gets its own, far looser source ceiling: the GPU renders
        # only the continuation, so the source's length prices as one ffmpeg
        # re-encode rather than as render passes. Holding it to the render
        # ceiling is what made extend-of-extend stop working at 5½ minutes.
        staged, source = await self._staged_source(
            job,
            "source_video",
            kind="video",
            limit_seconds=float(settings.ltx_max_extend_source_seconds),
        )
        extension_seconds = self._requested_seconds(job)

        # A video generated in Director mode keeps behaving like one when
        # extended. The API walked the source asset back to the job that made
        # it and attached the lineage; from it the CONTINUATION is planned —
        # same language, same people, the prior idea as the story-so-far —
        # and compiled through the same section compiler. The plan is
        # source-anchored by construction: the finished video's last frame is
        # the opening frame here, and it owns WHO and WHAT the way an
        # uploaded photograph does. A source with no lineage (an upload, a
        # standard generation, a pre-Director job) takes the path below,
        # byte-identical to what has always served extensions.
        lineage = continuation_lineage(job)
        continuation_plan = None
        if lineage is not None:
            await reporter.report(
                "preparing", 14, "Directing the continuation…", {"phase": "preparing"}
            )
            try:
                continuation_plan = await create_director_plan(
                    job, extension_seconds, lineage=lineage
                )
            except DirectorFailure as failure:
                raise AdapterError(
                    failure.user_message,
                    internal_detail=failure.internal_detail,
                    retriable=False,
                ) from failure

        prompt_plan: list[str] | None = None

        def prompt_for_step(step: ChainStep) -> str:
            nonlocal prompt_plan
            if prompt_plan is None:
                if continuation_plan is not None:
                    prompt_plan = compile_section_prompts(
                        continuation_plan,
                        step.total,
                        total_seconds=extension_seconds,
                        camera_continuity=bool(
                            job.execution.get("director_camera_continuity")
                        ),
                    )
                else:
                    # Timestamps in an extension prompt are relative to the
                    # EXTENSION, which is the only timeline the user is
                    # writing for.
                    prompt_plan = plan_section_prompts(
                        job.prompt,
                        step.total,
                        total_seconds=extension_seconds,
                        v2=bool(job.execution.get("prompt_structuring_v2")),
                    )
            return prompt_plan[step.index]

        await reporter.probing("Reading your video…")
        seed_frame = await self._final_frame_of(job, staged)
        # Present only on Director-lineage extensions of Image to Video: the
        # ORIGINAL uploaded image, carried forward by the API so the identity
        # anchor survives any number of extensions — the same low-strength
        # mid-window reference every I2V pass after the first has always used.
        identity = await self._conditioning_image(job, "identity_image")

        def conditioning(step: ChainStep) -> list[ConditioningFrame]:
            frame = step.previous_frame
            items = [ConditioningFrame(frame, 0, 1.0)] if frame else []
            if identity:
                anchor = self._identity_anchor(job, identity, step.seconds)
                if anchor is not None:
                    items.append(anchor)
            return items

        # The grid follows the SOURCE's aspect, not the request's: the I2V
        # benchmark showed a mismatched aspect makes the model keep the style
        # and replace the subject, which at a seam means a different video
        # after the join. Bound once so the pass ceiling is derived from the
        # shape actually rendered rather than the requested aspect's.
        grid = grid_for_source(source.width, source.height)

        per_pass = self._per_pass_seconds(job, grid)
        if continuation_plan is not None:
            # Directed passes hold their story for 30 seconds and not for 60 —
            # the same measurement that sectioned 60s generation (20 Aug 2026:
            # single-pass 60s dialogue repeated lines and returned departed
            # people; 30s passes measured clean). A long director extension
            # renders in the clean regime; standard extensions keep the
            # geometry they have always had.
            per_pass = min(per_pass, 30.0)

        rendered = await render_chain(
            job,
            extension_seconds,
            per_pass_seconds=per_pass,
            render=self._renderer(
                job, reporter,
                dimensions=grid,
                conditioning=conditioning,
                prompt_for_step=prompt_for_step,
            ),
            reporter=reporter,
            prefix="continuation",
            seed_frame=seed_frame,
        )

        await reporter.stitching()
        width, height = output_dimensions(source.width, source.height)
        fps = _delivery_fps(source)
        continuation_infos = [await probe_media(part) for part in rendered]
        keep_audio = source.has_audio or any(info.has_audio for info in continuation_infos)
        self._record_audio_mode(
            job,
            AudioMode.SOURCE_AUDIO
            if source.has_audio
            else (
                AudioMode.GENERATED_PER_SECTION_AUDIO
                if keep_audio
                else AudioMode.NO_AUDIO
            ),
        )
        output = job.workspace / "output.mp4"
        expected = (source.duration_seconds or 0.0) + extension_seconds

        async def assemble() -> Path:
            continuation = await self._assemble_generated_sections(
                job,
                rendered,
                job.workspace / "continuation.mp4",
                dimensions=(width, height),
                fps=fps,
                audio=keep_audio,
            )
            source_part = await normalize_clip(
                staged,
                job.workspace / "part-0000.mp4",
                width=width,
                height=height,
                fps=fps,
                audio=keep_audio,
            )
            return await concat_segments(
                [source_part, continuation], output
            )

        info = await self._assemble(
            job,
            assemble,
            output,
            OutputExpectation(
                expect_video=True,
                expect_audio=keep_audio,
                expected_seconds=expected,
                # Each re-timed part can drift a little; scale with length
                # instead of failing honest 60s extensions on frame rounding.
                tolerance_seconds=duration_tolerance(expected, floor=1.5),
            ),
            reporter,
        )

        await reporter.uploading()
        return _video_result(output, info)

    # ── video-to-video ───────────────────────────────────────────────────

    async def _run_restyle(
        self, job: AdapterJob, reporter: StageReporter
    ) -> AdapterResult:
        """Restyle footage, matching the source's own duration exactly.

        The customer picks no duration for this workflow — the API rejects one
        — so the target is whatever the probe measures, and a 42-second upload
        produces a 42-second result whether that is one pass or two.

        What keeps it a restyle rather than an unrelated generation is the
        conditioning: several stills lifted from the *same window of the source*
        that the pass is about to generate, so subject placement, framing and
        the direction of movement carry over while the prompt supplies the look.
        Passes after the first also take their frame 0 from the previous pass's
        last frame, which is what makes the joins invisible.
        """
        staged, source = await self._staged_source(job, "source_video", kind="video")
        await reporter.probing("Reading your video…")

        target_seconds = source.duration_seconds or 0.0
        reference = await self._conditioning_image(job, "reference_image")
        grid = grid_for_source(source.width, source.height)

        if str(job.execution.get("v2v_engine") or "").strip() == "transform":
            return await self._run_transform(
                job, reporter, staged, source, target_seconds, reference, grid
            )

        if job.execution.get("v2v_reference_identity") and reference is not None:
            # Identity replacement is built on the transform engine's control
            # and attention machinery. Running this job through the still-
            # conditioned restyle would deliver the source person unchanged
            # while the workflow claims replacement — the silent-success
            # failure mode, refused rather than shipped.
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=(
                    "v2v_reference_identity requires `v2v_engine: transform`; "
                    "the still-conditioned restyle cannot replace a person and "
                    "must not pretend to"
                ),
                retriable=False,
            )

        # An explicit count still wins — it is how a workflow pins conditioning
        # for footage where the derived density is wrong — but the default is
        # derived per pass from the duration it actually has to cover.
        explicit_keyframes = job.execution.get("v2v_keyframes")
        keyframe_seconds = max(
            0.5, job.execution_float("v2v_keyframe_seconds", _V2V_KEYFRAME_SECONDS)
        )
        floor, cap = _V2V_KEYFRAME_BOUNDS

        def keyframes_for(seconds: float) -> int:
            if explicit_keyframes is not None:
                return max(1, min(cap, int(explicit_keyframes)))
            return max(floor, min(cap, math.ceil(seconds / keyframe_seconds)))

        structure = job.execution_float("v2v_structure_strength", _V2V_STRUCTURE_STRENGTH)
        continuity = job.execution_float("v2v_continuity_strength", _V2V_CONTINUITY_STRENGTH)
        reference_strength = job.execution_float(
            "v2v_reference_strength", _V2V_REFERENCE_STRENGTH
        )

        async def conditioning(step: ChainStep) -> list[ConditioningFrame]:
            frames = self._frame_count(step.seconds)
            keyframes = keyframes_for(step.seconds)
            items: list[ConditioningFrame] = []

            # Frame 0 is the seam (or, on the first pass, the one place a
            # reference image can guide the look without displacing the
            # source's structure). Exactly one thing may own it.
            anchored = False
            if step.previous_frame is not None:
                items.append(ConditioningFrame(step.previous_frame, 0, continuity))
                anchored = True
            elif reference is not None and reference_strength > 0:
                items.append(ConditioningFrame(reference, 0, reference_strength))
                anchored = True

            # Source stills spread across the window this pass covers. The
            # half-step offsets keep them off both ends, so they never collide
            # with the anchor above and never fight the next pass's opening.
            offsets = [(index + 0.5) / keyframes for index in range(keyframes)]
            if not anchored:
                offsets.insert(0, 0.0)

            window = step.segment.start_seconds
            timestamps = [window + offset * step.seconds for offset in offsets]
            stills = await cancellable(
                job,
                extract_frames_at(
                    staged,
                    timestamps,
                    job.workspace / "keyframes",
                    prefix=f"pass-{step.index:04d}",
                ),
            )
            items += [
                ConditioningFrame(still, min(frames - 1, round(offset * (frames - 1))), structure)
                for still, offset in zip(stills, offsets, strict=False)
            ]
            return items

        per_pass = self._per_pass_seconds(job, grid)
        rendered = await render_chain(
            job,
            target_seconds,
            per_pass_seconds=per_pass,
            render=self._renderer(
                job,
                reporter,
                dimensions=grid,
                conditioning=conditioning,
                # `v2v_section_prompts` gives each section its own compiled
                # prompt instead of the same byte-identical text N times — the
                # only workflow family still without per-section prompts. Off
                # (the default) every section receives job.prompt exactly as
                # it always has; whether per-section text helps or destabilises
                # a restyle is a GPU A/B, which is why this is a flag and not
                # a behaviour change.
                prompt_for_step=self._v2v_prompt_for_step(job, target_seconds),
            ),
            reporter=reporter,
            prefix="restyled",
        )

        return await self._deliver_restyle(
            job, reporter, rendered, staged, source, target_seconds, per_pass
        )

    async def _run_transform(
        self,
        job: AdapterJob,
        reporter: StageReporter,
        staged: Path,
        source: MediaInfo,
        target_seconds: float,
        reference: Path | None,
        grid: tuple[int, int],
    ) -> AdapterResult:
        """Restyle by CONTROL SIGNAL rather than by stills — the strong path.

        The restyle above shows the model photographs of the source and asks it
        to draw something similar. That preserves the shot well and transforms
        it weakly, which is exactly the complaint: the source's colour, light
        and material come along with its geometry, and the prompt is left
        fighting them.

        This path separates the two. An edge map of the source window carries
        the geometry — subject outline, placement, camera motion, scene layout —
        and carries nothing else, so the prompt owns every pixel of the look
        with nothing to fight. Measured 17 Aug 2026: a daylight desert plate
        returned as a rain-soaked neon street with the subject's pose, the car's
        position and the road's perspective unchanged.

        Everything the restyle guarantees still holds, because it is the same
        chain: the length is the SOURCE's, the seams are the previous pass's
        final frame, and the source's own audio goes back on whole at the end.
        """
        await reporter.probing("Reading your video…")
        # `execution.v2v_person_lock` carries the subject's OWN pixels inside a
        # tracked matte, so a restyle stops inventing a new face and a new skin
        # tone for a person the prompt never asked to change. Off unless a
        # workflow says otherwise: it adds a matting pass per section, and the
        # existing engine is what has been serving customers.
        person_lock = bool(job.execution.get("v2v_person_lock"))
        # `execution.v2v_reference_identity` is the opposite ask: the person
        # should NOT survive — the uploaded reference image supplies who they
        # are, while the footage keeps supplying what they do. Three levers
        # move together (see the constants for why each exists): the reference
        # is re-anchored in EVERY pass rather than shown once, and the edge
        # map's grip is loosened over the person's own region so the source's
        # facial geometry stops being re-imposed. Without a reference image
        # the flag is inert and the job is an ordinary transform.
        identity = bool(job.execution.get("v2v_reference_identity")) and reference is not None
        if identity and person_lock:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=(
                    "v2v_person_lock and v2v_reference_identity are both set: "
                    "one preserves the source person, the other replaces them. "
                    "A workflow must choose."
                ),
                retriable=False,
            )

        if identity and job.execution.get("v2v_identity_describe_reference", True):
            # The model never sees the reference as anything but pixels, and
            # pixels lose to a prompt that says nothing about the person —
            # the first production identity job proved it: a customer prompt
            # made of meta-instructions ("preserve the exact face… no
            # flicker") named no visible attribute, and the render was
            # neither the source person nor the reference. So the worker
            # LOOKS at the photo and appends what it sees, in caption voice,
            # after the user's own text — which stays verbatim, first.
            # "" on any failure means the prompt simply goes through
            # unchanged; the description is reinforcement, not a dependency.
            facts = await cancellable(job, reference_person_facts(reference))
            if facts:
                # Caption voice, and NO photograph vocabulary. "The person
                # from the reference image, exactly as photographed" reads as
                # meta to a human and as CONTENT to the model — words like
                # "image" and "photographed" in the prompt invite a posed
                # photo-shoot shot, which is one of the two ways the first
                # customer job got a portrait cut into a dance video.
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
        strength = job.execution_float("v2v_control_strength", _V2V_CONTROL_STRENGTH)
        lora_strength = job.execution_float("v2v_lora_strength", _V2V_LORA_STRENGTH)
        continuity = job.execution_float("v2v_continuity_strength", _V2V_CONTINUITY_STRENGTH)
        reference_strength = job.execution_float(
            "v2v_reference_strength", _V2V_REFERENCE_STRENGTH
        )
        anchor_strength = job.execution_float(
            "v2v_identity_anchor_strength", _V2V_IDENTITY_ANCHOR_STRENGTH
        )
        # The composited anchor: the footage's own opening frame with the
        # reference person standing where the source person stood, anchored
        # at full strength. Built once per job, before any pass renders.
        # Falls back to the raw photo at the capped strength — the previously
        # shipped behaviour — when it cannot be built; the log says why.
        anchor = reference
        if identity and job.execution.get("v2v_identity_composited_anchor", True):
            built = await cancellable(
                job,
                build_identity_anchor(
                    staged,
                    reference,
                    job.workspace / "identity-anchor.png",
                    start_seconds=0.0,
                    width=grid[0],
                    height=grid[1],
                ),
            )
            if built is not None:
                anchor = built
            else:
                anchor_strength = min(
                    anchor_strength, _V2V_IDENTITY_RAW_ANCHOR_STRENGTH
                )
        elif identity:
            anchor_strength = min(anchor_strength, _V2V_IDENTITY_RAW_ANCHOR_STRENGTH)
        refresh_strength = job.execution_float(
            "v2v_identity_refresh_strength", _V2V_IDENTITY_REFRESH_STRENGTH
        )
        subject_attention = job.execution_float(
            "v2v_identity_subject_attention", _V2V_IDENTITY_SUBJECT_ATTENTION
        )
        low = job.execution_float("v2v_edge_low", DEFAULT_EDGE_LOW)
        high = job.execution_float("v2v_edge_high", DEFAULT_EDGE_HIGH)

        lora = LoraSpec(
            settings.ltx_models_root / self._optional_weight("union_control_lora"),
            lora_strength,
        )

        def conditioning(step: ChainStep) -> list[ConditioningFrame]:
            if identity:
                # The reference conditions EVERY pass, not only the first.
                # A continuity frame carries the replacement forward but
                # decays — each pass reproduces the previous pass's rendering,
                # and the errors compound back toward what the control signal
                # suggests, which is the source person. Ascending frame order,
                # as `_command` requires.
                if step.previous_frame is None:
                    return [ConditioningFrame(anchor, 0, anchor_strength)]
                items = [ConditioningFrame(step.previous_frame, 0, continuity)]
                if refresh_strength > 0:
                    frames = self._frame_count(step.seconds)
                    interior = min(frames - 1, max(1, frames // 3))
                    if interior > 0:
                        items.append(
                            ConditioningFrame(reference, interior, refresh_strength)
                        )
                return items
            # Frame 0 only. The control clip already states where everything is
            # for the whole window, so source stills would be a second, weaker
            # copy of the same instruction — and a redundant one that reasserts
            # the source's LOOK, which is the very thing this path exists to
            # discard.
            if step.previous_frame is not None:
                return [ConditioningFrame(step.previous_frame, 0, continuity)]
            if reference is not None and reference_strength > 0:
                return [ConditioningFrame(reference, 0, reference_strength)]
            return []

        async def control(step: ChainStep, frames: int) -> ControlConditioning:
            path = await cancellable(
                job,
                extract_edge_control(
                    staged,
                    job.workspace / f"control-{step.index:04d}.mp4",
                    start_seconds=step.segment.start_seconds,
                    duration_seconds=step.seconds,
                    width=grid[0],
                    height=grid[1],
                    fps=float(settings.ltx_frame_rate),
                    frames=frames,
                    low=low,
                    high=high,
                ),
            )
            if not person_lock:
                return ControlConditioning(path, strength)
            path = await self._person_locked_control(
                job, staged, path, step=step, frames=frames, grid=grid
            )
            return ControlConditioning(path, strength)

        async def attention(step: ChainStep, frames: int) -> MaskConditioning | None:
            """Per-region weights on the control signal, when a mode needs them.

            Person lock weights the subject ABOVE the scene (their real pixels
            are inside the control clip and must be followed hard). Identity
            replacement weights them BELOW it (the edges over the person are
            the source's facial geometry, which is exactly what must lose its
            grip). Neither mode active → no mask, and a mask of all one value
            would be a slower way of saying nothing.

            A matting failure here fails the pass. Delivering the source
            person at full control strength while claiming the reference
            replaced them is the one outcome this mode must never produce.
            """
            if identity:
                matte = await cancellable(
                    job,
                    build_person_matte(
                        staged,
                        job.workspace / f"matte-{step.index:04d}.mp4",
                        start_seconds=step.segment.start_seconds,
                        duration_seconds=step.seconds,
                        width=grid[0],
                        height=grid[1],
                        fps=float(settings.ltx_frame_rate),
                        frames=frames,
                    ),
                )
                weights = await cancellable(
                    job,
                    build_attention_mask(
                        matte,
                        job.workspace / f"attention-{step.index:04d}.mp4",
                        frames=frames,
                        background=1.0,
                        subject=subject_attention,
                    ),
                )
                return MaskConditioning(weights, 1.0)
            if not person_lock:
                return None
            matte = job.workspace / f"matte-{step.index:04d}.mp4"
            if not matte.exists():
                return None
            weights = await cancellable(
                job,
                build_attention_mask(
                    matte,
                    job.workspace / f"attention-{step.index:04d}.mp4",
                    frames=frames,
                    background=job.execution_float(
                        "v2v_background_attention", BACKGROUND_ATTENTION
                    ),
                ),
            )
            return MaskConditioning(weights, 1.0)

        # The tighter of the grid's distilled ceiling and the transform tier's
        # own measured pass length. The former exists so an unmeasured grid
        # still gets its pessimistic 10s; the latter is the binding one on the
        # product grids, where the distilled 60s says nothing about ic_lora.
        per_pass = min(
            self._per_pass_seconds(job, grid),
            max(1.0, job.execution_float(
                "transform_pass_seconds", _TRANSFORM_PASS_SECONDS
            )),
        )

        rendered = await render_chain(
            job,
            target_seconds,
            per_pass_seconds=per_pass,
            render=self._renderer(
                job, reporter, dimensions=grid,
                conditioning=conditioning,
                pipeline=_IC_LORA,
                loras=(lora,),
                control=control,
                mask=attention,
                # Same flag as the restyle engine: per-section prompts, off by
                # default. On the transform engine job.prompt already carries
                # the appended identity caption, so sectioning repeats the
                # person's description in every pass — the measured
                # anti-drift pattern Director mode uses.
                prompt_for_step=self._v2v_prompt_for_step(job, target_seconds),
            ),
            reporter=reporter,
            prefix="transformed",
        )

        return await self._deliver_restyle(
            job, reporter, rendered, staged, source, target_seconds, per_pass
        )

    async def _deliver_restyle(
        self,
        job: AdapterJob,
        reporter: StageReporter,
        rendered: list[Path],
        staged: Path,
        source: MediaInfo,
        target_seconds: float,
        per_pass_seconds: float,
    ) -> AdapterResult:
        """Assemble, restore the source's audio once, verify, hand back.

        Shared by both video-to-video engines because none of it is engine
        specific — and because "the source's audio survives exactly once" and
        "the result is the source's length" are promises that must not be able
        to hold on one path and quietly not on the other.

        `per_pass_seconds` re-derives the chain's own plan so each section can
        be delivered at exactly its planned length. This workflow lays ONE
        continuous soundtrack — the source's — over sections that were
        generated separately, so a section delivered even half a frame long
        pushes every later section's content later against that audio. The
        error is small and it only ever accumulates: measured arithmetically,
        a 37s source at 30fps drifts +33ms per seam and ends +133ms late,
        which is past the ~45ms threshold where a viewer sees a mouth move
        after the words. Sections whose planned lengths are whole frames were
        never affected, which is why this surfaced as "minor" and only on some
        uploads.
        """
        await reporter.stitching()
        width, height = output_dimensions(source.width, source.height)
        fps = _delivery_fps(source)
        keep_audio = source.has_audio
        self._record_audio_mode(
            job, AudioMode.SOURCE_AUDIO if keep_audio else AudioMode.NO_AUDIO
        )
        section_frames = self._planned_section_frames(
            rendered, target_seconds, per_pass_seconds, fps
        )
        output = job.workspace / "output.mp4"

        async def assemble() -> Path:
            # `audio=False` on purpose: the model generates its own soundtrack,
            # and a restyle that replaced the user's audio with an invented one
            # would be a bug nobody asked for. The source's own track goes back
            # on below, whole.
            picture = await self._assemble_generated_sections(
                job,
                rendered,
                job.workspace / "picture.mp4",
                dimensions=(width, height),
                fps=fps,
                audio=False,
                section_frames=section_frames,
            )
            if not keep_audio:
                return picture.replace(output)
            await reporter.muxing("Restoring your audio…")
            return await mux_audio(picture, staged, output)

        info = await self._assemble(
            job,
            assemble,
            output,
            OutputExpectation(
                expect_video=True,
                expect_audio=keep_audio,
                expected_seconds=target_seconds,
                tolerance_seconds=duration_tolerance(target_seconds, floor=1.0),
                expected_width=width,
                expected_height=height,
            ),
            reporter,
        )

        await reporter.uploading()
        return _video_result(output, info)

    # ── music-video ──────────────────────────────────────────────────────

    async def _run_music_video(
        self, job: AdapterJob, reporter: StageReporter
    ) -> AdapterResult:
        """Visuals for the whole uploaded track, with the track laid over once.

        The client's requirement is specific and it is about the audio, not the
        picture: the finished file carries the COMPLETE song, continuous, not
        restarting per visual section. That shape is enforced structurally —
        the chain produces silent picture, and `mux_audio` attaches the user's
        original file exactly once, at the end, as the only soundtrack.

        **Whether the MODEL hears the song is a separate question from whether
        the viewer does**, and the two were conflated for a long time. By
        default the picture is drawn from the text prompt alone: the song
        decides the length and where the cuts land, and nothing else. Under
        `execution.audio_conditioning` the audio tier renders instead, and each
        pass is given the master track seeked to its own moment in the song, so
        the model is drawing *to* the music it is about to be played under —
        which is what makes a performer's mouth follow a vocal at all.
        """
        staged, track = await self._staged_source(job, "source_audio", kind="audio")
        await reporter.probing("Listening to your track…")

        target_seconds = track.duration_seconds or 0.0
        self._record_audio_mode(job, AudioMode.SOURCE_AUDIO)
        dimensions = self._requested_dimensions(job)
        audio_conditioned = bool(job.execution.get("audio_conditioning"))
        if job.execution.get("require_audio_conditioning") and not audio_conditioned:
            # The client-test guard. The default music-video path generates
            # from the prompt alone and muxes the track afterwards — measured
            # 24 Aug 2026: the output envelope matched the input within 0.5 dB
            # and mouth motion did not correlate with the vocal. A client-test
            # environment that promises lip response must be UNABLE to run the
            # unconditioned route by a config slip; the two keys are set
            # together in the client-test YAML, and this refuses when they
            # disagree instead of shipping a video that quietly cannot sync.
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=(
                    "require_audio_conditioning is set but audio_conditioning "
                    "is not: the prompt-only + post-mux route is forbidden in "
                    "client-test mode"
                ),
                retriable=False,
            )
        per_pass = (
            self._audio_pass_seconds(job)
            if audio_conditioned
            else self._per_pass_seconds(job, dimensions)
        )

        conditioning_master = staged
        if audio_conditioned:
            # A container's declared duration can overstate its decodable
            # samples (MP3 encoder padding). Production, 27 Aug 2026: the
            # final pass of nine asked for its window at the track's tail,
            # the decoder came up one audio latent short (500 against 501),
            # and the pipeline's shape assert took down a 35-minute render at
            # its very last step — on every attempt, deterministically. The
            # conditioning copy is the master decoded ONCE to WAV (lossless,
            # no codec seams) with two seconds of appended silence, so every
            # window can always be filled. The delivered soundtrack remains
            # the untouched original, muxed once at the end.
            conditioning_master = job.workspace / "conditioning-master.wav"
            await cancellable(
                job,
                ffmpeg(
                    [
                        "-i", str(staged),
                        "-af", "apad=pad_dur=2.0",
                        str(conditioning_master), "-y",
                    ]
                ),
            )

        # Where the track is actually SUNG — so the performance direction can
        # match the audio instead of commanding singing throughout. The
        # client's 27 Aug audit measured mouth-vs-audio correlation at 0.027,
        # with the singer mouthing words through a forty-second instrumental
        # intro: the prompt's unconditional "she sings" out-shouted the audio
        # conditioning. None (separation unavailable) keeps old behaviour.
        vocal_spans: list[tuple[float, float]] | None = None
        if audio_conditioned and job.execution.get("vocal_aware_prompts", True):
            vocal_spans = await cancellable(job, vocal_activity(conditioning_master))
            logger.info(
                "vocal_spans",
                extra={
                    "job_id": job.job_id,
                    "spans": [
                        (round(a, 1), round(b, 1)) for a, b in (vocal_spans or [])
                    ][:20],
                },
            )

        prompt_plan: list[str] | None = None

        def prompt_for_step(step: ChainStep) -> str:
            nonlocal prompt_plan
            if prompt_plan is None:
                # Timestamped shots in a music-video prompt refer to positions
                # in the SONG, which is exactly the timeline of the chain.
                prompt_plan = plan_section_prompts(
                    job.prompt,
                    step.total,
                    total_seconds=target_seconds,
                    v2=bool(job.execution.get("prompt_structuring_v2")),
                )
            prompt = prompt_plan[step.index]
            if vocal_spans is not None:
                sung = vocal_fraction(
                    vocal_spans,
                    step.segment.start_seconds,
                    step.segment.start_seconds + step.segment.duration_seconds,
                )
                if sung < 0.2:
                    prompt += (
                        " The music in this passage is INSTRUMENTAL — nobody "
                        "sings here: the performer's lips stay closed, she "
                        "sways and feels the beat, no mouth movement of "
                        "singing at any point in this section."
                    )
                elif sung > 0.6:
                    prompt += (
                        " The vocal is active in this passage: she sings the "
                        "words, her mouth clearly articulating in time with "
                        "the voice in the music."
                    )
            return prompt

        boundaries = await self._musical_boundaries(job, staged, target_seconds, per_pass)

        anchor_state: dict[str, Path] = {}

        def conditioning(step: ChainStep) -> list[ConditioningFrame]:
            frame = step.previous_frame
            if not frame:
                return []
            frames = [ConditioningFrame(frame, 0, 1.0)]
            # The three-women problem (client frame-audit, 27 Aug: the
            # specified singer held 15 of 180 seconds). Each section inherits
            # identity only from its predecessor's final frame, and nine
            # copy-of-a-copy hops let the model's prior overwrite the face —
            # the restated captions alone could not hold it. Pin SECTION 1's
            # final frame as the identity reference for every later section:
            # the same anchor mechanism as image-to-video, including its
            # measured decoder guard — a two-image pass at an unmeasured
            # frame count drops the anchor with a log line rather than
            # gambling on the decode.
            if job.execution.get("mv_identity_anchor", True):
                anchor = anchor_state.setdefault("first_seam", frame)
                if anchor != frame:
                    extra = self._identity_anchor(
                        job, anchor, step.segment.duration_seconds
                    )
                    if extra is not None:
                        frames.append(extra)
            return frames

        def audio_window(step: ChainStep, frames: int) -> AudioConditioning:
            # The MASTER file, seeked — never a slice. Cutting the track into
            # per-section files would re-encode it N times and put a codec
            # boundary at every seam of the thing the model is synchronising
            # to. The window is measured from the frame count actually being
            # rendered, so a pass the shape tables nudged still gets audio that
            # covers it.
            return AudioConditioning(
                path=conditioning_master,
                start_seconds=step.segment.start_seconds,
                max_duration_seconds=self._audio_window_seconds(frames),
            )

        rendered = await render_chain(
            job,
            target_seconds,
            per_pass_seconds=per_pass,
            render=self._renderer(
                job, reporter, dimensions=dimensions,
                conditioning=conditioning,
                prompt_for_step=prompt_for_step,
                pipeline=_A2VID if audio_conditioned else _DISTILLED,
                audio=audio_window if audio_conditioned else None,
            ),
            reporter=reporter,
            prefix="scene",
            boundaries=boundaries,
        )

        await reporter.stitching()
        width, height = dimensions
        output = job.workspace / "output.mp4"
        # The same arithmetic video-to-video needs, for the same reason and
        # with the same evidence: ONE continuous soundtrack is laid over
        # sections that were generated separately, so a section delivered even
        # half a frame long pushes every later section's picture later against
        # the song. It only ever accumulates. Simulated through this planner:
        # a 300s track with cuts on the music drifts +21ms per seam and ends
        # +310ms late on the audio tier, +125ms on the default one — past the
        # ~45ms threshold where a viewer sees a mouth move after the words.
        #
        # `boundaries` is not optional here. Music-video windows come from the
        # track's own events, so re-deriving them from an even split would pin
        # every section to a length it was never rendered for.
        section_frames = self._planned_section_frames(
            rendered,
            target_seconds,
            per_pass,
            float(settings.ltx_frame_rate),
            boundaries=boundaries,
        )

        async def assemble() -> Path:
            picture = await self._assemble_generated_sections(
                job,
                rendered,
                job.workspace / "picture.mp4",
                dimensions=(width, height),
                audio=False,
                section_frames=section_frames,
            )
            await reporter.muxing("Adding your track…")
            return await mux_audio(picture, staged, output)

        info = await self._assemble(
            job,
            assemble,
            output,
            OutputExpectation(
                expect_video=True,
                # Both halves of the client's promise, checked on the real file:
                # the song is present, and the result is the song's length.
                expect_audio=True,
                expected_seconds=target_seconds,
                tolerance_seconds=duration_tolerance(target_seconds, floor=1.0),
            ),
            reporter,
        )

        await reporter.uploading()
        return _video_result(output, info)

    async def _musical_boundaries(
        self, job: AdapterJob, track: Path, total_seconds: float, per_pass: float
    ) -> list[float]:
        """Cut points taken from the music, or none — never a failed job.

        Timing analysis is an improvement to where the seams land, not a
        prerequisite for producing the video. A track this cannot measure still
        gets even windows, which is exactly what it would have got anyway.
        """
        if not job.execution.get("align_cuts_to_audio", True):
            return []
        try:
            onsets = await cancellable(job, audio_onsets(track))
        except FfmpegError as exc:
            logger.info("onset_analysis_skipped", extra={"detail": str(exc)})
            return []
        return plan_musical_boundaries(
            total_seconds, per_pass_seconds=per_pass, onsets=onsets
        )

    # ── Shared assembly and validation ───────────────────────────────────

    async def _assemble(
        self,
        job: AdapterJob,
        build,
        output: Path,
        expectation: OutputExpectation,
        reporter: StageReporter,
    ) -> MediaInfo:
        """Runs an assembly step, then refuses to ship what it produced unless
        the file is genuinely deliverable.

        Every workflow ends here. The whole point of a single exit is that
        "never mark a job successful unless the artifact validates" cannot be
        forgotten in one branch — there is only one branch.
        """
        try:
            await cancellable(job, build())
            await reporter.finalizing("Verifying your videoâ€¦")
            return await verify_output(output, expectation)
        except FfmpegError as exc:
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=f"assembly or validation failed: {exc}",
            ) from exc

    async def _assemble_generated_sections(
        self,
        job: AdapterJob,
        rendered: list[Path],
        output: Path,
        *,
        dimensions: tuple[int, int],
        fps: float | None = None,
        audio: bool,
        section_frames: Sequence[int] | None = None,
    ) -> Path:
        """Normalize FPS/timebase/streams before any generated-section concat.

        `section_frames` pins each normalized section to an exact frame count
        (see `_planned_section_frames`). Callers that lay ONE continuous
        soundtrack over the stitched picture need this; callers whose sections
        carry their own audio do not, because there any duration rounding moves
        both streams together.
        """
        width, height = dimensions
        normalized: list[Path] = []
        for index, part in enumerate(rendered):
            job.raise_if_cancelled()
            normalized.append(
                await normalize_clip(
                    part,
                    job.workspace / f"normalized-section-{index:04d}.mp4",
                    width=width,
                    height=height,
                    fps=fps or float(settings.ltx_frame_rate),
                    audio=audio,
                    frames=(
                        section_frames[index] if section_frames is not None else None
                    ),
                )
            )
        return await concat_segments(normalized, output)

    def _planned_section_frames(
        self,
        rendered: list[Path],
        target_seconds: float,
        per_pass_seconds: float,
        fps: float,
        boundaries: list[float] | None = None,
    ) -> list[int] | None:
        """Each section's delivery length in whole frames, from the plan.

        `plan_chain_segments` is pure, so calling it again with the chain's own
        inputs reproduces the chain's own windows. Counts are allocated from
        the CUMULATIVE boundary (`round(end · fps) − round(start · fps)`)
        rather than per-section, so rounding can never walk: every section
        starts within half a delivery frame of its planned timestamp, and the
        error at the last seam is the same as at the first.

        `boundaries` must be the SAME cut points the chain was given. A music
        video aligns its cuts to the track's own events, so its windows are not
        the even ones `plan_segments` would produce — asking without them
        returns a different plan, the section count usually still matches, and
        every section is then pinned to a length it was never rendered for.

        Returns None when the recomputed plan does not match what was actually
        rendered — that would mean this arithmetic has diverged from the
        chain's, and delivering with the historical quantization is strictly
        better than cutting sections at wrong lengths.
        """
        segments = plan_chain_segments(target_seconds, per_pass_seconds, boundaries)
        if len(segments) != len(rendered):
            logger.warning(
                "section_plan_mismatch",
                extra={
                    "planned_sections": len(segments),
                    "rendered_sections": len(rendered),
                },
            )
            return None
        counts: list[int] = []
        for segment in segments:
            start = round(segment.start_seconds * fps)
            end = round((segment.start_seconds + segment.duration_seconds) * fps)
            counts.append(max(1, end - start))
        return counts

    def _record_audio_mode(self, job: AdapterJob, mode: AudioMode) -> None:
        logger.info(
            "audio_mode_selected",
            extra={"workflow_id": job.workflow_id, "audio_mode": mode.value},
        )

    # ── Guardrails ───────────────────────────────────────────────────────

    def _require_generation_shape(self, job: AdapterJob) -> None:
        """Refuses job shapes this runtime cannot honestly produce.

        The failure mode to avoid is the quiet one: an audio job that produces
        a broken file three attempts later, or a source-conditioned job that
        ignores its source and returns unrelated text-to-video footage.
        """
        if job.execution.get("output_kind") == "audio":
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=(
                    "audio-only output is not supported by the LTX distilled entry "
                    "point (benchmarked 2026-08-12: it attaches libx264 to an mp3 "
                    "container and fails); music must route to the `music` runtime"
                ),
                retriable=False,
            )
        unsupported = [item.role for item in job.inputs if item.role != "source_image"]
        if unsupported:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=(
                    f"inputs {unsupported} reached the plain generation path on "
                    f"workflow '{job.workflow_id}'; source-conditioned workflows have "
                    "their own handlers and must not silently drop their inputs"
                ),
                retriable=False,
            )

    async def _staged_source(
        self,
        job: AdapterJob,
        role: str,
        *,
        kind: str,
        limit_seconds: float | None = None,
    ) -> tuple[Path, MediaInfo]:
        """The uploaded file and its measurements, or a clear refusal.

        Probing here rather than at first use is what turns "the GPU burned two
        minutes and produced nothing" into "that file could not be read",
        before any compute is spent. A corrupt upload is corrupt on every
        attempt, hence `retriable=False`.

        Every refusal below names WHICH thing is wrong. "That audio file could
        not be read" told a customer who had uploaded a silent video clip, one
        who had uploaded a nine-minute track, and one whose file was truncated
        exactly the same thing — and only one of those three is something they
        can act on by picking a different file.
        """
        item = job.input_for(role)
        if item is None:
            raise AdapterError(
                "This generation could not be started.",
                internal_detail=f"{job.workflow_id} job arrived without a {role} input",
                retriable=False,
            )
        staged = item.require_path()

        noun = "video" if kind == "video" else "audio file"
        try:
            info = await probe_media(staged)
        except FfmpegError as exc:
            raise AdapterError(
                f"That {noun} could not be read — it may be damaged or in an "
                "unsupported format. Please try another file.",
                internal_detail=f"probe of {role} failed: {exc}",
                retriable=False,
            ) from exc

        # The file opened, but does it contain the stream this workflow needs?
        # A video with no sound reaching music-video is the common case, and it
        # is a completely different mistake from a corrupt file.
        if kind == "video" and not info.has_video:
            raise AdapterError(
                "That file has no video in it. Please upload a video.",
                internal_detail=f"{role} has no video stream: {info}",
                retriable=False,
            )
        if kind == "audio" and not info.has_audio:
            raise AdapterError(
                "That file has no sound in it. Please upload an audio track, "
                "or a video that has sound.",
                internal_detail=f"{role} has no audio stream: {info}",
                retriable=False,
            )

        if not info.duration_seconds:
            raise AdapterError(
                f"That {noun} appears to be empty. Please try another file.",
                internal_detail=f"{role} has no measurable duration: {info}",
                retriable=False,
            )

        # The bound that stops a job running for hours. Nothing capped this,
        # so an hour-long upload became ~120 render passes: it could not finish
        # inside its own timeout, and it held the card until it failed.
        # `limit_seconds` lets a workflow whose render cost does NOT scale with
        # the source (extension: the source is re-encoded, never re-rendered)
        # carry a looser bound than the render-length default.
        limit = float(
            limit_seconds if limit_seconds is not None else settings.ltx_max_source_seconds
        )
        if info.duration_seconds > limit:
            raise AdapterError(
                f"That {noun} is {_minutes(info.duration_seconds)} long, and "
                f"the limit is {_minutes(limit)}. Please trim it and try again.",
                internal_detail=(
                    f"{role} is {info.duration_seconds:.1f}s, over the "
                    f"{limit:.0f}s source ceiling"
                ),
                retriable=False,
            )
        return staged, info

    async def _conditioning_image(self, job: AdapterJob, role: str) -> Path | None:
        """A staged still, verified decodable, or None when not supplied.

        This decodes one frame rather than probing: ffprobe's metadata pass
        accepts garbage (the `tty` demuxer will even claim ASCII text as
        "video"), while an actual decode rejects a truncated or mislabelled
        upload in milliseconds.
        """
        item = job.input_for(role)
        if item is None:
            return None
        staged = item.require_path()
        try:
            await ffmpeg(
                ["-i", str(staged), "-frames:v", "1", "-f", "null", "-"], timeout=30
            )
        except FfmpegError as exc:
            raise AdapterError(
                "That image could not be read. Please try another.",
                internal_detail=f"decode check of {role} failed: {exc}",
                retriable=False,
            ) from exc
        return staged

    async def _final_frame_of(self, job: AdapterJob, source: Path) -> Path:
        try:
            return await cancellable(
                job, extract_final_frame(source, job.workspace / "seed-frame.png")
            )
        except FfmpegError as exc:
            raise AdapterError(
                "That video could not be read. Please try another.",
                internal_detail=f"final-frame extraction from source failed: {exc}",
                retriable=False,
            ) from exc

    def _require_models(self) -> None:
        shared = ("text_encoder", "video_vae", "audio_vae", "duration_head", "spatial_upsampler")
        needed = [self._transformer_file(), *(_MODEL_FILES[key] for key in shared)]
        root = settings.ltx_models_root
        missing = [name for name in needed if not (root / name).exists()]
        if missing:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=f"LTX weights missing under {root}: {missing}",
                retriable=False,
            )

    def _requested_seconds(self, job: AdapterJob) -> float:
        """The requested length. NOT capped at the single-pass ceiling.

        Anything longer than one pass is chained, so the ceiling constrains
        each GPU invocation rather than the product. Which lengths a customer
        may ask for is the workflow definition's job, validated by the API
        before the request ever reaches a worker.

        Automatic-duration workflows never call this — their length comes from
        a probe of the uploaded file, which is the point of `duration_mode:
        source`.
        """
        seconds = parse_duration_seconds(job.parameters.get("duration"))
        if seconds is None:
            raise AdapterError(
                "This generation could not be started.",
                internal_detail=f"no usable duration in {job.parameters!r}",
                retriable=False,
            )
        return seconds

    def _requested_dimensions(self, job: AdapterJob) -> tuple[int, int]:
        return _DIMENSIONS.get(
            str(job.parameters.get("aspect_ratio") or ""), _DEFAULT_DIMENSIONS
        )

    def _per_pass_seconds(
        self, job: AdapterJob, dimensions: tuple[int, int] | None = None
    ) -> float:
        """The most this GPU may be asked for in one invocation, for THIS shape.

        The ceiling is a property of the grid, not of the product. 1024x576
        sustains 60s while 896x512 — fewer pixels — dies at 60s and survives 30s.
        A single global value therefore has to satisfy the worst shape offered,
        which is how every 60s render came to be split into six passes with five
        seams when only one aspect ratio needed it.

        `dimensions` is passed by every caller that knows the grid, which is all
        of them; the default exists so the ceiling can still be asked for
        generically. A workflow may lower it further via
        `execution.max_segment_seconds` — nothing raises it above what the grid
        was measured at, because that is where the kernel failure lives.
        """
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

    async def _person_locked_control(
        self,
        job: AdapterJob,
        staged: Path,
        edges: Path,
        *,
        step: ChainStep,
        frames: int,
        grid: tuple[int, int],
    ) -> Path:
        """An edge map carrying the subject's real pixels inside their matte.

        The transform engine's whole premise is that edges keep the geometry
        and discard the look — which is right for the scene and wrong for the
        person standing in it, because a face and a skin tone are exactly the
        "look" a customer expects to survive a change of weather.

        So the person is cut back in: matte the window, take the source's own
        pixels for it, and merge them into the edge map where the matte says
        the subject is. Everything outside stays an outline and is re-imagined
        as before. Measured 2026-08-18 — the same man, relit for the new scene,
        for about seven seconds of extra work per pass.

        Every clip here is built at the pass's grid, rate and ACTUAL frame
        count, because the merge is frame-for-frame and a misalignment would
        protect the wrong pixels rather than simply protect less.
        """
        index = step.index
        matte = await cancellable(
            job,
            build_person_matte(
                staged,
                job.workspace / f"matte-{index:04d}.mp4",
                start_seconds=step.segment.start_seconds,
                duration_seconds=step.seconds,
                width=grid[0],
                height=grid[1],
                fps=float(settings.ltx_frame_rate),
                frames=frames,
            ),
        )
        footage = await cancellable(
            job,
            extract_source_window(
                staged,
                job.workspace / f"window-{index:04d}.mp4",
                start_seconds=step.segment.start_seconds,
                duration_seconds=step.seconds,
                width=grid[0],
                height=grid[1],
                fps=float(settings.ltx_frame_rate),
                frames=frames,
            ),
        )
        return await cancellable(
            job,
            build_hybrid_control(
                edges,
                footage,
                matte,
                job.workspace / f"hybrid-{index:04d}.mp4",
                frames=frames,
                # False keeps the PERSON. Inverting keeps the scene and frees
                # the subject's region, which is what a future person
                # REPLACEMENT needs — the same builder, the other side.
                invert=False,
            ),
        )

    def _audio_pass_seconds(self, job: AdapterJob) -> float:
        """The pass ceiling for the AUDIO tier, which is its own measurement.

        `_GRID_CEILINGS` records what the distilled tier survives and none of it
        transfers: the audio tier runs a different checkpoint, carries a LoRA,
        runs unquantized and streams its weights from host RAM. The longest pass
        actually measured for it at the production grid is 481 frames — 20.04s
        at 1024x576, 17 Aug 2026 — so that, and not the distilled tier's 60s, is
        what a pass may ask for.

        Deliberately overridable: raising it is a measurement, not an opinion,
        and `execution.audio_pass_seconds` is where the measurement lands once
        someone has made it.
        """
        requested = job.execution_float("audio_pass_seconds", _AUDIO_PASS_SECONDS)
        return max(1.0, min(requested, float(settings.ltx_max_seconds)))

    def _guided_pass_seconds(self, job: AdapterJob) -> float:
        """The pass ceiling for the GUIDED tier — its own measurement, again.

        See `_GUIDED_PASS_SECONDS` for the 241-frame failure that keeps this
        at one measured landing. Deliberately overridable downward, and upward
        only alongside a new measurement: `execution.guided_pass_seconds` is
        where that lands, and `_GUIDED.measured_landings` is what actually
        keeps unmeasured counts away from the decoder in the meantime.
        """
        requested = job.execution_float("guided_pass_seconds", _GUIDED_PASS_SECONDS)
        return max(1.0, min(requested, float(settings.ltx_max_seconds)))

    @staticmethod
    def _audio_window_seconds(frames: int) -> float:
        """How much track a pass rendering `frames` frames must be given.

        Slightly MORE than the video covers, never less. The audio latent grid
        is coarser than the frame grid, so a window cut to exactly the video's
        length can land one latent short and the pipeline then reads past the
        end of what it was handed. The reference engine pads by one audio latent
        for the same reason; this is that pad.
        """
        return frames / float(settings.ltx_frame_rate) + _AUDIO_WINDOW_PAD_SECONDS

    def _frame_count(self, seconds: float) -> int:
        return max(1, round(seconds * settings.ltx_frame_rate))

    def _identity_anchor(
        self, job: AdapterJob, still: Path, pass_seconds: float
    ) -> ConditioningFrame | None:
        """The original upload as a mid-window identity reference — when the
        decoder is measured to survive it.

        A pass carrying this is a TWO-image pass, and the decoder's two-image
        failure set is its own lottery: 720 frames decodes fine with one image
        and crashed a production job with two, 736 fails identically (so the
        render-extra-and-trim dodge is useless here), while 120/240/360 pass.
        Measured 20 Aug 2026; `_TWO_IMAGE_SAFE_FRAMES` is the evidence.

        Returning None is not a degraded render: the pass keeps the seam frame
        at full strength and the compiled captions restate identity — the
        exact configuration the 60s two-section validation ran clean. A log
        line records the drop so an identity-drift report can be correlated
        with it.
        """
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

    def _v2v_prompt_for_step(
        self, job: AdapterJob, total_seconds: float
    ) -> Callable[[ChainStep], str] | None:
        """Per-section prompts for video-to-video, behind `v2v_section_prompts`.

        Video-to-video is the workflow family that chains longest (a 330 s
        source on the transform engine is 44 sections) and the only one whose
        sections all receive byte-identical prompt text. With the flag off
        (the default) this returns None and `_command` falls back to
        `job.prompt` exactly as it always has; whether per-section text helps
        or destabilises a restyle is a GPU A/B question, which is why this is
        a flag and not a behaviour change.
        """
        if not job.execution.get("v2v_section_prompts"):
            return None
        v2 = bool(job.execution.get("prompt_structuring_v2"))
        prompt_plan: list[str] | None = None

        def prompt_for_step(step: ChainStep) -> str:
            nonlocal prompt_plan
            if prompt_plan is None:
                prompt_plan = plan_section_prompts(
                    job.prompt, step.total, total_seconds=total_seconds, v2=v2
                )
            return prompt_plan[step.index]

        return prompt_for_step

    # ── The renderer handed to the chain ─────────────────────────────────

    def _renderer(
        self,
        job: AdapterJob,
        reporter: StageReporter,
        *,
        dimensions: tuple[int, int],
        conditioning,
        prompt_for_step: Callable[[ChainStep], str] | None = None,
        require_audio: bool = False,
        pipeline: LtxPipeline = _DISTILLED,
        loras: Sequence[LoraSpec] = (),
        control=None,
        mask=None,
        audio=None,
        skip_stage_2: bool = False,
    ) -> RenderStep:
        """Binds this job's fixed choices into the callable the chain drives.

        `conditioning` may be sync or async — building a restyle's conditioning
        means extracting stills from the source, which is I/O, while a
        text-to-video's is a one-line decision.

        `control` and `audio` are called with `(step, frames)` rather than just
        the step, and that is not incidental: both have to match the frame count
        the pass will ACTUALLY render, which is the measured-safe substitute
        rather than the requested one. A control clip or audio window built
        against the requested count drifts out of alignment on every pass the
        shape tables nudge.
        """

        async def render(step: ChainStep) -> None:
            items = conditioning(step)
            if asyncio.iscoroutine(items):
                items = await items
            # Dodge the decoder's measured bad shapes: render a few extra
            # frames where the exact count would crash the VAE, then trim back
            # so the delivered pass is exactly the planned length.
            requested_frames = self._frame_count(step.seconds)
            # A pass is "conditioned" for shape purposes if it carries ANY
            # conditioning, not only `--image`. A control video and an audio
            # window are conditioning too, and the decoder's bad-shape set
            # differs between the two classes.
            #
            # This cost a production job. The transform engine deliberately
            # drops the source stills the old restyle passed, so its first pass
            # — the common case, with no reference image — carried no `--image`
            # at all. `conditioned` flipped to False, the lattice snapped a
            # 14.976s source's 359 frames up to 361 instead of landing on the
            # measured-safe 360, and 361 is a count this decoder is already
            # documented to die on (see `_CONDITIONED_BANDS`). The old engine
            # never hit it only because its stills happened to keep the flag
            # True.
            conditioned = bool(items) or control is not None or audio is not None
            if pipeline.conforming_only:
                # The measured tables below are distilled-decoder evidence and
                # say nothing about this pipeline's decode path — see
                # `LtxPipeline.conforming_only` for the two production failures
                # that proved it in opposite directions.
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
            # Logged for EVERY pass, not only nudged ones. When a pass crashes
            # the decoder, its frame count and grid are the whole diagnosis —
            # and on 2026-08-16 a music-video failure could not be attributed
            # to a pass at all, because only the one nudged pass in four had
            # said what it was rendering. A line per pass costs nothing and is
            # the difference between "a pass failed" and "1381 frames
            # conditioned at 1024x576 fails".
            logger.info(
                "pass_frames",
                extra={
                    "workflow_id": job.workflow_id,
                    "pass_index": step.index,
                    "passes": step.total,
                    "grid": list(dimensions),
                    "seconds": round(step.seconds, 3),
                    "requested_frames": requested_frames,
                    "rendered_frames": frames,
                    "nudged": frames != requested_frames,
                    # The value the shape tables were consulted with, so the log
                    # explains the frame count it sits next to. Logging
                    # `bool(items)` here instead is what made the 361 crash read
                    # as "unconditioned" when the pass was carrying a control
                    # video the whole time.
                    "conditioned": conditioned,
                    "images": len(items),
                },
            )
            control_item = control(step, frames) if control else None
            if asyncio.iscoroutine(control_item):
                control_item = await control_item
            # After the control clip, never before: the mask is built from the
            # same matte that control step produced, and asking for it first
            # would race a file that does not exist yet.
            mask_item = mask(step, frames) if mask else None
            if asyncio.iscoroutine(mask_item):
                mask_item = await mask_item
            audio_item = audio(step, frames) if audio else None
            if asyncio.iscoroutine(audio_item):
                audio_item = await audio_item

            await self._execute(
                job=job,
                cmd=self._command(
                    job,
                    step.seconds,
                    step.output,
                    conditioning=items,
                    dimensions=dimensions,
                    prompt=(prompt_for_step(step) if prompt_for_step else None),
                    # Distinct per pass or every chained render replays the
                    # same noise; still deterministic so a retry reproduces.
                    seed=self._seed_for_step(job, step.index),
                    num_frames=frames,
                    pipeline=pipeline,
                    loras=loras,
                    control=control_item,
                    mask=mask_item,
                    audio=audio_item,
                    skip_stage_2=skip_stage_2,
                ),
                reporter=reporter,
                band=step.band,
                section=step.section_progress,
            )
            if frames != requested_frames:
                # BEFORE verification and before any continuity frame is
                # extracted, so the seam frame sits at the planned timestamp.
                #
                # Bounded by what WE added, never more. Snapping to the model's
                # lattice overshoots by at most 7 frames, and trimming exactly
                # that much is invisible. Trimming an arbitrary amount would
                # make this a length fixer — and a render that came back
                # seconds too long is a fault the verification below exists to
                # catch, not something to quietly cut down to size.
                overshoot = (frames - requested_frames) / float(settings.ltx_frame_rate)
                await self._trim_to(
                    job, step.output, step.seconds, tolerance_seconds=overshoot
                )
            if require_audio:
                try:
                    await verify_output(
                        step.output,
                        OutputExpectation(
                            expect_video=True,
                            expect_audio=True,
                            expected_seconds=step.seconds,
                        ),
                    )
                except FfmpegError as exc:
                    raise AdapterError(
                        "This generation could not be completed. Please try again.",
                        internal_detail=(
                            f"section {step.index + 1}/{step.total} failed media "
                            f"validation: {exc}"
                        ),
                    ) from exc

        return render

    # ── The command (pure — this is what unit tests pin) ─────────────────

    def _transformer_file(self) -> str:
        key = "transformer_nvfp4" if "nvfp4" in settings.ltx_quantization else "transformer_bf16"
        return _MODEL_FILES[key]

    def _launcher(self, module: str = _DISTILLED.module) -> list[str]:
        """The argv prefix that reaches the LTX environment.

        A separate seam so tests can substitute a plain Python stub and still
        exercise every real flag `_command` produces. The module is a parameter
        because the three tiers are three entry points in the same environment.
        """
        return ["uv", "run", "python", "-m", module]

    def _command(
        self,
        job: AdapterJob,
        seconds: float,
        output: Path,
        *,
        conditioning: Sequence[ConditioningFrame] = (),
        dimensions: tuple[int, int] | None = None,
        seed: int | None = None,
        prompt: str | None = None,
        num_frames: int | None = None,
        pipeline: LtxPipeline = _DISTILLED,
        loras: Sequence[LoraSpec] = (),
        control: ControlConditioning | None = None,
        mask: MaskConditioning | None = None,
        audio: AudioConditioning | None = None,
        skip_stage_2: bool = False,
    ) -> list[str]:
        root = settings.ltx_models_root
        # Explicit dimensions (extension and restyle: the source's aspect
        # decides) beat the requested aspect ratio's lookup.
        width, height = dimensions or self._requested_dimensions(job)
        # `num_frames` lets the renderer substitute a measured-safe count for a
        # frame count the VAE cannot decode (see `safe_frame_count`); the extra
        # material is trimmed after the render, never delivered.
        frames = num_frames if num_frames is not None else self._frame_count(seconds)
        if seed is None:
            # The pipeline's default seed is fixed, which would hand two users
            # with the same prompt the same video. CRC of the job id:
            # deterministic per job (a retried job reproduces), distinct
            # across jobs.
            seed = zlib.crc32(job.job_id.encode())

        transformer = (
            self._transformer_file()
            if pipeline.transformer_key is None
            else self._optional_weight(pipeline.transformer_key)
        )
        if pipeline.stage_1_only:
            # Ask for twice the target and stop after stage 1, which renders at
            # half — so the delivered grid is exactly `width x height`. See
            # `LtxPipeline.stage_1_only`.
            width, height = width * 2, height * 2
            skip_stage_2 = True

        cmd = [
            *self._launcher(pipeline.module),
            "--transformer-path", str(root / transformer),
            "--text-encoder-path", str(root / _MODEL_FILES["text_encoder"]),
            "--video-vae-path", str(root / _MODEL_FILES["video_vae"]),
            "--audio-vae-path", str(root / _MODEL_FILES["audio_vae"]),
            "--duration-head-path", str(root / _MODEL_FILES["duration_head"]),
            "--spatial-upsampler-path", str(root / _MODEL_FILES["spatial_upsampler"]),
        ]
        if pipeline.quantize:
            cmd += ["--quantization", settings.ltx_quantization]
        if pipeline.offload_cpu and settings.ltx_unquantized_offload != "none":
            # The other half of dropping quantization: the unquantized 22B
            # transformer is fitted by streaming weights from host RAM —
            # unless this node has the VRAM to keep it resident, which is
            # measured 23-30% faster (see ltx_unquantized_offload).
            cmd += ["--offload", settings.ltx_unquantized_offload]
        if pipeline.distilled_lora:
            cmd += [
                "--distilled-lora",
                str(root / self._optional_weight("distilled_lora")),
                "1.0",
            ]

        cmd += [
            # Exactly what the user typed, as one argv element: no quoting,
            # escaping, truncation or rewriting between the text field and
            # the model. Pinned by test_the_users_prompt_reaches_the_model_verbatim.
            "--prompt", job.prompt if prompt is None else prompt,
            "--num-frames", str(frames),
            "--height", str(height),
            "--width", str(width),
            "--frame-rate", str(settings.ltx_frame_rate),
            "--seed", str(seed),
            "--output-path", str(output),
        ]
        if pipeline.distilled_lora:
            # The guided family (dev transformer + distilled LoRA) is the ONLY
            # place these exist — the distilled and ic_lora entry points have
            # no guiders, and sending the flags there is a crash, not a hint.
            # Both stay unset by default: the pipeline's own defaults are the
            # measured baseline, and a guidance change is a quality/cost
            # judgement that belongs in a workflow's execution block.
            negative = str(job.execution.get("negative_prompt") or "").strip()
            if negative:
                cmd += ["--negative-prompt", negative]
            # Each scale also decides whether a whole extra transformer call
            # runs per step, which is what makes these cost levers as well as
            # quality ones. From `ltx_core/components/guiders.py`: a cfg other
            # than 1.0 buys an unconditional pass, an STG other than 0.0 buys a
            # perturbed pass, and a modality scale other than 1.0 buys the
            # isolated-modality pass. The shipped defaults (3.0 / 1.0 / 3.0)
            # are therefore FOUR calls per step, and the audio tier runs 30 of
            # them at half resolution before stage 2 begins.
            #
            # `a2v_guidance_scale` is the modality one, and on the audio tier
            # it is the lip-sync dial: the isolated pass it enables is the one
            # that skips the audio-to-video cross-attention, so the guider
            # pushes the picture AWAY from a version of itself that did not
            # listen. LTX's own help calls it out — "higher values may increase
            # lipsync quality" — and 1.0 means the pass does not run at all.
            for key, flag in (
                ("guidance_scale", "--video-cfg-guidance-scale"),
                ("stg_scale", "--video-stg-guidance-scale"),
                ("a2v_guidance_scale", "--a2v-guidance-scale"),
            ):
                value = job.execution.get(key)
                if value is not None:
                    cmd += [flag, str(round(float(value), 3))]
            steps = job.execution.get("inference_steps")
            if steps is not None:
                # Stage 1's step count. Stage 2 runs the distilled LoRA's own
                # fixed short schedule and is not affected.
                cmd += ["--num-inference-steps", str(max(1, int(steps)))]
        if job.execution.get("enhance_prompt"):
            # LTX's own enhancer expands a terse prompt into a detailed one,
            # which is the only adherence lever the distilled entry point
            # offers. It REWRITES the prompt, so it stays opt-in per workflow
            # and off by default: a user who typed something specific should
            # not silently get a machine's paraphrase of it.
            #
            # The gemma4 encode root cannot generate, so the pipeline REQUIRES
            # a separate instruct checkpoint alongside the bare flag — without
            # it the render exits with a ValueError before any GPU work. The
            # root is the same Apache-2.0 checkpoint the Director planner
            # runs, so one download serves both.
            cmd += [
                "--enhance-prompt",
                "--prompt-enhancer-gemma-root", str(settings.director_gemma_root),
            ]
        for item in conditioning:
            # `--image PATH FRAME_IDX STRENGTH`, once per conditioning frame,
            # in ascending frame order so the pipeline reads them as a timeline
            # rather than as an unordered set.
            cmd += item.as_args()
        for lora in loras:
            cmd += lora.as_args()
        if control is not None:
            cmd += control.as_args()
        if mask is not None:
            cmd += mask.as_args()
        if audio is not None:
            cmd += audio.as_args()
        if skip_stage_2:
            cmd.append("--skip-stage-2")
        return cmd

    def _optional_weight(self, key: str) -> str:
        """A named weights file, verified present, or a refusal naming it.

        Resolves against both tables: a non-default tier may need a file every
        tier needs (the IC-LoRA path runs the same distilled transformer as the
        default) as well as one only it needs.

        Existence is checked HERE rather than only in `_require_models` because
        those weights are checked once per job on the assumption every render
        needs them, while these are checked only on the path that selected them.
        A node without the audio tier's dev checkpoint still serves
        text-to-video perfectly, and must not be told it is broken.
        """
        relative = _OPTIONAL_MODEL_FILES.get(key) or _MODEL_FILES[key]
        if not (settings.ltx_models_root / relative).exists():
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=(
                    f"LTX weights missing for '{key}': "
                    f"{settings.ltx_models_root / relative}"
                ),
                retriable=False,
            )
        return relative

    async def _trim_to(
        self,
        job: AdapterJob,
        path: Path,
        seconds: float,
        *,
        tolerance_seconds: float = 0.0,
    ) -> None:
        """Cuts a nudged render back to the requested length, in place.

        A re-encode rather than a stream copy: `-c copy` can only cut on a
        keyframe, and the whole point is landing on the exact requested
        duration. Settings match the section normalizer so a trimmed pass is
        indistinguishable from an untrimmed one downstream.

        `tolerance_seconds` is how much overshoot this trim is entitled to
        remove — the amount snapping to the model's frame lattice added, and
        nothing beyond it. A render that came back materially longer than that
        did something the caller did not ask for, and the verification that
        follows is supposed to see it. Cutting it silently would turn this
        into a length fixer and delete the evidence.
        """
        if tolerance_seconds > 0:
            info = await probe_media(path)
            actual = info.duration_seconds or 0.0
            # A generous multiple: encoders round durations, and the point is
            # to catch "five seconds instead of two", not tenths.
            allowed = seconds + max(0.5, tolerance_seconds * 3)
            if actual > allowed:
                logger.warning(
                    "render_far_longer_than_planned",
                    extra={
                        "planned_seconds": round(seconds, 3),
                        "actual_seconds": round(actual, 3),
                        "trim_allowance": round(tolerance_seconds, 3),
                    },
                )
                return
        trimmed = path.with_name(path.stem + ".trimmed.mp4")
        try:
            await cancellable(
                job,
                ffmpeg([
                    "-i", str(path),
                    "-t", f"{seconds:.3f}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    str(trimmed),
                ]),
            )
        except FfmpegError as exc:
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=f"trimming a nudged render failed: {exc}",
            ) from exc
        trimmed.replace(path)

    def _seed_for_step(self, job: AdapterJob, index: int) -> int:
        requested = job.parameters.get("seed")
        try:
            base = int(requested) if requested is not None else None
        except (TypeError, ValueError):
            base = None
        if base is None:
            return zlib.crc32(f"{job.job_id}:{index}".encode())
        return (base + index) % (2**31)

    # ── Supervision ──────────────────────────────────────────────────────

    async def _execute(
        self,
        cmd: list[str],
        job: AdapterJob,
        reporter: StageReporter,
        *,
        band: tuple[int, int] = (GENERATE_FROM, GENERATE_TO),
        section: tuple[int, int, float, float] | None = None,
    ) -> None:
        """Runs the pipeline, streaming its output for progress and diagnostics.

        The contract with the rest of the platform:

          * `job.raise_if_cancelled()` is honoured within `_CANCEL_POLL_SECONDS`
            even while the pipeline is silent — cancellation and timeout both
            surface here as exceptions.
          * The child is killed on *every* non-completion path (the `finally`),
            because an orphaned render holds VRAM, and the runner is about to
            delete the workspace the child is writing into.

        `band` compresses the markers' sweep into a slice of it, so N chained
        passes produce one monotonic ramp instead of N restarts. `section`
        swaps the stage messages for "Generating section i of N…" — the
        machinery is only named when there are several.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(settings.ltx_repo_dir),
                # The launcher is `uv run python -m ...`, so the process this
                # handle refers to is uv and the one holding tens of gigabytes
                # of VRAM is its child. Its own session makes the pair a
                # process GROUP, which is the only handle that reaches both.
                # Ignored on Windows, where the tests run against a stub that
                # spawns nothing.
                start_new_session=True,
            )
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=f"could not launch LTX pipeline ({cmd[0]!r}): {exc}",
                retriable=False,
            ) from exc

        tail: deque[str] = deque(maxlen=_OUTPUT_TAIL_LINES)
        marker_from = 0
        try:
            assert process.stdout is not None
            while True:
                job.raise_if_cancelled()
                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(), timeout=_CANCEL_POLL_SECONDS
                    )
                except TimeoutError:
                    continue  # silence — loop back to the cancellation check
                if not line:
                    break
                text = line.decode("utf-8", "replace").rstrip()
                if text:
                    tail.append(text)
                matched = match_marker(text, marker_from)
                if matched is None:
                    continue
                _, nominal, message = _MARKERS[matched]
                marker_from = matched + 1
                low, high = band
                progress = low + (nominal - GENERATE_FROM) * (high - low) // (
                    GENERATE_TO - GENERATE_FROM
                )
                if section is None:
                    await reporter.generating(progress, message)
                else:
                    await reporter.section(
                        section[0],
                        section[1],
                        progress,
                        start_seconds=section[2],
                        end_seconds=section[3],
                    )

            returncode = await process.wait()
            if returncode != 0:
                output = " | ".join(tail)
                raise AdapterError(
                    "This generation could not be completed. Please try again.",
                    internal_detail=(
                        f"LTX pipeline exited {returncode}; output tail: {output}"
                    ),
                    retriable=not _is_deterministic_failure(output),
                )
        finally:
            await _terminate_render(process)


#: Pipeline failures that the SAME input reproduces every time, so a retry
#: spends the whole render again to reach the identical crash.
#:
#: These are shape bugs, not luck: the VAE's batched GEMM casts its dimensions
#: to int32 and fails on particular frame counts (see `_BAD_FRAME_BANDS`), and
#: a missing kernel is missing on every attempt. Observed 2026-08-16 — a
#: music-video job failed twice with CUBLAS_STATUS_INTERNAL_ERROR at the same
#: point, six minutes in, having burned twelve minutes of a card that other
#: customers were queued behind.
#:
#: OUT OF MEMORY IS DELIBERATELY ABSENT. That one genuinely does depend on what
#: else held the card at the time, and it is the case a retry exists for.
_DETERMINISTIC_FAILURES = (
    "CUBLAS_STATUS_INTERNAL_ERROR",
    "CUBLAS_STATUS_NOT_SUPPORTED",
    "CUBLAS_STATUS_INVALID_VALUE",
    "no kernel image is available",
    "CUDA error: invalid argument",
)


def _is_deterministic_failure(output: str) -> bool:
    return any(needle in output for needle in _DETERMINISTIC_FAILURES)


def _minutes(seconds: float) -> str:
    """A duration a customer can compare against their own file.

    "9 minutes 12 seconds", not "552.3s" — the refusal it appears in is asking
    someone to go and trim a track, and seconds are the wrong unit for that.
    """
    whole = int(round(seconds))
    minutes, remainder = divmod(whole, 60)
    if not minutes:
        return f"{remainder} seconds"
    if not remainder:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"{minutes} minute{'s' if minutes != 1 else ''} {remainder} seconds"


#: How long to wait for a killed render — the whole process group — to be gone
#: before giving up and saying so. A generous ceiling: the alternative to
#: waiting is claiming another job while the old one still owns the card.
_KILL_GRACE_SECONDS = 30.0


async def _terminate_render(process: asyncio.subprocess.Process) -> bool:
    """Kill a render and everything it spawned, and confirm the card is free.

    Killing the handle alone is not enough and the log proves it. On
    2026-08-16 a job lost its lease mid-generation, the cleanup here reported
    `ltx_kill_timeout`, the worker claimed the next job in the same second,
    and that job died with 43 MB free of a 102 GB card — the render that was
    supposedly killed still held ~56 GB. `uv run python -m ...` means the
    handle is uv and the memory belongs to its child, which a signal to the
    handle never reaches.

    So the group gets the signal, and then this WAITS for the group to empty
    rather than assuming. `killpg(pgid, 0)` raises once nothing is left in it,
    which is the only reliable "the card is actually free now" available
    without polling the driver.

    Returns whether the group is confirmed gone. A False is serious — it means
    a render is still holding VRAM that nothing is tracking.
    """
    if process.returncode is not None:
        return True

    loop = asyncio.get_running_loop()
    pgid: int | None = None
    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            pgid = os.getpgid(process.pid)
        except (ProcessLookupError, PermissionError, OSError):
            pgid = None

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError, OSError):
        # Already gone, or not ours to signal. Fall through to the wait, which
        # settles which of the two it was.
        pass

    # Reap the handle first so the process table entry goes away.
    try:
        await asyncio.wait_for(process.wait(), timeout=_KILL_GRACE_SECONDS)
    except (TimeoutError, asyncio.CancelledError):
        logger.warning("ltx_kill_timeout", extra={"pid": process.pid})

    if pgid is None:
        return process.returncode is not None

    # The grandchild is not ours to wait() on — it is reparented when uv dies —
    # so the group is polled instead. This is the part that was missing.
    deadline = loop.time() + _KILL_GRACE_SECONDS
    while loop.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except (PermissionError, OSError):
            return True
        await asyncio.sleep(0.25)

    logger.error(
        "ltx_render_still_holding_gpu",
        extra={"pid": process.pid, "pgid": pgid, "waited": _KILL_GRACE_SECONDS},
    )
    return False


def _delivery_fps(source: MediaInfo) -> float:
    """The rate a stitched result is delivered at.

    The source's own where it is sane, so a 30fps upload does not come back
    retimed. Clamped because ffprobe reports nonsense for some variable-rate
    phone recordings, and normalizing to a nonsense rate produces a file that
    plays at the wrong speed.
    """
    return min(60.0, max(10.0, source.fps or float(settings.ltx_frame_rate)))


def _video_result(output: Path, info: MediaInfo) -> AdapterResult:
    return AdapterResult(
        path=output,
        content_type="video/mp4",
        kind="video",
        duration_seconds=info.duration_seconds,
        width=info.width,
        height=info.height,
    )
