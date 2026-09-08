"""One music-video shot on the official LTX 2.5 audio-to-video graph.

The client's package renders every shot by launching `ltx_pipelines.a2vid_
two_stage` as a fresh process: the 42 GB development transformer plus the
distilled LoRA, unquantized, 24 denoising steps, and about 35 seconds of
process start and weight loading before the first step. Measured 8 Sep 2026
on the RTX PRO 6000: 96 seconds per 121-frame shot with the weights
resident, 120 with them streaming.

Lightricks ship the same job as a ComfyUI workflow on the fast path —
`example_workflows/2.5/LTX-2.5_A2V_Two_Stage_Distilled.json` in the
ComfyUI-LTXVideo pack. This module is that workflow, flattened by hand into
an API prompt, with the sizes and the audio window driven from a shot. Two
things make it cheaper than the CLI, and neither is a shortcut we invented:

  * **8 steps, not 24.** The distilled transformer carries its own schedule,
    and the sigmas below are byte-identical to the ones in the client's own
    FAST 1080 graph — the engine behind Text to Video, which they have
    already seen and accepted.
  * **The models stay loaded.** Every shot is one more prompt on a ComfyUI
    that is already holding the weights, so the per-shot loading disappears.

The conditioning mechanism is unchanged, which is the part that matters:
the song is encoded by the audio VAE, given an all-zero noise mask so the
sampler never denoises it, and concatenated with the video latent. That is
the same `frozen=True, noise_scale=0.0, initial_latent=encoded_audio` the
CLI pipeline sets. The picture is generated against the real audio, not
beside it.

What differs and must be watched: the distilled path guides at CFG 1, so
the extra audio-to-video guidance pass the CLI runs at scale 3.0 does not
happen here. Whether the mouth still follows the vocal as tightly is a
question for measurement, not for reasoning — `LTXVModalityGuidance` is
available if it turns out to be needed, at the cost of an extra transformer
call per step.

Weights, all already on the node:
  diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
  vae/ltx-2.5-video-vae-bf16.safetensors
  vae/ltx-2.5-audio-vae-bf16.safetensors
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
"""

from __future__ import annotations

from typing import Any

#: The distilled 8-step schedule. Identical to the client's FAST 1080 graph.
STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

#: The 3-step refine that runs after the 2x latent upscale.
STAGE2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"

TRANSFORMER = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
TEXT_ENCODER = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

#: Node classes the prompt uses, checked against the server before submitting
#: so a ComfyUI without the LTX pack fails with a sentence rather than a dump.
REQUIRED_NODES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "LatentUpscaleModelLoader",
    "CLIPTextEncode",
    "LTXVConditioning",
    "LoadImage",
    "LTXVPreprocess",
    "EmptyLTXVLatentVideo",
    "LTXVImgToVideoInplace",
    "LoadAudio",
    "TrimAudioDuration",
    "LTXVAudioVAEEncode",
    "SolidMask",
    "SetLatentNoiseMask",
    "LTXVConcatAVLatent",
    "LTXVSeparateAVLatent",
    "RandomNoise",
    "CFGGuider",
    "KSamplerSelect",
    "ManualSigmas",
    "SamplerCustomAdvanced",
    "LTXVLatentUpsampler",
    "VAEDecodeTiled",
    "CreateVideo",
    "SaveVideo",
)


def missing_nodes(catalogue: dict[str, Any] | set[str]) -> list[str]:
    names = set(catalogue.keys()) if isinstance(catalogue, dict) else set(catalogue)
    return [node for node in REQUIRED_NODES if node not in names]


def base_canvas(width: int, height: int) -> tuple[int, int]:
    """The first stage's size: half of the delivered frame on each side.

    The official graph generates 960x544 and upsamples to 1920x1088. Halving
    is the ratio the latent upsampler is trained for, and both halves must
    stay on the model's 32-pixel grid.
    """
    if width % 64 or height % 64:
        raise ValueError(
            f"a two-stage size must halve onto the 32-pixel grid, got {width}x{height}"
        )
    return width // 2, height // 2


def compile_a2v(
    *,
    positive: str,
    negative: str,
    audio: str,
    image: str,
    audio_start_seconds: float,
    seconds: float,
    frames: int,
    width: int,
    height: int,
    filename_prefix: str,
    seed: int = 0,
    fps: float = 24.0,
    two_stage: bool = True,
    steps_sigmas: str = STAGE1_SIGMAS,
    refine_sigmas: str = STAGE2_SIGMAS,
    sampler: str = "euler_ancestral",
    cfg: float = 1.0,
    image_strength: float = 0.7,
    refine_image_strength: float = 1.0,
    img_compression: int = 18,
    transformer: str = TRANSFORMER,
    text_encoder: str = TEXT_ENCODER,
    video_vae: str = VIDEO_VAE,
    audio_vae: str = AUDIO_VAE,
    upscaler: str = UPSCALER,
    tile: int = 512,
    tile_overlap: int = 64,
    temporal_size: int = 64,
    temporal_overlap: int = 8,
) -> dict[str, Any]:
    """The API prompt for one shot.

    `audio` and `image` are file names already inside ComfyUI's input folder.
    The audio is the job's whole conditioning master; `audio_start_seconds`
    and `seconds` cut this shot's window out of it, so the master is uploaded
    once per job and never sliced on disk — the same arrangement the CLI's
    `--audio-start-time` gives us today.

    `width` and `height` are the DELIVERED size. With `two_stage` the first
    stage runs at half that and the latent upsampler doubles it; without it
    the single stage runs at the delivered size, which is what the client's
    FAST 1080 graph does.
    """
    if frames < 1 or (frames - 1) % 8:
        raise ValueError(f"frames must sit on the model's 8k+1 lattice, got {frames}")
    if two_stage:
        stage1_width, stage1_height = base_canvas(width, height)
    else:
        if width % 32 or height % 32:
            raise ValueError(f"size must be a multiple of 32, got {width}x{height}")
        stage1_width, stage1_height = width, height

    api: dict[str, Any] = {
        # ── Models. One loader each; ComfyUI keeps them resident between
        #    prompts, which is the whole point of rendering here.
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": transformer, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": text_encoder, "type": "ltxv", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae}},
        # ── Text.
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": positive}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative}},
        "8": {
            "class_type": "LTXVConditioning",
            "inputs": {"positive": ["6", 0], "negative": ["7", 0], "frame_rate": float(fps)},
        },
        # ── The shot's canvas, with the anchor still written into frame 0.
        "12": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": int(stage1_width),
                "height": int(stage1_height),
                "length": int(frames),
                "batch_size": 1,
            },
        },
        "13": {
            "class_type": "LTXVImgToVideoInplace",
            "inputs": {
                "vae": ["3", 0],
                "image": ["11", 0],
                "latent": ["12", 0],
                "strength": float(image_strength),
                "bypass": False,
            },
        },
        # ── The song. Encoded, frozen, and joined to the video latent.
        "14": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "15": {
            "class_type": "TrimAudioDuration",
            "inputs": {
                "audio": ["14", 0],
                "start_index": float(audio_start_seconds),
                "duration": float(seconds),
            },
        },
        "16": {
            "class_type": "LTXVAudioVAEEncode",
            "inputs": {"audio": ["15", 0], "audio_vae": ["4", 0]},
        },
        "17": {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 1024, "height": 1024}},
        "18": {
            "class_type": "SetLatentNoiseMask",
            "inputs": {"samples": ["16", 0], "mask": ["17", 0]},
        },
        "19": {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {"video_latent": ["13", 0], "audio_latent": ["18", 0]},
        },
        # ── Stage 1: eight steps, CFG 1, the distilled schedule.
        "20": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "21": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["8", 0],
                "negative": ["8", 1],
                "cfg": float(cfg),
            },
        },
        "22": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
        "23": {"class_type": "ManualSigmas", "inputs": {"sigmas": steps_sigmas}},
        "24": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["20", 0],
                "guider": ["21", 0],
                "sampler": ["22", 0],
                "sigmas": ["23", 0],
                "latent_image": ["19", 0],
            },
        },
        "25": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["24", 0]}},
    }

    # The anchor, loaded and compressed the way the graph expects. Every
    # shot has one: the package draws an empty-scene still for the shots
    # with no performer in them, so this is never absent.
    api["10"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    api["11"] = {
        "class_type": "LTXVPreprocess",
        "inputs": {"image": ["10", 0], "img_compression": int(img_compression)},
    }

    video_latent: list[Any] = ["25", 0]
    if two_stage:
        api["5"] = {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": upscaler}}
        api["26"] = {
            "class_type": "LTXVLatentUpsampler",
            "inputs": {"samples": ["25", 0], "upscale_model": ["5", 0], "vae": ["3", 0]},
        }
        # The anchor is written in again at the delivered size, so the
        # refine does not drift off the first frame it was given.
        api["27"] = {
            "class_type": "LTXVImgToVideoInplace",
            "inputs": {
                "vae": ["3", 0],
                "image": ["10", 0],
                "latent": ["26", 0],
                "strength": float(refine_image_strength),
                "bypass": False,
            },
        }
        api["28"] = {
            "class_type": "SolidMask",
            "inputs": {"value": 0.0, "width": 1024, "height": 1024},
        }
        api["29"] = {
            "class_type": "SetLatentNoiseMask",
            "inputs": {"samples": ["18", 0], "mask": ["28", 0]},
        }
        api["30"] = {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {"video_latent": ["27", 0], "audio_latent": ["29", 0]},
        }
        api["31"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed) + 1}}
        api["32"] = {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["8", 0],
                "negative": ["8", 1],
                "cfg": float(cfg),
            },
        }
        api["33"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}}
        api["34"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": refine_sigmas}}
        api["35"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["31", 0],
                "guider": ["32", 0],
                "sampler": ["33", 0],
                "sigmas": ["34", 0],
                "latent_image": ["30", 0],
            },
        }
        api["36"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["35", 0]}}
        video_latent = ["36", 0]

    # ── Out. The soundtrack on the file is the customer's own audio, taken
    #    from the trim rather than decoded back from the latent.
    api["37"] = {
        "class_type": "VAEDecodeTiled",
        "inputs": {
            "samples": video_latent,
            "vae": ["3", 0],
            "tile_size": int(tile),
            "overlap": int(tile_overlap),
            "temporal_size": int(temporal_size),
            "temporal_overlap": int(temporal_overlap),
        },
    }
    api["38"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": ["37", 0], "fps": float(fps), "audio": ["15", 0]},
    }
    api["39"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["38", 0],
            "filename_prefix": filename_prefix,
            "format": "auto",
            "codec": "auto",
        },
    }
    return api


__all__ = [
    "AUDIO_VAE",
    "REQUIRED_NODES",
    "STAGE1_SIGMAS",
    "STAGE2_SIGMAS",
    "TEXT_ENCODER",
    "TRANSFORMER",
    "UPSCALER",
    "VIDEO_VAE",
    "base_canvas",
    "compile_a2v",
    "missing_nodes",
]
