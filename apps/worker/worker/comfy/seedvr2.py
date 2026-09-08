"""SeedVR2 — a temporal AI upscale of a finished clip, as a ComfyUI prompt.

The client's review of the 720p path (8 Sep 2026): the picture was right,
the speed was right, and "the 1080 conversion uses ordinary Lanczos scaling,
not a temporal AI upscaler". This is that upscaler. SeedVR2 (ByteDance,
ICLR 2026) is a one-step video restoration model that ComfyUI supports
natively; the graph here is ComfyUI's own `utility_seedvr2_3b_int8_upscale_
video` template, flattened by hand, with two changes: the trim switch is
gone, and an `ImageScale` lands the result on the exact delivered size.

Why "temporal": the model sees runs of frames together (`SeedVR2TemporalChunk`
/ `TemporalMerge` split a long clip into overlapping chunks and crossfade
them back), so what it adds is consistent from frame to frame rather than
re-invented per frame the way a still-image upscaler would.

The soundtrack rides through untouched: `GetVideoComponents` separates it
from the frames at the start and `CreateVideo` puts the same object back at
the end. Nothing in between sees audio.

Weights, exactly the two files ComfyUI's template names:
  models/diffusion_models/seedvr2_3b_int8_convrot.safetensors
  models/vae/seedvr2_ema_vae_fp16.safetensors
"""

from __future__ import annotations

from typing import Any

DIFFUSION_MODEL = "seedvr2_3b_int8_convrot.safetensors"
VAE = "seedvr2_ema_vae_fp16.safetensors"


def multiplier_for(source: tuple[int, int], target: tuple[int, int]) -> float:
    """The scale that makes the source cover the target on both sides; the
    closing centre-crop trims the rest. 1280x704 → 1920x1080 is 1.5341: the
    height is the binding side, and 14 px of width come off each edge."""
    sw, sh = source
    tw, th = target
    return round(max(tw / sw, th / sh), 4)


def compile_upscale(
    *,
    input_file: str,
    source: tuple[int, int],
    target: tuple[int, int],
    filename_prefix: str,
    seed: int = 0,
    diffusion_model: str = DIFFUSION_MODEL,
    vae: str = VAE,
    temporal_chunks: bool = True,
    tile: int = 512,
    tile_overlap: int = 128,
    temporal_size: int = 64,
    temporal_overlap: int = 8,
) -> dict[str, Any]:
    """The API prompt: load `input_file` from ComfyUI's input folder, upscale,
    deliver `target` with the original soundtrack under `filename_prefix`.

    `temporal_chunks` is the template's optional chunking, on by default here
    because the clips this serves are 121–721 frames at 1080p and a single
    latent of that size is what runs the card out of memory.
    """
    scale = multiplier_for(source, target)
    width, height = target
    api: dict[str, Any] = {
        "load": {"class_type": "LoadVideo", "inputs": {"file": input_file}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": ["load", 0]}},
        "resize": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {
                "input": ["parts", 0],
                "resize_type": "scale by multiplier",
                "resize_type.multiplier": scale,
                "scale_method": "lanczos",
            },
        },
        "pre": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["resize", 0]}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_model, "weight_dtype": "default"},
        },
        "enc": {
            "class_type": "VAEEncodeTiled",
            "inputs": {
                "pixels": ["pre", 0], "vae": ["vae", 0],
                "tile_size": tile, "overlap": tile_overlap,
                "temporal_size": temporal_size, "temporal_overlap": temporal_overlap,
            },
        },
    }
    if temporal_chunks:
        api["chunk"] = {
            "class_type": "SeedVR2TemporalChunk",
            "inputs": {"latent": ["enc", 0], "temporal_overlap": 0, "chunking_mode": "auto"},
        }
        latent_in: list[Any] = ["chunk", 0]
    else:
        latent_in = ["enc", 0]
    api["cond"] = {
        "class_type": "SeedVR2Conditioning",
        "inputs": {"model": ["unet", 0], "vae_conditioning": latent_in},
    }
    api["sample"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["unet", 0], "positive": ["cond", 0], "negative": ["cond", 1],
            "latent_image": latent_in, "seed": int(seed), "steps": 1, "cfg": 1,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
        },
    }
    if temporal_chunks:
        api["merge"] = {
            "class_type": "SeedVR2TemporalMerge",
            "inputs": {"latents": ["sample", 0], "temporal_overlap": ["chunk", 1]},
        }
        decoded_from: list[Any] = ["merge", 0]
    else:
        decoded_from = ["sample", 0]
    api.update({
        "dec": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": decoded_from, "vae": ["vae", 0],
                "tile_size": tile, "overlap": tile_overlap,
                "temporal_size": temporal_size, "temporal_overlap": temporal_overlap,
            },
        },
        "post": {
            "class_type": "SeedVR2PostProcessing",
            "inputs": {
                "images": ["dec", 0], "original_resized_images": ["resize", 0],
                "color_correction_method": "none",
            },
        },
        "fit": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["post", 0], "upscale_method": "lanczos",
                "width": int(width), "height": int(height), "crop": "center",
            },
        },
        "video": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["fit", 0], "audio": ["parts", 1],
                "fps": ["parts", 2], "bit_depth": ["parts", 3],
            },
        },
        "save": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["video", 0], "filename_prefix": filename_prefix,
                "format": "auto", "format.codec": "auto",
            },
        },
    })
    return api


__all__ = ["DIFFUSION_MODEL", "VAE", "compile_upscale", "multiplier_for"]
