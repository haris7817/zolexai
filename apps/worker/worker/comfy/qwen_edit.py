"""Anchor stills for the music-video worker, as a ComfyUI prompt (8 Sep 2026).

The client's music-video package renders every shot from a starting picture
it calls an anchor: a still that already shows the assigned performer(s) —
their real faces, from the customer's reference photos — composed in the
shot's location and light, at the LTX working size. Their contract for the
anchor generator is one command that reads a request (prompt, size,
reference paths) and writes a PNG; which image model does it is the
deployer's choice, and this node had none.

The model is Qwen-Image-Edit-2509 (Alibaba, Apache-2.0), which ComfyUI runs
natively: it takes up to three reference pictures and a text instruction and
produces a new composition that keeps the people in the references as they
are — the one capability an anchor needs that a text-only model cannot give.
With no references it is a text-to-image model, which is how the package's
fictional identity portraits are made. The graph is ComfyUI's own
`image_qwen_image_edit_2509` template flattened by hand, with the Lightning
4-step LoRA so a still costs seconds rather than a minute.

Weights (ComfyUI's `models/` folder; installed on the node 8 Sep 2026):
  diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors     (Comfy-Org)
  text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors               (Comfy-Org)
  vae/qwen_image_vae.safetensors                                    (Comfy-Org)
  loras/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors (lightx2v)
"""

from __future__ import annotations

from typing import Any

DIFFUSION_MODEL = "qwen_image_edit_2509_fp8_e4m3fn.safetensors"
TEXT_ENCODER = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
LIGHTNING_LORA = "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"

#: The edit node takes at most three pictures.
MAX_REFERENCES = 3

#: The node classes the prompt uses — checked against the server's catalogue
#: before submission so a ComfyUI without Qwen support fails with a message
#: rather than a validation dump.
REQUIRED_NODES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "LoraLoaderModelOnly",
    "ModelSamplingAuraFlow",
    "CFGNorm",
    "TextEncodeQwenImageEditPlus",
    "EmptySD3LatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
    "LoadImage",
)


def compile_anchor(
    *,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    filename_prefix: str,
    references: list[str] = (),
    negative: str = "",
    steps: int = 4,
    cfg: float = 1.0,
    shift: float = 3.0,
    lightning: bool = True,
    diffusion_model: str = DIFFUSION_MODEL,
    text_encoder: str = TEXT_ENCODER,
    vae: str = VAE,
    lora: str = LIGHTNING_LORA,
) -> dict[str, Any]:
    """The API prompt for one still.

    `references` are file names already inside ComfyUI's `input/` folder
    (at most `MAX_REFERENCES`); each feeds one image slot of the edit
    encoder, for the positive AND the negative conditioning, as the
    template does. With no references the same graph is text-to-image.
    """
    if len(references) > MAX_REFERENCES:
        raise ValueError(f"at most {MAX_REFERENCES} references, got {len(references)}")
    if width % 16 or height % 16:
        raise ValueError(f"anchor size must be a multiple of 16, got {width}x{height}")

    api: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_model, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": text_encoder, "type": "qwen_image", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    }
    model: list[Any] = ["1", 0]
    if lightning:
        api["4"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model, "lora_name": lora, "strength_model": 1.0},
        }
        model = ["4", 0]
    api["5"] = {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {"model": model, "shift": float(shift)},
    }
    api["6"] = {"class_type": "CFGNorm", "inputs": {"model": ["5", 0], "strength": 1.0}}

    image_inputs: dict[str, Any] = {}
    for index, name in enumerate(references, start=1):
        node_id = str(20 + index)
        api[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        image_inputs[f"image{index}"] = [node_id, 0]

    api["7"] = {
        "class_type": "TextEncodeQwenImageEditPlus",
        "inputs": {"clip": ["2", 0], "prompt": prompt, "vae": ["3", 0], **image_inputs},
    }
    api["8"] = {
        "class_type": "TextEncodeQwenImageEditPlus",
        "inputs": {"clip": ["2", 0], "prompt": negative, "vae": ["3", 0], **image_inputs},
    }
    api["9"] = {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": int(width), "height": int(height), "batch_size": 1},
    }
    api["10"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["6", 0],
            "positive": ["7", 0],
            "negative": ["8", 0],
            "latent_image": ["9", 0],
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    }
    api["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}}
    api["12"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["11", 0], "filename_prefix": filename_prefix},
    }
    return api


def missing_nodes(catalogue: dict[str, Any] | set[str]) -> list[str]:
    """Node classes the prompt needs that the server does not offer."""
    names = set(catalogue.keys()) if isinstance(catalogue, dict) else set(catalogue)
    return [node for node in REQUIRED_NODES if node not in names]
