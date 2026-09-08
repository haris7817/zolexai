# LTX-2.5 setup for the direct renderer

The worker's `ltx` renderer calls the official guided, two-stage audio-to-video module. This module uses the full development transformer in the first stage and the matching distilled LoRA in the second stage. Do not substitute the distilled transformer for the development transformer in this adapter.

The commands below mirror the official LTX repository layout. Review and accept the model terms before downloading. The easiest supported path is the packaged installer, which downloads these exact files together with Whisper and Qwen, verifies them, and writes the model-path environment file:

```bash
python -m pip install '.[models]'
hf auth login
zolex-music-video install-models --models-root /models
```

See `docs/MODEL-INSTALLER.md`. The manual equivalent is retained below for administrators who manage model storage separately.

```bash
cd /srv
git clone https://github.com/Lightricks/LTX-2.git ltx
cd /srv/ltx
uv sync --extra natten
hf auth login
hf download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
  --local-dir /models/ltx-2.5
```

Then set:

```text
LTX_PYTHON=/srv/ltx/.venv/bin/python
LTX_MODULE=ltx_pipelines.a2vid_two_stage
LTX_TRANSFORMER_PATH=/models/ltx-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors
LTX_TEXT_ENCODER_PATH=/models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
LTX_VIDEO_VAE_PATH=/models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors
LTX_AUDIO_VAE_PATH=/models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors
LTX_SPATIAL_UPSAMPLER_PATH=/models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
LTX_DISTILLED_LORA_PATH=/models/ltx-2.5/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors
```

Official sources:

- LTX repository and model list: https://github.com/Lightricks/LTX-2
- Audio-to-video implementation: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py
- LTX installation notes: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/docs/installation.md

Run `zolex-music-video doctor` after setting the environment. The command validates FFmpeg and reports all required LTX paths. A real GPU smoke test is still necessary because host CUDA, PyTorch, attention backend, VRAM, and model compatibility cannot be verified from the source package alone.
