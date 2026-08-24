# MiniMax H3 on the RTX PRO 6000 WS — runtime research

**Date:** 24 August 2026 · **No GPU work performed.** Every figure below comes
from a primary source, cited inline. Nothing here is measured; measurement is
Phase 12 and it has not run.

**Target hardware, as reported by the provider:**
95.6 GB VRAM · 128 GB host RAM · 879 GB disk · PCIe 5.0 x16 · CUDA 12.8.1 image.

---

## 1. Two corrections to our own frozen documents

Both were found by reading primary sources rather than trusting the freeze.

### 1.1 The licence territory is INVERTED in our documents

Our freeze manifest, `h3-pre-gpu-integration.md`, and the runbook all say the
open weights are *"limited to the EU, UK, South Korea and the US"* and require
an approved application. **That is backwards.**

The official licence text says:

> "Applicable Territory" means worldwide, excluding the Excluded Territories.

and the European Union, United Kingdom, Republic of Korea and United States are
the **Excluded** Territories — i.e. everywhere *except* those four is licensed
by default. ([LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE),
effective 2 August 2026.)

The ambiguity came from MiniMax's own FAQ, which says the licence is
"currently limited to the EU, UK, South Korea, and US" — a sentence that reads
either way in isolation. The next sentence settles it:

> Organizations in these regions can apply for a formal license. After
> reviewing the deployment scenario and confirming that appropriate compliance
> controls and safeguards are implemented, MiniMax may authorize usage.

Organizations *inside* a permitted territory would not need to apply. The four
named regions are the restricted ones.
([QA-about-License.md](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md))

Independent reporting agrees, and gives the reason: the US exclusion stems from
MiniMax's copyright litigation with Hollywood studios.
([The Batch](https://www.deeplearning.ai/the-batch/minimaxs-state-of-the-art-video-model-is-only-minimally-open))

**Consequence — the gate reverses.** The runbook's Phase 2 STOP ("stop if the
licence application has not been approved") is wrong for most of the world and
insufficient for the US:

| Physical GPU location | Status |
|---|---|
| Outside EU / UK / South Korea / US | **Licensed by default.** No application. H3 may run. |
| Inside any of those four | **Prohibited** without approved written authorisation. |

**This makes the physical location of the rented box a hard gate**, and it must
come from the provider's own machine record, not from IP geolocation and not
from the account's billing country.

Two further licence terms that bind us regardless of territory:

- **Revenue threshold.** Above ~US$20M annual revenue, commercial use needs
  separate written authorisation from MiniMax.
- **No model improvement.** "You may not use the MiniMax H3 Works or any of
  their Outputs or results to improve any other artificial intelligence model."
  Our hybrid runs LTX → H3, so no H3 output feeds LTX — but this also forbids
  using H3 output as training or tuning data for anything of ours, ever.
- **Attribution.** A "Powered by MiniMax H3" notice is encouraged and a licence
  notice file is required in any product built on it.

### 1.2 BF16 does not fit — but by more than we said, and it may not matter

Our documents estimate BF16 at "~110 GB (61.7 GB DiT + 48 GB text encoder)".
The actual repository contents:

| Component | Shards | Size |
|---|---:|---:|
| Transformer / DiT | 14 | **64.5 GB** |
| Text encoder (Qwen3-VL-32B) | 14 | **68.0 GB** |
| Visual VAE | 1 | 10.42 GB |
| Audio VAE | 1 | 0.61 GB |
| **All-resident BF16** | | **≈ 143.5 GB** |

([HF model index](https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3?blobs=true))

So co-residency is further out of reach than we thought — 143.5 GB against
95.6 GB. **But co-residency is not how this model runs.** The text encoder
produces embeddings once per generation and is then dead weight for the whole
denoising loop. Staged, the peak is per-stage:

```text
stage          resident                    VRAM      fits 95.6 GB
text encode    Qwen3-VL-32B  68.0 GB       68.0 GB   yes, 27.6 GB spare
denoise        DiT           64.5 GB       64.5 GB + activations
VAE decode     visual VAE    10.4 GB       10.4 GB   comfortably
```

This is precisely the configuration SGLang documents for consumer cards, and
**our card is on its supported list by name.**

## 2. Runtime evidence

The SGLang cookbook is the most specific official deployment documentation and
it addresses this hardware directly.
([SGLang · MiniMax-H3](https://lmsysorg.mintlify.app/cookbook/diffusion/MiniMax/MiniMax-H3))

**Supported hardware includes `RTX PRO 6000 (96GB)` explicitly**, alongside
B200/B300/H200/H100/MI300X and consumer cards down to 12 GB.

Documented consumer recipe:

```bash
--performance-mode memory \
--layerwise-offload-components dit,text_encoder,vae \
--layerwise-resident-layers video_vae=36
```

Datacenter alternative: `--performance-mode speed` (fully resident).

Other recorded facts:

- Native precision **BF16 / FP32** mixed — transformer BF16, projections FP32.
- **Online FP8** quantization exists, but verification coverage is stated as
  limited to "resident single-node B200/B300" — *not* validated on our card.
- Host RAM tiers offered: 32 GB · 48–64 GB · **96+ GB**. We have 128 GB, above
  the top tier. Host RAM determines whether DiT weights sit in pinned memory or
  are mapped from the checkpoint.
- Attention: FlashAttention default on NVIDIA; SageAttention alternative.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` recommended against
  fragmentation during decode.
- TP 1/2/4/8 with Ulysses and ring degrees; irrelevant at one GPU (TP=1).
- Both heads supported. Ref2VA modes include image+audio and video+soundtrack —
  the paths our music-video and reference-person cases need.

## 3. Provisional answer to the machine question

**The 128 GB host RAM worry looks survivable, and quantization looks optional
rather than mandatory.** That is better than the freeze predicted, and it is
still a hypothesis:

| Question | Provisional | Confidence |
|---|---|---|
| Does BF16 fit fully resident? | **No** — 143.5 GB vs 95.6 GB | High: arithmetic on published file sizes |
| Does BF16 fit staged with layerwise offload? | **Probably yes** | Medium: it is the documented consumer recipe, our card is named, per-stage peaks fit |
| Is 128 GB host RAM enough? | **Probably, with little margin** | Low–medium: evicted text encoder + VAE ≈ 79 GB pinned, before OS, runtime and video buffers |
| Is quantization required? | **No, likely an optimisation** | Medium |
| Should we use FP8? | **Not initially** | High: unverified outside B200/B300, and it would confound quality with fit |

The low-confidence row is the one that decides whether we keep the instance,
and it is measurable in under an hour by loading the model and watching RSS —
Phase 12, before any render.

## 4. Disk budget — the constraint nobody costed

The full repository is **≈ 354 GB**, because the `fl2va` and `ref2va` pipeline
folders each carry their own copies of the shared transformer, text encoder and
VAEs — roughly 142 GB of pure duplication.

Naively cloning it consumes 40% of the 879 GB disk, and a HuggingFace cache that
copies rather than symlinks would double that to ~708 GB and leave nothing for
LTX, outputs or intermediates.

```text
full repo, naive                      354 GB
  + HF cache duplication (worst)      708 GB   ← disk exhausted
selective (shared components once)    ~144 GB
  + LTX source, weights, env           ~40 GB
  + benchmark outputs/intermediates     ~80 GB
  + ACE-Step, logs, headroom            ~60 GB
                                      -------
                                      ~324 GB, leaving ~555 GB
```

**Therefore:** fetch with explicit `allow_patterns`, pull the shared components
once, set `HF_HOME=/workspace/cache/huggingface` on the same filesystem as the
model directory so the cache can symlink, and never `snapshot_download` the
whole repo. Download time is not the issue — at the reported 1.7 Gbps, 144 GB is
well under an hour, or roughly a dollar of rental.

## 5. Runtime decision matrix — provisional

Nothing has been installed, so no row is measured.

| Runtime | FL2VA | Ref2VA | fully_copy | 96 GB viable | 128 GB RAM | Blackwell | Integration | Verdict |
|---|---|---|---|---|---|---|---|---|
| **SGLang** | yes | yes | yes | **documented for this card** | 96+ tier documented | listed | service, like ACE-Step | **first choice** |
| vLLM | yes | yes | ? | not documented for this card | ? | ? | service | second |
| Diffusers | yes | yes | ? | manual offload | ? | ? | in-process | fallback / reference |
| ComfyUI | partial | partial | ? | community offload | ? | ? | poor — graph, not API | not for benchmarking |

SGLang leads on evidence, not familiarity: it is the only one whose official
documentation names our exact card, states host-RAM tiers, and gives a
layerwise offload recipe. It is also service-shaped, which matches how we
already run ACE-Step and suits the sequential model-switching in Phase 18.

**Community quantizations were found and are rejected for the baseline** —
NVFP4, INT4/INT8 and GGUF repackagings exist from third parties. None is
official, none has stated provenance against the base weights, and benchmarking
a community requantization would measure that repackager's choices rather than
H3. They remain available if BF16 staged loading fails.

## 6. What must be answered before anything is installed

1. **Where is this GPU physically?** Licence-critical, and now the *first*
   question rather than a footnote. From the provider's machine record.
2. Which head do we need first — `ref2va` alone covers D-group and E-group, the
   two highest-value comparisons. `fl2va` can wait.
3. Does the SGLang build support this CUDA 12.8.1 image, or does it want its own?

## 7. Sources

- [MiniMax-H3 LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE) — territory, revenue threshold, model-improvement clause, attribution
- [QA-about-License.md](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md) — application process for excluded regions
- [MiniMax-H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3) — 33B Omni-Transformer, Qwen3-VL-32B encoder, 4–15 s, 24 fps, 32 kHz stereo, aspect ratios, BF16 CFG-distilled
- [HF model index (blobs)](https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3?blobs=true) — per-shard file sizes
- [SGLang cookbook · MiniMax-H3](https://lmsysorg.mintlify.app/cookbook/diffusion/MiniMax/MiniMax-H3) — supported hardware, offload recipe, host RAM tiers, attention backends, FP8 coverage
- [The Batch — MiniMax's video model is only minimally open](https://www.deeplearning.ai/the-batch/minimaxs-state-of-the-art-video-model-is-only-minimally-open) — corroboration and the litigation rationale
