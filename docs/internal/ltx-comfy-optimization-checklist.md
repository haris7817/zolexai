# LTX 2.5 speed optimisation — checklist and benchmark protocol

**Phase 5 of the final milestone. STATUS: WAITING FOR GPU VALIDATION.**
Nothing here was run. The GPU node is unavailable (Sep 2026), so this
document prepares the work and states the rule it must follow: every change
ships with a before/after pair — runtime AND quality — on the same seed,
the same prompt and the same inputs, or it does not ship.

The client's graphs are the source of truth. An optimisation that changes a
graph's sampler, schedule, LoRA set or model files is a deviation, and a
deviation is an experiment behind an `execution` key that a deployment
chooses — never a silent default change.

---

## 0. The baseline, first

Before any optimisation, `scripts/ltx_comfy_bench.py` runs the matrix once
as shipped and writes `benchmarks/results/ltx25/<stamp>/results.{json,md}`:

| Cell | Runs |
|---|---|
| T2V 5/10/15/30 s × 16:9, 9:16, 1:1 | 12 |
| FLF first-only 5/10/15/30 s × 16:9 | 4 |
| FLF first+last 10 s | 1 |
| Character replacement, the ZIP sample source + one photo | 1 |
| Extend 30 s → +30 s (continuation.json records the seam) | 1 |

Each run records resolution, fps, output duration, wall-clock, VRAM peak and
mean, RAM peak. The pack's own samples (30 s T2V, 30 s FLF, 8 s replacement)
are the quality references for a human viewing.

## 1. Candidates, cheapest and least invasive first

| # | Lever | Where | What changes | Risk to quality | Measure |
|---|---|---|---|---|---|
| 1 | **Warm models between jobs** (already the default: `ltx_comfy_free_after_job=false`) | worker | No reload of ~30 GB of weights per job | none | first-job vs second-job wall on the same cell |
| 2 | **Stop ACE-Step during video jobs** / evict on engine switch | node | Frees ~24 GB; removes the measured 1-in-3 CUBLAS workspace failure class on the CLI tier | none | VRAM headroom; failure rate over 20 runs |
| 3 | **Attention backend** — the character graph already carries `ModelAttentionBackend` ("comfy kitchen attention"); T2V/FLF do not | graph-adjacent (experiment key) | sage / flash attention if the installed torch supports it on this card | low, but measure | wall; PSNR/SSIM against baseline at the same seed |
| 4 | **Stage-1 sigma count** (T2V/FLF run 8 + 3 sigmas) | graph deviation | fewer stage-1 steps | real — the schedule is the pack's | wall; side-by-side viewing; never below the pack's without a recorded verdict |
| 5 | **Canvas megapixels** (`ResolutionSelector` 0.9 MP) | graph deviation | 0.6 MP draft tier for previews | real | wall vs resolution; only as an explicit "draft" tier if the product wants one |
| 6 | **Q8 GGUF vs INT8 convrot transformer** for T2V/FLF | graph deviation | the INT8 file the character graph already loads | unknown | wall + VRAM + viewing; the pack chose Q8 for the T2V/FLF graphs |
| 7 | **FP8 / NVFP4** | graph deviation | quantised weights | unknown; the CLI runtime's NVFP4 tier is measured on a different pipeline | as above |
| 8 | **CPU offload / VRAM management** (`--lowvram`, `--reserve-vram`) | ComfyUI launch flags | trade wall for headroom on a shared card | none | wall; OOM rate |
| 9 | **Detailer LoRA at 0.3** off | graph deviation | one fewer LoRA in the stage-2 pass | real (detail) | wall; viewing |
| 10 | **Continuation overlap** (`SEAM_OVERLAP_FRAMES`) | engine | more overlap frames dropped per seam if the model's first frames flicker | none to length; seams | seam inspection at 24 fps |

Levers 1, 2 and 8 change nothing about the graphs and are the first to run.
Levers 3 and 6–7 are model-side and need a side-by-side. Levers 4, 5 and 9
change the pack's own settings and are experiments only.

## 2. The protocol for one lever

```
before:  bench cell × 3 runs, same seed, as shipped        → wall_b, vram_b, file_b
after:   bench cell × 3 runs, same seed, lever applied     → wall_a, vram_a, file_a
quality: PSNR/SSIM(file_a, file_b) + a human verdict on a 3-clip side-by-side
ship if: wall_a < wall_b by ≥ 15% AND the verdict is "no visible loss"
record:  docs/internal/ltx-comfy-optimization-results.md, one row per lever
```

A lever with a visible loss may still ship as a NAMED tier the customer
chooses (as Fast/Best once were), never as the default.

## 3. What is already measured on this box (other pipeline — not transferable)

From the CLI runtime (`ltx.py`), for context only: `--offload none` on the
unquantized tiers was 23–30 % faster once VRAM headroom existed; 15 steps on
the audio tier were 37 % cheaper; the guided tier cost 4.3× the distilled
one. None of these numbers describe the ComfyUI graphs.

## 4. Output of Phase 5 when the GPU returns

* `benchmarks/results/ltx25/<stamp>/results.md` — the baseline table.
* `docs/internal/ltx-comfy-optimization-results.md` — one row per lever
  with before/after runtime and the quality verdict.
* `ltx_comfy_expected_wall_per_output_second` set from the baseline so the
  progress bar paces honestly.
