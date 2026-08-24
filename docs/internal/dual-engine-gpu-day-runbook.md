# GPU day — the checklist

**Purpose:** the next session should execute, not design. Everything below is
decided; what is left is measurement.

**Branch:** `dual-engine-benchmark-prep`. **Nothing here deploys.** Production
routing is untouched and stays untouched until the decision table in §Phase 9
exists.

Companions: [ltx-h3-hybrid-benchmark.md](ltx-h3-hybrid-benchmark.md) (why the
hybrid is shaped the way it is), [golden-benchmark-pack.md](golden-benchmark-pack.md)
(the frozen inputs), [ltx-h3-comparison-framework.md](ltx-h3-comparison-framework.md)
(scoring and cases), [research-ltx25-zolexai-audit.md](research-ltx25-zolexai-audit.md)
(the verified LTX baseline).

---

## Before the card is switched on

The golden media does not exist yet. **Shoot or generate it first** — it is
the only remaining blocker that costs calendar time rather than GPU time, and
arriving at a rented GPU without it wastes the rental.

```bash
uv run python apps/worker/scripts/golden_pack.py --status     # 19 assets, all pending
# ... acquire, then for each file:
uv run python apps/worker/scripts/golden_pack.py --hash benchmarks/assets/<path>
# paste the hash into assets.manifest.json, set acquisition: acquired, record provenance
uv run python apps/worker/scripts/golden_pack.py --verify     # must exit 0
```

The song is the long pole: generate it with our own ACE-Step service so the
rights are unambiguous and the lyric sheet is known rather than transcribed.

---

## Phase 0 · Machine verification

Record every value; they go into every `RunRecord`.

```bash
nvidia-smi                      # GPU model, driver, total VRAM
nvcc --version                  # CUDA toolkit
free -g && df -h /workspace     # host RAM, disk (weights are ~100 GB across both engines)
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Gate: at least ~200 GB free disk, and VRAM at or above the purchased spec.

## Phase 1 · Restore and verify LTX

LTX is the baseline; nothing else matters if it moved.

```bash
cd /workspace/ltx2-benchmark && git log -1              # diff against upstream 400fd31
grep -c "materialize only the tensors" packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py   # must print 1
uv run python -c "import natten; print(natten.__version__)"
```

Then one known-good render and the golden argv check:

```bash
cd /workspace/zolexai/apps/worker
uv run python -m pytest tests/test_ltx_golden.py -q     # 11 shapes, must pass
uv run python scripts/ltx_smoke.py ...                  # one 5s T2V at 1024x576
```

**STOP if** the golden test fails, the render is green/black, or wall-clock is
far from the recorded ~34 s for 121 frames at 1024×576. Investigate before
anything H3 touches the card.

## Phase 2 · Install the H3 runtime

Do **not** guess the build. Decide it here, from what the purchased card
actually has, and record the decision.

- bf16 is ~110 GB (61.7 GB DiT + 48 GB Qwen3-VL-32B text encoder) and does not
  fit a 96 GB card. A quantized build is mandatory, not an optimisation.
- Official deployment paths: SGLang, vLLM, diffusers, ComfyUI. The
  service-shaped ones match how we already run ACE-Step.
- Sparse attention is not in the initial open-source release; full attention
  only.

Record exactly, into the result document: H3 model revision, quantization
build, runtime and version, CUDA, PyTorch, attention backend, and whether the
weights are the `fl2va` head, the `ref2va` head, or both.

**STOP if** the licence application has not been approved. This is a real
gate, but it turns on WHERE THIS GPU PHYSICALLY IS. The Applicable
Territory is worldwide **excluding** the EU, UK, South Korea and the US.
Outside those four, the community licence covers us by default; inside any of
them, an approved application is required first. Confirm the location from the
provider's machine record, not from IP geolocation or the billing country.
(Corrected 24 Aug 2026 — the earlier wording here was inverted. See
`h3-rtxpro6000-runtime-research.md` §1.1.)

## Phase 3 · H3 smoke

One of each, shortest supported length, before any comparison:

1. T2V (FL2VA, no images)
2. I2V (FL2VA, first frame)
3. Ref2VA with a subject image
4. Ref2VA with supplied audio in `fully_copy` — confirm the output audio IS
   the input waveform and that the mouth moves to it
5. Ref2VA video continuation

**STOP if** reference semantics differ from what
`worker/providers/h3_prompt.py` and `h3.py` assume. Fix the compiler first —
benchmarking a wrong request shape measures our bug, not the model.

## Phase 4 · Short A/B, LTX vs H3

Five seconds, one case per group, both engines, one run each. This is a
plumbing check, not a result: confirm both engines produce valid media, the
manifests match what the dry run predicted, and timings are in the expected
order of magnitude.

```bash
uv run python scripts/dual_engine_bench.py --dry-run --group A   # compare against reality
```

## Phase 5 · The high-value A/B/C

Three cells per case: LTX, H3, hybrid. Priority order — spend the budget top
down and stop when it runs out:

1. **D1–D5** reference-person V2V (the strongest a-priori H3 case)
2. **E1–E3** music video (lip-sync, `fully_copy` vs LTX a2vid)
3. **B1–B6, B8** image-to-video
4. **A2, A3, A5, A6** premium text-to-video

**STOP hybrid for a workflow** as soon as it is clearly behind direct H3 on
two consecutive cases. The question was never "can hybrid work" — it is
whether it earns a second inference pass.

## Phase 6 · Long form

30 s and 60 s from one global plan: A8, A9, B7, B8, H1, H2, E2, E3. Record
sections, seams, and per-section length for each engine — they differ by
design, and that difference is the finding.

## Phase 7 · Reliability

Repeat the shortlisted cells: 3 runs minimum, 5 on launch-critical paths.
Record the failure class every time, not just the count. A cell that passes
six times in isolation and fails five of fifteen under co-tenancy has already
happened here once.

## Phase 8 · Performance and cost

Fill in wall-clock, VRAM, host RAM and the model-switch penalty:

```text
LTX loaded → LTX generate → unload → H3 load → H3 generate
```

Measure the switch explicitly; a hybrid pays it on every job. Then compute
cost per 5 s / 30 s / 60 s / minute-of-music-video, per strategy, at the
actual GPU price. Keep quality and cost in separate columns.

## Phase 9 · The decision

Fill in the decision table. Every row cites quality, speed, VRAM, reliability
and cost. A mixed outcome is expected; "one model wins everything" should be
distrusted unless the table says so.

Only then: change `_AUTO_ROUTES`, with the evidence in the commit message.

---

## Stop conditions, in one place

| Condition | Action |
|---|---|
| An asset hash differs from the manifest | **Stop the comparison.** The inputs are not the same inputs. |
| The LTX golden argv test fails | **Stop.** The baseline moved; nothing measured against it means anything. |
| H3 repeatedly OOMs | **Stop tuning quality.** Fix runtime, build or memory first. |
| H3 reference semantics differ from our compiler | **Stop.** Correct the compiler, then re-run anything already measured. |
| Hybrid trails direct H3 on two consecutive cases in a workflow | **Stop hybrid for that workflow.** Spend the time on cells that can still change the answer. |
| A cell fails more than 1 run in 5 | Record it and treat reliability as the finding; do not average it away. |
| Licence not approved | **Stop.** H3 does not run. |

## The first ten commands

```bash
# 1  the pack must verify before anything is rendered
uv run python apps/worker/scripts/golden_pack.py --verify

# 2  machine
nvidia-smi && nvcc --version && free -g && df -h /workspace

# 3  LTX tree integrity
cd /workspace/ltx2-benchmark && git log -1 --oneline && \
  grep -c "materialize only the tensors" packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py

# 4  LTX argv baseline (no GPU needed, catches a moved baseline instantly)
cd /workspace/zolexai/apps/worker && uv run python -m pytest tests/test_ltx_golden.py tests/test_providers.py tests/test_hybrid.py -q

# 5  one real LTX render, 5s @1024x576 — confirm ~34s and real (not green) video
uv run python scripts/ltx_smoke.py --seconds 5 --width 1024 --height 576

# 6  what the benchmark expects, so reality can be compared against it
uv run python scripts/dual_engine_bench.py --dry-run --out /workspace/plans.json

# 7  empty result document for this session
uv run python scripts/dual_engine_bench.py --skeleton /workspace/results.json

# 8  install the H3 runtime — build chosen in Phase 2, then record it in results.json
#    (no command here on purpose: the build depends on the card you bought)

# 9  H3 smoke, shortest supported clip, one per head
#    T2V · I2V · Ref2VA subject · Ref2VA fully_copy audio · Ref2VA continuation

# 10 first real cell: D1, all three strategies, one run each
```
