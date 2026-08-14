# Issue: Triton neighbourhood-attention kernel fails on longer generations

**Opened:** 14 August 2026
**Status:** 🔴 **OPEN — live production impact**
**Severity:** High — customer-visible failures on the most-used workflow
**Box:** RTX PRO 6000 Blackwell (`sm_120`), instance `47698594`

---

## 1. Summary

The LTX video VAE's **fallback neighbourhood-attention path** crashes with a CUDA
error when decoding longer generations on this GPU. Short generations are
unaffected. The failure is in a *fallback* Triton kernel, which suggests the
optimised NATTEN kernel is missing on this box and was present on the RTX 5090
it replaced.

```text
RuntimeError: Triton Error [CUDA]: invalid argument
```

---

## 2. Customer impact

With production's current `LTX_MAX_SECONDS=30`:

| Workflow | Aspect | Duration | State |
|---|---|---|---|
| Text to Video | 16:9 | 5s / 10s / 15s | ✅ works |
| Text to Video | 16:9 | **30s / 60s** | ❌ **fails** |
| Text to Video | 9:16 | 5s / 10s | ✅ works |
| Text to Video | 9:16 | **15s / 30s / 60s** | ❌ **fails** |
| Text to Video | 1:1, 4:5 | all | ❓ **unmeasured** |
| Image to Video / Extend Video | all | 30s+ | ❌ **presumed failing** — same code path, unconfirmed |
| Music Video | all | any | ❌ **fails** — un-routed to `mock` 14 Aug as mitigation |
| Video to Video | — | — | ⚠️ untested since migration; worked on the 5090 |
| Music (audio) | — | — | ✅ unaffected, different runtime |

**Portrait is worse than landscape** — 9:16 breaks from 15s upward, so the
customer-visible surface is larger than "long videos are broken". The client uses
9:16 specifically.

**These are live.** A customer selecting 30s or 60s on zolexai.com gets a failed
generation. This was introduced by the 5090 → PRO 6000 migration and went
unnoticed because the migration parity test used **10s**, which passes.

---

## 3. Reproduction

On the GPU box:

```bash
cd /workspace/zolexai/apps/worker
LTX_REPO_DIR=/workspace/ltx2-benchmark LTX_QUANTIZATION=nvfp4-prequant \
LTX_MAX_SECONDS=30 DURATION=30s ASPECT_RATIO=16:9 \
/workspace/zolexai/.venv-worker/bin/python scripts/ltx_smoke.py a koi pond at dawn
```

Substituting `DURATION=10s` passes. This bypasses the API, database, storage and
routing entirely, so it isolates the model.

---

## 4. Evidence

All measured 14 Aug 2026 on `nvfp4-prequant`, via `scripts/ltx_smoke.py`.

### 4.1 The limit is DIMENSION-driven, not duration-driven

Single-pass ceilings differ per aspect ratio, at **identical pixel counts**:

| Aspect | Frame | 5s | 10s | 15s | 20s | 30s |
|---|---|---|---|---|---|---|
| 16:9 | 896×512 | ✅ | ✅ | ✅ | ✅ | ❌ |
| 9:16 | 512×896 | ✅ | ✅ | ❌ | — | ❌ |
| 1:1 | ? | — | — | — | — | — |
| 4:5 | ? | — | — | — | — | — |

896×512 and 512×896 hold the same number of pixels but have **different
ceilings**. `na3d` derives its launch grid from the spatial dimensions
individually, so the taller frame trips the limit sooner. 1:1 and 4:5 are
offered by the product and **have not been measured**.

This is why `LTX_MAX_SECONDS` — a single global per-pass value — must be set to
satisfy the *worst* aspect ratio, not the one that happens to be tested.

### 4.2 Conditioned passes have a lower ceiling than fresh ones

| Request | Ceiling | Pass plan | Result |
|---|---|---|---|
| 20s single, 16:9 | 20 | one fresh 20s | ✅ |
| 30s, 16:9 | 20 | two × 15s | ✅ |
| 60s, 16:9 | 20 | three × 20s | ❌ **section 2** |
| 60s, 16:9 | 15 | four × 15s | ✅ (59.9s, 148s wall) |

A fresh 20s pass succeeds; a *continuation* 20s pass fails. Chained sections
carry conditioning frames from the previous section, so their input is larger
than a fresh generation of the same length. **Testing a duration standalone does
not prove it works as a chained section.**

### 4.3 Two distinct CUDA errors

```text
Triton Error [CUDA]: invalid argument                    ← fresh pass over the limit
Triton Error [CUDA]: an illegal memory access was encountered  ← conditioned pass (60s §2)
```

Both originate from the same `_na3d_kernel[grid](` launch. Whether they are the
same underlying defect is unconfirmed.

### 4.4 Failing call stack, innermost frames

```text
ltx_core/model/video_vae/transformer/chunked/attn.py:157   _attn_on_chunk_impl
ltx_core/model/video_vae/transformer/fallback_na/__init__.py:128  __call__
ltx_core/model/video_vae/transformer/fallback_na/__init__.py:80   na_attention_triton
ltx_core/model/video_vae/transformer/fallback_na/triton_na.py:153 _na3d_kernel[grid](
triton/backends/nvidia/driver.py:328                       self.launch(gridX, gridY, gridZ, …)
RuntimeError: Triton Error [CUDA]: invalid argument
```

Two things the stack tells us:

- It is in the **video VAE decode**, not the diffusion transformer.
- The module is named **`fallback_na`** — this is the path taken when the
  optimised neighbourhood-attention kernel is unavailable.

`invalid argument` from a Triton grid launch typically means a launch dimension
exceeds a device limit — consistent with "works at small sizes, fails at large".

---

## 5. Music Video — largely explained, one gap left

Music Video failed generating **section 1 of 2** from a 29-second track, a pass
of roughly 15 seconds. That looked anomalous when 15s passed for Text to Video —
but §4.1 resolves most of it: 15s only passes at **16:9**, and it is not
established which dimensions Music Video derives. If it renders portrait or a
non-16:9 grid, a 15s pass failing is exactly consistent with the table above.

Remaining gap: Music Video's output dimensions were never captured from the
failing run. Confirm them before assuming it is fully explained — and note it
also conditions on audio, which may alter the latent shape independently of the
frame size.

---

## 6. Why the 5090 did not show this

The 5090 ran 30s single-pass successfully (recorded: 30s max, 60s hard-OOM). It
therefore never entered this fallback, which implies NATTEN was installed there —
either from the base image or as a side effect of that box's setup history. The
PRO 6000 was built from scratch and only `ltx-kernels` (the NVFP4 extension) was
compiled; **NATTEN was never installed**. See `gpu-worker-runbook.md` §34.2.

---

## 7. Options

### 7.1 Workaround — lower `LTX_MAX_SECONDS` (proven for 16:9, pending for portrait)

`LTX_MAX_SECONDS` is the **per-pass** ceiling; the chaining layer
(`worker/longform/chain.py`) already splits anything longer. One environment
variable on the worker, no API rebuild, **no durations removed from the product**.

Proven: `LTX_MAX_SECONDS=15` gives a working 60s at 16:9 (four passes, 148s wall).

**But it is not sufficient.** 9:16 fails at a 15s pass, so a global 15 leaves
portrait broken at 30s and 60s. The value must satisfy the worst aspect ratio the
product offers, and 1:1 / 4:5 are still unmeasured. If portrait tops out at 10s:

| Request | Passes at ceiling 10 | Seams | Est. wall |
|---|---|---|---|
| 30s | 3 | 2 | ~2 min |
| 60s | 6 | 5 | ~3–4 min |

That is a materially different product from the 5090's single-pass 30s. **The
cost of this workaround scales with how low the ceiling has to go**, which is the
main argument for pursuing 7.2 rather than settling here.

Does **not** fix Music Video (see §5).

### 7.2 Proper fix — install NATTEN (medium confidence)

If NATTEN builds for `sm_120`, the fallback is never used and the bug disappears.
Unknown until attempted: whether the LTX repo exposes it as a dependency group,
and whether it has Blackwell support.

### 7.3 Triton version (lower confidence)

The kernel may fail due to a Triton/driver combination on Blackwell rather than
anything about the shape. Bumping or pinning Triton is cheap to try.

### 7.4 Upstream

Worth reporting to Lightricks if 7.2 and 7.3 both fail — a fallback kernel that
cannot launch on a current-generation card is an upstream defect.

---

## 8. Actions taken

| When | What |
|---|---|
| 14 Aug 2026 | Music Video routed back to `runtime: mock` on the VPS, API rebuilt |
| 14 Aug 2026 | 16:9 ceiling established: 20s fresh passes, 30s fails |
| 14 Aug 2026 | Conditioned-pass effect identified (§4.2) |
| 14 Aug 2026 | 9:16 ceiling established: 10s passes, 15s fails |
| 14 Aug 2026 | `LTX_MAX_SECONDS=15` proven for 60s @ 16:9 — **not applied**, insufficient for portrait |

**Still open:**

- `LTX_MAX_SECONDS` **not changed** on the running worker — production still
  fails at 30s/60s in every aspect ratio
- **1:1 and 4:5 ceilings unmeasured** — both are offered to customers
- Chaining **not verified at whatever the final ceiling turns out to be**, in the
  worst-case aspect (this is the combination that failed at 20)
- **Image to Video and Extend Video not confirmed** — same code path, presumed
  affected
- **Music Video output dimensions not captured** (§5)
- **NATTEN not attempted** (§7.2)

---

## 9. Lessons

**A parity test at one duration and one aspect ratio proved much less than it
appeared to.** The migration was validated with a single 10s 16:9 clip, and the
box was declared good. The product offers five durations × four aspect ratios;
one cell of a twenty-cell matrix was checked, and it happened to be a passing one.

Worse, each narrowing step found the previous conclusion incomplete:

1. "30s is broken" → actually the whole duration ladder needed testing
2. "`LTX_MAX_SECONDS=20` fixes it" → failed at 60s, because conditioned passes
   have a lower ceiling than fresh ones
3. "`LTX_MAX_SECONDS=15` fixes it" → failed at 9:16, because the ceiling is
   dimension-dependent

**After any hardware change, exercise the product's actual parameter matrix** —
durations *and* aspect ratios — not one representative sample. The failure mode
here is size-dependent by construction, and no single sample can see it.
