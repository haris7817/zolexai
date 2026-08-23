# Pre-GPU freeze manifest

**Date:** 24 August 2026 · **Verified by:** a full audit-verify-freeze pass, no
redesign. This is the definitive handoff point: everything below was measured
on the day, not carried forward from a previous session's notes.

The next meaningful work requires a real GPU.

---

## 1. Repository

| | |
|---|---|
| Branch | `dual-engine-benchmark-prep` |
| HEAD | the freeze commit itself — this document; parent `97835a8` |
| LTX baseline commit | `c649c78` — tip of `ltx25-alignment-audit` |
| Stable main | `3bd8016` |
| Lineage | `main 3bd8016` → `ltx25-alignment-audit c649c78` → `dual-engine-benchmark-prep` |
| Audited LTX upstream | `Lightricks/LTX-2` @ `400fd31` |
| Working tree | clean |
| Remote backup | **none** — every feature commit is local-only (§6) |

## 2. What the dual-engine work changed in LTX

**Nothing.** `git diff c649c78..HEAD` is **7,178 insertions, 0 deletions**,
across 30 files of which 29 are new and one is a six-line addition to the
internal docs index. No file under `apps/worker/worker/adapters`,
`worker/longform`, `worker/director`, `worker/media`, `worker/music`,
`workflow-definitions`, `apps/api`, `apps/web` or `packages` appears in the
diff at all.

A second, independent check: `worker.providers` is **not imported** by
`worker.adapters.registry` or anything it pulls in. The provider layer is off
the production execution path structurally, not by convention.

Every experimental flag from the LTX audit remains commented out in the
committed YAML, and all six workflows still read `runtime: mock`.

## 3. Verified state

| Property | Verified how | Result |
|---|---|---|
| LTX golden invocations | `tests/test_ltx_golden.py`, 11 frozen argv shapes | PASS, unchanged |
| Auto routing | resolved at runtime for all five workflows | `auto → ltx` everywhere |
| Provider override | `parameters.provider` / `execution.provider` | `ltx` and `h3` both honoured; unknown refused, never substituted |
| H3 inference | grep for torch/diffusers/hf/subprocess/network in `h3.py`, `h3_prompt.py`, `hybrid.py` | none present |
| H3 `generate()` | called at runtime | raises `ProviderUnavailable`, naming the licence gate |
| H3 `health()` | called at runtime | `(False, "not installed … Licence requires an approved application")` |
| Capability matrix vs implementation | 10 cross-checks (15s ceiling, 4:5 absent, 4:3 present, 12-file cap, fully_copy, winner rules) | all consistent |
| Structural winners | recomputed | 5 of 28 rows; 20 rows marked GPU-test |
| Hybrid handoff | `hybrid.py` inspected + compiled | decoded RGB only; **no latent path exists** |
| Provenance | compiled I2V and reference-V2V hybrids | every final section carries `user_asset` **and** `generated_intermediate` |
| Hybrid scope | compiled | extend and plain restyle refused with their recorded reasons |
| Result schema | 33 required fields checked | all present; both hybrid passes and the model switch recorded separately |
| Failure taxonomy | `FailureClass` enum | 13 structured classes, not free text |
| Targeted suites | golden + providers + hybrid + pack | **55 passed** |
| Full worker suite | `pytest -q` | **879 passed / 1 failed / 1 skipped** (19m23s) |
| New regressions | compared against the 879/1/1 recorded before this pass | **0** |
| Lint | ruff over every file this work added | clean |

The single failure is `test_a_track_longer_than_one_pass_becomes_several_scenes`
— **pre-existing and environmental**: this machine's LAME encoder makes a 4.0 s
MP3 probe at exactly 4.0 s, so four windows are planned where the test expects
five. Verified pre-existing by stash-and-rerun on 17 and 21 August, and
unchanged by any work on this branch.

## 4. Long-form and audio, measured on the day

```text
60 s text-to-video     ltx  2 sections / 1 seam   [30, 30]
                       h3   4 sections / 3 seams  [15, 15, 15, 15]

30 s video-to-video    ltx  4 sections / 3 seams  [7.5 ×4]   (transform's 8 s ceiling)
                       h3   2 sections / 1 seam   [15, 15]
```

The asymmetry reverses by workflow — LTX chains less for long-form T2V, H3
chains less for V2V — which is why routing is per-workflow and why no default
has been chosen.

Music-video audio windows, all three strategies, advance strictly and cover
the track with no restart-at-zero and no drift:

```text
ltx     s0 audio 0.000 (+20.082)   s1 20.000   s2 40.000        mode frozen_latent
h3      s0 audio 0.000 (+15.000)   s1 15.000   s2 30.000  s3 45.000   mode fully_copy
hybrid  draft windows 0→15→30→45, and the customer's own track on every section
```

LTX's 20.082 s window is the measured +0.04 s pad — exactly one audio latent
at 25 latents/second.

## 5. Benchmark and assets

| Group | Cases | Cells | Runs | Hybrid |
|---|---:|---:|---:|---:|
| A Text to video | 9 | 22 | 84 | 4 |
| B Image to video | 8 | 23 | 87 | 7 |
| C Standard V2V | 6 | 12 | 44 | 0 |
| D Reference-person V2V | 5 | 15 | 69 | 5 |
| E Music video | 4 | 11 | 51 | 3 |
| F Dialogue | 4 | 8 | 32 | 0 |
| G Camera | 1 | 2 | 6 | 0 |
| H Long form | 2 | 4 | 20 | 0 |
| I Extend | 1 | 2 | 10 | 0 |
| J Multimodal | 1 | 1 | 3 | 0 |
| **Total** | **41** | **100** | **406** | **19** |

Scoring weights sum to 100; `lip_sync` and `audio_response` are scored
separately and never averaged in.

**Golden media: 19 declared, 0 acquired.** Every asset carries a stated
purpose, provenance requirement and the cases that need it. `--verify` exits 0
with all 19 reported as pending; a missing or changed acquired asset would
exit non-zero and stop a comparison.

Prompt versioning and asset hashing are both enforced by tests: a prompt
edited at an unchanged `prompt_version` fails, a bumped version passes, and
the four hash states (correct / missing / modified / pending) behave as
specified.

## 6. Backup status — the one operational risk

**14 commits exist only on this machine.** Neither feature branch has an
upstream; only `main` tracks `origin/main`.

```bash
# push the branch to the private repository (NOT executed — needs approval)
git push -u origin dual-engine-benchmark-prep
git push -u origin ltx25-alignment-audit        # the LTX baseline it stands on

# or an offline bundle, if pushing is not wanted yet
git bundle create ../zolexai-pre-gpu-freeze.bundle main ltx25-alignment-audit dual-engine-benchmark-prep
```

Until one of those runs, a disk failure loses the entire audit, the dual-engine
architecture and the frozen pack.

## 7. Fixed during this freeze pass

One objective issue, fixed minimally:

- **Dangling citations.** `research-ltx25-zolexai-audit.md` cited
  `audit/ZOLEXAI-CURRENT-STATE-AUDIT.md` and
  `audit/DIAGNOSIS-flash-and-camera.md` three times while both files were
  untracked — a clone received a document referencing files that did not
  exist. The evidence (408 KB of markdown, every verdict carrying its
  `file:line`) is now committed as `97835a8`.

No other change was made. No feature was added, no routing touched, no
snapshot regenerated.

## 8. Decisions deliberately left open for the GPU

| Decision | Why it cannot be made now | Criteria |
|---|---|---|
| H3 runtime (SGLang / vLLM / diffusers) | depends on the card purchased and on which path is stable with the chosen build | quantization support, 96 GB feasibility, Ref2VA + supplied audio, FL2VA, service stability, memory, latency, multi-GPU future, API fit |
| H3 quantization / precision / offload | bf16 is ~110 GB and does not fit a 96 GB card; the build is a quality variable | measured VRAM headroom and quality at the product grids |
| H3 steps / CFG | the open material does not state them; CFG-distilled weights exist | read from the checkpoint, then A/B |
| Final provider routing | no comparison has been run | the decision table in the comparison framework |
| Whether hybrid is ever worth it | costs two passes and a switch | `incremental_quality_gain` vs `incremental_cost_ratio` |

The config carries none of these as hard-coded values; the H3 manifest reports
`steps`, `guidance` and `quantization` as `UNKNOWN — GPU validation required`.

## 9. Documents to take into GPU day

1. [`dual-engine-gpu-day-runbook.md`](./dual-engine-gpu-day-runbook.md) — **execute this one.** Ten phases, stop conditions, first ten commands. Every path it references was verified to exist.
2. [`golden-benchmark-pack.md`](./golden-benchmark-pack.md) — the frozen inputs and the acquisition checklist.
3. [`ltx-h3-hybrid-benchmark.md`](./ltx-h3-hybrid-benchmark.md) — why the handoff is decoded RGB, and which cases carry a third cell.
4. [`ltx-h3-comparison-framework.md`](./ltx-h3-comparison-framework.md) — scoring, metrics, and the decision table to fill in.
5. [`h3-pre-gpu-integration.md`](./h3-pre-gpu-integration.md) — H3's official limits and the licence gate.
6. [`research-ltx25-zolexai-audit.md`](./research-ltx25-zolexai-audit.md) — the verified LTX baseline.

## 10. Non-code dependencies before a comparison can run

1. **Golden media** — 19 assets, none acquired. The only dependency that costs
   calendar time rather than GPU time.
2. **H3 licence authorisation** — open weights are limited to the EU, UK, South
   Korea and the US and require an approved application; the moderation
   obligation needs an owner.
3. **GPU purchase and runtime selection** — §8.

None of these is a code-side blocker.
