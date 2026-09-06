# Internal engineering notes

**Not customer-facing.** Everything in this directory may name models, providers,
hardware and vendors. Nothing here should be reproduced in the product UI, the
public API, marketing copy, or a document sent to an end user.

That split is a product rule, not a habit: the public surface is deliberately
provider-agnostic so a model can be replaced without a customer noticing, and
`apps/api/tests/test_workflows.py` fails if a provider name reaches a public
response. These notes are where the other half lives.

| File | Subject |
|---|---|
| [`client-readiness-report.md`](./client-readiness-report.md) | 🟡 **The final-milestone handover.** Implemented modules, workflows and models used, dependencies, what was verified without a GPU, limitations, rollback. Ends CLIENT TEST READY: WAITING FOR GPU VALIDATION. |
| [`extension-first-last-frame.md`](./extension-first-last-frame.md) | 🟢 **Extend Video, 6 Sep 2026.** Audit of why First/Last Frame was absent on Extend and why a second Extend read as dead (measured: the hand-off worked, the screen did not say so); optional first/last frame on every extension through the client's own graph; unlimited chained extensions with a stored chain record; failure behaviour, rollback, tests, GPU checks pending. |
| [`gpu-validation-checklist.md`](./gpu-validation-checklist.md) | 🟡 **Run when the GPU returns.** The nine-section validation of the LTX 2.5 client workflows: models, nodes, each graph, the extension seam, measurements, the ZIP-sample comparison. |
| [`ltx25-gpu-benchmark.md`](./ltx25-gpu-benchmark.md) | 🟢 **Measured 5 Sep 2026 on the RTX PRO 6000.** Official CLI 5 s in 41.5 s; client T2V ladder 48.8/76.3/106.3/215.2 s; First/Last, Character Replacement and Extend rendered; ZolexAI output bit-identical to the direct ComfyUI submission. LTX 2.5 GPU VALIDATION: PASS. |
| [`ltx25-gpu-model-validation.md`](./ltx25-gpu-model-validation.md) | The node as built: GPU facts, the fourteen official files with hashes, ComfyUI 0.34.5 and the node-pack pins that actually served (two forced deviations), each client graph checked file by file and rendered. |
| [`ltx25-model-inventory.md`](./ltx25-model-inventory.md) | Which models LTX 2.5 uses on the GPU: the official release file by file (precision, size, role), what the CLI runtime and the client graphs each load, the cross-version LoRAs, VRAM/disk arithmetic, licences, open items. Researched 5 Sep 2026, unmeasured. |
| [`ltx-comfy-runtime.md`](./ltx-comfy-runtime.md) | The second ComfyUI instance for the client's LTX 2.5 graphs: service, node-pack pins, weights and sources, install order, co-tenancy, rollback. |
| [`ltx-comfy-optimization-checklist.md`](./ltx-comfy-optimization-checklist.md) | Phase 5: the speed levers ranked by how far they deviate from the pack, and the before/after protocol each must pass. GPU-gated. |
| [`ltx-client-workflow-audit.md`](./ltx-client-workflow-audit.md) | 🟢 **Phase 0 of the LTX 2.5 integration (5 Sep 2026).** The client ZIP decoded node by node: models, LoRAs, node-pack pins, inputs/outputs, the migration design, risks, and the GPU validation checklist. |
| [`client-final-audit.md`](./client-final-audit.md) | 🟢 **Start here for the final client milestone.** Phase 0 audit (5 Sep 2026): what production runs today, the H3 footprint to flag, the client's LTX 2.5 ComfyUI pack decoded, the per-phase plan, risks and rollback. |
| [`ltx-2.5-licensing-review.md`](./ltx-2.5-licensing-review.md) | Licence and commercial-use review for the candidate video model |
| [`production-runbook.md`](./production-runbook.md) | Production server layout, deploy/rollback procedure, go-live checks (M1 stack) |
| [`gpu-worker-runbook.md`](./gpu-worker-runbook.md) | The GPU node, its restricted tunnel to the production API, GPU routing and rollback (M2). Covers the video and music runtimes and process supervision. |
| [`issue-triton-na-kernel.md`](./issue-triton-na-kernel.md) | 🔴 **Open bug.** LTX video fails above a dimension-dependent length on the PRO 6000. Read before touching video. |
| [`next-steps-2026-08-15.md`](./next-steps-2026-08-15.md) | Planned work after the 14 Aug migration: outstanding risks, generation-speed options with measurements, quality roadmap |
| [`research-ltx25-zolexai-audit.md`](./research-ltx25-zolexai-audit.md) | The LTX-2.5 alignment audit: our invocation against the official pipeline source, the flag-gated fixes, and the GPU validation checklist. **The verified baseline.** |
| [`h3-pre-gpu-integration.md`](./h3-pre-gpu-integration.md) | MiniMax H3's official capabilities and limits, what was built around it, and the licence gate. No GPU, no routing decision. |
| [`ltx-h3-comparison-framework.md`](./ltx-h3-comparison-framework.md) | How the two engines will be compared: provider architecture, capability matrix, 41 benchmark cases, scoring, and the decision process. |
| [`ltx-h3-hybrid-benchmark.md`](./ltx-h3-hybrid-benchmark.md) | The LTX→H3 hybrid: why the handoff is decoded RGB and not latents, which cases carry a third cell, and how its cost is accounted. |
| [`golden-benchmark-pack.md`](./golden-benchmark-pack.md) | The frozen benchmark inputs: assets, hashes, provenance, prompt versioning, and the result manifest. |
| [`dual-engine-gpu-day-runbook.md`](./dual-engine-gpu-day-runbook.md) | 🟢 **Execute this when the GPU arrives.** Ten phases, stop conditions, and the first ten commands. |

The runbook names the VPS, its addresses, the deployment paths and the
CloudPanel routing, which is why it lives here rather than beside the
client-facing control documents. It carries **no secrets** by design — no
passwords, keys, tokens or `.env` values — and it must stay that way: the
production `.env` is on the server at mode 600 and belongs nowhere else.

It documents the state as deployed for **M1 (mock worker)**. The M2 GPU runtime
is described separately in `gpu-worker-runbook.md`, written at deploy time from
what was actually done rather than in advance — same rule, same no-secrets
constraint.

The two runbooks answer different questions. `production-runbook.md` is "how the
VPS stack is built and rolled back"; `gpu-worker-runbook.md` is "how a rented
GPU box reaches that stack safely, and how a workflow is pointed at it". Read the
first before the second.
