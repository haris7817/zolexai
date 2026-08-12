# LTX-2.5 — licence and commercial-use review

**Internal. Revision 2 — 12 August 2026.**

> **Correction notice.** Revision 1 of this document claimed the licence contains
> no competing-products restriction and advised "do not plan around a
> restriction that was not written." **That was wrong.** The review was made
> against a truncated copy of Attachment A that ended at item 3; the full
> attachment has twenty items, and **item 20 is a competing-products clause**.
> It exists in every version of this licence, back to the original January 2026
> text. The third-party summaries revision 1 dismissed were right. Everything
> below is re-verified against complete documents, with provenance stated so it
> can be checked.

> This is an engineering review to establish whether integration can proceed and
> what it obliges us to build. It is **not legal advice**. §3 (the competing
> clause) and §4 (the revenue threshold) are decisions that need the client, and
> possibly Lightricks, not us.

**Bottom line, revised:** Non-production evaluation and benchmarking may proceed
under the evaluation provisions, subject to the applicable license, AUP and use
restrictions — so the client's RTX 5090 can be put to work on the LTX-2.5
benchmark in a non-production environment. But **LTX production use for ZolexAI
requires clarification**: the official commercial-use guidance and the
competing-products restriction (Attachment A #20) need to be reconciled for this
specific SaaS use case. **Obtain written clarification from Lightricks before
production release.** That makes licence terms a first-class scoring criterion
in the M2.04 model selection, next to quality and VRAM — not a launch-day
formality.

---

## 1. Which document governs, and its actual dates

Revision 1 called the licence "one day old." The fuller picture:

| Date | Event | Evidence |
|---|---|---|
| 5 Jan 2026 | **LTX-2 Community License Agreement** first committed (`LICENSE`, plain text), "License date: January 5, 2026" | commit `9ce438b3` |
| 9 Feb 2026 | Revised | commit `4dbd99e6` |
| **23 Jul 2026** | **LTX-2.5 weights published** on Hugging Face (`Lightricks/LTX-2.5`, `createdAt: 2026-07-23`) | HF API |
| **11 Aug 2026** | **LTX-2.x Community License Agreement** replaces it (`LICENSE.md`), "License date: August 11, 2026" | commit `23621616`, `2026-08-11T18:54Z` |

The HF model card's `license_link` points at the GitHub `LICENSE.md` on `main`,
which is the 11 August text, and it is what this review analyses. Which text
legally governs a given copy of the weights in a given situation is a question
for counsel, not for this document. **For current engineering planning, use the
license currently presented by the official LTX distribution and re-check the
applicable terms before production launch.**

One quirk worth recording for that re-check: §1.9 of the new text says it
applies to "all LTX-2.5 versions released **since August 11, 2026**" — but the
published weights predate that by three weeks. In practice this changes little
for planning (the clauses that matter to us — #20, the $10M threshold, the AUP
flow-down — are in **both** versions), with one exception: the
watermark/provenance obligations appear **only in the August text** (§7 below).
Plan against the August text.

Lightricks may revise again — this licence changed twice in seven months and the
incorporated Acceptable Use Policy is updatable at any time (Attachment A
preamble: "the version in effect at the time of your use governs"). Re-read at
launch.

## 2. What remains true from revision 1

- §3 explicitly permits SaaS hosting: *"You may host for third parties remote
  access purposes (e.g. software-as-a-service)…"*
- §5: *"Licensor claims no rights in the Output you generate"* — customers own
  their generations.
- §2.2 describes evaluation *"for testing, evaluation, or non-commercial
  research and development in a non-production or development environment"* at
  no cost, including for a Commercial Entity. So: non-production evaluation and
  benchmarking may proceed under the evaluation provisions, subject to the
  applicable license, AUP and use restrictions — the 5090 can be provisioned
  and used for the M2.04 benchmark while the production-licensing question is
  open, provided the work stays in a non-production environment.

## 3. Attachment A #20 — the competing-products clause, against ZolexAI

The clause, verbatim:

> **20)** To use LTX-2.x or Derivatives of LTX-2.x in any product, service, or
> application that directly competes with Licensor's commercial products or
> services, or is designed to replace or substitute Licensor's offerings in the
> market, without obtaining a separate commercial license from Licensor.

Three properties make it bite:

1. **No revenue floor.** Unlike §2.1's $10M threshold, #20 binds from the first
   dollar. A small operator is not exempt.
2. **It is a use restriction, not a commercial term** — §3.1 makes Attachment A
   flow into any downstream agreement, and §13 makes violating it grounds for
   termination.
3. **"Directly competes" is Lightricks' language to interpret first.** The
   licence gives no market-definition test.

**What Lightricks sells.** LTX Studio (`ltx.studio`) — a subscription SaaS where
a user types a prompt or supplies media and gets AI-generated video, with
image-to-video, storyboarding and editing around it. A paid hosted LTX-2 API
(there is a separate "LTX-2 API License Agreement" for it). Plus the mobile
creative apps (Videoleap, Facetune, Photoleap).

**What ZolexAI is.** A subscription SaaS (~$70/month) where a user types a
prompt or supplies media and gets AI-generated video — text-to-video,
image-to-video, video-to-video, extension — plus music and music video.

The tension: the licence's own commercial-use framing (free below $10M, §3's
explicit SaaS-hosting permission) points one way, while #20's plain words —
"directly competes … replace or substitute … in the market" — read against a
prompt-to-video subscription SaaS point the other, given LTX Studio exists.
There are arguments on both sides — ZolexAI is model-agnostic by design, LTX
would be one adapter among several, the music workflows have no LTX Studio
equivalent, and "designed to replace" implies intent we don't have — but they
are arguments to put to Lightricks, not grounds to ship on our own reading of
a clause drafted by the counterparty. **LTX production use for ZolexAI
therefore requires clarification: the official commercial-use guidance and the
competing-products restriction need to be reconciled for this specific SaaS use
case. Obtain written clarification from Lightricks before production release.**

**Consequence for M2.04 model selection.** The evaluation now scores licence
fitness alongside quality, VRAM and speed:

| Path | What it means |
|---|---|
| **(a) Ask Lightricks** | `ltxv-licensing@lightricks.com`. #20's own remedy is "a separate commercial license" — it is an invitation to negotiate, not a flat bar. Cost/terms unknown; a small operator may get friendly terms. Can run in parallel with benchmarking. |
| **(b) Benchmark an alternative alongside** | e.g. Wan 2.x (Apache-2.0 — no use restrictions of this kind), HunyuanVideo, or another current open-weight video model. The provider-adapter seam exists precisely so this is a config change. Quality vs LTX-2.5 is exactly what the benchmark measures. |
| **(c) Client accepts the risk** | Ship on the narrow reading. Recorded as an option only because the decision is the client's; not recommended, given §13 termination + the §2.1-style fee-recovery posture of this licensor. |

Recommendation: **(a) and (b) in parallel** — email Lightricks when M2
integration begins, and make the M2.04 benchmark a two-model comparison so a
"no" from Lightricks doesn't stall the milestone. This also honours the client's
own instruction to keep the provider layer modular.

## 4. The $10M threshold — now the *secondary* gate

Unchanged from revision 1, but demoted: even below $10M group revenue, #20
still applies. §2.1 requires a paid Commercial Use Agreement for entities with
≥$10M annual revenue, measured **aggregatively across all affiliates and
companies under common control** (§1.6). Unauthorised commercial use is a
material breach with back-fees payable at Lightricks' standard rates within 30
days of demand (§2.1), and §13 terminates automatically on material breach.

**Action unchanged: ask the client to confirm group-wide revenue.** If the
answer to #20 is a negotiated licence anyway, this question gets folded into
that conversation.

## 5. Our terms of service must bind our users — a build item (M3)

Unchanged. §3.1 + §4 require Attachment A and the Acceptable Use Policy to be
**enforceable provisions** of whatever agreement governs ZolexAI's users.
Attachment A #5 also independently requires that machine-generated content
placed anywhere be intelligibly disclosed as machine-generated — which lands on
ZolexAI's users, and therefore on our ToS and product copy.

## 6. Remote-restriction right — operational note, new in this revision

§6: *"Licensor reserves the right to restrict (remotely or otherwise) usage of
LTX-2.x in violation of this Agreement, update LTX-2.x through electronic
means, or modify the Output…"*, plus a duty of "reasonable efforts to use the
latest version." A production dependency that its licensor asserts a right to
remotely restrict or update is an availability consideration, not just a legal
one. Self-hosted weights make actual remote interference unlikely mechanically,
but the *right* exists and the version-currency duty is real. File under
deployment risk for M2.05.

## 7. Watermark / provenance — re-checked, corrected in scope

**The obligation is real, and it is new.** Every word of it arrived in the
11 August revision; the January text contains no occurrence of "watermark",
"provenance" or "latent disclosure" anywhere:

- §6: we *"shall not remove, disable, alter, or circumvent any … disclosures,
  metadata, watermarking, content provenance, latent disclosure, or other
  transparency features … included or embedded within LTX-2.x … or applied to
  any Output"*, and must *maintain* them in anything we ship. Breach carries an
  immediate-revocation right.
- Attachment A #19 (extended in August): no circumventing "watermarking, content
  provenance or latent disclosure functionalities."

**Where revision 1 overstated.** It claimed our `concat_segments` re-encode
fallback "strips a compliance feature." That presumed we know what LTX-2.5
embeds — we don't yet, and the two possibilities behave oppositely under
ffmpeg:

| Mechanism | Survives our pipeline? |
|---|---|
| **Latent (in-pixel) watermark** — §6's "latent disclosure" language suggests this exists | **Yes, by design.** Latent marks are built to survive re-encoding; stream-copy trivially preserves them, and even the re-encode fallback almost certainly does. |
| **Container/sidecar metadata (e.g. C2PA manifest)** | **No.** ffmpeg drops container metadata by default on both concat paths, and a C2PA manifest is invalidated by any byte change regardless. |

So the corrected engineering position, for the batch that integrates the model:

1. **Determine empirically what LTX-2.5 embeds** (probe raw output: container
   tags, C2PA manifest, documented latent mark) — first task on the GPU, part
   of M2.04.
2. If container-level provenance exists: carry it through assembly
   (`-map_metadata`), or re-sign the assembled file if it's C2PA — and add a
   post-assembly check that it survived, in the same measure-don't-trust style
   as `verify_duration`.
3. Stream-copy stays the concat default (it already is); the re-encode fallback
   gets a log marker so any job that used it is identifiable.
4. Nothing in the current codebase violates anything today — no LTX output has
   ever entered the pipeline. This is a gate on M2 integration, not a defect in
   Batch 1.

Separately, §6 makes us responsible as deployer for AI-transparency law
(EU AI Act, California AI Transparency Act) including disclosure that content is
AI-generated — an M3 item alongside the ToS work in §5.

## 8. Smaller items

| Clause | Effect on ZolexAI |
|---|---|
| §8 Trademarks | No Lightricks marks, no implied endorsement — consistent with our no-provider-names UI rule. |
| §3.2–3.4 | Bind only if we distribute weights/derivatives. We host inference; N/A unless we ship a fine-tune. |
| §3.5 | A fine-tune/LoRA transferred to a $10M+ entity triggers *their* paid licence regardless of author. Relevant only if M2 tuning produces shareable artefacts. |
| Attachment A #18 | Commercial users may not use LTX to train/improve other models (Derivatives excepted). Irrelevant to inference hosting. |
| §7 | Standard OFAC/EAR reps. |
| §9–10 | AS-IS, liability disclaimed — client carries model-behaviour risk; say so at handover. |
| §12/§14 | NY law, ICC arbitration in New York, jury/class waivers for non-consumers. |

## 9. Verdict — revised

| Question | Rev 1 said | **Now** |
|---|---|---|
| Evaluate/benchmark in M2, on the client's RTX 5090? | Yes | **Yes, in a non-production environment** — under the evaluation provisions, subject to the applicable license, AUP and use restrictions |
| Competing-product bar? | "No such clause exists" | **Yes — Attachment A #20, both licence versions, no revenue floor** |
| Run LTX-2.5 in ZolexAI production as-is? | Yes if under $10M | **Requires clarification** — the commercial-use guidance and #20 need to be reconciled for this SaaS use case; **written clarification from Lightricks before production release** |
| Anything blocking Batch 1 / Batch 2? | No | **No** — neither touches LTX |
| Before launch | revenue check, ToS, provenance | **Written Lightricks clarification (or an alternative model)**, then revenue check, ToS AUP flow-down, provenance verification, AI-disclosure labelling |

## 10. Actions

- [ ] **Now:** M2.04 benchmark plan becomes a two-model comparison (LTX-2.5 + one Apache-licensed alternative, e.g. Wan 2.x)
- [ ] **When M2 integration begins:** client requests **written clarification** from `ltxv-licensing@lightricks.com`, describing ZolexAI and asking how the commercial-use guidance and Attachment A #20 apply to it — before any production wiring
- [ ] Client confirms group-wide annual revenue (folds into the same conversation)
- [ ] On GPU: probe what LTX-2.5 embeds in Output; add provenance-preservation check to assembly
- [ ] M3: ToS carries Attachment A + AUP as enforceable terms; AI-generated disclosure on delivered media
- [ ] At launch: re-read licence + AUP at their then-current versions

## Sources

- [LTX-2.x Community License Agreement — 11 Aug 2026 text, as currently presented by the official distribution](https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md), commit `23621616`
- LTX-2 Community License Agreement — 5 Jan 2026 text (superseded), commits `9ce438b3` / `4dbd99e6`, retrieved via GitHub API
- [Lightricks Acceptable Use Policy](https://static.lightricks.com/legal/ltx-acceptable-use-policy.pdf) (incorporated by reference)
- [Lightricks/LTX-2.5 on Hugging Face](https://huggingface.co/Lightricks/LTX-2.5) — `createdAt` 2026-07-23; `license_link` → GitHub LICENSE.md
- [LTX-2 API License Agreement (hosted API — separate document)](https://static.lightricks.com/legal/ltx-2-api-license-agreement.pdf)
- [LTX-2.5 product page](https://ltx.io/model/ltx-2-5) — Python ≥3.12, CUDA ≥12.7, PyTorch ~2.7; fp8/int8 quantisation options (relevant to the 32 GB RTX 5090)
