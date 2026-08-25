# Client test readiness — 25 August 2026

Integration commit `d81326c` (+ this report). All six end-to-end workflows ran
through the real worker adapters on the client-test GPU node; every output was
probed and inspected. Review pack: `benchmarks/review/client-e2e/`.

```text
CLIENT TEST READINESS
=====================

LTX T2V:                        READY   (E2E 40.7 s incl. reload; render ~28 s)
H3 I2V:                         READY   (E2E 331 s / 5.17 s @1280x736)
H3 Reference V2V:               READY   (E2E 172 s / 5.17 s @960x544 quality tier)
LTX Music Video A2V:            READY   (E2E 164 s / 10.04 s, audio-conditioned)
LTX Extend:                     READY   (E2E 24.6 s, 10.0 s from a 5 s source)
H3 60s long-form:               READY   (E2E 716 s; identity/wardrobe/prop held
                                         across all 5 segments from GENERATED prompts;
                                         framing varies more than the hand-tuned run —
                                         Director-structured plan fields are the
                                         known refinement)

H3 prompt discipline integrated: YES    (worker/longform/h3_prompts.py, 10 tests)
H3 audio seams:                 NEEDS FIX (3/4 seams clean; one 9.8 dB loudness
                                         step, partially prompt-confounded;
                                         execution.h3_audio_context is the wired,
                                         default-off experiment)
H3 Draft tier:                  READY   (544x320, ~11-12x realtime)
H3 Quality tier:                READY   (960x544, ~33x realtime)

ComfyUI service adapter:        READY   (submit/wait/collect/interrupt/health;
                                         cancellation interrupts the server)
Job progress:                   READY   (house vocabulary; honest section
                                         counters on long runs — observed live:
                                         "Generating section 2 of 5")
Storage/history:                READY   (standard AdapterResult through the
                                         existing runner path; no new plumbing)
Frontend:                       READY   (no changes needed: I2V already declares
                                         source_image + the exact five durations;
                                         V2V already declares reference_image;
                                         duration_mode: source honoured via the
                                         nearest-preset fallback)
API:                            READY   (untouched)
Worker:                         READY   (new runtime registered; nothing shipped
                                         routes to it — pinned by test)

End-to-end workflows tested:    6  (T1-T6) + 1 negative (music-video guard REFUSES)
Full worker suite:              909 passed / 1 failed / 1 skipped — the failure is
                                the known pre-existing LAME probe case on this
                                dev machine, unchanged since 17 Aug
LTX regressions:                0  (golden argv suite green; guard is additive)

Production changed:             NO
Auto routing changed:           NO
Client-test environment:        READY (docs/internal/client-test-deployment-plan.md:
                                       routing YAML, services, health, rollback)

Estimated remaining work before client can test:  ~1-2 hours
  — apply the routing YAML edits in the client-test environment copy,
    start the three services per the plan, run the plan's health checks.
  (Optional before hand-over: one listening pass over 06's audio seams.)

CLIENT CAN TEST: YES
```

## Standing items, tracked separately

- **P0 LTX_REFERENCE_V2V_ANCHOR** — the opening-frame composite in LTX's
  `v2v_reference_identity`. Does not block client testing (that workflow
  routes to H3 in client-test); required before the final reference-V2V
  routing decision (`LTX_FIXED vs H3_INT8`).
- **Audio seam** — one loudness step at a 60 s boundary; A/B
  `h3_audio_context` with static-dynamics prompts on the next GPU session.
- **Long-form framing consistency** — feed Director scene fields into
  `H3ScenePlan` instead of the free-text fallback.
- Paused by decision, unchanged: Turbo, Hybrid, H3 API, Tier 2, the 406 pack.
