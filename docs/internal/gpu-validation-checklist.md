# GPU validation checklist — LTX 2.5 client workflows

> **Executed 5 September 2026 on the RTX PRO 6000 (`163.182.37.67:20577`).** Sections 1–8 done and recorded in `ltx25-gpu-model-validation.md` and `ltx25-gpu-benchmark.md` (LTX 2.5 GPU VALIDATION: PASS). Section 9 — the client-test deployment (routing profile, api+web rebuild, parity and e2e against it) — is the remaining step before CLIENT TEST READY: YES.

**Run this when the GPU node is available. Until every box is ticked with a
recorded number, the readiness report says CLIENT TEST READY: WAITING FOR
GPU VALIDATION.** Do not change the architecture during validation: the
integration is complete; this is measurement and, if needed, the Phase 5
optimisation protocol.

Everything below is scripted. Record results in
`benchmarks/results/ltx25/<stamp>/` and paste the tables into
`docs/internal/client-readiness-report.md`.

---

## 1. Load required models

- [ ] Weights present under `/workspace/ComfyUI-ltx/models/` at the exact
      paths in `docs/internal/ltx-comfy-runtime.md` §3.
- [ ] `sha256sum` and byte size of every file recorded in
      `benchmarks/results/ltx25/weights.json`.
- [ ] `cd apps/worker && .venv/bin/python scripts/ltx_comfy_health.py --deep`
      → `HEALTHY`. Every "not among the offered values" line is a file the
      server cannot see; fix before continuing.

## 2. Load required ComfyUI nodes

- [ ] ComfyUI core ≥ 0.34.0; the ten node packs at the pins in
      `ltx-comfy-runtime.md` §2 (record the commits that actually serve).
- [ ] `scripts/ltx_comfy_health.py` reports zero "not installed" lines for
      all three graphs.
- [ ] `ResolutionSelector` options include `16:9`, `9:16` and `1:1` labels
      (the script prints them).
- [ ] `supervisorctl status zolexai-ltx-comfy` → RUNNING; survives
      `supervisorctl restart`.

## 3. Execute Text-to-Video

- [ ] `scripts/ltx_comfy_bench.py t2v --seconds 5` — look at the file before
      anything else: motion, audio present, no burned-in text, 121 frames.
- [ ] Full matrix: `--seconds 5 10 15 30 --aspect 16:9 9:16 1:1` (12 cells).
- [ ] Output duration = seconds + 1/24 (±0.1 s); fps 24; audio 48 kHz stereo.
- [ ] Compare the 30 s 16:9 cell with the ZIP sample
      (`01_Text_to_Video.mp4`, 1280x704, 30.04 s): resolution class, motion
      quality, audio character.

## 4. Execute First/Last Frame

- [ ] `flf --first benchmarks/client-pack/ltx25/samples/first_last_frame_input.png --seconds 5 10 15 30`.
- [ ] `flf --first … --last … --seconds 10` (both stills): the first frame
      matches the first still; the last second lands on the last still.
- [ ] First-only cell: the bypassed last-frame node produces a clean pass
      (no validation error, no black tail).
- [ ] Identity held from first to last second (human verdict, recorded).

## 5. Execute Character Replacement

- [ ] `cr --video benchmarks/client-pack/ltx25/samples/character_replacement_source.mp4 --image <photo>`
      with a photo of a different person.
- [ ] Output: 8 s window, 193 frames, 736x1280, source audio present.
- [ ] Side-by-side with the ZIP sample (`character_replacement-output`):
      frame 0 is the photo, frame 4 onward follows the source's motion, the
      photo's setting is the video's setting.
- [ ] A landscape source produces 1280x736.

## 6. Test the extension system

- [ ] Extend a 30 s T2V result by 5, 10, 15 and 30 s (four jobs).
- [ ] Each `continuation.json`: one pass, `frames_kept = seconds × 24`,
      `measured_seconds ≈ promised_seconds` (±0.1 s).
- [ ] Inspect the seam at 24 fps: no held/duplicated frame, no black frame,
      audio click absent (edge fades).
- [ ] Extend an extension (chain of two) — the lineage keeps working.

## 7. Measure

For every cell the bench script records runtime, VRAM peak/mean (nvidia-smi
1 Hz), RAM peak (psutil), fps, output duration. Additionally:

- [ ] Set `LTX_COMFY_EXPECTED_WALL_PER_OUTPUT_SECOND` from the measured T2V
      rate so the progress bar paces honestly.
- [ ] Record VRAM with ACE-Step resident and with it stopped; decide the
      co-tenancy policy (`ltx-comfy-runtime.md` §5).
- [ ] Cancel a 30 s job mid-render from the UI: the ComfyUI queue empties,
      `nvidia-smi` shows the memory released, the job reads Cancelled.
- [ ] Run one job under `RUNTIMES` without `ltx_comfy`: the job stays
      queued (never claimed), proving the claim intersection.

## 8. Compare against the ZIP samples

- [ ] T2V 30 s and FLF 30 s: duration, resolution class, fps, audio layout
      match `ltx-client-workflow-audit.md` §2.1.
- [ ] Character replacement: the four-frame handoff behaviour reproduced.
- [ ] Verdicts (pass / fail / note) written into the readiness report,
      each with the cell name that produced it.

## 9. Only then

- [ ] `deploy/vps-local.sh --profile client-test` in the client-test
      environment; rebuild api and web; `qa:parity` PASS; `qa:e2e` PASS.
- [ ] Readiness report: CLIENT TEST READY: YES, with the tables filled.
