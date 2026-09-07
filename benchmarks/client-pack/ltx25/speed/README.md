# FAST 1080 speed work — the two copies

*8 Sep 2026. Report: `docs/internal/ltx25_speed_optimization_report.md`.*

| file | sha256 | what it is |
| --- | --- | --- |
| `LTX2.5_FAST_1080_original.json` | `19480e74…1f6e712` | the client's export, byte-identical |
| `LTX2.5_FAST_1080_optimized.json` | `19480e74…1f6e712` | **the same bytes** |

They are identical **on purpose, and that is the result**: the optimization
that survived measurement is a ComfyUI server flag (`--use-sage-attention`),
not a change to the workflow. Every graph-level change that was tested —
a different transformer, a shorter sigma schedule — either ran slower or
changed the video, and each is rejected with its numbers in the report.

So the optimized configuration is:

* this workflow, unchanged, driven by the same compiler as before;
* the NVFP4 transformer the client already specified (the fastest of the
  three tested, and the lightest on memory);
* ComfyUI started with `--use-sage-attention`.

`optimized.json` exists so the deliverable named in the brief exists and so a
future graph-level optimization has an obvious home. Until something is
written into it, routing anything at it changes nothing — which is why
production keeps pointing at the client's original file.

**Rollback** is the launcher's extra-args file, not these copies:

    : > /workspace/comfy_extra_args     # or: echo --no-sage > …
    supervisorctl restart zolexai-ltx-comfy
