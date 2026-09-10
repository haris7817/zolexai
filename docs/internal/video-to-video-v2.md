# Video to Video, reworked twice — 9 and 10 September 2026

The client changed this tool's shape twice in two days. This is the account of
both, what shipped, what was archived, and what the GPU node needs.

Read the second section first if you only want the current state.

## The two packages

**`zolexai-music-v2v-backend-v2-fixed.zip`, 9 Sep 21:32.** Video to Video as a
cast replacement: one to four people mapped onto the source from screen-left to
right, a required person reference, an optional background replacement, a 4K
finish, and a self-hosted voice-conversion service for milestone 2a.

**`zolexai-v2v-deploy-code.zip`, 10 Sep 14:56.** Seventeen hours later, and
after our questions went out, so it is their answer. It withdraws the cast
entirely — back to one OPTIONAL reference — and replaces the 4K finish with a
1080p/4K/8K ladder over a fixed 480-class render. **Confirmed deliberate by the
client on 10 Sep.**

Both packages carried a copy of our worker cut from our tree on 8 Sep, so both
predate the music lyrics v2 workflow, the music-video prompt enforcement, the
vendored `zolex_music_worker` and the LTX 2.5 guideline pack. **Neither was
merged.** Both were read as specifications and ported additively.

## What ships now

| Piece | Where |
|---|---|
| Identity gated by the upload, not a quality level | `LtxAdapter._uses_reference_identity` |
| Fast proxy grid matched to the source's aspect | `proxy_grid_for_source` |
| 1080p/4K/8K delivery frame, aspect preserved | `delivery_dimensions_for_source` |
| One resize after the audio is restored | `LtxAdapter._deliver_restyle` |
| `settings.sound` | `LtxAdapter._deliver_restyle` |
| The contract | `workflow-definitions/video-to-video.yaml` |

The delivery encode reuses `worker/media/upscale.py`, the same lanczos and
NVENC path Text to Video HD and Character Replacement finish through. The
client's package carried its own copy; using ours keeps the three tools from
drifting into three different 4K, and ours already falls back to the CPU
encoder and cancels cleanly when a job is abandoned.

### One tool, two paths

No photo is a prompt-only restyle. A photo turns on person replacement. That
is the client's 10 Sep answer to the question their 9 Sep package raised by
making the reference mandatory, and it is a better answer than the Fast/Best
pair it replaces: the levels existed to let a customer opt out of identity
replacement, and the upload says that directly.

`v2v_reference_identity` is therefore unconditionally true in the definition
and inert without an image.

### What the quality control actually sells

**It changes the container, not the detail.** Every level renders the same
512-class picture and resizes once. 8K is that render enlarged about 8 times.

This is the client's explicit design — their own test is named *quality
selection changes only final delivery* — and it is their call. Two measured
facts belong beside it:

* generation drops from 1024x576 to 896x512 at 16:9, which is 0.78x the
  pixels, so V2V output detail goes **down** relative to what this tool
  shipped before. The softness is visible in the acceptance clips below;
* a lanczos enlargement adds no detail. Measured 7 Sep 2026: a smaller render
  upscaled to the same frame scored 0.29x the detail and was visibly softer.

### Their 480 grid could not render at all

The client specifies a 480-class canvas on a /32 lattice. **This path needs
/64, and 480 is not a multiple of 64.** IC-LoRA encodes the reference video
through the VAE, which halves the latent, and the latent is the pixel size
over 32 — so 480 gives an odd latent of 15 and the encoder cannot halve it.
The first real clip through their setting died there, on the card:

```text
einops.EinopsError: ... "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w"
Input tensor shape: torch.Size([1, 1024, 50, 15, 26])
Shape mismatch, can't divide axis of length 15 in chunks of 2
```

26 is 832/32 and 15 is 480/32. **Every V2V job on their grid fails**, so there
was no version of their package that could have shipped as written. The short
side is therefore 512, the nearest legal neighbour above; 448 is the one below
and would be cheaper still, but detail is already what this feature spends.
`render_proxy: 480p` in their workflow keeps working and resolves to the legal
grid. `test_every_proxy_side_is_divisible_by_64` stops anyone reaching for 480
again.

This is also the client who rejected a 1.59x speed win on 9 Sep purely on how
it looked, having found no metric that faulted it. If the look is judged short,
**deleting the `render_proxy` line puts generation back on the measured grid**
and the ladder keeps working with less enlarging to do. That is the one-line
lever, and it is why the proxy is a workflow key rather than a code change.

`test_every_quality_level_renders_the_same_picture` pins this, so making 8K
render larger has to be a deliberate edit rather than a quiet one — the cost of
an 8K job would rise by more than an order of magnitude.

### One deliberate deviation from their package

Their workflow asks for `v2v_identity_refresh_strength: 0.30`, and their code
re-shows the **raw photograph** at that strength at an interior frame of every
later pass. That is the configuration that cut a reference portrait into a
customer's dance video on 19 Aug 2026, measured at 0.35 and again at I2V's
"safe" 0.2, which is why our default is 0.

We kept their 0.30 and changed what is shown: the **composited anchor**, which
carries the same face in a composition that belongs in the shot. The persistence
they asked for, without the defect. When the anchor cannot be built, `anchor`
is the photo again and the capped opening strength is the only protection left.

## What was archived, not deleted

The cast replacement is on the **`v2v-cast-replacement-archived`** branch
(commit `f6ac11a`): the four positional slots, the background replacement, the
strict multi-person anchor, its BiRefNet segmentation helper and the 4K finish.

It is kept because this tool has now changed shape twice in two days, and
because the parts that touch hardware were validated before it was withdrawn.

### What that validation found, and why it still matters

Run on the RTX PRO 6000 on 10 Sep against real two-person footage and the
performer photos from the music-video smoke runs.

| Case | Result | Cost |
|---|---|---|
| One person, one reference | mapped | 6 s |
| Two people, two references | mapped screen-left to right | 5 s |
| Two people plus a new background | mapped onto the new environment | 6 s |
| Two references, one-person source | refused, no file written | 5 s |
| Person matte, 193 frames at 1024x576 | correct grid, rate and count | 19 s |

**Three defects, all fixed, none catchable without hardware.** These are not
archived — they are in the shipped tree, because the matte and its segmentation
helper still run on every identity job:

1. **BiRefNet would not load.** Its published modelling file imports `kornia`
   and `timm`, and the client's installer installs neither. On a fresh node
   their documented sequence stops here.
2. **The first convolution crashed** with "Input type (float) and bias type
   (c10::Half) should be the same". The published weights are fp16 and their
   code fed them fp32. `_person_segmentation.py` now reads the dtype off the
   loaded weights, correct for either precision.
3. **The installer moved the CUDA stack**, silently downgrading
   `nvidia-cudnn-cu13` from 9.21.1.3 to 9.20.0.48 in the environment the LTX
   pipelines run in. `uv pip install --dry-run` did not predict it. The script
   now snapshots the `nvidia-*` versions and restores anything that moved, and
   no longer adds `opencv-python-headless` beside the `opencv-python` the
   checkout already has.

`install_v2v_models.sh` ends with a load check that would have caught the first
two in seconds.

## What the client's deploy package would have done

Their 10 Sep ZIP is a standalone deployment: a compose file, a preflight
script, and a `v2v-worker` container. It was not adopted, and these are the
reasons rather than a preference.

* **Every identity job would fail.** There is no `scripts/` directory in the
  ZIP and the Dockerfile copies only `worker/`, yet their code still shells out
  to `scripts/person_matte.py`, and their own docstring says a matting failure
  fails the pass. Their definition turns identity on.
* **Our API would not boot.** `ui.icon: video` is not in the icon literal. It
  is the only schema breakage in their file; everything else validates.
* **A second worker would claim our jobs.** Their compose declares
  `RUNTIMES=ltx`, so it would compete with the existing worker for every LTX
  job, text-to-video included, using a slice that lacks most of our code.
* **`runtime: ltx` is hard-coded**, which defeats `deploy/vps-local.sh`'s
  per-profile runtime switch and would leave uploads signed as PNG.

Not a defect, checked: their `JOB_TIMEOUT_SECONDS=1800` is overridden by the
workflow's `timeout_seconds`.

## Still open with the client

* **The environment variable rename.** Both packages ship `config.py` with
  `ACESTEP_LYRICS_*` and `HOSTED_LLM_*` in place of our Cerebras credentials.
  On our production node that silently reverts non-English lyrics to English
  only, which has caught us twice. We have not applied it and it remains
  unanswered.
* **Voice replacement, milestone 2a.** No audio input is declared and no
  `v2v_voice_conversion`, so nothing can be paid for and dropped. When it
  starts, the pieces are in the 9 Sep package: `worker/media/voices.py`,
  `scripts/voice_service_client.py`, the `CONVERTED_SOURCE_AUDIO` audio mode
  and the whole `voice-conversion-service/` directory. Two things to settle
  first: their voice slots map by first speaking turn while person slots map
  by screen position, which no interface can label honestly; and their YAML
  accepts `video/mp4` as a voice sample while our asset allowlist rejects video
  under the audio kind, so "voice from a video" needs an API change.
* **No consent gate.** Their `v2v_voice_consent_required` ships off. The
  product decision on 10 Sep was not to add a consent checkbox for face
  replacement. The 19 Aug recommendation stands, unactioned.

## Two pins were re-based

Both guards fired and both were doing their job. The protocol is the same as
music-video on 8 Sep and music on 9 Sep: the module note records why.

* `apps/api/tests/test_untouched_workflows.py` pins the definition by sha256.
* `apps/worker/tests/test_untouched_runtimes.py` pins `adapters/ltx.py`, and it
  matched HEAD exactly beforehand, so this change is the only thing that moved
  it. That pin exists so the milestone's new runtimes could not disturb a
  working tool; it was never a rule against the client changing the tool.

## What the GPU node needs

Nothing is bundled — neither package ships weights.

1. **BiRefNet**, for the matte on identity jobs:

   ```bash
   LTX_REPO_DIR=/workspace/ltx2-benchmark apps/worker/scripts/install_v2v_models.sh
   ```

   Already installed on `163.182.37.67:20577` on 10 Sep, along with `kornia`
   and `timm`, with nothing downgraded — verified by running a cuDNN
   convolution on the card afterwards. Both scripts load with
   `local_files_only=True`, so a customer job never waits on a download.
2. **The Union Control LoRA**,
   `loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`, under
   `LTX_MODEL_DIR`. Present on that node. A node without it refuses these jobs
   before spending GPU time.
3. **`MAX_CONCURRENCY=1`**, alongside `LTX_COMFY_FREE_AFTER_JOB=true` and the
   cgroup ceiling from 7 Sep.

## GPU acceptance of the shipped path, 10 Sep 2026

Deployed to `163.182.37.67:20577` at commit `ac6df3c` plus the /64 fix, and
driven through `LtxAdapter.run` by `scripts/v2v_smoke.py` — the production
code path, not a fixture. Source: a 6.92 s, 1280x704 shot from the music-video
smoke runs, which is 1.818:1 and therefore not exactly 16:9.

| Case | Delivered | Duration | Audio | Wall |
|---|---|---|---|---|
| Prompt-only, 1080p | 1920x1056 | 6.92 s | source had none | 78 s |
| Prompt-only with audio, 1080p | 1920x1056 | 7.00 s | restored | 60 s |
| Reference photo, 4K | 3840x2112 | 6.92 s | source had none | 108 s |
| Reference photo swapping the person, 1080p | 1920x1056 | 6.92 s | source had none | 108 s |

Everything the tool promises held. **The source's aspect survives** — 1056 and
2112, not 1080 and 2160, because the upload is 1.818:1 and it is not cropped
to fit a product ratio. **The length is exact.** **The soundtrack comes back**
when the source has one.

The restyle is a real restyle: the same alley, the same pose, the same hand on
the jacket, rendered in graphite. The replacement is a real replacement: the
bearded man in the patterned tracksuit becomes the curly-haired reference
person, with his olive trousers and his watch, in the same alley in the same
pose. **No portrait flash** at the client's 0.30 refresh, which is what the
anchor change was for.

Roughly 9 to 16 times realtime, so a five-minute upload is an hour or more.
Identity costs about 30 s more per short clip than a prompt-only restyle — the
anchor once, and the matte per pass.

## Still unmeasured

* **A source long enough to need several passes.** Everything above is one
  pass. The seams, the accumulating audio drift the delivery re-times, and the
  identity persistence across sections are all untested at the new grid.
* **8K on a real source.** 4K is proven; an 8K frame from a 5m30 upload is the
  largest thing this platform has ever written.
* **The grid's own ceiling.** 896x512 and 960x512 are not in `_GRID_CEILINGS`,
  so they chain at `_UNMEASURED_CEILING`. `transform_pass_seconds` is shorter
  still, so it does not bind today, but the frame-landing tables are per-grid
  too.
* **Whether the softness is acceptable.** It is visible in the clips above.
  This is a judgement no measurement makes, and the reason `render_proxy` is
  one line.

## Test status, 10 Sep 2026

**Worker suite: 1372 passed, 10 failed, 2 skipped.** The ten are pre-existing
and were verified identical on an unmodified `git worktree` at `6fd783c`:
three Director planner tests that reach the network, five `test_h3_comfy`
tests for the withdrawn engine, and two music-video tests. **This work adds no
regressions.** The skips are the 4K delivery tests on a box that cannot spare
the memory for the frame.

`tests/test_v2v_delivery.py` is the new suite, 16 tests. Its load-bearing one
is `test_every_quality_level_renders_the_same_picture`.

**API suite: cannot run on this machine** — every test errors on
`ConnectionRefused` because Docker Desktop is not running, including tests that
predate this work. The changed assertions were verified directly against the
registry instead: the definition loads, the ladder and the optional reference
are as written, the execution block does not reach the public projection, no
infrastructure name leaks, and the re-based checksum matches the file byte for
byte. **The API suite still has to be run on a machine with its database up.**

## Dev-machine note

`needs_4k_encode` in `apps/worker/tests/conftest.py` was probing with 12
frames, which never fills x264's lookahead, so it reported "this box can encode
4K" while both 4K delivery tests then died inside ffmpeg. Raised to 60 frames
on 10 Sep 2026. The probe runs once at import and free memory moves while the
suite works, so the delivery test also treats an allocation crash inside the
encoder as the environment limit it is, while every other assertion still runs
for real.
