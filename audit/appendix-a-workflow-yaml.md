# Appendix A (companion) — Full workflow-definition dumps, verbatim

Companion to [ZOLEXAI-CURRENT-STATE-AUDIT.md](ZOLEXAI-CURRENT-STATE-AUDIT.md).
Repo commit `3bd80165da35c52d1f0fd7cd16d5daacf0a7032a`, branch `main`, working tree clean.

All six files reproduced byte-for-byte, comments included. The comments carry the
measurement provenance the comparison phase needs (dates, GPU model, frame counts,
what failed and how), which is why nothing is elided.

---

## `workflow-definitions/text-to-video.yaml`

```yaml
# ZolexAI workflow definition — PUBLIC METADATA.
#
# These files are the single source of truth for what ZolexAI offers. The API
# validates them at startup and serves them at GET /api/v1/workflows; the
# frontend builds every tool surface from them; the worker dispatches on them.
#
# Anything provider-, model- or hardware-specific belongs under `execution:`,
# which is stripped from every public response. See docs/decisions/0002.
id: text-to-video
version: "1"
name: Text to Video
category: video
output_type: video

description: Describe a scene and generate cinematic video
short_description: From prompt to motion
marketing_description: Describe a scene and watch it come to life as cinematic video.

prompt:
  required: true
  placeholder: "Describe the video you want to create…"
  # Effectively unlimited (client request, 14 Aug 2026): 20 000 characters is
  # ~3 000 words and is the ceiling the API schema allows. Nobody writing a
  # prompt reaches it. It is not literally removed because the value bounds a
  # request body, and an unbounded one is a denial-of-service surface.
  max_length: 20000

# No uploaded assets — the prompt is the only input.
inputs: []

duration_mode: fixed
# The same ladder Image to Video and Extend Video offer. 60s was missing here
# only by oversight — the backend produces it exactly the way those two do, as
# chained safe-duration passes (see apps/worker/worker/longform/chain.py), and
# a customer has no reason to find the longest option on two video tools but
# not the third.
supported_durations: ["5s", "10s", "15s", "30s", "60s"]
supported_aspect_ratios: ["16:9", "9:16", "1:1", "4:5"]
supported_quality_levels: []

settings:
  # The current distilled runtime exposes no real quality, motion or guidance
  # controls. Keep fake sliders out of the product until the guided path lands.
  quality: false
  motion_strength: false
  prompt_adherence: false
  seed: true
  # Standard / Idea (Director) prompt-mode toggle. Standard is the default and
  # is byte-identical to the pre-feature behaviour; Director treats the text
  # as an IDEA and has the worker plan characters, dialogue, delivery and
  # timing before generating (see apps/worker/worker/director/). Declared here
  # and on image-to-video (where the plan is source-anchored); no other
  # workflow accepts a prompt mode.
  prompt_modes: true

capabilities:
  download: true
  extend: true
  reuse_settings: true
  variation: true

ui:
  icon: sparkles
  thumb: "linear-gradient(140deg, #26320C, #151C08)"

execution:
  # PRIVATE — stripped from every public API response.
  runtime: mock

  # ── Single-pass ceiling: 30 seconds (measured, 20 Aug 2026) ──
  #
  # The GPU sustains a 60s pass, but the MODEL does not sustain 60s of story:
  # a single-pass 60s render had the departed man flicker back at 43-48s and
  # stand fully returned for the final twelve seconds while the soundtrack
  # said he was gone, and single-pass 60s dialogue never measured cleaner
  # than one repeated line in 22. Thirty-second passes are the measured-clean
  # regime (0 repeats), so 60s renders as two of them through the long-form
  # chain — the machinery that gives each section only its own events and
  # carries state across the seam. 5/15/30s remain byte-identical single
  # passes; only 60s changes shape.
  max_segment_seconds: 30

  # Deterministic prompt structuring (worker-side). The user's text reaches the
  # model verbatim as the first block; the worker appends derived continuity
  # rules (counts, colours, persistence) — the measured fix for identity and
  # colour drift on the unguided distilled runtime. Rules, not a language
  # model: it cannot paraphrase away a detail. See
  # apps/worker/worker/longform/enhance.py.
  prompt_structuring: true

  # ── Guided quality tier (OFF — enabling is a product/pricing decision) ──
  #
  # generation_engine: guided
  #
  # Swaps the distilled entry point for the guided two-stage pipeline (dev
  # transformer + distilled LoRA, unquantized, CPU offload). This is the tier
  # with CFG, spatio-temporal guidance and a negative prompt — the adherence
  # levers the distilled runtime structurally lacks — and the tier the
  # client's own reference engine measures "follows the prompt" against.
  # Measured 17 Aug 2026 (RTX PRO 6000, 1024x576, 121 frames): 146s vs the
  # distilled tier's 34s for the same clip — ~4.3x the render time — with
  # video AND audio verified in the output. Passes are capped at one measured
  # landing (121 frames / 5s); 241 frames is a reproduced decoder FAIL on
  # this pipeline, so raising the cap requires a new measurement, not an
  # opinion:
  #
  # guided_pass_seconds: 5.0

  # M1 runs the mock runtime, which emits a placeholder image rather than
  # rendered media. `output_type` above stays the product truth; these two lines
  # describe what the current runtime actually writes, and the API signs the
  # worker's upload for exactly this type. Remove both when M2 wires up a real
  # provider.
  output_content_type: image/png
  output_kind: image
```

---

## `workflow-definitions/image-to-video.yaml`

```yaml
id: image-to-video
version: "1"
name: Image to Video
category: video
output_type: video

description: Animate a still image into natural motion
short_description: Animate any image
marketing_description: Animate any still image into smooth, natural motion.

prompt:
  required: true
  placeholder: "Describe how the image should move…"
  # See text-to-video.yaml — effectively unlimited, capped only because an
  # unbounded request body is a denial-of-service surface.
  max_length: 20000

inputs:
  - role: source_image
    kind: image
    required: true
    label: INPUT IMAGE
    drop_hint: an image
    accept: ["image/jpeg", "image/png", "image/webp"]
    max_size_mb: 25

duration_mode: fixed
# The same ladder Extend Video offers (client revision, 13 Aug 2026). Lengths
# beyond one GPU pass are rendered as chained segments behind the scenes —
# that is the backend's business, and the option list must not shrink because
# of it. See apps/worker/worker/adapters/ltx.py, `_render_chain`.
supported_durations: ["5s", "10s", "15s", "30s", "60s"]
supported_aspect_ratios: ["16:9", "9:16", "1:1"]
supported_quality_levels: []

settings:
  quality: false
  motion_strength: false
  prompt_adherence: false
  seed: true
  # Standard / Idea (Director) prompt-mode toggle, same control Text to Video
  # declares. Standard is byte-identical to the pre-feature behaviour. In
  # Director mode the text is an IDEA and the worker plans dialogue, actions
  # and timing before generating — SOURCE-ANCHORED here: the uploaded image
  # defines who and what exists (and remains the identity reference across
  # chained sections), the plan defines only what happens next. See
  # apps/worker/worker/director/.
  prompt_modes: true

capabilities:
  download: true
  extend: true
  reuse_settings: true
  variation: true

ui:
  icon: image
  thumb: "linear-gradient(140deg, #33430D, #111708)"

execution:
  # PRIVATE — stripped from every public API response.
  runtime: mock

  # Single-pass ceiling: 30 seconds — same measured reason as
  # text-to-video.yaml (the model does not hold 60s of story in one pass;
  # 30s passes are the measured-clean regime). 60s becomes two chained
  # sections; every shorter duration is unchanged. The uploaded still opens
  # pass one and rides later passes as the identity reference, exactly as the
  # chain has always conditioned I2V.
  max_segment_seconds: 30

  # Deterministic prompt structuring — see text-to-video.yaml for the full
  # note and apps/worker/worker/longform/enhance.py for the rules.
  prompt_structuring: true

  # ── Guided quality tier (OFF — a product/pricing decision) ──
  #
  # generation_engine: guided
  # guided_pass_seconds: 5.0
  #
  # Same switch and same measured costs as text-to-video.yaml (see the full
  # note there): guided two-stage pipeline, CFG + STG + negative prompt,
  # ~4.3x the render time, passes capped at the one measured 121-frame
  # landing. I2V conditioning (`--image` stills, including the low-strength
  # identity reference on later passes) rides along unchanged — the guided
  # pipeline takes the same conditioning flags the distilled one does.

  # M1 runs the mock runtime, which emits a placeholder image rather than
  # rendered media. `output_type` above stays the product truth; these two lines
  # describe what the current runtime actually writes, and the API signs the
  # worker's upload for exactly this type. Remove both when M2 wires up a real
  # provider.
  output_content_type: image/png
  output_kind: image
```

---

## `workflow-definitions/video-to-video.yaml`

```yaml
id: video-to-video
version: "1"
name: Video to Video
category: video
output_type: video

description: Restyle and transform existing footage
short_description: Restyle footage
marketing_description: Restyle and transform existing footage with a prompt.

prompt:
  required: true
  placeholder: "Describe the transformation…"
  # See text-to-video.yaml — effectively unlimited, capped only because an
  # unbounded request body is a denial-of-service surface.
  max_length: 20000

inputs:
  - role: source_video
    kind: video
    required: true
    label: INPUT VIDEO
    drop_hint: a video
    # Same reasoning as music-video: the result matches the source's length, so
    # length is what decides the work, and 512 MB permits far more of it than
    # a job can complete. Worker-side ceiling is LTX_MAX_SOURCE_SECONDS.
    help: Up to 5 minutes. The result matches your video's length.
    accept: ["video/mp4", "video/quicktime", "video/webm"]
    max_size_mb: 512

  # Optional reference image. M1 shipped the contract only (accepted, stored,
  # handed to the worker as a weak look hint); since 19 Aug 2026 it drives
  # person identity — see `v2v_reference_identity` in the execution block.
  # The help line is the customer promise: it says who the person will be,
  # and sets the single-person expectation, because a source with several
  # people has all of them re-imagined (measured, not a guess).
  - role: reference_image
    kind: image
    required: false
    label: REFERENCE IMAGE
    drop_hint: an optional reference image
    help: Optional. The person in the result follows this image. Works best
      when your video shows one person.
    accept: ["image/jpeg", "image/png", "image/webp"]
    max_size_mb: 25

# Duration is automatic (CR-006): the result matches the uploaded source video,
# so the user picks nothing and the API rejects a supplied duration. A source
# longer than one model pass is segmented and stitched by the worker — invisible
# to the customer.
duration_mode: source
supported_durations: []
supported_aspect_ratios: ["16:9", "9:16"]
supported_quality_levels: []

settings:
  # These public controls did reach the job, but the distilled adapter never
  # read them. Private v2v_* settings below are the only real conditioning
  # controls and remain covered by command-level tests.
  quality: false
  motion_strength: false
  prompt_adherence: false
  seed: false

capabilities:
  download: true
  extend: true
  reuse_settings: true
  variation: true

ui:
  icon: repeat
  thumb: "linear-gradient(140deg, #222C10, #141A0C)"

execution:
  # PRIVATE — stripped from every public API response.
  runtime: mock

  # A source longer than one generation pass is restyled in sections, so the
  # wall-clock budget scales with what a customer may upload rather than with
  # what a single render costs. Without this the default 30 minutes fails an
  # honest two-minute source part-way through and the user is told the
  # generation "took too long".
  timeout_seconds: 5400

  # Restyle conditioning. These are quality judgements to be made against real
  # footage on the GPU, which is exactly why they are configuration and not
  # constants: see `_run_restyle` in apps/worker/worker/adapters/ltx.py for
  # what each one does. Omitted keys fall back to the adapter's defaults.
  #
  #   v2v_keyframe_seconds:    how much output one source still anchors
  #                            (default 4.0 — a DENSITY; the count per pass is
  #                            derived from it, bounded 3..16)
  #   v2v_keyframes:           fixed count per pass, overriding the density
  #                            above. For footage where the derived number is
  #                            wrong; normally leave unset.
  #   v2v_structure_strength:  how hard they pull (default 0.45 — lowered with
  #                            the density above; the two cannot be read apart)
  #   v2v_continuity_strength: the seam between passes (default 0.85)
  #   v2v_reference_strength:  the optional reference image (default 0.3)

  # ── The transform engine (opt-in) ────────────────────────────────────────
  #
  # `v2v_engine: transform` swaps the still-conditioned restyle above for
  # structure conditioning: an edge map of the source drives the IC-LoRA Union
  # Control adapter, and the prompt supplies all content.
  #
  # This is the answer to "the source survives but the requested restyling is
  # too weak". Stills carry the source's colour, light and material along with
  # its geometry, so the prompt spends itself fighting them; an edge map carries
  # geometry alone. Verified on the RTX PRO 6000 on 17 Aug 2026 — a daylight
  # desert plate returned as a rain-soaked neon street with the subject's pose,
  # the car's position and the road's perspective unchanged.
  #
  # ENABLED 17 Aug 2026. The still-conditioned restyle remains in the adapter
  # and is one line away — delete this key and it is back, unchanged. Keeping
  # the old path rather than deleting it is deliberate: this is a different
  # product behaviour, and reverting must not require a code change.
  v2v_engine: transform
  #
  # Its own knobs, all with adapter defaults:
  #   v2v_control_strength:    how hard the edge map pulls (default 1.0 — full,
  #                            unlike the restyle's stills; lowering it does not
  #                            transform harder, it stops tracking the footage)
  #   v2v_lora_strength:       Union Control adapter strength (default 1.0)
  #   v2v_edge_low/_high:      canny thresholds (defaults 0.1 / 0.4 — raise on
  #                            grainy footage that turns into speckle)
  #
  # Requires `loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` on the
  # node; a node without it refuses these jobs before spending any GPU time
  # rather than rendering an unconditioned pass nobody asked for.

  # ── Person lock (OFF — awaiting a full-length validation) ───────────────
  #
  # v2v_person_lock: true
  #
  # An edge map says WHERE a face is and nothing about whose it is, so a
  # restyle invents a new one — different features, a different skin tone —
  # even when the prompt never asked for a different person. Reported as
  # "the person looks slightly changed from the source video", and it is not
  # a tuning problem: continuity strength was swept at 0.85 / 0.95 / 1.0 on
  # 18 Aug 2026 and all three drifted identically, because appearance is not
  # what that dial controls.
  #
  # With this on, each pass mattes the people in its own window (BiRefNet,
  # MIT-licensed, run in the GPU environment), carries THEIR OWN PIXELS
  # inside the matte and keeps the edge map everywhere else, and weights the
  # subject's region above the rest with the pipeline's native attention
  # mask. The model then reproduces the person from their own appearance and
  # relights them for the new scene.
  #
  # Measured 18 Aug 2026, 8s window at 1024x576: face, skin tone, hair, beard
  # and the bird's real plumage all survived a dusk blizzard — 61s against the
  # edge-only path's ~54, so roughly 13% more compute for the pass plus the
  # matting itself.
  #
  #   v2v_background_attention:  how much the edge map still steers everything
  #                              that is not the subject (default 0.5 — toward
  #                              0 restyles harder and loses the camera,
  #                              toward 1 re-imposes the source everywhere)
  #
  # Requires the matting model on the node. NOT YET ENABLED: proven on one
  # 8-second window, not across a chained multi-pass job, and the matte's
  # behaviour on several people, occlusion and fast motion is unmeasured.

  # ── Reference identity (ON since 19 Aug 2026) ───────────────────────────
  #
  # GPU-verified the day it was built (research-2026-08-19): a 15s speaking
  # man + a reference photo of a woman returned HER — face, hair, jacket —
  # performing his speech in his kitchen, stable across pass seams; a 30s
  # multi-pass source held the identity to the last section. Costs ~27s of
  # matting per pass on top of the transform.
  v2v_reference_identity: true
  #
  # Person lock's mirror image: the person should NOT survive — the uploaded
  # reference image says who they are, the footage keeps saying what they do.
  # Three levers move together (transform engine only; the flag with a
  # reference on any other engine is refused rather than silently ignored,
  # and without a reference image it is inert):
  #
  #   * the reference anchors frame 0 of the first pass and is RE-SHOWN at an
  #     interior frame of every later pass — the I2V identity-persistence
  #     mechanism, because a continuity frame alone decays back toward the
  #     source person over chained sections;
  #   * each pass mattes its people (BiRefNet, as person lock) and weights the
  #     control signal BELOW the scene inside that region, so the canny edges
  #     stop re-imposing the source person's facial geometry while pose and
  #     placement still track the footage;
  #   * a matting failure fails the job — never a delivered video carrying
  #     the source person under a claim of replacement.
  #
  #   v2v_identity_describe_reference: the worker LOOKS at the reference
  #                                    (gemma vision, ~6s once per job) and
  #                                    appends a caption of the person after
  #                                    the user's verbatim prompt (default
  #                                    true). Exists because the first
  #                                    production identity job shipped a
  #                                    prompt of meta-instructions naming no
  #                                    visible attribute, and the render was
  #                                    neither the source nor the reference.
  #                                    Failures degrade to the bare prompt.
  #   v2v_identity_anchor_strength:    reference at frame 0, first pass
  #                                    (default 0.65)
  #   v2v_identity_refresh_strength:   reference re-shown per later pass
  #                                    (default 0.2 — 0.35 flashed the
  #                                    reference PHOTO into a customer video
  #                                    at a pass's interior anchor, 19 Aug;
  #                                    raise only with that failure in mind)
  #   v2v_identity_subject_attention:  edge-map grip over the person
  #                                    (default 0.5 — toward 1 re-imposes the
  #                                    source's face, toward 0 loses their
  #                                    motion)
  #
  # Mutually exclusive with v2v_person_lock — carrying both is refused.
  # Honest scope until the matrix says more: single/primary person; a
  # multi-person source mattes ALL people, so identity assignment there is
  # undefined. A versioned, server-enforced consent checkbox for identity
  # replacement (the reference engine's pattern) remains the recorded
  # recommendation; enabling without it was a deliberate product decision on
  # 19 Aug 2026. `scripts/v2v_identity_matrix.sh` re-measures these defaults
  # when footage says they need moving; the sweep clips backing the current
  # values are on the GPU box in /workspace/idtest/.

  # M1 runs the mock runtime, which emits a placeholder image rather than
  # rendered media. `output_type` above stays the product truth; these two lines
  # describe what the current runtime actually writes, and the API signs the
  # worker's upload for exactly this type. Remove both when M2 wires up a real
  # provider.
  output_content_type: image/png
  output_kind: image
```

---

## `workflow-definitions/extend-video.yaml`

```yaml
id: extend-video
version: "1"
name: Extend Video
category: video
output_type: video

description: Continue a generated video seamlessly
short_description: Continue any clip
marketing_description: Continue any generated video seamlessly, as long as you need.

prompt:
  required: true
  placeholder: "Describe what happens next…"
  # See text-to-video.yaml — effectively unlimited, capped only because an
  # unbounded request body is a denial-of-service surface.
  max_length: 20000

inputs:
  - role: source_video
    kind: video
    required: true
    label: SOURCE VIDEO
    drop_hint: a video
    # Far looser than the 5-minute rule on Music Video and Video to Video, and
    # deliberately so: those render the whole source, an extension renders only
    # the continuation. The source is kept as-is and continued from its final
    # frame, so its length prices as encoding, not generation. Worker-side
    # ceiling is LTX_MAX_EXTEND_SOURCE_SECONDS; keep the two in step.
    help: Up to 30 minutes. Your video is kept as-is and continued from its final frame.
    accept: ["video/mp4", "video/quicktime", "video/webm"]
    max_size_mb: 512

# The five outcomes of CR-008, plus the long end (client ask #1, 17 Aug 2026:
# "make video extension unlimited"). Unlimited-the-noun is the pair of facts
# that every result can itself be extended and that the source ceiling above is
# far past anything a chain of extensions produces in practice; this list is
# just how far ONE step can go. Longer entries are produced by chained
# generation behind the scenes — that is the backend's business, and the option
# list must not shrink because of it.
duration_mode: fixed
supported_durations: ["5s", "10s", "15s", "30s", "60s", "2m", "5m"]
supported_aspect_ratios: ["16:9", "9:16"]
supported_quality_levels: []

settings:
  quality: false
  # The motion of an extension is inherited from the source clip, so exposing a
  # motion-strength control here would be a slider that changes nothing.
  motion_strength: false
  prompt_adherence: false
  seed: false

capabilities:
  download: true
  extend: true
  reuse_settings: true
  # A "variation" of an extension is just another extension of the same source —
  # the Extend action already covers it.
  variation: false

ui:
  icon: extend
  thumb: "linear-gradient(140deg, #2C3A0B, #141A08)"

execution:
  # PRIVATE — stripped from every public API response.
  runtime: mock

  # Deterministic prompt structuring — see text-to-video.yaml for the full
  # note and apps/worker/worker/longform/enhance.py for the rules.
  prompt_structuring: true

  # A five-minute continuation is the same order of work as Music Video's
  # longest job (same chained passes, same assembly), and a long source adds
  # re-encoding time on top. The default 30-minute budget would kill exactly
  # the jobs the 2m/5m options above exist to allow.
  timeout_seconds: 7200

  # M1 runs the mock runtime, which emits a placeholder image rather than
  # rendered media. `output_type` above stays the product truth; these two lines
  # describe what the current runtime actually writes, and the API signs the
  # worker's upload for exactly this type. Remove both when M2 wires up a real
  # provider.
  output_content_type: image/png
  output_kind: image
```

---

## `workflow-definitions/music-video.yaml`

```yaml
id: music-video
version: "1"
name: Music Video
category: audio
output_type: video

description: Pair audio with visual direction
short_description: Audio meets visuals
marketing_description: Pair audio with visual direction to produce full music videos.

prompt:
  required: true
  placeholder: "Describe the visual direction…"
  # See text-to-video.yaml — effectively unlimited, capped only because an
  # unbounded request body is a denial-of-service surface.
  max_length: 20000

inputs:
  - role: source_audio
    kind: audio
    required: true
    label: AUDIO
    drop_hint: an audio track
    # The length limit belongs HERE, before the upload, not in the refusal
    # afterwards. The video is generated in sections across the whole track, so
    # length is the single thing that decides how much work the job is — and
    # the size cap alone permits an hour of audio, which is a job that cannot
    # finish. Worker-side the ceiling is LTX_MAX_SOURCE_SECONDS; keep the two
    # in step.
    help: Up to 5 minutes. The video is generated to match your whole track.
    accept: ["audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/ogg"]
    # 200 matches the platform's audio ceiling (see MAX_SIZE_BYTES in
    # apps/api/app/services/storage.py). The old 64 was sized for MP3s and
    # silently refused honest lossless uploads — a five-minute stereo WAV is
    # 50-85 MB. Bytes were never the work bound anyway; the 5-minute LENGTH
    # rule above is, and it is enforced before any compute is spent.
    max_size_mb: 200

# Duration is automatic (CR-007): the final video matches the full uploaded
# track — a 30-second song yields a 30-second video, a 4-minute song a 4-minute
# video. Long tracks are produced in sections and assembled by the worker.
duration_mode: source
supported_durations: []
supported_aspect_ratios: ["16:9", "9:16", "1:1"]
supported_quality_levels: []

settings:
  quality: false
  motion_strength: false
  prompt_adherence: false
  seed: false

capabilities:
  download: true
  # The output is cut to the supplied track, so extending it past the audio is
  # not meaningful.
  extend: false
  reuse_settings: true
  variation: true

ui:
  icon: clapper
  thumb: "linear-gradient(140deg, #1C240C, #182008)"

execution:
  # PRIVATE — stripped from every public API response.
  runtime: mock

  # Deterministic prompt structuring — see text-to-video.yaml for the full
  # note and apps/worker/worker/longform/enhance.py for the rules.
  prompt_structuring: true

  # The longest job the product has. A four-minute track is eight visual
  # sections plus assembly and muxing, and the default 30-minute budget would
  # kill it around section three — after the compute had already been spent.
  timeout_seconds: 7200

  # Visual cuts are pulled back onto a rise in the track's energy where one is
  # available, so a section change lands on a hit rather than mid-phrase. This
  # is event alignment, not beat tracking; set it false for evenly spaced
  # sections. See worker/longform/timing.py.
  align_cuts_to_audio: true

  # ── Audio conditioning (opt-in) ──────────────────────────────────────────
  #
  # By default the MODEL NEVER HEARS THE SONG. The track decides the length and
  # where the cuts land; the picture is drawn from the prompt alone and the song
  # is laid over the finished result. That is why a singer in the output does
  # not move their mouth in time with the vocal — it is structural, not a
  # prompting problem, and no wording fixes it.
  #
  # `audio_conditioning: true` renders each section on the audio tier instead,
  # handing it the master track seeked to that section's own moment in the song
  # (the pipeline takes an offset, so the master is never sliced or re-encoded).
  # This is the same mechanism the client's reference engine uses, and it is
  # what makes audio-driven performance possible at all.
  #
  #   audio_conditioning: true
  #
  # WHAT IT COSTS. Measured on the RTX PRO 6000 at 1024x576 — 213s for a
  # 20.04-second pass, reproduced 17 Aug and 21 Aug 2026, so **10.6x real
  # time**. Against the default tier's measured 3.6x (from real production
  # jobs: a 300s track in 1085s, a 240s track in 863s), that is about three
  # times the compute. A five-minute song is ~53 minutes of GPU. It fits the
  # 2-hour budget above and it changes the unit cost of the workflow, which is
  # why this is a pricing decision and not simply switched on.
  #
  # WHY it costs that: the audio tier cannot run quantized (the distilled LoRA
  # it requires clashes with FP8/NVFP4 fusion), where the default tier serving
  # today's 3.6x runs NVFP4. Three times the cost is what "the model can hear
  # the song" costs on this hardware, and the knobs barely move it — the
  # guidance passes batch into ONE transformer call, so dropping one is not
  # worth a quarter. `--offload none` is the largest measured lever (23-30%)
  # and needs VRAM this card does not have spare.
  #
  # RELIABILITY, which matters more than the cost. The tier passed 6/6 at a
  # steady 208s with the GPU to itself, and failed 5 of 15 times at the same
  # frame count when the card was shared. A five-minute video is 15 CONSECUTIVE
  # passes and the chain has no per-pass retry, so per-pass reliability has to
  # be near 1. Read the 21 Aug research note before switching this on.
  #
  # WHAT IT DELIVERS, honestly: visuals that move with the music, and a
  # performer whose mouth follows the vocal. It is NOT verified phoneme-accurate
  # lip-sync.
  #
  #   audio_pass_seconds:  section length on the audio tier. Default is 481
  #                        frames' own duration (20.0417s) — a MEASURED landing
  #                        rather than a round number, so the planner's nominal
  #                        window is a count the decoder is known to survive.
  #                        The distilled tier's 60s does not transfer.
  #   inference_steps:     stage-1 denoising steps (pipeline default 30). The
  #                        one lever that moves wall time; a quality trade.
  #   a2v_guidance_scale:  how hard the model is pushed toward the audio
  #                        (pipeline default 3.0; 1.0 disables it entirely).
  #                        LTX's own help: "higher values may increase lipsync
  #                        quality".
  #   guidance_scale / stg_scale: CFG and spatio-temporal guidance. Cheap to
  #                        drop and they cost picture quality, not speed.
  #
  # Requires the dev transformer and the distilled LoRA on the node.

  # M1 runs the mock runtime, which emits a placeholder image rather than
  # rendered media. `output_type` above stays the product truth; these two lines
  # describe what the current runtime actually writes, and the API signs the
  # worker's upload for exactly this type. Remove both when M2 wires up a real
  # provider.
  output_content_type: image/png
  output_kind: image
```

---

## `workflow-definitions/music.yaml`

```yaml
id: music
version: "1"
name: Music
category: audio
output_type: audio

description: Generate original tracks from a description
short_description: Original tracks
marketing_description: Generate original tracks from a mood, style or description.

prompt:
  required: true
  placeholder: "Describe the track — mood, style, tempo…"
  # See text-to-video.yaml — effectively unlimited, capped only because an
  # unbounded request body is a denial-of-service surface.
  max_length: 20000

inputs: []

# The user picks a song length in minutes (CR-009) — music-appropriate, not
# video-style second presets. The 5-minute ceiling is PROVISIONAL: the safe
# maximum comes from benchmarking the selected music model in M2, and raising
# it is an edit to this list only. Long songs may be generated in continued
# sections and assembled — hidden from the customer.
duration_mode: minutes
supported_durations: ["1m", "2m", "3m", "4m", "5m"]
# Audio has no frame. The API rejects an aspect ratio on this workflow and the
# settings panel hides the section entirely.
supported_aspect_ratios: []
supported_quality_levels: []

settings:
  quality: false
  motion_strength: false
  prompt_adherence: true
  seed: true
  # A lyrics box (the customer's own words, never rewritten — the model sings
  # whatever language the sheet is in, 50+ supported) and a language choice
  # for generated lyrics.
  lyrics: true

capabilities:
  download: true
  # Audio cannot be extended — Extend must never appear on a music result.
  extend: false
  reuse_settings: true
  variation: true

ui:
  icon: music
  thumb: "linear-gradient(140deg, #18200A, #10160A)"

execution:
  # PRIVATE — stripped from every public API response.
  #
  # `music` is the runtime once a model is chosen. It stays `mock` because no
  # music model has been selected yet (docs/milestones.md tracks it as an open
  # decision) — the adapter, sectioning, lyric pass and validation are built and
  # tested; only the model is missing, and it plugs in through MUSIC_LAUNCHER.
  runtime: mock

  # A five-minute song is several generations plus crossfading and loudness
  # matching. Sized for the ceiling of the offered range.
  timeout_seconds: 3600

  # M1 runs the mock runtime, which emits a placeholder image rather than
  # rendered media. `output_type` above stays the product truth; these two lines
  # describe what the current runtime actually writes, and the API signs the
  # worker's upload for exactly this type. Remove both when M2 wires up a real
  # provider.
  output_content_type: image/png
  output_kind: image
```

