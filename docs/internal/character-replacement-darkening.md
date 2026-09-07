# The darkening in Character Replacement, and the client's MULTI4 fix

*7 Sep 2026. Source: `LTX2.5_Character-Replacement_MULTI4_FHD_COMBINED.json`,
sha256 `14561ea05c5f5065f5507524a350138bd56a87a096c4ec54bba802c603f09270`,
240,716 bytes, preserved unedited in `benchmarks/client-pack/ltx25/client_original/`.*

## The graphs are the same graph, twice, apart from two values

Every node, every link, every sampler setting, both VAEs, the CLIP, the
sigmas, the shift, the canvas and both guiders are identical between the
client's new graph and the one Character Replacement already runs. Two
widgets differ, and nothing else:

| | ours (`ltx25_character_replacement.json`) | client MULTI4 |
| --- | --- | --- |
| transformer (node 821) | `...22b-distilled-transformer-comfy-int8-convrot` | `...22b-distilled-transformer-nvfp4` |
| Ripple LoRA strength (node 1312) | 1.35 | 1.45 |

That is the whole model-level difference. The rest of their file is new
*plumbing* — an aspect switch, an FHD crop pair, and a post-render face-lock
branch — not new generation settings.

## The transformer change points the wrong way

We measured these two builds against each other on 6 Sep, on our Text to
Video graph, at 30 seconds and a fixed seed
(`client-ltx25-fast-1080-validation.md` §8, `ltx-t2v-legacy-adapters.md`):

> int8 190.3 s against NVFP4 192.1 s, the same speed, with **NVFP4 visibly
> darker and less saturated**.

So a change offered as the fix for darkening is, on our own measurement of
the same two files, a move towards the darker of the two. Character
Replacement is already on int8 — the lighter one — and has been all along.

**This is evidence, not proof, and the difference matters.** That comparison
ran on the Text to Video graph, not this one. A quantisation's effect on
exposure is not guaranteed to carry between graphs, and their file was
presumably built and judged on NVFP4 hardware where it looks right to them.
What it does establish is that adopting their transformer should be an A/B
with a luma measurement, not a copy — and that we should not describe it to
the client as a darkening fix until that A/B says so.

The LoRA move, 1.35 → 1.45, is untested here. Their own note reads:
"Increase the LoRA strength when the edit is not being carried through
strongly enough. Decrease it when the output changes content that should
remain unchanged." Exposure is content that should remain unchanged, so on
their own guidance 1.45 is the direction that risks *more* drift, not less.
`character_replacement_ripple_strength` already exists to test it.

## What in their pack IS a darkening fix

Three things, and two of them are now carried.

### 1. The negative names a direction — ADOPTED

Every exposure word our negative already had was symmetrical: "brightness
shifts", "exposure pumping", "skin-tone shifts", "white-balance shifts". Each
names a *change* without naming a way. The fault the client reported only
ever goes one way, and on an unguided runtime a symmetrical word gives the
model nothing to lean against.

Their negative says it outright, and ours now carries the same terms:
`darker face`, `darker person`, `darkening skin`, `underexposure`,
`crushed blacks`, `inconsistent face color`, `inconsistent hand color`. The
symmetrical words stay — flicker is a real and different fault.

### 2. A positive exposure lock — ADOPTED

`CHARACTER_REPLACEMENT_EXPOSURE`, condensed from their two sentences: the
scene's light is the reference video's own, and the character's face, neck,
arms and hands keep the brightness they had in the first frame.

Kept **relational**, like the hands clause beside it and for the same reason:
it names no colour and no brightness, only sameness. A clause saying "bright
skin" would pull every character towards one complexion, which is a worse
fault than the one being fixed and one no customer would forgive. A test
pins that it names none.

On by default (`character_replacement_exposure_clause`) because the client
asked for it, switchable per job so an A/B needs no redeploy, and applied to
single-window jobs as well as chained ones — the darkening was measured
*within* a window, not only across seams.

### 3. Never feed generated video back in — NOT ADOPTED, AND THIS IS THE BIG ONE

Their backend contract, twice, in their own words:

> `source_video`: always use the untouched original. **Never use a generated
> output as the next input.**

> `color_rule`: **never recursively process generated video**; repeated VAE
> passes compound darkness and identity distortion.

Our chain does exactly that. `character_replacement_chain_reference` defaults
to `previous_frame`: window *k* is seeded with the last frame window *k−1*
produced, decoded from its MP4 and re-encoded as the next window's reference.
Every window is therefore a generation of a generation, and the client is
naming that as the mechanism that compounds darkness.

We independently reached the same diagnosis: the 7 Sep skin-hold work found
the darkening present **at the seed** as well as per delivered frame, which is
what a compounding chain looks like from the other end. `skin_hold` corrects
it afterwards. Their design removes the cause.

**Why it is not simply switched.** The graph's reference picture *is* the
first frame of what it renders, so a window has to be seeded with something
that already shows the new character in the right pose. There are three
candidates and each costs something:

| seed | recursion | seam |
| --- | --- | --- |
| `previous_frame` (today) | yes — compounds | continuous; pose carries |
| `photo` (exists, untested on GPU) | none | pose snap at every seam |
| source frame + composited replacement (theirs) | none | continuous |

Their answer is the third, and it is the only one with neither cost. It needs
a backend stage this platform does not have: detect and track the people in
each chunk's first source frame, and composite the replacement identity onto
each one. Their own note says that belongs in backend preprocessing, not in
the graph — they are describing work to be built, not shipped code.

`photo` is the non-recursive option available today. It satisfies their rule
at the price of a hitch at each seam, and the comparison between it and
`previous_frame` has been waiting for GPU time since 6 Sep.

## The face lock cannot ship as delivered

Their post-render branch (nodes 1563-1570) corrects faces with ReActor and
blends only the masked face region back. Their own note states the blocker:

> ReActor documents the pre-trained InsightFace `inswapper_128.onnx` /
> `buffalo_l` assets as **non-commercial research models**. Replace them with
> models whose license permits production before deploying.

So the branch is a design, not a deliverable, until a commercially licensed
face-swap and analyser pair is chosen. It also needs `face_yolov8m.pt` and
`sam_vit_b_01ec64.pth`, plus two custom node packs. Nothing here should be
enabled on the client-test node on the strength of the graph alone.

## Also not adopted (out of scope for darkening)

* **1-4 people.** Needs the same detection/tracking/compositing stage as the
  non-recursive seed, and is the larger half of their file.
* **The aspect switch and FHD delivery** (1080×1920 / 1920×1080). Our canvas
  is the pack's 736×1280. A separate change with its own memory and time
  cost, unrelated to exposure.
* **8-second chunks.** Ours are 10, from the measured VRAM ceiling. Theirs is
  a smaller number, and moving to it is only meaningful alongside the
  non-recursive seed.

## What to do next, in order

1. **A/B the two seeds** on the GPU: `previous_frame` against `photo`, same
   source, same photo, same seed, `skin_hold` off in both so the raw
   mechanism is visible, measuring mean luma per window. This decides whether
   the recursion is worth its seam.
2. **A/B the exposure clause** on and off, same way. It is on by default now;
   the measurement should follow it rather than lead it.
3. **Only then** consider the transformer, against §"points the wrong way".
4. **Scope the compositor** if the client wants their design proper — it is
   the gate on multi-person, on the non-recursive seed without a seam, and on
   the face lock, and it needs a licence decision before any model is fetched.
