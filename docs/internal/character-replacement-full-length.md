# Character Replacement over the whole source — a chain of windows

**Status: implemented, tested, deployed to the GPU node and validated on the
live deployment with the client's own 32.7 s clip (6 Sep 2026, §7): four
windows, three seams, none visible as a cut, delivered length exact.**

Client request (6 Sep 2026): "character replacement works really well at
10 s; we want it for the length of the source video, like Video to Video,
without breaking what works."

## 1. Why there was a limit

One run of the client's graph (`ltx25_character_replacement.json`) follows
a fixed number of seconds set by its `Set Length (seconds)` constant. On
the RTX PRO 6000 a 10 s window peaks at 85 GB of VRAM and 110 GiB of
container RAM; a 20 s window was killed by the container's 241 GiB limit
(`ltx25-gpu-benchmark.md`, "Character Replacement ceiling"). So the worker
cut every source to 10 s. The graph's author's own note lists 5, 10 or 20 s
— it was never written for a whole video.

## 2. Research: what the graph allows

Read from the frozen graph, not assumed:

* **The loader resamples every source to 24 fps.** `VHS_LoadVideoFFmpeg.force_rate`
  is wired to the graph's "Set FPS" primitive (24), and its `frame_load_cap`
  to the frame formula `round((24·s − 1)/8)·8 + 1`. So a window cut on a
  24 fps grid is exactly what the graph would have made of that stretch.
* **The graph takes ONE reference picture, and that picture is the first
  frame it renders** (measured 5 Sep on the delivered sample: frame 0 is the
  photo, frame 4 onward is the source's motion). That is the only
  continuity mechanism the graph has — and it is enough: seed window k
  with the last frame window k−1 produced and the graph continues the
  character in the setting, at the pose the motion had reached, instead of
  starting again from the photo's pose.
* **The soundtrack is the source's, passed through** (the combiner's audio
  input is an `ifElse` fed by the loader's audio; the model also encodes it
  for conditioning). Laying the source's own track over the joined result
  reproduces that exactly, with no seam in the sound.
* **Video to Video's long-source machinery is the same idea**: even
  windows (`plan_segments`), each pass conditioned on the previous pass's
  final frame, one frame of overlap dropped at the seam, the parts joined,
  the sound laid over the whole (`worker/longform/chain.py`,
  `continuation.py`). This module reuses the rule set, not the CLI runtime
  — Video to Video itself is untouched.

Rejected: changing the loader's `start_time`/`frame_load_cap` directly
(the client's rule allows the worker to set prompt, seed, duration, input
media and output location, nothing else — cutting the window into its own
input file stays inside that rule); lowering the canvas or precision to fit
a longer window (forbidden: "do not solve a hardware limitation by silently
changing resolution"); cross-fading windows (the graph has no second image
to fade towards).

## 3. Design

```text
source (any fps, any length)
   │ probe → T = whole seconds, capped by max_total_seconds (default 120)
   │ plan_windows(T, 10): even whole-second windows, e.g. 25 s → 9 + 8 + 8
   │ window k starts on window k−1's LAST frame (one shared frame)
   ▼
for each window k:
   cut  zolex_<job>_windowKK.mp4  — exactly N_k frames at 24 fps from
        start_frame_k, the source's sound for that stretch (silence if none),
        a held final frame where the formula overshoots the source's tail
   ref  k = 0: the customer's photo
        k > 0: the last frame of window k−1's OUTPUT (mode previous_frame)
               or the photo again (mode photo)
   run  the unchanged graph: Set Length = s_k, canvas as before, same seeds
   ▼
join: drop frame 0 of every window after the first (it is the reference,
      rendered again), re-encode all parts alike, concat (stream copy)
sound: the source's own track over the whole picture, padded/silent as needed
verify: measured length = Σ kept frames / 24  (25 s → 601 frames = 25.04 s)
write: character-replacement.json — windows, references, seams, lengths
```

A source within one window (≤ 10 s) runs the path that ran before, byte for
byte: the clip uploaded as-is, one run, output prefix `output`. Pinned by
`test_a_source_within_one_window_runs_the_path_that_ran_before`.

Nothing in the graph changes per window: same nodes, models, LoRA and
canvas; the graph's own fixed seeds every window (a customer seed becomes
seed + window index, so windows differ the way passes do elsewhere).

## 4. Settings

| Setting | Default | Meaning |
|---|---|---|
| `character_replacement_max_seconds` / `execution.max_seconds` | 10 | the longest window one run follows (unchanged meaning: the measured ceiling) |
| `character_replacement_max_total_seconds` / `execution.max_total_seconds` | 120 | the longest source followed in total; above it the tail is not followed and the log says so |
| `character_replacement_chain_reference` / `execution.chain_reference` | `previous_frame` | `previous_frame`: continuity across seams, possible identity drift on very long chains; `photo`: no drift, a pose snap at every seam |
| `character_replacement_expected_wall_per_output_second` | 30 | progress pacing only |
| YAML `timeout_seconds` | 10800 (was 5400) | 12 windows × ~5.5 min plus cutting and joining, with margin |

**Why 2 minutes and not Video to Video's 5.** Measured cost is 323 s per
10 s window (164 s for an 8 s window). Two minutes of source is 12 windows
≈ 66 minutes on the node, which serves one job at a time for every
customer. Five minutes would hold the node for close to three hours. The
cap is one number to raise (`CHARACTER_REPLACEMENT_MAX_TOTAL_SECONDS` on
the worker, `max_total_seconds` in the definition's execution block);
raise `timeout_seconds` with it (≈ 55 s of budget per second of source).

An 8 s window costs 20.5 s per source second against 32 for a 10 s window
on the two measurements to date. If that holds under repetition, an 8 s
`max_seconds` makes long chains ~35 % faster at the price of more seams;
measure before switching — it is one setting.

## 5. Failure behaviour

* A window that fails mid-render fails the job with the same retriable
  error as before; no assembled output exists; the source is untouched.
* A window that cannot be cut (unreadable source) fails before any GPU
  time, non-retriable.
* A window whose output cannot be read for its last frame is a generation
  flake, retriable.
* Cancellation is honoured between windows and inside every ffmpeg step.
* The dead-ComfyUI fail-fast (`1554bc4`) applies to every window.

## 6. Tests (6 Sep 2026, no GPU)

`apps/worker/tests/test_character_replacement_chain.py` — 8 tests against
the fake ComfyUI with real ffmpeg files: the planner (even whole-second
windows, one shared frame, the formula's frame counts, the delivered total);
the ceiling logic; a 20 s source as two windows chained on the last frame
(uploads, per-window graph edits, cut clips on the 24 fps grid with sound,
481 delivered frames with one audio track, metadata, section progress);
an uneven 25 s source as 9 + 8 + 8; photo mode; the total ceiling; a silent
source; a failing window; and the short-source path unchanged.
`test_character_replacement.py`'s old "cut to the ceiling" test now pins
the chain. All 20 pass. The fake ComfyUI gained a per-window output queue.

## 7. GPU validation (6 Sep 2026, live deployment, job `597b8113`)

The client's own latest character-replacement source — `49709.mp4`, 32.73 s,
576×1024, 30 fps — with the client's own reference picture and prompt, run
through the live queue while it was otherwise idle.

| | |
|---|---|
| Plan | 32 whole seconds → 4 windows of 8 s, 193 frames each, seams at 8.0 / 16.0 / 24.0 s, reference mode `previous_frame` |
| Per window | 157.7 / 153.3 / 153.7 / 153.7 s wall (models resident after the first) |
| Whole job | 638 s end to end, including four cuts, four uploads, three last-frame extractions, the join and the soundtrack |
| Output | 736×1280 · 24 fps · **769 frames = 32.0417 s** (promised 32.042) · one audio track (the source's) |
| GPU / RAM during windows | 43 GB VRAM at the sampler, node RAM 89 GB used, no growth across windows |

**Seams, measured.** Frame-to-frame PSNR over the whole clip has a median
of 21.3 dB and a minimum of 16.0 dB (frame 287, inside a window — ordinary
fast motion). At the three seam frames the frame-to-frame PSNR is 19.3,
23.8 and 19.7 dB: inside the clip's own motion band, nowhere near a cut.

**Seams, looked at** (7-frame strips around each seam, kept with the
session's files): pose, framing, jewellery and tattoos continue across all
three seams; the graph's four-frame handoff is not visible as a jump. Two
observations for the record:

1. The client's reference picture carries a burned-in caption; window 0
   reproduces it as illegible text for its whole 8 s (as the single-window
   product does today), and window 1 lets it fade out over its first
   frames after the seam. That is content the customer put in the picture,
   not a seam artefact — but it is the one visible change at seam 1.
2. Identity drifts mildly over 32 s: the face and the jersey emblem in
   windows 3–4 sit a little further from the photo than window 1 does —
   the expected cost of seeding each window from a rendered frame. The
   `photo` reference mode is the lever if a customer prefers exact
   likeness over pose continuity; not yet compared on the GPU.

**Cost, re-measured.** 8 s windows cost 19.4 s of GPU per second of source
(155 s per window) against 32 s/s for the one 10 s window measured on
6 Sep (323 s). A 2-minute source at 8 s windows would be 15 windows ≈ 39
minutes; at 10 s windows 12 windows ≈ 65 minutes. The cap stays 10 s
because the client validated that window's quality and fewer seams is
the safer default; `character_replacement_max_seconds=8` is the one-line
speed option, to be judged on seams.

Still open: the `photo` mode comparison, and a 60 s+ chain for drift and
memory over more windows.

## 8. Rollback

The previous behaviour (cut to 10 s) is `git checkout <this commit>^ --
apps/worker/worker/adapters/character_replacement.py apps/worker/worker/core/config.py
workflow-definitions/character-replacement.yaml`, or set
`CHARACTER_REPLACEMENT_MAX_TOTAL_SECONDS=10` on the worker, which makes
every source a single window again without a code change.
