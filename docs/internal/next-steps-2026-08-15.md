# Next session — planned work

**Written:** 14 August 2026, end of the migration session
**Context:** production moved to the RTX PRO 6000 and music went live the same
day; a dimension-dependent kernel bug was found and worked around. See
[`gpu-worker-runbook.md`](./gpu-worker-runbook.md) §33–§37 and
[`issue-triton-na-kernel.md`](./issue-triton-na-kernel.md).

---

## 0. State at end of 14 Aug

| | |
|---|---|
| Production GPU | RTX PRO 6000 Blackwell, instance `47698594`, $1.310/hr. 5090 destroyed. |
| Worker | `ltx-6000-1`, runtimes `["ltx", "music"]`, `LTX_MAX_SECONDS=10` |
| Supervision | tunnel / worker / music all under supervisord, restart verified |
| Video | works across the full duration × aspect matrix, via chaining |
| Music (audio) | live, 2-minute song in ~15s |
| Music Video | passes on GPU at the new ceiling; re-routing to `ltx` in progress |

---

## 1. P0 — risk, do first

### 1.1 Commit the ACE-Step polling fix

`worker/music/acestep.py` `_poll_once` was patched to treat only entries
carrying `file` as finished. **Applied on the GPU box and in the local repo,
committed nowhere.** Any rebuild from git silently reintroduces
`the music service reported success but returned no audio`.

Wants a regression test — the fake provider in `tests/test_music.py` can
reproduce it by returning a progress-shaped record before a finished one.

### 1.2 Confirm what the VPS is actually running

The runbook records the VPS at commit `516b455`; GitHub `main` is at `3b9464c`,
which contains two client-facing bug fixes:

```text
643057d  fix(web,api): results display at their real shape and length
a1078ea  feat(web): landing revisions and a canvas that clears when work finishes
```

**If the VPS is behind, the client's reported display bugs were never deployed.**

```bash
cd /opt/zolexai && git log --oneline -1
```

### 1.3 Verify the three untested video workflows

Live and reachable by customers, never run on this hardware:

```text
image-to-video      extend-video      video-to-video
```

Fixtures are on the box at `/workspace/fixtures/`. Commands are in
`gpu-worker-runbook.md` §32.

### 1.4 Music Video at a realistic length

Verified at 30s (4 sections). A 2-minute track is **12 sections and 11 seams**,
6–8 minutes of generation, and nobody has watched one. Do not treat it as
generally available until someone has.

### 1.5 Decide the routing-vs-repo question

Six workflows are routed to real runtimes **only on the VPS**, uncommitted. The
repo still says `runtime: mock` everywhere. That drift is deliberate and
documented, but it means a rebuild from git produces a mock-only deployment.

---

## 2. P1 — generation speed

### 2.1 The measurement that frames everything

| Job | Passes | Wall | Per pass |
|---|---|---|---|
| 10s | 1 | 36.2s | 36s |
| 30s 9:16 | 3 | 89.6s | ~30s |
| 60s 9:16 | 6 | 179.6s | ~30s |

**Each pass costs ~30s regardless of how much video it produces.** That is fixed
overhead — process startup and model loading — not diffusion. The 5090 notes say
the same: *"~60s wall incl. ~45 GB weight reload per job"*.

So chaining is not slow because of the extra passes. It is slow because **every
pass reloads the model.**

**Measure first**, before building anything:

```bash
grep -iE "load|weights|elapsed|took|seconds" /tmp/p10-60s.log | head -20
```

- Load ≈ 20s of each 30s pass → §2.2 is the highest-value change available
- Load ≈ 5s → diffusion dominates, and §2.3 becomes the priority instead

### 2.2 Persistent LTX service (est. 2–3× faster)

The adapter shells out to the pipeline per pass, paying startup every time. Hold
the model resident in a long-running process and send it requests instead.

**The pattern is already proven in this codebase** — ACE-Step does exactly this:
~40s once, then 1.5s per request. 96 GB makes keeping LTX resident alongside it
trivial (~28 GB video + ~24 GB music = ~52 of 96).

Potential: 60s video from ~180s to ~60s.

Real engineering work. Also changes the failure model — a resident service that
dies takes all video with it, so it needs the supervisord treatment from §37.

### 2.3 NATTEN — fix the root cause

Removes the need to chain at all rather than making chaining cheaper. Faster
**and** eliminates every seam. See `issue-triton-na-kernel.md` §7.2.

Cheap to test, uncertain payoff. Timebox to an hour. First 2 minutes decide it:

```bash
grep -rn "natten" /workspace/ltx2-benchmark/pyproject.toml \
  /workspace/ltx2-benchmark/packages/*/pyproject.toml
sed -n '/dependency-groups/,/^\[/p' /workspace/ltx2-benchmark/pyproject.toml
```

**§2.2 and §2.3 compound.** Both together could mean single-pass 60s with no
reload — under a minute, versus three today, with zero seams.

### 2.4 Per-shape ceilings (easy, ~33% for landscape)

`LTX_MAX_SECONDS` is global and set to 10 to satisfy 9:16. But 16:9 tolerates
20s, so landscape is being split three ways when two would do. `output_dimensions`
in `adapters/ltx.py` already computes the shape — derive the ceiling from it.

Measured ceilings (fresh passes):

```text
16:9   896x512   20s
9:16   512x896   10s
1:1    640x640   >=15s   (upper bound not established)
4:5    512x640   >=15s   (upper bound not established)
```

Note **any conditioning lowers the ceiling** — continuation frames *or* audio —
so per-shape values must be validated in the conditioned case, not just fresh.

---

## 3. P2 — quality and product

### 3.1 Dev checkpoint — the prompt-adherence answer

42 GB on disk, **never run**. The non-distilled transformer exposing guidance
scale, negative prompt and step count — the actual fix for the client's
*"something missing in every video"*. Expect ~10× slower; wire it to the existing
`supported_quality_levels` so Standard stays fast and High/Ultra takes the slow
path. See [[ltx-quality-tuning-options]].

### 3.2 Re-measure the inherited ceilings

`_PIXEL_BUDGET = 896*512` and the `_DIMENSIONS` grids in `adapters/ltx.py` were
sized for the 5090's 32 GB. Higher resolutions may now be affordable — but
measure, do not assume, and remember the kernel bug is dimension-sensitive.

### 3.3 Music polish

- Surface the lyric-density warning in the UI (silent verse-dropping on
  customer-supplied lyrics)
- Expose `bpm` / `key` / `instrumental` / `vocal_type` in the public API — the
  service supports them, ZolexAI does not offer them
- Long intros: ~30s before vocals on a 120s track, tunable via structure tags

### 3.4 Reboot survival

Supervisord should bring everything back, but only a console stop/start proves
it. Untested.

---

## 4. Open questions

- Does music-video's audio conditioning have a *different* ceiling from frame
  conditioning? Both lower it; the amounts are unmeasured.
- Are 1:1 and 4:5 ceilings above 15s? Only lower bounds are established.
- Is a 12-section music video visually acceptable, or does seam count become the
  binding product constraint on track length?

---

## 5. The lesson from 14 Aug, worth re-reading before any hardware change

The migration was validated with **one 10s 16:9 clip**. The product is five
durations × four aspect ratios — one of twenty cells was checked, and it passed.
Each subsequent fix was also wrong: `LTX_MAX_SECONDS=20` failed on chaining,
`15` failed on portrait.

**Exercise the actual parameter matrix, not a representative sample.**
