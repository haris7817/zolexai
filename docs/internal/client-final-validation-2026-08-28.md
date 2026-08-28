# Final client iteration — deploy and validation, 28 Aug 2026

Companion to `research-2026-08-28-final-client-fixes.md`. That document says
what was wrong and why; this one is the order of operations, and the shortest
list of GPU checks that can honestly clear the build.

Nothing here has been run on a GPU. Read §3 before promising the client
anything.

---

## 1. Deploy

Two halves, as always, and they are not interchangeable.

### 1.1 GPU worker

```bash
cd /workspace/zolexai && git pull --ff-only && supervisorctl restart zolexai-worker
```

Then confirm the four changed behaviours are actually in the checkout:

```bash
grep -c "starting with their age in years" apps/worker/worker/director/vision.py   # 1
grep -c "_SUBJECT_HELD"                    apps/worker/worker/longform/music_video.py  # >= 3
grep -c "i2v_describe_identity"            apps/worker/worker/adapters/ltx.py       # 1
grep -c "h3_comfy_video_to_video"          apps/worker/worker/core/config.py        # >= 1
```

**Node prerequisites** — each of these is a silent degradation, not a crash,
so check them rather than assuming:

| Needed by | File | Missing means |
| --- | --- | --- |
| Best V2V identity | `scripts/person_anchor.py` + BiRefNet weights | anchor falls back to the raw photo; weaker identity |
| every identity caption | `models/gemma-4-e2b-it` (as an image-text model) | caption is `""`, prompts read as before the fix |
| V2V transform | `loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | job refused before spending GPU time |

`h3_comfy_video_to_video` needs no action: it defaults off, which is the
intended state.

### 1.2 VPS (api + web)

The YAML edits are the point of this deploy, so the §16 dance is mandatory and
`video-to-video.yaml` is again the file that conflicts.

```bash
cd /opt/zolexai
git diff                      # READ FIRST — six locally-modified YAMLs, two edits each
runuser -u zolexai -- git stash
runuser -u zolexai -- git pull --ff-only
runuser -u zolexai -- git stash pop
```

Resolving `video-to-video.yaml` must keep **all** of:

* the local `runtime` flip and the deletion of `output_content_type` /
  `output_kind` (§16 — a surviving `image/png` here is one of the three
  candidate causes of "I can't download anything");
* incoming: `v2v_reference_identity: false` at the top level and the new
  `execution_by_quality.best` block. Keeping the old top-level `true` and
  dropping the overlay silently gives Fast the matting cost it is not
  supposed to pay;
* incoming: the new `reference_image` help line and the revised Fast/Best
  comment.

**`runtime_by_quality` for video-to-video must now name an engine that serves
it at both levels.** The worker's new safety net will rescue a stale mapping,
but it logs a warning per job and that is a limp, not a configuration.

```bash
docker compose -f ... build api web
docker compose -f ... up -d --no-deps --force-recreate api web
```

Verify from outside the container, because the SSR catalog is the third reader
of this YAML and has its own copy of the strings:

```bash
curl -s https://zolexai.com/api/v1/workflows/video-to-video | grep -o "person in your video is replaced"
curl -s https://zolexai.com/api/v1/workflows/music-video    | grep -o "Who is on screen"
```

Both must hit. The web page's placeholder and help text must match — if the
API has them and the page does not, `web` was not rebuilt.

## 2. Read the production evidence

Before or after the deploy, but do it — the client asked directly ("check
generations") and one reading changes the diagnosis.

```sql
select g.id, g.workflow_id, g.status, g.error_code, g.created_at,
       o.asset_id, a.status as asset_status, a.content_type, a.size_bytes
from generation_jobs g
left join generation_job_outputs o on o.job_id = g.id
left join assets a on a.id = o.asset_id
where g.workflow_id = 'video-to-video'
order by g.created_at desc limit 20;
```

| Reading | Meaning |
| --- | --- |
| `status = failed` | the H3 no-reference refusal — fixed by this build |
| `succeeded`, `asset_status <> 'ready'` | upload confirm is failing; only READY assets get a URL |
| `content_type = 'image/png'` | the deployed YAML lost its mock-output deletion |

## 3. The GPU checks that clear the build

Ordered by what the client will press first. Each one has a stated pass
condition, because "looks better" is not a result.

**V1 — Video to Video, Best, with a reference photo.** Any 10-15 s clip of one
person, plus a photo of a different person.
*Pass:* the output is recognisably the SAME footage — same camera move, same
framing, same actions at the same times — with the reference person's face.
*Fail:* a different scene. That is the old routing still live; check the
worker log for `runtime_by_quality_unsupported` and check the VPS mapping.

**V2 — Video to Video, Fast, no reference photo.** The workflow's headline
promise, and the case that failed outright last week.
*Pass:* completes, and returns the customer's footage restyled.

**V3 — the download.** Press Download on V1's result.
*Pass:* an `.mp4` lands. If it fails, the new error line now says so — read
it, then §2's table tells you which of the three causes it is.

**A1 — age.** Image to Video, 20 s, from a photo of someone visibly over 50,
prompt naming no age.
*Pass:* `grep i2v_identity_facts` in the worker log shows a caption whose
first clause is a NUMBER, and the last second of the video is the same age as
the first.
*Fail (caption empty):* the vision checkpoint is not loading as an image-text
model — the run is unaffected but the fix is inert.

**M1 — music video, thin prompt.** A ~60 s vocal track and a deliberately thin
prompt ("a cinematic music video at sunset"), i.e. the exact 27 Aug failing
case.
*Pass:* `music_video_shot_plan` in the log shows no distant/overhead framing
for the sung sections, the performer is legible throughout, and one person
wears one outfit from first section to last.
*This is the weakest-evidence change in the build* — the framing floor is
reasoned from two runs, not swept. If M1 still drifts, the next lever is
enriching a thin subject once per job rather than per section, and that is a
model call nobody has costed yet.

**M2 — music video, detailed prompt.** Re-run the 27 Aug detailed prompt.
*Pass:* no worse than `24_mv_detailed_prompt.mp4`. This is the regression
guard: the framing change must not damage the case that already worked.

## 4. What to tell the client, and what not to

Tell them V2V now returns their own video at both settings, and that Best is
where the reference photo replaces the person. Tell them the age fix is in.
Tell them a music video follows the description they give it and that the
prompt box now says what to write.

Do not tell them the music-video thin-prompt case is solved. It is improved
by construction and unmeasured in fact, and M1 above is the run that decides
whether that sentence can be said at all.
