# Generation limits — what they are, and why each one exists

_Last reviewed: 17 August 2026 — full pass over every restriction in the product
(client ask #4, "review the generation restrictions and make the system more
flexible where possible")._

Every limit in ZolexAI is one of three kinds, and knowing the kind tells you
what it would take to move it:

| Kind | What it means | What moving it takes |
|---|---|---|
| **Product choice** | A menu we decided; the engine can already do more. | Editing one definition file. No engineering. |
| **Capacity protection** | Keeps one job from monopolising the render queue and blocking every other customer. | A deliberate capacity decision — possible, but it trades against everyone else's wait time. |
| **Measured stability** | Found by running the engine to its edge. Beyond it, jobs fail after burning their compute. | A new measurement pass first; the limit moves only when the measurement does. |

## Changed in this pass (17 August 2026)

1. **Video extension is now effectively unlimited.** A single extension step
   goes up to **5 minutes** (was 60 seconds), every extended result can itself
   be extended again, and the source a step continues from may now be up to
   **30 minutes** long (was 5½ minutes — this was the hidden cap that made
   "extend it again" stop working). Extension only ever generates the new
   part; your existing footage is kept exactly as it is, which is why it can
   be so much looser than the tools below.
2. **Music Video accepts audio uploads up to 200 MB** (was 64 MB). The old cap
   was sized for MP3s and silently refused honest lossless files — a
   five-minute stereo WAV is 50–85 MB. The 5-minute length rule was always the
   real bound; the byte cap now just mirrors the platform ceiling.

## The full table

### Prompts

| Limit | Value | Kind | Notes |
|---|---|---|---|
| Prompt length | 20,000 characters, every tool | Product choice | ~3,000 words; effectively unlimited. Exists only because an unbounded request is an abuse surface. Nobody has ever reached it. |
| Lyrics (Music) | 10,000 characters, optional | Product choice | Your own lyrics are sung exactly as written, in any of 50+ languages. The language selector applies when we write the lyrics for you (English today; more languages planned). |

### Durations

| Tool | Offered | Kind | Notes |
|---|---|---|---|
| Text to Video | 5s – 60s | Product choice | Longer runs are what Extend Video is for — generate, then extend without re-uploading. Raising the menu is a definition edit; the backend already produces long output in chained sections. |
| Image to Video | 5s – 60s | Product choice | Same as above. |
| Extend Video | 5s – 5m per step, **unlimited steps** | Product choice | New this pass. Each step continues from the final frame; chain steps for any total length. |
| Music | 1 – 5 minutes | Product choice | Proven across the full range in production. |
| Music Video | Matches your track | — | Not a limit — the output is generated to your whole song by design. |
| Video to Video | Matches your video | — | Same: the result is your footage's own length. |

### Source length (uploads that drive generation)

| Tool | Limit | Kind | Notes |
|---|---|---|---|
| Music Video, Video to Video | 5 minutes of source | **Capacity protection** | Every second of source is a second the engine must render. A one-hour upload is a job that cannot finish inside its slot — it would hold the queue for hours and then fail. The refusal is immediate and names both lengths, before any compute is spent. Moving this is a capacity decision, not a flag. |
| Extend Video | 30 minutes of source | Capacity protection | New this pass, and deliberately far looser: the source is kept as-is and continued, never re-rendered, so its length costs encoding time only. |

### Picture shape

| Limit | Value | Kind | Notes |
|---|---|---|---|
| Aspect ratios | Text to Video: 16:9, 9:16, 1:1, 4:5 · Image to Video and Music Video: 16:9, 9:16, 1:1 · others: 16:9, 9:16 | **Measured stability** | Each offered shape is one verified for long renders. The engine fails on *particular shapes*, not simply above a size, so a new ratio needs a measurement pass before it can be offered — it is on the roadmap, not a toggle. Extend and Video to Video follow the uploaded file's own shape regardless. |

### Uploads

| Limit | Value | Kind | Notes |
|---|---|---|---|
| Video files | 512 MB per file (platform ceiling 1 GB) | Product choice | Comfortable for 30 minutes of camera footage at normal bitrates. |
| Image files | 25 MB per file (platform ceiling 50 MB) | Product choice | Covers any photograph, including full-resolution phone panoramas. |
| Audio files | 200 MB per file (= platform ceiling) | Product choice | Raised this pass; fits any 5-minute track, lossless included. |
| File types | Video: MP4, MOV, WebM · Image: JPEG, PNG, WebP · Audio: MP3, WAV, M4A, OGG | Product choice | A deliberate allowlist: an unknown format is refused up front rather than failing mid-generation. Adding a format is a small, safe change. |

## For engineers

Every product-choice limit above lives in `workflow-definitions/*.yaml` — the
single source of truth the API validates at startup and every surface renders
from. The capacity ceilings are worker environment settings
(`LTX_MAX_SOURCE_SECONDS`, `LTX_MAX_EXTEND_SOURCE_SECONDS`), changeable per
deployment without a release. Measured-stability values are pinned by
regression tests and move only with a new measurement.
