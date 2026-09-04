# LTX 2.5 client pack — frozen graphs

The three ComfyUI graphs delivered in `LTX2.5 Pipeline-2.zip` (sha256
`fb1e7772921dcf093165aa1e57614c42bb1ffb9af03591dd1e095535a7a970a1`), byte-identical.
They are the production workflows for Text to Video, First/Last Frame Video and
Character Replacement, and they are used as shipped: the worker edits only prompt,
seed, duration, input media and output location (`worker/comfy/ltx_graphs.py`).

| File | sha256 | Serves |
|---|---|---|
| `ltx25_text_to_video.json` | `2dcd9661118c947cc1cae0e5aa59656b519387a8f8e86f8e4c06545bd07b914c` | text-to-video |
| `ltx25_first_last_frame.json` | `1926bd6dd4f897b45eb8f9e20072066f90fd01678107287f6b6459921e4da967` | image-to-video, extend-video, chained passes |
| `ltx25_character_replacement.json` | `2ea7547268f8742ba657fcf390800501e39ba7aff5d1736fa6b41ed988b1adc9` | character-replacement |

`samples/` holds the two small inputs from the ZIP (the first-frame still and the
character-replacement source clip). The three sample OUTPUTS (37 MB, 44 MB, 5.8 MB)
are not committed; their hashes and probes are in
`docs/internal/ltx-client-workflow-audit.md` §2.1 for the GPU-day comparison.

Do not edit these files. A changed graph is a new client deliverable, not a fix.
