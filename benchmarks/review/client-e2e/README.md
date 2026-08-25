# Client E2E review pack — 25 August 2026

Every file below was generated end-to-end through the real ZolexAI worker
adapters on the RTX PRO 6000 client-test node (commit `d81326c`) — not by
benchmark scripts. Inputs included so each result can be judged against what
went in.

| File | Workflow → engine | Prompt (abridged) | Duration | Resolution | Wall |
|---|---|---|---|---|---|
| `01_t2v_ltx.mp4` | Text-to-video → LTX 2.5 | koi pond at dawn | 5.013 s | 1024x576 | 40.7 s¹ |
| `02_i2v_h3.mp4` | Image animation → H3 INT8 (input: `input_reference_person.png`) | he begins to speak calmly to camera | 5.167 s | 1280x736 | 331.0 s |
| `03_reference_v2v_h3.mp4` | Reference video → H3 INT8, Quality tier (inputs: person + `input_source_video.mp4`) | sings at the microphone in the rehearsal room | 5.167 s | 960x544 | 172.0 s |
| `04_music_video_ltx_a2v.mp4` | Music video → LTX native A2V, guard on (input: `input_song_window.mp3`) | singer at a microphone, hard side light | 10.042 s | 1024x576 | 164.4 s |
| `05_extend_ltx.mp4` | Extend → LTX (input: source clip) | the dawn light strengthens | 10.004 s (from 5 s) | 1024x576 | 24.6 s |
| `06_h3_long_60s.mp4` | 60 s long-form → H3 INT8 + generated prompt discipline, Draft tier | grey-bearded man in navy coat sings at the microphone | 59.709 s | 544x320 | 716.0 s |

¹ includes the LTX model reload after ComfyUI's VRAM was freed; the render
itself is the known ~28 s.

Notes for review:
- `06`: identity, coat and microphone hold across all five segments; framing
  varies more than the hand-tuned benchmark run — the next refinement is
  feeding Director-structured scene fields into the prompt compiler.
- `04`: the model heard the song (audio-conditioned path); the guard test that
  the unconditioned route REFUSES also passed in the same session.
- No rejected experiments are included; Turbo and BF16 variants live only in
  the research folders, clearly labelled.
