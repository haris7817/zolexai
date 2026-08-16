"""Long-form orchestration: one architecture for every duration the product sells.

`worker.media` owns the tools (probe, plan, cut, join, mux, verify). This
package owns the *sequence* those tools are used in, and it is deliberately
separate from both the adapters above it and the media helpers below it:

  * `chain`    — a target duration becomes N safe passes, each conditioned on
                 the last, with cancellation honoured between them.
  * `progress` — one monotonic, customer-safe progress vocabulary, so fifteen
                 passes read as one job rather than fifteen restarts.
  * `timing`   — where a music video is allowed to cut, taken from the track
                 rather than from the pass ceiling.

Nothing here imports a provider. Video-to-video, music video, extension,
text-to-video and image-to-video all run the same code; they differ only in
what conditions the first pass and what happens to the parts afterwards.
"""

from worker.longform.chain import ChainStep, RenderStep, render_chain
from worker.longform.enhance import structure_prompt
from worker.longform.progress import (
    GENERATE_FROM,
    GENERATE_TO,
    StageReporter,
    band_for,
)
from worker.longform.prompts import plan_section_prompts
from worker.longform.timing import plan_musical_boundaries

__all__ = [
    "GENERATE_FROM",
    "GENERATE_TO",
    "ChainStep",
    "RenderStep",
    "StageReporter",
    "band_for",
    "plan_musical_boundaries",
    "plan_section_prompts",
    "render_chain",
    "structure_prompt",
]
