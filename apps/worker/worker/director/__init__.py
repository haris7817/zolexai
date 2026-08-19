"""Director mode: an idea in, a fully planned dialogue scene out.

The optional second prompt mode of Text to Video and Image to Video. The user
writes a one-line idea ("a detective confronts a corrupt police chief"); a
planning model turns it into a structured `DirectorPlan` — characters,
dialogue turns, delivery, actions, camera, timing — and
`compile_section_prompts` renders that plan into the caption style the LTX
model was actually trained on. Everything below the prompt (chaining,
rendering, stitching, validation) is the existing pipeline, untouched.

On Image to Video the plan is SOURCE-ANCHORED: the uploaded image defines WHO
and WHAT (and stays the identity anchor in every chained section, via the
adapter's existing conditioning), while the plan defines only WHAT HAPPENS —
the planner is forbidden to invent visual detail the idea does not state, and
the compiled captions tie identity to the conditioned first frame instead of
to a described look. See `worker/director/vision.py` for the optional
image-facts step that can ground the plan in what the photograph measurably
shows.

This is deliberately NOT part of prompt structuring or the section planner:
those are non-LLM by contract (they must never paraphrase a user's words),
while Director mode's whole job is to write words the user never typed. The
two postures cannot share a module without one of them lying.
"""

from worker.director.cerebras import (
    CerebrasDirectorProvider,
    DirectorProviderUnavailable,
)
from worker.director.compiler import compile_section_prompts
from worker.director.plan import (
    DirectorPlan,
    DirectorPlanError,
    pacing_problems,
    parse_plan,
    target_spoken_lines,
)
from worker.director.provider import (
    DirectorFailure,
    DirectorRequest,
    GemmaDirectorProvider,
    create_director_plan,
    default_providers,
    source_anchored,
    wants_director,
)
from worker.director.vision import source_image_facts

__all__ = [
    "CerebrasDirectorProvider",
    "DirectorFailure",
    "DirectorPlan",
    "DirectorPlanError",
    "DirectorProviderUnavailable",
    "DirectorRequest",
    "GemmaDirectorProvider",
    "compile_section_prompts",
    "create_director_plan",
    "default_providers",
    "pacing_problems",
    "parse_plan",
    "source_anchored",
    "source_image_facts",
    "target_spoken_lines",
    "wants_director",
]
