"""Director mode: an idea in, a fully planned dialogue scene out.

Text to Video's optional second prompt mode. The user writes a one-line idea
("a detective confronts a corrupt police chief"); a local instruct model turns
it into a structured `DirectorPlan` — characters, dialogue turns, delivery,
actions, camera, timing — and `compile_section_prompts` renders that plan into
the caption style the LTX model was actually trained on. Everything below the
prompt (chaining, rendering, stitching, validation) is the existing pipeline,
untouched.

This is deliberately NOT part of prompt structuring or the section planner:
those are non-LLM by contract (they must never paraphrase a user's words),
while Director mode's whole job is to write words the user never typed. The
two postures cannot share a module without one of them lying.
"""

from worker.director.compiler import compile_section_prompts
from worker.director.plan import DirectorPlan, DirectorPlanError, parse_plan
from worker.director.provider import (
    DirectorFailure,
    DirectorRequest,
    GemmaDirectorProvider,
    create_director_plan,
    wants_director,
)

__all__ = [
    "DirectorFailure",
    "DirectorPlan",
    "DirectorPlanError",
    "DirectorRequest",
    "GemmaDirectorProvider",
    "compile_section_prompts",
    "create_director_plan",
    "parse_plan",
    "wants_director",
]
