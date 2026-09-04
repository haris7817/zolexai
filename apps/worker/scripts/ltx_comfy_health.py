"""GPU-day check for the LTX 2.5 ComfyUI runtime — run this before any job.

    cd apps/worker && .venv/bin/python scripts/ltx_comfy_health.py [--deep] [--json]

What it answers, in the order a deploy should fix things:

  1. Is the service reachable at LTX_COMFY_BASE_URL?
  2. Are the frozen graphs byte-identical to the delivered ZIP?
  3. Does /object_info declare every node class the three compiled prompts
     use (missing → a node pack is not installed)?
  4. Does every combo value the prompts set exist on the server — model
     files, LoRA files, VAEs, the ResolutionSelector labels (missing → a
     weight is not in models/, or a pack revision renamed an option)?
  5. (--deep) Are the weights present on disk under LTX_COMFY_MODELS_DIR,
     and does the node have the VRAM and disk the runtime needs?

Exit status 0 when healthy, 1 otherwise. Nothing here renders.

STATUS: WAITING FOR GPU VALIDATION — written before the node existed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from worker.comfy.ltx_graphs import (
    combo_options,
    model_files_referenced,
    verify_against_object_info,
)
from worker.core.config import settings
from worker.providers.ltx_comfy import GRAPH_FILES, REQUIRED_WEIGHTS, LtxComfyService


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--deep", action="store_true", help="also check weights on disk, VRAM and free disk"
    )
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args()

    service = LtxComfyService()
    report: dict[str, object] = {
        "base_url": settings.ltx_comfy_base_url,
        "workflows_dir": str(settings.ltx_comfy_workflows_dir),
        "graphs": dict(GRAPH_FILES),
        "graph_drift": service.graphs_match_the_pack(),
        "required_weights": REQUIRED_WEIGHTS,
        "status": "WAITING FOR GPU VALIDATION",
    }

    ok, detail = await service.health(deep=args.deep)
    report["healthy"] = ok
    report["detail"] = detail

    info = await service.object_info()
    if info is not None:
        report["aspect_options"] = combo_options(info, "ResolutionSelector", "aspect_ratio")
        per_graph: dict[str, list[str]] = {}
        for name, api in service._probe_prompts().items():  # noqa: SLF001 - diagnostic script
            per_graph[name] = verify_against_object_info(api, info)
        report["problems_by_graph"] = per_graph
        report["models_referenced"] = {
            name: model_files_referenced(api)
            for name, api in service._probe_prompts().items()  # noqa: SLF001
        }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"LTX ComfyUI at {settings.ltx_comfy_base_url}: {'HEALTHY' if ok else 'NOT READY'}")
        print(f"  {detail}")
        if report["graph_drift"]:
            print("  graph drift:", "; ".join(report["graph_drift"]))
        for name, problems in (report.get("problems_by_graph") or {}).items():
            print(f"  {name}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
            for problem in problems[:25]:
                print(f"    - {problem}")
        print("  weights the pack loads:")
        for relative, source in REQUIRED_WEIGHTS.items():
            print(f"    - {relative}  ({source})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
