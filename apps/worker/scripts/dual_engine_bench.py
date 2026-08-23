#!/usr/bin/env python
"""Drive the LTX/H3 comparison — dry runs today, renders when a GPU exists.

    uv run python scripts/dual_engine_bench.py --list
    uv run python scripts/dual_engine_bench.py --dry-run --group A
    uv run python scripts/dual_engine_bench.py --dry-run --out plans.json

The dry run compiles every case through both providers and writes the
manifests. It touches no model, no network and no weights, which is what makes
it useful now: the section counts, seam counts, reference plans and audio
windows the GPU session will be comparing are all decided here, and they can
be reviewed and argued with before a single second of GPU time is bought.

`--run` is deliberately not implemented. H3 is not installed anywhere and LTX
renders belong to the worker, not to a benchmark script; when the hardware
lands, the runner that fills in `RunRecord` goes here, next to the schema it
fills.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Runnable from the repo without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker.adapters.base import AdapterInput, AdapterJob  # noqa: E402
from worker.providers import ProviderRefusal, compare  # noqa: E402
from worker.providers.benchmark import (  # noqa: E402
    CASES,
    GROUPS,
    SCORE_WEIGHTS,
    SEPARATE_SCORES,
    BenchmarkCase,
    cases_for_group,
    result_skeleton,
)

_SUFFIX = {"image": ".png", "video": ".mp4", "audio": ".mp3"}
_CONTENT_TYPE = {"image": "image/png", "video": "video/mp4", "audio": "audio/mpeg"}


def _job(case: BenchmarkCase, workspace: Path) -> AdapterJob:
    inputs = [
        AdapterInput(
            role=role,
            kind=kind,
            content_type=_CONTENT_TYPE[kind],
            download_url="https://storage.test/golden",
            path=workspace / "inputs" / f"{role}{_SUFFIX[kind]}",
        )
        for role, kind in case.inputs
    ]
    return AdapterJob(
        job_id=f"bench-{case.id.lower()}",
        workflow_id=case.workflow_id,
        workflow_version="1",
        prompt=case.prompt,
        parameters=dict(case.parameters),
        inputs=inputs,
        execution={
            "runtime": "ltx",
            # The cases are compiled the way the workflows SHIP, so a dry run
            # describes the product rather than a laboratory configuration:
            # structuring on everywhere but video-to-video, the 30s section
            # ceiling on t2v/i2v, and the transform engine on v2v — all of
            # which are the committed YAML's own values.
            "prompt_structuring": case.workflow_id != "video-to-video",
            **({"max_segment_seconds": 30}
               if case.workflow_id in ("text-to-video", "image-to-video") else {}),
            **({"v2v_engine": "transform"}
               if case.workflow_id == "video-to-video" else {}),
            **case.execution,
        },
        workspace=workspace,
    )


def _summarise(case: BenchmarkCase, plans: dict) -> str:
    parts = [f"{case.id:<3} {case.group}  {case.title}"]
    for engine in ("ltx", "h3"):
        plan = plans.get(engine, {})
        if "refused" in plan:
            parts.append(f"      {engine:<4} refused — {plan['refused']}")
            continue
        sections = plan.get("sections", [])
        seams = max(0, len(sections) - 1)
        lengths = ", ".join(f"{s['duration_seconds']:g}s" for s in sections[:4])
        if len(sections) > 4:
            lengths += ", …"
        audio = sections[0].get("audio") if sections else None
        audio_note = f"  audio={audio['mode']}" if audio else ""
        parts.append(
            f"      {engine:<4} {plan['pipeline']:<32} "
            f"{plan['width']}x{plan['height']}  "
            f"{len(sections)} section(s), {seams} seam(s)  [{lengths}]{audio_note}"
        )
    if not case.both_engines:
        parts.append("      (single-engine case — recorded as a capability, not a contest)")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument("--dry-run", action="store_true", help="compile both engines")
    parser.add_argument("--group", help="restrict to one group letter (A-J)")
    parser.add_argument("--case", help="restrict to one case id (e.g. A9)")
    parser.add_argument("--out", type=Path, help="write the manifests as JSON")
    parser.add_argument(
        "--skeleton", type=Path,
        help="write an empty result document for the GPU session to fill",
    )
    parser.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run:
        parser.error(
            "--run is not implemented: H3 is not installed on any node and LTX "
            "renders belong to the worker. Wire the runner when the GPU exists."
        )

    cases = CASES
    if args.group:
        cases = cases_for_group(args.group)
    if args.case:
        cases = tuple(c for c in cases if c.id.upper() == args.case.upper())
    if not cases:
        parser.error("no cases matched")

    if args.skeleton:
        args.skeleton.write_text(
            json.dumps(result_skeleton(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote empty result document to {args.skeleton}")
        return 0

    if args.list or not args.dry_run:
        print(f"{len(CASES)} cases across {len(GROUPS)} groups\n")
        for letter, title in GROUPS.items():
            group = cases_for_group(letter)
            if not group:
                continue
            print(f"  {letter} · {title}")
            for case in group:
                engines = "both" if case.both_engines else "h3 only"
                print(f"      {case.id:<3} {case.title:<38} "
                      f"{case.repeats} run(s), {engines}")
        print("\nScoring:", ", ".join(f"{k} {v}%" for k, v in SCORE_WEIGHTS.items()))
        print("Scored separately:", ", ".join(SEPARATE_SCORES))
        return 0

    workspace = Path("/workspace/bench")
    out: dict[str, dict] = {}
    for case in cases:
        job = _job(case, workspace)
        try:
            plans = compare(job)
        except ProviderRefusal as refusal:  # pragma: no cover - defensive
            plans = {"error": {"refused": refusal.reason}}
        out[case.id] = {"case": case.id, "title": case.title, "plans": plans}
        print(_summarise(case, plans))
        print()

    if args.out:
        args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(out)} case manifests to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
