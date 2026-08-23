#!/usr/bin/env python
"""Freeze and verify the golden benchmark pack.

    uv run python scripts/golden_pack.py --status
    uv run python scripts/golden_pack.py --verify
    uv run python scripts/golden_pack.py --freeze

`--verify` is the one that runs on GPU day, before any comparison: it checks
every acquired asset against its recorded SHA256 and exits non-zero on a
mismatch or a missing file. A pending asset is reported but does not fail the
check — not-yet-shot and quietly-different are different problems.

`--freeze` rewrites `benchmarks/frozen/cases.json` from the case definitions.
Run it deliberately, review the diff, and never to make a failing check pass:
a prompt that changed at the same `prompt_version` is the thing the pack
exists to catch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker.providers.golden import (  # noqa: E402
    ASSET_MANIFEST,
    FROZEN_CASES,
    PACK_ROOT,
    AssetStatus,
    blocking_statuses,
    compare_frozen,
    freeze_cases,
    load_assets,
    load_frozen,
    sha256_file,
    verify_assets,
)

_SYMBOL = {
    AssetStatus.OK: "ok",
    AssetStatus.PENDING_ACQUISITION: "pending",
    AssetStatus.MISSING: "MISSING",
    AssetStatus.UNHASHED: "unhashed",
    AssetStatus.MISMATCH: "MISMATCH",
}


def _status() -> int:
    assets = load_assets()
    statuses = verify_assets(assets)
    root = PACK_ROOT / "assets"
    print(f"Asset manifest : {ASSET_MANIFEST}")
    print(f"Media root     : {root}")
    print(f"{len(assets)} assets\n")
    for spec in assets:
        state = statuses[spec.id]
        cases = ",".join(spec.cases) or "-"
        print(f"  {_SYMBOL[state]:<9} {spec.id:<26} {spec.kind:<6} {cases}")
    pending = sum(1 for s in statuses.values() if s is AssetStatus.PENDING_ACQUISITION)
    blocking = blocking_statuses(statuses)
    print(f"\n{pending} pending acquisition, {len(blocking)} blocking")

    frozen = load_frozen()
    if frozen is None:
        print("\nCases: NOT FROZEN — run --freeze")
        return 0
    current = freeze_cases()
    problems = compare_frozen(current, frozen)
    print(
        f"\nCases: {frozen['case_count']} frozen, "
        f"{frozen['cell_count']} cells, {frozen['run_count']} runs with repeats"
    )
    for problem in problems:
        print(f"  DRIFT  {problem}")
    return 0


def _verify() -> int:
    """The GPU-day gate. Non-zero exit means do not start the comparison."""
    statuses = verify_assets()
    blocking = blocking_statuses(statuses)
    pending = [a for a, s in statuses.items() if s is AssetStatus.PENDING_ACQUISITION]

    frozen = load_frozen()
    problems: list[str] = []
    if frozen is None:
        problems.append("the case pack is not frozen — run --freeze")
    else:
        problems.extend(compare_frozen(freeze_cases(), frozen))

    for asset, status in blocking.items():
        print(f"STOP  {asset}: {status.value}")
    for problem in problems:
        print(f"STOP  {problem}")
    if pending:
        print(f"note  {len(pending)} asset(s) not yet acquired: {', '.join(sorted(pending))}")

    if blocking or problems:
        print("\nverification FAILED — do not run a comparison against this pack")
        return 1
    print("\nverification passed" + (" (with pending assets)" if pending else ""))
    return 0


def _freeze() -> int:
    FROZEN_CASES.parent.mkdir(parents=True, exist_ok=True)
    previous = load_frozen()
    current = freeze_cases()
    if previous is not None:
        for problem in compare_frozen(current, previous):
            print(f"  changing: {problem}")
    FROZEN_CASES.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    print(
        f"froze {current['case_count']} cases / {current['cell_count']} cells / "
        f"{current['run_count']} runs to {FROZEN_CASES}"
    )
    return 0


def _hash(paths: list[str]) -> int:
    """Print SHA256s so a newly shot asset can be pasted into the manifest."""
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            print(f"missing: {path}")
            continue
        print(f"{sha256_file(path)}  {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="report the pack")
    parser.add_argument("--verify", action="store_true", help="GPU-day gate")
    parser.add_argument("--freeze", action="store_true", help="rewrite cases.json")
    parser.add_argument("--hash", nargs="+", metavar="FILE", help="sha256 a file")
    args = parser.parse_args()

    if args.hash:
        return _hash(args.hash)
    if args.freeze:
        return _freeze()
    if args.verify:
        return _verify()
    return _status()


if __name__ == "__main__":
    raise SystemExit(main())
