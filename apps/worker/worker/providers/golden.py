"""The frozen benchmark pack: same inputs, same prompts, months apart.

A comparison is only worth the reproducibility of its inputs. Two runs of
"case A9" that used a slightly different prompt, or a source image someone
re-exported at a different quality, are not two results — they are one result
and one anecdote, and nothing in the output would say which.

So two things are frozen and checked.

**Media.** Every asset is declared with its SHA256, duration, geometry and
provenance. The pack verifies before a comparison runs, and a hash that
differs from the manifest is a STOP, not a warning: we cannot compare LTX
from image A against H3 from image A-as-edited-last-week without knowing.
Media itself stays out of git (binaries do not belong in this repository);
the manifest is the committed artifact and the files live beside it.

**Prompts.** The cases are code, so git already versions them — but git tracks
that a line changed, not that a *benchmark* changed. `cases.json` freezes each
case's prompt hash against its `prompt_version`, and the test refuses a
changed prompt at an unchanged version. Editing a prompt mid-benchmark is
legitimate; doing it silently is what this prevents.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from worker.providers.benchmark import CASES, BenchmarkCase

#: The pack lives at the repo root, beside the code it describes.
PACK_ROOT = Path(__file__).resolve().parents[4] / "benchmarks"
ASSET_MANIFEST = PACK_ROOT / "assets.manifest.json"
FROZEN_CASES = PACK_ROOT / "frozen" / "cases.json"


class AssetStatus(StrEnum):
    OK = "ok"
    """Present, and its hash matches the manifest."""

    PENDING_ACQUISITION = "pending_acquisition"
    """Declared but not yet shot or collected. Expected before GPU day."""

    MISSING = "missing"
    """The manifest says it exists and has a hash; the file is not there."""

    UNHASHED = "unhashed"
    """Present on disk but the manifest carries no hash yet — run --freeze."""

    MISMATCH = "mismatch"
    """Present and DIFFERENT. Stop the comparison."""


@dataclass(frozen=True)
class AssetSpec:
    id: str
    filename: str
    kind: str
    """image | video | audio."""

    purpose: str
    cases: list[str] = field(default_factory=list)
    """Benchmark cases that cannot run without it."""

    sha256: str | None = None
    duration_seconds: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    framing: str = ""
    """close-up | waist-up | full-body | landscape | n/a — the variable that
    several comparisons are organised around."""

    provenance: str = ""
    """Where it came from and under what right to use it. Required before an
    asset may be marked acquired: an unlicensed song in a benchmark is a legal
    problem that outlives the benchmark."""

    acquisition: str = "pending"
    """pending | acquired. Kept separate from the hash so "not shot yet" never
    reads as "someone deleted it"."""

    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_assets(path: Path | None = None) -> list[AssetSpec]:
    source = path or ASSET_MANIFEST
    if not source.exists():
        return []
    data = json.loads(source.read_text(encoding="utf-8"))
    return [AssetSpec(**entry) for entry in data["assets"]]


def asset_status(spec: AssetSpec, root: Path) -> AssetStatus:
    """Where this asset stands, without judging whether that is acceptable."""
    path = root / spec.filename
    if spec.acquisition != "acquired":
        return AssetStatus.PENDING_ACQUISITION
    if not path.exists():
        return AssetStatus.MISSING
    if not spec.sha256:
        return AssetStatus.UNHASHED
    return AssetStatus.OK if sha256_file(path) == spec.sha256 else AssetStatus.MISMATCH


def verify_assets(
    specs: list[AssetSpec] | None = None, root: Path | None = None
) -> dict[str, AssetStatus]:
    specs = specs if specs is not None else load_assets()
    root = root or (PACK_ROOT / "assets")
    return {spec.id: asset_status(spec, root) for spec in specs}


def blocking_statuses(statuses: dict[str, AssetStatus]) -> dict[str, AssetStatus]:
    """The ones that must stop a comparison, as opposed to merely delay it.

    A pending asset means the benchmark is not ready to run yet. A MISMATCH
    or a MISSING file means it is ready to run and would produce a result
    nobody could trust — a different failure entirely.
    """
    return {
        asset: status
        for asset, status in statuses.items()
        if status in (AssetStatus.MISMATCH, AssetStatus.MISSING)
    }


# ── Prompt freezing ──────────────────────────────────────────────────────


def freeze_case(case: BenchmarkCase) -> dict[str, Any]:
    """One case, reduced to what must not drift unnoticed."""
    return {
        "case_id": case.id,
        "group": case.group,
        "title": case.title,
        "workflow_id": case.workflow_id,
        "prompt_version": case.prompt_version,
        "prompt_sha256": sha256_text(case.prompt),
        "parameters": dict(case.parameters),
        "execution": dict(case.execution),
        "inputs": [list(pair) for pair in case.inputs],
        "strategies": [s.value for s in case.strategies],
        "repeats": case.repeats,
        "measures": list(case.measures),
    }


def freeze_cases(cases: tuple[BenchmarkCase, ...] = CASES) -> dict[str, Any]:
    return {
        "pack_version": 1,
        "case_count": len(cases),
        "cell_count": sum(len(c.strategies) for c in cases),
        "run_count": sum(len(c.strategies) * c.repeats for c in cases),
        "cases": [freeze_case(case) for case in sorted(cases, key=lambda c: c.id)],
    }


def load_frozen(path: Path | None = None) -> dict[str, Any] | None:
    source = path or FROZEN_CASES
    if not source.exists():
        return None
    return json.loads(source.read_text(encoding="utf-8"))


def compare_frozen(
    current: dict[str, Any], frozen: dict[str, Any]
) -> list[str]:
    """Every drift between the code and the frozen pack, in plain words."""
    problems: list[str] = []
    current_by_id = {c["case_id"]: c for c in current["cases"]}
    frozen_by_id = {c["case_id"]: c for c in frozen["cases"]}

    for case_id in sorted(set(frozen_by_id) - set(current_by_id)):
        problems.append(f"{case_id}: in the frozen pack but no longer defined in code")
    for case_id in sorted(set(current_by_id) - set(frozen_by_id)):
        problems.append(f"{case_id}: defined in code but not in the frozen pack")

    for case_id in sorted(set(current_by_id) & set(frozen_by_id)):
        now, before = current_by_id[case_id], frozen_by_id[case_id]
        if now["prompt_sha256"] != before["prompt_sha256"]:
            if now["prompt_version"] == before["prompt_version"]:
                problems.append(
                    f"{case_id}: the prompt changed but prompt_version is still "
                    f"{now['prompt_version']} — bump it, then re-freeze"
                )
            # A bumped version with a changed prompt is legitimate: it is a
            # new prompt, deliberately, and the pack should be re-frozen.
        for key in ("workflow_id", "parameters", "execution", "inputs", "strategies"):
            if now[key] != before[key]:
                problems.append(
                    f"{case_id}: {key} changed since the pack was frozen "
                    f"({before[key]!r} -> {now[key]!r})"
                )
    return problems
