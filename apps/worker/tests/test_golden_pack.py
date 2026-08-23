"""The frozen pack: the benchmark's inputs cannot drift unnoticed.

Two failures this exists to make impossible. A prompt quietly reworded between
sessions, so that two rows labelled "A9" were never the same request. And a
source asset re-exported, re-encoded or replaced, so that LTX rendered from
one image and H3 from another while the table showed a fair comparison.

Both are silent by nature — that is exactly why they need a test rather than a
convention.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from worker.providers.benchmark import CASES, RunRecord, result_skeleton
from worker.providers.golden import (
    ASSET_MANIFEST,
    PACK_ROOT,
    AssetSpec,
    AssetStatus,
    asset_status,
    blocking_statuses,
    compare_frozen,
    freeze_cases,
    load_assets,
    load_frozen,
    sha256_file,
    sha256_text,
    verify_assets,
)

# ── The frozen cases ─────────────────────────────────────────────────────


def test_the_pack_is_frozen_and_matches_the_cases_in_code() -> None:
    """If this fails, either re-freeze deliberately (and review the diff) or
    the change to the cases was not meant."""
    frozen = load_frozen()
    assert frozen is not None, "the pack is not frozen — run scripts/golden_pack.py --freeze"

    problems = compare_frozen(freeze_cases(), frozen)
    assert not problems, "\n".join(problems)


def test_a_silently_edited_prompt_is_caught() -> None:
    """The core guarantee. A changed prompt at an unchanged version is not a
    smaller version of the same benchmark — it is a different one."""
    frozen = freeze_cases()
    edited = tuple(
        replace(case, prompt=case.prompt + " Now at golden hour.")
        if case.id == "A1"
        else case
        for case in CASES
    )
    problems = compare_frozen(freeze_cases(edited), frozen)
    assert any("prompt_version is still" in problem for problem in problems)


def test_a_prompt_change_with_a_version_bump_is_allowed() -> None:
    """Editing a benchmark prompt is legitimate; doing it silently is not."""
    frozen = freeze_cases()
    revised = tuple(
        replace(
            case,
            prompt=case.prompt + " Now at golden hour.",
            prompt_version=case.prompt_version + 1,
        )
        if case.id == "A1"
        else case
        for case in CASES
    )
    problems = compare_frozen(freeze_cases(revised), frozen)
    assert not [p for p in problems if p.startswith("A1:")]


def test_adding_or_removing_a_case_is_reported() -> None:
    frozen = freeze_cases()
    fewer = tuple(case for case in CASES if case.id != "A1")
    problems = compare_frozen(freeze_cases(fewer), frozen)
    assert any("no longer defined in code" in problem for problem in problems)


def test_changing_a_cases_parameters_is_reported() -> None:
    frozen = freeze_cases()
    altered = tuple(
        replace(case, parameters={**case.parameters, "duration": "10s"})
        if case.id == "A1"
        else case
        for case in CASES
    )
    problems = compare_frozen(freeze_cases(altered), frozen)
    assert any("parameters changed" in problem for problem in problems)


def test_the_frozen_pack_records_every_cell_including_hybrids() -> None:
    frozen = load_frozen()
    assert frozen["case_count"] == len(CASES)
    assert frozen["cell_count"] == sum(len(c.strategies) for c in CASES)
    assert frozen["run_count"] == sum(len(c.strategies) * c.repeats for c in CASES)

    by_id = {case["case_id"]: case for case in frozen["cases"]}
    assert "ltx_to_h3_reference" in by_id["D3"]["strategies"]
    assert "ltx_to_h3_reference" not in by_id["A9"]["strategies"]
    assert by_id["J1"]["strategies"] == ["h3_only"]


# ── Assets ───────────────────────────────────────────────────────────────


def test_the_asset_manifest_is_loadable_and_complete() -> None:
    assets = load_assets()
    assert assets, "no assets declared"
    for spec in assets:
        assert spec.id and spec.filename and spec.kind
        assert spec.purpose, f"{spec.id} has no stated purpose"
        assert spec.provenance, f"{spec.id} has no provenance — required before use"
        assert spec.acquisition in ("pending", "acquired")


def test_every_declared_asset_is_needed_by_a_real_case() -> None:
    """An asset nobody uses is an asset nobody will keep correct."""
    known = {case.id for case in CASES}
    for spec in load_assets():
        for case_id in spec.cases:
            assert case_id in known, f"{spec.id} names unknown case {case_id}"


def test_every_case_that_needs_media_has_it_declared() -> None:
    """The other direction: a case whose asset was never declared would fail
    on GPU day, which is the most expensive place to discover it."""
    declared_for: dict[str, list[str]] = {}
    for spec in load_assets():
        for case_id in spec.cases:
            declared_for.setdefault(case_id, []).append(spec.id)

    for case in CASES:
        if not case.inputs:
            continue
        assert case.id in declared_for, (
            f"case {case.id} needs {[k for _, k in case.inputs]} but no asset "
            "declares it"
        )


def test_a_pending_asset_does_not_block_but_a_changed_one_does(
    tmp_path: Path,
) -> None:
    """Not-yet-shot and quietly-different are different problems, and only one
    of them invalidates a comparison."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"original bytes")
    digest = sha256_file(media)

    pending = AssetSpec(id="p", filename="clip.mp4", kind="video", purpose="x")
    assert asset_status(pending, tmp_path) is AssetStatus.PENDING_ACQUISITION

    good = replace(pending, acquisition="acquired", sha256=digest)
    assert asset_status(good, tmp_path) is AssetStatus.OK

    media.write_bytes(b"re-exported bytes")
    assert asset_status(good, tmp_path) is AssetStatus.MISMATCH

    media.unlink()
    assert asset_status(good, tmp_path) is AssetStatus.MISSING

    statuses = {
        "pending": AssetStatus.PENDING_ACQUISITION,
        "changed": AssetStatus.MISMATCH,
        "gone": AssetStatus.MISSING,
    }
    assert set(blocking_statuses(statuses)) == {"changed", "gone"}


def test_an_acquired_asset_without_a_hash_is_flagged(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"bytes")
    spec = AssetSpec(
        id="p", filename="clip.mp4", kind="video", purpose="x", acquisition="acquired"
    )
    assert asset_status(spec, tmp_path) is AssetStatus.UNHASHED


def test_the_shipped_manifest_verifies_as_pending_not_broken() -> None:
    """Today every asset is declared and none is shot. That is a readiness
    state, not a failure."""
    statuses = verify_assets()
    assert statuses, "no assets to verify"
    assert not blocking_statuses(statuses)
    assert all(s is AssetStatus.PENDING_ACQUISITION for s in statuses.values())


def test_sha256_helpers_agree_with_the_manifest_format(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello", encoding="utf-8")
    assert sha256_file(path) == sha256_text("hello")
    assert len(sha256_text("hello")) == 64


def test_the_media_directory_is_not_committed() -> None:
    """Binaries do not belong in this repository, and a benchmark asset in git
    history is a licensing problem that outlives the benchmark."""
    ignore = PACK_ROOT / "assets" / ".gitignore"
    assert ignore.exists()
    assert "*" in ignore.read_text(encoding="utf-8").splitlines()


def test_the_manifest_json_stays_human_editable() -> None:
    data = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    assert data["rules"], "the rules are the point of the file"
    assert data["assets"]


# ── The result schema ────────────────────────────────────────────────────


def test_the_result_skeleton_ships_empty() -> None:
    """A skeleton carrying plausible numbers is the easiest way for a
    fabricated benchmark to reach a decision."""
    skeleton = result_skeleton()
    assert skeleton["runs"] == []
    assert skeleton["gpu"] is None
    assert all(
        decision["provider"] is None for decision in skeleton["decisions"].values()
    )


def test_a_run_record_carries_enough_to_reproduce_it() -> None:
    record = RunRecord(
        case_id="D3",
        provider="h3",
        strategy="ltx_to_h3_reference",
        prompt_version=1,
        asset_hashes={"person-fullbody": "a" * 64},
        model_revision="MiniMax-H3@abc123",
        runtime_revision="sglang 0.x",
        gpu="RTX PRO 6000",
    )
    data = record.to_dict()
    for key in (
        "strategy", "prompt_version", "asset_hashes", "model_revision",
        "runtime_revision", "gpu", "seed", "ltx_generation_seconds",
        "h3_generation_seconds", "model_switch_seconds", "handoff_seconds",
        "generated_references", "duration_actual",
    ):
        assert key in data, f"a run record cannot be reproduced without {key}"


def test_an_unscored_run_has_no_overall(caplog: pytest.LogCaptureFixture) -> None:
    """A partial card is an incomplete result, not a lower score."""
    record = RunRecord(case_id="A1", provider="ltx")
    assert record.overall() is None
    record.scores = {"visual_quality": 8}
    assert record.overall() is None
