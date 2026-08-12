"""What a node declares it can run.

Routing has always been per workflow — the YAML's `execution.runtime` says which
adapter should execute a job. What was missing is the other half: nothing checked
that the worker claiming the job agreed.

That was harmless while every definition said `mock`. The moment one says
something else, a mock node claims that job, finds no adapter, and fails it with
`retriable=False` — a permanently dead job, and the customer is told the tool is
unavailable. These tests cover the worker's side of the fix; the API's
intersection is covered in `apps/api/tests/test_worker_protocol.py`.
"""

from __future__ import annotations

import pytest

from worker.core.config import WorkerSettings


def settings_with(**overrides) -> WorkerSettings:
    # `_env_file=None` so a developer's own .env cannot change the answer.
    return WorkerSettings(_env_file=None, **overrides)


def test_a_node_serves_its_primary_runtime_by_default() -> None:
    assert settings_with(runtime="mock").runtime_list == ["mock"]


def test_extra_runtimes_are_additive_not_replacements() -> None:
    """A node that lists others still runs its own — forgetting to repeat it
    would silently starve the node of the work it was deployed for."""
    settings = settings_with(runtime="ltx", runtimes="harness,mock")
    assert settings.runtime_list == ["ltx", "harness", "mock"]


def test_the_primary_runtime_is_not_duplicated() -> None:
    settings = settings_with(runtime="mock", runtimes="mock,harness")
    assert settings.runtime_list == ["mock", "harness"]


@pytest.mark.parametrize("value", ["", "   ", ",,", " , "])
def test_a_blank_declaration_degrades_to_the_primary_runtime(value: str) -> None:
    """An unset or malformed env var must not produce an empty capability list —
    the API would read that as "claims nothing" and the node would idle."""
    assert settings_with(runtime="mock", runtimes=value).runtime_list == ["mock"]


def test_whitespace_around_names_is_tolerated() -> None:
    settings = settings_with(runtime="mock", runtimes=" harness , ltx ")
    assert settings.runtime_list == ["mock", "harness", "ltx"]


def test_a_gpu_node_can_be_configured_to_serve_only_its_own_runtime() -> None:
    """The deployment that matters in M2: a GPU node must not pick up mock work
    and a mock node must not pick up GPU work."""
    gpu = settings_with(runtime="ltx")
    cpu = settings_with(runtime="mock")

    assert "mock" not in gpu.runtime_list
    assert "ltx" not in cpu.runtime_list
