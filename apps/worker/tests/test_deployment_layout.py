"""The worker must start in the container layout too.

Same defect as the API had: `parents[4]` is correct for
`apps/worker/worker/core/config.py` in the repo, and out of range for
`/app/worker/core/config.py` in the image.

The worker's image deliberately does NOT ship `workflow-definitions` — it
receives everything about a workflow in the claim payload, so it has no reason
to read those files. That makes its "nothing to find" case the normal one in
production, not an edge case, which is exactly why it needs a test.
"""

from __future__ import annotations

from pathlib import Path

from worker.core.config import _repo_root


def _make_container_layout(root: Path) -> Path:
    """Builds `<root>/worker/core/config.py`, mirroring the image."""
    module = root / "worker" / "core" / "config.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    return module


def test_starts_with_no_env_file_and_no_definitions(tmp_path: Path) -> None:
    """The production case: a bare image where config comes from the
    environment. Must resolve to something, never raise."""
    module = _make_container_layout(tmp_path)

    resolved = _repo_root(module)

    assert isinstance(resolved, Path)
    assert resolved in module.parents


def test_finds_an_env_file_when_one_is_mounted(tmp_path: Path) -> None:
    """Some deployments mount a .env rather than passing variables."""
    module = _make_container_layout(tmp_path)
    (tmp_path / ".env").write_text("WORKER_NAME=mounted\n", encoding="utf-8")

    assert _repo_root(module) == tmp_path


def test_shallow_paths_do_not_raise_indexerror(tmp_path: Path) -> None:
    """Guards the regression at every depth down to the filesystem root."""
    current = _make_container_layout(tmp_path)
    while current.parent != current:
        assert isinstance(_repo_root(current), Path)
        current = current.parent


def test_settings_import_works() -> None:
    """The import that crashed on boot."""
    from worker.core.config import settings

    assert settings.api_v1.endswith("/api/v1")
    assert settings.runtime
