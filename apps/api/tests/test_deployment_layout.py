"""Configuration must load in the CONTAINER layout, not just the repo layout.

## Why this file exists

`REPO_ROOT` was originally `Path(__file__).resolve().parents[4]`, which is
correct in the repository — `apps/api/app/core/config.py` really is four levels
below the root. Inside the image the same module sits at
`/app/app/core/config.py`, where only three parents exist, so that expression
raised `IndexError` at import time and the container died before FastAPI or
Alembic could start.

It reached production because every check ran against a host checkout. The unit
tests, the API suite and the browser end-to-end run all imported the module from
the repo, where the assumption holds. Building the image was not enough either —
the failure is at *runtime*, and `docker build` never imports the module.

These tests exercise the resolver against a simulated deployment layout, so the
assumption is checked rather than inherited from wherever the tests happen to
live.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import _repo_root


def _make_container_layout(root: Path, *, with_definitions: bool) -> Path:
    """Builds `<root>/app/core/config.py`, mirroring the image.

    The depth is what matters: in the image the module has exactly three
    parents above it before the filesystem root.
    """
    module = root / "app" / "core" / "config.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    if with_definitions:
        (root / "workflow-definitions").mkdir()
    return module


def test_resolves_when_definitions_sit_beside_the_app(tmp_path: Path) -> None:
    """The real container: `COPY workflow-definitions /workflow-definitions`."""
    module = _make_container_layout(tmp_path, with_definitions=True)

    assert _repo_root(module) == tmp_path


def test_does_not_crash_when_there_is_no_repository_at_all(tmp_path: Path) -> None:
    """A layout with nothing to find must still return a path, not raise.

    This is the exact production failure: the resolver has to degrade to
    something usable, because every value it would have read from disk is
    supplied by the environment in a deployed service.
    """
    module = _make_container_layout(tmp_path, with_definitions=False)

    resolved = _repo_root(module)

    assert isinstance(resolved, Path)
    assert resolved in module.parents


def test_shallow_paths_do_not_raise_indexerror(tmp_path: Path) -> None:
    """Guards the specific regression, at every depth down to the root.

    A future refactor that reintroduces a fixed `parents[N]` index passes the
    deep cases and fails here.
    """
    module = _make_container_layout(tmp_path, with_definitions=False)

    current = module
    while current.parent != current:
        # No assertion on the value — the point is that nothing raises,
        # however few parents the caller happens to have.
        assert isinstance(_repo_root(current), Path)
        current = current.parent


def test_repo_layout_still_finds_the_real_root() -> None:
    """The development path must keep working — the fix must not trade one
    layout for the other."""
    resolved = _repo_root()

    assert (resolved / "workflow-definitions").is_dir()
    assert (resolved / "workflow-definitions" / "text-to-video.yaml").is_file()


def test_settings_import_yields_a_usable_definitions_directory() -> None:
    """End result of all the above: the registry has somewhere to load from."""
    from app.core.config import settings

    assert settings.workflow_definitions_dir.is_dir()
