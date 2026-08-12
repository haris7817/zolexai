"""Per-job scratch space.

Every job gets one directory, and everything it writes — staged inputs,
intermediate segments, the final file — lives inside it. When the job ends, for
any reason, the directory goes.

That the cleanup is unconditional is the whole point. A GPU node runs jobs
continuously on a disk that is usually far smaller than the media it handles;
one leaked half-gigabyte source video per failed job fills it within a day, and
a full disk fails every *subsequent* job with an error that points nowhere near
the cause.

M1 had none of this because the mock worker never touched the filesystem.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from worker.adapters.base import AdapterError
from worker.core.config import settings
from worker.core.logging import get_logger

logger = get_logger(__name__)

_BYTES_PER_MB = 1024 * 1024


def assert_disk_available(root: Path) -> None:
    """Refuses to start when the workspace is nearly full.

    Cheap, and it converts a class of failure that is confusing and contagious
    into one clear retriable error before any compute is spent.
    """
    root.mkdir(parents=True, exist_ok=True)
    free_mb = shutil.disk_usage(root).free // _BYTES_PER_MB
    if free_mb < settings.min_free_disk_mb:
        raise AdapterError(
            "The service is briefly at capacity. Please try again shortly.",
            internal_detail=(
                f"workspace has {free_mb}MB free, "
                f"below MIN_FREE_DISK_MB={settings.min_free_disk_mb}"
            ),
        )


@contextmanager
def job_workspace(job_id: str) -> Iterator[Path]:
    """Creates the job's directory and removes it on the way out.

    `keep_workspace_on_failure` leaves it behind for debugging — off by default,
    because the failure mode of forgetting it is a disk that fills silently.
    """
    root = settings.workspace_root
    assert_disk_available(root)

    path = root / f"job-{job_id}"
    # A retry of the same job reuses the id, and stale files from the previous
    # attempt would be indistinguishable from this attempt's own.
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)

    failed = False
    try:
        yield path
    except BaseException:
        failed = True
        raise
    finally:
        if failed and settings.keep_workspace_on_failure:
            logger.warning("workspace_retained", extra={"path": str(path)})
        else:
            shutil.rmtree(path, ignore_errors=True)
