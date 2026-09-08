from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import ExternalCommandError, ValidationError


JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_job_id(value: str) -> str:
    if not JOB_ID_RE.fullmatch(value):
        raise ValidationError(
            "job_id must be 1-128 characters and contain only letters, numbers, '.', '_' or '-'"
        )
    return value


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Prevent two Linux worker processes from mutating the same job directory."""
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - production image is Linux.
        raise ValidationError("This worker requires POSIX file locking") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValidationError(f"Job is already running: {path.parent.name}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ensure_existing_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValidationError(f"{label} does not exist or is not a file: {path}")
    return path


def ensure_executable(name_or_path: str) -> str:
    from shutil import which

    found = which(name_or_path)
    if not found:
        raise ValidationError(f"Required executable was not found: {name_or_path}")
    return found


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Unable to read JSON {path}: {exc}") from exc


def run_command(
    argv: Iterable[str],
    *,
    timeout: int,
    cwd: Path | None = None,
    log_path: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    args = [str(item) for item in argv]
    if not args:
        raise ValidationError("External command cannot be empty")
    try:
        process_env = os.environ.copy()
        process_env.update(env_overrides or {})
        result = subprocess.run(
            args,
            cwd=cwd,
            env=process_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExternalCommandError(f"Command timed out after {timeout}s: {args[0]}") from exc
    except OSError as exc:
        raise ExternalCommandError(f"Unable to start command {args[0]}: {exc}") from exc

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "argv": args,
                    "returncode": result.returncode,
                    "env_overrides": env_overrides or {},
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if result.returncode != 0:
        stderr = result.stderr.strip()[-4000:]
        raise ExternalCommandError(
            f"Command failed with exit code {result.returncode}: {args[0]}\n{stderr}"
        )
    return result


def format_template_argv(template: list[str], values: dict[str, Any]) -> list[str]:
    rendered: list[str] = []
    for token in template:
        try:
            rendered.append(token.format_map(values))
        except KeyError as exc:
            raise ValidationError(f"Unknown command-template placeholder: {exc.args[0]}") from exc
    return rendered
