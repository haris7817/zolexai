from __future__ import annotations

import ipaddress
from pathlib import Path
from urllib.parse import urlparse

from .config import WorkerConfig
from .errors import ValidationError
from .utils import atomic_write_json, ensure_existing_file, format_template_argv, run_command


def _validated_url(url: str, allowed_hosts: tuple[str, ...]) -> tuple[str, str]:
    if len(url) > 2048:
        raise ValidationError("Reference-video URL must be 2048 characters or fewer")
    if any(ord(character) < 32 for character in url):
        raise ValidationError("Reference-video URL cannot contain control characters")
    parsed = urlparse(url)
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise ValidationError("Reference-video URL must use HTTPS")
    if parsed.username or parsed.password:
        raise ValidationError("Reference-video URL cannot contain credentials")
    host = parsed.hostname.casefold().rstrip(".")
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise ValidationError("Reference-video URL cannot use a local hostname")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValidationError("Reference-video URL cannot use an IP address")
    allowed = any(host == item or host.endswith(f".{item}") for item in allowed_hosts)
    if not allowed:
        raise ValidationError(
            f"Reference-video host {host} is not allowed; configure ZOLEX_REFERENCE_ALLOWED_HOSTS"
        )
    return url, host


def _validate_download(path: Path, config: WorkerConfig) -> Path:
    path = ensure_existing_file(path, "Downloaded reference video")
    size = path.stat().st_size
    if size <= 0:
        raise ValidationError("Downloaded reference video is empty")
    if size > config.reference_max_bytes:
        raise ValidationError(
            f"Downloaded reference video is {size} bytes; limit is {config.reference_max_bytes}"
        )
    return path


def _fetch_with_yt_dlp(url: str, output_dir: Path, config: WorkerConfig) -> Path:
    output_template = output_dir / "reference-video.%(ext)s"
    run_command(
        [
            config.reference_fetch_python,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--no-progress",
            "--no-warnings",
            "--socket-timeout",
            "30",
            "--retries",
            "3",
            "--max-filesize",
            str(config.reference_max_bytes),
            "-f",
            "bv*[height<=1080]+ba/b[height<=1080]/b",
            "--merge-output-format",
            "mp4",
            "--remux-video",
            "mp4",
            "-o",
            str(output_template),
            url,
        ],
        timeout=config.reference_fetch_timeout_seconds,
        log_path=output_dir / "fetch-command.json",
    )
    candidates = sorted(
        path
        for path in output_dir.glob("reference-video.*")
        if path.is_file() and path.suffix not in {".part", ".ytdl", ".json"}
    )
    if len(candidates) != 1:
        raise ValidationError("Reference fetcher did not create exactly one completed video file")
    return _validate_download(candidates[0], config)


def _fetch_with_command(url: str, output_dir: Path, config: WorkerConfig) -> Path:
    output = output_dir / "reference-video.mp4"
    request_path = output_dir / "fetch-request.json"
    atomic_write_json(
        request_path,
        {
            "schema_version": 1,
            "url": url,
            "output": str(output),
            "no_playlists": True,
            "max_bytes": config.reference_max_bytes,
            "max_duration_seconds": config.reference_max_seconds,
        },
    )
    run_command(
        format_template_argv(
            config.reference_fetch_command or [],
            {
                "request_json": str(request_path),
                "url": url,
                "output": str(output),
                "max_bytes": config.reference_max_bytes,
                "max_seconds": config.reference_max_seconds,
            },
        ),
        timeout=config.reference_fetch_timeout_seconds,
        log_path=output_dir / "fetch-command.json",
    )
    return _validate_download(output, config)


def fetch_reference_video(url: str, *, output_dir: Path, config: WorkerConfig) -> Path:
    url, host = _validated_url(url, config.reference_allowed_hosts)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "source.json"
    if metadata_path.exists():
        from .utils import read_json

        metadata = read_json(metadata_path)
        if isinstance(metadata, dict) and metadata.get("url") == url:
            existing = Path(str(metadata.get("local_path", "")))
            if existing.is_file():
                return _validate_download(existing, config)
    output = (
        _fetch_with_yt_dlp(url, output_dir, config)
        if config.reference_fetch_backend == "yt_dlp"
        else _fetch_with_command(url, output_dir, config)
    )
    atomic_write_json(
        metadata_path,
        {
            "schema_version": 1,
            "url": url,
            "host": host,
            "backend": config.reference_fetch_backend,
            "local_path": str(output),
            "bytes": output.stat().st_size,
        },
    )
    return output
