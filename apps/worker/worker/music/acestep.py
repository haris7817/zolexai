"""ACE-Step provider — the only file that knows which music model we run.

## How it runs

ACE-Step is a long-lived HTTP service, not a per-job subprocess. That is a
meaningful difference from the video runtime: LTX pays its model-load cost on
every render, while ACE-Step loads ~24 GB once and then answers requests in
seconds. So the worker talks to it over HTTP and never manages its lifecycle —
the service is started and supervised outside this process, exactly like a
database.

Measured on the client's RTX 5090 (2026-08-13), XL-turbo + the 4B LM:

  * 30s → 1.55s, 60s → 1.71s, 240s → 5.54s. Cost tracks *step count*, not
    length, so duration is nearly free.
  * Peak VRAM 23.9 GB and **flat across that whole range** — a four-minute
    song costs no more memory than a thirty-second one.
  * Durations come back exact (+24 ms, one MP3 frame).

The service natively covers 10s–600s, which spans the entire product range in
one pass. The adapter's sectioning machinery therefore never triggers; it stays
as insurance against a future provider that cannot do this.

## The API

Three calls, and the shapes are not guessable — each one below was verified
against the running service rather than taken from documentation:

    POST /release_task    → {"data": {"task_id": ...}}
    POST /query_result    → {"data": [{"task_id", "status", "result": "<JSON string>"}]}
    GET  /v1/audio?path=  → the audio bytes

Two traps worth naming, both of which cost time to find:

  * the poll parameter is **`task_id_list`**, not `task_ids`. The wrong name
    returns `200` with an empty list, which reads exactly like "not finished
    yet" and will poll until it times out.
  * `result` is a JSON string *inside* the JSON response, so it needs decoding
    twice.

## What this file must never do

Leak. No caller sees "ACE-Step", a checkpoint name, a step count or a sampler.
Errors raised from here carry provider detail in `internal_detail` only, and
`tests/test_acestep_provider.py` pins that.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from worker.core.config import settings
from worker.core.logging import get_logger
from worker.music.provider import (
    MusicRequest,
    MusicTake,
    ProviderGenerationError,
    ProviderProgress,
    ProviderUnavailable,
)

logger = get_logger(__name__)

#: Statuses the service reports for a task. Anything negative is a failure;
#: `result` being populated is what actually means "done", so these are used
#: for logging and early failure rather than as the completion test.
_STATUS_FAILED = {-1, 2, 3}


class AceStepProvider:
    name = "acestep"

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        max_seconds: float | None = None,
        request_timeout: float | None = None,
        generation_timeout: float | None = None,
        poll_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._transport = transport
        """
        Injected HTTP transport, for tests only.

        The alternative is running the real service to prove that a poll uses
        the right parameter name — and that parameter name is exactly the sort
        of thing that breaks silently, so it needs a test that runs everywhere.
        """

        # `is not None`, not `or`: an explicitly empty URL means "deliberately
        # unconfigured" and must reach the guard in `generate`, rather than
        # silently falling back to the default and failing later as a
        # connection refusal to somewhere nobody asked for.
        chosen = base_url if base_url is not None else settings.acestep_base_url
        self._base_url = (chosen or "").rstrip("/")
        self._api_key = api_key if api_key is not None else settings.acestep_api_key
        self.max_seconds = float(
            max_seconds if max_seconds is not None else settings.acestep_max_seconds
        )
        self._request_timeout = float(
            request_timeout if request_timeout is not None else settings.acestep_request_timeout
        )
        self._generation_timeout = float(
            generation_timeout
            if generation_timeout is not None
            else settings.acestep_generation_timeout
        )
        self._poll_seconds = float(
            poll_seconds if poll_seconds is not None else settings.acestep_poll_seconds
        )

    # ── The contract ─────────────────────────────────────────────────────

    async def generate(
        self,
        request: MusicRequest,
        workspace: Path,
        on_progress: ProviderProgress | None = None,
    ) -> list[MusicTake]:
        if not self._base_url:
            raise ProviderUnavailable(
                "ACESTEP_BASE_URL is empty; no music service is configured"
            )
        if request.duration_seconds > self.max_seconds:
            raise ProviderGenerationError(
                f"requested {request.duration_seconds:.0f}s exceeds the provider's "
                f"{self.max_seconds:.0f}s single-generation ceiling"
            )

        payload = self.build_payload(request)
        headers = {"Authorization": self._api_key} if self._api_key else {}

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._request_timeout),
            headers=headers,
            transport=self._transport,
        ) as client:
            task_id = await self._submit(client, payload)
            entries = await self._await_result(client, task_id, on_progress)
            return await self._download(client, entries, workspace)

    # ── Request construction (pure — this is what unit tests pin) ────────

    def build_payload(self, request: MusicRequest) -> dict[str, Any]:
        """Translates the platform's vocabulary into the service's.

        Kept pure and separate from the HTTP call so the mapping can be proven
        without a running service — including the parts that are easy to get
        quietly wrong, like an instrumental being an empty lyrics string rather
        than a flag.
        """
        payload: dict[str, Any] = {
            "caption": request.prompt,
            # Empty lyrics is how this service is told to produce an
            # instrumental. Verified on the GPU: an empty string yields no
            # vocals, rather than the model inventing its own words.
            "lyrics": "" if request.instrumental else (request.lyrics or ""),
            "audio_duration": float(request.duration_seconds),
        }

        if request.bpm is not None:
            payload["bpm"] = int(request.bpm)
        if request.key:
            payload["key_scale"] = request.key

        # The service defaults to a random seed. Sending one makes a retried
        # job reproduce its own result instead of surprising the user with a
        # different song.
        if request.seed is None:
            payload["use_random_seed"] = True
        else:
            payload["use_random_seed"] = False
            payload["seed"] = int(request.seed)

        if request.reference_audio is not None:
            payload["reference_audio"] = str(request.reference_audio)

        return payload

    # ── HTTP ─────────────────────────────────────────────────────────────

    async def _submit(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> str:
        try:
            response = await client.post("/release_task", json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderGenerationError(
                f"music service rejected the request ({exc.response.status_code}): "
                f"{exc.response.text[:400]}"
            ) from exc
        except httpx.RequestError as exc:
            # Connection refused / DNS / timeout on submit means the service is
            # not there, which retrying this job will not fix.
            raise ProviderUnavailable(
                f"music service unreachable at {self._base_url}: {exc}"
            ) from exc

        task_id = ((body or {}).get("data") or {}).get("task_id")
        if not task_id:
            raise ProviderGenerationError(
                f"music service returned no task id: {str(body)[:400]}"
            )
        logger.info("music_task_submitted", extra={"task_id": task_id})
        return str(task_id)

    async def _await_result(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        on_progress: ProviderProgress | None,
    ) -> list[dict[str, Any]]:
        """Polls until the task produces results, fails, or the budget runs out.

        Progress is time-based against the expected duration rather than
        reported by the service, which exposes none. It is therefore an
        estimate, and it is capped below 1.0 so the bar never claims completion
        the moment before something fails.
        """
        deadline = time.monotonic() + self._generation_timeout
        started = time.monotonic()

        while True:
            entries = await self._poll_once(client, task_id)
            if entries:
                return entries

            if time.monotonic() > deadline:
                raise ProviderGenerationError(
                    f"music service produced no result for task {task_id} within "
                    f"{self._generation_timeout:.0f}s"
                )

            if on_progress is not None:
                elapsed = time.monotonic() - started
                await on_progress(min(0.95, elapsed / max(1.0, self._generation_timeout)))

            await asyncio.sleep(self._poll_seconds)

    async def _poll_once(
        self, client: httpx.AsyncClient, task_id: str
    ) -> list[dict[str, Any]]:
        """One poll. Returns the finished entries, or [] if still working."""
        try:
            response = await client.post(
                "/query_result",
                # `task_id_list`, NOT `task_ids`. The wrong key returns 200
                # with an empty list, indistinguishable from "still running".
                json={"task_id_list": [task_id]},
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderGenerationError(
                f"music service failed while polling ({exc.response.status_code})"
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderGenerationError(
                f"lost contact with the music service while polling: {exc}"
            ) from exc

        for record in (body or {}).get("data") or []:
            status = record.get("status")
            if status in _STATUS_FAILED:
                raise ProviderGenerationError(
                    f"music service reported task {task_id} failed (status={status})"
                )

            raw = record.get("result")
            if not raw:
                continue
            # `result` is a JSON document encoded as a string inside the JSON
            # response — decoded twice, on purpose, not by accident.
            entries = json.loads(raw) if isinstance(raw, str) else raw
            if entries:
                return list(entries)

        return []

    async def _download(
        self,
        client: httpx.AsyncClient,
        entries: list[dict[str, Any]],
        workspace: Path,
    ) -> list[MusicTake]:
        destination = workspace / "provider"
        destination.mkdir(parents=True, exist_ok=True)

        takes: list[MusicTake] = []
        for index, entry in enumerate(entries):
            location = entry.get("file")
            if not location:
                continue

            path = destination / f"take-{index:02d}.mp3"
            try:
                response = await client.get(location)
                response.raise_for_status()
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                raise ProviderGenerationError(
                    f"could not fetch generated audio from the music service: {exc}"
                ) from exc

            path.write_bytes(response.content)
            if path.stat().st_size == 0:
                raise ProviderGenerationError(
                    "the music service returned an empty audio file"
                )

            takes.append(
                MusicTake(
                    path=path,
                    seed=_first_seed(entry.get("seed_value")),
                    metadata=dict(entry.get("metas") or {}),
                )
            )

        if not takes:
            raise ProviderGenerationError(
                "the music service reported success but returned no audio"
            )

        logger.info(
            "music_takes_downloaded",
            extra={"takes": len(takes), "bytes": sum(t.path.stat().st_size for t in takes)},
        )
        return takes


def _first_seed(value: Any) -> int | None:
    """The service reports one seed per take as a comma-separated string."""
    if value is None:
        return None
    first = str(value).split(",")[0].strip()
    try:
        return int(first)
    except ValueError:
        return None
