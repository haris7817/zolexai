"""HTTP client for the pinned ComfyUI service.

Same shape as the ACE-Step provider: the worker connects to a long-lived local
service, submits work, polls, and collects a file. It never launches ComfyUI,
never imports its code, and treats an unreachable service as "this runtime is
unavailable" rather than something to work around.

Progress honesty: ComfyUI's history endpoint reports queued/running/done — it
does not expose per-step progress for the Extender's monolithic node. The
client therefore reports elapsed time and stage transitions it actually
observed, and never invents section counters it cannot see.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from worker.adapters.base import AdapterJob

logger = logging.getLogger("zolexai.worker.comfy")


class ComfyError(Exception):
    """A ComfyUI-side failure, already split for the two audiences."""

    def __init__(
        self, user_message: str, *, internal_detail: str = "", retriable: bool = True
    ) -> None:
        self.user_message = user_message
        self.internal_detail = internal_detail or user_message
        self.retriable = retriable
        super().__init__(self.internal_detail)


class ComfyClient:
    def __init__(
        self,
        base_url: str,
        *,
        request_timeout: float = 30.0,
        poll_seconds: float = 3.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._request_timeout = request_timeout
        self._poll_seconds = poll_seconds
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._request_timeout,
            transport=self._transport,
        )

    # ── Health ───────────────────────────────────────────────────────────

    async def system_stats(self) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.get("/system_stats")
            resp.raise_for_status()
            return resp.json()

    async def reachable(self) -> tuple[bool, str]:
        """Cheap liveness: the stats endpoint, not the megabyte of object_info."""
        try:
            stats = await self.system_stats()
        except Exception as exc:  # noqa: BLE001 - any failure means unreachable
            return False, f"ComfyUI unreachable at {self._base_url}: {exc}"
        devices = stats.get("devices") or []
        vram = devices[0].get("vram_total", 0) if devices else 0
        return True, f"ComfyUI up, {len(devices)} device(s), vram_total={vram}"

    async def node_classes(self) -> set[str]:
        async with self._client() as client:
            resp = await client.get("/object_info")
            resp.raise_for_status()
            return set(resp.json().keys())

    # ── Submit / wait / collect ──────────────────────────────────────────

    async def submit(self, api_prompt: dict[str, Any], *, client_id: str) -> str:
        try:
            async with self._client() as client:
                resp = await client.post(
                    "/prompt", json={"prompt": api_prompt, "client_id": client_id}
                )
        except httpx.HTTPError as exc:
            raise ComfyError(
                "The video service is not responding.",
                internal_detail=f"POST /prompt failed: {exc}",
            ) from exc
        if resp.status_code >= 400:
            raise ComfyError(
                "This request could not be started.",
                internal_detail=f"POST /prompt {resp.status_code}: {resp.text[:2000]}",
                retriable=False,
            )
        body = resp.json()
        node_errors = body.get("node_errors") or {}
        if node_errors:
            # Validation failures are configuration problems (missing model,
            # bad widget) — retrying the identical prompt cannot help.
            raise ComfyError(
                "This request could not be started.",
                internal_detail=f"node_errors: {json.dumps(node_errors)[:2000]}",
                retriable=False,
            )
        prompt_id = body.get("prompt_id")
        if not prompt_id:
            raise ComfyError(
                "This request could not be started.",
                internal_detail=f"no prompt_id in response: {json.dumps(body)[:500]}",
                retriable=False,
            )
        return str(prompt_id)

    async def history(self, prompt_id: str) -> dict[str, Any] | None:
        async with self._client() as client:
            resp = await client.get(f"/history/{prompt_id}")
            resp.raise_for_status()
            body = resp.json()
        return body.get(prompt_id)

    async def free_memory(self) -> None:
        """Asks ComfyUI to unload models and release VRAM.

        On a co-tenanted GPU node this is what lets an LTX or ACE-Step job run
        after an H3 one: ComfyUI otherwise keeps ~52 GB resident between jobs,
        and 52 (idle ComfyUI) + 24 (ACE-Step) + an LTX pass does not fit the
        card. The next H3 job pays a model reload (~40-60 s) — measured, and
        cheap against an OOM'd customer job.
        """
        try:
            async with self._client() as client:
                await client.post(
                    "/free", json={"unload_models": True, "free_memory": True}
                )
        except Exception:  # noqa: BLE001 - best effort; health will catch worse
            logger.warning("comfy_free_failed", exc_info=True)

    async def interrupt(self) -> None:
        """Best-effort: a cancelled job should stop burning the GPU."""
        try:
            async with self._client() as client:
                await client.post("/interrupt")
        except Exception:  # noqa: BLE001 - the job is dying either way
            logger.warning("comfy_interrupt_failed", exc_info=True)

    async def cancel(self, prompt_id: str) -> None:
        """Best-effort removal of ONE prompt, wherever it sits in the queue.

        `/interrupt` only stops the prompt that is currently *executing*. A
        prompt still waiting in the queue survives it and runs later as an
        orphan — observed in production on 25 Aug 2026, when a budget-expired
        30s render held the GPU for twenty minutes nobody would collect while
        its own retry queued behind it. So: delete the prompt from the pending
        queue first, then interrupt only if it is the one actually running —
        never blindly, because a blind interrupt kills an innocent neighbour.
        """
        try:
            async with self._client() as client:
                await client.post("/queue", json={"delete": [prompt_id]})
                resp = await client.get("/queue")
                state = resp.json() if resp.status_code < 400 else {}
                running = {
                    entry[1]
                    for entry in state.get("queue_running", [])
                    if isinstance(entry, (list, tuple)) and len(entry) > 1
                }
                if prompt_id in running:
                    await client.post("/interrupt")
        except Exception:  # noqa: BLE001 - the job is dying either way
            logger.warning(
                "comfy_cancel_failed", extra={"prompt_id": prompt_id}, exc_info=True
            )

    async def wait(
        self,
        job: AdapterJob,
        prompt_id: str,
        *,
        timeout_seconds: float,
        on_tick: Callable[[float], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Polls until the prompt completes; cooperative with cancellation.

        Returns the history entry. Raises ComfyError on failure status or
        timeout, and lets JobCancelled/JobTimedOut from
        `job.raise_if_cancelled()` propagate after interrupting the server so
        the GPU is released.
        """
        started = time.monotonic()
        try:
            while True:
                elapsed = time.monotonic() - started
                if elapsed > timeout_seconds:
                    await self.cancel(prompt_id)
                    raise ComfyError(
                        "This generation took too long and was stopped.",
                        internal_detail=(
                            f"prompt {prompt_id} exceeded {timeout_seconds:.0f}s"
                        ),
                    )
                try:
                    job.raise_if_cancelled()
                except BaseException:
                    await self.cancel(prompt_id)
                    raise

                try:
                    entry = await self.history(prompt_id)
                except httpx.HTTPError as exc:
                    # One flaky poll is not a failed job; the next tick retries.
                    logger.warning("comfy_poll_failed", extra={"error": str(exc)})
                    entry = None

                if entry is not None:
                    status = (entry.get("status") or {}).get("status_str")
                    if status == "success":
                        return entry
                    messages = json.dumps(
                        (entry.get("status") or {}).get("messages", [])
                    )
                    raise ComfyError(
                        "This generation failed.",
                        internal_detail=(
                            f"prompt {prompt_id} status={status}: {messages[:2000]}"
                        ),
                    )

                if on_tick is not None:
                    await on_tick(elapsed)
                await asyncio.sleep(self._poll_seconds)
        except asyncio.CancelledError:
            # A hard task cancellation (attempt budget expiry, shutdown) must
            # not leave the prompt queued or rendering as an orphan. Shielded
            # because this coroutine is already being torn down.
            try:
                await asyncio.shield(self.cancel(prompt_id))
            except BaseException:  # noqa: BLE001 - best effort while dying
                logger.warning("comfy_cancel_on_teardown_failed", exc_info=True)
            raise


async def evict_comfy_vram(client: ComfyClient | None = None) -> None:
    """Frees ComfyUI's VRAM before another engine takes the card.

    The lazy half of the co-tenancy policy (25 Aug 2026): H3 keeps its ~52 GB
    warm between H3 jobs — saving the measured 40-60 s model reload every job
    used to pay — and the engine that actually needs the memory, LTX or
    music, calls this on its way in. Best-effort and cheap: a node with no
    ComfyUI, or one already empty, answers in milliseconds, and any failure
    is the health check's problem rather than this job's.
    """
    if client is None:
        from worker.core.config import settings

        client = ComfyClient(
            settings.h3_comfy_base_url,
            request_timeout=min(settings.h3_comfy_request_timeout, 10.0),
        )
    await client.free_memory()
