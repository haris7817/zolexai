"""The music provider seam.

This is the music half of the rule the video runtime already follows: the
platform speaks its own vocabulary, and exactly one class knows what the model
underneath is called. Everything above this file — the adapter, the job schema,
the API, the UI — describes a *song*; everything below describes a *model*.

Why it exists as a protocol rather than a base class: the two implementations
have nothing to share. `AceStepProvider` is an HTTP client for a long-lived
service; a future provider might be a subprocess, a queue, or a hosted API with
none of the same machinery. Inheritance would force a shape none of them
actually wants. The contract is four things — a name, a length ceiling, a
`generate` call, and what comes back — and that is all the adapter needs.

**A provider does one job: turn a `MusicRequest` into finished audio files.** It
does not plan song structure, write lyrics, crossfade, normalise loudness or
validate output — those are the adapter's, and they are identical whichever
model produced the notes. Keeping them out here is what stops a model swap
from becoming a rewrite.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: Reports progress *within one generation*, as a fraction from 0.0 to 1.0.
#:
#: Deliberately not the platform's (status, percent, message) callback: a
#: provider has no business knowing the job lifecycle or writing customer copy.
#: The adapter maps this fraction into whatever band the overall job is in,
#: which is what keeps the progress bar monotonic across planning, generation
#: and assembly.
ProviderProgress = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class MusicRequest:
    """One song to produce, in the platform's vocabulary.

    Every field here is something a customer chose or a plan derived. Nothing
    names a model, a checkpoint, a sampler or a step count — those belong to
    whichever provider is configured and are read from its own settings.
    """

    prompt: str
    """Style, genre, mood, instrumentation — free text, exactly as typed."""

    duration_seconds: float

    lyrics: str | None = None
    """
    Words to sing, with `[Verse 1]` / `[Chorus]` structure tags.

    `None` or empty means **instrumental** — see `instrumental` below. That
    equivalence is deliberate: it is one concept, so it gets one field rather
    than a flag that can contradict the lyrics beside it.
    """

    language: str | None = None
    """
    Canonical ISO 639-1 code for the language the vocals should be sung in.

    Not a hint about the words — `lyrics` already carries those, in whatever
    language they are written. This tells the model which *phonetics* to sing
    them with, which is a separate decision it will otherwise make wrongly:
    given a Spanish sheet and no language, the current provider defaults to
    English and sings Spanish words with an English accent.

    `None` means the caller expressed no preference and the provider's own
    default applies. It never means English.
    """

    bpm: int | None = None
    key: str | None = None
    """Musical key/scale, e.g. "C Major", "Am". None lets the model choose."""

    seed: int | None = None
    """None means the provider picks randomly. A value makes the result
    reproducible, which is what a retried job needs."""

    reference_audio: Path | None = None
    """Optional track whose character should guide the result."""

    takes: int = 1
    """How many alternatives to ask for. Providers may return fewer."""

    @property
    def instrumental(self) -> bool:
        return not (self.lyrics or "").strip()


@dataclass(frozen=True)
class MusicTake:
    """One finished audio file, plus what the provider knows about it."""

    path: Path
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    """
    Provider-reported facts about this take — measured tempo, key, the model
    that produced it.

    Diagnostics only. It reaches the log and may be stored alongside the asset,
    but nothing in the product branches on it and **it is never projected into
    a public API response**, because it is precisely the kind of thing that
    would leak a model name to a customer.
    """


class ProviderUnavailable(RuntimeError):
    """The provider cannot run at all: unconfigured, unreachable, or missing.

    Distinct from a generation *failing*. This one will not be fixed by
    retrying, so the adapter turns it into a non-retriable job failure with the
    reason in the log rather than burning three attempts against a service that
    is not there.
    """


class ProviderGenerationError(RuntimeError):
    """The provider ran and could not produce the song. Retrying may help."""


@runtime_checkable
class MusicGenerationProvider(Protocol):
    name: str
    """Internal identifier for logs and configuration. Never customer-facing."""

    max_seconds: float
    """
    Longest single generation this provider can produce.

    The adapter splits anything longer into sections and assembles them. A
    provider that natively covers the product's whole range therefore makes the
    sectioning path dead code rather than requiring it to be removed — which is
    the point of expressing the limit as a number instead of an assumption.
    """

    async def generate(
        self,
        request: MusicRequest,
        workspace: Path,
        on_progress: ProviderProgress | None = None,
    ) -> list[MusicTake]:
        """Produces audio for `request`, writing files inside `workspace`.

        Must raise `ProviderUnavailable` when it cannot run at all, and
        `ProviderGenerationError` when a run fails. Must return at least one
        take on success. Must be cancellable: the adapter races this call
        against the job's cancel signal, so it has to unwind on
        `asyncio.CancelledError` without leaving work behind.
        """
        ...
