"""Music runtime — minute-length songs through a swappable provider.

## What this file owns, and what it does not

It owns everything that is true of *a song* regardless of which model makes it:
reading a minute-based duration, planning structure per genre, sizing and
quality-checking the lyric sheet, deciding how many generations a length needs,
assembling them, matching loudness, and refusing to ship anything that does not
validate.

It owns none of the model. `worker/music/provider.py` is the seam and
`worker/music/acestep.py` is the current implementation; swapping models means
writing one class, not touching this file.

## Length

Music is chosen in minutes (client requirement) and the workflow offers 1–5.
Nothing here knows a ceiling — `provider.max_seconds` does. With the current
provider that is 600s, which spans the entire product range, so `_plan_sections`
returns a single section and the crossfade/assembly path below is a no-op
pass-through. It stays because the *next* provider may not be so generous, and
because a length ceiling is a property of a model rather than of the product.

## Long songs, if a provider ever needs them

Sections are where the obvious cheat lives — generate one, repeat it, and the
file is the right length and unmistakably a loop. Two things prevent it: each
section is generated from its own place in the song plan, and byte-identical
output across sections fails the job rather than shipping.
"""

from __future__ import annotations

import hashlib
import math
import zlib
from pathlib import Path

from worker.adapters.base import (
    AdapterError,
    AdapterJob,
    AdapterResult,
    ProgressCallback,
    cancellable,
    parse_duration_seconds,
)
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.longform import StageReporter, band_for
from worker.media import (
    FfmpegError,
    OutputExpectation,
    Segment,
    crossfade_concat,
    duration_tolerance,
    loudness_normalize,
    overlap_cost_seconds,
    plan_segments,
    verify_output,
)
from worker.music import (
    LyricBrief,
    LyricsWriter,
    MusicGenerationProvider,
    MusicRequest,
    MusicTake,
    ProviderGenerationError,
    ProviderUnavailable,
    SongPlan,
    check_lyric_fit,
    plan_song,
    write_lyrics,
)

logger = get_logger(__name__)

#: Sections are compared by content hash. Two identical sections mean the
#: provider ignored its per-section input, and the result would be a loop.
_DUPLICATE_CHECK_MINIMUM_SECTIONS = 2


class MusicAdapter:
    name = "music"

    def __init__(
        self,
        provider: MusicGenerationProvider | None = None,
        writer: LyricsWriter | None = None,
    ) -> None:
        self._provider = provider
        """
        What actually makes the audio. Resolved lazily from configuration when
        not injected, so constructing the adapter never touches the network and
        a test can supply a fake without any environment at all.
        """

        self._writer = writer
        """
        Whatever writes lyrics. Resolved lazily from configuration when not
        injected, exactly like the provider.

        This is NOT optional in practice: the current provider treats an empty
        lyric sheet as "make an instrumental" (verified on the GPU,
        2026-08-16), so running without a writer means no production track
        ever has sung words. That was the client's "lyrics not present"
        complaint. An earlier comment here claimed the provider writes its own
        words from the prompt — it does not.
        """

    def supports(self, workflow_id: str) -> bool:
        return workflow_id == "music"

    # ── The run ──────────────────────────────────────────────────────────

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        reporter = StageReporter(on_progress)
        await reporter.preparing("Setting up your track…")

        provider = self._resolve_provider()
        total_seconds = self._requested_seconds(job)

        # ── Plan the song before asking for a note of it ─────────────
        brief = LyricBrief.from_prompt(job.prompt)
        plan = plan_song(total_seconds, genre=brief.genre)
        logger.info(
            "music_planned",
            extra={
                "provider": provider.name,
                "genre": plan.genre,
                "total_seconds": round(total_seconds, 1),
                "line_budget": plan.line_budget,
                "outline": plan.outline,
            },
        )

        await reporter.report("preparing", 12, "Writing your song…")
        lyrics = await self._lyrics_for(job, plan, brief, total_seconds)

        # ── Generate ─────────────────────────────────────────────────
        fade = max(0.0, float(settings.music_crossfade_seconds))
        sections = self._plan_sections(job, provider, total_seconds, fade)
        rendered = await self._render_sections(
            job, reporter, provider, sections, plan, lyrics
        )

        # ── Assemble ─────────────────────────────────────────────────
        await reporter.stitching("Putting your track together…")
        output = job.workspace / "output.mp3"
        try:
            joined = await cancellable(
                job,
                crossfade_concat(
                    rendered, job.workspace / "joined.mp3", fade_seconds=fade or 0.05
                ),
            )
            await reporter.finalizing("Balancing the mix…")
            await cancellable(job, loudness_normalize(joined, output))
            info = await verify_output(
                output,
                OutputExpectation(
                    expect_audio=True,
                    expected_seconds=total_seconds,
                    tolerance_seconds=duration_tolerance(total_seconds, floor=2.0),
                ),
            )
        except FfmpegError as exc:
            raise AdapterError(
                "This track could not be completed. Please try again.",
                internal_detail=f"assembly or validation failed: {exc}",
            ) from exc

        await reporter.uploading()
        return AdapterResult(
            path=output,
            content_type="audio/mpeg",
            kind="audio",
            duration_seconds=info.duration_seconds,
        )

    # ── Provider resolution ──────────────────────────────────────────────

    def _resolve_provider(self) -> MusicGenerationProvider:
        """The configured provider, or a refusal naming what is missing.

        Imported here rather than at module scope so that adding a provider
        with heavy dependencies cannot slow down or break worker startup for
        nodes that never run music.
        """
        if self._provider is not None:
            return self._provider

        choice = (settings.music_provider or "").strip().lower()
        if choice == "acestep":
            from worker.music.acestep import AceStepProvider

            self._provider = AceStepProvider()
            return self._provider

        raise AdapterError(
            "This tool is temporarily unavailable.",
            internal_detail=(
                f"MUSIC_PROVIDER={choice!r} is not a known music provider; "
                "expected 'acestep'"
            ),
            retriable=False,
        )

    def _resolve_writer(self) -> LyricsWriter | None:
        """The configured lyrics writer, or None only when deliberately off.

        Mirrors `_resolve_provider`: lazy so construction stays free, injected
        writers win, and the import lives here so a writer with heavy
        dependencies never taxes worker startup. Unlike the provider, an
        unknown value degrades to no writer rather than failing the job —
        a misconfigured writer should cost lyric quality, not the track.
        """
        if self._writer is not None:
            return self._writer

        choice = (settings.music_lyrics_writer or "").strip().lower()
        if choice == "template":
            from worker.music.writer import TemplateLyricsWriter

            self._writer = TemplateLyricsWriter()
        elif choice:
            logger.warning(
                "unknown_lyrics_writer",
                extra={"configured": choice, "known": ["template"]},
            )
        return self._writer

    # ── Lyrics ───────────────────────────────────────────────────────────

    async def _lyrics_for(
        self,
        job: AdapterJob,
        plan: SongPlan,
        brief: LyricBrief,
        total_seconds: float,
    ) -> str | None:
        """The words to sing: the customer's own, or ours, or none.

        A user's own lyrics are passed through untouched — they are the one
        thing we must never rewrite. They are still *measured*, because a sheet
        longer than the song can hold gets silently truncated by the model, and
        a warning in the log is how anyone ever finds out.
        """
        supplied = str(job.parameters.get("lyrics") or "").strip()
        if supplied:
            fit = check_lyric_fit(supplied, total_seconds)
            if not fit.fits:
                # Deliberately not truncated here. Cutting a customer's words
                # to fit would be worse than the model doing it, because we
                # would be choosing which ones to lose.
                logger.warning(
                    "lyrics_exceed_duration",
                    extra={
                        "lines": fit.lines,
                        "budget": fit.budget,
                        "overflow": fit.overflow,
                        "total_seconds": round(total_seconds, 1),
                    },
                )
            return supplied

        if job.parameters.get("instrumental"):
            return None

        written = await write_lyrics(brief, plan, self._resolve_writer())
        if written is None:
            # Only two ways here: the plan is wordless (ambient/instrumental
            # genres) or the writer is deliberately disabled. Returning None
            # sends the provider an empty sheet, which it treats as a request
            # for an instrumental — never as an invitation to write its own
            # words. Verified on the GPU, 2026-08-16.
            return None

        text, review = written
        (job.workspace / "lyrics.txt").write_text(text, encoding="utf-8")
        logger.info(
            "lyrics_ready",
            extra={
                "rhyme_rate": round(review.rhyme_rate, 2),
                "unique_rate": round(review.unique_rate, 2),
                "unresolved_issues": len(review.issues),
            },
        )
        return text

    # ── Planning ─────────────────────────────────────────────────────────

    def _requested_seconds(self, job: AdapterJob) -> float:
        """The chosen length, in seconds, from a minute-based selection.

        `parse_duration_seconds` already reads "3m" as 180 — music is the
        reason that branch exists. Which lengths a customer may pick belongs to
        the workflow definition and is validated by the API, so an unusable
        value arriving here is a platform bug rather than a customer mistake.
        """
        seconds = parse_duration_seconds(job.parameters.get("duration"))
        if seconds is None:
            raise AdapterError(
                "This track could not be started.",
                internal_detail=f"no usable duration in {job.parameters!r}",
                retriable=False,
            )
        return seconds

    def _plan_sections(
        self,
        job: AdapterJob,
        provider: MusicGenerationProvider,
        total_seconds: float,
        fade_seconds: float,
    ) -> list[Segment]:
        """Generation windows, with the crossfade overlap paid for up front.

        The ceiling comes from the PROVIDER, never from a constant here. With a
        provider that covers the whole product range this returns exactly one
        section and no crossfade is ever applied.

        When it does split: each join consumes one fade of material, so
        generating exactly the requested length and then crossfading yields a
        song short by `(sections - 1) x fade`. Planning the overlap in is what
        makes the delivered length the length the user picked — which is the
        only thing the validation at the end will accept.
        """
        ceiling = max(
            5.0,
            float(job.execution_int("max_segment_seconds", int(provider.max_seconds))),
        )
        if total_seconds <= ceiling:
            return plan_segments(total_seconds, max_segment_seconds=ceiling)

        # Chicken and egg: the padding depends on how many sections there are,
        # and how many sections there are depends on the padding. Two or three
        # iterations always settle it; the bound is a guard, not a schedule.
        count = math.ceil(total_seconds / ceiling)
        for _ in range(8):
            padded = total_seconds + overlap_cost_seconds(count, fade_seconds)
            needed = max(1, math.ceil(padded / ceiling))
            if needed == count:
                break
            count = needed

        padded = total_seconds + overlap_cost_seconds(count, fade_seconds)
        # Even windows rather than "ceiling, ceiling, remainder": a five-minute
        # song split 60/60/60/60/60/7 ends on a seven-second fragment that
        # sounds like exactly what it is.
        return plan_segments(padded, max_segment_seconds=padded / count + 1e-6)

    # ── Generation ───────────────────────────────────────────────────────

    async def _render_sections(
        self,
        job: AdapterJob,
        reporter: StageReporter,
        provider: MusicGenerationProvider,
        sections: list[Segment],
        plan: SongPlan,
        lyrics: str | None,
    ) -> list[Path]:
        total = len(sections)
        rendered: list[Path] = []
        digests: dict[str, int] = {}

        for section in sections:
            job.raise_if_cancelled()
            low, high = band_for(section.index, total)
            await reporter.section(
                section.index + 1,
                total,
                low,
                start_seconds=section.start_seconds,
                end_seconds=section.start_seconds + section.duration_seconds,
            )

            request = MusicRequest(
                prompt=self._caption_for(job, plan, section, total),
                duration_seconds=section.duration_seconds,
                lyrics=lyrics,
                bpm=_optional_int(job.parameters.get("bpm")),
                key=_optional_str(job.parameters.get("key")),
                # Deterministic per section: a retried job reproduces its own
                # song rather than handing the user a different one.
                seed=zlib.crc32(f"{job.job_id}:{section.index}".encode()),
                reference_audio=self._reference_audio(job),
            )

            async def report(fraction: float, low: int = low, high: int = high) -> None:
                await reporter.generating(low + int((high - low) * max(0.0, min(1.0, fraction))))

            take = await self._generate(job, provider, request, report)

            digest = hashlib.sha256(take.path.read_bytes()).hexdigest()
            if total >= _DUPLICATE_CHECK_MINIMUM_SECTIONS and digest in digests:
                raise AdapterError(
                    "This track could not be completed. Please try again.",
                    internal_detail=(
                        f"section {section.index} is byte-identical to section "
                        f"{digests[digest]}; a repeated section is a loop, not a "
                        "longer song"
                    ),
                )
            digests[digest] = section.index

            rendered.append(take.path)
            await reporter.section(
                section.index + 1,
                total,
                high,
                start_seconds=section.start_seconds,
                end_seconds=section.start_seconds + section.duration_seconds,
            )

        return rendered

    async def _generate(
        self,
        job: AdapterJob,
        provider: MusicGenerationProvider,
        request: MusicRequest,
        report,
    ) -> MusicTake:
        """One provider call, with its failures translated for the customer.

        The provider's exception types carry the whole distinction that matters
        to the platform: unavailable is a deployment fault that retrying cannot
        fix, while a generation error is worth another attempt.
        """
        try:
            takes = await cancellable(
                job, provider.generate(request, job.workspace, report)
            )
        except ProviderUnavailable as exc:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=f"music provider '{provider.name}' unavailable: {exc}",
                retriable=False,
            ) from exc
        except ProviderGenerationError as exc:
            raise AdapterError(
                "This track could not be completed. Please try again.",
                internal_detail=f"music provider '{provider.name}' failed: {exc}",
            ) from exc

        if not takes:
            raise AdapterError(
                "This track could not be completed. Please try again.",
                internal_detail=f"music provider '{provider.name}' returned no audio",
            )
        # Providers may offer several alternatives; the product delivers one
        # result per job, so the first is the take. Surfacing the rest would be
        # a product decision, not an implementation one.
        return takes[0]

    def _caption_for(
        self, job: AdapterJob, plan: SongPlan, section: Segment, total: int
    ) -> str:
        """The user's prompt, plus which part of the song this section is.

        The prompt itself is never rewritten — the structure hint is appended,
        and only when there is more than one section to distinguish. A
        single-section song sends exactly what the user typed.
        """
        if total <= 1:
            return job.prompt

        start = section.start_seconds
        end = start + section.duration_seconds
        covered: list[str] = []
        cursor = 0.0
        for part in plan.sections:
            part_end = cursor + part.seconds
            if part_end > start and cursor < end:
                covered.append(part.kind)
            cursor = part_end

        outline = " → ".join(covered) if covered else plan.outline
        return f"{job.prompt}\n\n[section {section.index + 1} of {total}: {outline}]"

    def _reference_audio(self, job: AdapterJob) -> Path | None:
        item = job.input_for("reference_audio")
        return item.path if item is not None else None


def _optional_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None
