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

import dataclasses
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
from worker.comfy import evict_comfy_vram
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
    Language,
    LyricBrief,
    LyricsWriter,
    MusicGenerationProvider,
    MusicRequest,
    MusicTake,
    NoLyricsWriterAvailable,
    ProviderGenerationError,
    ProviderUnavailable,
    SongPlan,
    UnknownLanguage,
    UnsupportedLyricLanguage,
    check_lyric_fit,
    offered,
    plan_song,
    resolve_language,
    singable_details,
    vocal_intent,
    write_lyrics,
)
from worker.music.fallback import is_available
from typing import Any

from worker.music.gates import LyricsValidationFailed
from worker.music.trim import choose_window, cut_window
from worker.music.report import customer_result, job_report, write_report
from worker.music.verify import VerificationReport, verify_song
from worker.music.workflow import WORKFLOW_VERSION, LyricsOptions, prepare_song

logger = get_logger(__name__)

#: Sections are compared by content hash. Two identical sections mean the
#: provider ignored its per-section input, and the result would be a loop.
_DUPLICATE_CHECK_MINIMUM_SECTIONS = 2


#: Display names for the codes a writer reports it can handle, so a refusal
#: names "English" rather than "en". Sourced from the language catalogue rather
#: than written out again, because two lists of language names is how they
#: start disagreeing.
_LANGUAGE_NAMES = {language.code: language.name for language in offered()}


def _english_list(items: list[str]) -> str:
    """"a", "a and b", "a, b and c" — for a sentence a customer reads."""
    if len(items) <= 1:
        return items[0] if items else ""
    return f"{', '.join(items[:-1])} and {items[-1]}"


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

        # Lazy co-tenancy eviction — see ltx.py: H3's ComfyUI stays warm
        # between its own jobs; whoever needs the card next clears it.
        await evict_comfy_vram()

        provider = self._resolve_provider()
        total_seconds = self._requested_seconds(job)

        # ── Plan the song before asking for a note of it ─────────────
        brief = LyricBrief.from_prompt(job.prompt)
        # Applied HERE, once, so the writer and the reviewer agree on what a
        # detail is. `salient_details` collects every capitalised word, which
        # includes the sentence's first — "Two people falling in love" yields
        # "Two", and a writer told to keep it verbatim wedges an English word
        # into a Spanish chorus. Filtering in only one of the two places would
        # instead have the reviewer demand a word the writer was told to drop.
        brief = dataclasses.replace(
            brief, must_keep=singable_details(brief.must_keep)
        )
        language = self._language_for(job)
        if language is not None:
            brief = dataclasses.replace(brief, language=language.code)
        # Whether anyone sings is the customer's decision, not the genre
        # table's. `detect_genre` reads "a lo-fi pop song with soft female
        # vocals" as `ambient`, `ambient` is wordless, and the track came back
        # an instrumental with the words "female vocals" still sitting in the
        # request — the "beat but no lyrics" complaint, reproduced on demand.
        # The prompt is also the ONLY way to ask for an instrumental: there is
        # no such field on the API and no toggle in the panel.
        plan = plan_song(
            total_seconds, genre=brief.genre, vocals=vocal_intent(job.prompt)
        )
        logger.info(
            "music_planned",
            extra={
                "provider": provider.name,
                "genre": plan.genre,
                "total_seconds": round(total_seconds, 1),
                "line_budget": plan.line_budget,
                "outline": plan.outline,
                # A wordless plan produces a track with no vocals in it. That
                # is a legitimate answer to "an instrumental piano piece" and a
                # bug in answer to anything else, and the two are told apart
                # here or not at all.
                "wordless": plan.wordless,
                # The selection, and what it became. Both, because the whole
                # failure this replaced was a value that looked present at
                # every stage and meant nothing at the last one.
                "language_selected": job.parameters.get("lyrics_language"),
                "language_code": language.code if language else None,
            },
        )

        await reporter.report("preparing", 12, "Writing your song…")

        # ── Music Lyrics Workflow v2.0 (client specification, 9 Sep 2026) ─
        # Lyrics first, validated, then sung, then measured. An instrumental
        # request has no lyrics to validate and takes the older path as it
        # always did; so does a deployment or a job that asks for "v1".
        if (
            plan.has_lyrics
            and not job.parameters.get("instrumental")
            and self._lyrics_workflow(job) == "v2"
        ):
            return await self._run_v2(
                job, reporter, provider, plan, brief, language, total_seconds
            )

        lyrics = await self._lyrics_for(job, plan, brief, total_seconds)

        # ── Generate ─────────────────────────────────────────────────
        fade = max(0.0, float(settings.music_crossfade_seconds))
        sections = self._plan_sections(job, provider, total_seconds, fade)
        rendered = await self._render_sections(
            job, reporter, provider, sections, plan, lyrics, language
        )

        # ── Assemble ─────────────────────────────────────────────────
        output = job.workspace / "output.mp3"
        info = await self._assemble(job, reporter, rendered, fade, total_seconds, output)

        await reporter.uploading()
        return AdapterResult(
            path=output,
            content_type="audio/mpeg",
            kind="audio",
            duration_seconds=info.duration_seconds,
        )

    async def _assemble(
        self,
        job: AdapterJob,
        reporter: StageReporter,
        rendered: list[Path],
        fade: float,
        total_seconds: float,
        output: Path,
        *,
        extra_tolerance: float = 0.0,
    ):
        """Crossfade, loudness-match and validate the rendered sections."""
        await reporter.stitching("Putting your track together…")
        joined_path = output.with_name(output.stem + "-joined.mp3")
        try:
            joined = await cancellable(
                job,
                crossfade_concat(rendered, joined_path, fade_seconds=fade or 0.05),
            )
            await reporter.finalizing("Balancing the mix…")
            await cancellable(job, loudness_normalize(joined, output))
            return await verify_output(
                output,
                OutputExpectation(
                    expect_audio=True,
                    expected_seconds=total_seconds,
                    tolerance_seconds=duration_tolerance(total_seconds, floor=2.0) + extra_tolerance,
                ),
            )
        except FfmpegError as exc:
            raise AdapterError(
                "This track could not be completed. Please try again.",
                internal_detail=f"assembly or validation failed: {exc}",
            ) from exc

    # ── The v2 workflow ──────────────────────────────────────────────────

    @staticmethod
    def _lyrics_workflow(job: AdapterJob) -> str:
        chosen = str(job.execution.get("music_lyrics_workflow") or settings.music_lyrics_workflow)
        return "v1" if chosen.strip().lower() == "v1" else "v2"

    async def _run_v2(
        self,
        job: AdapterJob,
        reporter: StageReporter,
        provider: MusicGenerationProvider,
        plan: SongPlan,
        brief: LyricBrief,
        language: Language | None,
        total_seconds: float,
    ) -> AdapterResult:
        """Outline → lyrics → gates → (dry run stops) → longer take → cut
        around the singing → verify (coverage, breaks, recall, reference)
        → retry with a new seed → deliver or fail per policy.

        The sheet is written once and kept across retries; what a retry
        changes is the seed and the firmness of the production brief,
        because the model's arrangement — not the words — is what varies
        between takes. The cut is what makes the intro and the tail the
        model insists on stop counting against the song (10 Sep 2026).
        """
        options = LyricsOptions.from_job(job)
        supplied = bool(str(job.parameters.get("lyrics") or "").strip())
        writer: LyricsWriter | None = None
        if not supplied:
            writer = self._resolve_writer()
            self._refuse_a_language_the_writer_cannot_write(writer, brief, plan)

        work = job.workspace / "lyrics"
        try:
            prepared = await cancellable(
                job,
                prepare_song(
                    job=job,
                    plan=plan,
                    brief=brief,
                    writer=writer,
                    options=options,
                    workspace=work,
                    reference_path=self._reference_audio(job),
                ),
            )
        except LyricsValidationFailed as exc:
            raise self._failure_for(exc.code, exc.detail) from exc
        except (NoLyricsWriterAvailable, UnsupportedLyricLanguage) as exc:
            raise AdapterError(
                "We could not write lyrics for this track just now. Please try "
                "again, or paste your own lyrics and we will sing those.",
                internal_detail=f"[LYRICS_WRITER_FAILED] {exc}",
                retriable=True,
            ) from exc

        requested_language = job.parameters.get("lyrics_language")

        # ── Reference conditioning is mandatory when a reference exists ──
        conditioning: dict[str, Any] | None = None
        reference_file: Path | None = None
        if prepared.reference is not None:
            reference_file = self._reference_audio(job) or self._fetched_reference(work)
            if not getattr(provider, "supports_reference", False) or reference_file is None:
                raise self._failure_for(
                    "REFERENCE_CONDITIONING_FAILED",
                    f"provider {provider.name!r} cannot condition on a reference, or the file is gone",
                )
            conditioning = {
                "reference_audio_loaded": True,
                "reference_embedding_created": True,
                "reference_conditioning_applied": True,
                "reference_duration": round(prepared.reference.duration_seconds, 1),
                "target_bpm": prepared.reference.bpm,
                "target_key": (prepared.reference.key or "").replace(" ", "_") or None,
                "strength": options.reference_strength,
            }
            logger.info("music_reference_conditioning", extra=conditioning)

        if options.dry_run:
            report = job_report(
                job_id=job.job_id, prompt=job.prompt, prepared=prepared, options=options,
                verification=None, retries=0, retry_reasons=[], status="planned", failure_code=None,
                requested_language=requested_language, requested_seconds=total_seconds,
                actual_seconds=None, conditioning=conditioning,
            )
            path = write_report(work, report)
            self._log_report(report)
            raise AdapterError(
                "Dry run complete: the lyrics were planned and validated and no audio was "
                f"generated. Planned sung coverage {prepared.preflight.planned_vocal_coverage:.0%}, "
                f"rhyme groups passing {prepared.rhyme.pass_rate:.0%}, "
                f"{len(prepared.timed.lines)} timed lines.",
                internal_detail=f"[DRY_RUN] report at {path}",
                retriable=False,
            )

        # ── The take is longer than the song so the cut can choose ──────
        render_seconds = total_seconds
        if options.trim:
            # The provider's ceiling is per generation; `_plan_sections`
            # splits a longer take into sections, so the take itself is
            # not capped by it.
            render_seconds = total_seconds * options.overshoot + options.overshoot_seconds
        fade = max(0.0, float(settings.music_crossfade_seconds))
        sections = self._plan_sections(job, provider, render_seconds, fade)
        attempts = 1 + options.max_retries
        best: tuple[VerificationReport, Path, object, dict[str, Any] | None, dict[str, Any] | None] | None = None
        retry_reasons: list[str] = []
        caption = prepared.caption or job.prompt

        for attempt in range(attempts):
            job.raise_if_cancelled()
            rendered = await self._render_sections(
                job, reporter, provider, sections, plan, prepared.sheet, language,
                caption=caption, bpm=prepared.bpm, key=prepared.key, attempt=attempt,
                model_hears_reference=True, reference_strength=options.reference_strength,
            )
            suffix = "" if attempt == 0 else f"-take{attempt + 1}"
            raw = job.workspace / f"take{suffix}.mp3"
            output = job.workspace / f"output{suffix}.mp3"
            # The longer take is an intermediate; each crossfade join loses
            # a little beyond its planned fade, and the delivered file is
            # validated exactly after the cut.
            info = await self._assemble(
                job, reporter, rendered, fade, render_seconds, raw,
                extra_tolerance=1.0 * max(0, len(sections) - 1) if options.trim else 0.0,
            )

            # ── Cut the requested length around the singing ─────────────
            await reporter.finalizing("Fitting the song around the vocals…")
            window_dict: dict[str, Any] | None = None
            if options.trim and abs(render_seconds - total_seconds) > 0.5:
                spans = await cancellable(job, self._sung_spans(raw, brief.language))
                window = choose_window(
                    spans, source_seconds=float(info.duration_seconds or render_seconds), target_seconds=total_seconds
                )
                await cancellable(job, cut_window(raw, output, window))
                window_dict = window.to_dict()
                logger.info("music_take_cut", extra=window_dict)
            else:
                output = raw
            try:
                info = await verify_output(
                    output,
                    OutputExpectation(
                        expect_audio=True, expected_seconds=total_seconds,
                        tolerance_seconds=duration_tolerance(total_seconds, floor=2.0),
                    ),
                )
            except FfmpegError as exc:
                raise AdapterError(
                    "This track could not be completed. Please try again.",
                    internal_detail=f"cut validation failed: {exc}",
                ) from exc

            # ── Measure ─────────────────────────────────────────────────
            await reporter.finalizing("Checking the vocals…")
            verification = await cancellable(
                job,
                verify_song(
                    output, prepared.timed, language=brief.language, expected_seconds=total_seconds,
                    duration_seconds=float(info.duration_seconds or total_seconds),
                    coverage_target=options.coverage_target, recall_threshold=options.recall_threshold,
                    duration_tolerance_seconds=duration_tolerance(total_seconds, floor=2.0),
                    max_break_seconds=options.max_break_seconds,
                ),
            )
            comparison_dict: dict[str, Any] | None = None
            errors = list(verification.errors)
            if prepared.reference is not None:
                comparison = await cancellable(job, self._compare(prepared.reference, output, verification, options))
                comparison_dict = comparison.to_dict()
                mismatch = (
                    comparison.bpm_ok is False
                    or comparison.key_ok is False
                    or comparison.similarity < options.reference_similarity_threshold
                )
                if mismatch:
                    errors.append(("REFERENCE_MISMATCH", f"similarity {comparison.similarity:.0%}; " + "; ".join(comparison.reasons)))
            passed = not errors
            score = verification.score() + (comparison_dict["similarity"] if comparison_dict else 0.0)
            logger.info(
                "music_verified",
                extra={
                    "attempt": attempt + 1,
                    "passed": passed,
                    "measured_vocal_coverage": verification.measured_vocal_coverage,
                    "coverage_method": verification.coverage_method,
                    "longest_break_seconds": verification.longest_break_seconds,
                    "lyric_recall": verification.lyric_recall,
                    "recall_method": verification.recall_method,
                    "language_detected": verification.language_detected,
                    "window": window_dict,
                    "reference": comparison_dict,
                    "errors": [code for code, _ in errors],
                },
            )
            if best is None or score > best[0].score() + (best[4]["similarity"] if best[4] else 0.0):
                best = (verification, output, info, window_dict, comparison_dict)
                best_errors = errors
            if passed:
                break
            retry_reasons.append("; ".join(f"{code}: {detail}" for code, detail in errors))
            if attempt + 1 < attempts:
                caption = self._reinforced_caption(prepared.caption or job.prompt, verification)
                await reporter.report("generating", 30, "Re-recording the vocals…")

        assert best is not None
        verification, output, info, window_dict, comparison_dict = best
        failure_code = None if not best_errors else best_errors[0][0]
        delivering = not best_errors or options.verify_policy == "deliver_best"
        report = job_report(
            job_id=job.job_id, prompt=job.prompt, prepared=prepared, options=options,
            verification=verification, retries=len(retry_reasons), retry_reasons=retry_reasons,
            status="completed" if not best_errors else ("delivered_with_warnings" if delivering else "failed"),
            failure_code=failure_code, requested_language=requested_language,
            requested_seconds=total_seconds, actual_seconds=getattr(info, "duration_seconds", None),
            window=window_dict, comparison=comparison_dict, conditioning=conditioning,
        )
        write_report(work, report)
        self._log_report(report)

        if not delivering:
            raise self._failure_for(
                failure_code or "GENERATION_FAILED",
                retry_reasons[-1] if retry_reasons else "verification failed",
                after_render=True,
            )

        await reporter.uploading()
        result = customer_result(
            prepared, verification, retries=len(retry_reasons), window=window_dict, comparison=comparison_dict
        )
        if best_errors and options.verify_policy == "deliver_best":
            result["warnings"] = list(dict.fromkeys(result.get("warnings", []) + [d for _, d in best_errors]))
        return AdapterResult(
            path=output,
            content_type="audio/mpeg",
            kind="audio",
            duration_seconds=getattr(info, "duration_seconds", None),
            report=result,
        )

    @staticmethod
    def _fetched_reference(work: Path) -> Path | None:
        folder = work / "reference"
        if not folder.is_dir():
            return None
        files = [p for p in folder.glob("reference.*") if p.suffix not in {".part", ".ytdl", ".json"}]
        return files[0] if files else None

    @staticmethod
    async def _sung_spans(path: Path, language: str) -> list[tuple[float, float]] | None:
        """Where the singing is, for the cut: the stem, else the transcript."""
        from worker.media.vocals import spans_from_envelope, vocal_activity
        from worker.music.transcribe import transcribe

        spans = await vocal_activity(path)
        if spans is not None:
            return spans
        transcript = await transcribe(path, language=language)
        if transcript is None or not transcript.words:
            return None
        hop = 0.05
        last = max(w.end for w in transcript.words)
        env = [0.0] * (int(last / hop) + 2)
        for w in transcript.words:
            for i in range(int(w.start / hop), min(len(env), int(w.end / hop) + 1)):
                env[i] = 1.0
        return spans_from_envelope(env, abs_floor=0.5, rel_fraction=0.5)

    @staticmethod
    async def _compare(reference, output: Path, verification: VerificationReport, options: LyricsOptions):
        """The finished song measured the same way the reference was."""
        from worker.media import audio_envelope
        from worker.music.keys import analyse_tempo_and_key
        from worker.music.reference import ReferenceProfile, compare_to_reference, energy_curve

        tempo, key = await analyse_tempo_and_key(output)
        envelope = await audio_envelope(output, hop_seconds=0.05)
        candidate = ReferenceProfile(
            source="candidate", duration_seconds=verification.duration_seconds,
            bpm=tempo.bpm if tempo else None, tempo_stability=tempo.confidence if tempo else None,
            time_signature="4/4", energy_curve=tuple(energy_curve(envelope)), section_boundaries=(),
            vocal_density=verification.measured_vocal_coverage, vocal_cadence=None, words_per_second=None,
            language=verification.language_detected, language_probability=verification.language_probability,
            mood_hint="", key=key.name if key else None, mode=key.mode if key else None,
            key_confidence=key.confidence if key else None,
        )
        return compare_to_reference(reference, candidate, bpm_tolerance=options.reference_bpm_tolerance)

    @staticmethod
    def _reinforced_caption(caption: str, verification: VerificationReport) -> str:
        """A firmer production brief for the next take, naming the fault."""
        notes = []
        codes = {code for code, _ in verification.errors}
        if "VOCAL_COVERAGE_BELOW_90" in codes:
            notes.append(
                "the lead vocal must start in the first two seconds and sing continuously "
                "to the end with no instrumental break longer than a few seconds"
            )
        if "LYRIC_RECALL_TOO_LOW" in codes:
            notes.append("sing every written line clearly and in order, word for word, no improvised vocals")
        if not notes:
            notes.append("vocals throughout, every lyric line performed")
        return caption + "\n\n" + "; ".join(notes)

    @staticmethod
    def _log_report(report: dict) -> None:
        keys = (
            "workflow_version", "job_id", "request_sha256", "requested_language", "language",
            "detected_language", "requested_duration_seconds", "actual_duration_seconds",
            "blueprint_version", "lyrics_source", "writer_rounds", "rhyme_repairs",
            "planned_vocal_coverage", "filler_ratio", "measured_vocal_coverage",
            "coverage_method", "lyric_recall", "retry_count", "retry_reasons", "status",
            "failure_code",
        )
        extra = {key: report.get(key) for key in keys}
        extra["reference"] = report.get("reference", {}).get("source")
        extra["rhyme_pass_rate"] = report.get("rhyme", {}).get("pass_rate")
        extra["originality"] = report.get("originality", {}).get("result")
        extra["missing_lines"] = len(report.get("missing_lines") or [])
        extra["substituted_lines"] = len(report.get("substituted_lines") or [])
        logger.info("music_lyrics_report", extra=extra)

    @staticmethod
    def _failure_for(code: str, detail: str, *, after_render: bool = False) -> AdapterError:
        """The workflow's failure code as a job failure with customer copy."""
        messages = {
            "REFERENCE_UNAVAILABLE": (
                "We could not fetch the reference link. Check that it is public and try "
                "again, or upload the track instead.",
                False,
            ),
            "REFERENCE_AUDIO_INVALID": (
                "The reference audio could not be used. It needs to be an audible track "
                "between 5 seconds and 10 minutes long.",
                False,
            ),
            "VOCAL_COVERAGE_BELOW_90": (
                "The song came back with too much instrumental time and not enough singing, "
                "even after re-recording. Please try again."
                if after_render
                else "We could not write lyrics dense enough to fill this song. Please try "
                "again, or paste your own lyrics.",
                True,
            ),
            "LYRIC_RECALL_TOO_LOW": (
                "The vocals did not sing enough of the lyrics as written, even after "
                "re-recording. Please try again.",
                True,
            ),
            "RHYME_VALIDATION_FAILED": (
                "We could not make every line rhyme as required. Please try again, choose "
                "the relaxed rhyme mode, or paste your own lyrics.",
                True,
            ),
            "LANGUAGE_MISMATCH": (
                "The lyrics did not come out in the language you chose. Please try again.",
                True,
            ),
            "REFERENCE_SIMILARITY_TOO_HIGH": (
                "The lyrics came out too close to the reference track and were refused. "
                "Please try again.",
                True,
            ),
            "FILLER_ABOVE_LIMIT": (
                "The lyrics had too much filler in them. Please try again.",
                True,
            ),
            "BLUEPRINT_VALIDATION_FAILED": (
                "We could not plan this song's lyrics. Please try again.",
                True,
            ),
            "OUTPUT_DURATION_MISMATCH": (
                "This track could not be completed. Please try again.",
                True,
            ),
            "INSTRUMENTAL_BREAK_TOO_LONG": (
                "The song came back with a long instrumental break in the middle of the vocals, "
                "even after re-recording. Please try again.",
                True,
            ),
            "LYRIC_QUALITY_FAILED": (
                "We could not write lyrics that read naturally enough in the language you chose. "
                "Please try again, or paste your own lyrics.",
                True,
            ),
            "REFERENCE_MISMATCH": (
                "The song did not match the reference track's tempo, key or feel closely enough, "
                "even after re-recording. Please try again.",
                True,
            ),
            "REFERENCE_CONDITIONING_FAILED": (
                "The reference track could not be applied to the music model on this server.",
                False,
            ),
        }
        message, retriable = messages.get(
            code, ("This track could not be completed. Please try again.", True)
        )
        return AdapterError(message, internal_detail=f"[{code}] {detail}", retriable=retriable)

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

        from worker.music.fallback import FallbackLyricsWriter

        names = [
            name.strip().lower()
            for name in (settings.music_lyrics_writer or "").split(",")
            if name.strip()
        ]
        writers: list[LyricsWriter] = []
        for name in names:
            if name == "template":
                from worker.music.writer import TemplateLyricsWriter

                writers.append(TemplateLyricsWriter())
            elif name == "cerebras":
                from worker.music.cerebras import CerebrasLyricsWriter

                writers.append(CerebrasLyricsWriter())
            else:
                logger.warning(
                    "unknown_lyrics_writer",
                    extra={"configured": name, "known": ["cerebras", "template"]},
                )

        if not writers:
            return None
        # A single configured writer is returned unwrapped, so that a
        # deployment naming exactly one writer behaves — and logs — precisely
        # as it did before the chain existed.
        self._writer = writers[0] if len(writers) == 1 else FallbackLyricsWriter(writers)
        return self._writer

    # ── Language ─────────────────────────────────────────────────────────

    def _language_for(self, job: AdapterJob) -> Language | None:
        """The requested language as a canonical code, resolved exactly once.

        Resolved here, at the top of the run, rather than at each of the two
        places that need it — the lyric writer and the provider — because
        those two disagreeing about what "Spanish" means is the failure mode
        this whole path is being repaired for.
        """
        requested = job.parameters.get("lyrics_language")
        try:
            return resolve_language(str(requested) if requested is not None else None)
        except UnknownLanguage as exc:
            # A closed dropdown cannot produce this, so it is a client bug.
            # Failing names it; defaulting to English hides it and produces
            # exactly the song the customer complained about.
            raise AdapterError(
                "This track could not be started.",
                internal_detail=str(exc),
                retriable=False,
            ) from exc

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

        writer = self._resolve_writer()
        self._refuse_a_language_the_writer_cannot_write(writer, brief, plan)

        try:
            written = await write_lyrics(brief, plan, writer)
        except (NoLyricsWriterAvailable, UnsupportedLyricLanguage) as exc:
            # Every writer that could have served this language has been tried
            # and none produced a sheet. Failing here is the instruction: the
            # alternative is an empty sheet, and an empty sheet is how the music
            # model is told to make an INSTRUMENTAL — so a silent return would
            # hand a wordless track to someone who asked for a song with words,
            # which is the original complaint restored by a different route.
            raise AdapterError(
                "We could not write lyrics for this track just now. Please try "
                "again, or paste your own lyrics and we will sing those.",
                internal_detail=str(exc),
                # Retriable: the common cause is a rate limit or a timeout at
                # the writing service, and the next attempt costs one text call
                # rather than a GPU render.
                retriable=True,
            ) from exc

        if written is None:
            # Only two ways here: the plan is wordless or the writer is
            # deliberately disabled. Returning None sends the provider an empty
            # sheet, which it treats as a request for an instrumental — never
            # as an invitation to write its own words. Verified on the GPU,
            # 2026-08-16.
            #
            # Said out loud, because this is the exact line a "beat with no
            # lyrics" complaint has to be traced back to, and a silent return
            # makes that trace impossible. `plan_song` has already logged which
            # of the two it was and what the prompt asked for.
            logger.info(
                "music_instrumental_selected",
                extra={
                    "workflow_id": job.workflow_id,
                    "wordless_plan": plan.wordless,
                    "genre": plan.genre,
                    "reason": (
                        "the plan carries no sung sections"
                        if plan.wordless
                        else "the lyrics writer is disabled"
                    ),
                },
            )
            return None

        text, review = written
        (job.workspace / "lyrics.txt").write_text(text, encoding="utf-8")
        # Not decoration: it is the difference between "we wrote Spanish
        # lyrics" and "we wrote English lyrics and labelled them Spanish",
        # which is indistinguishable from the outside until someone listens.
        #
        # `lyrics_provider` is the operational half of that — whether the
        # hosted writer answered or the chain fell through to the local bank is
        # the first thing worth knowing when a song comes back in the wrong
        # register, and it is not visible anywhere else.
        logger.info(
            "lyrics_ready",
            extra={
                "lyrics_mode": "auto",
                "lyrics_provider": getattr(writer, "last_writer", "")
                or getattr(writer, "name", type(writer).__name__),
                "rhyme_rate": round(review.rhyme_rate, 2),
                "unique_rate": round(review.unique_rate, 2),
                "unresolved_issues": len(review.issues),
                "language": brief.language,
                "characters": len(text),
            },
        )
        return text

    def _refuse_a_language_the_writer_cannot_write(
        self, writer: LyricsWriter | None, brief: LyricBrief, plan: SongPlan
    ) -> None:
        """Stops before writing English words for a customer who asked for Urdu.

        This only ever fires on the AUTOMATIC lyric path. A customer's own
        lyrics are already in the language they wrote them in and are passed
        through untouched — the selection then only tells the model how to
        pronounce them, which every offered language supports. An instrumental
        has no words to be in any language at all.

        Refusing is the instruction, not a preference: a writer that cannot
        work in the requested language must not quietly substitute one it can.
        The message says what to do instead, because the paste-your-own-lyrics
        path genuinely does work in every language offered.
        """
        if writer is None or not plan.has_lyrics:
            return

        # Asked FIRST, because "no writer can run" and "the writer that can run
        # does not do this language" are different situations that deserve
        # different copy — and because the protocol expresses the first one as
        # an empty language set, which is indistinguishable from "any".
        if not is_available(writer):
            reason = getattr(writer, "unavailable_reason", lambda: "not configured")()
            raise AdapterError(
                "We cannot write lyrics automatically just now. Add your own "
                "lyrics in the language you picked and we will sing those.",
                internal_detail=f"no lyrics writer is available: {reason}",
                retriable=False,
            )

        supported: frozenset[str] = getattr(writer, "supported_languages", frozenset())
        # An empty set means "any" — a language model has no list to give.
        if not supported or brief.language in supported:
            return

        writable = sorted(supported)
        raise AdapterError(
            "We can only write lyrics for you in "
            + _english_list([_LANGUAGE_NAMES.get(code, code) for code in writable])
            + " at the moment. Add your own lyrics in the language you picked "
            "and we will sing those instead.",
            internal_detail=(
                f"lyrics writer cannot write {brief.language!r}; "
                f"it supports {writable}"
            ),
            retriable=False,
        )

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
        language: Language | None,
        *,
        caption: str | None = None,
        bpm: int | None = None,
        attempt: int = 0,
        model_hears_reference: bool = True,
        key: str | None = None,
        reference_strength: float | None = None,
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
                prompt=self._caption_for(job, plan, section, total, base=caption),
                duration_seconds=section.duration_seconds,
                lyrics=lyrics,
                # The words and the accent to sing them with are two decisions.
                # Passing only the words is what let a Spanish sheet be sung in
                # English — the provider defaults the language when nobody
                # states it, and "nobody stated it" was the whole bug.
                language=language.code if language else None,
                bpm=bpm if bpm is not None else _optional_int(job.parameters.get("bpm")),
                key=key or _optional_str(job.parameters.get("key")),
                reference_strength=reference_strength,
                # Deterministic per section: a retried job reproduces its own
                # song rather than handing the user a different one. A v2
                # verification retry is a different take on purpose, so it
                # salts the seed with the attempt.
                seed=zlib.crc32(
                    (
                        f"{job.job_id}:{section.index}"
                        if attempt == 0
                        else f"{job.job_id}:{section.index}:take{attempt}"
                    ).encode()
                ),
                # The v2 lyrics workflow keeps the reference away from the
                # model: given a track, the provider conditions on its audio
                # and can reproduce its melody, which the client's originality
                # rule forbids. There the reference shapes the blueprint and
                # the brief (tempo, energy, cadence) and nothing else.
                reference_audio=(self._reference_audio(job) or self._fetched_reference(job.workspace / "lyrics")) if model_hears_reference else None,
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
        self, job: AdapterJob, plan: SongPlan, section: Segment, total: int,
        *, base: str | None = None,
    ) -> str:
        """The user's prompt, plus which part of the song this section is.

        The prompt itself is never rewritten — the structure hint is appended,
        and only when there is more than one section to distinguish. A
        single-section song sends exactly what the user typed. `base` is the
        v2 workflow's production brief (the prompt plus its direction).
        """
        prompt = base if base is not None else job.prompt
        if total <= 1:
            return prompt

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
        return f"{prompt}\n\n[section {section.index + 1} of {total}: {outline}]"

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
