"""Smoke test for automatic lyrics — the text half, with no GPU involved.

`music_smoke.py` needs the ACE-Step service and therefore the GPU node. This
one needs neither: it drives the configured lyrics writer directly, so the
whole language matrix can be checked in seconds from a laptop with nothing but
a CEREBRAS_API_KEY.

That split is the point. "Did Cerebras write Spanish?" and "did ACE-Step sing
it?" are two questions, they fail for entirely different reasons, and answering
the first one cheaply is what makes the second one worth spending a GPU on.

Usage:

    # every language the product offers, one short song each
    CEREBRAS_API_KEY=... python scripts/lyrics_smoke.py

    # a specific set, and a specific length
    CEREBRAS_API_KEY=... LANGUAGES=es,fr,pt,ja DURATION=180 \\
        python scripts/lyrics_smoke.py romantic latin pop about a summer night

    # prove the fallback: no key, English still gets written, Spanish refuses
    LANGUAGES=en,es python scripts/lyrics_smoke.py

What it reports per language is exactly what the acceptance criteria ask for:
which writer answered, whether the returned text really is in the requested
language, how long it took, and how many lines came back against the budget the
duration allows.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker.adapters.music import MusicAdapter  # noqa: E402
from worker.music import (  # noqa: E402
    LyricBrief,
    NoLyricsWriterAvailable,
    UnsupportedLyricLanguage,
    offered,
    parse_sections,
    plan_song,
    resolve_language,
    write_lyrics,
    written_in,
)
from worker.music.fallback import is_available, writer_name  # noqa: E402

_DEFAULT_PROMPT = (
    "Romantic Latin pop song about two people falling in love during a summer "
    "night by the ocean. Warm female vocalist, acoustic guitar, soft piano and "
    "modern Latin percussion."
)


async def main() -> int:
    prompt = " ".join(sys.argv[1:]) or _DEFAULT_PROMPT
    seconds = float(os.getenv("DURATION", "120"))
    requested = os.getenv("LANGUAGES", "")
    codes = (
        [code.strip() for code in requested.split(",") if code.strip()]
        if requested
        else [language.code for language in offered()]
    )

    writer = MusicAdapter()._resolve_writer()
    if writer is None:
        print("no lyrics writer is configured (MUSIC_LYRICS_WRITER is empty)")
        return 1

    members = getattr(writer, "writers", [writer])
    print(f"prompt:    {prompt}")
    print(f"duration:  {seconds:.0f}s")
    print("writers:   " + ", ".join(
        f"{writer_name(m)}{'' if is_available(m) else ' (unavailable)'}"
        for m in members
    ))
    print("-" * 78)
    print(f"{'lang':<6} {'writer':<10} {'lang ok':<8} {'lines':<7} {'ms':<7} note")
    print("-" * 78)

    failures = 0
    for code in codes:
        language = resolve_language(code)
        if language is None:
            print(f"{code:<6} {'-':<10} {'-':<8} {'-':<7} {'-':<7} not an offered language")
            failures += 1
            continue

        brief = LyricBrief.from_prompt(prompt)
        brief = dataclasses.replace(brief, language=language.code)
        plan = plan_song(seconds, genre=brief.genre)

        started = time.monotonic()
        try:
            written = await write_lyrics(brief, plan, writer)
        except (NoLyricsWriterAvailable, UnsupportedLyricLanguage) as exc:
            elapsed = (time.monotonic() - started) * 1000
            print(
                f"{language.code:<6} {'-':<10} {'-':<8} {'-':<7} {elapsed:<7.0f} "
                f"REFUSED: {str(exc)[:120]}"
            )
            failures += 1
            continue
        elapsed = (time.monotonic() - started) * 1000

        if written is None:
            print(
                f"{language.code:<6} {'-':<10} {'-':<8} {'-':<7} {elapsed:<7.0f} "
                "no lyrics (wordless plan)"
            )
            continue

        text, review = written
        lines = sum(len(section) for _, section in parse_sections(text))
        verdict = written_in(text, language.code)
        # None means "too little text to tell", which is a real third state and
        # is not the same as passing.
        mark = {True: "yes", False: "NO", None: "?"}[verdict]
        if verdict is False:
            failures += 1

        used = getattr(writer, "last_writer", "") or writer_name(writer)
        note = f"budget {plan.line_budget}, issues {len(review.issues)}"
        print(
            f"{language.code:<6} {used:<10} {mark:<8} {lines:<7} {elapsed:<7.0f} {note}"
        )

        if os.getenv("SHOW_LYRICS"):
            for line in text.splitlines():
                print(f"       {line}")
            print()

    print("-" * 78)
    print(f"languages checked: {len(codes)}, problems: {failures}")
    # Non-zero on any wrong-language or refused result, so this is usable as a
    # gate rather than something a human has to read carefully every time.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
