"""The platform's language vocabulary for sung lyrics.

## Why this exists

Three parts of the pipeline had an opinion about what "Spanish" means and none
of them agreed. The UI offers an English display name ("Spanish"); the lyric
writer wants to know whether it can write in that language at all; and the
music model wants an ISO 639-1 code (`es`) and silently applies English to
anything it does not recognise. Resolving that in three places is how a
selection ends up meaning nothing.

So it is resolved here, once. **The canonical internal representation is the
ISO 639-1 code** — `es`, never "Spanish", never "es-ES". Everything above this
file may speak display names because that is what a customer picked;
everything below it speaks codes.

## What this file is not

It is not a provider mapping. Which codes the *model* can actually sing is a
property of the model, lives in `acestep.py`, and is checked there — this
catalogue is the product's offer, and the two are allowed to differ. They
happen to agree today (every code below is in the model's supported set, and
`tests/test_music_language.py` pins that they still do), but a future provider
with a smaller repertoire must fail loudly rather than silently redefine what
the product offers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    """One language the product offers for generated lyrics."""

    code: str
    """Canonical ISO 639-1 code. This is what travels through the platform."""

    name: str
    """English display name — what the UI shows and what it sends today."""


class UnknownLanguage(ValueError):
    """A requested language is not one the platform offers.

    Raised rather than defaulted, deliberately. A value that reaches here and
    is not recognised means a client sent something the dropdown cannot
    produce; quietly treating it as English is precisely the failure this
    module was written to end.
    """


#: The offered languages, in the order the UI lists them.
#:
#: This must stay in step with `LYRIC_LANGUAGES` in
#: `apps/web/src/features/generation/form.ts` — the web app's dropdown is the
#: only thing that produces these values, and a language offered there but
#: missing here is a job that fails instead of a song that sings.
_CATALOGUE: tuple[Language, ...] = (
    Language("en", "English"),
    Language("ur", "Urdu"),
    Language("hi", "Hindi"),
    Language("ar", "Arabic"),
    Language("es", "Spanish"),
    Language("fr", "French"),
    Language("de", "German"),
    Language("pt", "Portuguese"),
    Language("it", "Italian"),
    Language("tr", "Turkish"),
    Language("ru", "Russian"),
    Language("ja", "Japanese"),
    Language("ko", "Korean"),
    Language("zh", "Chinese"),
)

ENGLISH = _CATALOGUE[0]

_BY_KEY: dict[str, Language] = {}
for _language in _CATALOGUE:
    _BY_KEY[_language.code] = _language
    _BY_KEY[_language.name.lower()] = _language


def offered() -> tuple[Language, ...]:
    """Every language the product offers for generated lyrics."""
    return _CATALOGUE


def resolve_language(value: str | None) -> Language | None:
    """The canonical language for a requested value, or None if none was asked for.

    Accepts either representation the platform has ever used — the display
    name the UI sends ("Spanish") or the canonical code ("es") — because job
    history stores whatever was current when the job ran, and "Reuse settings"
    on an old job must not become a different song.

    Raises `UnknownLanguage` for anything else. There is no fallback: a
    misspelled language is a bug worth seeing, not a reason to sing English at
    someone who asked for Urdu.
    """
    if value is None:
        return None
    key = value.strip().lower()
    if not key:
        return None
    try:
        return _BY_KEY[key]
    except KeyError:
        raise UnknownLanguage(
            f"{value!r} is not a language this platform offers; "
            f"expected one of {[language.name for language in _CATALOGUE]}"
        ) from None
