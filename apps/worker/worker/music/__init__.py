"""Song planning, lyric quality, and the music provider seam.

Nothing in this package except `acestep.py` knows which model makes the audio.
Planning what a song of a given length and genre should contain, and measuring
whether a set of lyrics actually is that, are useful before a model is chosen
and stay useful after one is replaced — which is why they live here rather than
inside an adapter.
"""

from worker.music.language import (
    ENGLISH,
    Language,
    UnknownLanguage,
    offered,
    resolve_language,
)
from worker.music.lyrics import (
    LyricBrief,
    LyricFit,
    LyricIssue,
    LyricsReview,
    LyricsWriter,
    Section,
    SongPlan,
    UnsupportedLyricLanguage,
    check_lyric_fit,
    detect_genre,
    line_budget,
    lines_rhyme,
    parse_sections,
    plan_song,
    polish_lyrics,
    review_lyrics,
    rhyme_key,
    salient_details,
    write_lyrics,
)
from worker.music.provider import (
    MusicGenerationProvider,
    MusicRequest,
    MusicTake,
    ProviderGenerationError,
    ProviderProgress,
    ProviderUnavailable,
)
from worker.music.writer import TemplateLyricsWriter, detect_mood, extract_subject

__all__ = [
    "ENGLISH",
    "Language",
    "LyricBrief",
    "LyricFit",
    "LyricIssue",
    "LyricsReview",
    "LyricsWriter",
    "MusicGenerationProvider",
    "MusicRequest",
    "MusicTake",
    "ProviderGenerationError",
    "ProviderProgress",
    "ProviderUnavailable",
    "Section",
    "SongPlan",
    "TemplateLyricsWriter",
    "UnknownLanguage",
    "UnsupportedLyricLanguage",
    "check_lyric_fit",
    "detect_genre",
    "detect_mood",
    "extract_subject",
    "line_budget",
    "lines_rhyme",
    "offered",
    "parse_sections",
    "plan_song",
    "polish_lyrics",
    "resolve_language",
    "review_lyrics",
    "rhyme_key",
    "salient_details",
    "write_lyrics",
]
