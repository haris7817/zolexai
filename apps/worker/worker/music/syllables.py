"""Syllable estimates and filler detection, per script.

The lyrics workflow (`worker/music/workflow.py`) budgets words by time — a
verse of thirty seconds at a pop delivery holds about ninety syllables — and
measures a sheet against that budget before any audio is made. Both halves
need a syllable count that is *consistent* across the offered languages more
than one that is exact for any of them: the count decides whether a section
is thin, and a thin section is thin whichever way the individual words are
counted, as long as every line is counted the same way.

So this is a set of per-script estimators, not a pronunciation dictionary.
Latin and Cyrillic scripts count vowel groups; abugidas count consonant
clusters and free vowels; Arabic script has no written short vowels so it
counts letters against a measured average; CJK counts characters, which for
kana and hangul IS the syllable and for Han is close enough. The confidence
of each is recorded so a report never claims more than was measured.

Filler is the other thing measured here: the client's workflow allows at most
one tenth of a song to be "oh", "yeah", humming and sound effects, and a
writer told that will still sometimes pad a chorus with them.
"""

from __future__ import annotations

import re
import unicodedata

# ── Script detection ─────────────────────────────────────────────────────

_LATIN_VOWELS = "aeiouyàáâãäåæèéêëìíîïòóôõöøùúûüýÿœ"
_CYRILLIC_VOWELS = "аеёиоуыэюяіїєґ"

#: Devanagari: independent vowels, consonants, vowel signs, virama.
_DEVA_INDEPENDENT_VOWEL = re.compile(r"[ऄ-औॠॡॲ-ॷ]")
_DEVA_CONSONANT = re.compile(r"[क-हक़-य़ॹ-ॿ]")
_DEVA_VIRAMA = "्"

_ARABIC_LETTER = re.compile(r"[ء-يٱ-ۓۺ-ۼ]")
#: Letters per syllable in unpointed Arabic/Urdu text — a measured average
#: rather than a rule, since the short vowels are simply not written.
_ARABIC_LETTERS_PER_SYLLABLE = 2.4

_HANGUL = re.compile(r"[가-힣]")
_KANA = re.compile(r"[ぁ-ゖァ-ヺ]")
_SMALL_KANA = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")
_HAN = re.compile(r"[一-鿿㐀-䶿]")

_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def _latin_word(word: str, code: str) -> int:
    lowered = word.lower()
    folded = _strip_accents(lowered)
    groups = re.findall(r"[aeiouy]+", folded)
    count = len(groups)
    if count == 0:
        return 1
    if code == "en":
        # Silent final e ("time", "love"), but not when it is the only vowel
        # ("the") and not "-le" after a consonant ("table"), which sounds.
        if (
            folded.endswith("e")
            and count > 1
            and not folded.endswith("le")
            and not folded.endswith("ee")
        ):
            count -= 1
        if folded.endswith("ed") and count > 1 and not re.search(r"[td]ed$", folded):
            count -= 1
    elif code == "fr":
        # A final unstressed e (or -es, -ent) is silent in sung French
        # more often than not.
        if count > 1 and re.search(r"(e|es|ent)$", folded):
            count -= 1
    elif code in {"es", "pt", "it"}:
        # Diphthongs with a weak vowel count once; the vowel-group regex
        # already merges them. A "qu"/"gu" before e/i carries no vowel.
        count -= len(re.findall(r"[qg]u[ei]", folded))
        count = max(1, count)
    return max(1, count)


def _cyrillic_word(word: str) -> int:
    return max(1, sum(1 for ch in word.lower() if ch in _CYRILLIC_VOWELS))


def _devanagari(text: str) -> int:
    count = len(_DEVA_INDEPENDENT_VOWEL.findall(text))
    consonants = list(_DEVA_CONSONANT.finditer(text))
    for match in consonants:
        following = text[match.end() : match.end() + 1]
        # A consonant killed by a virama joins the next one; it carries no
        # vowel of its own.
        if following != _DEVA_VIRAMA:
            count += 1
    return count


def _arabic(text: str) -> int:
    letters = len(_ARABIC_LETTER.findall(text))
    return max(1, round(letters / _ARABIC_LETTERS_PER_SYLLABLE)) if letters else 0


def _kana(text: str) -> int:
    return sum(1 for ch in _KANA.findall(text) if ch not in _SMALL_KANA)


def syllables(text: str, code: str = "en") -> int:
    """Estimated syllable count of `text` in language `code`.

    Structure tags, punctuation and digits are ignored. Never negative; a
    line with letters in it is never zero.
    """
    body = _lyric_body(text)
    if not body.strip():
        return 0

    total = 0
    # CJK and abugidas are counted on the raw text; everything else per word.
    total += _kana(body)
    total += len(_HAN.findall(body))
    total += len(_HANGUL.findall(body))
    total += _devanagari(body)
    total += _arabic(body)

    for word in _WORD.findall(body):
        if _KANA.search(word) or _HAN.search(word) or _HANGUL.search(word):
            continue
        if _DEVA_CONSONANT.search(word) or _DEVA_INDEPENDENT_VOWEL.search(word):
            continue
        if _ARABIC_LETTER.search(word):
            continue
        if any(ch in _CYRILLIC_VOWELS for ch in word.lower()):
            total += _cyrillic_word(word)
        elif re.search(r"[A-Za-zÀ-ɏ]", word):
            total += _latin_word(word, code)
        else:
            total += 1
    return total


_TAG = re.compile(r"^\s*\[[^\]]+\]\s*$")


def _lyric_body(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _TAG.match(line))


def confidence(code: str) -> str:
    """How much to trust the estimate: "high" for scripts with written vowels,
    "medium" for kana/hangul/Han counts, "low" for unpointed Arabic script."""
    if code in {"ar", "ur"}:
        return "low"
    if code in {"ja", "ko", "zh", "hi"}:
        return "medium"
    return "high"


# ── Filler ───────────────────────────────────────────────────────────────

#: Tokens that are sung sound rather than words, across the offered
#: languages. A line made mostly of these is padding, whatever it is called.
_FILLER_TOKENS = frozenset(
    {
        "oh", "ooh", "oooh", "ohh", "ah", "aah", "ahh", "uh", "uhh", "mm", "mmm", "hmm",
        "hm", "la", "na", "da", "ba", "sha", "doo", "dum", "yeah", "yea", "yah", "hey",
        "ho", "whoa", "woah", "woo", "eh", "ay", "aye", "ey", "ya", "yo", "uh-huh",
        "oh-oh", "na-na", "la-la", "ooh-ooh", "eeh", "ohhh",
        # es / pt / it / fr
        "ey", "ea", "ae", "olé", "ole", "eh", "ah", "ohé", "lá", "lala", "nana",
        # ur / hi (romanised) and Arabic script
        "haan", "haaye", "haye", "aha", "yaara", "ya", "ah", "aa", "ha",
        "آه", "اوه", "ہاں", "हाँ", "आह", "ओह", "ला", "ना",
        # ru / de / tr
        "эй", "ах", "ох", "ла", "на", "hey", "ja", "ay", "of", "ah", "oy",
        # ja / ko / zh
        "ラ", "ナ", "ああ", "おお", "うう", "랄라", "라라", "나나", "오", "아", "啊", "哦", "噢",
    }
)

_PARENTHETICAL_FILLER = re.compile(
    r"^\s*[\(\[]\s*(humming|hum|instrumental|ad[- ]?libs?|vocalizing|vocalising|"
    r"scat|whistling|breath|break|solo|beat|drop|silence)\b[^\)\]]*[\)\]]\s*$",
    re.IGNORECASE,
)


def _tokens(line: str) -> list[str]:
    return [
        token.lower().strip("'’")
        for token in re.findall(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", line, re.UNICODE)
    ]


def filler_share(line: str) -> float:
    """Fraction of a line's tokens that are sung sound rather than words.

    A parenthetical stage direction — "(humming)", "[instrumental break]" —
    is entirely filler: nothing in it is sung as a word.
    """
    if _PARENTHETICAL_FILLER.match(line):
        return 1.0
    tokens = _tokens(line)
    if not tokens:
        return 0.0
    filler = sum(1 for token in tokens if token in _FILLER_TOKENS or _is_stretched(token))
    return filler / len(tokens)


def _is_stretched(token: str) -> bool:
    """"ooooh", "yeahhh", "lalala" — a filler token elongated or repeated."""
    collapsed = re.sub(r"(.)\1+", r"\1", token)
    if collapsed in _FILLER_TOKENS:
        return True
    # "lalala", "nanana", "dadada": one filler syllable repeated.
    for unit in ("la", "na", "da", "ba", "sha", "oh", "ah", "ла", "на"):
        if token == unit * (len(token) // len(unit)) and len(token) >= 2 * len(unit):
            return True
    return False


#: A line is filler when at least this share of it is non-words. Set so that
#: "oh baby I love you" (one filler token in five) is a lyric and "oh oh oh
#: yeah" is not.
FILLER_LINE_THRESHOLD = 0.6


def is_filler_line(line: str) -> bool:
    return filler_share(line) >= FILLER_LINE_THRESHOLD


__all__ = [
    "FILLER_LINE_THRESHOLD",
    "confidence",
    "filler_share",
    "is_filler_line",
    "syllables",
]
