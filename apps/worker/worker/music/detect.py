"""Is this text actually written in the language we asked for?

## Why this exists

A language model asked for Spanish lyrics will sometimes return English ones.
It does not announce this — it returns a confident, well-formed sheet, and
every layer below here treats a sheet as a sheet. The result is a song sung
with Spanish phonetics over English words, which is the exact failure the
language selector was built to end, arriving through a different door.

So the writer checks its own output before handing it on. This module is that
check, and it is deliberately shallow: no model, no downloaded corpus, no
dependency. It answers one narrow question — *does this look like the language
that was requested* — and it is tuned to be **precise rather than sensitive**.

## Why precision over sensitivity

A false negative costs a wasted retry. A false positive ships the wrong
language. But a false *alarm* — rejecting genuinely Spanish lyrics — is worse
than either in one specific way: it burns the retry budget and then falls
through to a fallback that may not speak Spanish at all, turning a good result
into a failed job. So every rule below refuses only on clear evidence and
returns `None` ("cannot tell") when the signal is thin.

## The two signals

**Script** is decisive and nearly free. Arabic, Devanagari, Hangul, kana, Han
and Cyrillic do not appear in English by accident, and English does not appear
in Korean by accident. For the seven offered languages that use a distinctive
script this settles the question on its own.

**Function words** handle the Latin-script languages, where script tells us
nothing. Articles, pronouns and prepositions are the highest-frequency words in
any text and the least susceptible to a topic word wandering in — a song about
"amor" proves nothing, a song full of "que / de / la / no / me" is Spanish.
Related languages (Spanish/Portuguese/Italian) share many of these, which is
why a rival language must beat the requested one by a clear margin before this
module will call it wrong.
"""

from __future__ import annotations

import re
import unicodedata

from worker.core.logging import get_logger

logger = get_logger(__name__)

#: Section tags are English by convention — `[Verse]`, `[Chorus]` — in every
#: language's sheet, because they are markup the music model reads rather than
#: words anybody sings. Counting them as evidence would push every short
#: non-English sheet towards "this is English".
_TAG_LINE = re.compile(r"^\s*\[[^\]]*\]\s*$")

#: What counts as a letter for the script share. Digits, punctuation and
#: whitespace are script-neutral and would only dilute the measurement.
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _lyric_text(text: str) -> str:
    """The sung words only, with structure markup removed."""
    return "\n".join(line for line in text.splitlines() if not _TAG_LINE.match(line))


# ── Script ───────────────────────────────────────────────────────────────

#: Unicode script name prefixes, as `unicodedata.name` reports them. Matching
#: on the name rather than hard-coded codepoint ranges is what keeps this
#: correct for the extension blocks each script has picked up over time —
#: Arabic Presentation Forms, Devanagari Extended, CJK Extension B and so on,
#: none of which are in the ranges people write from memory.
_SCRIPT_PREFIXES: dict[str, tuple[str, ...]] = {
    "latin": ("LATIN",),
    "arabic": ("ARABIC",),
    "devanagari": ("DEVANAGARI",),
    "cyrillic": ("CYRILLIC",),
    "hangul": ("HANGUL",),
    "kana": ("HIRAGANA", "KATAKANA"),
    "han": ("CJK", "IDEOGRAPHIC"),
}

#: Which script each offered language must actually be written in.
#:
#: Japanese is the interesting one: it uses Han characters too, so requiring
#: only Han would accept Chinese as Japanese. Kana are the discriminator, and
#: real Japanese lyrics are full of them, so `ja` is checked on kana alone.
#: Urdu and Arabic share a script and are not separated here — that distinction
#: needs vocabulary, and the two are never confused for English, which is the
#: failure this guards against.
_REQUIRED_SCRIPT: dict[str, str] = {
    "ar": "arabic",
    "ur": "arabic",
    "hi": "devanagari",
    "ru": "cyrillic",
    "ko": "hangul",
    "ja": "kana",
    "zh": "han",
}

#: Minimum share of letters that must belong to the required script.
#:
#: Not 1.0, and not close to it: real lyrics in every one of these languages
#: carry Latin fragments — a name, a brand, an English hook line, a romanised
#: word. A third is comfortably above what incidental borrowing produces and
#: comfortably below what a genuinely-wrong-language sheet would score.
_SCRIPT_SHARE = 0.33

#: How much non-Latin script disqualifies a language that should be Latin.
#: Higher than `_SCRIPT_SHARE` because the asymmetry is real — a Spanish sheet
#: containing one Japanese loan word is still Spanish, while a sheet that is
#: half kana is not.
_FOREIGN_SCRIPT_SHARE = 0.5


def _script_shares(text: str) -> dict[str, float]:
    """What fraction of the letters in `text` belongs to each script."""
    counts: dict[str, int] = {}
    total = 0
    for match in _WORD.finditer(text):
        for char in match.group(0):
            try:
                name = unicodedata.name(char)
            except ValueError:  # unnamed codepoint — no script to attribute
                continue
            total += 1
            for script, prefixes in _SCRIPT_PREFIXES.items():
                if name.startswith(prefixes):
                    counts[script] = counts.get(script, 0) + 1
                    break
    if not total:
        return {}
    return {script: count / total for script, count in counts.items()}


# ── Function words ───────────────────────────────────────────────────────

#: The highest-frequency function words of each Latin-script language the
#: product offers. Articles, pronouns, prepositions, conjunctions, copulas.
#:
#: Chosen for frequency, not for uniqueness — overlap between Spanish,
#: Portuguese and Italian is expected and is handled by the margin rule in
#: `_by_function_words` rather than by trying to curate it away. Words that
#: would be *lyric* vocabulary (amor, noche, coeur) are deliberately absent:
#: they say what a song is about, not what language it is in.
_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        """the and you your my me is are was were to of in on at with that this
        for but not all we they it be have has had will would can could when
        what where there here from about into like just now then than out
        over down up so if isn no yes dont im youre its ill youll cant"""
        .split()
    ),
    "es": frozenset(
        """el la los las un una unos unas y que de del en con por para no se me
        te nos le lo su sus mi mis tu tus es son era eran esta estan muy mas
        como cuando pero si ya todo toda todos al ni desde hasta sin sobre
        hay soy eres somos porque quien donde"""
        .split()
    ),
    "pt": frozenset(
        """o a os as um uma uns umas e que de do da dos das em no na nos nas com
        por para nao se me te lhe meu minha seu sua teu tua eh sao era eram
        esta estao mais como quando mas ja tudo todo todos ao nem desde
        ate sem sobre ha sou somos porque quem onde voce voces eu"""
        .split()
    ),
    "fr": frozenset(
        """le la les un une des du de et que qui en dans avec pour par ne pas je
        tu il elle nous vous ils elles mon ma mes ton ta tes son sa ses est
        sont etait etaient sur ce cette ces mais plus tout toute tous au aux
        si comme quand encore sans sous chez tres ou"""
        .split()
    ),
    "it": frozenset(
        """il lo la i gli le un uno una e che di da in con per non si mi ti ci vi
        ne mio mia tuo tua suo sua sono era erano piu come quando ma gia
        tutto tutta tutti al del dal nel sul io tu lui lei noi voi se anche
        solo senza dove perche"""
        .split()
    ),
    "de": frozenset(
        """der die das den dem des ein eine einen einem einer und ist sind war
        waren ich du er sie es wir ihr mit fur von zu auf in an nicht aber
        auch noch wie wenn mein dein sein sich mich dich uns hat haben hatte
        wird werden nur schon immer dass da so"""
        .split()
    ),
    "tr": frozenset(
        """bir ve bu su o ben sen biz siz icin ile ama gibi daha cok her ne var
        yok degil kadar sonra once sey ya da ki de den dan ise hic boyle
        simdi bile gore kendi"""
        .split()
    ),
}

#: Tokens, with an internal apostrophe kept so that French elisions
#: ("l'amour") and English contractions survive as one word rather than
#: fragmenting into evidence for neither language.
_TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)

#: Accents are stripped before matching so that a model writing "esta" for
#: "está" — or a sheet that lost its diacritics in transit — still scores as
#: the language it is. The word lists above are written unaccented to match.
_COMBINING = re.compile(r"[̀-ͯ]")


def _fold(token: str) -> str:
    """Lowercase, unaccented, apostrophe-free — the form the lists are in."""
    stripped = _COMBINING.sub("", unicodedata.normalize("NFD", token.lower()))
    return stripped.replace("'", "").replace("’", "")


#: Minimum tokens before the function-word rule will say anything at all.
#: A two-line fragment does not carry enough of them to distinguish Spanish
#: from Portuguese, and guessing on that little evidence is how a correct sheet
#: gets thrown away.
_MINIMUM_TOKENS = 12

#: How far ahead a rival language must score before the requested one is
#: called wrong. A rival that merely ties is the ordinary state of affairs
#: between Spanish and Portuguese and means nothing.
_MARGIN = 2.0

#: The rival must also clear this share of all tokens, so that three stray
#: matches in a long sheet cannot outvote the language actually being written.
_RIVAL_FLOOR = 0.08


def _by_function_words(text: str, code: str) -> bool | None:
    """Latin-script verdict, or None when the evidence is too thin."""
    if code not in _FUNCTION_WORDS:
        return None

    tokens = [_fold(token) for token in _TOKEN.findall(text)]
    if len(tokens) < _MINIMUM_TOKENS:
        return None

    scores = {
        language: sum(1 for token in tokens if token in words) / len(tokens)
        for language, words in _FUNCTION_WORDS.items()
    }
    mine = scores[code]
    rival_code, rival = max(
        ((other, score) for other, score in scores.items() if other != code),
        key=lambda pair: pair[1],
    )

    if rival < _RIVAL_FLOOR:
        # Nothing scored convincingly — the text may be sparse in function
        # words (a chant, a list, a very short chorus). Not evidence of
        # anything, so it is not treated as such.
        return True if mine >= _RIVAL_FLOOR else None

    if mine <= 0.0 or rival >= mine * _MARGIN:
        logger.debug(
            "lyrics_language_mismatch",
            extra={
                "requested": code,
                "looks_like": rival_code,
                "requested_score": round(mine, 3),
                "rival_score": round(rival, 3),
            },
        )
        return False
    return True


# ── The one function callers use ─────────────────────────────────────────


def written_in(text: str, code: str) -> bool | None:
    """Whether `text` reads as language `code`.

    True  — consistent with the requested language.
    False — clearly some other language; the caller should retry or refuse.
    None  — not enough evidence either way; the caller should accept, because
            refusing on no evidence throws away work for nothing.
    """
    body = _lyric_text(text).strip()
    if not body:
        return None

    shares = _script_shares(body)
    if not shares:
        return None

    required = _REQUIRED_SCRIPT.get(code)
    if required is not None:
        return shares.get(required, 0.0) >= _SCRIPT_SHARE

    # A Latin-script language. Any heavy presence of another script is
    # disqualifying on its own, before a single word is looked at.
    foreign = sum(share for script, share in shares.items() if script != "latin")
    if foreign >= _FOREIGN_SCRIPT_SHARE:
        return False

    return _by_function_words(body, code)
