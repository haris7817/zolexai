"""Language-aware rhyme validation for a lyric sheet.

## What this decides

The client's workflow assigns every line to a rhyme group and requires every
group to pass "language-aware phonetic rhyme validation" — strictly by
default. `worker/music/lyrics.py` already has an English ending heuristic
(`rhyme_key`) used to *rate* a draft; this module turns that idea into a
verdict per group, per language, with a scheme the caller chose.

## What "phonetic" means here, honestly

There is no pronunciation dictionary for fourteen languages in this
repository and none is added. What exists per language is a rule for where a
word's rhyming part begins — the stressed vowel — and how the spelling after
it sounds:

  * **Spanish / Italian / Portuguese** are close to phonetic and have real
    stress rules (a word ending in a vowel, n or s stresses the penultimate
    syllable; a written accent overrides). The key is exact from the
    stressed vowel. Relaxed mode keeps only the vowels — the *asonante*
    rhyme those traditions actually use.
  * **French** drops the silent final letters before keying.
  * **English** reuses the spelling→sound rules from `lyrics.py`.
  * **German / Turkish / Russian** key on the last vowel group onwards; the
    stress is not derivable from spelling, so the result is a fair
    approximation and marked as such.
  * **Arabic / Urdu / Hindi** compare the final letter(s) — the written
    ending, which IS what those poetic traditions rhyme on (qafia) — but
    without vowel marks the confidence is low.
  * **Korean** decomposes the final syllable block; **Japanese** compares
    the final mora; **Chinese** compares the final character, which
    rhymes only when it happens to share a final — low confidence.

Every group result carries the method and the confidence. A report says "the
validator with these rules passed this group", never "this rhymes" — and the
job-level policy in `worker/music/workflow.py` only *fails* a song over rhyme
where the confidence is high enough for the verdict to mean something.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from worker.music.lyrics import rhyme_key as _english_key

#: The schemes a customer may ask for. "auto" is resolved by the blueprint.
SCHEMES: tuple[str, ...] = ("auto", "AABB", "ABAB", "AAAA")

#: Sections whose lines repeat by design; an identical final word inside one
#: of these is a refrain, not laziness.
_REFRAIN_TAGS = frozenset({"chorus", "hook", "drop", "refrain"})


@dataclass(frozen=True)
class RhymeGroup:
    section_index: int
    section: str
    label: str
    line_indexes: tuple[int, ...]
    """Global line indexes (over the whole sheet, tags excluded)."""
    endings: tuple[str, ...]
    keys: tuple[str, ...]
    passed: bool
    reason: str = ""


@dataclass(frozen=True)
class RhymeReport:
    groups: tuple[RhymeGroup, ...]
    method: str
    confidence: str
    """"high", "medium" or "low" — how much the verdicts are worth."""
    mode: str
    scheme: str

    @property
    def required(self) -> tuple[RhymeGroup, ...]:
        return tuple(group for group in self.groups if len(group.line_indexes) >= 2)

    @property
    def pass_rate(self) -> float:
        required = self.required
        if not required:
            return 1.0
        return sum(1 for group in required if group.passed) / len(required)

    @property
    def failing(self) -> tuple[RhymeGroup, ...]:
        return tuple(group for group in self.required if not group.passed)

    @property
    def failing_lines(self) -> tuple[int, ...]:
        found: list[int] = []
        for group in self.failing:
            for index in group.line_indexes:
                if index not in found:
                    found.append(index)
        return tuple(found)

    @property
    def passed(self) -> bool:
        return not self.failing

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "confidence": self.confidence,
            "mode": self.mode,
            "scheme": self.scheme,
            "pass_rate": round(self.pass_rate, 3),
            "groups": [
                {
                    "section": group.section,
                    "section_index": group.section_index,
                    "label": group.label,
                    "lines": list(group.line_indexes),
                    "endings": list(group.endings),
                    "keys": list(group.keys),
                    "required": len(group.line_indexes) >= 2,
                    "passed": group.passed,
                    "reason": group.reason,
                }
                for group in self.groups
            ],
        }


# ── Scheme → labels ──────────────────────────────────────────────────────


def scheme_labels(count: int, scheme: str) -> list[str]:
    """Rhyme-group label per line for a section of `count` lines.

    AABB pairs consecutive lines; ABAB alternates within each four; AAAA is
    one group. A trailing odd line under AABB gets a label of its own and is
    therefore not required to rhyme with anything — a five-line verse is
    not a defect, and neither is its last line.
    """
    scheme = (scheme or "AABB").upper()
    if scheme == "AAAA":
        return ["A"] * count
    labels: list[str] = []
    if scheme == "ABAB":
        for index in range(count):
            block, position = divmod(index, 4)
            labels.append(_letter(block * 2 + (position % 2)))
        return labels
    # AABB and anything unrecognised.
    for index in range(count):
        labels.append(_letter(index // 2))
    return labels


def _letter(number: int) -> str:
    text = ""
    number += 1
    while number:
        number, remainder = divmod(number - 1, 26)
        text = chr(ord("A") + remainder) + text
    return text


# ── Endings ──────────────────────────────────────────────────────────────

_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
_CJK = re.compile(r"[一-鿿㐀-䶿ぁ-ゖァ-ヺ가-힣]")


def final_word(line: str) -> str:
    """The last word of a line, or the last CJK character."""
    stripped = re.sub(r"[\(\[].*?[\)\]]", " ", line).strip()
    words = _WORD.findall(stripped)
    if words:
        return words[-1]
    cjk = _CJK.findall(stripped)
    return cjk[-1] if cjk else ""


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


_ROMANCE_VOWELS = "aeiouáéíóúàèìòùâêôãõäëïöü"
_STRONG = set("aeoáéóàèòâêôãõ")


def _romance_syllables(word: str) -> list[tuple[int, int]]:
    """(start, end) of each vowel nucleus; diphthongs of a weak vowel merge."""
    nuclei: list[tuple[int, int]] = []
    index = 0
    while index < len(word):
        if word[index] in _ROMANCE_VOWELS:
            start = index
            while index < len(word) and word[index] in _ROMANCE_VOWELS:
                index += 1
            run = word[start:index]
            # Two strong vowels are two syllables (hiatus); split the run.
            cut = start
            for offset in range(1, len(run)):
                if run[offset] in _STRONG and run[offset - 1] in _STRONG:
                    nuclei.append((cut, start + offset))
                    cut = start + offset
            nuclei.append((cut, index))
        else:
            index += 1
    return nuclei


def _romance_key(word: str, *, vowels_only: bool) -> str:
    lowered = word.lower().replace("y", "i")
    nuclei = _romance_syllables(lowered)
    if not nuclei:
        return _strip_accents(lowered)[-2:]
    accented = [n for n in nuclei if any(ch in "áéíóúàèìòùâêô" for ch in lowered[n[0] : n[1]])]
    if accented:
        stressed = accented[-1]
    elif len(nuclei) >= 2 and (lowered[-1] in _ROMANCE_VOWELS or lowered[-1] in "ns"):
        stressed = nuclei[-2]
    else:
        stressed = nuclei[-1]
    start, end = stressed
    nucleus = lowered[start:end]
    # Inside a diphthong the rhyme begins at the vowel that carries the
    # stress: the written accent if there is one, else the strong vowel.
    # A rising diphthong's glide ("ci-ÓN", "c-IE-lo") is part of the onset.
    offset = next((i for i, ch in enumerate(nucleus) if ch in "áéíóúàèìòùâêô"), None)
    if offset is None:
        offset = next((i for i, ch in enumerate(nucleus) if ch in _STRONG), 0)
    tail = _strip_accents(lowered[start + offset :])
    if vowels_only:
        return "".join(ch for ch in tail if ch in "aeiou")
    return tail


_FRENCH_SILENT = re.compile(r"(?:es|ent|e|s|t|x|d|p|z)$")


def _french_key(word: str, *, vowels_only: bool) -> str:
    lowered = word.lower()
    if len(lowered) > 2:
        lowered = _FRENCH_SILENT.sub("", lowered) or lowered
    folded = _strip_accents(lowered)
    folded = folded.replace("eau", "o").replace("au", "o").replace("ou", "u").replace("oi", "wa")
    folded = folded.replace("ai", "e").replace("ei", "e").replace("ph", "f")
    match = list(re.finditer(r"[aeiouy]+", folded))
    if not match:
        return folded[-2:]
    tail = folded[match[-1].start() :]
    if vowels_only:
        return "".join(ch for ch in tail if ch in "aeiouy")
    return tail


def _last_vowel_key(word: str, vowels: str, *, vowels_only: bool) -> str:
    lowered = word.lower()
    positions = [index for index, ch in enumerate(lowered) if ch in vowels]
    if not positions:
        return lowered[-2:]
    tail = lowered[positions[-1] :]
    if vowels_only:
        return "".join(ch for ch in tail if ch in vowels)
    return tail


def _english_key_for(word: str, *, vowels_only: bool) -> str:
    key = _english_key(word)
    if vowels_only:
        return "".join(ch for ch in key if ch in "aeiou")
    return key


_CYRILLIC_VOWELS = "аеёиоуыэюяіїє"
_GERMAN_VOWELS = "aeiouyäöü"
_TURKISH_VOWELS = "aeıioöuü"

_ARABIC_MARKS = re.compile(r"[ً-ٰٟۖ-ۭ]")


def _arabic_key(word: str, *, vowels_only: bool) -> str:
    bare = _ARABIC_MARKS.sub("", word)
    # A final taa marbuta and haa sound alike in verse; fold them.
    bare = bare.replace("ة", "ه")
    return bare[-1:] if vowels_only else bare[-2:]


_DEVA_NUKTA = "़"


def _devanagari_key(word: str, *, vowels_only: bool) -> str:
    bare = word.replace(_DEVA_NUKTA, "")
    return bare[-1:] if vowels_only else bare[-2:]


def _hangul_key(word: str, *, vowels_only: bool) -> str:
    last = word[-1]
    code = ord(last) - 0xAC00
    if code < 0 or code > 11171:
        return last
    initial, rest = divmod(code, 588)
    vowel, final = divmod(rest, 28)
    return f"v{vowel}" if vowels_only else f"v{vowel}f{final}"


_SMALL_KANA = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")


def _japanese_key(word: str, *, vowels_only: bool) -> str:
    kana = [ch for ch in word if "ぁ" <= ch <= "ヺ"]
    if not kana:
        return word[-1:]
    # A small kana belongs to the mora before it.
    moras: list[str] = []
    for ch in kana:
        if ch in _SMALL_KANA and moras:
            moras[-1] += ch
        else:
            moras.append(ch)
    return moras[-1] if vowels_only else "".join(moras[-2:])


def _chinese_key(word: str, *, vowels_only: bool) -> str:
    return word[-1:]


#: language → (keying function, method label, confidence)
_RULES: dict[str, tuple] = {
    "en": (_english_key_for, "english_spelling_sound_rules", "high"),
    "es": (_romance_key, "romance_stress_rules", "high"),
    "it": (_romance_key, "romance_stress_rules", "high"),
    "pt": (_romance_key, "romance_stress_rules", "high"),
    "fr": (_french_key, "french_silent_final_rules", "high"),
    "de": (lambda w, *, vowels_only: _last_vowel_key(w, _GERMAN_VOWELS, vowels_only=vowels_only),
           "last_vowel_group", "medium"),
    "tr": (lambda w, *, vowels_only: _last_vowel_key(w, _TURKISH_VOWELS, vowels_only=vowels_only),
           "last_vowel_group", "medium"),
    "ru": (lambda w, *, vowels_only: _last_vowel_key(w, _CYRILLIC_VOWELS, vowels_only=vowels_only),
           "last_vowel_group", "medium"),
    "ar": (_arabic_key, "written_ending_unpointed", "low"),
    "ur": (_arabic_key, "written_ending_unpointed", "low"),
    "hi": (_devanagari_key, "written_ending", "medium"),
    "ko": (_hangul_key, "final_block_jamo", "high"),
    "ja": (_japanese_key, "final_mora", "medium"),
    "zh": (_chinese_key, "final_character", "low"),
}


def method_for(code: str) -> tuple[str, str]:
    """(method label, confidence) the validator uses for `code`."""
    _, method, confidence = _RULES.get(code, (None, "last_vowel_group", "low"))
    return method, confidence


def rhyme_key_for(word: str, code: str, *, mode: str = "strict") -> str:
    """The comparable ending of `word` in language `code`."""
    if not word:
        return ""
    rule = _RULES.get(code)
    vowels_only = mode == "relaxed"
    if rule is None:
        return _last_vowel_key(_strip_accents(word), "aeiouy", vowels_only=vowels_only)
    return rule[0](word, vowels_only=vowels_only)


# ── Validation ───────────────────────────────────────────────────────────


def validate_rhymes(
    sections: list[tuple[str, list[str]]],
    *,
    code: str,
    scheme: str = "AABB",
    mode: str = "strict",
) -> RhymeReport:
    """Every rhyme group in the sheet, with a verdict each.

    `sections` is `parse_sections` output. Groups of one line are reported
    but never required. An identical final word fails a group outside a
    refrain section — repetition is not rhyme — and is allowed inside one.
    """
    method, confidence = method_for(code)
    mode = "relaxed" if mode == "relaxed" else "strict"
    resolved_scheme = scheme.upper() if scheme and scheme.lower() != "auto" else "AABB"

    groups: list[RhymeGroup] = []
    global_index = 0
    for section_index, (tag, lines) in enumerate(sections):
        labels = scheme_labels(len(lines), resolved_scheme)
        by_label: dict[str, list[int]] = {}
        for offset, label in enumerate(labels):
            by_label.setdefault(label, []).append(global_index + offset)
        for label, indexes in by_label.items():
            endings = tuple(final_word(lines[i - global_index]) for i in indexes)
            keys = tuple(rhyme_key_for(ending, code, mode=mode) for ending in endings)
            passed, reason = _verdict(endings, keys, refrain=tag.lower() in _REFRAIN_TAGS)
            groups.append(
                RhymeGroup(
                    section_index=section_index,
                    section=tag,
                    label=label,
                    line_indexes=tuple(indexes),
                    endings=endings,
                    keys=keys,
                    passed=passed,
                    reason=reason,
                )
            )
        global_index += len(lines)

    return RhymeReport(
        groups=tuple(groups),
        method=method,
        confidence=confidence,
        mode=mode,
        scheme=resolved_scheme,
    )


def _verdict(endings: tuple[str, ...], keys: tuple[str, ...], *, refrain: bool) -> tuple[bool, str]:
    if len(endings) < 2:
        return True, "single line; not required"
    if any(not ending for ending in endings):
        return False, "a line has no final word"
    if any(not key for key in keys):
        return False, "no comparable ending"
    lowered = [ending.lower() for ending in endings]
    if len(set(lowered)) < len(lowered) and not refrain:
        return False, f"repeated final word {lowered[0]!r} is not a rhyme"
    if len(set(keys)) == 1:
        return True, ""
    return False, "endings do not share a sound: " + ", ".join(
        f"{ending}→{key}" for ending, key in zip(endings, keys, strict=True)
    )


def repair_instructions(report: RhymeReport, sections: list[tuple[str, list[str]]]) -> list[str]:
    """One instruction per failing group, naming the lines and the requirement."""
    flat: list[str] = [line for _, lines in sections for line in lines]
    notes: list[str] = []
    for group in report.failing:
        quoted = "; ".join(f"line {index + 1}: {flat[index]!r}" for index in group.line_indexes)
        notes.append(
            f"rhyme: in the [{group.section}] these lines must end on the same sound "
            f"(group {group.label}) and do not — {quoted}. Rewrite ONLY the line endings "
            f"so they rhyme exactly; keep every other line unchanged."
        )
    return notes


__all__ = [
    "SCHEMES",
    "RhymeGroup",
    "RhymeReport",
    "final_word",
    "method_for",
    "repair_instructions",
    "rhyme_key_for",
    "scheme_labels",
    "validate_rhymes",
]
