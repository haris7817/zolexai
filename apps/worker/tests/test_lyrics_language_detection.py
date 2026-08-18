"""Does the sheet that came back match the language that was asked for?

This is the guard that stands between "a model answered in English" and "a
customer's Spanish song is in English". It is worth testing carefully in both
directions, because the two ways it can be wrong have very different costs:

  * a MISS (English accepted as Spanish) ships the wrong song, silently;
  * a FALSE ALARM (real Spanish rejected) burns the retry budget and then falls
    through to a fallback that may not speak Spanish either, turning a correct
    result into a failed job.

So every language the product offers is checked against its own text and
against text in a language it is not.
"""

from __future__ import annotations

import pytest

from worker.music.detect import written_in
from worker.music.language import offered

# One short, realistic lyric sheet per language — same song, same imagery, so
# the only variable between them is the language itself.
SHEETS: dict[str, str] = {
    "en": """[verse]
Under the lights of a warm summer night
Your hand in mine and the sea is calling
I don't know if this is love but it feels right
[chorus]
And we dance until the sun comes up
With our hearts down in the golden sand""",
    "es": """[verse]
Bajo las luces de la noche de verano
Tu mano en la mia y el mar nos llama
No se si es amor pero se siente asi
[chorus]
Y bailamos hasta que salga el sol
Con el corazon en la arena dorada""",
    "fr": """[verse]
Sous les lumieres de la nuit d'ete
Ta main dans la mienne et la mer nous appelle
Je ne sais pas si c'est l'amour mais je le sens
[chorus]
Et nous dansons jusqu'au lever du soleil
Avec le coeur dans le sable dore""",
    "pt": """[verse]
Sob as luzes da noite de verao
Tua mao na minha e o mar nos chama
Nao sei se e amor mas parece que sim
[chorus]
E dancamos ate o sol nascer
Com o coracao na areia dourada""",
    "de": """[verse]
Unter den Lichtern der warmen Sommernacht
Deine Hand in meiner und das Meer ruft uns
Ich weiss nicht ob es Liebe ist aber es fuehlt sich so an
[chorus]
Und wir tanzen bis die Sonne aufgeht
Mit dem Herzen in dem goldenen Sand""",
    "it": """[verse]
Sotto le luci della notte d'estate
La tua mano nella mia e il mare ci chiama
Non so se e amore ma sembra cosi
[chorus]
E balliamo fino al sorgere del sole
Con il cuore nella sabbia dorata""",
    "tr": """[verse]
Yaz gecesinin isiklari altinda
Elin elimde ve deniz bizi cagiriyor
Bu ask mi bilmiyorum ama oyle geliyor
[chorus]
Ve gunes dogana kadar dans ediyoruz
Kalbimiz altin kumun icinde""",
    "ar": """[verse]
تحت أضواء ليلة الصيف الدافئة
يدك في يدي والبحر ينادينا
لا أعرف إن كان هذا حبا لكنه يبدو كذلك
[chorus]
ونرقص حتى تشرق الشمس
وقلوبنا في الرمال الذهبية""",
    "ur": """[verse]
گرمیوں کی رات کی روشنی کے نیچے
تمہارا ہاتھ میرے ہاتھ میں اور سمندر بلا رہا ہے
[chorus]
اور ہم ناچتے ہیں جب تک سورج نہ نکلے""",
    "hi": """[verse]
गर्मियों की रात की रोशनी के नीचे
तुम्हारा हाथ मेरे हाथ में और समंदर बुला रहा है
[chorus]
हम नाचते हैं जब तक सूरज न निकले
सुनहरी रेत में हमारा दिल""",
    "ru": """[verse]
Под огнями теплой летней ночи
Твоя рука в моей и море зовет нас
[chorus]
Мы танцуем пока не взойдет солнце
С сердцем в золотом песке""",
    "ja": """[verse]
夏の夜のあたたかい光の下で
きみの手をとって海が呼んでいる
[chorus]
太陽がのぼるまで踊ろう
金色の砂に心をおいて""",
    "ko": """[verse]
따뜻한 여름 밤의 불빛 아래에서
네 손을 잡고 바다가 부른다
[chorus]
해가 뜰 때까지 우리는 춤을 춘다
황금빛 모래 위에 마음을 두고""",
    "zh": """[verse]
在温暖夏夜的灯光下
你的手在我手中大海在呼唤
[chorus]
我们跳舞直到太阳升起
把心留在金色的沙滩上""",
}


def test_every_offered_language_has_a_sample_here() -> None:
    """A language the product offers and this suite never checks is a language
    whose wrong-language guard has never been run."""
    assert {language.code for language in offered()} == set(SHEETS)


@pytest.mark.parametrize("code", sorted(SHEETS))
def test_a_sheet_is_recognised_as_its_own_language(code: str) -> None:
    """No false alarms. Rejecting a correct sheet costs a retry and then, for
    any language the local bank cannot write, the whole job."""
    assert written_in(SHEETS[code], code) is not False


@pytest.mark.parametrize("code", sorted(set(SHEETS) - {"en"}))
def test_english_is_never_accepted_as_another_language(code: str) -> None:
    """The failure this module was written for, one language at a time.

    A model that answers a Spanish request in English produces a sheet that is
    well-formed, rhymes and fits the budget — indistinguishable from a correct
    one at every layer below the writer.
    """
    assert written_in(SHEETS["en"], code) is False


@pytest.mark.parametrize(
    ("text_code", "claimed"),
    [
        ("es", "en"),
        ("fr", "en"),
        ("de", "en"),
        ("ja", "zh"),
        ("zh", "ja"),
        ("ru", "en"),
        ("ar", "hi"),
        ("hi", "ar"),
        ("ko", "ja"),
    ],
)
def test_a_sheet_is_not_accepted_as_a_language_it_is_not(
    text_code: str, claimed: str
) -> None:
    """Including the pairs that share a writing system.

    Japanese and Chinese share Han characters, so kana are what separate them;
    Arabic and Devanagari share nothing and must not be confused at all.
    """
    assert written_in(SHEETS[text_code], claimed) is False


def test_section_tags_are_not_counted_as_english() -> None:
    """`[verse]` and `[chorus]` are markup the music model reads, and they are
    English in every language's sheet. Counting them as evidence would push
    every short non-English sheet towards "this is English"."""
    tags_only_spanish = "[verse]\n" + "\n".join(SHEETS["es"].splitlines()[1:3])
    assert written_in(tags_only_spanish, "es") is not False


def test_too_little_text_returns_no_verdict_rather_than_a_guess() -> None:
    """None means "cannot tell", and the caller accepts on None. Guessing from
    three words is how a correct sheet gets thrown away."""
    assert written_in("[verse]\nla mar", "es") is None
    assert written_in("", "es") is None
    assert written_in("[chorus]\n[verse]", "es") is None


def test_a_latin_sheet_is_rejected_for_a_language_that_is_not_written_in_latin() -> None:
    """A romanised answer is still the wrong answer: the music model is handed
    an ISO code that tells it which phonetics to sing, and romanised Hindi sung
    as Hindi is not what the customer picked."""
    assert written_in(SHEETS["en"], "hi") is False
    assert written_in(SHEETS["es"], "ja") is False


def test_a_borrowed_english_line_does_not_disqualify_a_sheet() -> None:
    """Real lyrics in every language carry the odd English hook or name. The
    threshold is a share, not a prohibition."""
    with_hook = SHEETS["es"] + "\nOh baby tonight"
    assert written_in(with_hook, "es") is not False

    with_name = SHEETS["ja"] + "\nTokyo Sunset"
    assert written_in(with_name, "ja") is not False


def test_accents_lost_in_transit_do_not_change_the_verdict() -> None:
    """A model writing "esta" for "está", or a sheet that lost its diacritics
    somewhere, is still Spanish and must not be rejected as English."""
    accented = """[verse]
Bajo las luces de la noche de verano
Tú mano en la mía y el mar nos está llamando
[chorus]
Y bailamos hasta que salga el sol
Con el corazón en la arena dorada"""
    assert written_in(accented, "es") is not False
