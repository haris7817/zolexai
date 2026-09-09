"""Music Lyrics Workflow v2.0 (client specification, 9 Sep 2026).

Lyrics first, validated, then sung, then measured. The music model lives
behind `MusicGenerationProvider` and the measuring tools (Demucs, Whisper)
behind two injectable functions, so the whole workflow — blueprint, writing,
rhyme, gates, dry run, verification, retry, delivery — runs here with a
fake provider that writes real MP3s and fakes that say what was "heard".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import collect, make_job, needs_ffmpeg
from tests.test_music import FakeProvider
from worker.adapters.base import AdapterError
from worker.adapters.music import MusicAdapter
from worker.music import LyricBrief, plan_song
from worker.music.blueprint import build_blueprint
from worker.music.gates import preflight, shared_phrases
from worker.music.lyrics import parse_sections
from worker.music.report import RESULT_MAX_BYTES, customer_result
from worker.music.rhyme import rhyme_key_for, scheme_labels, validate_rhymes
from worker.music.syllables import filler_share, is_filler_line, syllables
from worker.music.timing import lay_out
from worker.music.transcribe import Transcript, Word
from worker.music.verify import align_lines, verify_song
from worker.music.writer import TemplateLyricsWriter


def music_job(workspace: Path, duration: str = "1m", **overrides):
    defaults = dict(
        workflow_id="music",
        prompt="an upbeat pop song about summer in Lahore, hopeful, female vocals",
        parameters={"duration": duration},
        inputs=[],
        execution={"runtime": "music"},
        output_content_type="audio/mpeg",
    )
    return make_job(workspace, **{**defaults, **overrides})


# ── Syllables and filler ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "code", "low", "high"),
    [
        ("under the lights of a warm summer night", "en", 9, 11),
        ("bailamos hasta que salga el sol", "es", 9, 11),
        ("je t'aime encore ce soir", "fr", 5, 8),
        ("मेरा दिल तेरे नाम", "hi", 5, 8),
        ("東京の夜に", "ja", 5, 6),
        ("사랑해 오늘밤", "ko", 6, 6),
    ],
)
def test_syllables_are_estimated_per_script(text: str, code: str, low: int, high: int) -> None:
    assert low <= syllables(text, code) <= high


def test_filler_is_sound_not_words() -> None:
    assert is_filler_line("oh oh oh yeah")
    assert is_filler_line("(humming)")
    assert is_filler_line("la la la la la")
    assert not is_filler_line("oh baby I love you tonight")
    assert filler_share("yeah we ride tonight") == pytest.approx(0.25)


# ── Rhyme ────────────────────────────────────────────────────────────────


def test_schemes_label_lines_as_the_client_describes() -> None:
    assert scheme_labels(4, "AABB") == ["A", "A", "B", "B"]
    assert scheme_labels(8, "ABAB") == ["A", "B", "A", "B", "C", "D", "C", "D"]
    assert scheme_labels(3, "AAAA") == ["A", "A", "A"]
    # A trailing odd line under AABB is its own group and is not required.
    assert scheme_labels(5, "AABB")[-1] == "C"


@pytest.mark.parametrize(
    ("first", "second", "code", "rhymes"),
    [
        ("night", "light", "en", True),
        ("night", "cat", "en", False),
        ("corazón", "canción", "es", True),
        ("cielo", "suelo", "es", True),
        ("arena", "llama", "es", False),
        ("amour", "toujours", "fr", True),
        ("사랑", "바람", "ko", False),
        ("사랑", "가랑", "ko", True),
    ],
)
def test_rhyme_keys_follow_the_languages_own_rules(first: str, second: str, code: str, rhymes: bool) -> None:
    assert (rhyme_key_for(first, code) == rhyme_key_for(second, code)) is rhymes


def test_a_repeated_word_is_not_a_rhyme_outside_the_chorus() -> None:
    sections = [("verse", ["we ride tonight", "we ride tonight"]), ("chorus", ["hold me tight", "hold me tight"])]
    report = validate_rhymes(sections, code="en", scheme="AABB", mode="strict")
    verse, chorus = report.groups
    assert verse.passed is False and "repeated" in verse.reason
    assert chorus.passed is True


def test_strict_mode_needs_the_whole_ending_and_relaxed_only_the_vowel() -> None:
    sections = [("verse", ["walking home alone", "the night is cold", "call me on the phone", "I'm getting old"])]
    strict = validate_rhymes(sections, code="en", scheme="ABAB", mode="strict")
    relaxed = validate_rhymes(sections, code="en", scheme="ABAB", mode="relaxed")
    assert strict.pass_rate == 1.0
    assert relaxed.pass_rate == 1.0
    assert strict.confidence == "high"


# ── Blueprint and timing ─────────────────────────────────────────────────


@pytest.mark.parametrize("seconds", [60, 120, 180, 300])
def test_the_blueprint_keeps_wordless_time_inside_the_tenth(seconds: int) -> None:
    plan = plan_song(seconds, genre="pop")
    blueprint = build_blueprint(plan, language="en")
    wordless = sum(s.seconds for s in blueprint.sections if not s.vocal_required)
    assert wordless <= seconds * 0.10 + 1e-6
    assert blueprint.sections[-1].end == pytest.approx(seconds)
    assert all(s.target_lines >= 2 for s in blueprint.vocal_sections)
    # The chorus is written once however often it occurs.
    assert [tag for tag, _ in blueprint.writer_targets()].count("chorus") == 1


def test_a_full_sheet_lays_out_to_ninety_percent_and_reuses_the_chorus() -> None:
    plan = plan_song(60, genre="pop")
    blueprint = build_blueprint(plan, language="en")
    sheet: list[tuple[str, list[str]]] = []
    for tag, lines in blueprint.writer_targets():
        sheet.append((tag, [f"{tag} line number {i} carries a whole sung phrase along" for i in range(lines)]))
    timed = lay_out(sheet, blueprint)
    assert timed.planned_coverage >= 0.89
    occurrences = [s for s in timed.sections if s.blueprint.kind == "chorus"]
    if len(occurrences) > 1:
        assert [l.text for l in occurrences[0].lines] == [l.text for l in occurrences[1].lines]
    lrc = timed.to_lrc()
    assert lrc.startswith("[00:0")
    assert "-->" in timed.to_srt()
    # Times are contiguous and end on the song's length.
    assert timed.lines[-1].end <= 60.0 + 1e-6


def test_two_short_lines_over_thirty_seconds_are_not_thirty_seconds_of_singing() -> None:
    plan = plan_song(120, genre="pop")
    blueprint = build_blueprint(plan, language="en")
    thin = [(tag, ["oh no", "let go"]) for tag, _ in blueprint.writer_targets()]
    timed = lay_out(thin, blueprint)
    assert timed.planned_coverage < 0.5


# ── Gates ────────────────────────────────────────────────────────────────


def test_copied_runs_of_words_are_caught() -> None:
    reference = ("we dance until the sun comes up tonight",)
    assert shared_phrases(["and we dance until the sun comes up"], reference) == ["we dance until the sun comes up"]
    assert shared_phrases(["we dance"], reference) == []


def test_a_customers_own_sheet_is_measured_but_never_refused_over_rhyme_or_density() -> None:
    plan = plan_song(60, genre="pop")
    blueprint = build_blueprint(plan, language="en")
    sections = parse_sections("[verse]\nhello there\nnothing rhymes\n[chorus]\nla la la la\nooh yeah")
    timed = lay_out(sections, blueprint)
    rhyme = validate_rhymes(sections, code="en", scheme="AABB")
    brief = LyricBrief(topic="a song", genre="pop", language="en")
    report = preflight(timed, rhyme, brief, coverage_target=0.9, max_filler_ratio=0.1, rhyme_mode="strict", supplied=True)
    assert report.passed
    assert {p.code for p in report.warnings} >= {"VOCAL_COVERAGE_BELOW_90", "RHYME_VALIDATION_FAILED"}


def test_a_generated_sheet_fails_the_same_gates() -> None:
    plan = plan_song(60, genre="pop")
    blueprint = build_blueprint(plan, language="en")
    sections = parse_sections("[verse]\nhello there\nnothing rhymes\n[chorus]\nla la la la\nooh yeah")
    timed = lay_out(sections, blueprint)
    rhyme = validate_rhymes(sections, code="en", scheme="AABB")
    brief = LyricBrief(topic="a song", genre="pop", language="en")
    report = preflight(timed, rhyme, brief, coverage_target=0.9, max_filler_ratio=0.1, rhyme_mode="strict")
    assert not report.passed
    assert "VOCAL_COVERAGE_BELOW_90" in {p.code for p in report.errors}


# ── Verification ─────────────────────────────────────────────────────────


def _transcript_for(lines: list[str], *, drop: set[int] = frozenset(), seconds_per_line: float = 4.0) -> Transcript:
    words: list[Word] = []
    clock = 0.5
    for index, line in enumerate(lines):
        if index in drop:
            clock += seconds_per_line
            continue
        tokens = line.split()
        step = seconds_per_line / max(1, len(tokens))
        for token in tokens:
            words.append(Word(text=token, start=clock, end=clock + step * 0.9, probability=0.9))
            clock += step
    return Transcript(language="en", language_probability=0.99, words=tuple(words), model="fake")


def test_alignment_finds_the_lines_that_were_sung_and_names_the_ones_that_were_not() -> None:
    plan = plan_song(60, genre="pop")
    blueprint = build_blueprint(plan, language="en")
    pool = [
        "walking down the river road at dawn", "every window glowing gold and warm",
        "carry me across the summer rain", "nothing here will ever feel the same",
        "hold the moment like a paper kite", "we were burning brighter than the night",
        "call my name across the crowded square", "I will find you anywhere",
        "sugar on the tongue and salt on skin", "let the morning light come pouring in",
        "count the stars we never got to name", "every one of them a tiny flame",
        "lanterns floating on the harbour tide", "keep the secret we could never hide",
        "dancing barefoot on the kitchen floor", "always wanting just a little more",
    ]
    counter = iter(range(1000))
    sheet = [(tag, [pool[next(counter) % len(pool)] for _ in range(lines)]) for tag, lines in blueprint.writer_targets()]
    timed = lay_out(sheet, blueprint)
    texts = [line.text for line in timed.lines]
    transcript = _transcript_for(texts, drop={2})
    aligned = align_lines(timed, transcript, code="en")
    statuses = [line.status for line in aligned]
    assert statuses.count("matched") >= len(texts) - 2
    assert aligned[2].status in {"missing", "substituted"}


@needs_ffmpeg
async def test_verification_reports_unmeasured_without_tools(tmp_path: Path) -> None:
    from worker.media import ffmpeg

    track = tmp_path / "t.mp3"
    await ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-t", "3", "-c:a", "libmp3lame", str(track)])
    plan = plan_song(60, genre="pop")
    blueprint = build_blueprint(plan, language="en")
    timed = lay_out([("verse", ["a line", "b line"])], blueprint)

    async def no_transcript(path, *, language):
        return None

    async def no_stem(path):
        return None

    report = await verify_song(
        track, timed, language="en", expected_seconds=3, duration_seconds=3.0,
        coverage_target=0.9, recall_threshold=0.6, transcribe_fn=no_transcript, vocal_activity_fn=no_stem,
    )
    assert report.passed
    assert not report.measured
    assert report.coverage_method == "unmeasured"


# ── The adapter, end to end ──────────────────────────────────────────────


@pytest.fixture
def heard(monkeypatch):
    """Controls what the verifier 'hears': a list of sung fractions, one per
    take, and whether the transcript contains the lines."""
    state = {"coverages": [0.95], "calls": 0, "recall": True}

    async def fake_stem(path):
        index = min(state["calls"], len(state["coverages"]) - 1)
        state["calls"] += 1
        fraction = state["coverages"][index]
        return [(0.0, 60.0 * fraction)]

    async def fake_transcribe(path, *, language):
        return None if not state["recall"] else None

    monkeypatch.setattr("worker.music.verify.vocal_activity", fake_stem)
    monkeypatch.setattr("worker.music.verify.transcribe", fake_transcribe)
    return state


@needs_ffmpeg
async def test_the_workflow_writes_validates_sings_verifies_and_reports(workspace: Path, heard) -> None:
    provider = FakeProvider()
    result, _ = await collect(music_job(workspace, "1m"), MusicAdapter(provider, writer=TemplateLyricsWriter()))

    assert result.kind == "audio"
    assert result.report is not None
    report = result.report
    assert report["workflow"] == "music-lyrics/2.0"
    assert report["lyrics_source"] == "generated"
    assert report["planned_vocal_coverage"] >= 0.89
    assert report["measured_vocal_coverage"] == pytest.approx(0.95)
    assert report["coverage_method"] == "stem"
    assert "[00:" in report["lyrics_lrc"]
    assert report["retries"] == 0
    assert len(json.dumps(report).encode()) <= RESULT_MAX_BYTES

    # The sheet the model got is the expanded timeline, every chorus in place.
    sent = provider.requests[0].lyrics or ""
    assert sent.count("[chorus]") >= 1
    assert provider.requests[0].prompt.startswith("an upbeat pop song")
    assert "vocals begin" in provider.requests[0].prompt

    work = workspace / "lyrics"
    for name in ("lyrics.txt", "lyrics.json", "lyrics.lrc", "lyrics.srt", "blueprint.json",
                 "rhyme-validation.json", "preflight.json", "lyrics-report.json"):
        assert (work / name).exists(), name
    written = json.loads((work / "lyrics-report.json").read_text(encoding="utf-8"))
    assert written["status"] == "completed"
    assert written["request_sha256"]


@needs_ffmpeg
async def test_a_dry_run_validates_everything_and_makes_no_audio(workspace: Path, heard) -> None:
    provider = FakeProvider()
    job = music_job(workspace, "1m", parameters={"duration": "1m", "dry_run": True})
    with pytest.raises(AdapterError) as raised:
        await collect(job, MusicAdapter(provider, writer=TemplateLyricsWriter()))
    assert raised.value.retriable is False
    assert "Dry run complete" in raised.value.user_message
    assert provider.requests == []
    report = json.loads((workspace / "lyrics" / "lyrics-report.json").read_text(encoding="utf-8"))
    assert report["status"] == "planned"
    assert report["planned_vocal_coverage"] >= 0.89
    assert report["rhyme"]["pass_rate"] == 1.0


@needs_ffmpeg
async def test_a_take_with_too_little_singing_is_rerecorded_with_a_new_seed(workspace: Path, heard) -> None:
    heard["coverages"] = [0.5, 0.95]
    provider = FakeProvider()
    result, _ = await collect(music_job(workspace, "1m"), MusicAdapter(provider, writer=TemplateLyricsWriter()))
    assert len(provider.requests) == 2
    assert provider.requests[0].seed != provider.requests[1].seed
    assert provider.requests[0].lyrics == provider.requests[1].lyrics
    assert "first two seconds" in provider.requests[1].prompt
    assert result.report["retries"] == 1
    assert result.report["measured_vocal_coverage"] == pytest.approx(0.95)


@needs_ffmpeg
async def test_a_song_that_never_reaches_ninety_percent_fails_with_the_workflows_code(workspace: Path, heard) -> None:
    heard["coverages"] = [0.5]
    provider = FakeProvider()
    job = music_job(workspace, "1m", execution={"runtime": "music", "music_verify_max_retries": 1})
    with pytest.raises(AdapterError) as raised:
        await collect(job, MusicAdapter(provider, writer=TemplateLyricsWriter()))
    assert "[VOCAL_COVERAGE_BELOW_90]" in raised.value.internal_detail
    assert len(provider.requests) == 2
    assert raised.value.retriable is True


@needs_ffmpeg
async def test_deliver_best_ships_the_best_take_with_its_numbers(workspace: Path, heard) -> None:
    heard["coverages"] = [0.5, 0.7]
    provider = FakeProvider()
    job = music_job(
        workspace, "1m",
        execution={"runtime": "music", "music_verify_max_retries": 1, "music_verify_policy": "deliver_best"},
    )
    result, _ = await collect(job, MusicAdapter(provider, writer=TemplateLyricsWriter()))
    assert result.report["measured_vocal_coverage"] == pytest.approx(0.7)
    assert any("70%" in w for w in result.report["warnings"])
    assert result.path.name == "output-take2.mp3"


@needs_ffmpeg
async def test_a_customers_own_lyrics_are_sung_as_written_however_thin(workspace: Path, heard) -> None:
    provider = FakeProvider()
    own = "[verse]\nJust these words\nnothing more\n[chorus]\nsing it back to me"
    job = music_job(workspace, "1m", parameters={"duration": "1m", "lyrics": own})
    result, _ = await collect(job, MusicAdapter(provider, writer=TemplateLyricsWriter()))
    assert provider.requests[0].lyrics == own
    assert result.report["lyrics_source"] == "customer"
    assert result.report["lyrics"] == own
    assert any("coverage" in w for w in result.report["warnings"])


@needs_ffmpeg
async def test_a_reference_link_on_an_unknown_host_fails_before_any_work(workspace: Path, heard) -> None:
    provider = FakeProvider()
    job = music_job(
        workspace, "1m",
        parameters={"duration": "1m", "reference_audio_url": "https://drive.google.com/file/d/abc"},
    )
    with pytest.raises(AdapterError) as raised:
        await collect(job, MusicAdapter(provider, writer=TemplateLyricsWriter()))
    assert "[REFERENCE_UNAVAILABLE]" in raised.value.internal_detail
    assert raised.value.retriable is False
    assert provider.requests == []


@needs_ffmpeg
async def test_the_older_path_is_one_setting_away(workspace: Path, heard) -> None:
    provider = FakeProvider()
    job = music_job(workspace, "1m", execution={"runtime": "music", "music_lyrics_workflow": "v1"})
    result, _ = await collect(job, MusicAdapter(provider, writer=TemplateLyricsWriter()))
    assert result.report is None
    assert provider.requests[0].prompt == job.prompt


def test_the_customer_result_stays_within_the_apis_bound() -> None:
    plan = plan_song(300, genre="pop")
    blueprint = build_blueprint(plan, language="en")
    sheet = [(tag, ["a very long line of lyrics that goes on and on for the whole bar " * 3] * lines) for tag, lines in blueprint.writer_targets()]
    timed = lay_out(sheet, blueprint)
    sections = [(tag, lines) for tag, lines in sheet]
    rhyme = validate_rhymes(sections, code="en")
    brief = LyricBrief(topic="x", genre="pop", language="en")
    report = preflight(timed, rhyme, brief, coverage_target=0.9, max_filler_ratio=0.1, rhyme_mode="relaxed")
    from worker.music.workflow import PreparedSong

    prepared = PreparedSong(
        written="\n".join(f"[{t}]\n" + "\n".join(l) for t, l in sheet), sheet=timed.sheet(), blueprint=blueprint,
        timed=timed, rhyme=rhyme, preflight=report, brief=brief, caption="", bpm=None, supplied=False,
    )
    result = customer_result(prepared, None, retries=0)
    assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= RESULT_MAX_BYTES
    assert "lyrics" in result and "lyrics_lrc" in result


def test_rhyme_labels_a_writer_annotated_are_not_sung() -> None:
    from worker.music.workflow import strip_labels

    sheet = "[verse]\nMorning breeze dances through the flow (A)\nKids chase shadows on the lines [B]\nplain line\n[chorus]\nhold on (rhyme A)"
    assert strip_labels(sheet) == "[verse]\nMorning breeze dances through the flow\nKids chase shadows on the lines\nplain line\n[chorus]\nhold on"
