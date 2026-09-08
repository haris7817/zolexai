from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zolex_music_worker.adapters import LtxRenderAdapter
from zolex_music_worker.analysis import SongAnalysis, SongSection
from zolex_music_worker.config import WorkerConfig
from zolex_music_worker.director import expand_direction
from zolex_music_worker.lyrics import (
    LyricTranscript,
    apply_lyric_directions,
    lyrics_requested,
    parse_supplied_lyrics,
    prepare_lyrics,
)
from zolex_music_worker.media import ltx_window_frames
from zolex_music_worker.models import AudioInfo, EditorialPlan, MusicVideoRequest, Performer, Shot
from zolex_music_worker.planner import build_plan, compile_legacy_manifest, validate_plan
from zolex_music_worker.performance import build_latency_capacity_plan
from zolex_music_worker.prompts import compile_shot_prompts
from zolex_music_worker.reference_fetch import fetch_reference_video
from zolex_music_worker.reference_style import _apply_ai_analysis
from zolex_music_worker.upscaler import build_cuda_upscale_filter, upscale_master
from zolex_music_worker.utils import format_template_argv
from zolex_music_worker.vision_reference_analyzer import parse_style_json


def config(root: Path) -> WorkerConfig:
    return WorkerConfig(work_root=root, anchor_backend="mock", render_backend="mock")


class RequestTests(unittest.TestCase):
    def test_short_prompt_request_uses_safe_defaults(self) -> None:
        request = MusicVideoRequest.from_dict(
            {"job_id": "mv-love-1", "audio": "/audio/song.wav", "prompt": "make me a love video"}
        )
        self.assertEqual(request.formats, ("16:9",))
        self.assertTrue(request.automatic_creative_direction)
        self.assertTrue(request.lip_sync)
        self.assertEqual(request.performers, ())

    def test_one_line_lyric_direction_enables_automatic_transcription(self) -> None:
        request = MusicVideoRequest.from_dict(
            {
                "job_id": "mv-lyrics-1",
                "audio": "/audio/song.wav",
                "prompt": "make me a video according to the lyrics of the song",
            }
        )
        self.assertTrue(lyrics_requested(request))

    def test_reference_video_alias_is_accepted(self) -> None:
        request = MusicVideoRequest.from_dict(
            {
                "job_id": "mv-reference-1",
                "audio": "/audio/song.wav",
                "prompt": "make me a love video",
                "style_reference_video": "/video/reference.mp4",
            }
        )
        self.assertEqual(request.reference_video, "/video/reference.mp4")

    def test_https_reference_video_url_is_accepted(self) -> None:
        request = MusicVideoRequest.from_dict(
            {
                "job_id": "mv-reference-url-1",
                "audio": "/audio/song.wav",
                "prompt": "make me a love video",
                "reference_video_url": "https://youtu.be/example",
            }
        )
        self.assertEqual(request.reference_video_url, "https://youtu.be/example")

    def test_reference_path_and_url_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(Exception, "either reference_video"):
            MusicVideoRequest.from_dict(
                {
                    "job_id": "mv-reference-conflict",
                    "audio": "/audio/song.wav",
                    "prompt": "make me a love video",
                    "reference_video": "/video/reference.mp4",
                    "reference_video_url": "https://youtu.be/example",
                }
            )

    def test_reference_url_requires_https(self) -> None:
        with self.assertRaisesRegex(Exception, "HTTPS"):
            MusicVideoRequest.from_dict(
                {
                    "job_id": "mv-reference-http",
                    "audio": "/audio/song.wav",
                    "prompt": "make me a love video",
                    "reference_video_url": "http://example.com/video.mp4",
                }
            )

    def test_srt_lyrics_preserve_clean_timestamped_lines(self) -> None:
        segments = parse_supplied_lyrics(
            "1\n00:00:01,000 --> 00:00:03,500\nFirst line\n\n2\n00:00:03,500 --> 00:00:06,000\nSecond line\n",
            fps=24,
            total_frames=192,
            language="en",
        )
        self.assertEqual([item.text for item in segments], ["First line", "Second line"])
        self.assertEqual((segments[0].start_frame, segments[0].end_frame), (24, 84))

    def test_supplied_lyrics_bypass_transcription_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = MusicVideoRequest(
                "mv-supplied-lyrics",
                "song.wav",
                "follow the lyrics",
                lyrics="[00:00.00] We will rise\n[00:02.00] We will rise",
                lyrics_language="en",
            )
            audio = AudioInfo(
                source_path="song.wav",
                decoded_wav="decoded.wav",
                aligned_wav="aligned.wav",
                conditioning_wav="conditioning.wav",
                sample_rate=48000,
                channels=2,
                decoded_samples=192000,
                total_frames=96,
                fps=24,
                decoded_duration=4.0,
                aligned_duration=4.0,
                source_sha256="0" * 64,
            )
            transcript = prepare_lyrics(
                request,
                audio,
                WorkerConfig(
                    work_root=root,
                    anchor_backend="mock",
                    render_backend="mock",
                    transcription_backend="disabled",
                ),
                root / "analysis",
            )
            self.assertEqual(transcript.source, "supplied")
            self.assertTrue(all(item.is_refrain for item in transcript.segments))
            self.assertTrue((root / "analysis" / "lyrics.json").is_file())

    def test_invalid_job_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(Exception, "job_id"):
            MusicVideoRequest.from_dict(
                {"job_id": "../../escape", "audio": "/audio/song.wav", "prompt": "love video"}
            )

    def test_one_reference_produces_solo_love_treatment(self) -> None:
        request = MusicVideoRequest(
            job_id="mv-solo",
            audio="song.wav",
            prompt="make me a love video",
            performers=(Performer(id="artist", reference_images=("artist.png",)),),
        )
        analysis = SongAnalysis(
            fps=24,
            total_frames=240,
            tempo_bpm=100,
            transients=(0, 120, 240),
            sections=(SongSection("development", 0, 240, 0.5, True, 0.35),),
        )
        treatment = expand_direction(request, analysis)
        self.assertEqual(treatment["performer_strategy"], "solo")
        self.assertEqual(treatment["performer_ids"], ["artist"])

    def test_five_performer_band_defaults_to_five_visible_people(self) -> None:
        request = MusicVideoRequest.from_dict(
            {
                "job_id": "mv-band-five",
                "audio": "/audio/song.wav",
                "prompt": "make me a music video for this five-person band",
                "performers": [
                    {"id": f"member-{index}", "reference_images": [f"/images/member-{index}.jpg"]}
                    for index in range(1, 6)
                ],
            }
        )
        self.assertEqual(len(request.performers), 5)
        self.assertEqual(request.max_people_visible_per_shot, 5)

    def test_six_performers_are_rejected(self) -> None:
        with self.assertRaisesRegex(Exception, "zero to five"):
            MusicVideoRequest.from_dict(
                {
                    "job_id": "mv-band-six",
                    "audio": "/audio/song.wav",
                    "prompt": "make me a band video",
                    "performers": [{"id": f"member-{index}"} for index in range(1, 7)],
                }
            )


class PlanningTests(unittest.TestCase):
    def _analysis(self, seconds: int = 182) -> SongAnalysis:
        total = 24 * seconds
        section_names = ("opening", "introduction", "development", "emotional_middle", "climax", "ending")
        fractions = (0.0, 0.07, 0.25, 0.50, 0.70, 0.93)
        starts = tuple(round(total * value) for value in fractions)
        ends = starts[1:] + (total,)
        sections = tuple(
            SongSection(name, start, end, 0.6, name not in {"opening", "ending"}, 0.35)
            for name, start, end in zip(section_names, starts, ends)
        )
        transients = tuple(range(0, total + 1, 48))
        return SongAnalysis(24, total, 105.0, transients, sections)

    def test_plan_is_frame_exact_and_reference_style_paced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request = MusicVideoRequest("mv-plan", "song.wav", "make me a love video")
            analysis = self._analysis()
            treatment = expand_direction(request, analysis)
            plan = build_plan(request, treatment, analysis, "16:9", config(Path(tmp)))
            validate_plan(plan.shots, analysis.total_frames)
            self.assertGreaterEqual(len(plan.shots), 38)
            self.assertLessEqual(len(plan.shots), 46)
            self.assertEqual(sum(shot.frame_count for shot in plan.shots), analysis.total_frames)
            self.assertEqual(plan.shots[-1].end_frame, analysis.total_frames)

    def test_five_minute_plan_is_supported_and_frame_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            worker_config = config(Path(tmp))
            self.assertEqual(worker_config.max_source_seconds, 300)
            request = MusicVideoRequest("mv-five-minutes", "song.wav", "make me a love video")
            analysis = self._analysis(300)
            treatment = expand_direction(request, analysis)
            plan = build_plan(request, treatment, analysis, "16:9", worker_config)
            self.assertGreaterEqual(len(plan.shots), 62)
            self.assertLessEqual(len(plan.shots), 80)
            self.assertEqual(plan.total_frames, 7200)
            self.assertEqual(sum(shot.frame_count for shot in plan.shots), 7200)

    def test_prompt_and_portable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            anchor = base / "anchors" / "close.png"
            anchor.parent.mkdir()
            anchor.touch()
            shot = Shot(
                id="shot-001",
                start_frame=0,
                frame_count=96,
                section="development",
                family="close_performance",
                energy=0.7,
                vocals_present=True,
                performer_ids=["artist"],
                anchor_role="close_performance",
                anchor_image=str(anchor),
            )
            plan = EditorialPlan(2, 24, "16:9", 1024, 576, 96, [shot])
            request = MusicVideoRequest(
                "mv-prompt", "song.wav", "love video", performers=(Performer(id="artist"),)
            )
            treatment = expand_direction(
                request,
                SongAnalysis(
                    24,
                    96,
                    None,
                    (0, 96),
                    (SongSection("development", 0, 96, 0.7, True, 0.35),),
                ),
            )
            compile_shot_prompts(plan, request, treatment)
            manifest = compile_legacy_manifest(plan, base)
            self.assertEqual(
                set(manifest["shots"][0]),
                {"id", "start_seconds", "duration_seconds", "prompt", "image"},
            )
            self.assertEqual(manifest["shots"][0]["image"], "anchors/close.png")
            self.assertIn("one continuous take", shot.prompt.casefold())

    def test_timestamped_lyrics_drive_shots_and_4k_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            worker_config = config(Path(tmp))
            request = MusicVideoRequest(
                "mv-lyric-plan",
                "song.wav",
                "make me a rap video according to the lyrics of the song",
            )
            analysis = self._analysis(12)
            segments = parse_supplied_lyrics(
                "[00:00.00] I came from the street\n[00:04.00] Now I work to win\n[00:08.00] I remember the past",
                fps=24,
                total_frames=analysis.total_frames,
                language="en",
            )
            transcript = LyricTranscript(True, "supplied", "en", 1.0, segments)
            treatment = expand_direction(request, analysis, transcript)
            plan = build_plan(request, treatment, analysis, "16:9", worker_config, transcript)
            apply_lyric_directions(plan, transcript, treatment, worker_config, Path(tmp) / "lyrics")
            compile_shot_prompts(plan, request, treatment)

            self.assertEqual((plan.width, plan.height), (3840, 2160))
            self.assertEqual((plan.scene_width, plan.scene_height), (1280, 704))
            portrait = build_plan(request, treatment, analysis, "9:16", worker_config, transcript)
            self.assertEqual((portrait.width, portrait.height), (2160, 3840))
            self.assertEqual((portrait.scene_width, portrait.scene_height), (704, 1280))
            self.assertTrue(all(shot.lyric_segment_ids for shot in plan.shots))
            self.assertTrue(any("current lyric idea" in shot.prompt for shot in plan.shots))
            self.assertTrue(treatment["lyric_driven"])

    def test_reference_style_changes_pacing_and_enters_scene_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            worker_config = config(Path(tmp))
            request = MusicVideoRequest("mv-reference-plan", "song.wav", "make me a love video")
            analysis = self._analysis(12)
            reference_style = {
                "style_summary": "warm high-contrast imagery with energetic movement",
                "camera_language": "fast lateral tracking and controlled handheld energy",
                "shot_scale_and_angles": "low-angle wides and eye-level close-ups",
                "scene_archetypes": "generic waterside exteriors and sparse performance interiors",
                "lighting_style": "warm directional practical light",
                "color_palette": "warm amber highlights and deep blue shadows",
                "editing_rhythm": "brisk cuts around 2.2 seconds",
                "composition_style": "wide environmental framing",
                "motion_style": "energetic subject movement",
                "average_shot_seconds": 2.2,
                "originality_policy": "transfer filmmaking grammar only",
            }
            treatment = expand_direction(request, analysis, reference_style=reference_style)
            plan = build_plan(request, treatment, analysis, "16:9", worker_config)
            default_treatment = expand_direction(request, analysis)
            default_plan = build_plan(request, default_treatment, analysis, "16:9", worker_config)
            compile_shot_prompts(plan, request, treatment)
            self.assertTrue(treatment["reference_video_used"])
            self.assertGreater(len(plan.shots), len(default_plan.shots))
            self.assertTrue(all("reference video's general filmmaking grammar" in shot.prompt for shot in plan.shots))
            self.assertTrue(all("do not reproduce" in shot.prompt for shot in plan.shots))
            self.assertTrue(all("low-angle wides" in shot.prompt for shot in plan.shots))

    def test_five_person_band_gets_rotating_solos_and_full_group_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            worker_config = config(Path(tmp))
            request = MusicVideoRequest.from_dict(
                {
                    "job_id": "mv-band-plan",
                    "audio": "song.wav",
                    "prompt": "make me a performance video for this five-person band",
                    "performers": [
                        {
                            "id": f"member-{index}",
                            "role": role,
                            "description": f"member {index} locked wardrobe",
                            "reference_images": [f"member-{index}.jpg"],
                        }
                        for index, role in enumerate(
                            ("lead_vocalist", "guitarist", "bassist", "keyboardist", "drummer"),
                            start=1,
                        )
                    ],
                }
            )
            analysis = self._analysis(60)
            treatment = expand_direction(request, analysis)
            plan = build_plan(request, treatment, analysis, "16:9", worker_config)
            compile_shot_prompts(plan, request, treatment)
            group_shots = [shot for shot in plan.shots if len(shot.performer_ids) == 5]
            solo_shots = [shot for shot in plan.shots if len(shot.performer_ids) == 1]
            self.assertEqual(treatment["performer_strategy"], "band_or_group_ensemble")
            self.assertTrue(group_shots)
            self.assertTrue(solo_shots)
            self.assertTrue(all(f"member-{index}" in group_shots[0].prompt for index in range(1, 6)))
            self.assertIn("Never blend, duplicate or swap", group_shots[0].prompt)
            self.assertIn("every assigned person fully visible", group_shots[0].prompt)
            self.assertIn("role: drummer", group_shots[0].prompt)
            self.assertIn("Instrumental members perform their stated roles", group_shots[0].prompt)


class AdapterContractTests(unittest.TestCase):
    def test_ltx_window_is_legal(self) -> None:
        self.assertEqual(ltx_window_frames(48), 121)
        self.assertEqual(ltx_window_frames(121), 121)
        self.assertEqual((ltx_window_frames(168) - 1) % 8, 0)

    def test_command_template_remains_an_argument_array(self) -> None:
        rendered = format_template_argv(
            ["renderer", "--prompt", "{prompt}", "--output", "{output}"],
            {"prompt": "love; touch /tmp/never", "output": "/tmp/out.mp4"},
        )
        self.assertEqual(rendered[2], "love; touch /tmp/never")
        self.assertEqual(len(rendered), 5)

    def test_ltx_command_uses_frame_driven_audio_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_files = [root / f"model-{index}.bin" for index in range(6)]
            for path in model_files:
                path.touch()
            anchor = root / "anchor.png"
            audio_path = root / "conditioning.wav"
            anchor.touch()
            audio_path.touch()
            worker_config = WorkerConfig(
                work_root=root,
                ltx_python=sys.executable,
                ltx_transformer_path=model_files[0],
                ltx_text_encoder_path=model_files[1],
                ltx_video_vae_path=model_files[2],
                ltx_audio_vae_path=model_files[3],
                ltx_spatial_upsampler_path=model_files[4],
                ltx_distilled_lora_path=model_files[5],
            )
            shot = Shot(
                id="shot-001",
                start_frame=48,
                frame_count=96,
                section="development",
                family="close_performance",
                energy=0.5,
                vocals_present=True,
                performer_ids=["artist"],
                anchor_role="close_performance",
                anchor_image=str(anchor),
                prompt="performer sings while the camera slowly pushes in",
                seed=10,
            )
            plan = EditorialPlan(2, 24, "16:9", 1024, 576, 192, [shot])
            audio = AudioInfo(
                source_path=str(audio_path),
                decoded_wav=str(audio_path),
                aligned_wav=str(audio_path),
                conditioning_wav=str(audio_path),
                sample_rate=48000,
                channels=2,
                decoded_samples=1,
                total_frames=192,
                fps=24,
                decoded_duration=8.0,
                aligned_duration=8.0,
                source_sha256="0" * 64,
            )
            captured: list[str] = []
            captured_environment: dict[str, str] = {}

            def fake_run(argv, **kwargs):
                captured.extend(argv)
                captured_environment.update(kwargs.get("env_overrides") or {})
                Path(argv[argv.index("--output-path") + 1]).touch()

            with patch("zolex_music_worker.adapters.run_command", side_effect=fake_run):
                LtxRenderAdapter(worker_config).render(
                    shot=shot,
                    plan=plan,
                    audio=audio,
                    output=root / "raw.mp4",
                    attempt=1,
                    worker_slot=0,
                    gpu_id="3",
                    timeout=30,
                    log_path=root / "command.json",
                )
            self.assertIn("--num-frames", captured)
            self.assertNotIn("--audio-max-duration", captured)
            self.assertEqual(captured[captured.index("--audio-start-time") + 1], "2.000000000000")
            self.assertEqual(captured_environment["CUDA_VISIBLE_DEVICES"], "3")


class UpscaleContractTests(unittest.TestCase):
    def test_cuda_upscale_crops_ltx_padding_and_stream_copies_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "working-master.mp4"
            output = root / "output.mp4"
            source.touch()
            worker_config = WorkerConfig(
                work_root=root,
                upscale_backend="cuda",
                upscale_gpu_id="3",
            )
            plan = EditorialPlan(
                schema_version=3,
                fps=24,
                format="16:9",
                width=3840,
                height=2160,
                total_frames=240,
                render_width=1280,
                render_height=704,
            )
            captured: list[str] = []
            captured_environment: dict[str, str] = {}

            def fake_run(argv, **kwargs):
                captured.extend(argv)
                captured_environment.update(kwargs.get("env_overrides") or {})
                Path(argv[-1]).touch()

            with patch("zolex_music_worker.upscaler.run_command", side_effect=fake_run):
                upscale_master(
                    input_path=source,
                    output_path=output,
                    plan=plan,
                    config=worker_config,
                    log_path=root / "upscale-command.json",
                )
            filter_text = build_cuda_upscale_filter(plan)
            self.assertIn("crop=1252:704:14:0", filter_text)
            self.assertIn("scale_cuda=w=3840:h=2160", filter_text)
            self.assertEqual(captured[captured.index("-c:a") + 1], "copy")
            self.assertEqual(captured_environment["CUDA_VISIBLE_DEVICES"], "3")


class ReferenceFetchSecurityTests(unittest.TestCase):
    def test_disallowed_reference_host_is_rejected_before_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker_config = WorkerConfig(
                work_root=root,
                reference_fetch_backend="command",
                reference_fetch_command=["fetcher", "{url}", "{output}"],
                reference_allowed_hosts=("youtube.com", "youtu.be"),
            )
            with patch("zolex_music_worker.reference_fetch.run_command") as runner:
                with self.assertRaisesRegex(Exception, "not allowed"):
                    fetch_reference_video(
                        "https://untrusted.example/video.mp4",
                        output_dir=root / "reference",
                        config=worker_config,
                    )
            runner.assert_not_called()

    def test_ip_address_reference_host_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker_config = WorkerConfig(
                work_root=root,
                reference_fetch_backend="command",
                reference_fetch_command=["fetcher", "{url}", "{output}"],
                reference_allowed_hosts=("127.0.0.1",),
            )
            with self.assertRaisesRegex(Exception, "IP address"):
                fetch_reference_video(
                    "https://127.0.0.1/private",
                    output_dir=root / "reference",
                    config=worker_config,
                )


class ReferenceVisionTests(unittest.TestCase):
    def test_fenced_qwen_json_is_parsed_and_unknown_fields_are_ignored(self) -> None:
        result = parse_style_json(
            """```json
            {
              "style_summary": "grounded performance film",
              "camera_language": "controlled tracking and slow push-ins",
              "shot_scale_and_angles": "low-angle wides alternating with eye-level close-ups",
              "scene_archetypes": "urban night exterior and practical-lit interior",
              "unknown": "ignored"
            }
            ```"""
        )
        self.assertEqual(
            result["shot_scale_and_angles"],
            "low-angle wides alternating with eye-level close-ups",
        )
        self.assertNotIn("unknown", result)

    def test_local_qwen_profile_is_merged_without_loading_a_real_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contact_sheet = root / "contact-sheet.jpg"
            reference = root / "reference.mp4"
            contact_sheet.touch()
            reference.touch()
            worker_config = WorkerConfig(
                work_root=root,
                anchor_backend="mock",
                render_backend="mock",
                reference_vision_backend="qwen",
            )
            response = {
                "style_summary": "cinematic urban performance",
                "camera_language": "restrained handheld tracking",
                "shot_scale_and_angles": "low wides and eye-level close-ups",
                "scene_archetypes": "generic urban night exterior",
            }
            with patch(
                "zolex_music_worker.vision_reference_analyzer.analyze_contact_sheet_with_qwen",
                return_value=response,
            ) as analyzer:
                profile = _apply_ai_analysis(
                    {"style_summary": "technical fallback"},
                    reference_video=reference,
                    contact_sheet=contact_sheet,
                    output_dir=root,
                    config=worker_config,
                )
            analyzer.assert_called_once()
            self.assertEqual(
                profile["analysis_backend"],
                "built_in_metrics_plus_qwen2_5_vl_3b",
            )
            self.assertEqual(profile["shot_scale_and_angles"], "low wides and eye-level close-ups")


class PerformanceTargetTests(unittest.TestCase):
    def test_three_minute_five_minute_target_requires_three_times_baseline_capacity(self) -> None:
        worker_config = WorkerConfig(
            work_root=Path("/tmp/zolex-target-test"),
            render_backend="command",
            render_command=["renderer", "{request_json}", "{output}"],
            render_concurrency=12,
            baseline_job_seconds=900,
            baseline_audio_seconds=180,
            baseline_render_concurrency=4,
            target_job_seconds=300,
        )
        capacity = build_latency_capacity_plan(
            audio_seconds=180,
            format_count=1,
            render_backend="command",
            config=worker_config,
        )
        self.assertEqual(capacity["required_render_concurrency"], 12)
        self.assertEqual(capacity["required_capacity_multiplier"], 3.0)
        self.assertTrue(capacity["capacity_ready"])

    def test_two_output_formats_double_required_capacity(self) -> None:
        worker_config = WorkerConfig(
            work_root=Path("/tmp/zolex-target-test"),
            render_backend="command",
            render_command=["renderer", "{request_json}", "{output}"],
            render_concurrency=12,
            baseline_job_seconds=900,
            baseline_audio_seconds=180,
            baseline_render_concurrency=4,
            target_job_seconds=300,
        )
        capacity = build_latency_capacity_plan(
            audio_seconds=180,
            format_count=2,
            render_backend="command",
            config=worker_config,
        )
        self.assertEqual(capacity["required_render_concurrency"], 24)
        self.assertFalse(capacity["capacity_ready"])


if __name__ == "__main__":
    unittest.main()
