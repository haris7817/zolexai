from __future__ import annotations

import math
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from zolex_music_worker.config import WorkerConfig
from zolex_music_worker.media import video_probe
from zolex_music_worker.models import MusicVideoRequest
from zolex_music_worker.worker import run_job


def write_test_song(path: Path, seconds: float = 2.0, rate: int = 48000) -> None:
    sample_count = round(seconds * rate)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(rate)
        for index in range(sample_count):
            envelope = 0.35 + 0.15 * math.sin(2 * math.pi * 2 * index / rate)
            value = int(16000 * envelope * math.sin(2 * math.pi * 220 * index / rate))
            output.writeframesraw(struct.pack("<hh", value, value))


def write_reference_video(path: Path) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x180:d=1:r=24",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:d=1:r=24",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p",
            "-c:v",
            "libx264",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)


class MockEndToEndTests(unittest.TestCase):
    def test_complete_job_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song = root / "song.wav"
            write_test_song(song)
            reference = root / "reference.mp4"
            write_reference_video(reference)
            work_root = root / "runs"
            config = WorkerConfig(
                work_root=work_root,
                anchor_backend="mock",
                render_backend="mock",
                render_concurrency=3,
                command_timeout_seconds=300,
                reference_fetch_backend="command",
                reference_fetch_command=["cp", str(reference), "{output}"],
                reference_allowed_hosts=("example.com",),
            )
            request = MusicVideoRequest(
                job_id="mock-e2e",
                audio=str(song),
                prompt="make me a love video",
                formats=("16:9",),
                lip_sync=False,
                anchor_backend="mock",
                render_backend="mock",
                reference_video_url="https://example.com/style-reference.mp4",
            )

            first = run_job(request, config)
            output = Path(first["formats"]["16:9"]["output"])
            self.assertTrue(output.is_file())
            probe = video_probe(output, config, count_frames=True)
            self.assertEqual(probe["frame_count"], 48)
            self.assertAlmostEqual(probe["fps"], 24.0)
            self.assertEqual((probe["width"], probe["height"]), (3840, 2160))
            self.assertTrue(probe["has_audio"])
            working = work_root / "mock-e2e" / "16x9" / "final" / "working-master.mp4"
            working_probe = video_probe(working, config, count_frames=True)
            self.assertEqual((working_probe["width"], working_probe["height"]), (1280, 704))
            self.assertTrue(working_probe["has_audio"])
            self.assertEqual(first["formats"]["16:9"]["audio_during_upscale"], "stream_copy")
            self.assertEqual(first["latency_target"]["target_seconds"], 300)
            self.assertFalse(first["latency_target"]["guaranteed"])
            self.assertTrue(first["reference_video"]["used"])
            self.assertEqual(first["reference_video"]["input_type"], "url")
            self.assertEqual(
                first["reference_video"]["source_url"],
                "https://example.com/style-reference.mp4",
            )
            self.assertEqual(
                first["reference_video"]["analysis_backend"],
                "built_in_visual_metrics",
            )
            style_profile = work_root / "mock-e2e" / "analysis" / "reference-video" / "style-profile.json"
            contact_sheet = work_root / "mock-e2e" / "analysis" / "reference-video" / "contact-sheet.jpg"
            self.assertTrue(style_profile.is_file())
            self.assertTrue(contact_sheet.is_file())

            accepted = sorted((work_root / "mock-e2e" / "16x9" / "shots").glob("shot-*/accepted.mp4"))
            self.assertEqual(len(accepted), 1)
            first_mtimes = [path.stat().st_mtime_ns for path in accepted]
            second = run_job(request, config)
            self.assertEqual(second["status"], "complete")
            self.assertEqual([path.stat().st_mtime_ns for path in accepted], first_mtimes)


if __name__ == "__main__":
    unittest.main()
