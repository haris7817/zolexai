"""Compare two generated videos frame by frame — the Step 8 "must match" check.

    python scripts/compare_outputs.py A.mp4 B.mp4 [--label-a comfy --label-b zolexai] [--json out.json]

Reports, for the pair:

  * container facts side by side: duration, frame count, resolution, fps,
    audio layout (must be equal for "same result");
  * per-frame PSNR and SSIM (ffmpeg's own filters, luma and average), with
    mean / min / the worst frame index — identical renders score PSNR ∞ (or
    > 50 dB after a second encode), a different seed scores ~10-15 dB;
  * an audio similarity: normalised cross-correlation of the two soundtracks
    at zero lag plus the best lag in ±500 ms, from 16 kHz mono PCM.

Pure ffmpeg + Python stdlib on purpose (runs on the GPU box and on a dev
machine, no numpy). Exit status 0 when the containers agree AND the mean
PSNR clears --psnr-threshold (default 35 dB), 1 otherwise; the JSON carries
the numbers regardless.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def probe(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,r_frame_rate,avg_frame_rate,nb_frames,sample_rate,channels:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    data = json.loads(out)
    facts: dict = {"duration": float(data["format"]["duration"])}
    for stream in data["streams"]:
        if stream["codec_type"] == "video" and "width" not in facts:
            num, den = stream["avg_frame_rate"].split("/")
            facts.update(
                width=int(stream["width"]),
                height=int(stream["height"]),
                fps=round(int(num) / max(int(den), 1), 3),
                frames=int(stream.get("nb_frames") or 0),
            )
        elif stream["codec_type"] == "audio" and "sample_rate" not in facts:
            facts.update(sample_rate=int(stream["sample_rate"]), channels=int(stream["channels"]))
    facts.setdefault("sample_rate", None)
    facts.setdefault("channels", None)
    return facts


def frame_metrics(a: Path, b: Path, width: int, height: int, tmp: Path) -> dict:
    """ffmpeg psnr + ssim filters with per-frame stats files."""
    psnr_log = tmp / "psnr.txt"
    ssim_log = tmp / "ssim.txt"
    scale = f"scale={width}:{height},format=yuv420p,fps=24"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(a),
            "-i",
            str(b),
            "-lavfi",
            f"[0:v]{scale}[a];[1:v]{scale}[b];[a][b]psnr=stats_file={psnr_log.as_posix()}",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(a),
            "-i",
            str(b),
            "-lavfi",
            f"[0:v]{scale}[a];[1:v]{scale}[b];[a][b]ssim=stats_file={ssim_log.as_posix()}",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    psnr = [
        float(m.group(1)) if m.group(1) != "inf" else math.inf
        for m in re.finditer(r"psnr_avg:(inf|[0-9.]+)", psnr_log.read_text())
    ]
    ssim = [float(m.group(1)) for m in re.finditer(r"All:([0-9.]+)", ssim_log.read_text())]

    def summary(values: list[float]) -> dict:
        finite = [v for v in values if math.isfinite(v)]
        return {
            "frames": len(values),
            "mean": (sum(finite) / len(finite)) if finite else (math.inf if values else None),
            "min": min(values) if values else None,
            "worst_frame": (values.index(min(values)) if values else None),
            "all_identical": bool(values) and all(v == math.inf for v in values),
        }

    return {"psnr_db": summary(psnr), "ssim": summary(ssim)}


def audio_similarity(a: Path, b: Path, tmp: Path) -> dict | None:
    def pcm(path: Path, dest: Path) -> array.array | None:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "s16le",
                str(dest),
                "-y",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
            return None
        samples = array.array("h")
        samples.frombytes(dest.read_bytes())
        return samples

    sa, sb = pcm(a, tmp / "a.pcm"), pcm(b, tmp / "b.pcm")
    if sa is None or sb is None:
        return None
    n = min(len(sa), len(sb), 16000 * 30)
    xa = [float(v) for v in sa[:n]]
    xb = [float(v) for v in sb[:n]]

    def corr(lag: int) -> float:
        if lag >= 0:
            pa, pb = xa[lag:], xb[: n - lag]
        else:
            pa, pb = xa[: n + lag], xb[-lag:]
        m = min(len(pa), len(pb))
        if m < 1600:
            return 0.0
        ma = sum(pa[:m]) / m
        mb = sum(pb[:m]) / m
        num = sum((pa[i] - ma) * (pb[i] - mb) for i in range(m))
        da = math.sqrt(sum((pa[i] - ma) ** 2 for i in range(m)))
        db = math.sqrt(sum((pb[i] - mb) ** 2 for i in range(m)))
        return num / (da * db) if da and db else 0.0

    zero = corr(0)
    best_lag, best = 0, zero
    for lag in range(-8000, 8001, 160):  # ±500 ms in 10 ms steps
        c = corr(lag)
        if c > best:
            best, best_lag = c, lag
    return {
        "zero_lag_correlation": round(zero, 4),
        "best_correlation": round(best, 4),
        "best_lag_ms": round(best_lag / 16.0, 1),
        "seconds_compared": round(n / 16000.0, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--psnr-threshold", type=float, default=35.0)
    args = parser.parse_args()
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ffmpeg/ffprobe required", file=sys.stderr)
        return 2

    fa, fb = probe(args.a), probe(args.b)
    same_container = (
        all(
            fa.get(k) == fb.get(k)
            for k in ("width", "height", "fps", "frames", "sample_rate", "channels")
        )
        and abs(fa["duration"] - fb["duration"]) < 0.05
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        metrics = frame_metrics(args.a, args.b, fa["width"], fa["height"], tmp)
        audio = audio_similarity(args.a, args.b, tmp)

    mean_psnr = metrics["psnr_db"]["mean"]
    passed = (
        same_container
        and mean_psnr is not None
        and (mean_psnr == math.inf or mean_psnr >= args.psnr_threshold)
    )
    report = {
        args.label_a: {"path": str(args.a), **fa},
        args.label_b: {"path": str(args.b), **fb},
        "same_container": same_container,
        "video": metrics,
        "audio": audio,
        "verdict": "MATCH" if passed else "DIFFERENT",
        "psnr_threshold_db": args.psnr_threshold,
    }
    text = json.dumps(report, indent=2, default=lambda v: "inf" if v == math.inf else v)
    print(text)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
