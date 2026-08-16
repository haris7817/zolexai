#!/usr/bin/env bash
# Build the matrix fixtures on the GPU box, from the box's own generated output.
#
# The conditioned workflows need a source image, a source video and an audio
# track. Nothing has to be uploaded: today's smoke runs already left real
# generated videos on disk, and LTX writes an audio stream into every
# text-to-video result, so all three can be cut locally with ffmpeg.
#
# Deliberately SHORT sources. Video-to-Video and Music Video take their length
# from the source, so a two-minute track would turn four matrix cells into
# two-minute renders. 8s of video and 20s of audio keeps those cells honest
# without making them the slowest thing in the run.
#
#   ./scripts/ltx_fixtures.sh
#   ls -la /workspace/fixtures

set -euo pipefail

DEST="${DEST:-/workspace/fixtures}"
mkdir -p "$DEST"

# Newest generated mp4 that actually decodes, preferring the smoke workspaces.
pick_source () {
  local f
  for f in $(ls -t /tmp/ltx-smoke-*/output.mp4 /tmp/grid-*.mp4 /tmp/*.mp4 2>/dev/null); do
    if ffprobe -v error -select_streams v:0 -show_entries stream=codec_type \
         -of csv=p=0 "$f" 2>/dev/null | grep -q video; then
      echo "$f"; return 0
    fi
  done
  return 1
}

SRC="$(pick_source)" || {
  echo "No generated mp4 found under /tmp." >&2
  echo "Run one smoke test first, then re-run this script:" >&2
  echo "  cd /workspace/zolexai/apps/worker && \\" >&2
  echo "  LTX_REPO_DIR=/workspace/ltx2-benchmark LTX_QUANTIZATION=nvfp4-prequant \\" >&2
  echo "  DURATION=10s ASPECT_RATIO=16:9 /workspace/zolexai/.venv-worker/bin/python \\" >&2
  echo "    scripts/ltx_smoke.py a koi pond at dawn" >&2
  exit 1
}

echo "source: $SRC"
ffprobe -v error -show_entries format=duration:stream=codec_type,width,height \
  -of default=nw=1 "$SRC" | sed 's/^/  /'
echo

# ── still.png — a frame from a third of the way in ───────────────────────
# Not frame zero: the opening frame of a generated clip is often the least
# settled, and this still becomes the identity anchor for every I2V cell.
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SRC" | cut -d. -f1)
SEEK=$(( DUR / 3 )); [ "$SEEK" -lt 1 ] && SEEK=1
ffmpeg -v error -y -ss "$SEEK" -i "$SRC" -frames:v 1 "$DEST/still.png"
echo "still.png   $(ffprobe -v error -show_entries stream=width,height -of csv=p=0:s=x "$DEST/still.png" | head -1)"

# ── clip.mp4 — 8 seconds, re-encoded so timestamps start clean ───────────
# A stream copy would keep the source's start offset and hand the adapter a
# file whose first frame is not at t=0, which is a different test than the one
# we mean to run.
ffmpeg -v error -y -i "$SRC" -t 8 -c:v libx264 -preset veryfast -crf 18 \
  -c:a aac -movflags +faststart "$DEST/clip.mp4"
echo "clip.mp4    $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$DEST/clip.mp4")s"

# ── song.mp3 — 20s. Real generated audio if the source has a stream ──────
if ffprobe -v error -select_streams a:0 -show_entries stream=codec_type \
     -of csv=p=0 "$SRC" 2>/dev/null | grep -q audio; then
  ffmpeg -v error -y -i "$SRC" -t 20 -vn -c:a libmp3lame -b:a 192k "$DEST/song.mp3"
  echo "song.mp3    from generated audio"
else
  # Fallback: a 20s tone. Mechanically valid for a shape test — it exercises the
  # same probe, sectioning and mux path — but it has no musical structure, so
  # cut alignment will fall back to even spacing. Fine for measuring whether a
  # shape renders; not a test of cut placement.
  ffmpeg -v error -y -f lavfi -i "sine=frequency=220:duration=20" \
    -c:a libmp3lame -b:a 192k "$DEST/song.mp3"
  echo "song.mp3    SYNTHETIC TONE — source had no audio stream"
  echo "            cut alignment is not meaningfully tested by this fixture"
fi

echo
echo "fixtures in $DEST:"
ls -la "$DEST"
echo
echo "now run:  cd /workspace/zolexai/apps/worker && TIER=1 ./scripts/ltx_matrix.sh"
