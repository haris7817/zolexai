#!/usr/bin/env bash
# Does a restyle still resemble its source at the END, or only at the start?
#
# The client's complaint about video-to-video was not that the look was wrong,
# it was that the video stopped being their video: "after 20 secs it changes,
# then there is no woman present". That is a claim about TIME, and it is
# measurable without watching anything — compare the output against its source
# frame by frame and look at how the similarity behaves as the clip runs.
#
# A restyle is *supposed* to differ from its source, so the absolute number is
# not the signal. The SHAPE is: a restyle that holds its subject keeps a flat
# line, and one that drifts away falls off a cliff somewhere in the middle and
# never recovers.
#
#   ./scripts/drift_check.sh SOURCE.mp4 OUTPUT.mp4 [MORE_OUTPUTS...]
#
# Every output is compared against the same source and printed side by side,
# which is how two conditioning settings get judged against each other.

set -uo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 SOURCE.mp4 OUTPUT.mp4 [OUTPUT2.mp4 ...]" >&2
  exit 2
fi

SOURCE=$1; shift
WINDOW="${WINDOW:-5}"     # seconds per reported bucket
FPS="${FPS:-24}"          # both sides resampled to this before comparing
GRID="${GRID:-1024:576}"  # and to this size, so shape differences do not count

[ -f "$SOURCE" ] || { echo "source not found: $SOURCE" >&2; exit 2; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# ── Measure each output against the source ───────────────────────────────
labels=(); files=()
for output in "$@"; do
  [ -f "$output" ] || { echo "skipping missing output: $output" >&2; continue; }
  label=$(basename "$(dirname "$output")")
  stats="$TMP/$label.ssim"

  # setpts + fps on BOTH sides: the filter pairs frames in order, so a source
  # at a different frame rate would compare frame 100 against frame 130 and
  # report drift that is really just misalignment.
  ffmpeg -v error -y \
    -i "$output" -i "$SOURCE" \
    -lavfi "[0:v]fps=$FPS,scale=$GRID,setpts=PTS-STARTPTS[a];\
[1:v]fps=$FPS,scale=$GRID,setpts=PTS-STARTPTS[b];\
[a][b]ssim=stats_file=$stats" \
    -f null - 2>/dev/null

  if [ ! -s "$stats" ]; then
    echo "could not compare $output — is it the same length as the source?" >&2
    continue
  fi
  labels+=("$label"); files+=("$stats")
done

[ ${#files[@]} -gt 0 ] || { echo "nothing to compare" >&2; exit 1; }

# ── Bucket by time and print ─────────────────────────────────────────────
python3 - "$WINDOW" "$FPS" "${labels[@]}" -- "${files[@]}" <<'PY'
import re, sys

window = float(sys.argv[1]); fps = float(sys.argv[2])
rest = sys.argv[3:]
split = rest.index("--")
labels, files = rest[:split], rest[split + 1:]

series = []
for path in files:
    buckets = {}
    with open(path) as handle:
        for line in handle:
            frame = re.search(r"\bn:(\d+)", line)
            value = re.search(r"\bAll:([0-9.]+)", line)
            if not (frame and value):
                continue
            bucket = int((int(frame.group(1)) - 1) / fps / window)
            buckets.setdefault(bucket, []).append(float(value.group(1)))
    series.append({k: sum(v) / len(v) for k, v in buckets.items()})

span = max((max(s) for s in series if s), default=-1) + 1
width = max((len(l) for l in labels), default=8)

print()
print("Similarity to the SOURCE, by time window.")
print("Flat = the output still follows the upload. Falling = it is drifting away.")
print()
header = "  window   " + "".join(f"{l:>{width + 2}}" for l in labels)
print(header)
print("  " + "-" * (len(header) - 2))
for bucket in range(int(span)):
    start, end = bucket * window, (bucket + 1) * window
    row = f"  {start:>4.0f}-{end:<4.0f} "
    for s in series:
        row += f"{s.get(bucket, float('nan')):>{width + 2}.3f}"
    print(row)

print()
for label, s in zip(labels, series):
    if len(s) < 2:
        continue
    ordered = [s[k] for k in sorted(s)]
    half = max(1, len(ordered) // 2)
    first, second = ordered[:half], ordered[half:]
    opening = sum(first) / len(first)
    closing = sum(second) / len(second)
    drop = (opening - closing) / opening * 100 if opening else 0.0
    verdict = "HOLDS" if drop < 5 else ("DRIFTS" if drop < 12 else "DRIFTS BADLY")
    print(f"  {label:>{width}}: first half {opening:.3f} -> second half "
          f"{closing:.3f}   {drop:+.1f}%   {verdict}")
print()
PY
