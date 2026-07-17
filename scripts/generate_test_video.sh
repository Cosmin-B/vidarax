#!/usr/bin/env bash
# Generate a test MP4 for Vidarax integration tests.
#
# Usage:
#   ./scripts/generate_test_video.sh [output_path]
#
# Output path defaults to /tmp/vidarax-e2e-test.mp4
#
# The video is a 10-second clip with 3 solid-colour scenes (red/blue/green)
# that produce 2 abrupt scene cuts to exercise vidarax marker detection.
#
set -euo pipefail

OUTPUT="${1:-/tmp/vidarax-e2e-test.mp4}"

# Check for ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg is not installed or not in PATH" >&2
    echo "Install it with:" >&2
    echo "  macOS:  brew install ffmpeg" >&2
    echo "  Debian: apt-get install ffmpeg" >&2
    exit 1
fi

echo "[generate-test-video] Generating $OUTPUT ..."

ffmpeg -y \
    -f lavfi -i "color=c=0xCC2222:s=640x480:d=3" \
    -f lavfi -i "color=c=0x2244CC:s=640x480:d=4" \
    -f lavfi -i "color=c=0x22AA44:s=640x480:d=3" \
    -filter_complex "[0][1][2]concat=n=3:v=1:a=0[out]" \
    -map "[out]" \
    -c:v libx264 -pix_fmt yuv420p -r 24 \
    "$OUTPUT" \
    2>&1 | grep -v "^$" | tail -5

SIZE=$(wc -c < "$OUTPUT")
echo "[generate-test-video] OK: $OUTPUT (${SIZE} bytes, 10s, 3 scenes: red/blue/green)"
