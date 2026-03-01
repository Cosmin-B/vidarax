#!/usr/bin/env python3
"""Generate a 10-second test video with 3 distinct scenes for E2E testing.

Scene 1 (0-3s):  solid red   (c=0xCC2222)
Scene 2 (3-7s):  solid blue  (c=0x2244CC)
Scene 3 (7-10s): solid green (c=0x22AA44)

The abrupt color changes trigger vidarax scene_cut detection.
drawtext is intentionally avoided — not available in all ffmpeg builds.

Expected vidarax results:
- 2 scene_cut markers at ~3s and ~7s
- VLM descriptions mentioning color for each scene
"""
import subprocess, sys, os

output = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vidarax-e2e-test.mp4"

cmd = [
    "ffmpeg", "-y",
    "-f", "lavfi", "-i", "color=c=0xCC2222:s=640x480:d=3",
    "-f", "lavfi", "-i", "color=c=0x2244CC:s=640x480:d=4",
    "-f", "lavfi", "-i", "color=c=0x22AA44:s=640x480:d=3",
    "-filter_complex",
    "[0][1][2]concat=n=3:v=1:a=0[out]",
    "-map", "[out]",
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-r", "24",
    output
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"FAIL: {result.stderr}", file=sys.stderr)
    sys.exit(1)

size = os.path.getsize(output)
print(f"OK: {output} ({size} bytes, 10s, 3 scenes: red/blue/green)")
