"""MOCK_COMFY=1: encode a real, playable MP4 with ffmpeg instead of rendering on a GPU.
Python port of render-worker/src/mock.js.

This is how you exercise the ENTIRE path — queue intake, retry, delivery, the status
webhook — with no GPU, no models and no spend. ffmpeg is already in the image (the
Dockerfile installs it for ComfyUI's video encode), and the entrypoint skips booting
ComfyUI entirely in mock mode.
"""
import os
import subprocess
import tempfile
import time

# How long to SIMULATE a render for (real ComfyUI is minutes; keep the POC snappy).
MOCK_RENDER_SECONDS = float(os.environ.get("MOCK_RENDER_SECONDS") or 12)

# drawtext HARD-FAILS on a missing fontfile, and the path differs per base image
# (Debian/Ubuntu vs Alpine vs a bare RunPod image), so probe instead of assuming. No
# font anywhere -> render a plain colour card rather than fail the mock.
_FONT_CANDIDATES = (
    os.environ.get("MOCK_FONT") or "",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


def _font():
    return next((p for p in _FONT_CANDIDATES if p and os.path.isfile(p)), None)


def _wrap(text, width=34):
    lines, line = [], ""
    for word in (text or "").split():
        if len((line + " " + word).strip()) > width:
            lines.append(line.strip())
            line = word
        else:
            line += " " + word
    if line.strip():
        lines.append(line.strip())
    return "\n".join(lines)


def render_mock(job, on_progress=None):
    """Simulate progress, then encode a colour card + the spoken line. Returns bytes."""
    steps = 20
    for i in range(1, steps + 1):
        time.sleep(MOCK_RENDER_SECONDS / steps)
        if on_progress:
            on_progress(i, steps, "mock-sampler")

    name = str(job.get("personalitySlug") or "AI").upper()
    dialogue = _wrap(job.get("dialogue") or "(no dialogue)")
    try:
        duration = max(3, round(float(job.get("durationSeconds") or 8)))
    except (TypeError, ValueError):
        duration = 8

    with tempfile.TemporaryDirectory(prefix="render-") as d:
        txt = os.path.join(d, "line.txt")
        out = os.path.join(d, "out.mp4")
        with open(txt, "w") as f:
            f.write(f"{name}\n\n{dialogue}")

        cmd = [
            "ffmpeg", "-y",
            # libx264 + yuv420p require EVEN width AND height.
            "-f", "lavfi", "-i", f"color=c=0x10243a:s=720x404:d={duration}",
        ]
        font = _font()
        if font:
            cmd += [
                "-vf",
                f"drawtext=fontfile={font}:textfile={txt}:fontcolor=white:fontsize=26:"
                "x=(w-text_w)/2:y=(h-text_h)/2:line_spacing=12:box=1:boxcolor=0x00000088:boxborderw=24",
            ]
        cmd += ["-r", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]

        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-5:]
            raise RuntimeError("ffmpeg mock render failed: " + " | ".join(tail))
        with open(out, "rb") as f:
            return f.read()
