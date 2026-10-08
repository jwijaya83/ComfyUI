"""MOCK_COMFY=1: encode a real, playable file with ffmpeg instead of rendering on a GPU.
Python port of render-worker/src/mock.js.

This is how you exercise the ENTIRE path — queue intake, retry, delivery, the status
webhook — with no GPU, no models and no spend. ffmpeg is already in the image (the
Dockerfile installs it for ComfyUI's video encode), and the entrypoint skips booting
ComfyUI entirely in mock mode. The job's `kind` picks the file: an MP4 for a video (the
default), a PNG for an `image` (a Krea 2 job), a short tone for `audio` (Music 3).
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


def _size(job, default):
    """The job's width × height, else `default`. libx264 + yuv420p require EVEN width
    AND height; every real tier value is already a multiple of 32, so this only matters
    for an ad-hoc/manual job."""
    try:
        width = int(job.get("width") or default[0])
        height = int(job.get("height") or default[1])
    except (TypeError, ValueError):
        width, height = default
    return width - width % 2, height - height % 2


def _ffmpeg(cmd, out):
    proc = subprocess.run(cmd + [out], capture_output=True)
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-5:]
        raise RuntimeError("ffmpeg mock render failed: " + " | ".join(tail))
    with open(out, "rb") as f:
        return f.read()


def render_mock(job, on_progress=None):
    """Simulate progress, then encode a card with the job's words on it.
    Returns (bytes, filename): the filename's extension is the file's type."""
    steps = 20
    for i in range(1, steps + 1):
        time.sleep(MOCK_RENDER_SECONDS / steps)
        if on_progress:
            on_progress(i, steps, "mock-sampler")

    kind = job.get("kind")
    name = str(job.get("personalitySlug") or "AI").upper()
    try:
        duration = max(3, round(float(job.get("durationSeconds") or 8)))
    except (TypeError, ValueError):
        duration = 8

    with tempfile.TemporaryDirectory(prefix="render-") as d:
        if kind == "latent":
            # A refmod job (ai-chat milestone D): a stub, so the path from the worker to the row
            # runs with no GPU. It never matches a real tensor, so a scene given it just misses.
            return b"refmod stub: not a safetensors file\n", "mock.safetensors"
        if kind == "audio":
            out = os.path.join(d, "out.flac")
            return _ffmpeg(["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-c:a", "flac"], out), "mock.flac"

        # A picture shows its prompt (it has no line); a clip shows the spoken line.
        if kind == "image":
            text = _wrap((job.get("positive") or "(no prompt)")[:300])
            width, height = _size(job, (720, 1280))
        else:
            text = _wrap(job.get("dialogue") or "(no dialogue)")
            # Mirror the job's tier resolution so MOCK_COMFY=1 actually exercises the
            # width/height wiring end to end (no GPU needed to sanity-check a tier's
            # aspect ratio).
            width, height = _size(job, (720, 404))
        txt = os.path.join(d, "line.txt")
        with open(txt, "w") as f:
            f.write(f"{name}\n\n{text}")

        source = f"color=c=0x10243a:s={width}x{height}" + ("" if kind == "image" else f":d={duration}")
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", source]
        font = _font()
        if font:
            cmd += [
                "-vf",
                f"drawtext=fontfile={font}:textfile={txt}:fontcolor=white:fontsize=26:"
                "x=(w-text_w)/2:y=(h-text_h)/2:line_spacing=12:box=1:boxcolor=0x00000088:boxborderw=24",
            ]
        if kind == "image":
            return _ffmpeg(cmd + ["-frames:v", "1"], os.path.join(d, "out.png")), "mock.png"
        cmd += ["-r", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        return _ffmpeg(cmd, os.path.join(d, "out.mp4")), "mock.mp4"
