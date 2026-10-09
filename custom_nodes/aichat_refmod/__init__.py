"""aichat_refmod: REFMODS for MiniMax H3 (ai-chat docs/v1.1.1/RenderEnginesDesign.md §18).

On every scene, MiniMaxH3ReferenceToVideo prepares each reference (loads the file and resizes it),
VAE-encodes it, and hands it to the Qwen3-VL text encoder. On an RTX 4080 a 124-frame reference
video cost ~12.7 s of loading and resizing and 11.7 s of VAE, a character sheet ~0.4 s and
0.4-0.8 s. A character's references are the same from one scene to the next, and so is all of
that. A REFMOD keeps it, made once:

  - the VAE latents: a picture's, or a video's frames and its soundtrack's
  - the prepared pixels Qwen3-VL reads: the resized picture, or the video's frames at 2 fps. As
    8-bit, because the H3 node resizes through 8-bit pixels (comfy.utils.lanczos), so they come
    back exactly; to_u8 refuses any that would not.

Qwen3-VL's own reading is not kept. Its vision tower is cheap (0.16 s a picture, 2.6 s a video)
and its output large (~30 MB a picture, ~500 MB a video), and its language model reads the
references together with each new prompt. So the text encoder runs as before, on the same pixels,
and the scene is the same scene.

  MiniMaxH3EncodeRefmod            output node: a picture (with the scene's canvas), or a video and
                                   its soundtrack, prepared exactly as the H3 node prepares them,
                                   encoded, and written as one .safetensors.
  MiniMaxH3ReferenceToVideoCached  MiniMaxH3ReferenceToVideo, given refmods. A reference whose
                                   refmod fits (its format, the VAEs it was made with, its size at
                                   this canvas, its length against this scene) is built from it, and
                                   its loader never runs, because each reference is a lazy input;
                                   any other is loaded and prepared as the H3 node does. The same
                                   conditioning and latent, so a refmod can only ever miss.

The worker's builder puts the cached node in the H3 node's place only on a job with refmods, and
fills the VAE names from the loaders. Run from the ComfyUI directory in its venv,
`python custom_nodes/aichat_refmod/__init__.py` checks the cached node against the H3 node with a
stub VAE and text encoder. Run it after updating ComfyUI: the cached node copies the H3 node's
reference code, and the check is what says they still agree.
"""
import math
import os
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "2"
IMAGES, VIDEOS = 9, 3  # MiniMaxH3ReferenceToVideo's own limits


def from_u8(u8):
    """8-bit pixels back to the floats comfy.utils.lanczos made them as."""
    return torch.from_numpy(u8.cpu().numpy().astype(np.float32) / 255.0)


def to_u8(pixels):
    """Prepared pixels as 8-bit, refusing any that are not exactly 8-bit: a refmod that changed the
    picture would not be a cache."""
    u8 = (pixels.cpu() * 255.0).round().clamp(0, 255).to(torch.uint8)
    if not torch.equal(from_u8(u8), pixels.cpu()):
        raise ValueError("these pixels are not 8-bit: the H3 node's resize must have changed")
    return u8


def image_size(w, h, width, height, ref_image_size):
    """A picture's size in a scene of width x height, as MiniMaxH3ReferenceToVideo sizes it."""
    from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE, REF_IMAGE_SHORT_EDGE

    if ref_image_size == "match":
        scale = min(1.0, math.sqrt((width * height) / (w * h)))
    else:
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
    tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return tw, th


def prepare_image(image, width, height, ref_image_size="match"):
    """A picture as MiniMaxH3ReferenceToVideo prepares it for a scene of width x height."""
    from comfy_extras.nodes_minimax_h3 import _resize

    tw, th = image_size(image.shape[2], image.shape[1], width, height, ref_image_size)
    return _resize(image[:1], tw, th, "disabled")


def video_size(vw, vh):
    """A reference video's canvas, as MiniMaxH3ReferenceToVideo sizes it: from its own shape."""
    from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE, adapt_canvas

    cw, ch = adapt_canvas(vw, vh)
    if vw * vh < cw * ch:
        cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return cw, ch


def prepare_video(video_frames, frame_count=None):
    """A reference video's frames as MiniMaxH3ReferenceToVideo prepares them: resized, cut to the
    scene's frame_count when it has more, and snapped down to 17k+5 frames. A refmod is made with
    no cut, and fits only a scene at least as long as the video (the worker's loader caps it at
    124 frames, the shortest scene)."""
    from comfy_extras.nodes_minimax_h3 import _resize

    cw, ch = video_size(video_frames.shape[2], video_frames.shape[1])
    frames = _resize(video_frames, cw, ch, "disabled")
    if frame_count is not None and frames.shape[0] > frame_count:
        frames = frames[:frame_count]
    n = frames.shape[0]
    if n < 5:
        raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
    while n % 17 != 5:
        n -= 1
    return frames[:n]


def qwen_frames(frames):
    """The frames Qwen3-VL reads of a reference video, sampled as the H3 node samples them."""
    from comfy_extras.nodes_minimax_h3 import FPS

    return frames[list(range(0, frames.shape[0], FPS // 2))]


def write_refmod(path, tensors, meta):
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, path,
              metadata={"format": FORMAT, **{k: str(v) for k, v in meta.items()}})


def read_meta(path):
    """A refmod's metadata, or None when it can't be read or is another format: a miss."""
    try:
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
        return meta if meta.get("format") == FORMAT else None
    except Exception as e:  # noqa: BLE001 - a bad cache file costs the saving, nothing else
        print(f"[refmod] can't read {os.path.basename(path)}: {e}", flush=True)
        return None


def read_tensors(path):
    with safe_open(path, framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def misfit(meta, kind, vae_name, audio_vae_name, width, height, ref_image_size, frame_count):
    """Why this refmod can't stand in for its reference in this scene, or None when it can."""
    if meta.get("kind") != kind:
        return f"a {meta.get('kind')} refmod for a {kind}"
    if meta.get("vae") != vae_name:
        return f"made with {meta.get('vae')}, not {vae_name}"
    if kind == "image":
        size = image_size(int(meta["w"]), int(meta["h"]), width, height, ref_image_size)
        if size != (int(meta["tw"]), int(meta["th"])):
            return f"made at {meta['tw']}x{meta['th']}, this canvas wants {size[0]}x{size[1]}"
        return None
    if int(meta["frames_in"]) > frame_count:
        return f"{meta['frames_in']} frames, longer than this scene's {frame_count}"
    if meta.get("soundtrack") == "1" and meta.get("audio_vae") != audio_vae_name:
        return f"its soundtrack made with {meta.get('audio_vae')}, not {audio_vae_name}"
    return None


def _slot(name):
    """ref_image_3 -> ("image", 3); ref_video_audio_1 -> ("video", 1): a soundtrack is its video's."""
    return ("image" if name.startswith("ref_image_") else "video"), int(name.rsplit("_", 1)[1])


def _output_device(vae, t):
    return t.to(getattr(vae, "output_device", t.device))


class MiniMaxH3EncodeRefmod:
    CATEGORY = "aichat"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 - ComfyUI's naming
        return {
            "required": {
                "vae": ("VAE",),
                "vae_name": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "refmod/refmod"}),
            },
            # A video (and its soundtrack), or a picture with the scene's canvas.
            "optional": {
                "video": ("IMAGE",),
                "audio": ("AUDIO",),
                "audio_vae": ("VAE",),
                "audio_vae_name": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "width": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            },
        }

    def encode(self, vae, vae_name, filename_prefix, video=None, audio=None, audio_vae=None, audio_vae_name="", image=None, width=0, height=0):
        import folder_paths
        from comfy_extras.nodes_minimax_h3 import _encode_ref_audio

        t = time.perf_counter()
        if image is not None:
            if not (width and height):
                raise ValueError("a picture's refmod needs the scene's width and height")
            pixels = prepare_image(image, width, height)
            tensors = {"latent": vae.encode(pixels), "pixels": to_u8(pixels)}
            meta = {"kind": "image", "vae": vae_name, "w": image.shape[2], "h": image.shape[1], "tw": pixels.shape[2], "th": pixels.shape[1]}
        elif video is not None:
            frames = prepare_video(video)
            tensors = {"latent": vae.encode(frames), "pixels": to_u8(qwen_frames(frames))}
            meta = {"kind": "video", "vae": vae_name, "frames_in": video.shape[0], "cw": frames.shape[2], "ch": frames.shape[1], "soundtrack": int(audio is not None)}
            if audio is not None and audio_vae is not None:
                tensors["audio_latent"], _ = _encode_ref_audio(audio_vae, audio)
                meta["audio_vae"] = audio_vae_name
        else:
            raise ValueError("a refmod needs a video or a picture")
        out_dir = folder_paths.get_output_directory()
        full_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        file = f"{name}_{counter:05}_.safetensors"
        write_refmod(os.path.join(full_dir, file), tensors, meta)
        print(f"[refmod] made {file}: a {'picture' if image is not None else 'video'} {tuple(tensors['pixels'].shape)}"
              f"{' and a soundtrack' if 'audio_latent' in tensors else ''} in {time.perf_counter() - t:.1f}s", flush=True)
        return {"ui": {"refmods": [{"filename": file, "subfolder": subfolder, "type": "output"}]}}


class MiniMaxH3ReferenceToVideoCached:
    CATEGORY = "aichat"
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        lazy = lambda kind: (kind, {"lazy": True})  # noqa: E731
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 16384}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16384}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600}),
                "ref_image_size": (["match", "max"],),
                # The job's refmods, one per line: "<image|video> <index> <file in the input folder>".
                "refmods": ("STRING", {"multiline": True, "default": ""}),
                "vae_name": ("STRING", {"default": ""}),
                "audio_vae_name": ("STRING", {"default": ""}),
            },
            "optional": {
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                **{f"ref_image_{i}": lazy("IMAGE") for i in range(IMAGES)},
                **{f"ref_video_{i}": lazy("IMAGE") for i in range(VIDEOS)},
                **{f"ref_video_audio_{i}": lazy("AUDIO") for i in range(VIDEOS)},
            },
        }

    def _fitting(self, refmods, vae_name, audio_vae_name, width, height, length, ref_image_size):
        """{slot: path} of the refmods that fit this scene. Once per prompt: ComfyUI asks
        check_lazy_status again after each input it loads, then runs execute."""
        key = (refmods, vae_name, audio_vae_name, width, height, length, ref_image_size)
        if getattr(self, "_fit_key", None) == key:
            return self._fit
        import folder_paths
        from comfy_extras.nodes_minimax_h3 import temporal_shape

        frame_count = temporal_shape(length)[0]
        fit = {}
        for line in str(refmods or "").splitlines():
            parts = line.split()
            if len(parts) != 3 or parts[0] not in ("image", "video") or not parts[1].isdigit():
                continue
            name = os.path.basename(parts[2])
            path = os.path.join(folder_paths.get_input_directory(), name)
            meta = read_meta(path)
            why = "can't be read" if meta is None else misfit(meta, parts[0], vae_name, audio_vae_name, width, height, ref_image_size, frame_count)
            if why:
                print(f"[refmod] miss {name}: {why}", flush=True)
            else:
                fit[(parts[0], int(parts[1]))] = (path, meta)
        self._fit_key, self._fit = key, fit
        return fit

    def check_lazy_status(self, refmods, vae_name, audio_vae_name, width, height, length, ref_image_size, **inputs):
        fit = self._fitting(refmods, vae_name, audio_vae_name, width, height, length, ref_image_size)
        return [n for n, v in inputs.items() if n.startswith("ref_") and v is None and _slot(n) not in fit]

    def execute(self, clip, prompt, width, height, length, ref_image_size, refmods, vae_name, audio_vae_name, vae=None, audio_vae=None, **refs):
        # The H3 node's own execute, reference by reference, with a fitting refmod in place of
        # its file. Keep the raw branches in step with comfy_extras/nodes_minimax_h3.py.
        import node_helpers
        from comfy_extras.nodes_minimax_h3 import FPS, _empty_av_latent, _encode_ref_audio

        fit = self._fitting(refmods, vae_name, audio_vae_name, width, height, length, ref_image_size)
        latent, frame_count = _empty_av_latent(width, height, length)
        ref_items, ref_blocks = [], []

        for i in range(IMAGES):
            if ("image", i) in fit:
                path, _ = fit[("image", i)]
                r = read_tensors(path)
                resized = from_u8(r["pixels"])
                print(f"[refmod] hit {os.path.basename(path)} (picture {i + 1})", flush=True)
                z = _output_device(vae, r["latent"]) if vae is not None else None
            else:
                img = refs.get(f"ref_image_{i}")
                if img is None:
                    continue
                resized = prepare_image(img, width, height, ref_image_size)
                z = vae.encode(resized) if vae is not None else None
            ref_items.append({"type": "image", "data": resized})
            if z is not None:
                ref_blocks.append({"kind": "image", "latent_h": resized.shape[1] // 16, "latent_w": resized.shape[2] // 16, "latent": z})

        for k in range(VIDEOS):
            if ("video", k) in fit:
                path, meta = fit[("video", k)]
                r = read_tensors(path)
                qwen = from_u8(r["pixels"])
                has_sound, cw, ch = meta.get("soundtrack") == "1", int(meta["cw"]), int(meta["ch"])
                print(f"[refmod] hit {os.path.basename(path)} (video {k + 1})", flush=True)
            else:
                video_frames = refs.get(f"ref_video_{k}")
                if video_frames is None:
                    continue
                soundtrack = refs.get(f"ref_video_audio_{k}")
                frames = prepare_video(video_frames, frame_count)
                qwen, has_sound, cw, ch = qwen_frames(frames), soundtrack is not None, frames.shape[2], frames.shape[1]
            if has_sound:
                # the soundtrack gets its own <Audio j> label, emitted before <Video k>
                ref_items.append({"type": "audio"})
            ref_items.append({"type": "video", "data": qwen, "timestamps": [j / 2.0 for j in range(qwen.shape[0])]})
            if vae is None:
                continue
            audio_latent, ref_audio_t = (None, 0)
            if ("video", k) in fit:
                z = _output_device(vae, r["latent"])
                if has_sound and audio_vae is not None and "audio_latent" in r:
                    audio_latent = _output_device(audio_vae, r["audio_latent"])
                    ref_audio_t = audio_latent.shape[-1]
            else:
                z = vae.encode(frames)
                if soundtrack is not None and audio_vae is not None:
                    audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                               "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})
        assert FPS == 24, "a refmod's video frames are sampled at the H3 node's 2 per second"

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return (cond, latent)


NODE_CLASS_MAPPINGS = {"MiniMaxH3EncodeRefmod": MiniMaxH3EncodeRefmod, "MiniMaxH3ReferenceToVideoCached": MiniMaxH3ReferenceToVideoCached}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3EncodeRefmod": "MiniMax H3 Encode Refmod", "MiniMaxH3ReferenceToVideoCached": "MiniMax H3 Reference to Video (refmods)"}


if __name__ == "__main__":
    # The self-check: the cached node against ComfyUI's own H3 node, with stubs for the VAEs and
    # the text encoder. Run from the ComfyUI directory, in its venv.
    import sys
    import tempfile

    sys.path.insert(0, os.getcwd())
    import folder_paths
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

    class StubVAE:
        audio_sample_rate = 32000
        output_device = torch.device("cpu")

        def __init__(self):
            self.calls = 0

        def encode(self, samples):  # any deterministic function of its input
            self.calls += 1
            return samples.float().mean(dim=-1, keepdim=True).unsqueeze(0) * 2 + 1

    class StubCLIP:
        def tokenize(self, prompt, minimax_ref_items=None):
            return {"prompt": prompt, "items": minimax_ref_items}

        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.zeros(1), {"tokens": tokens}]]

    def same(a, b, where="root"):
        if hasattr(a, "tensors") and hasattr(b, "tensors"):  # comfy.nested_tensor.NestedTensor
            same(list(a.tensors), list(b.tensors), f"{where}.tensors")
        elif torch.is_tensor(a) or torch.is_tensor(b):
            assert torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b), where
        elif isinstance(a, dict):
            assert isinstance(b, dict) and a.keys() == b.keys(), (where, a.keys(), getattr(b, "keys", lambda: b)())
            for key in a:
                same(a[key], b[key], f"{where}.{key}")
        elif isinstance(a, (list, tuple)):
            assert isinstance(b, (list, tuple)) and len(a) == len(b), where
            for n, (x, y) in enumerate(zip(a, b)):
                same(x, y, f"{where}[{n}]")
        else:
            assert a == b, (where, a, b)

    torch.manual_seed(0)
    sheet = torch.rand(1, 1000, 600, 3)  # big enough to be scaled to the canvas
    place = torch.rand(1, 260, 200, 3)
    video = torch.rand(124, 96, 64, 3)
    sound = {"waveform": torch.rand(1, 2, 32000 * 5), "sample_rate": 32000}
    vae, audio_vae, clip = StubVAE(), StubVAE(), StubCLIP()
    scene = dict(prompt="P", width=480, height=640, length=124, ref_image_size="match")

    native = MiniMaxH3ReferenceToVideo.execute(clip, vae=vae, audio_vae=audio_vae, **scene,
                                               ref_images={"ref_image_0": sheet, "ref_image_1": place},
                                               ref_videos={"ref_video_0": video}, ref_video_audios={"ref_video_audio_0": sound}).args
    raw = {"ref_image_0": sheet, "ref_image_1": place, "ref_video_0": video, "ref_video_audio_0": sound}
    names = dict(vae_name="video.safetensors", audio_vae_name="audio.safetensors")
    node = MiniMaxH3ReferenceToVideoCached()
    same(native, node.execute(clip, vae=vae, audio_vae=audio_vae, refmods="", **names, **scene, **raw), "no refmods")

    with tempfile.TemporaryDirectory() as tmp:
        folder_paths.set_output_directory(tmp)
        folder_paths.set_input_directory(os.path.join(tmp, "refmod"))
        enc = MiniMaxH3EncodeRefmod()
        enc.encode(vae, "video.safetensors", "refmod/sheet", image=sheet, width=480, height=640)
        enc.encode(vae, "video.safetensors", "refmod/video", video=video, audio=sound, audio_vae=audio_vae, audio_vae_name="audio.safetensors")
        refmods = "image 0 sheet_00001_.safetensors\nvideo 0 video_00001_.safetensors"

        # Both refmods fit: their loaders are never asked for, and the scene is the same.
        node = MiniMaxH3ReferenceToVideoCached()
        lazy = {"ref_image_0": None, "ref_image_1": None, "ref_video_0": None, "ref_video_audio_0": None}
        assert node.check_lazy_status(refmods=refmods, **names, **scene, **lazy) == ["ref_image_1"], "only the place is loaded"
        calls = vae.calls + audio_vae.calls
        same(native, node.execute(clip, vae=vae, audio_vae=audio_vae, refmods=refmods, **names, **scene, ref_image_1=place), "both refmods")
        assert vae.calls + audio_vae.calls == calls + 1, "only the place is encoded"

        # Each kind of misfit loads the file instead, and the scene is still the same.
        for why, change, loads in (
            ("another canvas", dict(width=576, height=768), {"ref_image_0", "ref_image_1"}),
            ("another VAE", dict(vae_name="other.safetensors"), set(lazy)),
            ("another audio VAE", dict(audio_vae_name="other.safetensors"), {"ref_image_1", "ref_video_0", "ref_video_audio_0"}),
        ):
            node = MiniMaxH3ReferenceToVideoCached()
            s, n = {**scene, **{k: v for k, v in change.items() if k in scene}}, {**names, **{k: v for k, v in change.items() if k in names}}
            needed = node.check_lazy_status(refmods=refmods, **n, **s, **lazy)
            assert set(needed) == loads, (why, needed)
            expected = MiniMaxH3ReferenceToVideo.execute(clip, vae=vae, audio_vae=audio_vae, **s,
                                                         ref_images={"ref_image_0": sheet, "ref_image_1": place},
                                                         ref_videos={"ref_video_0": video}, ref_video_audios={"ref_video_audio_0": sound}).args
            same(expected, node.execute(clip, vae=vae, audio_vae=audio_vae, refmods=refmods, **n, **s, **{k: raw[k] for k in needed}), why)

        node = MiniMaxH3ReferenceToVideoCached()
        assert node.check_lazy_status(refmods="image 0 missing.safetensors", **names, **scene, **lazy) == list(lazy), "a missing refmod is a miss"

    try:
        to_u8(torch.full((1, 2, 2, 3), 0.5))
        raise AssertionError("pixels that are not 8-bit must be refused")
    except ValueError:
        pass
    print("aichat_refmod self-check ok")
