"""aichat_refmod: REFMODS for MiniMax H3 (ai-chat docs/v1.1.1/RenderEnginesDesign.md §18).

On every scene, MiniMaxH3ReferenceToVideo VAE-encodes each reference: a 124-frame reference
video took 11.7 s on an RTX 4080 (plan P6.4), against well under a second for a picture. A
character's reference video is the same from one scene to the next, so its latents are too. A
REFMOD is that video's latents (its frames through the video VAE, its soundtrack through the
audio VAE), computed once and kept as a small .safetensors file.

  MiniMaxH3EncodeRefmod  an output node: a video's frames and soundtrack in, prepared exactly as
                         MiniMaxH3ReferenceToVideo prepares them (its own helpers), encoded, and
                         written with the sha256 of each exact tensor it encoded and the VAEs'
                         file names.
  MiniMaxH3RefmodVAE     a VAE and the job's refmod files in, a pass-through VAE out. Its encode
                         returns a stored latent when the tensor's hash and the VAE's file name
                         match one, and calls the real VAE otherwise, so a refmod can only ever
                         miss: after a VAE change or a new video the hash or the name differs and
                         the reference is encoded as before. ComfyUI's H3 node is never forked.

The worker's workflow builder fills each node's VAE file names from the loaders it is wired to,
and takes the pass-through out of a job that has no refmods.

`python __init__.py` (in ComfyUI's venv: it needs torch and safetensors) runs the self-check
with a stub VAE.
"""
import hashlib
import json
import os
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "1"


def tensor_sha256(t):
    """The hash of a tensor's exact bytes, with its shape and dtype."""
    a = t.detach().contiguous().cpu()
    h = hashlib.sha256(json.dumps([list(a.shape), str(a.dtype)]).encode())
    h.update(a.reshape(-1).view(torch.uint8).numpy())
    return h.hexdigest()


class _Recorder:
    """A VAE that remembers the hash of what it was asked to encode (the encode node's)."""

    def __init__(self, vae):
        self.__dict__["_vae"] = vae
        self.__dict__["sha"] = None

    def __getattr__(self, name):  # audio_sample_rate and the rest are the real VAE's
        return getattr(self._vae, name)

    def encode(self, samples):
        self.__dict__["sha"] = tensor_sha256(samples)
        return self._vae.encode(samples)


class RefmodVAE:
    """Stands in for a VAE in front of MiniMaxH3ReferenceToVideo. `latents` = { sha256: (tensor,
    refmod file name) }, only those made with this VAE."""

    def __init__(self, vae, vae_name, latents):
        self.__dict__.update(_vae=vae, _vae_name=vae_name, _latents=latents)

    def __getattr__(self, name):
        return getattr(self._vae, name)

    def encode(self, samples):
        key = tensor_sha256(samples)
        hit = self._latents.get(key)
        if hit is not None:
            latent, name = hit
            print(f"[refmod] hit {name} ({self._vae_name})", flush=True)
            return latent.to(getattr(self._vae, "output_device", latent.device))
        print(f"[refmod] miss {tuple(samples.shape)} ({self._vae_name})", flush=True)
        return self._vae.encode(samples)


def read_refmods(paths, vae_name):
    """The latents in `paths` made with `vae_name`: the video's under its frames' hash, the
    soundtrack's under its own. A file that can't be read is a miss, never an error."""
    latents = {}
    for path in paths:
        try:
            with safe_open(path, framework="pt") as f:
                meta = f.metadata() or {}
                if meta.get("format") != FORMAT:
                    continue
                for tensor, sha, made_with in (("latent", "pixels_sha256", "vae"), ("audio_latent", "audio_sha256", "audio_vae")):
                    if meta.get(made_with) == vae_name and meta.get(sha) and tensor in f.keys():
                        latents[meta[sha]] = (f.get_tensor(tensor), os.path.basename(path))
        except Exception as e:  # noqa: BLE001 - a bad cache file costs the saving, nothing else
            print(f"[refmod] can't read {os.path.basename(path)}: {e}", flush=True)
    return latents


def prepare_video(video_frames):
    """A reference video's frames as MiniMaxH3ReferenceToVideo encodes them: its canvas (from
    the video's own size, so the scene's doesn't matter), the resize, and the snap down to 17k+5
    frames. It also cuts a video to the scene's length: never at 124 frames or fewer, the
    worker's cap and the shortest scene. If the H3 node's preparation ever changes, refmods made
    here stop matching and scenes encode as before."""
    from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE, _resize, adapt_canvas

    vh, vw = video_frames.shape[1], video_frames.shape[2]
    cw, ch = adapt_canvas(vw, vh)
    if vw * vh < cw * ch:
        cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    frames = _resize(video_frames, cw, ch, "disabled")
    n = frames.shape[0]
    if n < 5:
        raise ValueError("a reference video needs at least 5 frames")
    while n % 17 != 5:
        n -= 1
    return frames[:n]


def write_refmod(path, video_latent, pixels_sha, vae_name, audio_latent=None, audio_sha=None, audio_vae_name=""):
    tensors = {"latent": video_latent.contiguous().cpu()}
    meta = {"format": FORMAT, "kind": "video", "pixels_sha256": pixels_sha, "vae": vae_name}
    if audio_latent is not None:
        tensors["audio_latent"] = audio_latent.contiguous().cpu()
        meta.update(kind="video_audio", audio_sha256=audio_sha, audio_vae=audio_vae_name)
    save_file(tensors, path, metadata=meta)


class MiniMaxH3EncodeRefmod:
    CATEGORY = "aichat"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 - ComfyUI's naming
        return {
            "required": {
                "video": ("IMAGE",),
                "vae": ("VAE",),
                "vae_name": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "refmod/refmod"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_vae": ("VAE",),
                "audio_vae_name": ("STRING", {"default": ""}),
            },
        }

    def encode(self, video, vae, vae_name, filename_prefix, audio=None, audio_vae=None, audio_vae_name=""):
        import folder_paths
        from comfy_extras.nodes_minimax_h3 import _encode_ref_audio

        t = time.perf_counter()
        frames = prepare_video(video)
        latent = vae.encode(frames)
        audio_latent = audio_sha = None
        if audio is not None and audio_vae is not None:
            recorder = _Recorder(audio_vae)
            audio_latent, _ = _encode_ref_audio(recorder, audio)
            audio_sha = recorder.sha
        out_dir = folder_paths.get_output_directory()
        full_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        file = f"{name}_{counter:05}_.safetensors"
        write_refmod(os.path.join(full_dir, file), latent, tensor_sha256(frames), vae_name, audio_latent, audio_sha, audio_vae_name)
        print(f"[refmod] made {file}: {tuple(frames.shape)} frames{' and a soundtrack' if audio_latent is not None else ''} in {time.perf_counter() - t:.1f}s", flush=True)
        return {"ui": {"refmods": [{"filename": file, "subfolder": subfolder, "type": "output"}]}}


class MiniMaxH3RefmodVAE:
    CATEGORY = "aichat"
    RETURN_TYPES = ("VAE",)
    FUNCTION = "wrap"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "vae": ("VAE",),
                "vae_name": ("STRING", {"default": ""}),
                # The job's refmod files, uploaded to ComfyUI's input folder, one name per line.
                "refmods": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def wrap(self, vae, vae_name, refmods):
        import folder_paths

        names = [n.strip() for n in str(refmods or "").splitlines() if n.strip()]
        paths = [os.path.join(folder_paths.get_input_directory(), os.path.basename(n)) for n in names]
        latents = read_refmods(paths, vae_name)
        print(f"[refmod] {len(latents)} latent(s) for {vae_name} from {len(names)} refmod(s)", flush=True)
        return (RefmodVAE(vae, vae_name, latents) if latents else vae,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3EncodeRefmod": MiniMaxH3EncodeRefmod, "MiniMaxH3RefmodVAE": MiniMaxH3RefmodVAE}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3EncodeRefmod": "MiniMax H3 Encode Refmod", "MiniMaxH3RefmodVAE": "MiniMax H3 Refmod VAE"}


if __name__ == "__main__":
    import tempfile

    class StubVAE:
        audio_sample_rate = 32000
        output_device = torch.device("cpu")

        def __init__(self):
            self.calls = 0

        def encode(self, samples):
            self.calls += 1
            return samples.mean(dim=-1, keepdim=True) * 2 + 1  # any deterministic function of its input

    stub = StubVAE()
    frames = torch.rand(124, 64, 48, 3)
    other = frames.clone()
    other[0, 0, 0, 0] += 0.5
    sound = torch.rand(1, 1000, 2)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "r.safetensors")
        rec = _Recorder(stub)
        made = stub.encode(frames)
        audio_made = rec.encode(sound)
        write_refmod(path, made, tensor_sha256(frames), "video.safetensors", audio_made, rec.sha, "audio.safetensors")
        assert stub.calls == 2

        video_vae = RefmodVAE(stub, "video.safetensors", read_refmods([path], "video.safetensors"))
        assert torch.equal(video_vae.encode(frames), made), "the same tensor: the stored latent"
        assert stub.calls == 2, "a hit never calls the VAE"
        video_vae.encode(other)
        assert stub.calls == 3, "another tensor: the real VAE"
        assert video_vae.audio_sample_rate == 32000, "everything else is the real VAE's"

        assert read_refmods([path], "another_vae.safetensors") == {}, "another VAE name: nothing"
        RefmodVAE(stub, "another_vae.safetensors", read_refmods([path], "another_vae.safetensors")).encode(frames)
        assert stub.calls == 4, "another VAE name: the real VAE"

        audio_vae = RefmodVAE(stub, "audio.safetensors", read_refmods([path], "audio.safetensors"))
        assert torch.equal(audio_vae.encode(sound), audio_made) and stub.calls == 4, "the soundtrack by its own hash"
        assert read_refmods([os.path.join(tmp, "missing.safetensors")], "video.safetensors") == {}, "a missing file is a miss"
    print("aichat_refmod self-check ok")
