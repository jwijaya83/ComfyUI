"""Build a ComfyUI API-format graph from a workflow template + its meta sidecar.

A faithful Python port of the render-worker's workflowLoader.js: load
`workflows/<name>.json`, patch the nodes the meta sidecar names (prompt, lora,
frame count, length in seconds, source video, reference image, a scene's reference
pictures and videos, save prefix), return the graph. The SEED NODES keep the template's fixed seed, so a tested graph renders the way
it was tested — unless the job sends a `seed`. Only one job does: a Krea 2 picture drawn
again after the vision check rejected it, because the same seed and prompt would draw the
same picture. Each turn's `positive` prompt differs, so ComfyUI's input-hash cache never
collides.

Run it directly for the assert-based self-check: `python workflow_builder.py`.
"""
import json
import math
import os
import random
import re

WORKFLOWS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")

# The length of a seed clip when the job doesn't say (mirrors render-plane's
# SEED_DURATION_SECONDS — this worker is stateless and can't import it). Seeds are no
# longer all this long: a full render follows its line, so render-plane sends the source
# seed's real length as `sourceSeconds`, and this is only the fallback for an older caller.
SEED_DURATION_SECONDS = 20


def _safe_name(name):
    s = re.sub(r"[^a-zA-Z0-9_-]", "", str(name))
    if not s:
        raise ValueError(f"Invalid workflow name: {name}")
    return s


def _load_meta(name):
    path = os.path.join(WORKFLOWS_DIR, f"{name}.meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Workflow '{name}' is missing its meta sidecar at {name}.meta.json"
        )
    with open(path) as f:
        return json.load(f)


def build_workflow(
    *,
    name,
    prompt,
    reference_image=None,
    use_reference_image=True,
    lora_name=None,
    duration_seconds=None,
    width=None,
    height=None,
    filename_prefix=None,
    source_video=None,
    source_seconds=None,
    lora_strength=None,
    seed=None,
    ref_images=None,
    ref_videos=None,
    refmods=None,
):
    safe = _safe_name(name)
    with open(os.path.join(WORKFLOWS_DIR, f"{safe}.json")) as f:
        workflow = json.load(f)
    meta = _load_meta(safe)

    # Every template but the refmod one (which encodes a video, and has no prompt) names one.
    pos = meta.get("positivePromptNode")
    if pos:
        if pos not in workflow:
            raise ValueError(f"Workflow '{safe}' has no positive-prompt node '{pos}'.")
        # A CLIPTextEncode takes `text`; a primitive text box (H3's node 138) takes `value`.
        pos_inputs = workflow[pos]["inputs"]
        pos_inputs["text" if "text" in pos_inputs else "value"] = prompt

    # Latent injection: point the VHS_LoadVideo node at the uploaded seed clip. Only
    # workflows whose meta declares a sourceVideoNode have one.
    src_node = meta.get("sourceVideoNode")
    if source_video and src_node and src_node in workflow:
        workflow[src_node]["inputs"]["video"] = source_video

    # The LoadImage node always needs a resolvable filename even when i2v is bypassed:
    # swap to the declared filler when reference use is off so the graph still validates.
    ref_node = meta.get("referenceImageNode")
    if ref_node and ref_node in workflow:
        if use_reference_image and reference_image:
            workflow[ref_node]["inputs"]["image"] = reference_image
        elif not use_reference_image and meta.get("defaultReferenceImage"):
            workflow[ref_node]["inputs"]["image"] = meta["defaultReferenceImage"]

    # Flip the i2v bypass switch in sync with the caller's choice.
    bypass = meta.get("bypassReferenceImageNode")
    if bypass and bypass in workflow:
        workflow[bypass]["inputs"]["value"] = not use_reference_image

    lora_node = meta.get("loraNode")
    if lora_name and lora_node and lora_node in workflow:
        workflow[lora_node]["inputs"]["lora_name"] = lora_name
    # The LoRA's strength, when the job sends one. Krea 2 switches her LoRA per picture:
    # 0 (the export's value) leaves her out of a view or a side character's sheet.
    if lora_strength is not None and lora_node and lora_node in workflow:
        workflow[lora_node]["inputs"]["strength_model"] = float(lora_strength)

    # Another seed, only when the job sends one (see the module docstring).
    if seed is not None:
        for node in meta.get("seedNodes") or []:
            if node in workflow:
                inputs = workflow[node]["inputs"]
                inputs["noise_seed" if "noise_seed" in inputs else "seed"] = int(seed)

    # Video dimensions: EmptyLTXVLatentVideo (basic_workflow) takes them as width/height;
    # VHS_LoadVideo (latent_injection) takes them as custom_width/custom_height, where 0
    # means "keep the source clip's native size" — so detect which key the node has.
    width_node = meta.get("widthNode")
    if width and width_node and width_node in workflow:
        inputs = workflow[width_node]["inputs"]
        inputs["width" if "width" in inputs else "custom_width"] = int(width)

    height_node = meta.get("heightNode")
    if height and height_node and height_node in workflow:
        inputs = workflow[height_node]["inputs"]
        inputs["height" if "height" in inputs else "custom_height"] = int(height)

    # A template that takes its length in SECONDS (H3's node 132) and snaps the frames
    # itself; the framesNode/fpsNode path below is LTX's.
    duration_node = meta.get("durationNode")
    if duration_seconds and duration_node and duration_node in workflow:
        workflow[duration_node]["inputs"]["value"] = float(duration_seconds)

    frames_node = meta.get("framesNode")
    fps_node = meta.get("fpsNode")
    if duration_seconds and frames_node and fps_node and frames_node in workflow and fps_node in workflow:
        fps = float(workflow[fps_node]["inputs"].get("value") or 24)
        target = max(1, round(duration_seconds * fps))
        # LTX requires frame counts of form 8n+1; snap up so we never undershoot.
        frames = math.ceil((target - 1) / 8) * 8 + 1
        workflow[frames_node]["inputs"]["value"] = frames
        # Latent injection: cap frames pulled from the SOURCE clip to the same length
        # so a reused reply can be shorter than the full seed (matches Node behaviour).
        cap_node = meta.get("frameLoadCapNode")
        if cap_node and cap_node in workflow:
            workflow[cap_node]["inputs"]["frame_load_cap"] = frames

    # Seed-reuse start offset: a latent-injection reuse loads the window
    # [skip_first_frames, skip_first_frames + frame_load_cap] out of the source seed.
    # From the seed's length (`source_seconds`, else SEED_DURATION_SECONDS) we compute how
    # many frames it has and start at a RANDOM in-bounds point — otherwise every reuse of a
    # clip begins on the same frame and they all look alike. Clamped so the window never
    # runs past the seed's end. (Port of workflowLoader.js; the Node worker is stateless
    # too, hence the duplicated constant.)
    cap_node = meta.get("frameLoadCapNode")
    if source_video and cap_node and cap_node in workflow:
        loader = workflow[cap_node]
        fps = 24
        if fps_node and fps_node in workflow:
            fps = float(workflow[fps_node]["inputs"].get("value") or 24)
        total_seed_frames = round((source_seconds or SEED_DURATION_SECONDS) * fps)
        load_cap = int(loader["inputs"].get("frame_load_cap") or total_seed_frames)
        max_skip = max(0, total_seed_frames - load_cap)
        loader["inputs"]["skip_first_frames"] = random.randint(0, max_skip) if max_skip > 0 else 0

    # A scene's pictures (H3 ref2va, ai-chat v1.1.1): the reference node's
    # ref_images.ref_image_0…N-1 point at the first N loaders, in order, each loader set to
    # its uploaded picture, and every other ref_image key is removed. The order is the
    # prompt's <Picture N> numbering. The loaders left over are dropped: nothing reads them,
    # but ComfyUI would still look for the export's test pictures they name.
    refs_node, loaders = meta.get("refImageNode"), meta.get("refImageLoaders") or []
    if ref_images and refs_node in workflow:
        if len(ref_images) > len(loaders):
            raise ValueError(f"Workflow '{safe}' takes at most {len(loaders)} reference pictures, not {len(ref_images)}.")
        inputs = workflow[refs_node]["inputs"]
        for key in [k for k in inputs if k.startswith("ref_images.ref_image_")]:
            del inputs[key]
        for i, image in enumerate(ref_images):
            inputs[f"ref_images.ref_image_{i}"] = [loaders[i], 0]
            workflow[loaders[i]]["inputs"]["image"] = image
        for unused in loaders[len(ref_images):]:
            workflow.pop(unused, None)

    # A scene's reference VIDEOS (ai-chat milestone D): each one's frames on
    # ref_videos.ref_video_i and its soundtrack on ref_video_audios.ref_video_audio_i of the same
    # node, from the i-th of `refVideoLoaders` (VHS_LoadVideo at 24 fps, capped at 124 frames,
    # the shortest scene's length, so H3 never cuts one). The order is the prompt's <Video k>.
    # Loaders left over are dropped, as for pictures, and so are all three on a job with none.
    vloaders = meta.get("refVideoLoaders") or []
    if vloaders and refs_node in workflow:
        videos = ref_videos or []
        if len(videos) > len(vloaders):
            raise ValueError(f"Workflow '{safe}' takes at most {len(vloaders)} reference videos, not {len(videos)}.")
        inputs = workflow[refs_node]["inputs"]
        for key in [k for k in inputs if k.startswith(("ref_videos.", "ref_video_audios."))]:
            del inputs[key]
        for i, video in enumerate(videos):
            inputs[f"ref_videos.ref_video_{i}"] = [vloaders[i], 0]
            inputs[f"ref_video_audios.ref_video_audio_{i}"] = [vloaders[i], 2]
            workflow[vloaders[i]]["inputs"]["video"] = video
        for unused in vloaders[len(videos):]:
            workflow.pop(unused, None)

    # REFMODS (ai-chat milestone D, custom_nodes/aichat_refmod): a reference's prepared pixels
    # and latents, made once. `refmods` = {"image": [...], "video": [...]}, a file or None per
    # reference, in ref_images / ref_videos order. On a job with any, the H3 node becomes
    # MiniMaxH3ReferenceToVideoCached: the same node, with each reference's loader a lazy input
    # it runs only when that refmod doesn't fit. A job with none keeps the H3 node itself.
    lines = [f"{kind} {i} {name}" for kind in ("image", "video") for i, name in enumerate((refmods or {}).get(kind) or []) if name]
    cached = []
    if lines and refs_node in workflow and workflow[refs_node]["class_type"] == "MiniMaxH3ReferenceToVideo":
        node = workflow[refs_node]
        node["class_type"] = "MiniMaxH3ReferenceToVideoCached"
        node["inputs"] = {k.split(".", 1)[-1]: v for k, v in node["inputs"].items()}  # ref_images.ref_image_0 -> ref_image_0
        node["inputs"]["refmods"] = "\n".join(lines)
        cached = [refs_node]
    # A refmod is made with, and only fits, the VAEs named in it, so every refmod node takes its
    # VAE names from the loaders it is wired to: a template whose VAE changes never uses a stale one.
    for node_id in [n for n in (meta.get("refmodNodes") or []) + cached if n in workflow]:
        inputs = workflow[node_id]["inputs"]
        for k in ("vae", "audio_vae"):
            src = inputs.get(k)
            if isinstance(src, list) and src[0] in workflow:
                inputs[f"{k}_name"] = workflow[src[0]]["inputs"].get("vae_name", "")

    save_node = meta.get("saveVideoNode")
    if filename_prefix and save_node and save_node in workflow:
        workflow[save_node]["inputs"]["filename_prefix"] = filename_prefix

    return workflow


if __name__ == "__main__":
    # The self-check (ai-chat docs/v1.1.1/ImplementationPlan.md W3).
    def template(name):
        with open(os.path.join(WORKFLOWS_DIR, f"{name}.json")) as f:
            return json.load(f)

    h3, h3_t = build_workflow(name="minimax_h3_r2v_hybrid", prompt="P", duration_seconds=10), template("minimax_h3_r2v_hybrid")
    assert h3["138"]["inputs"]["value"] == "P"
    assert h3["132"]["inputs"]["value"] == 10
    assert h3["129"] == h3_t["129"], "a job without a seed keeps the template's"

    krea, krea_t = build_workflow(name="krea2_image_creator", prompt="K", lora_strength=0.8), template("krea2_image_creator")
    assert krea["70"]["inputs"]["text"] == "K"
    assert krea["57"]["inputs"]["strength_model"] == 0.8
    assert krea["52"] == krea_t["52"]
    krea7 = build_workflow(name="krea2_image_creator", prompt="K", seed=7)
    assert krea7["52"]["inputs"]["seed"] == 7
    assert krea7["57"]["inputs"]["strength_model"] == krea_t["57"]["inputs"]["strength_model"]

    # W2: two pictures wire exactly the first two loaders, in order; a job with none keeps the export's.
    h3r = build_workflow(name="minimax_h3_r2v_hybrid", prompt="P", ref_images=["sheet.png", "place.png"])
    refs = {k: v for k, v in h3r["136"]["inputs"].items() if k.startswith("ref_images.")}
    assert refs == {"ref_images.ref_image_0": ["137", 0], "ref_images.ref_image_1": ["148", 0]}, refs
    assert (h3r["137"]["inputs"]["image"], h3r["148"]["inputs"]["image"]) == ("sheet.png", "place.png")
    dropped = set(h3_t) - set(h3r)
    assert dropped == {"147", "151", "152", "153", "154", "155", "156", "157", "158", "159"}, dropped
    assert not any(isinstance(v, list) and v and v[0] in dropped for n in h3r.values() for v in n["inputs"].values()), \
        "nothing points at a dropped loader"
    assert h3["136"] == h3_t["136"], "no pictures: the template's own wiring"

    # W5: one picture and one video leave exactly ref_image_0, ref_video_0 and ref_video_audio_0
    # on node 136, and no other loader.
    h3v = build_workflow(name="minimax_h3_r2v_hybrid", prompt="P", ref_images=["sheet.png"], ref_videos=["voice.mp4"])
    refs = {k: v for k, v in h3v["136"]["inputs"].items() if k.startswith(("ref_images.", "ref_videos.", "ref_video_audios."))}
    assert refs == {
        "ref_images.ref_image_0": ["137", 0],
        "ref_videos.ref_video_0": ["157", 0],
        "ref_video_audios.ref_video_audio_0": ["157", 2],
    }, refs
    assert h3v["157"]["inputs"]["video"] == "voice.mp4"
    assert (h3v["157"]["inputs"]["force_rate"], h3v["157"]["inputs"]["frame_load_cap"]) == (24, 124)
    loaders = {n for n, v in h3v.items() if v["class_type"] in ("LoadImage", "VHS_LoadVideo")}
    assert loaders == {"137", "157"}, loaders

    # Refmods: the H3 node becomes the cached one, each reference still wired (a lazy fallback),
    # with the job's refmods by slot and the VAE names of its loaders. None means no refmod.
    h3m = build_workflow(name="minimax_h3_r2v_hybrid", prompt="P", ref_images=["s.png", "p.png"], ref_videos=["v.mp4"],
                         refmods={"image": ["s.safetensors", None], "video": ["v.safetensors"]})
    m = h3m["136"]["inputs"]
    assert h3m["136"]["class_type"] == "MiniMaxH3ReferenceToVideoCached"
    assert m["refmods"] == "image 0 s.safetensors\nvideo 0 v.safetensors"
    assert (m["ref_image_0"], m["ref_image_1"], m["ref_video_0"], m["ref_video_audio_0"]) == (["137", 0], ["148", 0], ["157", 0], ["157", 2])
    assert not any("." in k for k in m), "no autogrow keys left"
    assert (m["vae_name"], m["audio_vae_name"]) == (h3_t["119"]["inputs"]["vae_name"], h3_t["120"]["inputs"]["vae_name"])
    assert {k: v for k, v in m.items() if not k.startswith(("ref_", "refmods", "vae_name", "audio_vae_name"))} == \
        {k: v for k, v in h3v["136"]["inputs"].items() if not k.startswith("ref_")}, "every other input as the H3 node had it"
    assert build_workflow(name="minimax_h3_r2v_hybrid", prompt="P", ref_images=["s.png"], refmods={"image": [None]})["136"]["class_type"] \
        == "MiniMaxH3ReferenceToVideo", "no refmod: the H3 node itself"
    # The refmod template: no prompt, the reference video into its loader, the VAE names copied.
    rm, rm_t = build_workflow(name="minimax_h3_refmod", prompt=None, source_video="ref.mp4"), template("minimax_h3_refmod")
    assert rm["3"]["inputs"]["video"] == "ref.mp4"
    assert (rm["4"]["inputs"]["vae_name"], rm["4"]["inputs"]["audio_vae_name"]) == (h3_t["119"]["inputs"]["vae_name"], h3_t["120"]["inputs"]["vae_name"])
    assert {k: v for k, v in rm["3"]["inputs"].items() if k != "video"} == {k: v for k, v in h3_t["157"]["inputs"].items() if k != "video"}, \
        "the refmod's video is loaded exactly as a scene loads it, or its hash never matches"
    # A picture's refmod (a character sheet): the picture into its loader, at the scene's canvas,
    # which is the H3 template's own unless the job sends one.
    rmi, rmi_t = build_workflow(name="minimax_h3_refmod_image", prompt=None, reference_image="sheet_g.png"), template("minimax_h3_refmod_image")
    assert rmi["2"]["inputs"]["image"] == "sheet_g.png"
    assert rmi["4"]["inputs"]["vae_name"] == h3_t["119"]["inputs"]["vae_name"]
    assert rmi_t["3"]["inputs"] == h3_t["115"]["inputs"] and (rmi["4"]["inputs"]["width"], rmi["4"]["inputs"]["height"]) == (["3", 0], ["3", 1]), \
        "the template's canvas is the scene's, or a sheet's hash never matches"
    rmi_sized = build_workflow(name="minimax_h3_refmod_image", prompt=None, reference_image="s.png", width=480, height=640)
    assert (rmi_sized["4"]["inputs"]["width"], rmi_sized["4"]["inputs"]["height"]) == (480, 640)
    assert h3["136"]["inputs"]["ref_image_size"] == "match", "a sheet's refmod resizes as 'match' does"

    # basic_workflow exactly as before: only the nodes a job has always patched change.
    ltx, ltx_t = build_workflow(
        name="basic_workflow", prompt="B", lora_name="x.safetensors", duration_seconds=5,
        width=576, height=768, filename_prefix="p", use_reference_image=False,
    ), template("basic_workflow")
    changed = {n for n in ltx_t if ltx_t[n] != ltx[n]}
    assert changed <= {"2483", "4990", "4979", "3059", "4977", "2004", "4852"}, changed
    assert ltx["4832"] == ltx_t["4832"], "the fixed seed"
    assert ltx["4990"]["inputs"]["strength_model"] == ltx_t["4990"]["inputs"]["strength_model"]
    print("workflow_builder self-check ok")
