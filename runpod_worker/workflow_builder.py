"""Build a ComfyUI API-format graph from a workflow template + its meta sidecar.

A faithful Python port of the render-worker's workflowLoader.js: load
`workflows/<name>.json`, patch the nodes the meta sidecar names (prompt, lora,
frame count, source video, reference image, save prefix), return the graph.
Note: like the Node version, this deliberately does NOT touch the seed nodes —
each turn's `positive` prompt differs, so ComfyUI's input-hash cache never collides.
"""
import json
import math
import os
import re

WORKFLOWS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")


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
    filename_prefix=None,
    source_video=None,
):
    safe = _safe_name(name)
    with open(os.path.join(WORKFLOWS_DIR, f"{safe}.json")) as f:
        workflow = json.load(f)
    meta = _load_meta(safe)

    pos = meta.get("positivePromptNode")
    if pos not in workflow:
        raise ValueError(f"Workflow '{safe}' has no positive-prompt node '{pos}'.")
    workflow[pos]["inputs"]["text"] = prompt

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

    save_node = meta.get("saveVideoNode")
    if filename_prefix and save_node and save_node in workflow:
        workflow[save_node]["inputs"]["filename_prefix"] = filename_prefix

    return workflow
