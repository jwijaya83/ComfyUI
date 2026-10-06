"""Render-batching experiment harness — Stage 1 of ai-chat docs/RenderBatchingDesign.md.

NOT product code. Run it on the GPU you intend to buy (a rented 5090 — a 4080 result does
not carry over), against the ComfyUI inside this image (127.0.0.1:8188), with no other
work on the card.

It answers one question per run: how many GPU-seconds does one CLIP cost? It submits a
graph, times it end to end, samples VRAM + utilisation with nvidia-smi while it runs, and
saves every output so quality can be judged blind, side by side.

    E0  batch 1, reuse tier, 10 s, ×10         (built here from the real workflow)
        python bench/batch_bench.py --run E0 --workflow latent_injection --seconds 10 \\
            --count 10 --source-video seed.mp4 --prompts bench/prompts.txt
    E4  the same for the full tier (no source video)
        python bench/batch_bench.py --run E4 --workflow basic_workflow --seconds 10 --count 10 \\
            --prompts bench/prompts.txt

    E1/E2/E3  batch 2 and 4. A batched LTX graph (batch dimension > 1, per-item
        conditioning) is built by hand in the ComfyUI editor — that is the experiment —
        and exported with "Save (API format)". Pass it with --graph and say how many clips
        one execution yields:
        python bench/batch_bench.py --run E2-b2 --graph bench/e2_batch2_api.json --items 2 --count 5

Results append to bench/results.jsonl, and a markdown row per run is printed for the
"Results" table in RenderBatchingDesign.md. The GATE: proceed to Stage 2 only if E2
batch-2 gives >= 1.4x clips per GPU-second over E0 with no visible quality loss in a blind
side-by-side of 10 pairs.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from comfy_client import (  # noqa: E402
    collect_outputs,
    download_output,
    get_history,
    submit_prompt,
    upload_video,
    wait_for_ready,
    watch_prompt,
)
from workflow_builder import build_workflow  # noqa: E402


class GpuSampler:
    """Polls nvidia-smi every 500 ms while a graph runs: peak VRAM (MiB), mean util (%)."""

    def __init__(self):
        self.mem, self.util, self._stop = [], [], threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().splitlines()[0]
                m, u = (float(x) for x in out.split(","))
                self.mem.append(m)
                self.util.append(u)
            except Exception:  # noqa: BLE001 - no GPU / no nvidia-smi: record nothing
                pass
            self._stop.wait(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._t.join(timeout=2)

    def summary(self):
        return {
            "peak_vram_mib": max(self.mem) if self.mem else None,
            "mean_gpu_util": round(sum(self.util) / len(self.util), 1) if self.util else None,
        }


def run_graph(graph, out_dir, label):
    prompt_id, client_id = submit_prompt(graph)
    with GpuSampler() as gpu:
        started = time.monotonic()
        watch_prompt(client_id, prompt_id)
        elapsed = time.monotonic() - started
    files = collect_outputs(get_history(prompt_id))
    saved = []
    for i, f in enumerate(o for o in files if o.get("kind") == "videos"):
        path = os.path.join(out_dir, f"{label}_{i}.mp4")
        with open(path, "wb") as fh:
            fh.write(download_output(f))
        saved.append(path)
    return elapsed, gpu.summary(), saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="label, e.g. E0, E2-b2, E4")
    ap.add_argument("--count", type=int, default=10, help="executions of the graph")
    ap.add_argument("--items", type=int, default=1, help="clips one execution yields (the batch size)")
    ap.add_argument("--graph", help="a hand-built API-format graph (E1-E3); used as-is")
    ap.add_argument("--workflow", help="build batch-1 from workflows/<name> (E0/E4)")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--width", type=int, default=576)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--source-video", help="seed clip for latent injection (reuse tier)")
    ap.add_argument("--prompts", help="one prompt per line; cycled")
    ap.add_argument("--lora")
    args = ap.parse_args()
    if not args.graph and not args.workflow:
        ap.error("pass --graph (a batched API graph) or --workflow (build batch 1)")

    wait_for_ready(timeout=600)
    out_dir = os.path.join(HERE, "out", args.run)
    os.makedirs(out_dir, exist_ok=True)
    prompts = ["A woman talks to camera in a quiet cafe, warm light."]
    if args.prompts:
        with open(args.prompts) as fh:
            prompts = [line.strip() for line in fh if line.strip()] or prompts
    source = None
    if args.source_video:
        with open(args.source_video, "rb") as fh:
            source = upload_video(fh.read(), filename=os.path.basename(args.source_video))
    fixed_graph = json.load(open(args.graph)) if args.graph else None

    # One warm-up execution, not timed: the first graph pays the model load, and a cold
    # start is a separate number (UnitEconomics records it per job) from throughput.
    runs = []
    for i in range(args.count + 1):
        graph = fixed_graph or build_workflow(
            name=args.workflow, prompt=prompts[i % len(prompts)], use_reference_image=False,
            lora_name=args.lora, duration_seconds=args.seconds, width=args.width, height=args.height,
            source_video=source, filename_prefix=f"bench/{args.run}",
        )
        elapsed, gpu, saved = run_graph(graph, out_dir, f"{i:02d}")
        label = "warm-up" if i == 0 else f"{i}/{args.count}"
        print(f"[{args.run}] {label}: {elapsed:.1f}s for {args.items} clip(s) · {gpu} · {len(saved)} saved", flush=True)
        if i:
            runs.append({"elapsed": elapsed, **gpu, "outputs": saved})

    secs = sorted(r["elapsed"] for r in runs)
    per_clip = sum(secs) / len(secs) / args.items
    result = {
        "run": args.run, "items": args.items, "count": args.count, "seconds": args.seconds,
        "width": args.width, "height": args.height, "graph": args.graph, "workflow": args.workflow,
        "seconds_per_execution_p50": secs[len(secs) // 2],
        "gpu_seconds_per_clip": round(per_clip, 2),
        "clips_per_gpu_second": round(1 / per_clip, 4),
        "peak_vram_mib": max((r["peak_vram_mib"] or 0) for r in runs) or None,
        "mean_gpu_util": round(sum((r["mean_gpu_util"] or 0) for r in runs) / len(runs), 1),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(HERE, "results.jsonl"), "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))
    print(
        f"\n| {args.run} | batch {args.items} | {args.seconds:g}s | {result['gpu_seconds_per_clip']}s/clip | "
        f"{result['clips_per_gpu_second']} clips/GPU-s | {result['peak_vram_mib']} MiB | {result['mean_gpu_util']}% util |"
    )
    print("Speed-up vs E0 = this run's clips_per_gpu_second ÷ E0's (results.jsonl). Gate: >= 1.4 on E2 batch 2.")


if __name__ == "__main__":
    main()
