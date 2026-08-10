"""The worker's small HTTP surface — port of render-worker/src/server.js.

Always up, on every intake, because container orchestration needs a liveness probe:

  GET  /health   {status, workerId, mock, queue} — the docker-compose healthcheck.
  POST /render   the QUEUE_DRIVER=http intake: chat-api pushes ONE job (guarded by
                 x-internal-token), we ACK 202 and render fire-and-forget, streaming
                 status back over the same webhook the queue path uses.

stdlib only — the handler venv deliberately carries no web framework.
"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config


def serve(process_job, port=None):
    """Start the server on a daemon thread and return it. Non-blocking, so the caller
    goes on to run its intake loop (or RunPod's)."""
    port = config.RENDER_PORT if port is None else port
    pool = ThreadPoolExecutor(max_workers=max(1, config.RENDER_CONCURRENCY), thread_name_prefix="http-render")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
            if self.path.split("?")[0] != "/health":
                return self._send(404, {"error": "not found"})
            self._send(200, {
                "status": "ok",
                "workerId": config.WORKER_ID,
                "mock": config.MOCK_COMFY,
                "queue": config.QUEUE_DRIVER,
            })

        def do_POST(self):  # noqa: N802
            if self.path.split("?")[0] != "/render":
                return self._send(404, {"error": "not found"})
            if (self.headers.get("x-internal-token") or "") != config.INTERNAL_TOKEN:
                return self._send(401, {"error": "bad internal token"})
            try:
                length = int(self.headers.get("content-length") or 0)
                job = json.loads(self.rfile.read(length) or b"{}")
            except Exception:  # noqa: BLE001
                return self._send(400, {"error": "invalid json body"})
            if not job.get("jobId") or not job.get("workflow"):
                return self._send(400, {"error": "jobId and workflow are required"})
            # ACK first: the render takes minutes and chat-api must not wait on it.
            self._send(202, {"ok": True, "jobId": job["jobId"], "accepted": True})
            pool.submit(_guarded, process_job, job)

        def log_message(self, fmt, *args):  # quieter than the default stderr access log
            pass

    def _guarded(fn, job):
        try:
            fn(job)
        except Exception as e:  # noqa: BLE001 - process_job already reported `failed`
            print(f"[http] processJob crashed: {e}", flush=True)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    print(f"[http] listening on 0.0.0.0:{port} (/health, /render)", flush=True)
    return server
