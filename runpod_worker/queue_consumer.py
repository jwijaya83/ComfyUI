"""QUEUE_DRIVER=redis — drain ai-chat's MAIN render queue.

Python port of render-worker/src/consumer.js, with the same guarantees. Many workers
share ONE consumer group (`renderers`) across the per-lane streams (render:high,
render:low); each job is delivered to exactly one of them. That is what makes this
image horizontally scalable: point REDIS_URL at production and the box joins the pool —
no registration, no config on the chat-api side, no redeploy. Stop it and its share of
the work is simply picked up by whoever is left.

  Priority      render:high is read before render:low on every pass (paid/admin lane
                first); when everything is idle we BLOCK across all lanes so the loop
                doesn't spin.
  At-least-once A job stays in the group's pending list until XACK, which happens
                AFTER the render. process_job is crash-safe (it reports `failed` on its
                own exhaustion and returns), so the only redelivery path is a real
                worker crash before the ACK — and render_jobs + idempotency_key make
                that safe.
  Crash recovery XAUTOCLAIM reclaims jobs idle longer than RENDER_VISIBILITY_MS (a dead
                worker). Past RENDER_MAX_DELIVERIES a poison job is dead-lettered to
                render:dead and marked failed, so the UI never hangs on it.
  Concurrency   RENDER_CONCURRENCY jobs at a time (keep at 1 per GPU). The GLOBAL render
                ceiling is (instances) x RENDER_CONCURRENCY; demand above it waits in
                the stream — latency, not cost.
"""
import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import config
from reporter import report

_RECLAIM_INTERVAL_S = 30


def _stream_for(lane):
    return f"{config.QUEUE_STREAM_PREFIX}:{lane}"


def _job_from(fields):
    """Stream entries carry the job as a single JSON `job` field (chat-api's
    renderQueue.js: XADD … 'job' <json>)."""
    raw = (fields or {}).get("job")
    if not raw:
        raise ValueError("stream entry has no 'job' field")
    return json.loads(raw), raw


class _Slots:
    """In-flight counter guarding RENDER_CONCURRENCY across the worker threads."""

    def __init__(self, limit):
        self.limit = limit
        self._used = 0
        self._lock = threading.Lock()

    @property
    def free(self):
        with self._lock:
            return self.limit - self._used

    def take(self):
        with self._lock:
            self._used += 1

    def give_back(self):
        with self._lock:
            self._used -= 1


def start_consumer(process_job):
    """Block forever draining the lane streams. `process_job(job)` is the shared render
    entry point (handler.process_job) — identical to what the runpod/http intakes call."""
    import redis as redis_lib

    # socket_timeout MUST exceed the blocking read's BLOCK window. redis-py applies a
    # default (5s in redis-py 8.x) which the idle XREADGROUP ... BLOCK RENDER_BLOCK_MS
    # then races: the socket times out at the same moment the server is about to reply
    # nil, so every idle cycle raised TimeoutError, hit the loop's error branch and slept
    # a second. Be explicit rather than inheriting whatever the installed version defaults
    # to — the margin also keeps the connection from tripping on a slow broker.
    client = redis_lib.Redis.from_url(
        config.REDIS_URL,
        decode_responses=True,
        socket_timeout=config.RENDER_BLOCK_MS / 1000.0 + 15,
        socket_connect_timeout=10,
        health_check_interval=30,
    )
    # MUST be unique per instance: every container is PID 1, so pid alone collides.
    consumer_name = os.environ.get("CONSUMER_NAME") or f"{config.WORKER_ID}-{socket.gethostname()}-{os.getpid()}"

    # Ensure the group exists on each lane (MKSTREAM creates an empty stream).
    for lane in config.RENDER_LANES:
        try:
            client.xgroup_create(_stream_for(lane), config.RENDER_GROUP, id="$", mkstream=True)
        except redis_lib.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    print(
        f"comfy-worker '{config.WORKER_ID}' consuming [{', '.join(config.RENDER_LANES)}] "
        f"group={config.RENDER_GROUP} consumer={consumer_name} "
        f"concurrency={config.RENDER_CONCURRENCY} -> chat-api {config.CHAT_API_URL}",
        flush=True,
    )

    slots = _Slots(config.RENDER_CONCURRENCY)
    pool = ThreadPoolExecutor(max_workers=config.RENDER_CONCURRENCY, thread_name_prefix="render")

    def run(lane, entry_id, fields):
        """Render one entry, then ACK. Any failure inside process_job has already been
        reported as `failed`, so the ACK is unconditional — a job we can't parse must not
        be redelivered forever."""
        slots.take()

        def task():
            try:
                job, _ = _job_from(fields)
                process_job(job)
            except Exception as e:  # noqa: BLE001 - never let one job kill the loop
                print(f"[consumer] {lane} {entry_id}: {e}", flush=True)
            finally:
                try:
                    client.xack(_stream_for(lane), config.RENDER_GROUP, entry_id)
                except Exception:  # noqa: BLE001 - visibility timeout covers a missed ACK
                    pass
                slots.give_back()

        pool.submit(task)

    def deliveries_for(stream, entry_id):
        """How many times this entry has been delivered (the poison-job counter)."""
        try:
            pending = client.xpending_range(stream, config.RENDER_GROUP, min=entry_id, max=entry_id, count=1)
            if pending:
                return int(pending[0].get("times_delivered") or 1)
        except Exception:  # noqa: BLE001
            pass
        return 1

    def dead_letter(lane, stream, entry_id, fields, deliveries):
        raw = (fields or {}).get("job") or ""
        try:
            client.xadd(config.RENDER_DEAD_STREAM, {"lane": lane, "origId": str(entry_id), "job": raw})
        except Exception:  # noqa: BLE001
            pass
        try:
            client.xack(stream, config.RENDER_GROUP, entry_id)
        except Exception:  # noqa: BLE001
            pass
        try:
            job_id = json.loads(raw).get("jobId")
            if job_id:
                report({
                    "jobId": job_id,
                    "status": "failed",
                    "error": f"dead-lettered after {deliveries} delivery attempts",
                })
        except Exception:  # noqa: BLE001 - best effort
            pass
        print(f"[consumer] ☠ dead-letter {lane} {entry_id} after {deliveries} deliveries", flush=True)

    def reclaim(available):
        """Take over jobs whose owning consumer died (idle > visibility timeout)."""
        for lane in config.RENDER_LANES:
            if available <= 0:
                break
            stream = _stream_for(lane)
            try:
                res = client.xautoclaim(
                    stream, config.RENDER_GROUP, consumer_name,
                    min_idle_time=config.RENDER_VISIBILITY_MS, start_id="0-0", count=available,
                )
            except Exception:  # noqa: BLE001 - stream/group not there yet
                continue
            # Redis 7 returns (next_cursor, messages, deleted); Redis 6 omits the third.
            entries = res[1] if isinstance(res, (list, tuple)) and len(res) > 1 else []
            for entry_id, fields in entries or []:
                if available <= 0:
                    break
                if not fields:  # entry deleted since it was pended
                    try:
                        client.xack(stream, config.RENDER_GROUP, entry_id)
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                deliveries = deliveries_for(stream, entry_id)
                if deliveries > config.RENDER_MAX_DELIVERIES:
                    dead_letter(lane, stream, entry_id, fields, deliveries)
                else:
                    run(lane, entry_id, fields)
                    available -= 1

    def read_new(count):
        """High lane first (non-blocking); if every lane is empty, block across all."""
        for lane in config.RENDER_LANES:
            res = client.xreadgroup(
                config.RENDER_GROUP, consumer_name, {_stream_for(lane): ">"}, count=count,
            )
            if res and res[0][1]:
                return [(lane, res[0][1])]
        try:
            res = client.xreadgroup(
                config.RENDER_GROUP, consumer_name,
                {_stream_for(lane): ">" for lane in config.RENDER_LANES},
                count=count, block=config.RENDER_BLOCK_MS,
            )
        except redis_lib.exceptions.TimeoutError:
            return None  # idle window elapsed — not an error, just no work
        if not res:
            return None
        return [(stream[len(config.QUEUE_STREAM_PREFIX) + 1:], entries) for stream, entries in res]

    last_reclaim = 0.0
    while True:
        try:
            available = slots.free
            if available <= 0:
                time.sleep(0.05)
                continue

            now = time.time()
            if now - last_reclaim > _RECLAIM_INTERVAL_S:
                last_reclaim = now
                reclaim(available)
                continue

            batches = read_new(available)
            if not batches:
                continue
            for lane, entries in batches:
                for entry_id, fields in entries:
                    run(lane, entry_id, fields)
        except Exception as e:  # noqa: BLE001 - a broker blip must not kill the worker
            print(f"[consumer] loop error: {e}", flush=True)
            time.sleep(1)
