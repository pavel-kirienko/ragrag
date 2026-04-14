"""HTTP dashboard and status endpoint for the ragrag daemon.

A tiny ``http.server.ThreadingHTTPServer`` runs alongside the JSON-RPC
Unix socket and serves:

  - ``GET /``                        → static HTML dashboard
  - ``GET /status``                  → JSON snapshot
  - ``GET /pages/<sha>/<page>.webp`` → cached page image
  - ``GET /log``                     → last ~8 KiB of daemon.log
  - ``POST /shutdown``               → graceful daemon exit (requires
    ``X-Ragrag-Confirm: yes`` header)

The status server is intentionally tiny, single-threaded per request,
and does not share the RPC lock. Handlers only read daemon state, so a
long indexing pass on the RPC side never blocks a dashboard refresh.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlparse

from ragrag.daemon.static import DASHBOARD_HTML


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from ragrag.daemon.server import DaemonServer


class StatusServer:
    """Runs the dashboard HTTP server in a background thread."""

    def __init__(
        self,
        server: "DaemonServer",
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._daemon = server
        self._host = host
        self._requested_port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.actual_port: int = 0

    def start(self) -> int:
        """Bind the server and start serving. Returns the actual port used."""
        handler_factory = _make_handler_class(self._daemon)
        try:
            self._httpd = ThreadingHTTPServer((self._host, self._requested_port), handler_factory)
        except OSError as exc:
            if self._requested_port != 0:
                logger.warning(
                    "HTTP status server: port %d in use (%s); falling back to a free port",
                    self._requested_port, exc,
                )
                self._httpd = ThreadingHTTPServer((self._host, 0), handler_factory)
            else:
                raise
        self.actual_port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
            name="ragrag-http",
        )
        self._thread.start()
        logger.info("Dashboard at http://%s:%d/", self._host, self.actual_port)
        return self.actual_port

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
        self._httpd = None
        self._thread = None


# --------------------------------------------------------------------------- #
# Handler factory
# --------------------------------------------------------------------------- #

def _make_handler_class(daemon: "DaemonServer") -> type[BaseHTTPRequestHandler]:
    """Return a handler class that closes over the daemon reference."""

    class _Handler(BaseHTTPRequestHandler):
        # Quiet access log by default — routed through our logger.
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            logger.debug("http %s " + format, self.address_string(), *args)

        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, default=_json_default).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str) -> None:
            try:
                data = path.read_bytes()
            except OSError:
                self._send_text("not found", status=404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=300")
            self.end_headers()
            self.wfile.write(data)

        # -- GET ------------------------------------------------------ #

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/" or path == "/index.html":
                self._send_text(DASHBOARD_HTML, content_type="text/html; charset=utf-8")
                return

            if path == "/status":
                payload = daemon.dispatcher._status({})  # noqa: SLF001
                payload.update(_collect_resources(daemon))
                self._send_json(payload)
                return

            if path.startswith("/pages/"):
                # /pages/<sha>/<page>.webp
                parts = path.split("/")
                if len(parts) == 4 and parts[-1].endswith(".webp"):
                    sha = parts[2]
                    try:
                        page = int(parts[3].removesuffix(".webp"))
                    except ValueError:
                        self._send_text("bad page", status=400)
                        return
                    cache_path = _find_cached_page(daemon, sha, page)
                    if cache_path is not None:
                        self._send_file(cache_path, "image/webp")
                        return
                self._send_text("not found", status=404)
                return

            if path == "/log":
                log_path = daemon.pid_path.parent / "daemon.log"
                try:
                    with open(log_path, "rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - 8192))
                        tail = f.read()
                    self._send_text(tail.decode("utf-8", errors="replace"))
                except OSError:
                    self._send_text("no log yet", status=404)
                return

            self._send_text("not found", status=404)

        # -- POST ----------------------------------------------------- #

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/shutdown":
                if self.headers.get("X-Ragrag-Confirm") != "yes":
                    self._send_text("confirmation header missing", status=400)
                    return
                daemon.shutdown_event.set()
                self._send_json({"ack": True})
                return
            self._send_text("not found", status=404)

    return _Handler


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def _collect_resources(daemon: "DaemonServer") -> dict[str, Any]:
    """Gather CPU / RAM / GPU / VRAM utilisation for the dashboard."""
    snapshot: dict[str, Any] = {"resources": {}}

    try:
        import psutil

        proc = psutil.Process()
        res = snapshot["resources"]
        res["cpu_pct"] = round(proc.cpu_percent(interval=None), 1)
        res["mem_rss_mib"] = proc.memory_info().rss // (1024 * 1024)
        vm = psutil.virtual_memory()
        res["system_mem_total_mib"] = vm.total // (1024 * 1024)
        res["system_mem_available_mib"] = vm.available // (1024 * 1024)
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            snapshot["resources"]["vram_free_mib"] = free // (1024 * 1024)
            snapshot["resources"]["vram_total_mib"] = total // (1024 * 1024)
            snapshot["resources"]["vram_used_mib"] = (total - free) // (1024 * 1024)
    except Exception:
        pass

    gpu_util = _gpu_utilisation()
    if gpu_util is not None:
        snapshot["resources"]["gpu_pct"] = gpu_util

    return snapshot


_GPU_UTIL_CACHE = {"ts": 0.0, "value": None}


def _gpu_utilisation() -> float | None:
    """Shell out to ``nvidia-smi`` for GPU util. 5 s cache, best-effort."""
    now = time.time()
    if now - _GPU_UTIL_CACHE["ts"] < 5 and _GPU_UTIL_CACHE["value"] is not None:
        return _GPU_UTIL_CACHE["value"]
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        if result.returncode == 0 and result.stdout:
            value = float(result.stdout.strip().splitlines()[0])
            _GPU_UTIL_CACHE["ts"] = now
            _GPU_UTIL_CACHE["value"] = value
            return value
    except Exception:
        pass
    _GPU_UTIL_CACHE["ts"] = now
    _GPU_UTIL_CACHE["value"] = None
    return None


def _find_cached_page(daemon: "DaemonServer", sha: str, page: int) -> Path | None:
    """Walk every engine's page cache looking for a matching WebP."""
    for engine in daemon.engine_cache._engines.values():  # noqa: SLF001
        ingest = getattr(engine, "ingest_manager", None)
        if ingest is None:
            continue
        cache = getattr(ingest, "page_cache", None)
        if cache is None:
            continue
        try:
            candidate = cache.get(sha, page)
        except Exception:
            candidate = None
        if candidate is not None:
            return candidate
    return None
