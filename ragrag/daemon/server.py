"""Ragrag daemon server.

Long-lived process that owns loaded models and per-index ``SearchEngine``
instances. Speaks JSON-RPC 2.0 over a Unix domain socket at
``<index>/.ragrag/daemon.sock``.

Architecture:

  - one acceptor thread accepts connections from the Unix socket
  - a small thread pool services individual connections (parses one
    request, runs the dispatcher, writes one response, closes)
  - ``threading.RLock`` serialises access to the search pipeline because the
    GPU forward pass is single-tenant
  - models are loaded **lazily** on the first request that needs them, so
    starting the daemon for a quick ``status`` call does not pay the
    20-second cost
  - an idle-timeout thread tears the daemon down after
    ``Settings.daemon_idle_timeout_s`` seconds of inactivity (12 h default)
  - SIGTERM / SIGINT trigger graceful shutdown; SIGHUP reloads config;
    SIGUSR1 dumps a status JSON to ``daemon.log``

The protocol surface (methods accepted) lives in ``Dispatcher.methods``.
"""
from __future__ import annotations

import os

# Must be set before any transitive torch import. Reduces CUDA allocator
# fragmentation during the two-pass indexing swap.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import errno
import fcntl
import json
import logging
import signal
import socket
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ragrag import __version__
from ragrag.config import Settings, find_index_root, get_settings
from ragrag.daemon import rpc
from ragrag.daemon.rpc import (
    ERROR_INTERNAL,
    ERROR_INVALID_PARAMS,
    ERROR_METHOD_NOT_FOUND,
    JsonRpcError,
    PROTOCOL_VERSION,
    Request,
    Response,
    decode_request,
    encode_response,
    error_response,
)


logger = logging.getLogger("ragrag.daemon")


# --------------------------------------------------------------------------- #
# Status / state objects
# --------------------------------------------------------------------------- #

@dataclass
class RecentQuery:
    query: str
    wall_ms: int
    status: str
    top1_path: str | None
    timestamp: float


@dataclass
class DaemonState:
    """Mutable runtime state owned by the daemon, surfaced via ``status``.

    Everything here is read by the HTTP status server (Phase E) without
    locking — fields are simple and the worst case is a slightly stale
    snapshot.
    """

    started_at: float
    device_mode: str  # "cuda" | "cpu" | "mps"
    last_request_at: float = 0.0
    recent_queries: deque[RecentQuery] = field(default_factory=lambda: deque(maxlen=20))
    indexing: dict[str, Any] = field(default_factory=dict)
    open_indexes: dict[str, dict[str, Any]] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Lazy SearchEngine cache (one per index path)
# --------------------------------------------------------------------------- #

class EngineCache:
    """Holds at most one ``SearchEngine`` per index path.

    The first request for a given path constructs the embedder, the store,
    and the engine. Subsequent requests for the same path reuse them. The
    cache is intentionally tiny — there is one daemon per index directory,
    so usually only one engine ever lives here.
    """

    def __init__(self, settings_factory: Callable[[str], Settings]) -> None:
        self._settings_factory = settings_factory
        self._lock = threading.RLock()
        self._engines: dict[str, Any] = {}

    def get(self, index_path: str) -> Any:
        with self._lock:
            cached = self._engines.get(index_path)
            if cached is not None:
                return cached
            engine = self._build_engine(index_path)
            self._engines[index_path] = engine
            return engine

    def _build_engine(self, index_path: str):
        # Imported lazily so that ``ragrag daemon --idle`` (no model needed)
        # boots in under a second for tests.
        from ragrag.embedding.colqwen_embedder import ColQwenEmbedder
        from ragrag.embedding.vlm_loader import load_vlm
        from ragrag.extractors.vlm_topic_client import VLMTopicClient
        from ragrag.index.ingest_manager import IngestManager
        from ragrag.index.qdrant_store import COLLECTION_NAME, QdrantStore
        from ragrag.retrieval.search_engine import SearchEngine

        settings = self._settings_factory(index_path)
        logger.info("Constructing ColQwen3 handle for index %s (deferred load) ...", index_path)
        embedder = ColQwenEmbedder(
            settings.model_id,
            settings.max_visual_tokens,
            quantization=settings.quantization,
            defer_load=True,
        )

        # VLM topic client loads LAZILY per indexing pass. Keeping it in
        # VRAM across calls would break search on 8 GB cards (ColQwen3 at
        # 2.5 GB + Qwen2.5-VL-3B at 2.5 GB + activations > 8 GB).
        def _vlm_factory(device: str | None = None) -> VLMTopicClient:
            logger.info(
                "Loading VLM topic client '%s' (device=%s) ...",
                settings.vlm_model_id, device or "auto",
            )
            handle = load_vlm(
                settings.vlm_model_id,
                quantization=settings.vlm_quantization,
                device=device,
            )
            return VLMTopicClient(handle, image_max_side=settings.chunker_vlm_image_max_side)

        store = QdrantStore(settings.index_path, COLLECTION_NAME, embedder.embedding_dim)
        ingest = IngestManager(
            embedder, store, settings, vlm_factory=_vlm_factory,
        )

        reranker = None
        if settings.reranker_model and settings.reranker_model.lower() not in {"none", ""}:
            try:
                from ragrag.retrieval.reranker import VLMReranker

                reranker = VLMReranker(settings)
                logger.info("VLM reranker armed (model=%s)", settings.vlm_model_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reranker init failed: %s — falling back to MaxSim-only", exc)
                reranker = None

        engine = SearchEngine(embedder, store, ingest, settings, reranker=reranker)
        logger.info("Engine ready for %s", index_path)
        return engine

    def unload_all(self) -> None:
        with self._lock:
            self._engines.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover — torch is mandatory in deployment
            pass


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

class Dispatcher:
    """Routes RPC method names to handler functions.

    Holds the daemon-wide ``RLock`` so only one search/index runs at a time
    (the GPU is not multi-tenant). Read-only methods like ``status`` and
    ``shutdown`` skip the lock to stay responsive even during a long index.
    """

    def __init__(
        self,
        engine_cache: EngineCache,
        state: DaemonState,
        shutdown_event: threading.Event,
    ) -> None:
        self.engine_cache = engine_cache
        self.state = state
        self.shutdown_event = shutdown_event
        self.lock = threading.RLock()
        self.methods: dict[str, Callable[[dict[str, Any]], Any]] = {
            "search": self._search,
            "index": self._index,
            "status": self._status,
            "shutdown": self._shutdown,
            "reload_config": self._reload_config,
        }

    # ---- helpers ---------------------------------------------------------- #

    def _require_param(self, params: dict[str, Any], name: str) -> Any:
        if name not in params:
            raise JsonRpcError(ERROR_INVALID_PARAMS, f"missing required param: {name}")
        return params[name]

    def _resolve_index_path(self, params: dict[str, Any]) -> str:
        # Each request must say which index it targets. The daemon process
        # itself was started with one --index-path, but we let clients
        # override per-request so a future multi-index daemon is possible.
        path = params.get("index_path")
        if not path or not isinstance(path, str):
            raise JsonRpcError(ERROR_INVALID_PARAMS, "missing or non-string index_path")
        return os.path.abspath(path)

    # ---- methods --------------------------------------------------------- #

    def _search(self, params: dict[str, Any]) -> Any:
        from ragrag.models import SearchRequest

        index_path = self._resolve_index_path(params)
        engine = self.engine_cache.get(index_path)
        request = SearchRequest(
            query=self._require_param(params, "query"),
            paths=params.get("paths") or [index_path],
            top_k=int(params.get("top_k", 10)),
            include_markdown=bool(params.get("include_markdown", False)),
        )
        with self.lock:
            self.state.last_request_at = time.time()
            t0 = time.time()
            response = engine.search(request)
            wall_ms = int((time.time() - t0) * 1000)
            top1_path = response.results[0].path if response.results else None
            self.state.recent_queries.appendleft(
                RecentQuery(
                    query=request.query,
                    wall_ms=wall_ms,
                    status=response.status,
                    top1_path=top1_path,
                    timestamp=time.time(),
                )
            )
        return response

    def _index(self, params: dict[str, Any]) -> Any:
        from ragrag.models import IndexingStats

        index_path = self._resolve_index_path(params)
        engine = self.engine_cache.get(index_path)
        paths = params.get("paths") or [index_path]
        with self.lock:
            self.state.last_request_at = time.time()
            stats, skipped, _files = engine.ingest_manager.ingest_paths(list(paths))
        return {
            "stats": stats.model_dump() if hasattr(stats, "model_dump") else stats.__dict__,
            "skipped": [s.model_dump() if hasattr(s, "model_dump") else s.__dict__ for s in skipped],
        }

    def _status(self, params: dict[str, Any]) -> dict[str, Any]:
        # No lock — read-only snapshot. Slightly stale is fine.
        uptime = time.time() - self.state.started_at
        idle = time.time() - self.state.last_request_at if self.state.last_request_at else uptime
        return {
            "version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "device_mode": self.state.device_mode,
            "started_at": self.state.started_at,
            "uptime_s": round(uptime, 1),
            "idle_s": round(idle, 1),
            "last_request_at": self.state.last_request_at,
            "models_loaded": list(self.engine_cache._engines.keys()),  # noqa: SLF001
            "indexing": dict(self.state.indexing),
            "recent_queries": [
                {
                    "query": q.query,
                    "wall_ms": q.wall_ms,
                    "status": q.status,
                    "top1_path": q.top1_path,
                    "timestamp": q.timestamp,
                }
                for q in self.state.recent_queries
            ],
        }

    def _shutdown(self, params: dict[str, Any]) -> dict[str, Any]:
        # Ack first, exit on the next idle-checker tick.
        logger.info("Shutdown requested")
        self.shutdown_event.set()
        return {"ack": True}

    def _reload_config(self, params: dict[str, Any]) -> dict[str, Any]:
        # Clearing the cache forces the next request to rebuild engines
        # from a freshly-read config.
        self.engine_cache.unload_all()
        return {"reloaded": True}

    # ---- entry point ----------------------------------------------------- #

    def dispatch(self, request: Request) -> Response:
        handler = self.methods.get(request.method)
        if handler is None:
            return error_response(
                request.id,
                JsonRpcError(ERROR_METHOD_NOT_FOUND, f"unknown method: {request.method}"),
            )
        try:
            result = handler(request.params)
        except JsonRpcError as exc:
            return error_response(request.id, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("RPC handler %s raised: %s\n%s", request.method, exc, traceback.format_exc())
            return error_response(request.id, JsonRpcError(ERROR_INTERNAL, str(exc)))
        return Response(id=request.id, result=result)


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #

def _detect_device_mode() -> str:
    try:
        import torch
    except Exception:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _read_one_line(conn: socket.socket, max_bytes: int = 16 * 1024 * 1024) -> bytes:
    """Read until ``\\n`` or close. Capped to defend against unbounded clients."""
    chunks: list[bytes] = []
    total = 0
    while total < max_bytes:
        chunk = conn.recv(8192)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if b"\n" in chunk:
            break
    line = b"".join(chunks)
    nl = line.find(b"\n")
    if nl >= 0:
        return line[:nl]
    return line


class DaemonServer:
    def __init__(
        self,
        index_path: str,
        socket_path: Path,
        pid_path: Path,
        idle_timeout_s: float,
        http_port: int = 0,
    ) -> None:
        self.index_path = index_path
        self.socket_path = socket_path
        self.pid_path = pid_path
        self.idle_timeout_s = float(idle_timeout_s)
        self.http_port = int(http_port)
        self.shutdown_event = threading.Event()
        self.state = DaemonState(
            started_at=time.time(),
            device_mode=_detect_device_mode(),
            last_request_at=time.time(),
        )
        self.engine_cache = EngineCache(settings_factory=lambda p: get_settings(p))
        self.dispatcher = Dispatcher(self.engine_cache, self.state, self.shutdown_event)
        self._sock: socket.socket | None = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ragrag-rpc")
        self._status_server = None  # lazily constructed in run()

    def _bind(self) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(self.socket_path))
        self.socket_path.chmod(0o600)
        self._sock.listen(16)
        self._sock.settimeout(1.0)  # so accept loop wakes up to check shutdown

    def _write_pid_file(self) -> None:
        self.pid_path.write_text(
            f"{os.getpid()}\n{self.state.started_at}\n{PROTOCOL_VERSION}\n{self.http_port}\n",
            encoding="utf-8",
        )

    def _cleanup(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()
        except Exception:
            pass
        try:
            if self.pid_path.exists():
                self.pid_path.unlink()
        except Exception:
            pass
        try:
            if self._status_server is not None:
                self._status_server.stop()
        except Exception:
            pass
        self.engine_cache.unload_all()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _handle_connection(self, conn: socket.socket) -> None:
        try:
            line = _read_one_line(conn)
            if not line:
                return
            try:
                request = decode_request(line)
            except JsonRpcError as exc:
                response = error_response(None, exc)
            else:
                response = self.dispatcher.dispatch(request)
            try:
                conn.sendall(encode_response(response))
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _idle_checker(self) -> None:
        while not self.shutdown_event.is_set():
            time.sleep(min(60.0, self.idle_timeout_s / 60))
            if self.shutdown_event.is_set():
                return
            idle = time.time() - self.state.last_request_at
            if idle >= self.idle_timeout_s:
                logger.info("Idle for %.0fs ≥ %.0fs, exiting", idle, self.idle_timeout_s)
                self.shutdown_event.set()

    def _install_signals(self) -> None:
        # signal.signal() only works on the main thread of the main interpreter.
        # When the daemon runs inside a test as a thread we silently skip — the
        # test drives shutdown via shutdown_event directly.
        if threading.current_thread() is not threading.main_thread():
            return

        def _term(signum, _frame):
            logger.info("Signal %d received, shutting down", signum)
            self.shutdown_event.set()

        def _hup(_signum, _frame):
            logger.info("SIGHUP received, reloading engines")
            self.engine_cache.unload_all()

        def _usr1(_signum, _frame):
            try:
                snapshot = self.dispatcher._status({})  # noqa: SLF001
                logger.info("STATUS: %s", json.dumps(snapshot))
            except Exception as exc:
                logger.warning("SIGUSR1 status dump failed: %s", exc)

        signal.signal(signal.SIGTERM, _term)
        signal.signal(signal.SIGINT, _term)
        try:
            signal.signal(signal.SIGHUP, _hup)
            signal.signal(signal.SIGUSR1, _usr1)
        except (AttributeError, ValueError):  # pragma: no cover — Windows
            pass

    def run(self) -> int:
        self._bind()
        # Start the HTTP status server before writing the pid file so the
        # actual bound port lands in daemon.pid line 4.
        try:
            from ragrag.daemon.http_status import StatusServer

            settings = get_settings(self.index_path)
            self._status_server = StatusServer(
                self,
                host=settings.daemon_status_host,
                port=self.http_port or settings.daemon_status_port,
            )
            self.http_port = self._status_server.start()
        except Exception as exc:  # noqa: BLE001
            logger.warning("HTTP status server failed to start: %s", exc)
            self._status_server = None
        self._write_pid_file()
        self._install_signals()
        idle_thread = threading.Thread(target=self._idle_checker, daemon=True, name="ragrag-idle")
        idle_thread.start()
        logger.info(
            "Daemon listening on %s (device=%s, idle_timeout=%.0fs, dashboard=http://%s:%d/)",
            self.socket_path, self.state.device_mode, self.idle_timeout_s,
            settings.daemon_status_host if self._status_server else "?",
            self.http_port if self._status_server else 0,
        )
        try:
            assert self._sock is not None
            while not self.shutdown_event.is_set():
                try:
                    conn, _ = self._sock.accept()
                except socket.timeout:
                    continue
                except OSError as exc:
                    if exc.errno in (errno.EBADF, errno.EINVAL):
                        break
                    raise
                self._executor.submit(self._handle_connection, conn)
        finally:
            self._cleanup()
        return 0


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def _resolve_paths(index_path: str | None) -> tuple[str, Path, Path]:
    """Normalise the user-supplied path into ``(index_root, sock_path, pid_path)``.

    Accepts either the knowledge-base root or the ``.ragrag`` directory inside
    it; the two are equivalent from the daemon's point of view.
    """
    if index_path is None:
        try:
            root, _settings = find_index_root()
        except SystemExit:
            root = os.getcwd()
        index_path = root
    abs_path = Path(os.path.abspath(index_path))
    if abs_path.name == ".ragrag":
        ragrag_dir = abs_path
        abs_index = str(abs_path.parent)
    else:
        ragrag_dir = abs_path / ".ragrag"
        abs_index = str(abs_path)
    ragrag_dir.mkdir(parents=True, exist_ok=True)
    return abs_index, ragrag_dir / "daemon.sock", ragrag_dir / "daemon.pid"


def _double_fork() -> bool:
    """Detach from the controlling terminal. Returns True in the child."""
    pid = os.fork()
    if pid > 0:
        return False  # parent
    os.setsid()
    pid = os.fork()
    if pid > 0:
        os._exit(0)
    # Child: redirect stdio to /dev/null
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        try:
            os.dup2(devnull, fd)
        except OSError:
            pass
    return True


def _setup_logging(log_path: Path, level: str) -> None:
    handlers: list[logging.Handler] = []
    try:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handlers.append(fh)
    except Exception:
        pass
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handlers.append(sh)
    logging.basicConfig(level=getattr(logging, level), handlers=handlers, force=True)
    for noisy in ("transformers", "huggingface_hub", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragrag.daemon")
    parser.add_argument("--index-path", default=None, help="Index root directory")
    parser.add_argument("--detach", action="store_true", help="Fork into the background")
    parser.add_argument("--idle-timeout", type=float, default=None, help="Override idle timeout (seconds)")
    parser.add_argument("--http-port", type=int, default=0, help="Reserved for the dashboard (Phase E)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    abs_index, sock_path, pid_path = _resolve_paths(args.index_path)
    log_path = pid_path.parent / "daemon.log"

    if args.detach:
        if not _double_fork():
            return 0  # parent returns to caller

    _setup_logging(log_path, args.log_level)

    settings = get_settings(abs_index)
    idle = float(args.idle_timeout if args.idle_timeout is not None else settings.daemon_idle_timeout_s)

    server = DaemonServer(
        index_path=abs_index,
        socket_path=sock_path,
        pid_path=pid_path,
        idle_timeout_s=idle,
        http_port=args.http_port,
    )
    return server.run()


if __name__ == "__main__":
    raise SystemExit(main())
