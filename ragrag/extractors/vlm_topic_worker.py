"""Subprocess worker for the VLM topic chunker.

The parent process reserves its CUDA context exclusively for the
ColQwen3 embedder. The VLM topic chunker — which uses bnb 4-bit and
leaves non-PyTorch CUDA state behind after unload — runs here in an
isolated child. When the child exits, the driver reclaims every byte
of its CUDA context, leaving the parent's allocator pristine.

Protocol: one JSON object on stdin, one JSON object per file on stdout
as a JSON Lines stream, then the ``{"status": "done"}`` sentinel.

Request shape:
    {
      "settings": { ... Settings.model_dump() ... },
      "files": [
        {"path": "...", "sha256": "...", "type": "pdf|text"}
      ]
    }

Per-file response:
    {
      "path": "...",
      "status": "ok" | "error",
      "chunks": [ ... Chunk.model_dump() ... ],
      "error": "... when status == error ..."
    }

Final sentinel:
    {"status": "done"}

Errors during worker startup emit a single JSON object
``{"status": "fatal", "error": "..."}`` and exit non-zero.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

# Offline HF before torch imports so the worker does not hit the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _fatal(error: str, exit_code: int = 2) -> None:
    _emit({"status": "fatal", "error": error})
    sys.exit(exit_code)


def main() -> int:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            _fatal("empty request on stdin")
        request = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        _fatal(f"invalid request JSON: {exc}")
        return 2

    settings_payload = request.get("settings") or {}
    files = request.get("files") or []
    if not isinstance(files, list):
        _fatal("'files' must be a list")
        return 2

    # Heavy imports go here, AFTER the request has been parsed, so a
    # malformed stdin does not pay the ~3 s transformers warm-up.
    try:
        from ragrag.config import Settings
        from ragrag.embedding.vlm_loader import load_vlm
        from ragrag.extractors.text_topic_segmenter import TextTopicSegmenter
        from ragrag.extractors.vlm_topic_chunker import VLMTopicChunker
        from ragrag.extractors.vlm_topic_client import VLMTopicClient

        settings = Settings(**settings_payload)
    except Exception as exc:  # noqa: BLE001
        _fatal(f"failed to rebuild settings: {exc}\n{traceback.format_exc()}")
        return 2

    try:
        handle = load_vlm(
            settings.vlm_model_id,
            quantization=settings.vlm_quantization,
        )
    except Exception as exc:  # noqa: BLE001
        _fatal(f"vlm load failed: {exc}\n{traceback.format_exc()}")
        return 2

    try:
        client = VLMTopicClient(
            handle, image_max_side=settings.chunker_vlm_image_max_side
        )

        for entry in files:
            path = entry.get("path")
            sha256 = entry.get("sha256") or ""
            ftype = entry.get("type") or ""
            try:
                chunks = _plan_one(client, settings, path, sha256, ftype)
                _emit(
                    {
                        "path": path,
                        "status": "ok",
                        "chunks": [c.model_dump(mode="json") for c in chunks],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                _emit(
                    {
                        "path": path,
                        "status": "error",
                        "chunks": [],
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )
    finally:
        try:
            handle.unload()
        except Exception:
            pass

    _emit({"status": "done"})
    return 0


def _plan_one(client, settings, path: str, sha256: str, ftype: str):
    """Run the chunker for one file and return a list of ``Chunk``."""
    from pathlib import Path

    from ragrag.extractors.text_topic_segmenter import TextTopicSegmenter
    from ragrag.extractors.vlm_topic_chunker import VLMTopicChunker

    if ftype == "pdf":
        # Stream pages through PyMuPDF, same as IngestManager's own
        # plan-phase helper, but entirely inside this subprocess.
        from ragrag.extractors.pdf_extractor import iter_pdf_segments

        def _page_stream():
            current_page = None
            current_text = ""
            current_image = None
            for segment, image in iter_pdf_segments(path, settings):
                if segment.modality.value == "image":
                    if current_page is not None and current_image is not None:
                        yield (current_page, current_image, current_text)
                    current_page = segment.page or 0
                    current_image = image
                    current_text = segment.excerpt or ""
                elif current_page == segment.page:
                    current_text = (current_text + "\n" + (segment.excerpt or "")).strip()
            if current_page is not None and current_image is not None:
                yield (current_page, current_image, current_text)

        chunker = VLMTopicChunker(client, settings)
        return chunker.chunk(str(Path(path).resolve()), sha256, _page_stream())

    if ftype == "text":
        segmenter = TextTopicSegmenter(client, settings)
        return segmenter.segment(path)

    raise ValueError(f"unsupported file type: {ftype!r}")


if __name__ == "__main__":
    raise SystemExit(main())
