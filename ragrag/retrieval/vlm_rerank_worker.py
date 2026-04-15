"""Long-lived VLM reranker subprocess.

Loads the VLM once, then serves an arbitrary number of rerank
requests over stdin/stdout. Each request is one line of JSON
followed by a newline; each response is one line of JSON followed
by a newline.

Protocol:

    request  {"cmd": "ping"}
    response {"status": "pong"}

    request  {"cmd": "shutdown"}
    response {"status": "bye"}       (worker exits afterwards)

    request  {
        "cmd": "rerank",
        "query": "<user query>",
        "candidates": [
            {
                "id": 0,
                "title": "...",
                "summary": "...",
                "pages": "1-3, 7",
                "excerpt": "...",
                "image_path": "/path/to/hero.webp"   (optional)
            }, ...
        ]
    }
    response {
        "status": "ok",
        "ranks": [
            {"id": 0, "rank": 1, "score": 8, "reason": "..."},
            ...
        ]
    }

On parse or generation errors the response carries
``{"status": "error", "error": "..."}`` so the parent can fall
back to the pre-rerank order.

Like the topic chunker worker, this process is isolated so its
CUDA context — and any bnb 4-bit state it pulls in — stays out
of the embedder's allocator.
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
from typing import Any

# Offline HF before transitive torch imports.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    # Heavy imports after argparse so ``--help`` stays snappy.
    import argparse

    parser = argparse.ArgumentParser(description="VLM reranker worker")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--quantization", default="auto")
    parser.add_argument("--image-max-side", type=int, default=640)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    try:
        from ragrag.embedding.vlm_loader import load_vlm

        handle = load_vlm(args.model_id, quantization=args.quantization)
    except Exception as exc:  # noqa: BLE001
        _emit({"status": "fatal", "error": f"vlm load failed: {exc}\n{traceback.format_exc()}"})
        return 2

    _emit({"status": "ready", "device": handle.device, "model_id": handle.model_id})

    try:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                _emit({"status": "error", "error": f"bad json: {exc}"})
                continue

            cmd = request.get("cmd")
            if cmd == "ping":
                _emit({"status": "pong"})
                continue
            if cmd == "shutdown":
                _emit({"status": "bye"})
                break
            if cmd != "rerank":
                _emit({"status": "error", "error": f"unknown cmd: {cmd!r}"})
                continue

            try:
                response = _handle_rerank(handle, request, args)
            except Exception as exc:  # noqa: BLE001
                response = {
                    "status": "error",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "trace": traceback.format_exc()[-400:],
                }
            _emit(response)
    finally:
        try:
            handle.unload()
        except Exception:
            pass

    return 0


def _handle_rerank(handle, request: dict, args) -> dict:
    query = request.get("query") or ""
    candidates = request.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return {"status": "error", "error": "no candidates"}

    prompt = _build_prompt(query, candidates)
    # On CPU the vision encoder is the dominant cost (a 10-image
    # rerank prompt is ~10k image tokens of prefill, several minutes
    # per query). The candidate text — title, summary, excerpt — is
    # already informative enough for listwise ranking in text-only
    # mode, so we drop images entirely when the handle is on CPU.
    handle_device = str(getattr(handle, "device", "") or "").lower()
    if handle_device == "cpu":
        images: list[Any] = []
    else:
        images = _load_images(candidates, args.image_max_side)

    raw = handle.generate(
        prompt,
        images=images if images else None,
        max_new_tokens=int(args.max_new_tokens),
        temperature=0.0,
    )
    ranks = _parse_ranks(raw, candidates)
    if not ranks:
        return {"status": "error", "error": f"unparseable response: {raw[:200]!r}"}
    return {"status": "ok", "ranks": ranks, "raw": raw[:400]}


def _build_prompt(query: str, candidates: list[dict]) -> str:
    lines: list[str] = []
    lines.append(
        f'A user is searching a technical document corpus for: "{query}"\n'
    )
    lines.append(
        f"Below are {len(candidates)} candidate topic chunks. For each candidate you see "
        "its title, a one-sentence summary, the pages it covers, a short excerpt "
        "and (when available) a representative page image.\n"
    )
    lines.append(
        "Rank the candidates from most to least relevant to the query. "
        "Use ONLY the information visible in the candidate block — do not "
        "invent pages or facts.\n"
    )
    lines.append("Candidates:\n")
    for c in candidates:
        lines.append(
            f"[{c.get('id')}] {c.get('title') or '(untitled)'} (pages {c.get('pages') or '?'})"
        )
        summary = (c.get("summary") or "").strip()
        if summary:
            lines.append(f"  summary: {summary[:240]}")
        excerpt = (c.get("excerpt") or "").strip()
        if excerpt:
            lines.append(f"  excerpt: {excerpt[:240]}")
        lines.append("")
    lines.append(
        "Output a JSON array. Each entry must be an object with:\n"
        '  "id" — the [N] identifier above\n'
        '  "rank" — 1 = most relevant\n'
        '  "score" — 0 (irrelevant) to 9 (directly answers the query)\n'
        '  "reason" — one short sentence explaining the placement\n'
        "Output ONLY the JSON array, no prose, no fences."
    )
    return "\n".join(lines)


def _load_images(candidates: list[dict], max_side: int) -> list[Any]:
    images: list[Any] = []
    try:
        from PIL import Image
    except Exception:
        return images
    for c in candidates:
        path = c.get("image_path")
        if not path:
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / float(longest)
            img = img.resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                Image.Resampling.LANCZOS,
            )
        images.append(img)
    return images


_RANK_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_ranks(raw: str, candidates: list[dict]) -> list[dict]:
    if not raw:
        return []
    stripped = raw.strip()
    if stripped.startswith("```"):
        nl = stripped.find("\n")
        if nl >= 0:
            stripped = stripped[nl + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()

    match = _RANK_ARRAY_RE.search(stripped)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    valid_ids = {int(c.get("id", -1)) for c in candidates if isinstance(c.get("id"), int)}
    out: list[dict] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        try:
            eid = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        if eid not in valid_ids:
            continue
        try:
            rank_val = int(entry.get("rank", 0))
        except (TypeError, ValueError):
            rank_val = 0
        try:
            score_val = float(entry.get("score", 0))
        except (TypeError, ValueError):
            score_val = 0.0
        reason = str(entry.get("reason") or "").strip()[:200]
        out.append({"id": eid, "rank": rank_val, "score": score_val, "reason": reason})
    return out


if __name__ == "__main__":
    raise SystemExit(main())
