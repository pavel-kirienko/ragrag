"""Minimal JSON-RPC 2.0 framing for the ragrag daemon protocol.

We use a deliberately tiny subset:

  * Each request and response is a single JSON object terminated by ``\\n``.
  * One request per connection. The client opens, writes one request line,
    reads one response line, and closes. There is no batching, no streaming,
    no notifications, no server-pushed events.
  * Errors follow the JSON-RPC 2.0 shape but with a small fixed code table.

This file has no third-party dependencies and does no I/O — it is only a
codec. Sockets live in ``ragrag.daemon.server`` and ``ragrag.daemon.client``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


PROTOCOL_VERSION = 1
JSONRPC_VERSION = "2.0"


# Error codes — ours match JSON-RPC's reserved ranges where possible.
ERROR_PARSE = -32700
ERROR_INVALID_REQUEST = -32600
ERROR_METHOD_NOT_FOUND = -32601
ERROR_INVALID_PARAMS = -32602
ERROR_INTERNAL = -32603
ERROR_SERVER = -32000  # generic server-side error


class JsonRpcError(Exception):
    """Raised by the server-side dispatcher to send a structured error reply."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            out["data"] = self.data
        return out


@dataclass
class Request:
    method: str
    params: dict[str, Any]
    id: str | int | None


@dataclass
class Response:
    id: str | int | None
    result: Any = None
    error: dict[str, Any] | None = None


def encode_request(method: str, params: dict[str, Any] | None, id: str | int) -> bytes:
    """Encode a JSON-RPC request as one ``\\n``-terminated line."""
    payload = {
        "jsonrpc": JSONRPC_VERSION,
        "method": method,
        "params": params or {},
        "id": id,
    }
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def decode_request(line: bytes) -> Request:
    """Parse a single request line. Raises ``JsonRpcError`` on bad input."""
    try:
        text = line.decode("utf-8")
        payload = json.loads(text)
    except UnicodeDecodeError as exc:
        raise JsonRpcError(ERROR_PARSE, f"invalid utf-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise JsonRpcError(ERROR_PARSE, f"invalid json: {exc}") from exc

    if not isinstance(payload, dict):
        raise JsonRpcError(ERROR_INVALID_REQUEST, "request must be a JSON object")
    if payload.get("jsonrpc") != JSONRPC_VERSION:
        raise JsonRpcError(ERROR_INVALID_REQUEST, "missing or wrong jsonrpc version")
    method = payload.get("method")
    if not isinstance(method, str) or not method:
        raise JsonRpcError(ERROR_INVALID_REQUEST, "missing or non-string method")
    params = payload.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise JsonRpcError(ERROR_INVALID_PARAMS, "params must be an object")
    request_id = payload.get("id")
    if request_id is not None and not isinstance(request_id, (str, int)):
        raise JsonRpcError(ERROR_INVALID_REQUEST, "id must be string, integer, or null")
    return Request(method=method, params=params, id=request_id)


def encode_response(response: Response) -> bytes:
    """Encode a response object as one ``\\n``-terminated line."""
    payload: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": response.id}
    if response.error is not None:
        payload["error"] = response.error
    else:
        payload["result"] = response.result
    return (json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n").encode("utf-8")


def decode_response(line: bytes) -> Response:
    """Parse a single response line returned from the daemon."""
    try:
        text = line.decode("utf-8")
        payload = json.loads(text)
    except UnicodeDecodeError as exc:
        raise JsonRpcError(ERROR_PARSE, f"invalid utf-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise JsonRpcError(ERROR_PARSE, f"invalid json: {exc}") from exc
    if not isinstance(payload, dict):
        raise JsonRpcError(ERROR_INVALID_REQUEST, "response must be an object")
    return Response(
        id=payload.get("id"),
        result=payload.get("result"),
        error=payload.get("error"),
    )


def error_response(request_id: str | int | None, error: JsonRpcError) -> Response:
    """Build a Response carrying the given error."""
    return Response(id=request_id, error=error.to_dict())


def _json_default(obj: Any) -> Any:
    """Fallback JSON encoder used by encode_response.

    Pydantic v2 models expose ``.model_dump()``; we delegate to it so that
    the daemon can return ``SearchResponse`` (or similar) directly without
    callers having to convert.
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")
