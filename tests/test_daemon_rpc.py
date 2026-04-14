"""Tests for the JSON-RPC codec used by the daemon."""
from __future__ import annotations

import json

import pytest

from ragrag.daemon import rpc
from ragrag.daemon.rpc import (
    ERROR_INVALID_PARAMS,
    ERROR_INVALID_REQUEST,
    ERROR_PARSE,
    JsonRpcError,
    Response,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
    error_response,
)


def test_encode_decode_request_roundtrip() -> None:
    line = encode_request("search", {"query": "foo", "top_k": 5}, "abc-123")
    assert line.endswith(b"\n")
    request = decode_request(line.rstrip(b"\n"))
    assert request.method == "search"
    assert request.params == {"query": "foo", "top_k": 5}
    assert request.id == "abc-123"


def test_decode_request_strips_trailing_newline() -> None:
    line = encode_request("status", None, 1)
    request = decode_request(line)  # newline still attached
    assert request.method == "status"
    assert request.params == {}
    assert request.id == 1


def test_decode_request_invalid_json() -> None:
    with pytest.raises(JsonRpcError) as exc_info:
        decode_request(b"{not json")
    assert exc_info.value.code == ERROR_PARSE


def test_decode_request_wrong_top_level_type() -> None:
    with pytest.raises(JsonRpcError) as exc_info:
        decode_request(b"[1,2,3]")
    assert exc_info.value.code == ERROR_INVALID_REQUEST


def test_decode_request_missing_method() -> None:
    payload = json.dumps({"jsonrpc": "2.0", "id": 1}).encode()
    with pytest.raises(JsonRpcError) as exc_info:
        decode_request(payload)
    assert exc_info.value.code == ERROR_INVALID_REQUEST


def test_decode_request_wrong_jsonrpc_version() -> None:
    payload = json.dumps({"jsonrpc": "1.0", "method": "search", "id": 1}).encode()
    with pytest.raises(JsonRpcError) as exc_info:
        decode_request(payload)
    assert exc_info.value.code == ERROR_INVALID_REQUEST


def test_decode_request_non_object_params() -> None:
    payload = json.dumps({"jsonrpc": "2.0", "method": "search", "params": [1, 2], "id": 1}).encode()
    with pytest.raises(JsonRpcError) as exc_info:
        decode_request(payload)
    assert exc_info.value.code == ERROR_INVALID_PARAMS


def test_encode_response_with_result() -> None:
    response = Response(id="r-1", result={"ok": True})
    line = encode_response(response)
    decoded = decode_response(line.rstrip(b"\n"))
    assert decoded.id == "r-1"
    assert decoded.result == {"ok": True}
    assert decoded.error is None


def test_encode_response_with_error() -> None:
    response = error_response("r-2", JsonRpcError(-32000, "boom", data={"why": "test"}))
    line = encode_response(response)
    decoded = decode_response(line)
    assert decoded.id == "r-2"
    assert decoded.error is not None
    assert decoded.error["code"] == -32000
    assert decoded.error["message"] == "boom"
    assert decoded.error["data"] == {"why": "test"}


def test_encode_response_uses_pydantic_model_dump() -> None:
    """Result objects with a model_dump() method serialise via that hook."""
    class Dummy:
        def model_dump(self) -> dict:
            return {"x": 1, "y": "two"}

    response = Response(id=1, result=Dummy())
    line = encode_response(response)
    decoded = decode_response(line)
    assert decoded.result == {"x": 1, "y": "two"}


def test_jsonrpc_error_to_dict() -> None:
    err = JsonRpcError(-42, "no", data=[1, 2])
    assert err.to_dict() == {"code": -42, "message": "no", "data": [1, 2]}
    err_no_data = JsonRpcError(-7, "soft")
    assert err_no_data.to_dict() == {"code": -7, "message": "soft"}
