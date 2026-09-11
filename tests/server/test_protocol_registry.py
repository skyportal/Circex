"""Tests for the JSON request protocol + tool registry dispatch."""

from __future__ import annotations

import pytest

from circex.server.protocol import ToolRequest, ToolResponse
from circex.server.registry import TOOLS, ToolContext, dispatch


def test_tool_request_from_dict() -> None:
    req = ToolRequest.from_dict({"tool": "get_redshift", "arguments": {"event": "GRB X"}})
    assert req.tool == "get_redshift"
    assert req.arguments == {"event": "GRB X"}


def test_tool_request_missing_tool_field_raises() -> None:
    with pytest.raises(ValueError, match="tool"):
        ToolRequest.from_dict({"arguments": {}})


def test_tool_request_bad_arguments_raises() -> None:
    with pytest.raises(ValueError, match="arguments"):
        ToolRequest.from_dict({"tool": "x", "arguments": "not a dict"})


def test_tool_response_success_serializes() -> None:
    resp = ToolResponse(ok=True, result={"x": 1}, id="req-1")
    payload = resp.to_dict()
    assert payload == {"ok": True, "result": {"x": 1}, "id": "req-1"}


def test_tool_response_error_serializes() -> None:
    resp = ToolResponse(ok=False, error="boom", id="req-2")
    payload = resp.to_dict()
    assert payload == {"ok": False, "error": "boom", "id": "req-2"}


def test_core_tools_registered() -> None:
    expected = {
        "extract_properties",
        "extract_text",
        "get_redshift",
        "get_photometry",
        "get_classification",
        "find_counterparts",
        "search_by_position",
        "search_gcn_circulars",
        "fetch_gcn_circulars",
    }
    assert expected <= set(TOOLS.keys())


def test_dispatch_unknown_tool_raises() -> None:
    ctx = ToolContext(store=None)
    with pytest.raises(KeyError, match="unknown tool"):
        dispatch(ctx, "no_such_tool", {})
