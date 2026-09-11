"""Integration tests for the demo web bridge against a fake worker socket."""

from __future__ import annotations

import http.client
import importlib.util
import json
import socket
import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

_BRIDGE_PATH = Path(__file__).parents[2] / "demo" / "web" / "serve.py"
_spec = importlib.util.spec_from_file_location("circex_web_serve", _BRIDGE_PATH)
assert _spec and _spec.loader
serve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _FakeWorker(threading.Thread):
    """Accepts one connection, reads a JSON line, replies with a canned response."""

    def __init__(self, port: int, response: dict[str, object]) -> None:
        super().__init__(daemon=True)
        self._port = port
        self._response = response
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", port))
        self._sock.listen(5)
        self.requests: list[dict[str, object]] = []
        self._stop = False

    def run(self) -> None:
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                buf = b""
                while b"\n" not in buf:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                if buf:
                    self.requests.append(json.loads(buf.split(b"\n", 1)[0]))
                conn.sendall((json.dumps(self._response) + "\n").encode("utf-8"))

    def stop(self) -> None:
        self._stop = True
        self._sock.close()


@pytest.fixture
def bridge_and_worker() -> Iterator[tuple[int, _FakeWorker]]:
    worker_port = _free_port()
    bridge_port = _free_port()
    worker = _FakeWorker(worker_port, {"ok": True, "result": {"redshift": 1.61}, "id": None})
    worker.start()

    serve.WORKER_HOST = "127.0.0.1"
    serve.WORKER_PORT = worker_port

    httpd = ThreadingHTTPServer(("127.0.0.1", bridge_port), serve.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield bridge_port, worker
    finally:
        httpd.shutdown()
        worker.stop()


def _post(port: int, path: str, body: dict[str, object]) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    conn.close()
    return resp.status, data


def _get(port: int, path: str) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    out = resp.read()
    conn.close()
    return resp.status, out


def test_index_served(bridge_and_worker: tuple[int, _FakeWorker]) -> None:
    port, _ = bridge_and_worker
    status, body = _get(port, "/")
    assert status == 200
    assert b"<title>Circex" in body


def test_tool_call_forwarded(bridge_and_worker: tuple[int, _FakeWorker]) -> None:
    port, worker = bridge_and_worker
    status, data = _post(
        port,
        "/api/tool",
        {"tool": "get_redshift", "arguments": {"event": "GRB 990123"}},
    )
    assert status == 200
    assert data["ok"] is True
    assert data["result"]["redshift"] == 1.61
    assert worker.requests[0]["tool"] == "get_redshift"
    assert worker.requests[0]["arguments"] == {"event": "GRB 990123"}


def test_tool_not_in_allowlist_rejected(bridge_and_worker: tuple[int, _FakeWorker]) -> None:
    port, _ = bridge_and_worker
    status, data = _post(port, "/api/tool", {"tool": "rm_rf", "arguments": {}})
    assert status == 400
    assert "not allowed" in data["error"]


def test_bad_arguments_rejected(bridge_and_worker: tuple[int, _FakeWorker]) -> None:
    port, _ = bridge_and_worker
    status, data = _post(port, "/api/tool", {"tool": "get_redshift", "arguments": "oops"})
    assert status == 400


def test_health_reports_up(bridge_and_worker: tuple[int, _FakeWorker]) -> None:
    port, _ = bridge_and_worker
    status, body = _get(port, "/api/health")
    assert status == 200
    assert json.loads(body)["worker"] == "up"


def test_unknown_path_404(bridge_and_worker: tuple[int, _FakeWorker]) -> None:
    port, _ = bridge_and_worker
    status, _body = _get(port, "/secrets")
    assert status == 404


def test_allowed_tools_matches_worker_registry() -> None:
    """The bridge allow-list must equal the worker's registered tool set."""
    from circex.server import TOOLS

    assert frozenset(TOOLS.keys()) == serve.ALLOWED_TOOLS
