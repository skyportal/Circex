"""Tests for the llama.cpp-server extractor (grammar-constrained JSON)."""

from __future__ import annotations

import json
from pathlib import Path

import requests

from circex.cache.llm import LLMCache
from circex.extract.llm import LlamaServerExtractor
from circex.extract.llm.prompt import llm_grammar_schema
from circex.extract.protocol import Circular


class _FakeResp:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeSession:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[tuple[str, dict]] = []
        self.headers: dict | None = None

    def post(self, url: str, json: dict, timeout: float, headers: dict | None = None) -> _FakeResp:  # noqa: A002
        self.calls.append((url, json))
        self.headers = headers
        return _FakeResp(self.content)


def test_llama_server_parses_grammar_constrained_output() -> None:
    model_output = json.dumps(
        {"event": {"event_name": "GRB 260604C"}, "redshift": {"redshift": 0.5}, "provenance": {}}
    )
    session = _FakeSession(model_output)
    ext = LlamaServerExtractor(base_url="http://agc03:8080", session=session)
    result = ext.extract(Circular(circular_id=44877, subject="", body="GRB 260604C at z = 0.5"))

    assert result.circular_id == 44877
    assert result.event is not None and result.event.event_name == "GRB 260604C"
    assert result.redshift is not None and result.redshift.redshift == 0.5
    assert result.extraction_meta.extractor == "llama-server:mistral-7b"


def test_llama_server_uses_grammar_constrained_request() -> None:
    session = _FakeSession(json.dumps({"provenance": {}}))
    LlamaServerExtractor(session=session).extract(Circular(circular_id=1, subject="", body="x"))
    url, payload = session.calls[0]
    assert url.endswith("/v1/chat/completions")
    assert payload["temperature"] == 0
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"]["type"] == "object"
    json.dumps(payload)  # messages must be JSON-serializable for the real POST


class _FailSession:
    """Session whose POST always dies mid-flight, like a dropped tunnel."""

    def post(self, url: str, json: dict, timeout: float, headers: dict | None = None) -> _FakeResp:  # noqa: A002
        raise requests.ConnectionError("('Connection aborted.', ConnectionResetError(54, ...))")


def test_llama_server_does_not_cache_on_transport_failure(tmp_path: Path) -> None:
    # A transport error must fail soft (no crash) AND leave the cache empty, so a
    # re-run re-hits the model instead of returning a poisoned empty extraction.
    cache = LLMCache(tmp_path / "llm.sqlite")
    ext = LlamaServerExtractor(cache=cache, session=_FailSession())
    result = ext.extract(Circular(circular_id=6418, subject="", body="GRB 260604C at z = 0.5"))

    assert result.circular_id == 6418  # fail-soft: returns an (empty) extraction
    assert cache.count() == 0  # but nothing was persisted


def test_llama_server_binds_body_observation_epoch_to_untimed_rows() -> None:
    """A table row with no date column gets obs_mjd from a single body-level
    observation time, matching the regex extractor (GCN 45198 regression)."""
    model_output = json.dumps(
        {"event": {"event_name": "AT2026vts"}, "photometry": [{"filter": "r", "mag": 19.78}]}
    )
    body = "We began observations at 2026-07-23 04:54 UTC and measured the candidate."
    ext = LlamaServerExtractor(session=_FakeSession(model_output))
    result = ext.extract(Circular(circular_id=45198, subject="", body=body))
    assert result.photometry[0].obs_mjd is not None
    assert result.photometry[0].obs_time is not None


def test_grammar_requires_no_fields_by_default():
    """Mistral pads the photometry array to maxItems when the fields are required."""
    assert llm_grammar_schema()["required"] == []


def test_require_fields_names_every_top_level_field():
    """Qwen3 returns {} otherwise — the empty object satisfies an empty `required`."""
    schema = llm_grammar_schema(require_fields=True)
    assert set(schema["required"]) == set(schema["properties"])


def test_require_fields_changes_only_the_required_list():
    lean, strict = llm_grammar_schema(), llm_grammar_schema(require_fields=True)
    assert {k: v for k, v in lean.items() if k != "required"} == {
        k: v for k, v in strict.items() if k != "required"
    }


def test_required_fields_stay_nullable():
    """Requiring a field compels the model to mention it, not to invent a value."""
    schema = llm_grammar_schema(require_fields=True)
    event = schema["properties"]["event"]
    assert {"type": "null"} in event["anyOf"]


def test_require_fields_is_part_of_the_cache_key():
    """The grammar changes the output, so the two variants must not share cache entries."""
    lean = LlamaServerExtractor(session=object())
    strict = LlamaServerExtractor(session=object(), require_fields=True)
    assert lean.prompt_version != strict.prompt_version


def test_require_fields_reaches_the_request_schema():
    class Recorder:
        def __init__(self):
            self.payload = None

        def post(self, url, json, timeout, headers=None):
            self.payload = json
            self.headers = headers
            raise requests.RequestException("stop here; the schema is what we assert on")

    for require, expected in ((False, False), (True, True)):
        session = Recorder()
        LlamaServerExtractor(session=session, require_fields=require).extract(
            Circular(circular_id=1, subject="s", body="b")
        )
        sent = session.payload["response_format"]["json_schema"]["schema"]
        assert bool(sent["required"]) is expected


def test_no_auth_header_without_an_api_key():
    """A llama-server on localhost needs none; don't send an empty Bearer."""
    session = _FakeSession(json.dumps({"choices": [{"message": {"content": "{}"}}]}))
    LlamaServerExtractor(session=session).extract(Circular(circular_id=1, subject="", body="b"))
    assert not session.headers


def test_api_key_is_sent_as_a_bearer_token():
    session = _FakeSession(json.dumps({"choices": [{"message": {"content": "{}"}}]}))
    LlamaServerExtractor(session=session, api_key="sekrit").extract(
        Circular(circular_id=1, subject="", body="b")
    )
    assert session.headers == {"Authorization": "Bearer sekrit"}


def test_api_key_is_not_part_of_the_cache_key():
    """The same model and grammar give the same answer however you authenticate."""
    plain = LlamaServerExtractor(session=object())
    keyed = LlamaServerExtractor(session=object(), api_key="sekrit")
    assert (plain.extractor_id, plain.prompt_version) == (
        keyed.extractor_id,
        keyed.prompt_version,
    )


def test_unconstrained_mode_sends_no_response_format_and_reads_fenced_json():
    """A server that answers a schema with nulls needs asking in prose instead."""
    import json as _json

    from circex.extract.llm.llama_server import LlamaServerExtractor
    from circex.extract.protocol import Circular

    sent = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            body = _json.dumps({"photometry": [{"filter": "r", "mag": 19.5}]})
            return {"choices": [{"message": {"content": f"Here you go:\n```json\n{body}\n```"}}]}

    class FakeSession:
        def post(self, url, json=None, headers=None, timeout=None):
            sent.update(json or {})
            return FakeResponse()

    extractor = LlamaServerExtractor(
        session=FakeSession(), structured=False, extra_body={"reasoning_effort": "low"}
    )
    result = extractor.extract(Circular(circular_id=1, subject="s", body="b"))

    assert "response_format" not in sent, "the schema is what the server mishandles"
    assert sent["reasoning_effort"] == "low"
    assert extractor.extractor_id.endswith(":free")
    assert [p.mag for p in result.photometry] == [19.5]
