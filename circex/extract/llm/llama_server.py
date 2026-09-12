"""LlamaServerExtractor — Mistral-7B via a llama.cpp OpenAI-compatible server.

Where the Ollama path uses loose JSON mode + a repair retry + fail-soft (Mistral
regularly emits schema-non-conforming JSON), a llama.cpp server enforces the
output schema *at decode time* via grammar-constrained decoding
(`response_format: {type: json_schema}`) — the model cannot emit non-conforming
JSON, so no repair loop is needed. Same shared prompt and `llm_input_schema()` as
the other backends, so eval numbers stay comparable.

Set up on MSI `agc03:8080` (co-located with BOOM), which makes it both the box for
the full LLM eval and the production LLM backend for the SkyPortal service — the
service calls `localhost:8080`, no network hop. Point at it with `CIRCEX_LLAMA_URL`.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import requests
import structlog
from pydantic import ValidationError

from circex.cache.llm import LLMCache, cache_key
from circex.extract.llm.chunker import chunk_body, merge_extractions
from circex.extract.llm.ollama import OllamaExtractor  # reuse the parse/sanitize helpers
from circex.extract.llm.prompt import (
    PROMPT_V1,
    build_messages,
    build_system_text,
    llm_grammar_schema,
)
from circex.extract.protocol import Circular, Extractor
from circex.extract.timing import (
    resolve_inline_offsets,
    resolve_observation_epoch,
    resolve_relative_epochs,
)
from circex.schema import CircularExtraction, ExtractionMeta

log = structlog.get_logger(__name__)

DEFAULT_LLAMA_URL = os.environ.get("CIRCEX_LLAMA_URL", "http://localhost:8080")
DEFAULT_LLAMA_MODEL = os.environ.get("CIRCEX_LLAMA_MODEL", "mistral-7b")
# Grammar-constrained decoding on the full CircularExtraction schema is far slower
# than free generation on a dense circular; give it room. Override for tuning.
DEFAULT_LLAMA_TIMEOUT = float(os.environ.get("CIRCEX_LLAMA_TIMEOUT", "300"))
# Hard cap on generated tokens. Belt-and-braces with the schema's maxItems: even a
# looping model stops here instead of grinding out 10k tokens under constrained
# sampling. A rich extraction fits comfortably in this budget.
DEFAULT_LLAMA_MAX_TOKENS = int(os.environ.get("CIRCEX_LLAMA_MAX_TOKENS", "2048"))
# Whether the grammar names every field in `required`. Model-dependent — see
# llm_grammar_schema. Off suits Mistral-7B; Qwen3 extracts nothing without it.
DEFAULT_LLAMA_REQUIRE_FIELDS = os.environ.get("CIRCEX_LLAMA_REQUIRE_FIELDS", "").lower() in (
    "1",
    "true",
    "yes",
)
# Bearer token, for a server reached over the open internet rather than through
# an ssh tunnel. A llama-server on localhost needs none.
DEFAULT_LLAMA_API_KEY = os.environ.get("CIRCEX_LLAMA_API_KEY") or None
# Constrained decoding is the default because a conforming answer cannot then be
# malformed. Not every server honours it usefully: a reasoning model asked for a
# schema may satisfy it with an all-null object and stop. Turning this off asks
# for the same JSON in prose and validates what comes back, which is weaker but
# is the difference between an answer and an empty one on such a server.
DEFAULT_LLAMA_STRUCTURED = os.environ.get("CIRCEX_LLAMA_STRUCTURED", "1").lower() not in (
    "0",
    "false",
    "no",
)
# Passed through to the server, for options it defines and this client does not:
# a reasoning model's effort level, say. JSON object, e.g. {"reasoning_effort": "low"}.
DEFAULT_LLAMA_EXTRA_BODY = os.environ.get("CIRCEX_LLAMA_EXTRA_BODY") or None


def _json_object(content: str) -> str:
    """The JSON object in a free-form reply.

    Without a grammar a model may fence its answer or introduce it in prose, so
    take the outermost braces rather than requiring the whole reply to parse.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S)
    if fenced:
        return str(fenced.group(1))
    start, end = content.find("{"), content.rfind("}")
    return content[start : end + 1] if 0 <= start < end else content


class LlamaServerExtractor(Extractor):
    """Grammar-constrained extraction against a llama.cpp OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str = DEFAULT_LLAMA_URL,
        model_id: str = DEFAULT_LLAMA_MODEL,
        cache: LLMCache | None = None,
        timeout: float = DEFAULT_LLAMA_TIMEOUT,
        session: Any | None = None,
        require_fields: bool = DEFAULT_LLAMA_REQUIRE_FIELDS,
        api_key: str | None = DEFAULT_LLAMA_API_KEY,
        structured: bool = DEFAULT_LLAMA_STRUCTURED,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._cache = cache
        self._timeout = timeout
        self._session = session or requests  # injectable for tests
        self._require_fields = require_fields
        self._api_key = api_key
        self._structured = structured
        if extra_body is None and DEFAULT_LLAMA_EXTRA_BODY:
            extra_body = json.loads(DEFAULT_LLAMA_EXTRA_BODY)
        self._extra_body = extra_body or {}

    @property
    def extractor_id(self) -> str:
        # The mode belongs in the id: the same model constrained and unconstrained
        # is two different extractors, and a cache must not confuse them.
        suffix = "" if self._structured else ":free"
        return f"llama-server:{self._model_id}{suffix}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def prompt_version(self) -> str:
        # The grammar shape changes what the model emits, so it belongs in the
        # cache key: entries from the two variants must not collide.
        return f"{PROMPT_V1}+required" if self._require_fields else PROMPT_V1

    def extract(self, circular: Circular) -> CircularExtraction:
        body_hash = cache_key(circular.body)
        if self._cache is not None:
            cached = self._cache.get(
                self.extractor_id,
                self._model_id,
                self.prompt_version,
                circular.circular_id,
                body_hash,
            )
            if cached is not None:
                meta = cached.extraction.extraction_meta.model_copy(update={"cache_hit": True})
                result = cached.extraction.model_copy(update={"extraction_meta": meta})
                resolve_observation_epoch(result, circular.body)
                resolve_inline_offsets(result, circular.body, circular.trigger_time)
                resolve_relative_epochs(result, circular.trigger_time)
                return result

        started = time.perf_counter()
        chunk_results: list[CircularExtraction] = []
        had_error = False
        for chunk in chunk_body(circular.body):
            try:
                payload = self._call(
                    Circular(
                        circular_id=circular.circular_id,
                        subject=circular.subject,
                        body=chunk,
                        event_id=circular.event_id,
                    )
                )
            except (ValidationError, json.JSONDecodeError, requests.RequestException) as exc:
                # Grammar-constrained output shouldn't fail validation, but a
                # server/transport error still shouldn't crash the run.
                log.warning(
                    "llama_server_extract_failed",
                    circular_id=circular.circular_id,
                    error=str(exc)[:300],
                )
                payload = {}
                had_error = True
            payload["circular_id"] = circular.circular_id
            llm_notes = payload.pop("_llm_notes", [])
            if not isinstance(llm_notes, list):
                llm_notes = []
            payload["extraction_meta"] = {
                "extractor": self.extractor_id,
                "model_id": self._model_id,
                "notes": llm_notes,
            }
            chunk_results.append(CircularExtraction.model_validate(payload))

        latency_ms = (time.perf_counter() - started) * 1000.0
        meta = ExtractionMeta(
            extractor=self.extractor_id,
            model_id=self._model_id,
            prompt_version=self.prompt_version,
            latency_ms=latency_ms,
            cache_hit=False,
        )
        merged = merge_extractions(circular.circular_id, chunk_results, meta)
        # Never cache a partial/empty result produced by a transport failure — a
        # tunnel blip would otherwise poison the cache and be silently skipped on
        # re-run. Only persist extractions where every chunk actually returned.
        if self._cache is not None and not had_error:
            self._cache.put(
                extractor_id=self.extractor_id,
                model_id=self._model_id,
                prompt_version=self.prompt_version,
                circular_id=circular.circular_id,
                body_sha1=body_hash,
                extraction=merged,
                latency_ms=latency_ms,
            )
        # An observation datetime stated in the body is exact; a relative offset is
        # rounded ("~4.5 days post-burst"), so it only fills what is still untimed.
        resolve_observation_epoch(merged, circular.body)
        resolve_inline_offsets(merged, circular.body, circular.trigger_time)
        resolve_relative_epochs(merged, circular.trigger_time)
        return merged

    @property
    def _headers(self) -> dict[str, str]:
        """Auth header, when the server is behind one. Never logged."""
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _call(self, circular: Circular) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": build_system_text()},
            *build_messages(circular),
        ]
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": DEFAULT_LLAMA_MAX_TOKENS,
            **self._extra_body,
        }
        if self._structured:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "circular_extraction",
                    # lean: scored fields only
                    "schema": llm_grammar_schema(require_fields=self._require_fields),
                },
            }
        resp = self._session.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"] or ""
        if not self._structured:
            content = _json_object(content)
        payload = OllamaExtractor._parse_and_strip_meta(content)
        return OllamaExtractor._sanitize_payload(payload, body=circular.body)
