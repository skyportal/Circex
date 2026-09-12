"""OllamaExtractor — JSON-mode constrained extraction with one repair retry.

Mistral-7B-Instruct-v0.2 doesn't have first-class tool use. We use `format="json"`
and pass the input schema in the prompt. On Pydantic validation failure, retry
once with the validation error appended to the messages.

Cost is always None (open-source local model). Latency is wall-clock.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, get_args

import ollama
import structlog
from pydantic import ValidationError

from circex.cache.llm import LLMCache, cache_key
from circex.extract.llm.chunker import chunk_body, merge_extractions
from circex.extract.llm.prompt import (
    PROMPT_V1,
    build_messages,
    build_system_text,
    llm_input_schema,
)
from circex.extract.protocol import Circular, Extractor
from circex.extract.timing import (
    resolve_inline_offsets,
    resolve_relative_epochs,
)
from circex.schema import CircularExtraction, ExtractionMeta
from circex.schema.photometry import CalibrationReference
from circex.taxonomy import normalize_classification

log = structlog.get_logger(__name__)

# The bare `mistral:7b-instruct-v0.2` is not a pullable tag in the Ollama
# registry; the v0.2 variants are all quantizations. Q4_K_M is the standard
# balanced choice (~4 GB, near-FP16 quality). Override with CIRCEX_OLLAMA_MODEL.
DEFAULT_OLLAMA_MODEL = os.environ.get("CIRCEX_OLLAMA_MODEL", "mistral:7b-instruct-v0.2-q4_K_M")


class OllamaExtractor(Extractor):
    """LLM extractor backed by a local Ollama model."""

    def __init__(
        self,
        model_id: str = DEFAULT_OLLAMA_MODEL,
        cache: LLMCache | None = None,
        client: Any | None = None,
    ) -> None:
        self._model_id = model_id
        self._cache = cache
        self._client = client or ollama

    @property
    def extractor_id(self) -> str:
        return f"ollama:{self._model_id}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def prompt_version(self) -> str:
        return PROMPT_V1

    def extract(self, circular: Circular) -> CircularExtraction:
        body_hash = cache_key(circular.body)

        if self._cache is not None:
            cached = self._cache.get(
                self.extractor_id, self._model_id, PROMPT_V1, circular.circular_id, body_hash
            )
            if cached is not None:
                meta = cached.extraction.extraction_meta.model_copy(update={"cache_hit": True})
                result = cached.extraction.model_copy(update={"extraction_meta": meta})
                # Relative epochs are resolved per-call (T0-dependent), not cached.
                resolve_inline_offsets(result, circular.body, circular.trigger_time)
                resolve_relative_epochs(result, circular.trigger_time)
                return result

        chunks = chunk_body(circular.body)

        started = time.perf_counter()
        chunk_results: list[CircularExtraction] = []

        for chunk in chunks:
            try:
                payload = self._call_with_repair(
                    Circular(
                        circular_id=circular.circular_id,
                        subject=circular.subject,
                        body=chunk,
                        event_id=circular.event_id,
                    )
                )
            except (ValidationError, json.JSONDecodeError) as exc:
                # Mistral-7B regularly emits structurally-broken JSON or
                # schema-non-conforming objects on dense circulars even after
                # repair. Treat that as a null extraction so the eval scores it
                # as a model failure rather than crashing the whole run.
                log.warning(
                    "ollama_extract_failed",
                    circular_id=circular.circular_id,
                    error=str(exc)[:300],
                )
                payload = {}
            payload["circular_id"] = circular.circular_id
            # Preserve any `notes` the LLM emitted (e.g. redshift_bound phrases)
            # while overwriting the run-level fields. `_parse_and_strip_meta`
            # stashes them under `_llm_notes` (the full extraction_meta is
            # stripped because the runner owns it). Notes flow through the merge
            # step in chunker.py back into the final extraction_meta.
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
            prompt_version=PROMPT_V1,
            tokens_in=None,
            tokens_out=None,
            cost_usd=None,
            latency_ms=latency_ms,
            cache_hit=False,
        )
        merged = merge_extractions(circular.circular_id, chunk_results, meta)

        # Cache the T0-independent (absolute-epoch) extraction; put() serializes
        # immediately, so the relative resolution below doesn't leak into cache.
        if self._cache is not None:
            self._cache.put(
                extractor_id=self.extractor_id,
                model_id=self._model_id,
                prompt_version=PROMPT_V1,
                circular_id=circular.circular_id,
                body_sha1=body_hash,
                extraction=merged,
                latency_ms=latency_ms,
            )

        resolve_inline_offsets(merged, circular.body, circular.trigger_time)
        resolve_relative_epochs(merged, circular.trigger_time)
        return merged

    # ---- internal ----

    def _system_text_with_schema(self) -> str:
        """Ollama lacks tool-use; we embed the schema in the system message."""
        schema_json = json.dumps(llm_input_schema(), indent=2)
        return (
            build_system_text()
            + "\n\nReturn a JSON object conforming to this schema:\n"
            + schema_json
        )

    def _call_with_repair(self, circular: Circular) -> dict[str, Any]:
        """Call once; on validation error, retry once with the error appended."""
        system_text = self._system_text_with_schema()
        messages = [
            {"role": "system", "content": system_text},
            *build_messages(circular),
        ]

        first = self._client.chat(model=self._model_id, messages=messages, format="json")
        first_content = first["message"]["content"]
        try:
            return self._parse_validate(first_content, circular.body)
        except (ValidationError, json.JSONDecodeError) as exc:
            error_message = str(exc)
            log.warning("ollama_repair_retry", error=error_message)

        # Repair retry: append the error and ask for a corrected JSON.
        messages.append({"role": "assistant", "content": first_content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "The previous response failed schema validation with: "
                    f"{error_message}. Emit a corrected JSON object only."
                ),
            }
        )
        second = self._client.chat(model=self._model_id, messages=messages, format="json")
        return self._parse_validate(second["message"]["content"], circular.body)

    def _parse_validate(self, content: str, body: str) -> dict[str, Any]:
        """Parse, sanitize, then trial-validate. Raises on hard schema failures."""
        payload = self._parse_and_strip_meta(content)
        payload = self._sanitize_payload(payload, body=body)
        trial = dict(payload)
        trial.setdefault("circular_id", 0)
        trial.setdefault("extraction_meta", {"extractor": "trial"})
        CircularExtraction.model_validate(trial)
        return payload

    @staticmethod
    def _parse_and_strip_meta(content: str) -> dict[str, Any]:
        """Parse JSON, strip the runner-owned extraction_meta, but stash any
        model-authored `notes` under `_llm_notes` so the caller can fold them
        into the final extraction_meta (CircularExtraction ignores the extra
        `_llm_notes` key during validation)."""
        loaded = json.loads(content)
        if not isinstance(loaded, dict):
            raise json.JSONDecodeError("expected JSON object, got non-dict", content, 0)
        payload: dict[str, Any] = dict(loaded)
        meta = payload.pop("extraction_meta", None)
        if isinstance(meta, dict) and isinstance(meta.get("notes"), list):
            payload["_llm_notes"] = meta["notes"]
        return payload

    @staticmethod
    def _sanitize_payload(payload: dict[str, Any], body: str) -> dict[str, Any]:
        """Tolerant fixups for common small-model output quirks.

        Applied to the post-LLM JSON before strict Pydantic validation:
        - drop provenance entries that can't form a valid Span (missing or
          non-int offsets, out-of-range offsets); recompute snippet from
          body[start:end] when the model omits or corrupts it
        - collapse `{"X": {"X": null}}` to `"X": null` (a common Mistral
          shape-confusion on optional nested objects)
        - coerce `follow_up.reference.gcn_circulars` list-of-anything to a
          comma-joined string (the schema expects str | float)
        """
        # --- provenance ---
        prov = payload.get("provenance")
        if isinstance(prov, dict):
            cleaned: dict[str, dict[str, Any]] = {}
            for key, val in prov.items():
                if not isinstance(val, dict):
                    continue
                start = val.get("start")
                end = val.get("end")
                if not (isinstance(start, int) and isinstance(end, int)):
                    continue
                if start < 0 or end <= start or end > len(body):
                    continue
                snippet = val.get("snippet")
                if not isinstance(snippet, str) or snippet != body[start:end]:
                    snippet = body[start:end]
                cleaned[key] = {"start": start, "end": end, "snippet": snippet}
            payload["provenance"] = cleaned
        else:
            payload["provenance"] = {}

        # --- nested-null collapse ---
        for parent_key in ("classification", "redshift", "event", "follow_up", "localization"):
            v = payload.get(parent_key)
            if isinstance(v, dict):
                non_null = {k: vv for k, vv in v.items() if vv not in (None, [], {}, "")}
                if not non_null:
                    payload[parent_key] = None

        # --- classification: normalize aliases to canonical, drop unknown ---
        cls = payload.get("classification")
        if isinstance(cls, dict):
            raw = cls.get("classification")
            if isinstance(raw, str):
                canonical = normalize_classification(raw)
                if canonical is None:
                    payload["classification"] = None
                else:
                    cls["classification"] = canonical

        # --- follow_up.reference list coercion ---
        fu = payload.get("follow_up")
        if isinstance(fu, dict):
            ref = fu.get("reference")
            if isinstance(ref, dict):
                for rk, rv in list(ref.items()):
                    if isinstance(rv, list):
                        flat: list[str] = []
                        for item in rv:
                            if isinstance(item, dict):
                                for kk in ("id", "circular_id", "value", "name", "event_name"):
                                    if kk in item and item[kk] is not None:
                                        flat.append(str(item[kk]))
                                        break
                            elif item is not None:
                                flat.append(str(item))
                        ref[rk] = ",".join(flat) if flat else None
                    elif rv is not None and not isinstance(rv, str | int | float):
                        ref[rk] = str(rv)

        # --- calibration reference ---
        # Unconstrained, a model names the catalogue it actually read ("USNO
        # B-1.0"), which is not one the enum lists. That is a true statement
        # about one field, so it becomes the enum's own escape hatch rather
        # than failing validation and discarding the whole extraction with it.
        rows = payload.get("photometry")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                ref_value = row.get("calibration_reference")
                if isinstance(ref_value, str) and ref_value not in get_args(CalibrationReference):
                    row["calibration_reference"] = "Other"

        return payload
