"""OllamaExtractor tests with mocked ollama client (no live daemon)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from circex.extract.llm.ollama import OllamaExtractor
from circex.extract.protocol import Circular


def _mock_ollama(*responses: dict[str, object]) -> MagicMock:
    """Return a MagicMock whose .chat returns each given response in turn."""
    client = MagicMock()
    client.chat.side_effect = [{"message": {"content": json.dumps(r)}} for r in responses]
    return client


def test_extractor_id() -> None:
    # Pass model_id explicitly so the test is independent of the env-overridable
    # DEFAULT_OLLAMA_MODEL (currently the q4_K_M quantization of v0.2).
    ext = OllamaExtractor(model_id="mistral:7b-instruct-v0.2", client=_mock_ollama({}))
    assert ext.extractor_id == "ollama:mistral:7b-instruct-v0.2"


def test_extract_basic_success() -> None:
    client = _mock_ollama({"event": {"event_name": "GRB 240101A"}, "photometry": []})
    ext = OllamaExtractor(client=client)
    result = ext.extract(Circular(circular_id=7, subject="", body="body"))
    assert result.circular_id == 7
    assert result.event is not None and result.event.event_name == "GRB 240101A"
    assert client.chat.call_count == 1


def test_extract_repairs_on_validation_error() -> None:
    """First response invalid (type-level error in `redshift`); repair returns valid.

    The sanitizer can fix a number of cosmetic Mistral output quirks (bogus
    classification aliases, nested-null collapsing, list-shaped follow_up refs),
    so this test uses a type error the sanitizer cannot rescue — `redshift`
    given a string where the schema requires a float.
    """
    bad = {"redshift": {"redshift": "not-a-float"}}
    good = {"event": {"event_name": "GRB X"}}
    client = _mock_ollama(bad, good)
    ext = OllamaExtractor(client=client)
    result = ext.extract(Circular(circular_id=1, subject="", body="b"))
    assert client.chat.call_count == 2
    assert result.event is not None


def test_meta_populated_without_cost() -> None:
    client = _mock_ollama({})
    ext = OllamaExtractor(client=client)
    result = ext.extract(Circular(circular_id=1, subject="", body=""))
    assert result.extraction_meta.cost_usd is None
    assert result.extraction_meta.latency_ms is not None


# ---- notes preservation (P2 #11) ----


def test_ollama_preserves_llm_notes() -> None:
    """A redshift_bound note emitted by the model survives into the final meta."""
    payload = {
        "redshift": None,
        "extraction_meta": {"notes": ["redshift_bound: z <= 1.61"]},
    }
    client = _mock_ollama(payload)
    ext = OllamaExtractor(client=client)
    result = ext.extract(Circular(circular_id=216, subject="", body="z <= 1.61"))
    assert "redshift_bound: z <= 1.61" in result.extraction_meta.notes
    # Run-level fields are still overwritten by the extractor, not the model.
    assert result.extraction_meta.extractor == ext.extractor_id


def test_ollama_handles_missing_notes_gracefully() -> None:
    """No notes in the LLM payload → empty notes, no crash."""
    client = _mock_ollama({"event": {"event_name": "GRB X"}})
    ext = OllamaExtractor(client=client)
    result = ext.extract(Circular(circular_id=1, subject="", body="b"))
    assert result.extraction_meta.notes == []


def test_ollama_ignores_non_list_notes() -> None:
    """A malformed `notes` (string instead of list) is coerced to empty, not crashed."""
    payload = {"event": {"event_name": "GRB X"}, "extraction_meta": {"notes": "oops"}}
    client = _mock_ollama(payload)
    ext = OllamaExtractor(client=client)
    result = ext.extract(Circular(circular_id=1, subject="", body="b"))
    assert result.extraction_meta.notes == []
