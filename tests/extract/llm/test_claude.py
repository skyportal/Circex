"""ClaudeExtractor tests with mocked Anthropic client (no live API)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from circex.cache.llm import LLMCache
from circex.extract.llm.claude import ClaudeExtractor
from circex.extract.protocol import Circular


def _mock_anthropic(tool_input: dict[str, object]) -> MagicMock:
    """Build a MagicMock that mimics anthropic.Anthropic() with messages.create."""
    tool_use_block = SimpleNamespace(type="tool_use", name="submit_extraction", input=tool_input)
    usage = SimpleNamespace(
        input_tokens=1200,
        output_tokens=180,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    response = SimpleNamespace(content=[tool_use_block], usage=usage)
    client = MagicMock()
    client.messages.create.return_value = response
    return client


def test_extractor_id_and_model_id() -> None:
    ext = ClaudeExtractor(
        model_id="claude-haiku-4-5-20251001",
        client=_mock_anthropic({}),
    )
    assert ext.extractor_id == "claude:claude-haiku-4-5-20251001"
    assert ext.model_id == "claude-haiku-4-5-20251001"


def test_extract_calls_anthropic_with_tool_use() -> None:
    client = _mock_anthropic({"event": {"event_name": "GRB 240101A"}, "photometry": []})
    ext = ClaudeExtractor(model_id="claude-haiku-4-5-20251001", client=client)

    result = ext.extract(Circular(circular_id=42, subject="GRB", body="Body"))

    assert result.circular_id == 42
    assert result.event is not None and result.event.event_name == "GRB 240101A"
    assert client.messages.create.called

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    assert kwargs["tool_choice"]["name"] == "submit_extraction"
    assert kwargs["tools"][0]["name"] == "submit_extraction"
    # System block has cache_control: ephemeral.
    assert kwargs["system"][0]["cache_control"]["type"] == "ephemeral"


def test_extract_populates_meta() -> None:
    client = _mock_anthropic({})
    ext = ClaudeExtractor(model_id="claude-haiku-4-5-20251001", client=client)
    result = ext.extract(Circular(circular_id=1, subject="", body="body"))
    meta = result.extraction_meta
    assert meta.tokens_in == 1200
    assert meta.tokens_out == 180
    assert meta.prompt_version is not None
    assert meta.cost_usd is not None and meta.cost_usd > 0
    assert meta.cache_hit is False


def test_cache_hit_returns_cached_without_api_call(tmp_path: Path) -> None:
    cache = LLMCache(tmp_path / "c.sqlite")
    client = _mock_anthropic({"event": {"event_name": "GRB X"}})
    ext = ClaudeExtractor(model_id="claude-haiku-4-5-20251001", cache=cache, client=client)

    circ = Circular(circular_id=99, subject="s", body="b")
    ext.extract(circ)
    assert client.messages.create.call_count == 1

    # second call hits the cache
    result2 = ext.extract(circ)
    assert client.messages.create.call_count == 1
    assert result2.extraction_meta.cache_hit is True


def test_raises_when_no_tool_use_block_in_response() -> None:
    text_block = SimpleNamespace(type="text", text="ignore me")
    response = SimpleNamespace(content=[text_block], usage=SimpleNamespace())
    client = MagicMock()
    client.messages.create.return_value = response
    ext = ClaudeExtractor(model_id="claude-haiku-4-5-20251001", client=client)
    with pytest.raises(ValueError, match="submit_extraction"):
        ext.extract(Circular(circular_id=1, subject="", body=""))


def test_claude_preserves_llm_notes() -> None:
    """A redshift_bound note in tool_use input survives into the final meta."""
    client = _mock_anthropic(
        {"redshift": None, "extraction_meta": {"notes": ["redshift_bound: z <= 1.61"]}}
    )
    ext = ClaudeExtractor(model_id="claude-haiku-4-5-20251001", client=client)
    result = ext.extract(Circular(circular_id=216, subject="", body="z <= 1.61"))
    assert "redshift_bound: z <= 1.61" in result.extraction_meta.notes
    # Run-level fields are still set by the runner, not the model.
    assert result.extraction_meta.extractor == ext.extractor_id
    assert result.extraction_meta.model_id == "claude-haiku-4-5-20251001"


def test_claude_no_notes_yields_empty_list() -> None:
    client = _mock_anthropic({"event": {"event_name": "GRB X"}})
    ext = ClaudeExtractor(model_id="claude-haiku-4-5-20251001", client=client)
    result = ext.extract(Circular(circular_id=1, subject="", body="b"))
    assert result.extraction_meta.notes == []
