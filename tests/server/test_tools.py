"""Tests for the MCP tool implementations (server-side)."""

from __future__ import annotations

from pathlib import Path

import pytest

from circex.schema import (
    CircularExtraction,
    Classification,
    Event,
    ExtractionMeta,
    Localization,
    PhotometryExt,
    Redshift,
)
from circex.server.registry import ToolContext, dispatch
from circex.server.store import ExtractionStore


def _full_extraction(
    circular_id: int,
    event: str,
    ra: float | None = None,
    dec: float | None = None,
) -> CircularExtraction:
    return CircularExtraction(
        circular_id=circular_id,
        event=Event(event_name=event),
        redshift=Redshift(redshift=0.215, redshift_type="host"),
        classification=Classification(classification="Ic-BL"),
        localization=(Localization(ra=ra, dec=dec) if ra is not None and dec is not None else None),
        photometry=[PhotometryExt(filter="r", mag=18.5, mag_system="AB")],
        extraction_meta=ExtractionMeta(extractor="regex-v1"),
    )


@pytest.fixture
def populated_ctx(tmp_path: Path) -> ToolContext:
    store = ExtractionStore(tmp_path / "s.sqlite")
    store.put(_full_extraction(1, "GRB 240101A"))
    store.put(_full_extraction(2, "GRB 240202B"))
    return ToolContext(store=store)


def test_get_redshift_returns_first_match(populated_ctx: ToolContext) -> None:
    result = dispatch(populated_ctx, "get_redshift", {"event": "GRB 240101A"})
    assert result is not None
    assert result["redshift"] == 0.215


def test_get_redshift_unknown_event_returns_none(populated_ctx: ToolContext) -> None:
    assert dispatch(populated_ctx, "get_redshift", {"event": "GRB XYZ"}) is None


def test_get_photometry_returns_all_rows(populated_ctx: ToolContext) -> None:
    rows = dispatch(populated_ctx, "get_photometry", {"event": "GRB 240101A"})
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["filter"] == "r"
    assert rows[0]["mag"] == 18.5


def test_get_classification_returns_canonical(populated_ctx: ToolContext) -> None:
    result = dispatch(populated_ctx, "get_classification", {"event": "GRB 240101A"})
    assert result is not None
    assert result["classification"] == "Ic-BL"


def test_get_redshift_missing_event_arg_errors(populated_ctx: ToolContext) -> None:
    with pytest.raises(ValueError, match="event"):
        dispatch(populated_ctx, "get_redshift", {})


def test_extract_properties_without_default_extractor_errors(
    populated_ctx: ToolContext,
) -> None:
    """No default_extractor in ctx; missing circular_id 999."""
    with pytest.raises(ValueError, match="no default extractor"):
        dispatch(populated_ctx, "extract_properties", {"circular_id": 999})


def test_extract_properties_serves_from_store_when_cached(
    populated_ctx: ToolContext,
) -> None:
    """Stored extractions returned without re-extracting (no extractor in ctx)."""

    class _StubExtractor:
        extractor_id = "regex-v1"
        model_id = ""
        prompt_version = ""

        def extract(self, _: object) -> CircularExtraction:
            raise AssertionError("should not be called when cached")

    populated_ctx.default_extractor = _StubExtractor()
    result = dispatch(populated_ctx, "extract_properties", {"circular_id": 1})
    assert result["circular_id"] == 1
    assert result["event"]["event_name"] == "GRB 240101A"


# ---- extract_text (P0 #1) ----


class _RecordingExtractor:
    """Captures the Circular it was handed and returns a canned extraction."""

    extractor_id = "regex-v1"
    model_id = ""
    prompt_version = ""

    def __init__(self) -> None:
        self.last_circular: object = None

    def extract(self, circular: object) -> CircularExtraction:
        self.last_circular = circular
        cid = getattr(circular, "circular_id", 0)
        return CircularExtraction(
            circular_id=cid,
            event=Event(event_name="GRB 250601A"),
            extraction_meta=ExtractionMeta(extractor="regex-v1"),
        )


def test_extract_text_requires_default_extractor(populated_ctx: ToolContext) -> None:
    populated_ctx.default_extractor = None
    with pytest.raises(ValueError, match="requires a default extractor"):
        dispatch(populated_ctx, "extract_text", {"body": "GRB 250601A z = 0.5"})


def test_extract_text_requires_body(populated_ctx: ToolContext) -> None:
    populated_ctx.default_extractor = _RecordingExtractor()
    with pytest.raises(ValueError, match="body"):
        dispatch(populated_ctx, "extract_text", {"subject": "no body here"})


def test_extract_text_extracts_from_raw_body(populated_ctx: ToolContext) -> None:
    ext = _RecordingExtractor()
    populated_ctx.default_extractor = ext
    result = dispatch(
        populated_ctx,
        "extract_text",
        {
            "circular_id": 99999,
            "subject": "GRB 250601A",
            "body": "z = 0.5",
            "event_id": "GRB 250601A",
        },
    )
    assert result["circular_id"] == 99999
    assert result["event"]["event_name"] == "GRB 250601A"
    # The body was forwarded verbatim into the Circular (no archive lookup).
    assert getattr(ext.last_circular, "body") == "z = 0.5"
    assert getattr(ext.last_circular, "event_id") == "GRB 250601A"


def test_extract_text_persists_real_id_to_store(populated_ctx: ToolContext) -> None:
    populated_ctx.default_extractor = _RecordingExtractor()
    dispatch(populated_ctx, "extract_text", {"circular_id": 12345, "body": "GRB 250601A"})
    stored = populated_ctx.store.get(circular_id=12345, extractor_id="regex-v1")
    assert stored is not None
    assert stored.circular_id == 12345


def test_extract_text_sentinel_id_not_persisted(populated_ctx: ToolContext) -> None:
    """circular_id defaults to 0; sentinel rows must not collide in the store."""
    populated_ctx.default_extractor = _RecordingExtractor()
    result = dispatch(populated_ctx, "extract_text", {"body": "GRB 250601A"})
    assert result["circular_id"] == 0
    # Nothing persisted under the sentinel id.
    assert populated_ctx.store.get(circular_id=0, extractor_id="regex-v1") is None


def test_extract_text_defaults_subject_and_event_id(populated_ctx: ToolContext) -> None:
    ext = _RecordingExtractor()
    populated_ctx.default_extractor = ext
    dispatch(populated_ctx, "extract_text", {"body": "just a body"})
    assert getattr(ext.last_circular, "subject") == ""
    assert getattr(ext.last_circular, "event_id") is None


def test_extract_text_rejects_non_int_circular_id(populated_ctx: ToolContext) -> None:
    populated_ctx.default_extractor = _RecordingExtractor()
    with pytest.raises(ValueError, match="circular_id"):
        dispatch(populated_ctx, "extract_text", {"body": "b", "circular_id": "99"})


def test_extract_text_rejects_non_str_subject(populated_ctx: ToolContext) -> None:
    populated_ctx.default_extractor = _RecordingExtractor()
    with pytest.raises(ValueError, match="subject"):
        dispatch(populated_ctx, "extract_text", {"body": "b", "subject": 42})


def test_extract_text_rejects_non_str_event_id(populated_ctx: ToolContext) -> None:
    populated_ctx.default_extractor = _RecordingExtractor()
    with pytest.raises(ValueError, match="event_id"):
        dispatch(populated_ctx, "extract_text", {"body": "b", "event_id": 7})


def test_extract_text_idempotent_via_caching_extractor(populated_ctx: ToolContext) -> None:
    """Re-delivered Kafka message (same body+id) must not re-invoke the extractor.

    Simulates the extractor's own LLM cache: a body-keyed cache means the
    second call returns the first result without re-running extraction.
    """

    class _CachingExtractor:
        extractor_id = "regex-v1"
        model_id = ""
        prompt_version = ""

        def __init__(self) -> None:
            self.calls = 0
            self._cache: dict[tuple[int, str], CircularExtraction] = {}

        def extract(self, circular: object) -> CircularExtraction:
            key = (getattr(circular, "circular_id"), getattr(circular, "body"))
            if key in self._cache:
                return self._cache[key]
            self.calls += 1
            ex = CircularExtraction(
                circular_id=getattr(circular, "circular_id"),
                extraction_meta=ExtractionMeta(extractor="regex-v1"),
            )
            self._cache[key] = ex
            return ex

    ext = _CachingExtractor()
    populated_ctx.default_extractor = ext
    args = {"circular_id": 55555, "body": "duplicate kafka payload"}
    dispatch(populated_ctx, "extract_text", args)
    dispatch(populated_ctx, "extract_text", args)
    assert ext.calls == 1  # second delivery served from the extractor's cache


def test_extract_text_resolves_relative_epoch_from_trigger_time(
    populated_ctx: ToolContext,
) -> None:
    from circex.extract.regex import RegexExtractor

    populated_ctx.default_extractor = RegexExtractor()
    result = dispatch(
        populated_ctx,
        "extract_text",
        {
            "circular_id": 80000,
            "body": "We observed at T+1 h and measured r = 19.5 mag.",
            "trigger_time": "2024-01-01T00:00:00Z",
        },
    )
    rows = [p for p in result["photometry"] if p.get("obs_mjd") is not None]
    assert rows, "expected the relative offset to resolve to an obs_mjd"


def test_extract_text_rejects_bad_trigger_time(populated_ctx: ToolContext) -> None:
    from circex.extract.regex import RegexExtractor

    populated_ctx.default_extractor = RegexExtractor()
    with pytest.raises(ValueError, match="trigger_time"):
        dispatch(
            populated_ctx,
            "extract_text",
            {"body": "r = 19.5 mag", "trigger_time": "not-a-date"},
        )


def test_extract_text_provenance_survives_to_serialized_output(
    populated_ctx: ToolContext,
) -> None:
    """Real regex extractor through extract_text: provenance lands in the dict."""
    from circex.extract.regex import RegexExtractor

    populated_ctx.default_extractor = RegexExtractor()
    result = dispatch(
        populated_ctx,
        "extract_text",
        {"circular_id": 70000, "body": "Spectroscopy gives z = 0.198 for the host."},
    )
    assert "provenance" in result
    assert "redshift" in result["provenance"]


# ---- search_by_position (P1 #3) ----


@pytest.fixture
def positioned_ctx(tmp_path: Path) -> ToolContext:
    store = ExtractionStore(tmp_path / "p.sqlite")
    store.put(_full_extraction(1, "AT2024aaa", ra=150.0, dec=2.0))
    store.put(_full_extraction(2, "AT2024bbb", ra=150.0, dec=2.5))  # 0.5 deg away
    return ToolContext(store=store)


def test_search_by_position_finds_within_radius(positioned_ctx: ToolContext) -> None:
    hits = dispatch(
        positioned_ctx,
        "search_by_position",
        {"ra": 150.0, "dec": 2.0, "radius_arcsec": 10.0},
    )
    assert len(hits) == 1
    assert hits[0]["circular_id"] == 1
    assert hits[0]["event_name"] == "AT2024aaa"
    assert hits[0]["separation_arcsec"] < 1.0


def test_search_by_position_excludes_outside_radius(positioned_ctx: ToolContext) -> None:
    hits = dispatch(
        positioned_ctx,
        "search_by_position",
        {"ra": 150.0, "dec": 2.0, "radius_arcsec": 30.0},
    )
    assert [h["circular_id"] for h in hits] == [1]


def test_search_by_position_requires_numeric_args(positioned_ctx: ToolContext) -> None:
    with pytest.raises(ValueError, match="radius_arcsec"):
        dispatch(
            positioned_ctx,
            "search_by_position",
            {"ra": 150.0, "dec": 2.0, "radius_arcsec": "wide"},
        )


def test_search_by_position_wide_cone_sorted_by_separation(
    positioned_ctx: ToolContext,
) -> None:
    hits = dispatch(
        positioned_ctx,
        "search_by_position",
        {"ra": 150.0, "dec": 2.0, "radius_arcsec": 3600.0},
    )
    assert [h["circular_id"] for h in hits] == [1, 2]
    seps = [h["separation_arcsec"] for h in hits]
    assert seps == sorted(seps)
