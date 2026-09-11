"""Tests for the ExtractionStore."""

from __future__ import annotations

from pathlib import Path

from circex.schema import (
    CircularExtraction,
    Event,
    ExtractionMeta,
    Localization,
    Redshift,
)
from circex.server.store import ExtractionStore


def _make_extraction(
    circular_id: int = 1,
    event_name: str | None = "GRB 240101A",
    redshift: float | None = None,
    extractor: str = "regex-v1",
    prompt_version: str = "",
    ra: float | None = None,
    dec: float | None = None,
) -> CircularExtraction:
    return CircularExtraction(
        circular_id=circular_id,
        event=Event(event_name=event_name) if event_name else None,
        redshift=Redshift(redshift=redshift) if redshift is not None else None,
        localization=(Localization(ra=ra, dec=dec) if ra is not None and dec is not None else None),
        extraction_meta=ExtractionMeta(extractor=extractor, prompt_version=prompt_version),
    )


def test_put_get_roundtrip(tmp_path: Path) -> None:
    with ExtractionStore(tmp_path / "s.sqlite") as store:
        ex = _make_extraction(circular_id=42, redshift=0.5)
        store.put(ex)
        got = store.get(circular_id=42, extractor_id="regex-v1")
        assert got is not None
        assert got.circular_id == 42
        assert got.redshift is not None
        assert got.redshift.redshift == 0.5


def test_put_replaces_same_key(tmp_path: Path) -> None:
    with ExtractionStore(tmp_path / "s.sqlite") as store:
        store.put(_make_extraction(redshift=0.5))
        store.put(_make_extraction(redshift=0.8))  # same composite key
        got = store.get(circular_id=1, extractor_id="regex-v1")
        assert got is not None
        assert got.redshift is not None and got.redshift.redshift == 0.8
        assert store.count() == 1


def test_different_extractors_coexist(tmp_path: Path) -> None:
    with ExtractionStore(tmp_path / "s.sqlite") as store:
        store.put(_make_extraction(extractor="regex-v1"))
        store.put(_make_extraction(extractor="claude:claude-haiku-4-5"))
        assert store.count() == 2


def test_find_by_event_returns_matching(tmp_path: Path) -> None:
    with ExtractionStore(tmp_path / "s.sqlite") as store:
        store.put(_make_extraction(circular_id=1, event_name="GRB 240101A"))
        store.put(_make_extraction(circular_id=2, event_name="GRB 240101A"))
        store.put(_make_extraction(circular_id=3, event_name="GRB 240202B"))
        matches = list(store.find_by_event("GRB 240101A"))
        assert len(matches) == 2
        assert {m.circular_id for m in matches} == {1, 2}


def test_find_by_event_filters_by_extractor(tmp_path: Path) -> None:
    with ExtractionStore(tmp_path / "s.sqlite") as store:
        store.put(_make_extraction(circular_id=1, extractor="regex-v1"))
        store.put(_make_extraction(circular_id=2, extractor="claude:M"))
        regex_only = list(store.find_by_event("GRB 240101A", extractor_id="regex-v1"))
        assert len(regex_only) == 1
        assert regex_only[0].circular_id == 1


def test_event_name_list_uses_first(tmp_path: Path) -> None:
    with ExtractionStore(tmp_path / "s.sqlite") as store:
        ex = CircularExtraction(
            circular_id=99,
            event=Event(event_name=["GW170817", "AT2017gfo"]),
            extraction_meta=ExtractionMeta(extractor="test"),
        )
        store.put(ex)
        assert list(store.find_by_event("GW170817"))
        # The second name isn't indexed (only first is); document this.
        assert not list(store.find_by_event("AT2017gfo"))


def test_list_circular_ids_distinct(tmp_path: Path) -> None:
    with ExtractionStore(tmp_path / "s.sqlite") as store:
        store.put(_make_extraction(circular_id=1, extractor="regex-v1"))
        store.put(_make_extraction(circular_id=1, extractor="claude:M"))
        store.put(_make_extraction(circular_id=2, extractor="regex-v1"))
        assert store.list_circular_ids() == [1, 2]


# ---- cone search (P1 #3) ----


def test_cone_search_finds_nearby(tmp_path: Path) -> None:
    with ExtractionStore(tmp_path / "s.sqlite") as store:
        store.put(_make_extraction(circular_id=1, ra=150.0, dec=2.0))
        # ~3.6 arcsec away in dec (0.001 deg)
        hits = store.find_by_cone(150.0, 2.0, radius_arcsec=10.0)
        assert len(hits) == 1
        sep, ex = hits[0]
        assert ex.circular_id == 1
        assert sep < 1.0


def test_cone_search_excludes_far(tmp_path: Path) -> None:
    with ExtractionStore(tmp_path / "s.sqlite") as store:
        store.put(_make_extraction(circular_id=1, ra=150.0, dec=2.0))
        store.put(_make_extraction(circular_id=2, ra=150.0, dec=2.5))  # 0.5 deg away
        hits = store.find_by_cone(150.0, 2.0, radius_arcsec=30.0)
        assert [ex.circular_id for _, ex in hits] == [1]


def test_cone_search_sorts_by_separation(tmp_path: Path) -> None:
    with ExtractionStore(tmp_path / "s.sqlite") as store:
        store.put(_make_extraction(circular_id=1, ra=100.0, dec=0.002))  # ~7.2"
        store.put(_make_extraction(circular_id=2, ra=100.0, dec=0.0005))  # ~1.8"
        store.put(_make_extraction(circular_id=3, ra=100.0, dec=0.001))  # ~3.6"
        hits = store.find_by_cone(100.0, 0.0, radius_arcsec=60.0)
        assert [ex.circular_id for _, ex in hits] == [2, 3, 1]
        seps = [s for s, _ in hits]
        assert seps == sorted(seps)


def test_cone_search_ignores_null_localization(tmp_path: Path) -> None:
    with ExtractionStore(tmp_path / "s.sqlite") as store:
        store.put(_make_extraction(circular_id=1, ra=None, dec=None))  # no position
        store.put(_make_extraction(circular_id=2, ra=100.0, dec=0.0))
        hits = store.find_by_cone(100.0, 0.0, radius_arcsec=10.0)
        assert [ex.circular_id for _, ex in hits] == [2]


def test_cone_search_respects_limit(tmp_path: Path) -> None:
    with ExtractionStore(tmp_path / "s.sqlite") as store:
        for i in range(5):
            store.put(_make_extraction(circular_id=i, ra=100.0, dec=0.0001 * i))
        hits = store.find_by_cone(100.0, 0.0, radius_arcsec=60.0, limit=2)
        assert len(hits) == 2


def test_cone_search_migrates_legacy_db(tmp_path: Path) -> None:
    """A store created before the ra/dec columns gets them via ALTER on reopen."""
    import sqlite3

    db = tmp_path / "legacy.sqlite"
    # Build a legacy table without ra/dec.
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE extractions (
            circular_id INTEGER NOT NULL,
            extractor_id TEXT NOT NULL,
            model_id TEXT NOT NULL DEFAULT '',
            prompt_version TEXT NOT NULL DEFAULT '',
            primary_event TEXT,
            extraction_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (circular_id, extractor_id, model_id, prompt_version)
        );
        """
    )
    conn.commit()
    conn.close()

    # Reopening through ExtractionStore should add ra/dec and accept a put.
    with ExtractionStore(db) as store:
        store.put(_make_extraction(circular_id=1, ra=100.0, dec=0.0))
        hits = store.find_by_cone(100.0, 0.0, radius_arcsec=10.0)
        assert len(hits) == 1
