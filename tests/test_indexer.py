"""Tests for circex.db.indexer (ported from GCNMCP tests/test_indexer.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from circex.db.connection import get_connection
from circex.db.indexer import (
    ingest_path,
    iter_json_records,
    parse_circular_id,
    sha1_text,
    upsert_circular,
)


def make_record(
    circular_id: int = 43493,
    subject: str = "GRB 260120B: Swift-BAT refined analysis",
    body: str = "Using T-769 to T+303 sec, we report further analysis of BAT GRB 260120B.",
    event_id: str | None = "GRB 260120B",
    created_on: int = 1769036892952,
    submitter: str = "Test Submitter <test@example.com>",
) -> dict[str, Any]:
    return {
        "circularId": circular_id,
        "subject": subject,
        "eventId": event_id,
        "createdOn": created_on,
        "submitter": submitter,
        "format": "text/plain",
        "body": body,
    }


def test_sha1_is_deterministic() -> None:
    assert sha1_text("hello") == sha1_text("hello")


def test_sha1_sensitive_to_input() -> None:
    assert sha1_text("abc") != sha1_text("abcd")


def test_sha1_returns_40_char_hex() -> None:
    h = sha1_text("anything")
    assert len(h) == 40
    assert all(c in "0123456789abcdef" for c in h)


@pytest.mark.parametrize(
    "value, expected_raw, expected_int",
    [
        (43493, "43493", 43493),
        (43493.0, "43493", 43493),
        ("43493", "43493", 43493),
        ("43493.0", "43493", 43493),
        (None, None, None),
        ("", None, None),
    ],
)
def test_parse_circular_id_standard_cases(
    value: Any, expected_raw: str | None, expected_int: int | None
) -> None:
    raw, integer = parse_circular_id(value)
    assert raw == expected_raw
    assert integer == expected_int


def test_parse_circular_id_non_numeric_string() -> None:
    raw, integer = parse_circular_id("CIRCULAR-XYZ")
    assert raw == "CIRCULAR-XYZ"
    assert integer is None


def test_upsert_inserts_into_circulars(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.sqlite")
    upsert_circular(conn, make_record())
    conn.commit()
    row = conn.execute("SELECT * FROM circulars WHERE circular_id_raw = ?", ("43493",)).fetchone()
    assert row is not None
    assert row["primary_event_norm"] == "GRB260120B"
    assert row["circular_id_int"] == 43493
    assert row["extraction_source"] == "eventId"
    conn.close()


def test_upsert_populates_circular_events(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.sqlite")
    upsert_circular(conn, make_record())
    conn.commit()
    rows = conn.execute(
        "SELECT * FROM circular_events WHERE circular_id_raw = ?", ("43493",)
    ).fetchall()
    primary_rows = [r for r in rows if r["is_primary"] == 1]
    assert len(primary_rows) == 1
    assert primary_rows[0]["event_norm"] == "GRB260120B"
    conn.close()


def test_upsert_populates_fts(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.sqlite")
    upsert_circular(conn, make_record())
    conn.commit()
    rows = conn.execute(
        "SELECT * FROM circulars_fts WHERE circulars_fts MATCH ?", ("refined",)
    ).fetchall()
    assert len(rows) == 1
    conn.close()


def test_upsert_skips_unchanged_record(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.sqlite")
    record = make_record()
    upsert_circular(conn, record)
    conn.commit()
    upsert_circular(conn, record)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM circulars").fetchone()[0] == 1
    conn.close()


def test_upsert_updates_subject_when_record_changes(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.sqlite")
    record = make_record()
    upsert_circular(conn, record)
    conn.commit()
    updated = dict(record, subject="GRB 260120B: Updated subject")
    upsert_circular(conn, updated)
    conn.commit()
    row = conn.execute(
        "SELECT subject FROM circulars WHERE circular_id_raw = ?", ("43493",)
    ).fetchone()
    assert row["subject"] == "GRB 260120B: Updated subject"
    conn.close()


def test_upsert_extracts_event_from_body_as_last_resort(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.sqlite")
    record = make_record(
        event_id=None,
        subject="Optical follow-up observations",
        body="We observed GRB 260120B and found a fading optical source.",
    )
    upsert_circular(conn, record)
    conn.commit()
    row = conn.execute(
        "SELECT primary_event_norm, extraction_source FROM circulars WHERE circular_id_raw = ?",
        ("43493",),
    ).fetchone()
    assert row["primary_event_norm"] == "GRB260120B"
    assert row["extraction_source"] == "body"
    conn.close()


def test_upsert_raises_on_missing_circular_id(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.sqlite")
    record = make_record()
    del record["circularId"]
    with pytest.raises(ValueError, match="circularId"):
        upsert_circular(conn, record)
    conn.close()


def test_iter_single_json_object(tmp_path: Path) -> None:
    path = tmp_path / "one.json"
    path.write_text(json.dumps(make_record()), encoding="utf-8")
    records = list(iter_json_records(path))
    assert len(records) == 1
    assert records[0]["circularId"] == 43493


def test_iter_json_array(tmp_path: Path) -> None:
    path = tmp_path / "many.json"
    recs = [make_record(1), make_record(2), make_record(3)]
    path.write_text(json.dumps(recs), encoding="utf-8")
    records = list(iter_json_records(path))
    assert {r["circularId"] for r in records} == {1, 2, 3}


def test_iter_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    lines = "\n".join(json.dumps(make_record(i)) for i in [10, 20, 30])
    path.write_text(lines, encoding="utf-8")
    records = list(iter_json_records(path))
    assert {r["circularId"] for r in records} == {10, 20, 30}


def test_iter_raises_for_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("col1,col2\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_json_records(path))


def test_iter_raises_for_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_json_records(tmp_path / "nonexistent_dir"))


def test_ingest_returns_correct_count(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps([make_record(i) for i in range(5)]), encoding="utf-8")
    assert ingest_path(db_path, json_path) == 5


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(make_record()), encoding="utf-8")
    ingest_path(db_path, json_path)
    ingest_path(db_path, json_path)
    conn = get_connection(db_path)
    assert conn.execute("SELECT COUNT(*) FROM circulars").fetchone()[0] == 1
    conn.close()
