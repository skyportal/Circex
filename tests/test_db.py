"""Tests for circex.db.connection (ported from GCNMCP tests/test_db.py)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from circex.db.connection import get_connection


def _open(tmp_path: Path) -> sqlite3.Connection:
    return get_connection(tmp_path / "test.sqlite")


def _insert_circular(
    conn: sqlite3.Connection,
    circular_id_raw: str = "43493",
    circular_id_int: int = 43493,
    record_hash: str = "abc",
) -> None:
    conn.execute(
        """
        INSERT INTO circulars (
            circular_id_raw, circular_id_int,
            subject, body, created_on, submitter, format,
            raw_event_id, primary_event_raw, primary_event_norm,
            extraction_source, llm_confidence, record_hash
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            circular_id_raw,
            circular_id_int,
            "GRB 260120B: Swift-BAT refined analysis",
            "Further analysis of BAT GRB 260120B.",
            1769036892952,
            "Tester",
            "text/plain",
            "GRB 260120B",
            "GRB 260120B",
            "GRB260120B",
            "eventId",
            None,
            record_hash,
        ),
    )
    conn.commit()


def test_returns_sqlite_connection(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    try:
        assert isinstance(conn, sqlite3.Connection)
    finally:
        conn.close()


def test_uses_row_factory(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    try:
        row = conn.execute("SELECT 42 AS answer").fetchone()
        assert isinstance(row, sqlite3.Row)
        assert row["answer"] == 42
    finally:
        conn.close()


def test_journal_mode_is_wal(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    try:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode == "wal"
    finally:
        conn.close()


def test_foreign_keys_enabled(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    try:
        assert conn.execute("PRAGMA foreign_keys;").fetchone()[0] == 1
    finally:
        conn.close()


def test_schema_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    c1 = get_connection(db_path)
    c1.close()
    c2 = get_connection(db_path)
    c2.close()


@pytest.mark.parametrize("table", ["circulars", "circular_events", "circulars_fts"])
def test_table_exists(tmp_path: Path, table: str) -> None:
    conn = _open(tmp_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        assert row is not None
    finally:
        conn.close()


@pytest.mark.parametrize(
    "idx",
    [
        "idx_circulars_circular_id_int",
        "idx_circulars_primary_event_norm",
        "idx_circular_events_event_norm",
        "idx_circulars_created_on",
    ],
)
def test_index_exists(tmp_path: Path, idx: str) -> None:
    conn = _open(tmp_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name=?", (idx,)
        ).fetchone()
        assert row is not None
    finally:
        conn.close()


def test_circulars_has_all_expected_columns(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(circulars)").fetchall()}
        required = {
            "circular_id_raw",
            "circular_id_int",
            "subject",
            "body",
            "created_on",
            "submitter",
            "format",
            "raw_event_id",
            "primary_event_raw",
            "primary_event_norm",
            "extraction_source",
            "llm_confidence",
            "record_hash",
        }
        assert required <= cols
    finally:
        conn.close()


def test_can_insert_and_retrieve_circular(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    try:
        _insert_circular(conn)
        row = conn.execute(
            "SELECT * FROM circulars WHERE circular_id_raw = ?", ("43493",)
        ).fetchone()
        assert row is not None
        assert row["primary_event_norm"] == "GRB260120B"
        assert row["circular_id_int"] == 43493
    finally:
        conn.close()


def test_circular_id_raw_is_unique(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    try:
        _insert_circular(conn)
        with pytest.raises(sqlite3.IntegrityError):
            _insert_circular(conn)
    finally:
        conn.close()


def test_fts_match_on_subject(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    try:
        conn.execute(
            "INSERT INTO circulars_fts (circular_id_raw, subject, body) VALUES (?,?,?)",
            ("43493", "GRB 260120B Swift-BAT refined analysis", "Further analysis."),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT * FROM circulars_fts WHERE circulars_fts MATCH ?", ("refined",)
        ).fetchall()
        assert len(rows) == 1
    finally:
        conn.close()


def test_fts_no_match_returns_empty(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    try:
        conn.execute(
            "INSERT INTO circulars_fts (circular_id_raw, subject, body) VALUES (?,?,?)",
            ("43493", "Some subject", "Some body text."),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT * FROM circulars_fts WHERE circulars_fts MATCH ?", ("xyznonexistent",)
        ).fetchall()
        assert rows == []
    finally:
        conn.close()
