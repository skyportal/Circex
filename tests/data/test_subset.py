"""Tests for circex.data.subset — stratification heuristics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from circex.data.subset import (
    StratifiedCircular,
    build_stratified_subset,
    classify_stratum,
    load_subset,
    save_subset,
)


def test_classify_multi_row_mag_table() -> None:
    body = (
        "Photometry table:\n"
        "Date    Filter Mag      Error\n"
        "Jan 1   r      19.42    0.05\n"
        "Jan 2   r      19.51    0.05\n"
    )
    assert classify_stratum(body) == "multi_row_mag_table"


def test_classify_photometric_upper_limit() -> None:
    body = "We obtained images with no detection; 3-sigma upper limit of r > 22.5."
    assert classify_stratum(body) == "photometric_upper_limit"


def test_classify_spec_z() -> None:
    body = (
        "Spectroscopy of the optical transient with the VLT yields a redshift "
        "of z = 0.95, consistent with the host galaxy."
    )
    assert classify_stratum(body) == "spec_z_classification"


def test_classify_gw_counterpart() -> None:
    body = "We report the optical counterpart to GW170817 detected with Swope."
    assert classify_stratum(body) == "gw_neutrino_counterpart"


def test_classify_single_mag() -> None:
    body = "The OT is detected at r = 18.42 in the Sloan filter."
    assert classify_stratum(body) == "single_row_mag"


def test_classify_returns_none_for_unmatched() -> None:
    body = "We observed the field and report no findings."
    assert classify_stratum(body) is None


def test_build_stratified_subset_per_stratum_cap() -> None:
    circulars: list[dict[str, Any]] = []
    for i in range(20):
        circulars.append({"circularId": i, "body": f"r = 18.{i:02d} in the Sloan filter."})
    subset = build_stratified_subset(circulars, per_stratum=5, seed=42)
    assert len(subset) == 5
    assert all(s.stratum == "single_row_mag" for s in subset)


def test_save_and_load_subset_roundtrip(tmp_path: Path) -> None:
    items = [
        StratifiedCircular(circular_id=1, stratum="single_row_mag"),
        StratifiedCircular(circular_id=2, stratum="multi_row_mag_table"),
    ]
    path = tmp_path / "subset.json"
    save_subset(items, path)
    loaded = load_subset(path)
    assert loaded == items
