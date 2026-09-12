"""Dump Pydantic schema modules to JSON Schema files for the upstream PR.

Run via `circex schema-dump --out schemas/`. The three output files conform to
gcn-schema conventions ($id URL, JSON Schema draft 2020-12, every property has a
description). The Photometry file is the upstream extension patch; SpectralLines
and Classification are net-new.

Each artifact carries a `version` (semver, SCHEMA_VERSION below) so downstream
consumers — ICARE/SkyPortal and the upstream gcn-schema PR — can pin to a
version and detect breakage. CI requires SCHEMA_VERSION to be bumped whenever
the generated artifacts change (and a separate gate requires the artifacts to
match the models). Bump rules: PATCH for additive/descriptive changes, MINOR
for new optional fields, MAJOR for removed/renamed/retyped fields or tightened
enums (anything that can break an existing consumer).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from circex.schema.classification import Classification
from circex.schema.photometry import PhotometryExt
from circex.schema.spectral_lines import SpectralLine

GCN_SCHEMA_ID_PREFIX = "https://gcn.nasa.gov/schema/main/gcn/notices/core"

# Semantic version of the dumped JSON Schema artifacts. Bump on every change to
# the generated schemas (CI enforces this). Distinct from the package version in
# pyproject.toml: code-only changes do not bump this; schema-shape changes do.
SCHEMA_VERSION = "0.6.0"


def _attach_gcn_id(schema: dict[str, Any], title: str, filename: str) -> dict[str, Any]:
    """Prepend gcn-schema-style $id + $schema + version + title to a Pydantic dump."""
    schema.pop("$defs", None)  # SpectralLine $defs handled separately below
    out = {
        "$id": f"{GCN_SCHEMA_ID_PREFIX}/{filename}",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "version": SCHEMA_VERSION,
        "title": title,
    }
    out.update(schema)
    return out


def dump_photometry() -> dict[str, Any]:
    """Photometry.schema.json — the extension patch."""
    raw = PhotometryExt.model_json_schema()
    return _attach_gcn_id(raw, "Photometry: Magnitude Schema", "Photometry.schema.json")


def dump_spectral_lines() -> dict[str, Any]:
    """SpectralLines.schema.json — NEW. Array of SpectralLine entries."""
    line_schema = SpectralLine.model_json_schema()
    line_schema.pop("title", None)
    schema = {
        "type": "array",
        "items": line_schema,
        "description": "Identified spectral lines from one optical-spectroscopy observation.",
    }
    return _attach_gcn_id(schema, "SpectralLines", "SpectralLines.schema.json")


def dump_classification() -> dict[str, Any]:
    """Classification.schema.json — NEW. Uses the time-domain taxonomy controlled vocab."""
    raw = Classification.model_json_schema()
    return _attach_gcn_id(raw, "Classification", "Classification.schema.json")


def write_all(out_dir: Path) -> list[Path]:
    """Write all three schemas + a VERSION file to out_dir. Returns paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, payload in (
        ("Photometry.schema.json", dump_photometry()),
        ("SpectralLines.schema.json", dump_spectral_lines()),
        ("Classification.schema.json", dump_classification()),
    ):
        path = out_dir / filename
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path)
    # Single source of truth for the artifact version; CI keys the version-bump
    # gate off changes to this file.
    version_path = out_dir / "VERSION"
    version_path.write_text(SCHEMA_VERSION + "\n", encoding="utf-8")
    written.append(version_path)
    return written
