"""End-to-end test for the markdown report generator."""

from __future__ import annotations

from pathlib import Path

from circex.eval.report import (
    evaluate_extractor,
    render_report,
    write_report,
)
from circex.schema import CircularExtraction


def _ext(circular_id: int, redshift: float | None = None) -> CircularExtraction:
    payload = {
        "circular_id": circular_id,
        "extraction_meta": {
            "extractor": "test",
            "cost_usd": 0.001,
            "latency_ms": 100.0,
            "tokens_in": 1000,
            "tokens_out": 100,
        },
    }
    if redshift is not None:
        payload["redshift"] = {"redshift": redshift}
    return CircularExtraction.model_validate(payload)


def test_render_report_includes_headline_table() -> None:
    gold = [_ext(1, 0.5), _ext(2, 1.0)]
    perfect = [_ext(1, 0.5), _ext(2, 1.0)]
    half_right = [_ext(1, 0.5), _ext(2, 99.0)]  # second mismatches

    reports = [
        evaluate_extractor("perfect-ext", perfect, gold),
        evaluate_extractor("half-right-ext", half_right, gold),
    ]
    md = render_report(reports)
    assert "Per-field F1" in md
    assert "perfect-ext" in md
    assert "half-right-ext" in md
    assert "redshift.redshift" in md
    # Perfect should achieve F1=1.000 on redshift
    assert "1.000" in md


def test_render_report_writes_to_file(tmp_path: Path) -> None:
    gold = [_ext(1, 0.5)]
    pred = [_ext(1, 0.5)]
    reports = [evaluate_extractor("e", pred, gold)]
    out = tmp_path / "subdir" / "eval.md"
    write_report(reports, out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Per-field F1" in text


def test_vidushi_delta_appears_when_present() -> None:
    gold = [_ext(1, 0.5), _ext(2, 1.0)]
    perfect = [_ext(1, 0.5), _ext(2, 1.0)]
    vidushi = [_ext(1, 0.5), _ext(2, 99.0)]  # gets 1 right, 1 wrong

    reports = [
        evaluate_extractor("perfect-ext", perfect, gold),
        evaluate_extractor("vidushi-mistral", vidushi, gold),
    ]
    md = render_report(reports)
    assert "Δ F1 vs Vidushi" in md
    # Perfect beats Vidushi by 1.0 - 0.667 ≈ 0.333 on redshift.
    assert "perfect-ext" in md.split("Δ F1 vs Vidushi")[1]


def test_failure_browser_section_present() -> None:
    gold = [_ext(1, 0.5)]
    bad = [_ext(1, 99.0)]
    reports = [evaluate_extractor("bad", bad, gold)]
    md = render_report(reports)
    assert "Failure-case browser" in md
    assert "redshift.redshift" in md.split("Failure-case browser")[1]
