"""Tests for the eval runner."""

from __future__ import annotations

from circex.eval.runner import run_extractor
from circex.extract.protocol import Circular
from circex.schema import CircularExtraction, ExtractionMeta


class _MockExtractor:
    extractor_id = "mock"

    def __init__(self, cost_per_call: float = 0.0, fail_at: int | None = None) -> None:
        self.cost = cost_per_call
        self.fail_at = fail_at
        self.calls = 0

    def extract(self, circular: Circular) -> CircularExtraction:
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError("simulated failure")
        return CircularExtraction(
            circular_id=circular.circular_id,
            extraction_meta=ExtractionMeta(extractor=self.extractor_id, cost_usd=self.cost),
        )


def _circulars(n: int) -> list[Circular]:
    return [Circular(circular_id=i, subject="", body="") for i in range(n)]


def test_runner_completes_all() -> None:
    ext = _MockExtractor()
    results, stats = run_extractor(ext, _circulars(5))
    assert len(results) == 5
    assert stats.n_succeeded == 5
    assert stats.n_failed == 0
    assert not stats.aborted_for_cost


def test_runner_skips_failed_circulars() -> None:
    ext = _MockExtractor(fail_at=3)
    results, stats = run_extractor(ext, _circulars(5))
    assert len(results) == 4
    assert stats.n_succeeded == 4
    assert stats.n_failed == 1


def test_runner_honors_cost_cap() -> None:
    ext = _MockExtractor(cost_per_call=0.50)
    results, stats = run_extractor(ext, _circulars(10), max_usd=1.0)
    # Cap triggers AFTER cost accumulates past 1.0, then halts on next iteration.
    # Two calls at $0.50 each = $1.00, then the third loop check sees cost >= 1.0 and stops.
    assert stats.aborted_for_cost is True
    assert len(results) == 2
    assert stats.cost_usd == 1.0
