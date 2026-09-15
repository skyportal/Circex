"""Extractor protocol — common interface for regex, Claude, Ollama extractors.

A Circular is the input shape; CircularExtraction is the output. Every extractor
takes the same input and produces the same shape, so the eval harness can swap
implementations without changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from circex.schema import CircularExtraction


@dataclass(frozen=True)
class Circular:
    """One GCN circular as input to an extractor."""

    circular_id: int
    subject: str
    body: str
    event_id: str | None = None
    submitter: str | None = None
    created_on: int | None = None
    bibcode: str | None = None
    # Trigger time T0 for resolving relative offsets ("T+234s") into obs_mjd.
    # Caller-supplied (the GCN broker has it); NOT created_on, which is the
    # circular's submission time. None when unknown — relative epochs stay null.
    trigger_time: datetime | None = None

    @property
    def published_at(self) -> datetime | None:
        """When the circular was issued, which is the year a bare "Sep 25" means."""
        if self.created_on is None:
            return None
        try:
            return datetime.fromtimestamp(self.created_on / 1000, UTC)
        except (OverflowError, OSError, ValueError):
            return None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Circular:
        """Build a Circular from the raw archive JSON record."""
        try:
            cid = int(record["circularId"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"record missing/malformed circularId: {record!r}") from exc
        return cls(
            circular_id=cid,
            subject=str(record.get("subject") or ""),
            body=str(record.get("body") or ""),
            event_id=record.get("eventId") or None,
            submitter=record.get("submitter") or None,
            created_on=record.get("createdOn"),
            bibcode=record.get("bibcode") or None,
        )


@runtime_checkable
class Extractor(Protocol):
    """The interface every extractor implements.

    Implementations:
      - circex.extract.regex.RegexExtractor (Sprint 2)
      - circex.extract.llm.ClaudeExtractor (Sprint 3)
      - circex.extract.llm.OllamaExtractor (Sprint 3)
    """

    @property
    def extractor_id(self) -> str:
        """Stable identifier for this extractor (e.g., 'regex-v1')."""
        ...

    def extract(self, circular: Circular) -> CircularExtraction:
        """Extract structured CircularExtraction from one Circular."""
        ...
