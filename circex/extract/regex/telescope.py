"""Finds the telescope a circular observed with.

Precision first: a name from the alias map is taken wherever it appears, and
otherwise only an explicit "<name> telescope" construction is trusted. The text
is returned as written, since a downstream consumer canonicalizes separately.
"""

from __future__ import annotations

import re
from functools import lru_cache

from circex.data.telescopes import _alias_source
from circex.schema.span import Span


@lru_cache(maxsize=1)
def _alias_pattern() -> re.Pattern[str]:
    """One alternation over every known spelling, longest first."""
    spellings = set()
    for canonical, aliases in (_alias_source().get("telescopes") or {}).items():
        spellings.add(canonical)
        spellings.update(aliases or [])
    ordered = sorted((a for a in spellings if len(a) > 2), key=len, reverse=True)
    # An acronym is written in capitals; matching it case-insensitively turns
    # NOT (Nordic Optical Telescope) into the English word "not".
    acronyms = [a for a in ordered if a.isupper()]
    words = [a for a in ordered if not a.isupper()]
    parts = []
    if acronyms:
        parts.append("(?:" + "|".join(re.escape(a) for a in acronyms) + ")")
    if words:
        parts.append("(?i:" + "|".join(re.escape(a) for a in words) + ")")
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b")


# "with the 2.5-m Nordic Optical Telescope", "at the Palomar 60-inch telescope".
# The name is bounded to a few words so a whole clause is never swallowed.
_NAMED_RE = re.compile(
    r"\b(?:with|at|using|on|from)\s+the\s+"
    r"((?:[A-Z][\w.+-]*|\d[\w.'\"-]*)(?:[ -](?:[A-Z][\w.+-]*|\d[\w.'\"-]*|of|de)){0,4})"
    r"\s+(?:telescope|observatory)\b",
    re.IGNORECASE,
)

# A telescope named without the word "telescope": "images from Keck", where Keck
# is only trusted because the alias map knows it.
_PREPOSITION_RE = re.compile(r"\b(?:with|at|using|on|from)\s+(?:the\s+)?", re.IGNORECASE)


def _clean(name: str) -> str:
    """Drop a leading article the alias map carries for prose matching."""
    return re.sub(r"^the\s+", "", name.strip(), flags=re.IGNORECASE)


# "calibrated against Pan-STARRS", "relative to the SDSS catalog": the survey
# supplying the reference magnitudes is not the telescope that observed. Bounded
# to the clause so the guard cannot reach back into a previous sentence.
_CALIBRATION_CLAUSE_RE = re.compile(
    r"(?:calibrat\w*|photometr\w+\s+zero\s*point|relative\s+to|with\s+respect\s+to"
    r"|compared\s+(?:to|with)|reference\s+(?:star|catalog)\w*|against)"
    r"\b[^.;\n]{0,80}$",
    re.IGNORECASE,
)


def _in_calibration_clause(text: str, start: int) -> bool:
    return _CALIBRATION_CLAUSE_RE.search(text[max(0, start - 100) : start]) is not None


def parse_telescope_with_span(text: str) -> tuple[str, Span] | None:
    """The telescope the observation used, as written, or None."""
    alias = next(
        (m for m in _alias_pattern().finditer(text) if not _in_calibration_clause(text, m.start())),
        None,
    )
    named = next(
        (m for m in _NAMED_RE.finditer(text) if not _in_calibration_clause(text, m.start(1))),
        None,
    )

    # A known name wins wherever the two overlap: the generic pattern tends to
    # take a qualifier with it ("2.5-m Nordic Optical"). Otherwise whichever is
    # stated first, since a later mention is usually a comparison.
    if alias is not None and named is not None:
        overlaps = alias.start() < named.end(1) and named.start(1) < alias.end()
        if overlaps or alias.start() <= named.start(1):
            named = None
    if alias is not None and named is None:
        return _clean(alias.group(0)), Span(
            start=alias.start(), end=alias.end(), snippet=alias.group(0)
        )
    if named is not None:
        return _clean(named.group(1)), Span(
            start=named.start(1), end=named.end(1), snippet=named.group(0)
        )
    return None


def parse_telescope(text: str) -> str | None:
    result = parse_telescope_with_span(text)
    return result[0] if result is not None else None
