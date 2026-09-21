"""Redshift parser: 'z = X.XXX' with optional error + method/type heuristics.

The plan's headline regex is `z\\s*[=~]\\s*(\\d+\\.\\d+)` plus ±200-char context
windows for spec/photo and host/em/abs. We implement both.
"""

from __future__ import annotations

import re

from circex.schema import Redshift, RedshiftMeasure, RedshiftType, Span

# The negative lookbehind rejects a color index ("g-z = 0.48", "i-z = 0.9"): the
# trailing "z" is a filter band, not a redshift. "photo-z" (preceded by "o-") is
# not a filter letter, so it still matches (and is filtered by context instead).
_Z_RE = re.compile(
    r"(?<![ugrizyUBVRIJHK]-)\bz\s*[=~≈]\s*(\d+\.\d+)(?:\s*±\s*(\d+\.\d+))?",
    re.IGNORECASE,
)
_ALT_RE = re.compile(
    r"\b(?:redshift\s+(?:of\s+)?)(\d+\.\d+)",
    re.IGNORECASE,
)
# Bound-redshift phrasings the current Redshift schema cannot represent.
# Examples: "z <= 1.61", "z =< 1.61", "z >= 0.2", "z < 2.5".
# Per docs/labeling_spec.md, when one of these fires the extractor sets
# redshift = None and appends the literal phrase to extraction_meta.notes.
_Z_BOUND_RE = re.compile(
    r"\bz\s*(<=?|>=?|=<|=>|≤|≥)\s*(\d+\.\d+)",
    re.IGNORECASE,
)

# Context-window heuristics. Search ±200 chars around the redshift match.
_CONTEXT_WINDOW = 200

# A real redshift is well below this; a "z = 19.21" match is the Sloan z-band
# MAGNITUDE, not a redshift (the highest known spectroscopic redshifts are ~11).
_MAX_PLAUSIBLE_Z = 12.0

# A redshift can belong to a *nearby, explicitly-unassociated* object (typically a
# catalog galaxy offset from the transient) rather than to the transient itself —
# e.g. 44834's "red galaxy ... at 18.9\" from the optical counterpart ... photo-z =
# 0.343 ... association very unlikely". Grabbing that z would post a wrong redshift.
# If the context around the match carries an offset/association-doubt cue, skip it.
_NEARBY_OBJECT_RE = re.compile(
    r"association\s+(?:is\s+)?(?:very\s+)?unlikely"
    r"|unlikely\s+to\s+be\s+associated"
    r"|\bP_?cc\b"  # chance-coincidence probability, cited to argue non-association
    r"|\d+(?:\.\d+)?\s*(?:\"|''|arcsec|arcseconds?)\s+(?:from|away|offset)"
    r"|offset\s+of\s+\d",
    re.IGNORECASE,
)

_SPEC_RE = re.compile(r"\bspectroscop(?:ic|y|ically)\b", re.IGNORECASE)
_PHOTO_RE = re.compile(r"\bphotomet(?:ric|ry|rically)\b", re.IGNORECASE)
_HOST_RE = re.compile(r"\bhost(?:\s+galaxy)?\b", re.IGNORECASE)
# "absorption line" is one phrasing among several -- a trough, a feature, a
# system. Bare "absorption" is not enough: Galactic absorption is extinction.
_EMISSION_RE = re.compile(r"\bemission[- ](?:lines?|features?|components?)", re.IGNORECASE)
_ABSORPTION_RE = re.compile(
    r"\babsorption[- ](?:lines?|features?|troughs?|systems?|components?)",
    re.IGNORECASE,
)


def _classify_measure(context: str, at: int | None = None) -> RedshiftMeasure | None:
    """Spectroscopic or photometric, from whichever cue sits nearest the value.

    Both words appear together whenever a circular compares its own measurement
    with someone else's: "the photometric redshift of 0.995 +- 0.352, which is
    marginally consistent with the redshift of 1.673 obtained with GTC
    spectroscopy". Taking the first rule labelled that photometric value
    spectroscopic, so the cue is chosen by distance instead.
    """
    spec = _SPEC_RE.search(context)
    photo = _PHOTO_RE.search(context)
    if spec is None and photo is None:
        return None
    if spec is None:
        return "photometric"
    if photo is None:
        return "spectroscopic"
    if at is None:
        return "spectroscopic"

    def distance(m: re.Match[str]) -> int:
        return min(abs(m.start() - at), abs(m.end() - at))

    return "spectroscopic" if distance(spec) <= distance(photo) else "photometric"


def _classify_type(context: str) -> RedshiftType | None:
    if _HOST_RE.search(context):
        return "host"
    if _EMISSION_RE.search(context):
        return "emission"
    if _ABSORPTION_RE.search(context):
        return "absorption"
    return None


def parse_redshift(text: str) -> Redshift | None:
    """Return a Redshift if the text contains z = X.XXX or 'redshift of X.XXX'."""
    result = parse_redshift_with_span(text)
    return result[0] if result is not None else None


def parse_redshift_bound(text: str) -> tuple[str, Span] | None:
    """Return (phrase, Span) for the first bound-redshift mention in `text`, or None.

    Bound redshifts (e.g. "z <= 1.61") cannot be represented in the current
    Redshift point-value schema. The composer in RegexExtractor writes the
    matched phrase to extraction_meta.notes and records a "_redshift_bound"
    provenance key; it does NOT populate Redshift.redshift.
    """
    match = _Z_BOUND_RE.search(text)
    if not match:
        return None
    phrase = match.group(0)
    span = Span(start=match.start(), end=match.end(), snippet=phrase)
    return phrase, span


def context_around(text: str, start: int, end: int) -> str:
    """The +/-200 characters a redshift is read in."""
    return text[max(0, start - _CONTEXT_WINDOW) : min(len(text), end + _CONTEXT_WINDOW)]


def disclaimed_at(text: str, start: int, end: int) -> str | None:
    """The cue, if this position sits in a nearby-or-conditional-object context."""
    match = _NEARBY_OBJECT_RE.search(context_around(text, start, end))
    return match.group(0) if match else None


def disclaimed_value(text: str, value: float) -> str | None:
    """The cue, if every mention of `value` in `text` is one the circular hedges.

    A circular can name a candidate host's catalogue redshift and the
    transient's own; only a value that appears nowhere undisclaimed is dropped.
    """
    mentions = list(re.finditer(re.escape(f"{value:g}"), text))
    if not mentions:
        return None
    cues = [disclaimed_at(text, m.start(), m.end()) for m in mentions]
    return cues[0] if all(cues) else None


def parse_redshift_with_span(text: str) -> tuple[Redshift, Span] | None:
    """Same as parse_redshift, but also return a Span pointing at the z-match.

    Every match is considered, not just the first: photometry writes `z = 21.3`
    for the Sloan band, and stopping there would miss the redshift below it.
    """
    # Each pattern is exhausted before the next is tried, as it always was:
    # merging them by position promotes a superseded value over the correction
    # that follows it ("the redshift of 1.7 ... was due to mis-identification
    # ... the correct redshift is z=2.42").
    matches = [*_Z_RE.finditer(text), *_ALT_RE.finditer(text)]
    for match in matches:
        z = float(match.group(1))
        if z >= _MAX_PLAUSIBLE_Z:
            continue  # a "z = 19.21" match is a z-band magnitude, not a redshift
        err: float | None = None
        if _Z_RE.match(match.group(0)) is not None:
            try:
                err_str = match.group(2)
            except IndexError:
                err_str = None
            if err_str:
                err = float(err_str)

        ctx_start = max(0, match.start() - _CONTEXT_WINDOW)
        context = context_around(text, match.start(), match.end())

        # Skip a redshift that the circular attributes to a nearby object, or
        # associates only conditionally, rather than to the transient.
        if _NEARBY_OBJECT_RE.search(context):
            continue

        redshift = Redshift(
            redshift=z,
            redshift_error=err,
            redshift_measure=_classify_measure(context, match.start() - ctx_start),
            redshift_type=_classify_type(context),
        )
        span = Span(start=match.start(), end=match.end(), snippet=match.group(0))
        return redshift, span
    return None
