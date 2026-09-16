"""Date / time parsers: UTC, MJD, and literal T+offset captures.

Per PDF decision 4, T+offset captures are stored as TimeOffset records literally —
no resolution against T0 in v1.
"""

from __future__ import annotations

import re

from circex.schema import Span, TimeOffset, TimeOffsetUnit

# T+234s, T+8.5h, T+12 d, T-300 s, T + 4 hours
_T_OFFSET_RE = re.compile(
    r"""
    \bT
    \s*(?P<sign>[+\-−–])\s*
    (?P<value>\d+(?:\.\d+)?)
    \s*
    (?P<unit>ks|s(?:ec(?:ond)?s?)?|m(?:in(?:ute)?s?)?|h(?:our)?s?|d(?:ay)?s?)
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# "T-T0 = 11h", "observations mid-time at T-To=11h", "T_mid-T_0 12.2 h". The
# elapsed time written as a difference rather than as an offset from T; the
# subscript is a zero, a letter o, or absent.
_T_MINUS_T0_RE = re.compile(
    r"""
    \bT(?:_?mid)?
    \s*[-−–]\s*
    T[_\s]?(?:0|o)\b
    \s*(?:[=:~]|\s)\s*
    (?P<value>\d+(?:\.\d+)?)
    \s*
    (?P<unit>ks|hrs?|s(?:ec(?:ond)?s?)?|m(?:in(?:ute)?s?)?|h(?:our)?s?|d(?:ay)?s?)
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# "mid. time = 8.1358 hours", "observations mid-time at 2.3 hr" — the elapsed
# time to the middle of a stacked exposure, with the trigger left implicit. A
# clock time ("mid time = 03:41 UT") carries no bare unit and does not match.
_MID_TIME_RE = re.compile(
    r"""
    \bmid[-.\s]*time\b
    \s*(?:[=:~]|\s+(?:of|at)\s+)?\s*
    (?P<value>\d+(?:\.\d+)?)
    \s*
    (?P<unit>ks|hrs?|s(?:ec(?:ond)?s?)?|m(?:in(?:ute)?s?)?|h(?:our)?s?|d(?:ay)?s?)
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# "approximately X hours after the trigger", "X minutes post-burst", "5.9d after
# the Swift/BAT trigger". Circulars say "burst" as often as "trigger", and name
# the instrument in between, so both are accepted. "the GRB" and "the merger"
# name the same instant. "the notice" and "the alert" deliberately do not: those
# are issued after the event they describe, by seconds to minutes.
_POST_TRIGGER_RE = re.compile(
    r"""
    \b(?P<value>\d+(?:\.\d+)?)\s*
    (?P<unit>ks|seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|[smhd])\s+
    (?:
        after\s+(?:the\s+)?(?:\S+\s+){0,2}?(?:trigger|burst|explosion|GRB|merger)
      | post[-\s]?(?:trigger|burst|explosion|GRB|merger)
    )
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)

_UNIT_MAP: dict[str, TimeOffsetUnit] = {
    "ks": "s",
    "s": "s",
    "sec": "s",
    "secs": "s",
    "second": "s",
    "seconds": "s",
    "m": "m",
    "min": "m",
    "mins": "m",
    "minute": "m",
    "minutes": "m",
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    "d": "d",
    "day": "d",
    "days": "d",
}


# Swift and UVOT circulars quote offsets in kiloseconds ("19.2 ks after the BAT
# trigger"). The schema's units stop at seconds, so a ks value is scaled instead.
_UNIT_SCALE: dict[str, float] = {"ks": 1000.0}


def _normalize_unit(token: str) -> TimeOffsetUnit | None:
    return _UNIT_MAP.get(token.lower())


def _scaled(value: float, token: str) -> float:
    return value * _UNIT_SCALE.get(token.lower(), 1.0)


def parse_time_offsets(text: str) -> list[TimeOffset]:
    """Find all literal T+offset and 'X hours after trigger' phrasings."""
    return [t for t, _ in parse_time_offsets_with_spans(text)]


def parse_time_offsets_with_spans(text: str) -> list[tuple[TimeOffset, Span]]:
    """Same as parse_time_offsets, plus a Span per offset pointing at the phrasing."""
    out: list[tuple[TimeOffset, Span]] = []

    for match in _T_OFFSET_RE.finditer(text):
        unit = _normalize_unit(match.group("unit"))
        if unit is None:
            continue
        sign = match.group("sign")
        value = _scaled(float(match.group("value")), match.group("unit"))
        if sign in ("-", "−", "–"):
            value = -value
            reference = "T-"
        else:
            reference = "T+"
        out.append(
            (
                TimeOffset(value=value, unit=unit, reference=reference),
                Span(start=match.start(), end=match.end(), snippet=match.group(0)),
            )
        )

    for match in _T_MINUS_T0_RE.finditer(text):
        unit = _normalize_unit(match.group("unit"))
        if unit is None:
            continue
        out.append(
            (
                TimeOffset(
                    value=_scaled(float(match.group("value")), match.group("unit")),
                    unit=unit,
                    reference="trigger",
                ),
                Span(start=match.start(), end=match.end(), snippet=match.group(0)),
            )
        )

    for match in _POST_TRIGGER_RE.finditer(text):
        unit = _normalize_unit(match.group("unit"))
        if unit is None:
            continue
        out.append(
            (
                TimeOffset(
                    value=_scaled(float(match.group("value")), match.group("unit")),
                    unit=unit,
                    reference="trigger",
                ),
                Span(start=match.start(), end=match.end(), snippet=match.group(0)),
            )
        )

    # Last, and only where nothing else claimed the text: "mid-time 38.66 min
    # after the trigger" is already an offset by the clause above.
    for match in _MID_TIME_RE.finditer(text):
        unit = _normalize_unit(match.group("unit"))
        if unit is None:
            continue
        if any(sp.start < match.end() and match.start() < sp.end for _, sp in out):
            continue
        out.append(
            (
                TimeOffset(
                    value=_scaled(float(match.group("value")), match.group("unit")),
                    unit=unit,
                    reference="trigger",
                ),
                Span(start=match.start(), end=match.end(), snippet=match.group(0)),
            )
        )

    return out


def parse_mid_time_offset(text: str) -> TimeOffset | None:
    """The mid-exposure offset from the trigger, when the circular states one.

    Only one: a circular listing several mid-times is describing several epochs,
    and which row each belongs to is decided per line, not here.
    """
    matches = _MID_TIME_RE.findall(text)
    if len(matches) != 1:
        return None
    match = _MID_TIME_RE.search(text)
    assert match is not None
    unit = _normalize_unit(match.group("unit"))
    if unit is None:
        return None
    return TimeOffset(
        value=_scaled(float(match.group("value")), match.group("unit")),
        unit=unit,
        reference="trigger",
    )
