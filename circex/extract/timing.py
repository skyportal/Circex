"""Observation-epoch resolution for photometry rows (ICARE P0 #2).

Two sources of a row's epoch:

1. An absolute UT / MJD stated in the row (a table's Date/MJD column) — parsed
   here with no trigger time needed.
2. A relative offset ("T+234s") plus a trigger time T0 — resolved here.

Both produce an `(obs_mjd, obs_time)` pair: MJD (UTC) as a float and an ISO-8601
UTC string. T0 is treated as immutable per circular (true for a published GRB
trigger), so resolving relative offsets at extraction time and caching the
result is safe.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from astropy.time import Time
from dateutil import parser as date_parser

from circex.schema import CircularExtraction, PhotometryExt

# Plausible MJD range so a bare number is read as MJD, not a year or a count:
# 40000 = 1968-05-24, 90000 = 2031-09-04. GCN circulars fall well inside this.
_MJD_LO = 40000.0
_MJD_HI = 90000.0

_UNIT_SECONDS: dict[str, float] = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def _to_pair(dt_or_mjd: datetime | float) -> tuple[float, str]:
    """Normalize a datetime or MJD float to (obs_mjd, obs_time ISO-8601 UTC)."""
    if isinstance(dt_or_mjd, datetime):
        dt = dt_or_mjd if dt_or_mjd.tzinfo else dt_or_mjd.replace(tzinfo=UTC)
        t = Time(dt.astimezone(UTC))
        return float(t.mjd), dt.astimezone(UTC).isoformat().replace("+00:00", "Z")
    t = Time(dt_or_mjd, format="mjd")
    iso = str(t.utc.isot)
    return float(dt_or_mjd), iso if iso.endswith("Z") else iso + "Z"


def epoch_from_absolute(token: str | None) -> tuple[float, str] | None:
    """Parse an absolute date/MJD token into (obs_mjd, obs_time), or None.

    Accepts bare MJD numbers (within the plausible range), ISO-8601, and the
    loose 'YYYY-MM-DD HH:MM' forms common in circular tables, including the
    'YYYY-MM-DD_HH:MM' and 'YYYY-MM-DD at HH:MM' separators radio circulars use.
    """
    if not token:
        return None
    token = re.sub(r"(?:_|\s+at\s+)(?=\d{2}:\d{2})", " ", token.strip())
    if not token:
        return None
    # Bare MJD number.
    try:
        val = float(token)
    except ValueError:
        pass
    else:
        if _MJD_LO < val < _MJD_HI:
            return _to_pair(val)
        return None  # a number outside MJD range isn't a date we trust
    # Calendar date/time.
    token, day_fraction = _split_fractional_day(token)
    dt = _parse_complete_date(token)
    if dt is None:
        return None
    if day_fraction:
        return _to_pair(float(Time(dt.replace(tzinfo=UTC)).mjd) + day_fraction)
    return _to_pair(dt)


# Two defaults a year apart: whatever dateutil fills in from the default rather
# than from the token comes out different, and a date it had to invent is worse
# than no date at all.
_DEFAULT_A = datetime(1904, 1, 1, tzinfo=UTC)
_DEFAULT_B = datetime(1905, 2, 2, tzinfo=UTC)


def _parse_complete_date(token: str) -> datetime | None:
    """Parse a token only if it states its own year, month and day."""
    try:
        a = date_parser.parse(token, default=_DEFAULT_A)
        b = date_parser.parse(token, default=_DEFAULT_B)
    except (ValueError, OverflowError):
        return None
    return a if a == b else None


def _split_fractional_day(token: str) -> tuple[str, float]:
    """Separate a decimal day from its date: "Aug 18.85" is the 18th at 20:24 UT.

    A token carrying a clock time already states it, so it is left alone.
    """
    if ":" in token:
        return token, 0.0
    match = re.search(r"\b(\d{1,2})\.(\d+)\b", token)
    if match is None:
        return token, 0.0
    cleaned = token[: match.start()] + match.group(1) + token[match.end() :]
    return cleaned, float("0." + match.group(2))


def normalize_pair(obs_mjd: float | None, obs_time: str | None) -> tuple[float, str] | None:
    """Given whichever of (obs_mjd, obs_time) is set, return both. None if neither.

    obs_mjd wins when both are present (numeric, unambiguous). Used to backfill
    the missing half of the pair on PhotometryExt (e.g. an LLM that set only the
    ISO time).
    """
    if obs_mjd is not None:
        return _to_pair(obs_mjd)
    return epoch_from_absolute(obs_time)


def epoch_from_offset(trigger_time: datetime, value: float, unit: str) -> tuple[float, str] | None:
    """Resolve T0 + offset into (obs_mjd, obs_time). None if the unit is unknown."""
    seconds = _UNIT_SECONDS.get(unit)
    if seconds is None:
        return None
    base = trigger_time if trigger_time.tzinfo else trigger_time.replace(tzinfo=UTC)
    mjd = float(Time(base.astimezone(UTC)).mjd) + (value * seconds) / 86400.0
    return _to_pair(mjd)


# Month names as circulars write them: "August", "Aug", "Sept".
_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
# "2023-03-12", "2017 August 18", "2017-Aug-19", "20 August 2017",
# "August 18 2017", "August 18th 2017".
_ORDINAL = r"(?:st|nd|rd|th)?"
# Astronomical dates carry the time as a decimal day ("2017 Aug 18.85 UT"), so
# the day-of-month may be fractional wherever it appears.
_DAY = r"\d{1,2}(?:\.\d+)?"
_DATE = (
    rf"(?:\d{{4}}[-.]\d{{2}}[-.]\d{{2}}(?:\.\d+)?"
    rf"|\d{{4}}[-.\s]+{_MONTH}[-.\s]+{_DAY}{_ORDINAL}"
    rf"|{_DAY}{_ORDINAL}[-.\s]+{_MONTH}[-.\s]+\d{{4}}"
    rf"|{_MONTH}\s+{_DAY}{_ORDINAL}[,\s]+\d{{4}})"
)
# Radio circulars separate date from time with "_" or " at " as often as a space.
_TIME = r"(?:(?:[ T_]|\s+at\s+)\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
# Anything up to the end of the sentence, so "et al." and "2024 Dec 14" do not
# cut the search short the way a bare period would. A period after a single
# capital is an initial ("PI: G. Anderson"), not a sentence end.
_WITHIN_SENTENCE = r"(?:(?!(?<![A-Z])\.\s+[A-Z])[\s\S]){0,200}?"

# An absolute observation datetime stated in prose, near observation language:
# "We observed from 2026-06-05 03:41 to 03:51 UTC", "beginning at 2017 August 18
# 02:09:00 UT". Captures the first datetime that follows an observation verb.
_VERB = (
    r"(?:observ|imag|obtain|acquir|expos|integrat|trigger|detect|begin|began"
    r"|start|monitor|carried\s+out|on\s+target)\w*"
)
# "Observations started at 23:15 UT on August 18th 2017" puts the time before
# the date. Without this the date still parses and the time silently becomes
# midnight, which is wrong by up to a day.
_TIME_FIRST_RE = re.compile(
    rf"\bat\s+(?P<clock>\d{{1,2}}:\d{{2}}(?::\d{{2}}(?:\.\d+)?)?)\s*(?:UTC?\b)?\s*"
    rf"on\s+(?P<date>{_DATE})",
    re.IGNORECASE,
)

_OBS_EPOCH_RE = re.compile(
    rf"{_VERB}{_WITHIN_SENTENCE}({_DATE}{_TIME})",
    re.IGNORECASE,
)
# The same sentence with the date first: "On 2017 Aug 18 UT in the process of
# observing several galaxies". The verb still has to be there, so a bare date in
# a citation is not mistaken for an observation epoch.
_OBS_EPOCH_REVERSED_RE = re.compile(
    rf"\bon\s+({_DATE}{_TIME})\s*(?:UTC?\b)?{_WITHIN_SENTENCE}{_VERB}",
    re.IGNORECASE,
)


def parse_observation_epoch(text: str) -> tuple[float, str] | None:
    """First absolute observation datetime stated in prose, as (obs_mjd, obs_time).

    Looks for a calendar datetime immediately following observation language
    ("observed ... 2026-06-05 03:41"). Used to time prose photometry lists whose
    epoch lives in a separate sentence rather than a per-row column. None if none.
    """
    clock = _TIME_FIRST_RE.search(text)
    if clock is not None:
        return epoch_from_absolute(f"{clock.group('date')} {clock.group('clock')}")
    match = _OBS_EPOCH_RE.search(text) or _OBS_EPOCH_REVERSED_RE.search(text)
    if match is None:
        return None
    return epoch_from_absolute(match.group(1))


def _stated_mid_time(body: str, trigger_time: datetime | None) -> tuple[float, str] | None:
    """T0 plus a mid-exposure offset the circular states, when both are known."""
    if trigger_time is None:
        return None
    from circex.extract.regex.dates import parse_mid_time_offset

    offset = parse_mid_time_offset(body)
    if offset is None:
        return None
    return epoch_from_offset(trigger_time, offset.value, offset.unit)


def resolve_object_epochs(extraction: CircularExtraction, body: str) -> set[int]:
    """Time each candidate's rows from the paragraph that discusses it.

    A circular tabulating several candidates gives the table one epoch column at
    most, then describes each object in its own paragraph with the time it was
    seen. Applying the circular's single epoch to every row would put one
    object's measurement at another's time, so each object is timed from the
    paragraph that names it -- and only when that paragraph names no other.

    Returns the identities of the rows timed this way, so the circular-level
    backfill can tell them from a table that carried its own dates.
    """
    rows_by_object: dict[str, list[PhotometryExt]] = {}
    for row in extraction.photometry:
        if row.obs_mjd is not None or row.obs_time is not None or not row.object_name:
            continue
        rows_by_object.setdefault(row.object_name, []).append(row)
    if not rows_by_object:
        return set()

    # Every designation an object answers to, so a paragraph using the survey's
    # internal name is still recognised as being about the IAU-named object.
    aliases: dict[str, set[str]] = {}
    for row in extraction.photometry:
        if row.object_name:
            aliases.setdefault(row.object_name, {row.object_name}).update(row.object_aliases)

    timed: set[int] = set()
    for paragraph in re.split(r"\n\s*\n", body):
        named = [
            name for name, names in aliases.items() if any(alias in paragraph for alias in names)
        ]
        # Two objects in one paragraph leaves the time ambiguous; skip it rather
        # than attach one object's epoch to the other.
        if len(named) != 1:
            continue
        rows = rows_by_object.get(named[0])
        if not rows:
            continue
        pair = parse_observation_epoch(paragraph)
        if pair is None:
            continue
        mjd, iso = pair
        for row in rows:
            row.obs_mjd, row.obs_time = mjd, iso
            timed.add(id(row))
        rows_by_object.pop(named[0], None)
    return timed


def resolve_observation_epoch(
    extraction: CircularExtraction,
    body: str,
    trigger_time: datetime | None = None,
    timed_per_object: set[int] | None = None,
) -> None:
    """Backfill a single circular-level observation epoch onto untimed rows, in place.

    For prose photometry lists ("g = 19.69 +/- 0.04", magnitudes on their own
    lines) the observation time is stated once, in a separate sentence, not per
    row. When EVERY photometry row lacks an epoch and the body states one
    observation datetime, apply it to all rows. Guarded to all-untimed so a
    partially-dated table is never clobbered; notes the inference for the record.
    """
    if not extraction.photometry:
        return
    # Rows already timed from their own object's paragraph are not evidence that
    # this table carried dates, so they neither block the backfill nor take it.
    per_object = timed_per_object or set()
    pending = [r for r in extraction.photometry if id(r) not in per_object]
    if not pending:
        return
    if any(r.obs_mjd is not None or r.obs_time is not None for r in pending):
        return
    # A datetime in the prose is usually when the exposures began; where the
    # circular also states the mid-time, that is the epoch of the measurement.
    pair = _stated_mid_time(body, trigger_time) or parse_observation_epoch(body)
    if pair is None:
        return
    mjd, iso = pair
    for row in pending:
        row.obs_mjd = mjd
        row.obs_time = iso
    extraction.extraction_meta.notes.append(
        f"observation epoch {iso} (single circular-level time) applied to all "
        f"{len(extraction.photometry)} photometry row(s)"
    )


def resolve_inline_offsets(
    extraction: CircularExtraction, body: str, trigger_time: datetime | None
) -> None:
    """Pair each row with an offset stated on its own line, in place.

    A circular that lists several epochs usually writes each one beside its
    magnitude ("r = 20.92 +/- 0.19 AB (mid-time 38.66 min after the trigger)").
    The offsets are only ambiguous when they sit apart from the measurements, so
    a row is timed here when exactly one offset shares its line.
    """
    from circex.extract.regex.dates import parse_time_offsets

    if trigger_time is None or not extraction.photometry:
        return

    untimed = [r for r in extraction.photometry if r.obs_mjd is None and r.obs_time is None]
    if not untimed:
        return

    for row in untimed:
        value = row.mag if row.mag is not None else row.limiting_mag
        if value is None:
            continue
        line = _line_containing(body, value)
        if line is None:
            continue
        offsets = parse_time_offsets(line)
        if len(offsets) != 1:
            continue
        pair = epoch_from_offset(trigger_time, offsets[0].value, offsets[0].unit)
        if pair is None:
            continue
        row.obs_mjd, row.obs_time = pair


def _line_containing(body: str, value: float) -> str | None:
    """The one line stating this magnitude, or None if it is not unique."""
    token = f"{value:g}"
    hits = [line for line in body.splitlines() if token in line]
    return hits[0] if len(hits) == 1 else None


def resolve_relative_epochs(extraction: CircularExtraction, trigger_time: datetime | None) -> None:
    """Fill obs_mjd/obs_time on rows that lack an epoch, in place.

    Conservative single-epoch rule: only applied when the circular has exactly
    one distinct relative offset (one time_offset, or several that resolve to
    the same MJD) and a trigger time is known. That single epoch is applied to
    every photometry row still missing a time. Multiple distinct offsets are
    ambiguous to pair with rows, so those are left null rather than guessed.
    """
    if trigger_time is None or not extraction.photometry:
        return
    resolved: set[tuple[float, str]] = set()
    for off in extraction.time_offsets:
        pair = epoch_from_offset(trigger_time, off.value, off.unit)
        if pair is not None:
            resolved.add(pair)
    if len(resolved) != 1:
        return
    (mjd, iso) = next(iter(resolved))
    for row in extraction.photometry:
        if row.obs_mjd is None and row.obs_time is None:
            row.obs_mjd = mjd
            row.obs_time = iso
