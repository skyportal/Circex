"""Radio/mm photometry: flux densities rather than magnitudes.

Radio circulars report a flux density at an observing frequency ("a flux density
of 120+/-30 microJy/beam at 9 GHz"), so neither `filter` nor `mag` applies. The
band is identified by frequency and crosswalked to a radio-* bandpass.

The parser is anchored on the "flux density" / "upper limit" phrases and only
emits a row when the value and frequency counts agree, so a distributive
"...170 +/- 30 and 150 +/- 20 microJy/beam, respectively" pairs correctly and an
ambiguous clause yields nothing rather than a guess.
"""

from __future__ import annotations

import math
import re
from typing import Final

from circex.extract.regex.mag_table import _contaminant_ranges, _in_ranges
from circex.schema import FluxDensityUnit, PhotometryExt, Span

# Canonical bandpass name per centre frequency, from SkyPortal's
# `additional_bandpasses`. Below 230 GHz these are the VLA bands plus ATCA's
# 3mm pair; above, the submillimetre bands.
_BANDPASS_BY_GHZ: Final[tuple[tuple[float, str], ...]] = (
    (0.34, "radio-0.34GHz"),
    (1.4, "radio-1.4GHz"),
    (3.0, "radio-3GHz"),
    (6.0, "radio-6GHz"),
    (10.0, "radio-10GHz"),
    (15.0, "radio-15GHz"),
    (22.0, "radio-22GHz"),
    (33.0, "radio-33GHz"),
    (45.0, "radio-45GHz"),
    (93.0, "radio-93GHz"),
    (95.0, "radio-95GHz"),
    (230.0, "sma-230GHz"),
    (345.0, "sma-345GHz"),
    (400.0, "sma-400GHz"),
)

# Widest ratio between a reported frequency and the nearest bandpass centre that
# still counts as the same band. Receiver bands are broad (VLA C is 4-8 GHz for
# a 6 GHz centre), so the tolerance has to be generous; beyond it there is no
# representable bandpass and the row is dropped rather than mislabelled.
_MAX_FREQ_RATIO: Final[float] = 1.5

_UJY_PER: Final[dict[str, float]] = {"uJy": 1.0, "mJy": 1.0e3, "Jy": 1.0e6}

_UNIT_ALIASES: Final[dict[str, FluxDensityUnit]] = {
    "microjy": "uJy",
    "micro-jy": "uJy",
    "ujy": "uJy",
    "μjy": "uJy",
    "µjy": "uJy",
    "microjansky": "uJy",
    "mjy": "mJy",
    "millijy": "mJy",
    "millijansky": "mJy",
    "jy": "Jy",
    "jansky": "Jy",
}

_UNIT_RE = re.compile(
    r"\b(microJansky|micro[\s-]?Jy|milliJansky|milliJy|uJy|μJy|µJy|mJy|Jansky|Jy)"
    r"(?:\s*/\s*beam)?",
    re.IGNORECASE,
)

# A run of numbers sharing one trailing unit: "9 GHz", "5.5 and 9 GHz".
_FREQ_RE = re.compile(
    r"((?:\d+(?:\.\d+)?\s*(?:,\s*|\s+and\s+|\s*&\s*|\s*/\s*))*\d+(?:\.\d+)?)\s*(GHz|MHz)",
    re.IGNORECASE,
)

# GCN bodies carry mis-encoded minus signs, so accept U+FFFD after "+/".
_VAL_ERR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\+/[-−–�]|±)\s*(\d+(?:\.\d+)?)")
_BARE_VAL_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])")

_DETECTION_ANCHOR = r"flux\s+densit(?:y|ies)"
# A bare "limit" is too loose (FRB fluence limits, sensitivity limits), so require
# either "upper" or an explicit N-sigma prefix.
_LIMIT_ANCHOR = r"(?:\d+(?:\.\d+)?\s*[-\s]?sigma\s+(?:upper\s+)?limits?|upper[\s-]?limits?)"
_ANCHOR_RE = re.compile(rf"({_LIMIT_ANCHOR}|{_DETECTION_ANCHOR})", re.IGNORECASE)

_SIGMA_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-\s]?sigma", re.IGNORECASE)
_COMPARATIVE_RE = re.compile(
    r"(?:than|compared\s+(?:with|to)|versus|vs\.?)\s+(?:the\s+)?$", re.IGNORECASE
)
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

# "a 10 GHz flux density of ~5.5 mJy", "the 1390 MHz band flux density of...":
# the frequency qualifies the anchor from in front. Only "band" may intervene,
# so a frequency merely earlier in the clause is not mistaken for this one's.
_ADJACENT_FREQ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(GHz|MHz)(?:\s+bands?)?[\s-]*$", re.IGNORECASE)
# "17/21 GHz flux density < 42 uJy" states one value over a pair of frequencies.
# Which one it belongs to is not recoverable, so the shortcut declines it.
_FREQ_LIST_TAIL_RE = re.compile(r"[\d.]\s*(?:/|,|&|\band\b)\s*$", re.IGNORECASE)

# One value with a unit of its own, anchoring a single-measurement line.
_LINE_FREQ_FIRST_RE = re.compile(
    r"^\s*(?P<freq>\d+(?:\.\d+)?)\s*(?P<fu>GHz|MHz)\s*[:=]?\s*"
    r"(?P<op><|>)?\s*~?\s*(?P<val>\d+(?:\.\d+)?)"
    r"(?:\s*(?:\+/[-−–\ufffd]|±)\s*(?P<err>\d+(?:\.\d+)?))?\s*"
    r"(?P<unit>microJy|micro[\s-]?Jy|uJy|μJy|µJy|mJy|Jy)(?:\s*/\s*beam)?\s*[.,;]?\s*$",
    re.IGNORECASE,
)
_LINE_VAL_FIRST_RE = re.compile(
    r"^\s*(?P<op><|>)?\s*~?\s*(?P<val>\d+(?:\.\d+)?)"
    r"(?:\s*(?:\+/[-−–\ufffd]|±)\s*(?P<err>\d+(?:\.\d+)?))?\s*"
    r"(?P<unit>microJy|micro[\s-]?Jy|uJy|μJy|µJy|mJy|Jy)(?:\s*/\s*beam)?\s+at\s+"
    r"(?P<freq>\d+(?:\.\d+)?)\s*(?P<fu>GHz|MHz)\s*[.,;]?\s*$",
    re.IGNORECASE,
)


def normalize_flux_unit(token: str) -> FluxDensityUnit | None:
    """Map a written unit ('microJy/beam', 'mJy') to the canonical enum."""
    cleaned = re.sub(r"\s*/\s*beam", "", token.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"[\s-]+", "", cleaned)
    return _UNIT_ALIASES.get(cleaned.lower())


def to_ujy(value: float, unit: FluxDensityUnit) -> float:
    """Flux density in microjanskys."""
    return value * _UJY_PER[unit]


def bandpass_for_frequency(ghz: float) -> str | None:
    """Nearest bandpass in log-frequency, or None if nothing is close enough."""
    if ghz <= 0:
        return None
    centre, name = min(_BANDPASS_BY_GHZ, key=lambda b: abs(math.log(ghz / b[0])))
    return name if max(ghz / centre, centre / ghz) <= _MAX_FREQ_RATIO else None


def _frequencies(text: str) -> list[float]:
    """Every frequency in `text`, in GHz, in order of appearance.

    A frequency introduced by a comparison belongs to some other instrument's
    result being cited, not to the measurement in hand, so it is skipped.
    """
    out: list[float] = []
    for match in _FREQ_RE.finditer(text):
        if _COMPARATIVE_RE.search(text[: match.start()]):
            continue
        scale = 1.0e-3 if match.group(2).lower() == "mhz" else 1.0
        for token in re.split(r",|\band\b|&|/", match.group(1)):
            token = token.strip()
            if token:
                out.append(float(token) * scale)
    return out


def _values(segment: str) -> list[tuple[float, float | None]]:
    """(value, error) pairs in `segment`; bare values when no +/- form is present."""
    pairs: list[tuple[float, float | None]] = [
        (float(m.group(1)), float(m.group(2))) for m in _VAL_ERR_RE.finditer(segment)
    ]
    if pairs:
        return pairs
    return [(float(m.group(1)), None) for m in _BARE_VAL_RE.finditer(segment)]


def _row(
    ghz: float,
    unit: FluxDensityUnit,
    value: float,
    error: float | None,
    is_limit: bool,
    sigma: float | None,
) -> PhotometryExt:
    row = PhotometryExt(
        frequency_ghz=ghz,
        flux_density_unit=unit,
        bandpass=bandpass_for_frequency(ghz),
        mag_system="AB",
    )
    if is_limit:
        row.limiting_flux_density = value
        row.limiting_mag_sigma = sigma if sigma is not None else 3.0
    else:
        row.flux_density = value
        row.flux_density_error = error
    row.is_detection = not is_limit
    return row


def parse_radio_with_spans(text: str) -> list[tuple[PhotometryExt, Span]]:
    """Radio flux-density rows with Spans into the source text.

    Two strict shapes are accepted: prose where the whole clause shares a single
    unit token, and a standalone line carrying exactly one measurement. Anything
    with several units in one clause (column tables, per-epoch lists) is left to
    the LLM path rather than paired by guesswork.
    """
    excluded = _contaminant_ranges(text)
    rows = _parse_prose(text, excluded)
    claimed = [(sp.start, sp.end) for _, sp in rows]
    rows.extend(_parse_lines(text, excluded, claimed))
    return sorted(rows, key=lambda pair: pair[1].start)


def _parse_prose(text: str, excluded: list[tuple[int, int]]) -> list[tuple[PhotometryExt, Span]]:
    out: list[tuple[PhotometryExt, Span]] = []
    offset = 0
    for clause in _CLAUSE_SPLIT_RE.split(text):
        start = text.index(clause, offset)
        offset = start + len(clause)
        out.extend(_parse_clause(clause, start, excluded))
    return out


def _parse_clause(
    clause: str, base: int, excluded: list[tuple[int, int]]
) -> list[tuple[PhotometryExt, Span]]:
    anchors = list(_ANCHOR_RE.finditer(clause))
    if not anchors:
        return []
    clause_freqs = _frequencies(clause)

    out: list[tuple[PhotometryExt, Span]] = []
    for i, anchor in enumerate(anchors):
        if _in_ranges(base + anchor.start(), excluded):
            continue
        end = anchors[i + 1].start() if i + 1 < len(anchors) else len(clause)
        segment = clause[anchor.start() : end]

        # Several units in one segment means per-value units (a column table or a
        # per-epoch list); the value/frequency pairing is not recoverable here.
        units = list(_UNIT_RE.finditer(segment))
        if len(units) != 1:
            continue
        unit = normalize_flux_unit(units[0].group(0))
        if unit is None:
            continue

        anchor_end = anchor.end() - anchor.start()
        values = _values(segment[anchor_end : units[0].start()])
        if not values:
            continue
        # A frequency qualifying the anchor from in front binds to this one
        # measurement, and stops the segment's reach into the next one's.
        span_end = end
        leading = _ADJACENT_FREQ_RE.search(clause[: anchor.start()])
        if leading is not None and _FREQ_LIST_TAIL_RE.search(clause[: leading.start()]):
            leading = None
        if leading is not None and len(values) == 1:
            scale = 1.0e-3 if leading.group(2).lower() == "mhz" else 1.0
            freqs = [float(leading.group(1)) * scale]
            span_end = anchor.start() + units[0].end()
        else:
            # Frequencies stated after the anchor, else the clause's own —
            # accepted only when the counts line up, which is what makes
            # "respectively" safe.
            freqs = _frequencies(segment)
            if len(freqs) != len(values):
                freqs = clause_freqs
            if len(freqs) != len(values):
                continue

        is_limit = "limit" in anchor.group(0).lower()
        sigma_match = _SIGMA_RE.search(segment) or _SIGMA_RE.search(clause)
        sigma = float(sigma_match.group(1)) if sigma_match else None

        for (value, error), ghz in zip(values, freqs, strict=True):
            out.append(
                (
                    _row(ghz, unit, value, error, is_limit, sigma),
                    Span(
                        start=base + anchor.start(),
                        end=base + span_end,
                        snippet=clause[anchor.start() : span_end],
                    ),
                )
            )
    return out


def _parse_lines(
    text: str,
    excluded: list[tuple[int, int]],
    claimed: list[tuple[int, int]],
) -> list[tuple[PhotometryExt, Span]]:
    """Standalone one-measurement lines, as written in flux-density listings."""
    out: list[tuple[PhotometryExt, Span]] = []
    pos = 0
    for line in re.split(r"(?<=\n)", text):
        start, pos = pos, pos + len(line)
        stripped = line.strip()
        if not stripped or _in_ranges(start, excluded):
            continue
        if any(s <= start < e for s, e in claimed):
            continue
        match = _LINE_FREQ_FIRST_RE.match(stripped) or _LINE_VAL_FIRST_RE.match(stripped)
        if match is None:
            continue
        unit = normalize_flux_unit(match.group("unit"))
        if unit is None:
            continue
        scale = 1.0e-3 if match.group("fu").lower() == "mhz" else 1.0
        ghz = float(match.group("freq")) * scale
        err = match.group("err")
        is_limit = match.group("op") == "<"
        sigma_match = _SIGMA_RE.search(text[max(0, start - 400) : start])
        out.append(
            (
                _row(
                    ghz,
                    unit,
                    float(match.group("val")),
                    float(err) if err else None,
                    is_limit,
                    float(sigma_match.group(1)) if sigma_match else None,
                ),
                Span(start=start, end=start + len(line.rstrip()), snippet=line.rstrip()),
            )
        )
    return out
