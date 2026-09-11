"""Magnitude parsers: single-row prose mentions + multi-row tables.

Per PDF Phase 2, multi-row tables are where regex visibly fails — the parser is
intentionally conservative. It should return ROWS on cleanly-formatted column-aligned
tables, and return NOTHING on irregular prose (rather than fabricating).

Mag-system inference rules (matching docs/labeling_spec.md):
  Sloan g/r/i/z/y → AB
  Bessel U/B/V/R/I → Vega
  2MASS J/H/K/Ks → Vega
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Final

from circex.extract.timing import epoch_from_absolute, epoch_from_offset
from circex.schema import MagSystem, PhotometryExt, Span

# Filter classification.
# Filter tokens circulars use, beyond the Johnson/Sloan/NIR letters: Swift's UVOT
# set, HST's F###W names, and the several spellings of "no filter".
_UVOT = ("uvw1", "uvw2", "uvm2", "white")
_SVOM_VT = ("VT_B", "VT_R")
_UNFILTERED = ("unfiltered", "clear", "CR", "CV")

_SLOAN: Final[frozenset[str]] = frozenset({"u", "g", "r", "i", "z", "y"})
_BESSEL: Final[frozenset[str]] = frozenset({"U", "B", "V", "R", "I"})
# Y is the NIR band near 1.02 um, distinct from the Sloan/PS1 y near 0.96 um
# that the lowercase token means. Case is the only thing telling them apart.
_NIR: Final[frozenset[str]] = frozenset({"Y", "J", "H", "K", "Ks"})

# Wide survey filters with no sncosmo equivalent. Recognised so their rows are
# read and attributed, with bandpass left unset rather than guessed.
_WIDE: Final[frozenset[str]] = frozenset({"L"})

_HST: Final[frozenset[str]] = frozenset(
    {"F450W", "F555W", "F606W", "F702W", "F775W", "F814W", "F850LP", "F160W", "F110W"}
)
_KNOWN_FILTERS: Final[frozenset[str]] = (
    _SLOAN
    | _BESSEL
    | _NIR
    | _WIDE
    | _HST
    | frozenset(_UVOT)
    | frozenset(_SVOM_VT)
    | frozenset(_UNFILTERED)
    | frozenset({"clear", "C", "W", "Rc", "Ic"})
)

# Circulars write the Sloan prime with an ASCII quote or, when the text has been
# through a word processor, a typographic one.
_PRIMES = "'\u2019\u02b9\u2032"

# Single-mag patterns. Matches "r = 18.42 ± 0.05", "R=22.1+/-0.3", "K_s = 19.0",
_FILTER_TOKEN = (
    r"(?:F\d{3}[A-Z]{1,2}"
    r"|" + "|".join(_SVOM_VT) + r""
    r"|" + "|".join(_UVOT) + r""
    r"|" + "|".join(_UNFILTERED) + r""
    r"|[UBVRI]c|[ugriz][p" + _PRIMES + r"]|[UBVRIJHKYLgrizyuCW]s?)"
)

# "5-sigma upper limit: J = 19.07" states a limit in the syntax of a detection.
# Scoped to the clause before the value so a limit mentioned in an earlier
# sentence does not turn a real detection into a non-detection.
_LIMIT_CLAUSE_RE = re.compile(
    r"(?:(?P<sigma>\d+(?:\.\d+)?)\s*[-\s]?\s*sigma\s+)?"
    r"(?:upper[-\s]?limits?|limiting\s+magnitudes?|non[-\s]?detections?"
    # bare "limit" is too loose, but "down to a limit of" is not
    r"|down\s+to\s+(?:a|the)\s+limit)"
    r"[^.;:]*[:\s]\s*$",
    re.IGNORECASE,
)

# "down to a 3-sigma upper limit of about 20.6 mag": the depth is stated in
# prose and the band named in a neighbouring sentence, so no filter sits beside
# the number. `mag` is mandatory, because the same phrasing carries X-ray limits
# in ph/cm2/s and counts/s, which are not magnitudes.
_BARE_LIMIT_RE = re.compile(
    r"(?:(?P<sigma>\d+(?:\.\d+)?)\s*[-–\s]?\s*sigma\s+)?"
    r"(?:upper[-\s]?limits?|limiting\s+magnitudes?)"
    r"\s*(?:of|is|was|are|were|[:=~])\s*"
    r"(?:a\s+|about\s+|approximately\s+|around\s+|~\s*)*"
    r"(?P<limit>\d{1,2}\.\d{1,3})\s*mag\b"
    r"(?:[^.;]{0,30}?\(?\s*(?P<sigma_after>\d+(?:\.\d+)?)\s*[-–\s]?\s*sigma)?",
    re.IGNORECASE,
)

# "an r'-band magnitude of 22.1 +/- 0.1", "the i-band magnitude is about 21.8":
# the band names the measurement from in front rather than sitting beside the
# value. The phrase welds the two together, so there is no pairing to guess.
_BAND_PROSE_RE = re.compile(
    rf"""
    (?<![A-Za-z])
    (?P<filter>{_FILTER_TOKEN})
    \s*[-\s]?\s*band\s+
    (?:AB\s+|Vega\s+)?
    (?:magnitudes?|mag|brightness)\s+
    (?:of|is|was|=|:)\s*
    (?:about\s+|approximately\s+|around\s+|~\s*)*
    (?P<mag>\d{{1,2}}\.\d{{1,3}})
    (?:\s*(?:±|\+/[-−]|\+[-−])\s*(?P<err>\d+\.\d+))?
    """,
    re.VERBOSE | re.IGNORECASE,
)

# "with a limiting mag of ~20.5": the anchor names the unit itself, so unlike
# _BARE_LIMIT_RE there is no trailing "mag" to demand -- an X-ray limit in
# counts/s is never introduced as a limiting magnitude.
_LIMITING_MAG_RE = re.compile(
    r"(?:(?P<sigma>\d+(?:\.\d+)?)\s*[-–\s]?\s*sigma\s+)?"
    r"limiting\s+mag(?:nitude)?s?\b"
    r"\s*(?:of|is|was|are|were|[:=~])\s*"
    r"(?:a\s+|about\s+|approximately\s+|around\s+|~\s*)*"
    r"(?P<limit>\d{1,2}\.\d{1,3})"
    r"(?:[^.;]{0,30}?\(?\s*(?P<sigma_after>\d+(?:\.\d+)?)\s*[-–\s]?\s*sigma)?",
    re.IGNORECASE,
)

# "in R-band", "the r' filter": the band a bare limit refers to.
_BAND_CONTEXT_RE = re.compile(
    rf"(?<![A-Za-z])(?P<filter>{_FILTER_TOKEN})\s*[-\s]?\s*(?:band|filter)\b",
    re.IGNORECASE,
)


# upper limits: "r > 22.5", "m > 22 (3-sigma)".
_DETECTION_RE = re.compile(
    rf"""
    (?<![A-Za-z])                          # not preceded by a letter (avoid "Ar=...")
    (?P<filter>{_FILTER_TOKEN})            # filter
    \s*[=~]\s*
    (?P<mag>\d{{1,2}}\.\d{{1,3}})              # magnitude
    (?:\s*(?:±|\+/[-−]|\+[-−]|[+\-])\s*(?P<err>\d+\.\d+))?  # error: ±, +/-, +-
    """,
    re.VERBOSE,
)

_UPPER_LIMIT_RE = re.compile(
    rf"""
    (?<![A-Za-z])
    (?P<filter>{_FILTER_TOKEN})
    \s*>\s*
    (?P<limit>\d{{1,2}}\.\d{{1,3}})
    (?:.{{0,40}}?(?P<sigma>\d+(?:\.\d+)?)\s*[-–]?\s*sigma)?
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Space-separated detection, as written in fixed-width single-row tables and
# terse prose: "Rc     23.08 +/- 0.18", "r 19.5 ± 0.05". The mandatory +/- (or
# ±) error term after a whitespace-separated mag is a strong precision guard —
# it distinguishes a real magnitude measurement from incidental "<letter>
# <number>" pairs. Cousins (Rc, Ic) and primed-Sloan (r', rp) filters are
# accepted here and normalized to their base band. This intentionally does NOT
# parse multi-row tables (the documented regex/LLM boundary); it recovers the
# common single-detection line the column-split parser drops.
_SPACED_DETECTION_RE = re.compile(
    rf"""
    (?<![A-Za-z])
    (?P<filter>{_FILTER_TOKEN})
    \s+
    (?P<mag>\d{{1,2}}\.\d{{1,3}})
    \s*(?:±|\+/-|\+/−)\s*
    (?P<err>\d+\.\d+)
    """,
    re.VERBOSE,
)


def normalize_filter(token: str) -> str:
    """Strip a Cousins 'c' suffix (Rc->R) or a prime marker (r'/rp->r)."""
    if len(token) == 2 and token[0] in "UBVRI" and token[1] == "c":
        return token[0]
    if len(token) == 2 and token[0] in "ugriz" and token[1] in "p" + _PRIMES:
        return token[0]
    return token


# Circulars that state their system say so plainly, and it overrides the filter
# convention: NIR is usually Vega, but "J = 19.07 mag (AB)" is not.
_STATED_AB_RE = re.compile(
    r"\(\s*AB\s*\)|\bAB\s+(?:mag(?:nitude)?s?|system|photometric)"
    r"|magnitudes?\s+(?:are\s+)?(?:given\s+|reported\s+|calibrated\s+)?in\s+"
    r"(?:the\s+)?AB\b",
    re.IGNORECASE,
)
_STATED_VEGA_RE = re.compile(
    r"\(\s*Vega\s*\)|\bVega\s+(?:mag(?:nitude)?s?|system|photometric)"
    r"|magnitudes?\s+(?:are\s+)?(?:given\s+|reported\s+|calibrated\s+)?in\s+"
    r"(?:the\s+)?Vega\b",
    re.IGNORECASE,
)


def _clause_before(text: str, at: int, window: int = 90) -> str:
    """Text back to the previous sentence break, so a clause cannot reach past it."""
    start = max(0, at - window)
    segment = text[start:at]
    for stop in (". ", "\n\n"):
        cut = segment.rfind(stop)
        if cut != -1:
            segment = segment[cut + len(stop) :]
    return segment


def stated_mag_system(text: str) -> MagSystem | None:
    """The system the text names, or None when it names neither or both.

    A circular aggregating several telescopes can quote both, and there is no
    way to tell here which row is which, so it defers to the filter convention.
    """
    ab = _STATED_AB_RE.search(text) is not None
    vega = _STATED_VEGA_RE.search(text) is not None
    if ab and not vega:
        return "AB"
    if vega and not ab:
        return "Vega"
    return None


def infer_mag_system(filter_name: str) -> MagSystem | None:
    """Heuristic AB/Vega inference based on filter name conventions."""
    if filter_name in _SLOAN:
        return "AB"
    if filter_name in _BESSEL or filter_name in _NIR:
        return "Vega"
    return None


# Canonical-bandpass crosswalk (sncosmo / SkyPortal vocabulary). The raw filter
# token is always kept on PhotometryExt.filter; this populates the sibling
# `bandpass` field so downstream consumers (e.g. SkyPortal/ICARE) get a
# normalized name. `clear`/`C` (unfiltered) have no canonical bandpass.
_BANDPASS_CROSSWALK: Final[dict[str, str]] = {
    # Sloan (AB)
    "u": "sdssu",
    "g": "sdssg",
    "r": "sdssr",
    "i": "sdssi",
    "z": "sdssz",
    "y": "ps1::y",
    # Bessel (Vega)
    "U": "bessellu",
    "B": "bessellb",
    "V": "bessellv",
    "R": "bessellr",
    "I": "besselli",
    # Cousins R/I, as the circulars write them.
    "Rc": "bessellr",
    "Ic": "besselli",
    # Swift/UVOT, which names its filters in lower case.
    "uvw1": "uvot::uvw1",
    "uvw2": "uvot::uvw2",
    "uvm2": "uvot::uvm2",
    "white": "uvot::white",
    # SVOM's Visible Telescope, whose blue and red channels SkyPortal defines.
    "VT_B": "svomvtb",
    "VT_R": "svomvtr",
    # HST, where sncosmo registers the bare name. F814W is deliberately absent:
    # it exists only as uvf814w (WFC3/UVIS), and ACS carries an F814W too, so the
    # bare name does not say which instrument took the measurement.
    "F555W": "f555w",
    "F606W": "f606w",
    "F775W": "f775w",
    "F850LP": "f850lp",
    "F110W": "f110w",
    "F125W": "f125w",
    "F160W": "f160w",
    # 2MASS / NIR (Vega)
    # sncosmo carries no NIR Y, so this approximates it with the PS1 y that
    # sits ~60 nm blueward. Revisit once a Y band lands upstream.
    "Y": "ps1::y",
    "J": "2massj",
    "H": "2massh",
    "K": "2massks",
    "Ks": "2massks",
}


def infer_bandpass(filter_name: str) -> str | None:
    """Map a recognized filter token to its canonical bandpass name, or None."""
    return _BANDPASS_CROSSWALK.get(filter_name)


# Optical magnitudes run roughly 5 to 30; the deepest reported limits sit just
# under 30, so a larger number is something else wearing a filter name.
_MAG_MIN, _MAG_MAX = 5.0, 30.0


def _plausible_mag(filter_name: str, mag: float) -> bool:
    """Reject obvious non-photometric values (e.g., 'z = 1.61' for redshift).

    The lowercase Sloan filter 'z' overlaps with redshift notation, so z-band
    mags must additionally be > 10.
    """
    if not _MAG_MIN < mag < _MAG_MAX:
        return False
    return not (filter_name == "z" and mag < 10.0)


# Clauses describing a NON-transient object — a nearby/host galaxy, a reference or
# comparison star — can state magnitudes that are not the transient's (e.g. 44834's
# "a red galaxy with g = 21.69 ..."). Photometry inside such a clause is excluded.
_CONTAMINANT_RE = re.compile(
    r"red\s+galaxy|host\s+galaxy|nearby\s+galaxy|underlying\s+galaxy"
    r"|background\s+galaxy|foreground\s+galaxy"
    r"|reference\s+star|comparison\s+star|calibration\s+star|field\s+star|nearby\s+star",
    re.IGNORECASE,
)


# "GRB240618.80 (trigger No 740430582, 22h 48m 28.80s, +71d 41m 24.0s, R=74.88)
# errorbox" reports the radius of an error box. R is a length here, and 58 of
# these fall inside the plausible magnitude range, so no bound separates them
# from real photometry; the surrounding template is what identifies them.
_ERRORBOX_RE = re.compile(r"\(\s*trigger\s+No\b[^)]*\)\s*errorbox", re.IGNORECASE)


def _errorbox_ranges(text: str) -> list[tuple[int, int]]:
    """Char ranges of error-box declarations, whose R is a radius, not a band."""
    return [(m.start(), m.end()) for m in _ERRORBOX_RE.finditer(text)]


def _contaminant_ranges(text: str) -> list[tuple[int, int]]:
    """Char ranges of clauses about non-transient objects (galaxy / reference star).

    Each runs from the contaminant phrase to the end of its sentence, so
    magnitudes attributed to that object are dropped from the transient's photometry.
    """
    ranges: list[tuple[int, int]] = []
    for m in _CONTAMINANT_RE.finditer(text):
        end_match = re.search(r"\.(?:\s|$)", text[m.start() :])
        end = m.start() + end_match.end() if end_match else len(text)
        ranges.append((m.start(), end))
    return ranges


def _in_ranges(pos: int, ranges: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in ranges)


# "our clear coadd images": an unfiltered token is itself the band, so it needs
# no following "band"/"filter" the way a lettered one does.
_UNFILTERED_CONTEXT_RE = re.compile(
    r"(?<![A-Za-z])(?P<filter>" + "|".join(_UNFILTERED) + r")\s+"
    r"(?:co-?add|stack|combined|image|exposure|frame)\w*",
    re.IGNORECASE,
)


def _context_filter(text: str, pos: int, window: int = 400) -> str | None:
    """The band named most recently before `pos`, for a limit that names none."""
    start = max(0, pos - window)
    matches = list(_BAND_CONTEXT_RE.finditer(text, start, pos))
    matches += list(_UNFILTERED_CONTEXT_RE.finditer(text, start, pos))
    for match in sorted(matches, key=lambda m: m.start(), reverse=True):
        raw = match.group("filter")
        if raw.lower() in _UNFILTERED:
            return raw.lower()
        name = raw if raw in _KNOWN_FILTERS else normalize_filter(raw)
        if name in _KNOWN_FILTERS:
            return name
    return None


def parse_single_mags(text: str) -> list[PhotometryExt]:
    """Extract single-row magnitude mentions ('r = 18.42 ± 0.05', 'R > 22.5')."""
    return [p for p, _ in parse_single_mags_with_spans(text)]


def parse_single_mags_with_spans(text: str) -> list[tuple[PhotometryExt, Span]]:
    """Same as parse_single_mags, plus per-row Spans into the source text."""
    rows: list[tuple[PhotometryExt, Span]] = []
    consumed: list[tuple[int, int]] = []  # char ranges already claimed
    # non-transient (galaxy / ref-star) clauses, and error-box radii
    excluded = _contaminant_ranges(text) + _errorbox_ranges(text)

    for match in _DETECTION_RE.finditer(text):
        raw = match.group("filter")
        # Rc stays Rc; r' and rp fold to r. Checking the raw token first keeps a
        # Cousins band distinguishable from its Johnson counterpart.
        filter_name = raw if raw in _KNOWN_FILTERS else normalize_filter(raw)
        if filter_name not in _KNOWN_FILTERS:
            continue
        if _in_ranges(match.start(), excluded):
            continue
        mag = float(match.group("mag"))
        if not _plausible_mag(filter_name, mag):
            continue
        err = float(match.group("err")) if match.group("err") else None
        limit_clause = _LIMIT_CLAUSE_RE.search(_clause_before(text, match.start()))
        consumed.append((match.start(), match.end()))
        rows.append(
            (
                PhotometryExt(
                    filter=filter_name,
                    mag=None if limit_clause else mag,
                    mag_error=None if limit_clause else err,
                    limiting_mag=mag if limit_clause else None,
                    limiting_mag_sigma=(
                        float(limit_clause.group("sigma"))
                        if limit_clause and limit_clause.group("sigma")
                        else None
                    ),
                    mag_system=infer_mag_system(filter_name),
                    bandpass=infer_bandpass(filter_name),
                ),
                Span(start=match.start(), end=match.end(), snippet=match.group(0)),
            )
        )

    # Space-separated form ("Rc  23.08 +/- 0.18"). Skip ranges already matched
    # by the '=' form above so a value isn't double-counted.
    for match in _SPACED_DETECTION_RE.finditer(text):
        if any(s < match.end() and match.start() < e for s, e in consumed):
            continue
        if _in_ranges(match.start(), excluded):
            continue
        base = normalize_filter(match.group("filter"))
        if base not in _KNOWN_FILTERS:
            continue
        mag = float(match.group("mag"))
        if not _plausible_mag(base, mag):
            continue
        rows.append(
            (
                PhotometryExt(
                    filter=base,
                    mag=mag,
                    mag_error=float(match.group("err")),
                    mag_system=infer_mag_system(base),
                    bandpass=infer_bandpass(base),
                ),
                Span(start=match.start(), end=match.end(), snippet=match.group(0)),
            )
        )

    # Band-before-value prose ("an r'-band magnitude of 22.1 +/- 0.1").
    for match in _BAND_PROSE_RE.finditer(text):
        if any(s < match.end() and match.start() < e for s, e in consumed):
            continue
        if _in_ranges(match.start(), excluded):
            continue
        raw = match.group("filter")
        filter_name = raw if raw in _KNOWN_FILTERS else normalize_filter(raw)
        if filter_name not in _KNOWN_FILTERS:
            continue
        mag = float(match.group("mag"))
        if not _plausible_mag(filter_name, mag):
            continue
        err = float(match.group("err")) if match.group("err") else None
        limit_clause = _LIMIT_CLAUSE_RE.search(_clause_before(text, match.start()))
        consumed.append((match.start(), match.end()))
        rows.append(
            (
                PhotometryExt(
                    filter=filter_name,
                    mag=None if limit_clause else mag,
                    mag_error=None if limit_clause else err,
                    limiting_mag=mag if limit_clause else None,
                    limiting_mag_sigma=(
                        float(limit_clause.group("sigma"))
                        if limit_clause and limit_clause.group("sigma")
                        else None
                    ),
                    is_detection=not limit_clause,
                    mag_system=infer_mag_system(filter_name),
                    bandpass=infer_bandpass(filter_name),
                ),
                Span(start=match.start(), end=match.end(), snippet=match.group(0)),
            )
        )

    for match in _UPPER_LIMIT_RE.finditer(text):
        raw = match.group("filter")
        # Rc stays Rc; r' and rp fold to r. Checking the raw token first keeps a
        # Cousins band distinguishable from its Johnson counterpart.
        filter_name = raw if raw in _KNOWN_FILTERS else normalize_filter(raw)
        if filter_name not in _KNOWN_FILTERS:
            continue
        if _in_ranges(match.start(), excluded):
            continue
        limit = float(match.group("limit"))
        sigma = float(match.group("sigma")) if match.group("sigma") else None
        rows.append(
            (
                PhotometryExt(
                    filter=filter_name,
                    limiting_mag=limit,
                    limiting_mag_sigma=sigma,
                    mag_system=infer_mag_system(filter_name),
                    bandpass=infer_bandpass(filter_name),
                ),
                Span(start=match.start(), end=match.end(), snippet=match.group(0)),
            )
        )

    # Limits stated without a filter beside them, the band taken from context.
    claimed = [(span.start, span.end) for _, span in rows]
    for match in _LIMITING_MAG_RE.finditer(text):
        if _in_ranges(match.start(), excluded) or _in_ranges(match.start(), claimed):
            continue
        filter_name = _context_filter(text, match.start())
        if filter_name is None:
            continue
        limit = float(match.group("limit"))
        if not _plausible_mag(filter_name, limit):
            continue
        stated_sigma = match.group("sigma") or match.group("sigma_after")
        claimed.append((match.start(), match.end()))
        rows.append(
            (
                PhotometryExt(
                    filter=filter_name,
                    limiting_mag=limit,
                    limiting_mag_sigma=float(stated_sigma) if stated_sigma else None,
                    is_detection=False,
                    mag_system=infer_mag_system(filter_name),
                    bandpass=infer_bandpass(filter_name),
                ),
                Span(start=match.start(), end=match.end(), snippet=match.group(0)),
            )
        )

    for match in _BARE_LIMIT_RE.finditer(text):
        if _in_ranges(match.start(), excluded) or _in_ranges(match.start(), claimed):
            continue
        filter_name = _context_filter(text, match.start())
        if filter_name is None:
            continue
        limit = float(match.group("limit"))
        if not _plausible_mag(filter_name, limit):
            continue
        stated_sigma = match.group("sigma") or match.group("sigma_after")
        rows.append(
            (
                PhotometryExt(
                    filter=filter_name,
                    limiting_mag=limit,
                    limiting_mag_sigma=float(stated_sigma) if stated_sigma else None,
                    is_detection=False,
                    mag_system=infer_mag_system(filter_name),
                    bandpass=infer_bandpass(filter_name),
                ),
                Span(start=match.start(), end=match.end(), snippet=match.group(0)),
            )
        )

    return rows


# ---- Multi-row table detector ----

_COLUMN_SPLIT_RE = re.compile(r"\s{2,}|\t")
_HEADER_KEYWORDS = {
    "date",
    "mjd",
    "epoch",
    "filter",
    "band",
    "mag",
    "magnitude",
    "err",
    "error",
    "exp",
    "exptime",
    "exposure",
}


def _looks_like_header(line: str) -> bool:
    """Cheap heuristic: line is a header if it contains 2+ table keywords."""
    tokens = {_header_word(token) for token in _COLUMN_SPLIT_RE.split(line)}
    return len(tokens & _HEADER_KEYWORDS) >= 2


def _looks_like_table_row(line: str, expected_columns: int) -> bool:
    """A data row has roughly the expected number of fields and at least one numeric."""
    fields = [f for f in _COLUMN_SPLIT_RE.split(line.strip()) if f]
    if not (expected_columns - 1 <= len(fields) <= expected_columns + 1):
        return False
    return any(re.search(r"\d", f) for f in fields)


# A rule drawn under a header: "-----", "=====", "_____".
_RULE_RE = re.compile(r"^\s*[-=_]{3,}\s*$")


_ERR_WORDS = frozenset({"err", "err.", "error", "magerr", "merr", "uncertainty"})


def _strip_word(word: str) -> str:
    return word.strip(":.,()[]").lower()


def _header_word(field: str) -> str:
    """The word a header cell is named by: "Mag. (AB)" -> "mag", "MJD (mid)" -> "mjd".

    A cell often carries a unit or qualifier after the name, and matching the
    whole cell leaves the column unclassified and its values dropped.
    """
    first = field.strip().split()[0] if field.strip() else ""
    return _strip_word(first)


def _classify_columns(header_line: str) -> dict[int, str]:
    """Return a dict mapping column index -> semantic role (filter, mag, err, ...).

    A cell is named by its first word, stripped of the punctuation headers carry
    ("Err\\n", "mag.", "Filter:", "Mag. (AB)"); matching the whole cell leaves the
    column unclassified and its values silently dropped.
    """
    fields = [f for f in _COLUMN_SPLIT_RE.split(header_line) if f]
    classification: dict[int, str] = {}
    for i, field in enumerate(fields):
        # An uncertainty keyword anywhere in the cell settles it: "Mag err" names
        # the error column, not a second magnitude one.
        if _ERR_WORDS & {_strip_word(w) for w in field.split()}:
            classification[i] = "mag_error"
            continue
        token = _header_word(field)
        if token in classification.values():
            continue  # "Mag" then "Mag. Range": the first column carrying a role owns it
        if token in {"filter", "band"}:
            classification[i] = "filter"
        elif token in {"mag", "magnitude"}:
            classification[i] = "mag"
        elif token in {"date", "mjd", "epoch"}:
            classification[i] = "date"
        elif token in {"exp", "exptime", "exposure"}:
            classification[i] = "exposure"
    return classification


# ---- Markdown / pipe-delimited tables ----
#
# "| Tmid-T0 (h) | Filter | Mag (AB) |" — common in modern circulars (GRANDMA,
# GOTO, LAST, ICARE style). Columns split on "|"; the "| --- | --- |" separator
# row is skipped. Time columns are an absolute UT datetime OR a relative offset
# (Tmid-T0 / t-t0 / TGRB, in hours) resolved against a supplied trigger time.
# Mag cells carry "X +/- Y" and optionally "(... limit: Z)".

_PIPE_MAG_RE = re.compile(r"(?P<mag>\d{1,2}\.\d{1,4})\s*(?:±|\+/-|\+/−)\s*(?P<err>\d+\.\d+)")
_PIPE_LIMIT_RE = re.compile(r"limit[:\s]*(?P<lim>\d{1,2}\.\d{1,4})", re.IGNORECASE)


def _pipe_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r"[-:\s]*", c) for c in cells)


def _classify_pipe_columns(cells: list[str]) -> dict[int, str]:
    roles: dict[int, str] = {}
    for i, tok in enumerate(cells):
        t = tok.lower()
        if "filter" in t or t == "band":
            roles[i] = "filter"
        elif re.search(r"mag\s*err|magerr|\berr", t):
            # Checked BEFORE "mag": a "MagErr" column would otherwise classify as
            # a second mag column and overwrite the real one (ZTF/GROWTH template).
            roles[i] = "mag_err"
        elif "mag" in t:  # mag / magnitude / abmag / mag (ab)
            roles[i] = "mag"
        elif re.search(r"\bra\b", t):
            roles[i] = "ra"
        elif re.search(r"\bdec\b", t):
            roles[i] = "dec"
        elif "name" in t:  # "ZTF Name" / "IAU Name" — counterpart designation
            roles[i] = "name"
        elif re.search(r"t0|tgrb|t-t|tmid", t) and re.search(r"h\b|hour|hr", t):
            roles[i] = "rel_hours"
        elif re.search(r"mid-?time|date|\but\b|utc", t):
            roles[i] = "abs_time"
    return roles


# "+----+" / "+===+" frame lines of ASCII-boxed tables (no "|" in them), which
# would otherwise terminate the data-row scan before the first row is reached.
_PIPE_FRAME_RE = re.compile(r"^\s*\+[-=+\s]*\+?\s*$")


def _decimal_degrees(cell: str | None) -> float | None:
    """A coordinate cell as decimal degrees, or None when it is not one."""
    if not cell:
        return None
    m = re.fullmatch(r"([+-]?\d{1,3}(?:\.\d+)?)", cell.strip())
    return float(m.group(1)) if m else None


def _parse_pipe_row(
    cells: list[str],
    roles: dict[int, str],
    trigger_time: datetime | None,
    prose_filter: str | None = None,
) -> PhotometryExt | None:
    by = {role: cells[idx] for idx, role in roles.items() if idx < len(cells)}
    m = _PIPE_MAG_RE.search(by.get("mag", ""))
    mag = float(m.group("mag")) if m else None
    err = float(m.group("err")) if m else None
    if mag is None:
        # Bare magnitude cell ("19.78") — the error lives in its own MagErr column.
        bare = re.fullmatch(r"(\d{1,2}\.\d{1,4})", by.get("mag", "").strip())
        if bare:
            mag = float(bare.group(1))
    if err is None and by.get("mag_err"):
        err_m = re.search(r"\d+(?:\.\d+)?", by["mag_err"])
        if err_m:
            err = float(err_m.group())
    lim_m = _PIPE_LIMIT_RE.search(by.get("mag", ""))
    limit = float(lim_m.group("lim")) if lim_m else None
    if limit is None:
        # "> 24.3": a table states a non-detection as an inequality rather than
        # spelling out the word the cell pattern above looks for.
        bound = re.fullmatch(r"[><]\s*(\d{1,2}\.\d{1,4})", by.get("mag", "").strip())
        if bound:
            limit = float(bound.group(1))
    if mag is None and limit is None:
        return None
    # A table listing several candidates gives each row its own object; without
    # carrying that, two candidates' magnitudes would land on one light curve.
    # A row may carry both an internal and an IAU designation, in separate
    # columns, and either may be blank: take the first that is filled, in column
    # order, so the identifier is the one the table always supplies.
    object_name = next(
        (
            cells[idx].strip()
            for idx in sorted(roles)
            if roles[idx] == "name" and idx < len(cells) and cells[idx].strip()
        ),
        None,
    )
    row_ra = _decimal_degrees(by.get("ra"))
    row_dec = _decimal_degrees(by.get("dec"))
    # Filter: "Rc (Vega)" -> "Rc" -> R. Require a recognized, mappable filter.
    # A candidate table names no band per row -- it states one in prose for the
    # whole circular -- so that is accepted in place of a column.
    raw = by.get("filter")
    base = normalize_filter(re.split(r"[ (]", raw.strip())[0]) if raw else None
    if base is None and prose_filter is not None:
        base = prose_filter
    if base is None or base not in _KNOWN_FILTERS:
        return None
    # A wide table repeating filter/mag/error per band can have its mag column
    # misread, so hold a pipe row to the same plausibility the prose parsers use.
    for value in (mag, limit):
        if value is not None and not _plausible_mag(base, value):
            return None
    obs_mjd = obs_time = None
    if by.get("abs_time") and (ep := epoch_from_absolute(by["abs_time"])) is not None:
        obs_mjd, obs_time = ep
    if obs_mjd is None and by.get("rel_hours") and trigger_time is not None:
        num = re.search(r"[-+]?\d+(?:\.\d+)?", by["rel_hours"])
        if num and (ep := epoch_from_offset(trigger_time, float(num.group()), "h")) is not None:
            obs_mjd, obs_time = ep
    return PhotometryExt(
        filter=base,
        mag=mag,
        mag_error=err,
        limiting_mag=limit,
        mag_system=infer_mag_system(base),
        bandpass=infer_bandpass(base),
        obs_mjd=obs_mjd,
        obs_time=obs_time,
        object_name=object_name,
        ra=row_ra,
        dec=row_dec,
    )


def parse_pipe_table_with_spans(
    text: str, trigger_time: datetime | None = None
) -> list[tuple[PhotometryExt, Span]]:
    """Parse pipe-delimited magnitude tables. Per-row Spans; relative times use T0."""
    rows: list[tuple[PhotometryExt, Span]] = []
    # A candidate table names no band per row; the circular states one in prose.
    prose_filter = _context_filter(text, len(text), window=len(text))
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    i = 0
    while i < len(lines):
        if "|" not in lines[i]:
            i += 1
            continue
        roles = _classify_pipe_columns(_pipe_cells(lines[i]))
        # Header needs a mag column plus at least one more recognized column
        # (filter / time) — guards against a stray prose line with a "|".
        if "mag" not in roles.values() or len(roles) < 2:
            i += 1
            continue
        j = i + 1
        while j < len(lines):
            if not lines[j].strip():
                # Some templates space their rows out. A blank run only ends the
                # table when nothing table-shaped follows it.
                k = j
                while k < len(lines) and not lines[k].strip():
                    k += 1
                if k < len(lines) and ("|" in lines[k] or _PIPE_FRAME_RE.match(lines[k])):
                    j = k
                    continue
                break
            if "|" not in lines[j] and not _PIPE_FRAME_RE.match(lines[j]):
                break
            if _PIPE_FRAME_RE.match(lines[j]):
                j += 1  # ASCII box frame, not a data row — but not the end either
                continue
            cells = _pipe_cells(lines[j])
            if not _is_separator_row(cells):
                row = _parse_pipe_row(cells, roles, trigger_time, prose_filter)
                if row is not None:
                    row_text = lines[j].rstrip("\r\n")
                    rows.append(
                        (
                            row,
                            Span(
                                start=offsets[j], end=offsets[j] + len(row_text), snippet=row_text
                            ),
                        )
                    )
            j += 1
        i = max(j, i + 1)
    return rows


def parse_pipe_candidate_with_span(text: str) -> tuple[list[str], float, float, Span] | None:
    """Counterpart designation + decimal-degree position from a candidate table.

    The ZTF/GROWTH follow-up template announces a counterpart candidate as a
    pipe table with Name / RA (deg) / DEC (deg) / Filter / Mag columns (e.g.
    GCN 45198). Returns (names, ra, dec, span-of-row) ONLY when the table has
    exactly one data row: a single named candidate is a claimed counterpart,
    while a multi-row list is a survey product that must not become one source
    (same convention as the labeling spec's wide-field-search rule).
    """
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    i = 0
    while i < len(lines):
        if "|" not in lines[i]:
            i += 1
            continue
        roles = _classify_pipe_columns(_pipe_cells(lines[i]))
        vals = set(roles.values())
        if not ({"ra", "dec"} <= vals):
            i += 1
            continue
        hits: list[tuple[list[str], float, float, Span]] = []
        j = i + 1
        while j < len(lines) and ("|" in lines[j] or _PIPE_FRAME_RE.match(lines[j])):
            if _PIPE_FRAME_RE.match(lines[j]):
                j += 1
                continue
            cells = _pipe_cells(lines[j])
            if not _is_separator_row(cells):
                by: dict[str, list[str]] = {}
                for idx, role in roles.items():
                    if idx < len(cells):
                        by.setdefault(role, []).append(cells[idx])
                try:
                    ra = float(by["ra"][0])
                    dec = float(by["dec"][0])
                except (KeyError, IndexError, ValueError):
                    j += 1
                    continue
                if 0.0 <= ra <= 360.0 and -90.0 <= dec <= 90.0:
                    names = [n for n in by.get("name", []) if n]
                    row_text = lines[j].rstrip("\r\n")
                    span = Span(start=offsets[j], end=offsets[j] + len(row_text), snippet=row_text)
                    hits.append((names, ra, dec, span))
            j += 1
        if len(hits) == 1:
            return hits[0]
        i = max(j, i + 1)
    return None


# ---- Fixed-width "Date UTstart t-T0 Exp. Filter Mag Err. UL" tables ----
#
# The SAO-RAS / IKI GRB follow-up template (GCN 44852, 44858, 44862, 44877):
#
#   Date       UTstart  t-T0    Exp.   Filter Mag +/- Err.    UL
#   2026.06.08 20:01:14 4.01102 12*300 Rc     23.08 +/- 0.18  23.8
#
# The Date and UTstart columns are ONE datetime token in the data but TWO header
# columns, which defeats the whitespace-column parser. Parse the row directly,
# anchored on a leading datetime with mandatory seconds (so looser generic tables
# like "2024-01-02 04:30 r ..." are left to parse_mag_table). Mag/Err come as
# "23.08 +/- 0.18" or space-separated "21.35  0.14"; the trailing float is the UL.

_FIXEDW_ROW_RE = re.compile(
    r"""^[ \t]*
    (?P<dt>\d{4}[.\-]\d{2}[.\-]\d{2}[ T]\d{2}:\d{2}:\d{2})   # date + time (secs required)
    \s+(?P<offset_d>[\d.]+)                                   # t-T0, mid-exposure (days)
    \s+\S+                                                    # exposure (12*300, 26x150)
    \s+(?P<filter>[A-Za-z']{1,4})                             # filter (Rc, R)
    \s+(?P<mag>\d{1,2}\.\d{1,3})                              # mag
    \s*(?:\+/-\s*)?(?P<err>\d{1,2}\.\d{1,3})                  # err (+/- optional)
    (?:\s+(?P<ul>\d{1,2}\.\d{1,3}))?                          # UL (3-sigma limit)
    """,
    re.VERBOSE,
)


def _looks_like_fixedw_header(line: str) -> bool:
    low = line.lower()
    return (
        "date" in low
        and "filter" in low
        and "mag" in low
        and ("utstart" in low or "t-t0" in low or "exp" in low)
    )


def parse_fixed_width_table_with_spans(
    text: str, trigger_time: datetime | None = None
) -> list[tuple[PhotometryExt, Span]]:
    """Parse the SAO-RAS/IKI fixed-width photometry template. Per-row Spans.

    A row is only accepted when a matching header sits within the preceding few
    lines (a units subheader can intervene), keeping the datetime-anchored regex
    from matching stray prose elsewhere in the body.
    """
    rows: list[tuple[PhotometryExt, Span]] = []
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    for j, line in enumerate(lines):
        if not any(_looks_like_fixedw_header(lines[k]) for k in range(max(0, j - 3), j)):
            continue
        m = _FIXEDW_ROW_RE.match(line)
        if m is None:
            continue
        base = normalize_filter(m.group("filter"))
        if base not in _KNOWN_FILTERS:
            continue
        # The datetime column is when the exposures started; t-T0 is their
        # mid-point, which is the epoch of the measurement.
        ep = None
        if trigger_time is not None:
            ep = epoch_from_offset(trigger_time, float(m.group("offset_d")), "d")
        if ep is None:
            ep = epoch_from_absolute(m.group("dt").replace(".", "-"))
        obs_mjd, obs_time = ep if ep is not None else (None, None)
        row_text = line.rstrip("\r\n")
        rows.append(
            (
                PhotometryExt(
                    filter=base,
                    mag=float(m.group("mag")),
                    mag_error=float(m.group("err")),
                    limiting_mag=float(m.group("ul")) if m.group("ul") else None,
                    mag_system=infer_mag_system(base),
                    bandpass=infer_bandpass(base),
                    obs_mjd=obs_mjd,
                    obs_time=obs_time,
                ),
                Span(start=offsets[j], end=offsets[j] + len(row_text), snippet=row_text),
            )
        )
    return rows


def parse_mag_table(text: str) -> list[PhotometryExt]:
    """Detect column-aligned magnitude tables and parse rows.

    Conservative: requires an explicit header line with table keywords. Prose-style
    "we measured r = 18.42" is NOT picked up here (use parse_single_mags). This
    parser is the one that the PDF expects to lose on irregular table layouts.
    """
    return [p for p, _ in parse_mag_table_with_spans(text)]


_CELL_MAG_RE = re.compile(
    r"^(?P<limit>[<>])?\s*(?P<mag>\d{1,2}(?:\.\d+)?)"
    r"(?:\s*(?:±|\+/[-−]|\+-)\s*(?P<err>\d+(?:\.\d+)?))?$"
)


def _parse_mag_cell(cell: str) -> tuple[float | None, float | None, float | None] | None:
    """Split a magnitude cell into (mag, error, limit).

    A single column often carries the whole measurement — "22.73 +/- 0.26" for a
    detection, "> 22.37" for an upper limit — so a bare float() drops the row.
    """
    m = _CELL_MAG_RE.match(cell.strip().replace("−", "-"))
    if m is None:
        return None
    value = float(m.group("mag"))
    if m.group("limit"):
        return None, None, value
    err = float(m.group("err")) if m.group("err") else None
    return value, err, None


def parse_mag_table_with_spans(text: str) -> list[tuple[PhotometryExt, Span]]:
    """Same as parse_mag_table, plus per-row Spans covering each data line."""
    rows: list[tuple[PhotometryExt, Span]] = []
    # Use keepends=True so we can compute absolute offsets for each line.
    lines = text.splitlines(keepends=True)
    line_offsets: list[int] = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    i = 0
    while i < len(lines):
        line = lines[i]
        if not _looks_like_header(line):
            i += 1
            continue

        cols = _classify_columns(line)
        if "filter" not in cols.values() or "mag" not in cols.values():
            i += 1
            continue

        header_fields = [f for f in _COLUMN_SPLIT_RE.split(line.strip()) if f]
        expected = len(header_fields)

        j = i + 1
        # A rule drawn under the header is not a row, but nor is it the end.
        while j < len(lines) and _RULE_RE.match(lines[j]):
            j += 1
        while j < len(lines) and _looks_like_table_row(lines[j], expected):
            data_fields = [f for f in _COLUMN_SPLIT_RE.split(lines[j].strip()) if f]
            row_data: dict[str, str] = {}
            for idx, role in cols.items():
                if idx < len(data_fields):
                    row_data[role] = data_fields[idx]

            filter_token = row_data.get("filter")
            mag_token = row_data.get("mag")
            parsed = _parse_mag_cell(mag_token) if mag_token else None
            if filter_token and parsed is not None:
                mag, err, limit = parsed
                if err is None and (err_token := row_data.get("mag_error")) is not None:
                    try:
                        err = float(err_token)
                    except ValueError:
                        err = None
                row_start = line_offsets[j]
                row_text = lines[j].rstrip("\r\n")
                row_end = row_start + len(row_text)
                obs_mjd, obs_time = (None, None)
                if (epoch := epoch_from_absolute(row_data.get("date"))) is not None:
                    obs_mjd, obs_time = epoch
                rows.append(
                    (
                        PhotometryExt(
                            filter=filter_token,
                            mag=mag,
                            mag_error=err,
                            limiting_mag=limit,
                            mag_system=infer_mag_system(filter_token),
                            bandpass=infer_bandpass(filter_token),
                            obs_mjd=obs_mjd,
                            obs_time=obs_time,
                        ),
                        Span(start=row_start, end=row_end, snippet=row_text),
                    )
                )
            j += 1
        i = j if j > i + 1 else i + 1

    return rows
