"""Tests for observation-epoch resolution (ICARE P0 #2)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from astropy.time import Time

from circex.extract.protocol import Circular
from circex.extract.regex import RegexExtractor
from circex.extract.regex.dates import parse_time_offsets
from circex.extract.timing import (
    epoch_from_absolute,
    epoch_from_offset,
    normalize_pair,
    parse_observation_epoch,
    resolve_inline_offsets,
    resolve_observation_epoch,
    resolve_relative_epochs,
)
from circex.schema import CircularExtraction, ExtractionMeta, PhotometryExt, TimeOffset

# ---- epoch_from_absolute ----


def test_absolute_iso_date() -> None:
    pair = epoch_from_absolute("2024-01-02 04:30")
    assert pair is not None
    mjd, iso = pair
    assert abs(mjd - 60311.1875) < 1e-6
    assert iso.startswith("2024-01-02T04:30")


def test_absolute_bare_mjd() -> None:
    pair = epoch_from_absolute("60311.5")
    assert pair is not None
    mjd, _ = pair
    assert mjd == 60311.5


def test_absolute_number_outside_mjd_range_is_none() -> None:
    # A year or a small count is not a date we trust.
    assert epoch_from_absolute("2024") is None
    assert epoch_from_absolute("12") is None


def test_a_date_missing_its_year_or_month_is_none() -> None:
    # Filling the gap from today's clock yields a confident wrong date.
    assert epoch_from_absolute("Jan 24.16") is None
    assert epoch_from_absolute("Dec.12.22") is None
    assert epoch_from_absolute("15:13:13") is None
    assert epoch_from_absolute("2004 Oct 10.05") is not None


def test_absolute_garbage_is_none() -> None:
    assert epoch_from_absolute("not a date") is None
    assert epoch_from_absolute(None) is None
    assert epoch_from_absolute("") is None


# ---- epoch_from_offset ----


def test_offset_hours() -> None:
    t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    pair = epoch_from_offset(t0, 1.0, "h")
    assert pair is not None
    mjd, iso = pair
    assert abs(mjd - (60310.0 + 1.0 / 24.0)) < 1e-6
    assert iso.startswith("2024-01-01T01:00")


def test_offset_unknown_unit_is_none() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    assert epoch_from_offset(t0, 5.0, "weeks") is None


def test_offset_naive_datetime_treated_as_utc() -> None:
    naive = datetime(2024, 1, 1, 0, 0, 0)
    aware = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert epoch_from_offset(naive, 30.0, "s") == epoch_from_offset(aware, 30.0, "s")


# ---- normalize_pair (backfill) ----


def test_normalize_pair_from_mjd() -> None:
    pair = normalize_pair(60311.5, None)
    assert pair is not None and pair[0] == 60311.5


def test_normalize_pair_from_iso() -> None:
    pair = normalize_pair(None, "2024-01-02T04:30:00Z")
    assert pair is not None and abs(pair[0] - 60311.1875) < 1e-6


def test_normalize_pair_none() -> None:
    assert normalize_pair(None, None) is None


# ---- PhotometryExt pair backfill ----


def test_photometry_backfills_mjd_from_obs_time() -> None:
    p = PhotometryExt(filter="r", mag=20.0, obs_time="2024-01-02T04:30:00Z")
    assert p.obs_mjd is not None and abs(p.obs_mjd - 60311.1875) < 1e-6


def test_photometry_backfills_obs_time_from_mjd() -> None:
    p = PhotometryExt(filter="r", mag=20.0, obs_mjd=60311.5)
    assert p.obs_time is not None and p.obs_time.startswith("2024-01-02")


def test_photometry_no_epoch_stays_null() -> None:
    p = PhotometryExt(filter="r", mag=20.0)
    assert p.obs_mjd is None and p.obs_time is None


# ---- resolve_relative_epochs (single-epoch rule) ----


def _extraction(rows: list[PhotometryExt], offsets: list[TimeOffset]) -> CircularExtraction:
    return CircularExtraction(
        circular_id=1,
        photometry=rows,
        time_offsets=offsets,
        extraction_meta=ExtractionMeta(extractor="test"),
    )


def test_resolve_single_offset_applies_to_all_rows() -> None:
    ex = _extraction(
        [PhotometryExt(filter="r", mag=19.5), PhotometryExt(filter="g", mag=20.1)],
        [TimeOffset(value=1.0, unit="h", reference="T+")],
    )
    resolve_relative_epochs(ex, datetime(2024, 1, 1, tzinfo=UTC))
    assert all(p.obs_mjd is not None for p in ex.photometry)
    assert ex.photometry[0].obs_mjd == ex.photometry[1].obs_mjd


def test_resolve_multiple_distinct_offsets_is_ambiguous_noop() -> None:
    ex = _extraction(
        [PhotometryExt(filter="r", mag=19.5)],
        [
            TimeOffset(value=1.0, unit="h", reference="T+"),
            TimeOffset(value=5.0, unit="h", reference="T+"),
        ],
    )
    resolve_relative_epochs(ex, datetime(2024, 1, 1, tzinfo=UTC))
    assert ex.photometry[0].obs_mjd is None


def test_resolve_no_trigger_time_noop() -> None:
    ex = _extraction(
        [PhotometryExt(filter="r", mag=19.5)],
        [TimeOffset(value=1.0, unit="h", reference="T+")],
    )
    resolve_relative_epochs(ex, None)
    assert ex.photometry[0].obs_mjd is None


def test_resolve_does_not_overwrite_absolute_epoch() -> None:
    row = PhotometryExt(filter="r", mag=19.5, obs_mjd=60000.0)
    ex = _extraction([row], [TimeOffset(value=1.0, unit="h", reference="T+")])
    resolve_relative_epochs(ex, datetime(2024, 1, 1, tzinfo=UTC))
    assert ex.photometry[0].obs_mjd == 60000.0


# ---- regex extractor integration ----


def test_regex_table_resolves_absolute_epoch() -> None:
    body = (
        "Date              Filter  Mag     Err\n"
        "2024-01-02 04:30  r       20.42   0.05\n"
        "60311.5           g       21.10   0.07"
    )
    r = RegexExtractor().extract(Circular(circular_id=1, subject="", body=body))
    mjds = sorted(p.obs_mjd for p in r.photometry if p.obs_mjd is not None)
    assert len(mjds) == 2
    assert abs(mjds[0] - 60311.1875) < 1e-6


def test_regex_relative_resolved_with_trigger_time() -> None:
    body = "We observed at T+1 h and measured r = 19.5 mag."
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    r = RegexExtractor().extract(Circular(circular_id=2, subject="", body=body, trigger_time=t0))
    detected = [p for p in r.photometry if p.filter == "r"]
    assert detected and detected[0].obs_mjd is not None


def test_regex_relative_unresolved_without_trigger_time() -> None:
    body = "We observed at T+1 h and measured r = 19.5 mag."
    r = RegexExtractor().extract(Circular(circular_id=3, subject="", body=body))
    assert all(p.obs_mjd is None for p in r.photometry)


# ---- parse_observation_epoch / resolve_observation_epoch ----


def test_parse_observation_epoch_from_prose() -> None:
    from circex.extract.timing import parse_observation_epoch

    pair = parse_observation_epoch("We observed from 2026-06-05 03:41 to 03:51 UTC.")
    assert pair is not None
    mjd, iso = pair
    assert iso.startswith("2026-06-05T03:41")


def test_resolve_observation_epoch_backfills_untimed_prose_rows() -> None:
    from circex.extract.timing import resolve_observation_epoch

    ex = CircularExtraction(
        circular_id=44834,
        photometry=[PhotometryExt(filter="g", mag=19.69), PhotometryExt(filter="r", mag=19.56)],
        extraction_meta=ExtractionMeta(extractor="test"),
    )
    resolve_observation_epoch(ex, "We observed on 2026-06-05 03:41 UTC and obtained griz.")
    assert all(p.obs_mjd is not None for p in ex.photometry)
    assert ex.photometry[0].obs_mjd == ex.photometry[1].obs_mjd
    assert any("observation epoch" in n for n in ex.extraction_meta.notes)


def test_resolve_observation_epoch_does_not_clobber_timed_rows() -> None:
    from circex.extract.timing import resolve_observation_epoch

    ex = CircularExtraction(
        circular_id=1,
        photometry=[
            PhotometryExt(filter="g", mag=19.0, obs_mjd=60000.0),
            PhotometryExt(filter="r", mag=19.5),
        ],
        extraction_meta=ExtractionMeta(extractor="test"),
    )
    resolve_observation_epoch(ex, "observed on 2026-06-05 03:41 UTC")
    assert ex.photometry[0].obs_mjd == 60000.0  # any-timed guard: leave all as-is
    assert ex.photometry[1].obs_mjd is None


# Date and offset forms that radio circulars use, each drawn from a real GCN.


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # ATCA separates date from time with an underscore (GCN 33475).
        ("We observed with ATCA between 2023-03-12_02:30 UT", "2023-03-12T02:30:00Z"),
        # ...or with "at" (GCN 35155).
        ("ATCA observed on 2023-11-20 at 14:30 UT for 3 hours", "2023-11-20T14:30:00Z"),
        # Spelled-out month, year first (GCN 21545).
        ("observations beginning at 2017 August 18 02:09:00 UT", "2017-08-18T02:09:00Z"),
        # Abbreviated month with hyphens (GCN 21613).
        ("observation started on 2017-Aug-19 22:01:48 UT", "2017-08-19T22:01:48Z"),
        # Day first, with the UT label leading (GCN 21708).
        ("observations carried out on UT 20 August 2017 08:00", "2017-08-20T08:00:00Z"),
        # Abbreviated month and "at" (GCN 38640).
        ("started observing on 2024 Dec 14 at 01:36:13 UTC", "2024-12-14T01:36:13Z"),
        # A date with no time of day (GCN 34843).
        ("ATCA observed the burst on 2023-09-11 UT", "2023-09-11T00:00:00Z"),
    ],
)
def test_observation_epochs_radio_circulars_write(text: str, expected: str) -> None:
    result = parse_observation_epoch(text)
    assert result is not None
    assert result[1] == expected


def test_an_initial_is_not_read_as_the_end_of_the_sentence() -> None:
    """ "PI: G. Anderson" sits between the verb and the date in every PanRadio circular."""
    text = (
        "ATCA observed the long GRB 231118A, first detected by the Fermi GRB Team "
        '(GCN 35100), as part of the ATCA "PanRadio GRB" Large Project C3542 '
        "(PI: G. Anderson) on 2023-11-20 at 14:30 UT for 3 hours."
    )
    result = parse_observation_epoch(text)
    assert result is not None
    assert result[1] == "2023-11-20T14:30:00Z"


@pytest.mark.parametrize(
    ("text", "value", "unit"),
    [
        ("just 27 minutes post-burst", 27.0, "m"),
        ("(~4.5 days post-burst)", 4.5, "d"),
        ("5.9d after the Swift/BAT trigger", 5.9, "d"),
        ("53.84 days after the EP trigger", 53.84, "d"),
        ("13 hours after the Fermi trigger time", 13.0, "h"),
    ],
)
def test_offsets_measured_from_the_burst_as_well_as_the_trigger(
    text: str, value: float, unit: str
) -> None:
    offsets = parse_time_offsets(text)
    assert [(o.value, o.unit) for o in offsets] == [(value, unit)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Month DD YYYY, which the year-first and day-first forms both missed.
        (
            "attached to the Subaru telescope on August 18 2017 UT we observed the field",
            "2017-08-18T00:00:00Z",
        ),
        # an ordinal day
        ("GROND started observing on August 19th 2017", "2017-08-19T00:00:00Z"),
        # the date ahead of the verb
        (
            "On 2017 Aug 18 UT in the process of observing several galaxies",
            "2017-08-18T00:00:00Z",
        ),
        # the clock ahead of the date: without this the time became midnight
        ("Observations started at 23:15 UT on August 18th 2017.", "2017-08-18T23:15:00Z"),
    ],
)
def test_more_observation_epoch_forms(text: str, expected: str) -> None:
    result = parse_observation_epoch(text)
    assert result is not None
    assert result[1] == expected


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("24.3 ks after the EP trigger", 24300.0),
        ("19.2 ks after the BAT trigger", 19200.0),
        ("T+19.2 ks", 19200.0),
    ],
)
def test_kilosecond_offsets(text: str, seconds: float) -> None:
    """Swift and UVOT quote offsets in ks; the schema's units stop at seconds."""
    offsets = parse_time_offsets(text)
    assert [(o.value, o.unit) for o in offsets] == [(seconds, "s")]


@pytest.mark.parametrize(
    ("text", "expected_mjd"),
    [
        # A decimal day carries the time: Aug 18.85 is the 18th at 20:24 UT.
        ("We observed on 2017 Aug 18.85 UT.", 57983.85),
        ("Observations began on 2017 August 18.99 UT.", 57983.99),
        ("We observed the field on 2017-08-18.85 UT.", 57983.85),
        ("We observed on 2017 Aug 20.424 UT.", 57985.424),
        # An explicit clock time still wins, and a whole day stays whole.
        ("We observed on 2017 Aug 18 at 20:24 UT.", 57983.85),
        ("We observed on 2017 Aug 18 UT.", 57983.0),
    ],
)
def test_fractional_day_epoch(text: str, expected_mjd: float) -> None:
    extraction = CircularExtraction(
        circular_id=1,
        photometry=[PhotometryExt(filter="r", mag=20.0)],
        extraction_meta=ExtractionMeta(extractor="test"),
    )
    resolve_observation_epoch(extraction, text)
    assert extraction.photometry[0].obs_mjd == pytest.approx(expected_mjd, abs=1e-3)


def test_each_row_takes_the_offset_on_its_own_line():
    """GCN 45505 reported three epochs, each timed beside its magnitude.

    The circular-level rule refuses several distinct offsets as ambiguous; these
    are not ambiguous, because each sits in the clause with its measurement.
    """
    body = (
        "r = 20.92 +/- 0.19 AB (mid-time 38.66 min after the trigger);\n"
        "r = 20.46 +/- 0.19 AB (mid-time 2.30 hr after the trigger);\n"
        "r = 21.8 +/- 0.4 AB (mid-time 2.74 hr after the trigger).\n"
    )
    trigger = datetime(2026, 9, 3, 12, 36, 47, tzinfo=UTC)
    extraction = CircularExtraction(
        circular_id=45505,
        photometry=[
            PhotometryExt(filter="r", mag=20.92),
            PhotometryExt(filter="r", mag=20.46),
            PhotometryExt(filter="r", mag=21.8),
        ],
        extraction_meta=ExtractionMeta(extractor="test"),
    )
    resolve_inline_offsets(extraction, body, trigger)
    hours = [(row.obs_mjd - Time(trigger).mjd) * 24 for row in extraction.photometry]
    assert hours[0] == pytest.approx(38.66 / 60, abs=1e-3)
    assert hours[1] == pytest.approx(2.30, abs=1e-3)
    assert hours[2] == pytest.approx(2.74, abs=1e-3)


def test_an_offset_away_from_the_measurement_is_left_alone():
    # One offset in prose, magnitudes elsewhere: the per-row rule must not guess.
    body = "We observed 3 hours after the trigger.\nr = 20.9\nz = 21.4\n"
    extraction = CircularExtraction(
        circular_id=1,
        photometry=[PhotometryExt(filter="r", mag=20.9)],
        extraction_meta=ExtractionMeta(extractor="test"),
    )
    resolve_inline_offsets(extraction, body, datetime(2026, 9, 3, tzinfo=UTC))
    assert extraction.photometry[0].obs_mjd is None


def test_each_candidate_is_timed_from_its_own_paragraph():
    """GCN 45552: two candidates, each described with the time it was seen."""
    from circex.extract.timing import resolve_object_epochs
    from circex.schema import CircularExtraction, ExtractionMeta, PhotometryExt

    body = (
        "GOTO26jjg / AT 2026abfp is coincident with a galaxy. The source is "
        "detected with L = 20.58 at 09:02:46 UTC on 2026-09-11.\n\n"
        "GOTO26jjj is coincident with a galaxy. The source was covered in the "
        "single epoch reported above, beginning at 09:30:34 UTC on 2026-09-11.\n"
    )
    extraction = CircularExtraction(
        circular_id=45552,
        photometry=[
            PhotometryExt(object_name="AT 2026abfp", object_aliases=["GOTO26jjg"], mag=20.58),
            PhotometryExt(object_name="GOTO26jjj", mag=20.28),
        ],
        extraction_meta=ExtractionMeta(extractor="test", latency_ms=0.0),
    )
    assert len(resolve_object_epochs(extraction, body)) == 2
    assert [p.obs_time for p in extraction.photometry] == [
        "2026-09-11T09:02:46Z",
        "2026-09-11T09:30:34Z",
    ]


def test_a_paragraph_naming_two_objects_times_neither():
    from circex.extract.timing import resolve_object_epochs
    from circex.schema import CircularExtraction, ExtractionMeta, PhotometryExt

    body = "AT 2026aaa and AT 2026bbb were both observed at 09:02:46 UTC on 2026-09-11.\n"
    extraction = CircularExtraction(
        circular_id=1,
        photometry=[
            PhotometryExt(object_name="AT 2026aaa", mag=20.0),
            PhotometryExt(object_name="AT 2026bbb", mag=20.0),
        ],
        extraction_meta=ExtractionMeta(extractor="test", latency_ms=0.0),
    )
    assert resolve_object_epochs(extraction, body) == set()
    assert all(p.obs_time is None for p in extraction.photometry)
