"""Tests for date / T+offset parsers."""

from __future__ import annotations

from circex.extract.regex.dates import parse_time_offsets


def test_t_plus_seconds() -> None:
    offsets = parse_time_offsets("Observations began at T+234s.")
    assert len(offsets) == 1
    o = offsets[0]
    assert o.value == 234.0
    assert o.unit == "s"
    assert o.reference == "T+"


def test_t_minus_seconds() -> None:
    offsets = parse_time_offsets("Pre-trigger frame at T-30 s.")
    assert len(offsets) == 1
    assert offsets[0].value == -30.0
    assert offsets[0].reference == "T-"


def test_t_plus_hours() -> None:
    offsets = parse_time_offsets("T+8.5 hours after burst.")
    assert any(o.value == 8.5 and o.unit == "h" for o in offsets)


def test_post_trigger_phrasing() -> None:
    offsets = parse_time_offsets("4 hours after the trigger we observed.")
    assert any(o.value == 4.0 and o.unit == "h" and o.reference == "trigger" for o in offsets)


def test_multiple_offsets() -> None:
    text = "First epoch at T+100s, second at T+30 minutes, third at T+2 hours."
    offsets = parse_time_offsets(text)
    units = [o.unit for o in offsets]
    assert "s" in units and "m" in units and "h" in units


def test_no_offsets() -> None:
    assert parse_time_offsets("No time offsets mentioned here.") == []


def test_elapsed_time_written_as_a_difference():
    """ "T-To=11h" states the same offset as "11 hours after the trigger"."""
    assert [(o.value, o.unit, o.reference) for o in parse_time_offsets("mid-time at T-To=11h")] == [
        (11.0, "h", "trigger")
    ]
    assert [(o.value, o.unit, o.reference) for o in parse_time_offsets("t-t0 = 2.30 hr")] == [
        (2.3, "h", "trigger")
    ]


def test_a_stated_mid_time_is_an_offset_from_the_trigger():
    assert [(o.value, o.unit) for o in parse_time_offsets("(mid. time = 8.1358 hours)")] == [
        (8.1358, "h")
    ]
    # A clock time is not an elapsed time.
    assert parse_time_offsets("mid time = 03:41 UT") == []


def test_a_mid_time_already_tied_to_the_trigger_is_counted_once():
    assert len(parse_time_offsets("mid-time 38.66 min after the trigger")) == 1


def test_an_offset_can_be_measured_from_the_grb_or_the_merger():
    """Circulars name the event as often as they name the trigger."""
    from circex.extract.regex.dates import parse_time_offsets

    assert parse_time_offsets("(5.48 days after the GRB)")[0].value == 5.48
    assert parse_time_offsets("2.3 days after the merger")[0].unit == "d"
    assert parse_time_offsets("16.2 hours after the burst")[0].value == 16.2


def test_a_notice_is_not_the_event_it_announces():
    """A notice goes out seconds to minutes after the trigger, so an offset
    measured from it is not an offset from T0."""
    from circex.extract.regex.dates import parse_time_offsets

    assert parse_time_offsets("12 s after the notice") == []
    assert parse_time_offsets("3 hours after the alert") == []
