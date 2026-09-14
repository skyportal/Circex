"""Dropped photometry rows are named, not silently lost."""

from circex.bot.skyportal_map import to_actions
from circex.schema import (
    CircularExtraction,
    Event,
    ExtractionMeta,
    Localization,
    PhotometryExt,
)


def _extraction(**row):
    return CircularExtraction(
        circular_id=1,
        event=Event(event_name="GRB 260604C"),
        localization=Localization(ra=1.0, dec=2.0),
        photometry=[PhotometryExt(**row)],
        extraction_meta=ExtractionMeta(extractor="test"),
    )


def test_row_without_a_bandpass_is_dropped_and_counted():
    actions = to_actions(
        # L is read and attributed, but sncosmo carries no equivalent, so the
        # row reaches this point with no bandpass to post with
        _extraction(filter="L", obs_mjd=61195.0, mag=19.0),
        default_instrument_id=4,
    )
    assert actions.photometry == []
    assert actions.skipped_rows == 1


def test_row_with_a_known_filter_survives():
    actions = to_actions(
        _extraction(filter="r", obs_mjd=61195.0, mag=19.0, limiting_mag=20.0),
        default_instrument_id=4,
    )
    assert len(actions.photometry) == 1
    assert actions.photometry[0].filter == "sdssr"
