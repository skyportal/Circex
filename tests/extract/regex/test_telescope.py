"""Naming the telescope a circular observed with."""

from __future__ import annotations

import pytest

from circex.extract.regex.telescope import parse_telescope


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("images obtained at the Keck-II telescope on UT 1998", "Keck-II"),
        ("with the 2.5-m Nordic Optical Telescope. We discovered", "Nordic Optical Telescope"),
        ("images obtained at the Palomar 60-inch telescope, the initial", "Palomar 60-inch"),
        ("We observed with the VLT under programme 099.D", "VLT"),
    ],
)
def test_telescope_is_read_as_written(text: str, expected: str) -> None:
    assert parse_telescope(text) == expected


def test_an_acronym_is_not_an_english_word():
    # NOT is the Nordic Optical Telescope; "not" is not a telescope.
    assert parse_telescope("This is not a detection of anything") is None
    assert parse_telescope("We used NOT to observe the field") == "NOT"


def test_prose_without_a_telescope():
    assert parse_telescope("The burst was bright in gamma rays.") is None


def test_the_article_is_not_part_of_the_name():
    assert parse_telescope("observations with the VLT") == "VLT"


def test_the_calibration_survey_is_not_the_observer():
    """The catalogue supplying reference magnitudes did not take the image."""
    assert (
        parse_telescope(
            "We observed with LATIOS on SVOM/C-GFT. The photometry was calibrated "
            "against Pan-STARRS1 DR1 catalogue."
        )
        == "SVOM/C-GFT"
    )
    assert (
        parse_telescope("differential PSF photometry of DECam images relative to Pan-STARRs")
        is None
    )


def test_a_telescope_named_only_in_a_calibration_clause_is_not_taken():
    assert parse_telescope("The limit is derived by calibrating against Pan-STARRS1.") is None
