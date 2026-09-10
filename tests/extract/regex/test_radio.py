"""Radio flux-density parsing, the frequency crosswalk, and flux-space mapping."""

from __future__ import annotations

import pytest

from circex.bot.skyportal_map import ASSUMED_FLUX_ERROR_FRACTION, to_actions
from circex.extract.protocol import Circular
from circex.extract.regex.extractor import RegexExtractor
from circex.extract.regex.radio import (
    bandpass_for_frequency,
    normalize_flux_unit,
    parse_radio_with_spans,
    to_ujy,
)
from circex.schema import (
    CircularExtraction,
    Event,
    ExtractionMeta,
    Localization,
    PhotometryExt,
)

# GCN 33475, the shape that carries a detection and a limit in one circular.
GCN_33475 = (
    "We observed GRB 230703A (Fermi GBM Team GCN 33405) with the Australia Telescope\n"
    "Compact Array (ATCA) between 2023-03-12_02:30 UT and 2023-03-12_07:30 UT\n"
    "(~4.5 days post-burst). In our preliminary analysis, we detect a radio source\n"
    "coincident with the X-ray (Burrows et al. GCN 33429) and optical (Levan et al.\n"
    "GCN 33439) counterpart with a flux density of 120+/-30 microJy/beam at 9 GHz.\n"
    "We also obtain an 3 sigma upper limit of 90 microJy/beam at 5.5 GHz.\n"
)

# GCN 33433, where two flux densities are distributed over two frequencies.
GCN_33433 = (
    "The Australia Telescope Compact Array (ATCA) automatically triggered on\n"
    "the Swift-BAT detection of the short GRB 230217A at 5.5 and 9 GHz.\n"
    "At the position of the proposed counterpart, we detected a source at both\n"
    "5.5 GHz and 9 GHz with flux densities of 170 +/- 30 and 150 +/- 20\n"
    "microJy/beam, respectively.\n"
)


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("microJy/beam", "uJy"),
        ("micro Jy", "uJy"),
        ("uJy", "uJy"),
        ("μJy", "uJy"),
        ("mJy", "mJy"),
        ("Jy", "Jy"),
    ],
)
def test_unit_aliases_fold_to_the_enum(written: str, expected: str) -> None:
    assert normalize_flux_unit(written) == expected


def test_flux_density_converts_to_microjansky() -> None:
    assert to_ujy(1.0, "mJy") == 1000.0
    assert to_ujy(2.5, "Jy") == 2_500_000.0


@pytest.mark.parametrize(
    ("ghz", "expected"),
    [
        (5.5, "radio-6GHz"),
        (9.0, "radio-10GHz"),
        (1.284, "radio-1.4GHz"),
        (93.0, "radio-93GHz"),
        (230.0, "sma-230GHz"),
    ],
)
def test_frequency_maps_to_nearest_bandpass(ghz: float, expected: str) -> None:
    assert bandpass_for_frequency(ghz) == expected


@pytest.mark.parametrize("ghz", [0.05, 650.0])
def test_frequency_outside_every_band_has_no_bandpass(ghz: float) -> None:
    """Nothing is representable there, so the row is dropped rather than mislabelled."""
    assert bandpass_for_frequency(ghz) is None


def test_detection_and_upper_limit_in_one_circular() -> None:
    rows = [row for row, _ in parse_radio_with_spans(GCN_33475)]
    assert len(rows) == 2

    detection, limit = rows
    assert detection.is_detection is True
    assert (detection.flux_density, detection.flux_density_error) == (120.0, 30.0)
    assert detection.frequency_ghz == 9.0
    assert detection.bandpass == "radio-10GHz"

    assert limit.is_detection is False
    assert limit.limiting_flux_density == 90.0
    assert limit.limiting_mag_sigma == 3.0
    assert limit.frequency_ghz == 5.5


def test_respectively_pairs_values_with_frequencies_in_order() -> None:
    rows = [row for row, _ in parse_radio_with_spans(GCN_33433)]
    assert [(r.flux_density, r.frequency_ghz) for r in rows] == [(170.0, 5.5), (150.0, 9.0)]


def test_several_units_in_one_clause_yield_nothing() -> None:
    """A column table cannot be paired safely, so it is left to the LLM path."""
    text = "Flux Density 3-sigma Limit for SSS17a 8.5 GHz: 0.66 mJy 120 uJy 10.5 GHz: 0.54 mJy"
    assert parse_radio_with_spans(text) == []


def test_frequency_cited_for_comparison_is_not_borrowed() -> None:
    """The 15 GHz belongs to the instrument being compared against, not this row."""
    text = (
        "The afterglow is detected at a flux density of ~2 mJy, indicating that the peak "
        "emission is still at higher frequencies than the 15 GHz detection by AMI-LA."
    )
    assert parse_radio_with_spans(text) == []


def test_standalone_measurement_lines_are_parsed() -> None:
    text = "We report preliminary flux densities of:\n~0.6 mJy at 8.5 GHz\n~0.5 mJy at 10.5 GHz\n"
    rows = [row for row, _ in parse_radio_with_spans(text)]
    assert [(r.flux_density, r.frequency_ghz) for r in rows] == [(0.6, 8.5), (0.5, 10.5)]


def test_host_galaxy_flux_is_not_transient_photometry() -> None:
    text = "We detect the host galaxy at a flux density of 0.22 mJy at 6 GHz."
    assert parse_radio_with_spans(text) == []


def test_radio_rows_reach_the_extractor() -> None:
    extraction = RegexExtractor().extract(
        Circular(circular_id=33475, subject="GRB 230307A: ATCA radio detection", body=GCN_33475)
    )
    radio = [r for r in extraction.photometry if r.frequency_ghz is not None]
    assert len(radio) == 2
    assert all(r.obs_mjd is not None for r in radio), "the ATCA underscore date must resolve"


def _radio_extraction(row: PhotometryExt) -> CircularExtraction:
    return CircularExtraction(
        circular_id=1,
        event=Event(event_name="GRB 230307A"),
        localization=Localization(ra=10.0, dec=-20.0),
        photometry=[row],
        extraction_meta=ExtractionMeta(extractor="regex"),
    )


def test_detection_becomes_a_flux_space_point() -> None:
    row = PhotometryExt(
        frequency_ghz=9.0,
        flux_density=120.0,
        flux_density_error=30.0,
        flux_density_unit="uJy",
        obs_mjd=60015.1,
    )
    actions = to_actions(_radio_extraction(row), default_instrument_id=7)
    payload = actions.photometry[0].to_payload()
    assert payload["flux"] == 120.0
    assert payload["fluxerr"] == 30.0
    assert payload["zp"] == 23.9
    assert payload["filter"] == "radio-10GHz"
    assert "mag" not in payload


def test_upper_limit_becomes_a_null_flux_with_sigma_scaled_error() -> None:
    """SkyPortal derives the limiting magnitude from fluxerr, so the limit is divided by sigma."""
    row = PhotometryExt(
        frequency_ghz=5.5,
        limiting_flux_density=90.0,
        limiting_mag_sigma=3.0,
        flux_density_unit="uJy",
        obs_mjd=60015.1,
    )
    actions = to_actions(_radio_extraction(row), default_instrument_id=7)
    payload = actions.photometry[0].to_payload()
    assert payload["flux"] is None
    assert payload["fluxerr"] == 30.0


def test_millijansky_is_converted_to_microjansky() -> None:
    row = PhotometryExt(
        frequency_ghz=6.0,
        flux_density=2.0,
        flux_density_error=0.1,
        flux_density_unit="mJy",
        obs_mjd=60015.1,
    )
    actions = to_actions(_radio_extraction(row), default_instrument_id=7)
    payload = actions.photometry[0].to_payload()
    assert (payload["flux"], payload["fluxerr"]) == (2000.0, 100.0)


def test_detection_without_an_uncertainty_gets_a_flagged_nominal_error() -> None:
    """About half of radio detections quote no error; they are posted, but marked assumed."""
    row = PhotometryExt(
        frequency_ghz=9.0, flux_density=400.0, flux_density_unit="uJy", obs_mjd=60015.1
    )
    actions = to_actions(_radio_extraction(row), default_instrument_id=7)
    point = actions.photometry[0]
    payload = point.to_payload()
    assert payload["flux"] == 400.0
    assert payload["fluxerr"] == 400.0 * ASSUMED_FLUX_ERROR_FRACTION
    assert point.altdata["flux_density_error_assumed"] is True


def test_a_reported_uncertainty_is_never_marked_assumed() -> None:
    row = PhotometryExt(
        frequency_ghz=9.0,
        flux_density=400.0,
        flux_density_error=25.0,
        flux_density_unit="uJy",
        obs_mjd=60015.1,
    )
    point = to_actions(_radio_extraction(row), default_instrument_id=7).photometry[0]
    assert point.to_payload()["fluxerr"] == 25.0
    assert "flux_density_error_assumed" not in point.altdata


def test_the_extraction_itself_keeps_the_uncertainty_null() -> None:
    """The nominal error is a write-time policy; the stored extraction stays truthful."""
    rows = [row for row, _ in parse_radio_with_spans("A flux density of ~0.4 mJy at 9 GHz.")]
    assert rows[0].flux_density_error is None


@pytest.mark.parametrize(
    "text",
    [
        "no significant X-ray source was detected within the 2.4 arcmin radius",
        "We observed the field with ATCA at radio frequencies",
        "The burst was detected by the Fermi GBM team",
        "a source was found in the error box",
    ],
)
def test_band_and_function_words_do_not_classify(text: str) -> None:
    """The taxonomy lists 'x-ray'/'radio'/'by'/'in' as other names; prose is not a class."""
    from circex.extract.regex.classification import parse_classification_with_span

    assert parse_classification_with_span(text) is None


def test_a_real_class_name_still_classifies() -> None:
    from circex.extract.regex.classification import parse_classification_with_span

    hit = parse_classification_with_span("We classify this as a Type Ia supernova.")
    assert hit is not None


# --- error-box radii are not magnitudes -------------------------------------

MASTER_ERRORBOX = (
    "MASTER-Tavrida was pointed to GRB240618.80 (trigger No 740430582,"
    "22h 48m 28.80s , +71d 41m 24.0s, R=74.88) errorbox 2162 sec after notice time."
)


def test_an_error_box_radius_is_not_photometry() -> None:
    """R is a radius in this template; 58 such values fall inside the mag range."""
    from circex.extract.regex.mag_table import parse_single_mags

    assert parse_single_mags(MASTER_ERRORBOX) == []


def test_an_error_box_radius_inside_the_magnitude_range_is_still_rejected() -> None:
    from circex.extract.regex.mag_table import parse_single_mags

    text = MASTER_ERRORBOX.replace("R=74.88", "R=19.5")
    assert parse_single_mags(text) == []


def test_a_real_magnitude_of_the_same_value_survives() -> None:
    from circex.extract.regex.mag_table import parse_single_mags

    rows = parse_single_mags("We detect the afterglow at R = 19.5 +/- 0.1.")
    assert [(r.filter, r.mag) for r in rows] == [("R", 19.5)]


@pytest.mark.parametrize("mag", [31.0, 74.88, 4.9])
def test_magnitudes_outside_the_optical_range_are_rejected(mag: float) -> None:
    from circex.extract.regex.mag_table import parse_single_mags

    assert parse_single_mags(f"the source was at R = {mag} in our image") == []


# --- classification triggers that are not classifications --------------------


@pytest.mark.parametrize(
    "text",
    [
        # software that shares a class alias, masking the real classification
        "Using SNID (Blondin & Tonry 2007) we classify this transient",
        # institutions and funders in the acknowledgements
        "We thank the Tata Institute of Fundamental Research for support.",
        "This work was supported by funding from DST-SERB and IUSSTF.",
        "We acknowledge the Siding Spring Observatory (SSO), Australia.",
        # a class named as the target of a search
        "We searched the data to look for any coincident hard X-ray flash.",
    ],
)
def test_prose_that_names_no_class_is_not_classified(text: str) -> None:
    from circex.extract.regex.classification import parse_classification_with_span

    assert parse_classification_with_span(text) is None


def test_a_masked_classification_is_recovered() -> None:
    """SNID sorted before the real class, so the tool name used to win."""
    from circex.extract.regex.classification import parse_classification_with_span

    hit = parse_classification_with_span(
        "Using SNID we classify the source as a Type Ia supernova at redshift z = 0.1"
    )
    assert hit is not None
    assert hit[0].classification == "Ia"


def test_a_declared_x_ray_flash_still_classifies() -> None:
    from circex.extract.regex.classification import parse_classification_with_span

    hit = parse_classification_with_span("The event is classified as an X-ray Flash.")
    assert hit is not None
    assert hit[0].classification == "X-ray Flash"


def test_the_nearest_cue_decides_the_redshift_measure() -> None:
    """A circular comparing its own value with someone else's names both methods."""
    from circex.extract.regex.redshift import parse_redshift

    result = parse_redshift(
        "the photometric redshift of 0.995 +- 0.352, which is marginally "
        "consistent with the redshift of 1.673 obtained with GTC spectroscopy"
    )
    assert result is not None
    assert result.redshift == 0.995
    assert result.redshift_measure == "photometric"


@pytest.mark.parametrize(
    ("text", "measure"),
    [
        ("We measure a spectroscopic redshift of z = 0.151", "spectroscopic"),
        ("The host has a photometric redshift of 0.63.", "photometric"),
        ("The burst is at z = 1.2.", None),
    ],
)
def test_an_unambiguous_measure_is_unchanged(text: str, measure: str | None) -> None:
    from circex.extract.regex.redshift import parse_redshift

    result = parse_redshift(text)
    assert result is not None
    assert result.redshift_measure == measure


def test_a_frequency_stated_before_the_anchor_binds_to_its_own_measurement():
    """GCN 45547: "a 10 GHz flux density of ~5.5 mJy and a 104.5 GHz ... of ~50 mJy"."""
    from circex.extract.regex.radio import parse_radio_with_spans

    body = (
        "From a preliminary analysis, we detect a bright radio source with a "
        "10 GHz flux density of ~5.5 mJy and a 104.5 GHz flux density of ~50 mJy."
    )
    rows = [r for r, _ in parse_radio_with_spans(body)]
    assert [(r.frequency_ghz, r.flux_density) for r in rows] == [(10.0, 5.5), (104.5, 50.0)]
    assert all(r.is_detection for r in rows)


def test_a_distributive_respectively_clause_still_pairs_in_order():
    """The leading-frequency rule must not disturb trailing "respectively" lists."""
    from circex.extract.regex.radio import parse_radio_with_spans

    body = (
        "We measure flux densities of 170 +/- 30 and 150 +/- 20 microJy/beam "
        "at 6 and 10 GHz, respectively."
    )
    rows = [r for r, _ in parse_radio_with_spans(body)]
    assert [(r.frequency_ghz, r.flux_density) for r in rows] == [(6.0, 170.0), (10.0, 150.0)]


def test_a_frequency_qualifying_the_anchor_may_carry_the_word_band():
    """GCN 15002: "The 1390 MHz band flux density ... is 792+/-44 uJy"."""
    from circex.extract.regex.radio import parse_radio_with_spans

    body = (
        "The 1390 MHz band flux density of the afterglow is 792+/-44 uJy and "
        "610 MHz flux density of the afterglow is 457+/-75 uJy."
    )
    rows = [r for r, _ in parse_radio_with_spans(body)]
    assert [(round(r.frequency_ghz, 2), r.flux_density) for r in rows] == [
        (1.39, 792.0),
        (0.61, 457.0),
    ]


def test_one_value_over_a_pair_of_frequencies_is_declined():
    """GCN 21900: "17/21 GHz flux density < 42 uJy" names no single frequency."""
    from circex.extract.regex.radio import parse_radio_with_spans

    assert parse_radio_with_spans("17/21 GHz flux density < 42 uJy") == []
