def test_a_filter_the_llm_read_gets_its_bandpass():
    # The LLM owns photometry and writes the filter it read, not a canonical
    # bandpass, so rows arrive as filter="J" with bandpass=None. On fritz every
    # row but one carried a magnitude with no bandpass, which cannot be plotted
    # or fitted.
    from circex.extract.hybrid import _fill_bandpasses
    from circex.schema.photometry import PhotometryExt

    rows = [
        PhotometryExt(mag=17.13, filter="J"),
        PhotometryExt(mag=14.68, filter="white"),
        PhotometryExt(mag=19.0, filter="not-a-filter"),
        PhotometryExt(mag=18.0, filter="R", bandpass="already-set"),
    ]
    _fill_bandpasses({"photometry": rows})
    assert rows[0].bandpass == "2massj"
    assert rows[1].bandpass == "uvot::white"
    # An unrecognised filter stays empty rather than being guessed at.
    assert rows[2].bandpass is None
    # A bandpass the extractor already settled is not overwritten.
    assert rows[3].bandpass == "already-set"


def test_filling_bandpasses_tolerates_no_photometry():
    from circex.extract.hybrid import _fill_bandpasses

    _fill_bandpasses({})
    _fill_bandpasses({"photometry": None})
    _fill_bandpasses({"photometry": []})


def test_the_telescope_settles_a_filter_that_names_two_bands():
    # Swift/UVOT's v, b and u are its own filters, not Bessell's or Sloan's.
    # The filter token alone sends them to the wrong instrument's response.
    from circex.extract.regex.mag_table import infer_bandpass_for

    assert infer_bandpass_for("u", "Swift/UVOT") == "uvot::u"
    assert infer_bandpass_for("u", "SDSS") == "sdssu"
    assert infer_bandpass_for("v", "Swift/UVOT") == "uvot::v"
    assert infer_bandpass_for("white_FC", "Swift/UVOT") == "uvot::white"
    assert infer_bandpass_for("B", "SVOM/VT") == "svomvtb"
    assert infer_bandpass_for("R", "SVOM/VT") == "svomvtr"
    assert infer_bandpass_for("R", "NOT") == "bessellr"


def test_an_instrument_with_one_band_states_no_filter():
    # A Swift/XRT row says "XRT" where a filter would go, and an SVOM/GRM row
    # says an energy range. The band is the instrument.
    from circex.extract.regex.mag_table import infer_bandpass_for

    assert infer_bandpass_for("XRT", "Swift XRT") == "swiftxrt"
    assert infer_bandpass_for("15-5000 keV", "SVOM/GRM") == "svomgrm"
    assert infer_bandpass_for(None, "Einstein Probe EP-WXT") == "epwxt"


def test_a_primed_sloan_filter_is_the_unprimed_band():
    from circex.extract.regex.mag_table import infer_bandpass_for

    assert infer_bandpass_for("r'", "GROWTH-India Telescope (GIT)") == "sdssr"
    assert infer_bandpass_for("g'", "D50 Ondrejov") == "sdssg"


def test_a_band_skyportal_does_not_carry_stays_unnamed():
    # Fermi GBM and AstroSat CZTI state a gamma-ray energy range and SkyPortal
    # registers no bandpass for either. Naming one would be worse than leaving
    # the row without: it would be plotted against the wrong response.
    from circex.extract.regex.mag_table import infer_bandpass_for

    assert infer_bandpass_for("50-300 keV", "Fermi GBM") is None
    assert infer_bandpass_for("20-200 keV", "AstroSat CZTI (CZT detectors)") is None
    assert infer_bandpass_for("15-150 keV", "Swift-BAT") is None
