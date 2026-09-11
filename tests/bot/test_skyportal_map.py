"""Tests for the CircularExtraction -> SkyPortal mapping (docs/design_skyportal_bot.md)."""

from __future__ import annotations

from circex.bot import to_actions
from circex.bot.poster import SkyPortalPoster
from circex.schema import (
    CircularExtraction,
    Event,
    ExtractionMeta,
    Localization,
    PhotometryExt,
    Redshift,
    Span,
)


def _meta() -> ExtractionMeta:
    return ExtractionMeta(extractor="regex-v1")


def test_source_from_event_and_localization() -> None:
    ex = CircularExtraction(
        circular_id=1,
        event=Event(event_name="GRB 260608A"),
        localization=Localization(ra=224.5, dec=28.8),
        extraction_meta=_meta(),
    )
    a = to_actions(ex)
    assert a.source is not None
    assert a.source.to_payload() == {"id": "GRB260608A", "ra": 224.5, "dec": 28.8}


def test_no_source_without_event_name() -> None:
    ex = CircularExtraction(circular_id=1, extraction_meta=_meta())
    assert to_actions(ex).source is None


def test_prefers_optical_at_name_for_obj_id() -> None:
    ex = CircularExtraction(
        circular_id=1,
        event=Event(event_name=["GW170817", "AT2017gfo"]),
        localization=Localization(ra=197.45, dec=-23.38),
        extraction_meta=_meta(),
    )
    source = to_actions(ex).source
    assert source is not None
    assert source.id == "AT2017gfo"


def test_timed_photometry_becomes_a_point() -> None:
    ex = CircularExtraction(
        circular_id=7,
        event=Event(event_name="AT2026xyz"),
        photometry=[
            PhotometryExt(
                filter="r",
                bandpass="sdssr",
                mag=20.4,
                mag_error=0.05,
                mag_system="AB",
                obs_mjd=61199.0,
                telescope="NOT",
            )
        ],
        extraction_meta=_meta(),
    )
    a = to_actions(ex, instrument_map={"NOT": 7})
    assert len(a.photometry) == 1
    p = a.photometry[0].to_payload()
    assert p["mjd"] == 61199.0 and p["filter"] == "sdssr" and p["magsys"] == "ab"
    assert p["mag"] == 20.4 and p["magerr"] == 0.05 and p["instrument_id"] == 7


def test_untimed_photometry_is_not_posted_but_noted() -> None:
    ex = CircularExtraction(
        circular_id=1,
        event=Event(event_name="AT2026xyz"),
        photometry=[PhotometryExt(filter="r", bandpass="sdssr", mag=20.4)],  # no obs_mjd
        extraction_meta=_meta(),
    )
    a = to_actions(ex)
    assert a.photometry == []
    assert a.skipped_rows == 1
    assert any("could not be posted" in c for c in a.comments)


def test_non_detection_maps_to_limiting_mag() -> None:
    ex = CircularExtraction(
        circular_id=1,
        event=Event(event_name="AT2026xyz"),
        photometry=[
            PhotometryExt(
                filter="r", bandpass="sdssr", limiting_mag=22.5, mag_system="AB", obs_mjd=61199.0
            )
        ],
        extraction_meta=_meta(),
    )
    p = to_actions(ex, default_instrument_id=1).photometry[0].to_payload()
    assert p["mag"] is None and p["limiting_mag"] == 22.5


def test_scalar_redshift_becomes_patch_and_comment() -> None:
    ex = CircularExtraction(
        circular_id=1,
        event=Event(event_name="AT2026xyz"),
        redshift=Redshift(redshift=0.5, redshift_error=0.01),
        provenance={"redshift": Span(start=0, end=7, snippet="z = 0.5")},
        extraction_meta=_meta(),
    )
    a = to_actions(ex)
    assert a.redshift == (0.5, 0.01)
    assert any("Redshift z=0.5" in c for c in a.comments)


def test_bound_redshift_is_a_comment_not_a_value() -> None:
    ex = CircularExtraction(
        circular_id=1,
        event=Event(event_name="AT2026xyz"),
        extraction_meta=ExtractionMeta(extractor="regex-v1", notes=["redshift_bound: z <= 1.61"]),
    )
    a = to_actions(ex)
    assert a.redshift is None
    assert any("redshift_bound" in c for c in a.comments)


def test_provenance_lands_in_photometry_altdata() -> None:
    ex = CircularExtraction(
        circular_id=42,
        event=Event(event_name="AT2026xyz"),
        photometry=[PhotometryExt(filter="r", bandpass="sdssr", mag=20.4, obs_mjd=61199.0)],
        provenance={"photometry[0]": Span(start=0, end=5, snippet="r=20.4")},
        extraction_meta=_meta(),
    )
    alt = to_actions(ex, default_instrument_id=1).photometry[0].to_payload()["altdata"]
    assert alt["note"] == 'photometry[0]: "r=20.4"'
    assert alt["circex_circular_id"] == 42


def _photometry_ex(telescope: str | None = None) -> CircularExtraction:
    return CircularExtraction(
        circular_id=1,
        event=Event(event_name="AT2026xyz"),
        photometry=[
            PhotometryExt(
                filter="r", bandpass="sdssr", mag=20.4, obs_mjd=61199.0, telescope=telescope
            )
        ],
        extraction_meta=_meta(),
    )


def test_unmapped_telescope_without_default_is_not_postable() -> None:
    """SkyPortal requires instrument_id; no map entry and no default -> not posted."""
    a = to_actions(_photometry_ex(telescope="VLT"), instrument_map={})
    assert a.photometry == []
    assert a.skipped_rows == 1
    assert any("instrument_id" in c for c in a.comments)


def test_unmapped_telescope_uses_generic_default_and_flags_it() -> None:
    """With a generic GCN instrument id (ICARE's fallback), the row IS postable."""
    a = to_actions(_photometry_ex(telescope="VLT"), instrument_map={}, default_instrument_id=1)
    assert len(a.photometry) == 1
    p = a.photometry[0].to_payload()
    assert p["instrument_id"] == 1
    assert p["altdata"]["instrument_fallback"] is True
    assert p["altdata"]["telescope_as_written"] == "VLT"


def test_mapped_telescope_uses_its_id_no_fallback_flag() -> None:
    a = to_actions(
        _photometry_ex(telescope="VLT"), instrument_map={"VLT": 42}, default_instrument_id=1
    )
    p = a.photometry[0].to_payload()
    assert p["instrument_id"] == 42
    assert "instrument_fallback" not in p["altdata"]


def test_dry_run_poster_sends_nothing_and_plans_in_order() -> None:
    ex = CircularExtraction(
        circular_id=1,
        event=Event(event_name="AT2026xyz"),
        localization=Localization(ra=10.0, dec=20.0),
        photometry=[PhotometryExt(filter="r", bandpass="sdssr", mag=20.4, obs_mjd=61199.0)],
        redshift=Redshift(redshift=0.5),
        extraction_meta=_meta(),
    )
    plan = SkyPortalPoster().post(to_actions(ex, default_instrument_id=1))  # dry-run
    methods = [(r["method"], r["path"]) for r in plan]
    assert methods[0] == ("POST", "/sources")
    assert ("POST", "/photometry") in methods
    assert ("PATCH", "/sources/AT2026xyz") in methods


def test_live_post_requires_token() -> None:
    ex = CircularExtraction(
        circular_id=1, event=Event(event_name="AT2026xyz"), extraction_meta=_meta()
    )
    # live=True but no token -> still dry-run (returns plan, sends nothing)
    plan = SkyPortalPoster(live=True, token=None).post(to_actions(ex))
    assert isinstance(plan, list)


def test_deterministic_band_overrides_llm_mislabel() -> None:
    """The filter crosswalk corrects an LLM that mislabeled the band/magsys.

    Mistral tagged Cousins 'Rc' as sdssr/ab; it is bessellr/vega.
    """
    ex = CircularExtraction(
        circular_id=1,
        event=Event(event_name="AT2026xyz"),
        photometry=[
            PhotometryExt(
                filter="Rc",
                bandpass="sdssr",
                mag_system="AB",
                mag=23.08,
                mag_error=0.18,
                obs_mjd=61199.83,
            )
        ],
        extraction_meta=_meta(),
    )
    p = to_actions(ex, default_instrument_id=1).photometry[0].to_payload()
    assert p["filter"] == "bessellr" and p["magsys"] == "vega"


def test_unrecognized_filter_falls_back_to_row_bandpass() -> None:
    """A filter the crosswalk doesn't know keeps the extractor's bandpass."""
    ex = CircularExtraction(
        circular_id=1,
        event=Event(event_name="AT2026xyz"),
        photometry=[
            PhotometryExt(
                filter="Zband", bandpass="ztfg", mag_system="AB", mag=20.0, obs_mjd=61199.83
            )
        ],
        extraction_meta=_meta(),
    )
    p = to_actions(ex, default_instrument_id=1).photometry[0].to_payload()
    assert p["filter"] == "ztfg"


def test_no_positionless_source_create() -> None:
    """A named event with no RA/Dec must NOT emit a source-create (SkyPortal 400s)."""
    ex = CircularExtraction(
        circular_id=44877,
        event=Event(event_name="GRB 260604C"),
        extraction_meta=_meta(),
    )
    a = to_actions(ex)
    assert a.source is None
    assert any("not created" in c and "RA/Dec" in c for c in a.comments)


def test_detection_without_limit_backfills_limiting_mag() -> None:
    """SkyPortal requires a non-null limiting_mag; fall back to the detection mag, flagged."""
    ex = CircularExtraction(
        circular_id=44834,
        event=Event(event_name="GRB 260604C"),
        localization=Localization(ra=224.4566, dec=28.8175),
        photometry=[PhotometryExt(filter="g", mag=19.69, mag_error=0.04, obs_mjd=61200.15)],
        extraction_meta=_meta(),
    )
    a = to_actions(ex, default_instrument_id=4)
    assert len(a.photometry) == 1
    p = a.photometry[0].to_payload()
    assert p["limiting_mag"] == 19.69
    assert p["mag"] == 19.69
    assert a.photometry[0].altdata.get("limiting_mag_assumed") is True


def test_row_with_neither_mag_nor_limit_is_unpostable() -> None:
    ex = CircularExtraction(
        circular_id=1,
        event=Event(event_name="GRB 260604C"),
        localization=Localization(ra=1.0, dec=2.0),
        photometry=[PhotometryExt(filter="g", obs_mjd=61200.0)],
        extraction_meta=_meta(),
    )
    a = to_actions(ex, default_instrument_id=4)
    assert a.photometry == []
    assert a.skipped_rows == 1


def test_poster_continue_on_error(monkeypatch) -> None:
    """continue_on_error tolerates a failed POST (unattended); default re-raises."""
    import pytest
    import requests

    from circex.bot.poster import SkyPortalPoster
    from circex.bot.skyportal_map import SkyPortalActions, SourceUpsert

    class _Resp:
        status_code = 400

        def raise_for_status(self) -> None:
            raise requests.HTTPError("400 Bad Request")

        def json(self) -> dict:
            return {}

    monkeypatch.setattr(requests, "request", lambda *a, **k: _Resp())
    actions = SkyPortalActions(
        source=SourceUpsert(id="X", ra=1.0, dec=2.0, group_ids=[1]),
        photometry=[],
        redshift=None,
        comments=[],
        skipped_rows=0,
    )
    # tolerant: logs and continues, no raise
    SkyPortalPoster(token="t", live=True, continue_on_error=True).post(actions)
    # strict (default): propagates
    with pytest.raises(requests.HTTPError):
        SkyPortalPoster(token="t", live=True).post(actions)


def test_an_error_region_localizes_the_event_but_makes_no_source():
    """An IPN error-box centre is not a transient; writing one would invent a source."""
    ex = CircularExtraction(
        circular_id=21517,
        event=Event(event_name="GRB 170816A"),
        localization=Localization(ra=350.841, dec=12.853, ra_dec_error=[1.225, 0.158]),
        extraction_meta=_meta(),
    )
    actions = to_actions(ex, default_instrument_id=4)
    assert actions.source is None
    assert any("error region" in c for c in actions.comments)


def test_a_counterpart_position_still_makes_a_source():
    ex = CircularExtraction(
        circular_id=1,
        event=Event(event_name="GRB 230307A"),
        localization=Localization(ra=45.12, dec=-75.38, ra_dec_error=0.0003),
        extraction_meta=_meta(),
    )
    actions = to_actions(ex, default_instrument_id=4)
    assert actions.source is not None
    assert actions.source.ra == 45.12


def test_a_candidate_table_becomes_one_source_per_object():
    """GCN 45552: two GOTO candidates, each its own source at its own position."""
    from circex.bot.skyportal_map import to_actions
    from circex.schema import CircularExtraction, Event, ExtractionMeta, PhotometryExt

    extraction = CircularExtraction(
        circular_id=45552,
        event=Event(event_name="GRB 260910B"),
        photometry=[
            PhotometryExt(
                object_name="GOTO26jjj",
                ra=264.309139,
                dec=10.532956,
                filter="L",
                mag=20.28,
                mag_error=0.18,
            ),
            PhotometryExt(
                object_name="AT 2026abfp",
                object_aliases=["GOTO26jjg"],
                ra=261.677596,
                dec=2.050792,
                filter="L",
                mag=20.58,
                mag_error=0.19,
            ),
        ],
        extraction_meta=ExtractionMeta(extractor="test", latency_ms=0.0),
    )
    actions = to_actions(extraction, default_instrument_id=1, group_ids=[3])
    assert [(c.id, c.ra) for c in actions.candidate_sources] == [
        ("GOTO26jjj", 264.309139),
        ("AT2026abfp", 261.677596),
    ]
    assert all(c.group_ids == [3] for c in actions.candidate_sources)


def test_a_candidate_without_a_position_is_not_made_a_source():
    """SkyPortal cannot create a source with no RA/Dec."""
    from circex.bot.skyportal_map import to_actions
    from circex.schema import CircularExtraction, ExtractionMeta, PhotometryExt

    extraction = CircularExtraction(
        circular_id=1,
        photometry=[PhotometryExt(object_name="AT 2026zzz", filter="r", mag=19.0)],
        extraction_meta=ExtractionMeta(extractor="test", latency_ms=0.0),
    )
    assert to_actions(extraction, default_instrument_id=1).candidate_sources == ()


def test_gotos_wide_l_maps_to_its_own_bandpass():
    """L means gotol on GOTO, and nothing at all from another telescope."""
    from circex.bot.skyportal_map import _effective_band
    from circex.schema import PhotometryExt

    goto = PhotometryExt(filter="L", mag=20.28, telescope="GOTO")
    assert _effective_band(goto) == ("gotol", "ab")

    # LCO has no L filter, so the same letter stays unmapped rather than
    # borrowing GOTO's band.
    elsewhere = PhotometryExt(filter="L", mag=20.28, telescope="LCO")
    assert _effective_band(elsewhere)[0] is None
