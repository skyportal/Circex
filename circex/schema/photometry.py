"""Extended Photometry schema.

Base mirrors gcn-schema gcn/notices/core/Photometry.schema.json. Sprint 1 adds:
telescope, instrument, calibration_reference, galactic_extinction_corrected, seeing,
airmass; tightens mag_system to enum [AB, Vega, STMag] (BREAKING for the upstream PR).

These additions are the optical-specific extensions the plan calls for.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

MagSystem = Literal["AB", "Vega", "STMag"]
FluxDensityUnit = Literal["uJy", "mJy", "Jy"]
CalibrationReference = Literal["PS1", "SDSS", "APASS", "2MASS", "Gaia", "Other"]


class PhotometryExt(BaseModel):
    """Magnitude-based UV/optical/NIR photometry for one source in one filter (one row)."""

    filter: str | None = Field(
        default=None,
        description="Filter used for the observation (e.g., u, g, r, R, clear).",
    )
    bandpass: str | None = Field(
        default=None,
        description=(
            "Canonical bandpass name for downstream consumers (sncosmo/SkyPortal "
            "vocabulary), derived from `filter` + `mag_system` where possible. "
            "Sloan u/g/r/i/z -> sdss{u,g,r,i,z}; y -> ps1::y; "
            "Bessel U/B/V/R/I -> bessell{u,b,v,r,i}; "
            "NIR J/H/K/Ks -> 2mass{j,h,ks}. Null for unfiltered/clear or when the "
            "filter cannot be mapped (the raw `filter` string is always retained)."
        ),
    )
    mag: float | None = Field(
        default=None,
        description="Measured apparent magnitude [mag] of the source in the specified filter.",
    )
    mag_error: float | None = Field(
        default=None,
        description="1-sigma statistical uncertainty on magnitude [mag].",
    )
    mag_system: MagSystem | None = Field(
        default=None,
        description=(
            "Photometric magnitude system (zeropoint convention) for mag and limiting_mag. "
            "One of [AB, Vega, STMag]. Tightened from upstream open-string default."
        ),
    )
    limiting_mag: float | None = Field(
        default=None,
        description="Limiting magnitude (upper limit) for the observation [mag].",
    )
    limiting_mag_sigma: float | None = Field(
        default=None,
        description="Significance level [sigma] associated with limiting_mag (default 5).",
    )

    # ---- radio: flux density rather than magnitude ----
    flux_density: float | None = Field(
        default=None,
        description=(
            "Measured flux density in `flux_density_unit`, for radio/mm rows that "
            "report a flux rather than a magnitude. May be negative for a marginal "
            "measurement; null for a non-detection."
        ),
    )
    flux_density_error: float | None = Field(
        default=None,
        description="1-sigma statistical uncertainty on flux_density, same unit.",
    )
    limiting_flux_density: float | None = Field(
        default=None,
        description=(
            "Upper limit on flux density, same unit, at `limiting_mag_sigma` sigma "
            "(circulars almost always quote 3-sigma)."
        ),
    )
    flux_density_unit: FluxDensityUnit | None = Field(
        default=None,
        description="Unit for the flux-density fields. One of [uJy, mJy, Jy].",
    )
    frequency_ghz: float | None = Field(
        default=None,
        description=(
            "Observing frequency in GHz. What identifies the band for radio rows, "
            "where `filter` is meaningless; crosswalked to a radio-* bandpass."
        ),
    )

    # ---- X-ray/gamma-ray: energy flux over a stated band ----
    energy_flux: float | None = Field(
        default=None,
        description=(
            "Band-integrated energy flux in erg cm^-2 s^-1, for X-ray and "
            "gamma-ray rows. Distinct from a fluence (erg cm^-2) and from a "
            "luminosity (erg s^-1), neither of which is photometry."
        ),
    )
    energy_flux_error: float | None = Field(
        default=None,
        description="1-sigma uncertainty on energy_flux, erg cm^-2 s^-1.",
    )
    limiting_energy_flux: float | None = Field(
        default=None,
        description=(
            "Upper limit on the energy flux, erg cm^-2 s^-1, at `limiting_mag_sigma` sigma."
        ),
    )
    energy_band_kev: list[float] | None = Field(
        default=None,
        description=(
            "Energy band the flux is integrated over, [low, high] in keV. What "
            "identifies the band for X-ray rows, where `filter` is meaningless."
        ),
    )

    # ---- per-row observation epoch (ICARE P0 #2) ----
    obs_mjd: float | None = Field(
        default=None,
        description=(
            "Observation epoch as Modified Julian Date (UTC). Resolved from an "
            "absolute UT/MJD stated in the row, else from the caller-supplied "
            "trigger time plus the circular's relative offset. Null when neither "
            "is available. SkyPortal consumes this directly as the point's mjd."
        ),
    )
    obs_time: str | None = Field(
        default=None,
        description="Same epoch as obs_mjd in ISO-8601 UTC (human-legible mirror).",
    )

    # ---- optical-specific extensions added by Circex ----
    object_name: str | None = Field(
        default=None,
        description=(
            "Designation of the object this row measures, for a circular that "
            "reports several candidates in one table. None when the circular "
            "concerns a single target named elsewhere in the extraction."
        ),
    )
    object_aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Other designations the circular gives this object, such as a "
            "survey's internal name alongside the IAU one."
        ),
    )
    ra: float | None = Field(
        default=None,
        description="Right ascension of this row's object, J2000 degrees.",
    )
    dec: float | None = Field(
        default=None,
        description="Declination of this row's object, J2000 degrees.",
    )
    telescope: str | None = Field(
        default=None, description="Name of the telescope (e.g., GTC, ZTF, Pan-STARRS1)."
    )
    telescope_canonical: str | None = Field(
        default=None,
        description=(
            "Canonical telescope name from the alias map (e.g. 'the VLT' -> 'VLT'). "
            "Auto-derived from `telescope`; null when the name isn't in the seed map "
            "(the raw `telescope` string is always retained). Extend the map from "
            "ICARE's instrument_id table."
        ),
    )
    instrument: str | None = Field(
        default=None, description="Name of the instrument (e.g., OSIRIS, ZTF Camera)."
    )
    instrument_canonical: str | None = Field(
        default=None,
        description=(
            "Canonical instrument name from the alias map. Auto-derived from "
            "`instrument`; null when not in the seed map (raw `instrument` retained)."
        ),
    )
    calibration_reference: CalibrationReference | None = Field(
        default=None,
        description=(
            "Photometric calibration reference catalog. One of "
            "[PS1, SDSS, APASS, 2MASS, Gaia, Other]."
        ),
    )
    galactic_extinction_corrected: bool | None = Field(
        default=None,
        description="True if the reported magnitude has been corrected for Galactic extinction.",
    )
    seeing: float | None = Field(
        default=None,
        description="Seeing FWHM during the observation [arcsec].",
    )
    airmass: float | None = Field(
        default=None,
        description="Airmass during the observation [dimensionless].",
    )

    # ---- detection / non-detection flag ----
    is_detection: bool | None = Field(
        default=None,
        description=(
            "True if this row reports a measured magnitude; False if it reports only an "
            "upper limit (non-detection). Inferred automatically when not set explicitly: "
            "mag present ⇒ True, only limiting_mag present ⇒ False. When both are present "
            "(a detection plus the night's depth), this is True. When both are null the "
            "value is left null."
        ),
    )

    @model_validator(mode="after")
    def _infer_is_detection(self) -> PhotometryExt:
        """Auto-set is_detection when the caller leaves it null."""
        if self.is_detection is None:
            if (
                self.mag is not None
                or self.flux_density is not None
                or self.energy_flux is not None
            ):
                self.is_detection = True
            elif (
                self.limiting_mag is not None
                or self.limiting_flux_density is not None
                or self.limiting_energy_flux is not None
            ):
                self.is_detection = False
        return self

    @model_validator(mode="after")
    def _fill_obs_pair(self) -> PhotometryExt:
        """Backfill the missing half of (obs_mjd, obs_time) from the other.

        The regex path sets both; an LLM may set only obs_time. When exactly one
        is present, derive the other so SkyPortal always gets obs_mjd.
        """
        if (self.obs_mjd is None) != (self.obs_time is None):
            from circex.extract.timing import normalize_pair

            pair = normalize_pair(self.obs_mjd, self.obs_time)
            if pair is not None:
                self.obs_mjd, self.obs_time = pair
        return self

    @model_validator(mode="after")
    def _fill_canonical_names(self) -> PhotometryExt:
        """Derive telescope_canonical / instrument_canonical from the raw names.

        Always re-derives (a pure function of the raw name + alias map), so an
        LLM-supplied canonical can't drift from the map; round-trips are stable.
        """
        # Local import avoids a schema->data import cycle at module load.
        from circex.data.telescopes import canonicalize_instrument, canonicalize_telescope

        self.telescope_canonical = canonicalize_telescope(self.telescope)
        self.instrument_canonical = canonicalize_instrument(self.instrument)
        return self
