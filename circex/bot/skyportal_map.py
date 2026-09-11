"""Map a CircularExtraction to SkyPortal API writes.

Pure functions, no network. `to_actions()` returns a SkyPortalActions bundle —
a source upsert, photometry points, an optional redshift patch, and comments —
shaped as the dicts SkyPortal's REST API expects. The poster turns these into
HTTP calls (or prints them in dry-run). See docs/design_skyportal_bot.md.

Everything the ICARE work added feeds in here: obs_mjd -> mjd, bandpass ->
filter, telescope_canonical -> instrument_id, is_detection, redshift bounds ->
comment, provenance -> altdata.note.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import structlog

from circex.extract.regex.mag_table import (
    infer_bandpass,
    infer_mag_system,
    normalize_filter,
)
from circex.extract.regex.radio import bandpass_for_frequency, to_ujy
from circex.schema import CircularExtraction, PhotometryExt

# mag_system (our enum) -> SkyPortal magsys (lowercase). STMag has no direct
# SkyPortal equivalent; map to the closest and flag it in a comment upstream.
log = structlog.get_logger(__name__)

_MAGSYS = {"AB": "ab", "Vega": "vega", "STMag": "ab"}


# A position whose stated uncertainty is coarser than this describes a region,
# not a transient. An IPN error-box centre is somewhere to put a localization,
# never somewhere to put a source.
MAX_SOURCE_POSITION_ERROR_DEG = 0.02


def position_error_deg(extraction: CircularExtraction) -> float | None:
    """Stated semi-major uncertainty on the position [deg], if the circular gives one."""
    loc = extraction.localization
    error = loc.ra_dec_error if loc is not None else None
    if isinstance(error, list):
        error = next((e for e in error if isinstance(e, int | float)), None)
    return float(error) if isinstance(error, int | float) and error > 0 else None


def _obj_id(extraction: CircularExtraction) -> str | None:
    """SkyPortal source id from the event name (spaces removed). None if unnamed.

    Prefers an AT/optical designation over a bare GRB/GW trigger when the
    extraction carries a list (the optical name is what SkyPortal keys on).
    """
    if extraction.event is None or extraction.event.event_name is None:
        return None
    name = extraction.event.event_name
    names = name if isinstance(name, list) else [name]
    optical = [n for n in names if re.match(r"(?i)^(AT|SN)\s?\d", n)]
    chosen = optical[0] if optical else (names[0] if names else None)
    return re.sub(r"\s+", "", chosen) if chosen else None


def _as_source_id(name: str | None) -> str | None:
    """A designation as SkyPortal keys it: "AT 2026abfp" -> "AT2026abfp"."""
    if not name:
        return None
    return re.sub(r"\s+", "", name.strip()) or None


@dataclass(frozen=True)
class SourceUpsert:
    """`POST /api/sources` payload."""

    id: str
    ra: float | None
    dec: float | None
    group_ids: list[int] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id}
        if self.ra is not None:
            out["ra"] = self.ra
        if self.dec is not None:
            out["dec"] = self.dec
        if self.group_ids:
            out["group_ids"] = self.group_ids
        return out


@dataclass(frozen=True)
class PhotometryPoint:
    """`POST /api/photometry` payload for one row."""

    obj_id: str
    mjd: float
    filter: str
    magsys: str
    instrument_id: int | None
    mag: float | None
    magerr: float | None
    limiting_mag: float | None
    altdata: dict[str, Any] = field(default_factory=dict)
    # Flux space, for radio rows. `flux` is null for a non-detection, where
    # SkyPortal derives the limiting magnitude from fluxerr and the sigma level.
    flux: float | None = None
    fluxerr: float | None = None
    zp: float | None = None

    @property
    def is_flux_space(self) -> bool:
        return self.zp is not None

    def to_payload(self) -> dict[str, Any]:
        if self.is_flux_space:
            out: dict[str, Any] = {
                "obj_id": self.obj_id,
                "mjd": self.mjd,
                "filter": self.filter,
                "magsys": self.magsys,
                "flux": self.flux,
                "fluxerr": self.fluxerr,
                "zp": self.zp,
            }
            if self.instrument_id is not None:
                out["instrument_id"] = self.instrument_id
            if self.altdata:
                out["altdata"] = self.altdata
            return out
        out = {
            "obj_id": self.obj_id,
            "mjd": self.mjd,
            "filter": self.filter,
            "magsys": self.magsys,
            "mag": self.mag,
            "magerr": self.magerr,
            "limiting_mag": self.limiting_mag,
        }
        if self.instrument_id is not None:
            out["instrument_id"] = self.instrument_id
        if self.altdata:
            out["altdata"] = self.altdata
        return out


@dataclass(frozen=True)
class SkyPortalActions:
    """Everything to post for one circular."""

    source: SourceUpsert | None
    photometry: list[PhotometryPoint]
    redshift: tuple[float, float | None] | None  # (z, z_err)
    comments: list[str]
    skipped_rows: int  # photometry rows we could not post (no mjd/filter)
    # Why each was dropped, so a reader can tell a thin light curve from lost data.
    skipped_reasons: tuple[str, ...] = ()
    # A circular announcing several counterpart candidates gives each its own
    # designation and position, so each becomes a source in its own right rather
    # than collapsing onto the event's.
    candidate_sources: tuple[SourceUpsert, ...] = ()
    # The extractions these actions were built from. Callers that need fields with
    # no place in the SkyPortal write bundle (event designations, classification)
    # would otherwise have to run the extractor a second time.
    extractions: tuple[CircularExtraction, ...] = ()


def _provenance_note(extraction: CircularExtraction, path: str) -> str | None:
    span = extraction.provenance.get(path)
    return f'{path}: "{span.snippet}"' if span is not None else None


def to_actions(
    extraction: CircularExtraction,
    *,
    instrument_map: dict[str, int] | None = None,
    bandpass_instrument_map: dict[str, int] | None = None,
    default_instrument_id: int | None = None,
    group_ids: list[int] | None = None,
) -> SkyPortalActions:
    """Build the SkyPortal write bundle for one extraction.

    `bandpass_instrument_map` maps a bandpass -> SkyPortal instrument_id and
    is tried first: a bandpass names one instrument exactly, where a telescope
    may carry several (EP has both WXT and FXT), and the radio and X-ray rows
    carry a bandpass but no telescope.
    `instrument_map` maps `telescope_canonical` -> SkyPortal instrument_id
    (ICARE's table). `default_instrument_id` is the generic GCN instrument used
    when a telescope isn't in the map (matching ICARE's fall-back). SkyPortal's
    photometry endpoint REQUIRES an instrument_id, so a row that resolves to
    neither a mapped nor a default id cannot be posted — it becomes a comment.
    """
    instrument_map = instrument_map or {}
    bandpass_instrument_map = bandpass_instrument_map or {}
    group_ids = group_ids or []

    obj_id = _obj_id(extraction)
    loc = extraction.localization
    ra = loc.ra if loc is not None else None
    dec = loc.dec if loc is not None else None

    photometry: list[PhotometryPoint] = []
    comments: list[str] = []
    dropped: list[str] = []

    # A NEW SkyPortal source requires ra/dec. A follow-up circular usually has no
    # position (it lives in the discovery circular), so guard against emitting a
    # positionless source-create that SkyPortal would reject with a 400.
    source: SourceUpsert | None = None
    error_deg = position_error_deg(extraction)
    if extraction.retraction:
        comments.append(
            "This circular withdraws the trigger; no source created and no photometry posted."
        )
        dropped.append("retraction")
    elif error_deg is not None and error_deg > MAX_SOURCE_POSITION_ERROR_DEG:
        # A region, not a counterpart; it still localizes the event.
        comments.append(
            f"Position is an error region ({error_deg:.3f} deg), not a counterpart; "
            f"no source created."
        )
    elif obj_id is not None and ra is not None and dec is not None:
        source = SourceUpsert(id=obj_id, ra=ra, dec=dec, group_ids=group_ids)
    elif obj_id is not None:
        comments.append(
            f"Source {obj_id} not created: no RA/Dec in this circular "
            f"(position comes from the discovery circular)."
        )

    # One source per candidate the circular tabulates, keyed on its designation.
    # Only a row carrying its own position can become a source: without one
    # SkyPortal has nothing to create.
    candidates: dict[str, SourceUpsert] = {}
    for row in extraction.photometry if not extraction.retraction else []:
        cid = _as_source_id(row.object_name)
        if cid is None or row.ra is None or row.dec is None or cid in candidates:
            continue
        candidates[cid] = SourceUpsert(id=cid, ra=row.ra, dec=row.dec, group_ids=group_ids)

    for idx, row in enumerate(extraction.photometry if not extraction.retraction else []):
        point = _row_to_point(
            extraction,
            _as_source_id(row.object_name) or obj_id,
            idx,
            row,
            instrument_map,
            bandpass_instrument_map,
            default_instrument_id,
            dropped,
        )
        if point is not None:
            photometry.append(point)

    # Redshift: scalar only. Bounds (in notes) become a comment, not a value.
    redshift: tuple[float, float | None] | None = None
    if extraction.redshift is not None and extraction.redshift.redshift is not None:
        err = extraction.redshift.redshift_error
        z_err = err if isinstance(err, float) else None
        redshift = (extraction.redshift.redshift, z_err)
        if (note := _provenance_note(extraction, "redshift")) is not None:
            comments.append(f"Redshift z={extraction.redshift.redshift} from {note}")

    # Bound redshifts and other extractor notes -> comment.
    for note in extraction.extraction_meta.notes:
        comments.append(f"Note (not posted as a value): {note}")

    if dropped:
        detail = ", ".join(f"{n}x {reason}" for reason, n in sorted(Counter(dropped).items()))
        comments.append(
            f"{len(dropped)} photometry row(s) could not be posted ({detail}); "
            f"kept in the extraction only."
        )

    return SkyPortalActions(
        source=source,
        candidate_sources=tuple(candidates.values()),
        photometry=photometry,
        redshift=redshift,
        comments=comments,
        skipped_rows=len(dropped),
        skipped_reasons=tuple(dropped),
        extractions=(extraction,),
    )


def _effective_band(row: PhotometryExt) -> tuple[str | None, str]:
    """Canonical (bandpass, magsys) for a row.

    The deterministic filter crosswalk is authoritative for recognized filters
    and OVERRIDES the extractor's values — an LLM may mislabel a band (e.g.
    Mistral tagging Cousins "Rc" as sdssr/ab when it is bessellr/vega). Falls
    back to the row's own bandpass/mag_system when the filter isn't recognized.
    """
    if row.energy_band_kev is not None:
        return row.bandpass, "ab"
    if row.frequency_ghz is not None:
        return (row.bandpass or bandpass_for_frequency(row.frequency_ghz)), "ab"
    base = normalize_filter(row.filter) if row.filter else None
    band = infer_bandpass(base) if base else None
    if band is not None:
        system = infer_mag_system(base) if base else None
        return band, _MAGSYS.get(system or "", "ab")
    return row.bandpass, _MAGSYS.get(row.mag_system or "", "ab")


def _row_to_point(
    extraction: CircularExtraction,
    obj_id: str | None,
    idx: int,
    row: PhotometryExt,
    instrument_map: dict[str, int],
    bandpass_instrument_map: dict[str, int],
    default_instrument_id: int | None,
    dropped: list[str],
) -> PhotometryPoint | None:
    """One photometry row -> a SkyPortal point, or None if unpostable.

    A row needs an obj_id, an obs_mjd, a resolvable bandpass, AND an
    instrument_id (mapped, or the generic default) — SkyPortal requires all of
    these. Otherwise it cannot be posted.
    """
    band, magsys = _effective_band(row)
    # A bandpass names one instrument exactly; a telescope may carry several.
    mapped = bandpass_instrument_map.get(band) if band else None
    if mapped is None and row.telescope_canonical:
        mapped = instrument_map.get(row.telescope_canonical)
    instrument_id = mapped if mapped is not None else default_instrument_id
    if row.energy_band_kev is not None:
        return _xray_row_to_point(
            extraction, obj_id, idx, row, band, instrument_id, mapped is None, dropped
        )
    if row.frequency_ghz is not None:
        return _radio_row_to_point(
            extraction, obj_id, idx, row, band, instrument_id, mapped is None, dropped
        )
    if obj_id is None or row.obs_mjd is None or band is None or instrument_id is None:
        # Name the reason: a filter with no bandpass is a crosswalk gap and the
        # row is lost silently otherwise, which reads as a thin light curve
        # rather than as missing data.
        reason = (
            "no bandpass"
            if band is None
            else "no obs_mjd"
            if row.obs_mjd is None
            else "no obj_id"
            if obj_id is None
            else "no instrument_id"
        )
        log.info(
            "photometry_row_dropped",
            circular_id=extraction.circular_id,
            reason=reason,
            filter=row.filter,
            telescope=row.telescope,
        )
        dropped.append(reason)
        return None

    # SkyPortal's photometry endpoint REQUIRES a non-null limiting_mag for
    # mag-space points. Circulars often report a detection with no explicit
    # per-point limit, so fall back to the detection mag itself — a conservative,
    # truthful depth floor (the field was seen at least this faint) — and flag it.
    limiting_mag = row.limiting_mag
    limit_assumed = False
    if limiting_mag is None:
        if row.mag is None:
            # Neither a detection nor a stated limit — nothing to post.
            log.info(
                "photometry_row_dropped",
                circular_id=extraction.circular_id,
                reason="no mag or limit",
                filter=row.filter,
                telescope=row.telescope,
            )
            dropped.append("no mag or limit")
            return None
        limiting_mag = row.mag
        limit_assumed = True

    altdata: dict[str, Any] = {}
    if (note := _provenance_note(extraction, f"photometry[{idx}]")) is not None:
        altdata["note"] = note
    altdata["circex_circular_id"] = extraction.circular_id
    if mapped is None:
        # Fell back to the generic instrument; record what we actually saw.
        altdata["instrument_fallback"] = True
        if row.telescope:
            altdata["telescope_as_written"] = row.telescope
    if limit_assumed:
        altdata["limiting_mag_assumed"] = True
    return PhotometryPoint(
        obj_id=obj_id,
        mjd=row.obs_mjd,
        filter=band,
        magsys=magsys,
        instrument_id=instrument_id,
        mag=row.mag,
        magerr=row.mag_error,
        limiting_mag=limiting_mag,
        altdata=altdata,
    )


# AB zeropoint for a flux density in microjanskys: m = -2.5 log10(f_uJy) + 23.9.
_UJY_AB_ZP = 23.9

# Roughly half of radio detections quote no uncertainty, and in the archive every
# one of those is written as "~0.4 mJy" — an approximate quick-look value. Rather
# than drop them, assume this fraction of the flux, which is the right order for
# cm-wave absolute flux calibration. Rows carrying it are flagged in altdata so a
# consumer can tell an assumed error from a reported one.
ASSUMED_FLUX_ERROR_FRACTION = 0.2


def _radio_row_to_point(
    extraction: CircularExtraction,
    obj_id: str | None,
    idx: int,
    row: PhotometryExt,
    band: str | None,
    instrument_id: int | None,
    instrument_fallback: bool,
    dropped: list[str],
) -> PhotometryPoint | None:
    """A radio row as a flux-space point, or None if unpostable.

    SkyPortal requires a non-null fluxerr, so a detection quoted without an
    uncertainty cannot be posted; inventing one would be worse than dropping it.
    """

    def drop(reason: str) -> PhotometryPoint | None:
        log.info(
            "photometry_row_dropped",
            circular_id=extraction.circular_id,
            reason=reason,
            frequency_ghz=row.frequency_ghz,
            telescope=row.telescope,
        )
        dropped.append(reason)
        return None

    unit = row.flux_density_unit
    if obj_id is None:
        return drop("no obj_id")
    if row.obs_mjd is None:
        return drop("no obs_mjd")
    if band is None:
        return drop("no bandpass")
    if instrument_id is None:
        return drop("no instrument_id")
    if unit is None:
        return drop("no flux unit")

    flux: float | None = None
    error_assumed = False
    if row.flux_density is not None:
        flux = to_ujy(row.flux_density, unit)
        if row.flux_density_error is not None:
            fluxerr = to_ujy(row.flux_density_error, unit)
        else:
            fluxerr = abs(flux) * ASSUMED_FLUX_ERROR_FRACTION
            error_assumed = True
            if fluxerr == 0.0:
                return drop("zero flux with no uncertainty")
    elif row.limiting_flux_density is not None:
        sigma = row.limiting_mag_sigma or 3.0
        fluxerr = to_ujy(row.limiting_flux_density, unit) / sigma
    else:
        return drop("no flux density")

    altdata: dict[str, Any] = {"circex_circular_id": extraction.circular_id}
    if (note := _provenance_note(extraction, f"photometry[{idx}]")) is not None:
        altdata["note"] = note
    altdata["frequency_ghz"] = row.frequency_ghz
    if error_assumed:
        altdata["flux_density_error_assumed"] = True
        altdata["flux_density_error_fraction"] = ASSUMED_FLUX_ERROR_FRACTION
    if row.limiting_flux_density is not None:
        altdata["limiting_flux_density_sigma"] = row.limiting_mag_sigma or 3.0
    if instrument_fallback:
        altdata["instrument_fallback"] = True
        if row.telescope:
            altdata["telescope_as_written"] = row.telescope
    return PhotometryPoint(
        obj_id=obj_id,
        mjd=row.obs_mjd,
        filter=band,
        magsys="ab",
        instrument_id=instrument_id,
        mag=None,
        magerr=None,
        limiting_mag=None,
        flux=flux,
        fluxerr=fluxerr,
        zp=_UJY_AB_ZP,
        altdata=altdata,
    )


# Planck's constant in keV s, for converting an energy band to a frequency width.
_PLANCK_KEV_S = 4.135667696e-18
# 1 microjansky in erg cm^-2 s^-1 Hz^-1.
_UJY_CGS = 1.0e-29


def energy_flux_to_ujy(flux: float, band_kev: list[float]) -> float | None:
    """Band-integrated energy flux to a band-averaged flux density in microjansky.

    Dividing by the width of the band in frequency assumes the spectrum is flat
    across it, which is the usual convention for placing an X-ray point on a
    broadband SED and is what makes the value comparable with the optical and
    radio rows beside it.
    """
    if len(band_kev) != 2:
        return None
    lo, hi = band_kev
    if not hi > lo > 0:
        return None
    delta_nu = (hi - lo) / _PLANCK_KEV_S
    return flux / delta_nu / _UJY_CGS


def _xray_row_to_point(
    extraction: CircularExtraction,
    obj_id: str | None,
    idx: int,
    row: PhotometryExt,
    band: str | None,
    instrument_id: int | None,
    instrument_fallback: bool,
    dropped: list[str],
) -> PhotometryPoint | None:
    """An X-ray row as a flux-space point, or None if unpostable."""

    def drop(reason: str) -> PhotometryPoint | None:
        log.info(
            "photometry_row_dropped",
            circular_id=extraction.circular_id,
            reason=reason,
            energy_band_kev=row.energy_band_kev,
            telescope=row.telescope,
        )
        dropped.append(reason)
        return None

    if obj_id is None:
        return drop("no obj_id")
    if row.obs_mjd is None:
        return drop("no obs_mjd")
    if band is None:
        return drop("no bandpass")
    if instrument_id is None:
        return drop("no instrument_id")
    band_kev = row.energy_band_kev or []

    flux: float | None = None
    error_assumed = False
    if row.energy_flux is not None:
        flux = energy_flux_to_ujy(row.energy_flux, band_kev)
        if flux is None:
            return drop("unusable energy band")
        if row.energy_flux_error is not None:
            fluxerr = energy_flux_to_ujy(row.energy_flux_error, band_kev) or 0.0
        else:
            fluxerr = abs(flux) * ASSUMED_FLUX_ERROR_FRACTION
            error_assumed = True
    elif row.limiting_energy_flux is not None:
        limit = energy_flux_to_ujy(row.limiting_energy_flux, band_kev)
        if limit is None:
            return drop("unusable energy band")
        fluxerr = limit / (row.limiting_mag_sigma or 3.0)
    else:
        return drop("no energy flux")
    if fluxerr <= 0:
        return drop("no flux uncertainty")

    altdata: dict[str, Any] = {"circex_circular_id": extraction.circular_id}
    if (note := _provenance_note(extraction, f"photometry[{idx}]")) is not None:
        altdata["note"] = note
    altdata["energy_band_kev"] = band_kev
    altdata["energy_flux_cgs"] = row.energy_flux or row.limiting_energy_flux
    if row.limiting_energy_flux is not None:
        altdata["limiting_energy_flux_sigma"] = row.limiting_mag_sigma or 3.0
    if error_assumed:
        altdata["flux_density_error_assumed"] = True
        altdata["flux_density_error_fraction"] = ASSUMED_FLUX_ERROR_FRACTION
    if instrument_fallback:
        altdata["instrument_fallback"] = True
        if row.telescope:
            altdata["telescope_as_written"] = row.telescope
    return PhotometryPoint(
        obj_id=obj_id,
        mjd=row.obs_mjd,
        filter=band,
        magsys="ab",
        instrument_id=instrument_id,
        mag=None,
        magerr=None,
        limiting_mag=None,
        flux=flux,
        fluxerr=fluxerr,
        zp=_UJY_AB_ZP,
        altdata=altdata,
    )
