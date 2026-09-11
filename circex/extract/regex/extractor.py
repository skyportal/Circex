"""RegexExtractor — composes the regex sub-extractors into a CircularExtraction.

This is the baseline the LLM extractor must beat on per-field P/R/F1. Per the PDF,
regex is EXPECTED to fail visibly on multi-row mag tables and in-prose
classifications — measure the gap, do not over-engineer.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Final

from circex.data.telescopes import canonicalize_telescope
from circex.extract.protocol import Circular, Extractor
from circex.extract.regex.classification import (
    parse_classification_with_span,
    parse_stellar_flare,
    parse_xrf_subtype,
)
from circex.extract.regex.coords import (
    parse_coords_with_span,
    parse_ipn_error_box_with_span,
)
from circex.extract.regex.dates import parse_time_offsets_with_spans
from circex.extract.regex.mag_table import (
    parse_fixed_width_table_with_spans,
    parse_mag_table_with_spans,
    parse_pipe_candidate_with_span,
    parse_pipe_table_with_spans,
    parse_single_mags_with_spans,
    stated_mag_system,
)
from circex.extract.regex.radio import parse_radio_with_spans
from circex.extract.regex.redshift import (
    parse_redshift_bound,
    parse_redshift_with_span,
)
from circex.extract.regex.regex_events import (
    extract_events,
    extract_gcn_xrefs_with_positions,
    extract_matches_with_positions,
)
from circex.extract.regex.retraction import is_retraction
from circex.extract.regex.telescope import parse_telescope_with_span
from circex.extract.regex.xray import parse_xray_with_spans
from circex.extract.timing import (
    resolve_inline_offsets,
    resolve_object_epochs,
    resolve_observation_epoch,
    resolve_relative_epochs,
)
from circex.schema import (
    CircularExtraction,
    Classification,
    Event,
    ExtractionMeta,
    FollowUp,
    Localization,
    Span,
)

if TYPE_CHECKING:
    from circex.classify import SNTypeClassifier

REGEX_EXTRACTOR_ID: Final[str] = "regex-v1"


def _apply_telescope(
    extraction: CircularExtraction, body: str, provenance: dict[str, Span]
) -> None:
    """Name the telescope on rows that state none; a circular observes with one."""
    if not extraction.photometry:
        return
    if all(row.telescope for row in extraction.photometry):
        return
    found = parse_telescope_with_span(body)
    if found is None:
        return
    name, span = found
    for index, row in enumerate(extraction.photometry):
        if row.telescope:
            continue
        row.telescope = name
        row.telescope_canonical = canonicalize_telescope(name)
        provenance.setdefault(f"photometry[{index}].telescope", span)


def _apply_stated_mag_system(extraction: CircularExtraction, body: str) -> None:
    """Let a stated magnitude system override the one inferred from the filter."""
    stated = stated_mag_system(body)
    if stated is None:
        return
    for row in extraction.photometry:
        if row.mag is not None or row.limiting_mag is not None:
            row.mag_system = stated


class RegexExtractor(Extractor):
    """Regex baseline. Implements the Extractor protocol.

    An optional `sn_classifier` (the trained NB model) overrides the weak regex
    classification field — the classifier learns to abstain, which regex can't.
    """

    def __init__(self, sn_classifier: SNTypeClassifier | None = None) -> None:
        self._sn_classifier = sn_classifier

    @property
    def extractor_id(self) -> str:
        return REGEX_EXTRACTOR_ID

    def extract(self, circular: Circular) -> CircularExtraction:
        started = time.perf_counter()
        body = circular.body
        subject = circular.subject
        provenance: dict[str, Span] = {}

        # ---- event identification ----
        record = {"eventId": circular.event_id, "subject": subject, "body": body}
        primary_event_raw, all_events, _ = extract_events(record)
        event: Event | None = None
        if primary_event_raw:
            # Multi-event circulars (e.g. GW170817 + AT2017gfo) emit the full
            # list so the AT/optical name doesn't get dropped — matches the
            # Event.event_name `str | list[str]` schema and the LLM-extractor
            # behavior demonstrated in few-shot #4.
            name_value: str | list[str] = all_events if len(all_events) > 1 else primary_event_raw
            event = Event(event_name=name_value)
            # Try to ground the primary event in body text. Only spans that point
            # into circular.body are recorded; eventId / subject sources are not
            # addressable by a body offset.
            body_matches = extract_matches_with_positions(body)
            primary_norm = primary_event_raw.replace(" ", "")
            for name, start, end in body_matches:
                if name == primary_event_raw or name.replace(" ", "") == primary_norm:
                    provenance["event"] = Span(start=start, end=end, snippet=body[start:end])
                    break

        # ---- GCN cross-references → follow_up ----
        xref_hits = extract_gcn_xrefs_with_positions(body)
        follow_up: FollowUp | None = None
        if xref_hits:
            xref_ids = [cid for cid, _, _ in xref_hits]
            follow_up = FollowUp(
                reference={"gcn_circulars": ",".join(str(x) for x in xref_ids)},
            )
            # Provenance: span the entire range from the first to the last xref.
            first_start = xref_hits[0][1]
            last_end = xref_hits[-1][2]
            provenance["follow_up"] = Span(
                start=first_start, end=last_end, snippet=body[first_start:last_end]
            )

        # ---- localization ----
        localization: Localization | None = None
        coord_hit = parse_coords_with_span(body)
        if coord_hit is not None:
            (ra, dec), loc_span = coord_hit
            localization = Localization(ra=ra, dec=dec)
            provenance["localization"] = loc_span
        if localization is None:
            # IPN triangulation error box. A Konus/IPN-localized burst gets its
            # only position here, and the stated box dimensions are the only
            # uncertainty the circular gives.
            ipn = parse_ipn_error_box_with_span(body)
            if ipn is not None:
                (ipn_ra, ipn_dec, extent), ipn_span = ipn
                localization = Localization(ra=ipn_ra, dec=ipn_dec, ra_dec_error=extent)
                provenance["localization"] = ipn_span
        if localization is None:
            # Single-candidate counterpart table (ZTF/GROWTH neutrino/GW template,
            # e.g. GCN 45198): decimal-degree position + the counterpart's own
            # designation, which SkyPortal keys the source on (AT name preferred).
            cand = parse_pipe_candidate_with_span(body)
            if cand is not None:
                cand_names, cand_ra, cand_dec, cand_span = cand
                localization = Localization(ra=cand_ra, dec=cand_dec)
                provenance["localization"] = cand_span
                if cand_names:
                    existing = event.event_name if event is not None else None
                    merged = (
                        existing if isinstance(existing, list) else ([existing] if existing else [])
                    )
                    merged += [n for n in cand_names if n not in merged]
                    if event is not None:
                        event = event.model_copy(update={"event_name": merged})
                    else:
                        event = Event(event_name=merged if len(merged) > 1 else merged[0])

        # ---- photometry: prefer pipe table, then whitespace table, then prose ----
        photo_hits = (
            parse_pipe_table_with_spans(body, circular.trigger_time)
            or parse_fixed_width_table_with_spans(body, circular.trigger_time)
            or parse_mag_table_with_spans(body)
            or parse_single_mags_with_spans(body)
        )
        # Radio and X-ray rows are additive: they carry a flux density and an
        # energy flux, so neither competes with the magnitude parsers above.
        photo_hits = list(photo_hits) + parse_radio_with_spans(body) + parse_xray_with_spans(body)
        photometry = [row for row, _ in photo_hits]
        for idx, (_, span) in enumerate(photo_hits):
            provenance[f"photometry[{idx}]"] = span

        # ---- redshift (point) ----
        redshift = None
        z_hit = parse_redshift_with_span(body)
        if z_hit is not None:
            redshift, z_span = z_hit
            provenance["redshift"] = z_span

        # ---- redshift (bound) — schema gap, written to extraction_meta.notes ----
        # Only fires when no point value was matched, to avoid double-counting
        # the same circular's redshift line.
        bound_notes: list[str] = []
        if redshift is None:
            bound_hit = parse_redshift_bound(body)
            if bound_hit is not None:
                phrase, bound_span = bound_hit
                bound_notes.append(f"redshift_bound: {phrase}")
                provenance["_redshift_bound"] = bound_span

        # ---- classification ----
        classification = None
        cls_hit = parse_classification_with_span(body)
        if cls_hit is not None:
            classification, cls_span = cls_hit
            provenance["classification"] = cls_span
        # The trained classifier, when supplied, is authoritative: it abstains on
        # non-classification circulars (where regex over-fires) and predicts a
        # canonical SN type otherwise.
        if self._sn_classifier is not None:
            predicted = self._sn_classifier.predict_type(f"{subject}\n{body}")
            classification = Classification(classification=predicted) if predicted else None
            provenance.pop("classification", None)

        # An X-ray flash is a soft GRB the taxonomy has no node for, so it rides
        # on the subtype. The circular states it outright or hedges it, and the
        # difference is what a reader acts on.
        # The subject often carries the terse claim ("...is likely a flaring star").
        if (flare := parse_stellar_flare(f"{subject}\n{body}")) is not None:
            classification = Classification(classification="UV Ceti")
            if flare.endswith("candidate"):
                classification.subtype = flare

        if (xrf := parse_xrf_subtype(body)) is not None:
            if classification is None:
                classification = Classification(classification="GRB")
            classification.subtype = xrf

        # ---- time offsets ----
        offset_hits = parse_time_offsets_with_spans(body)
        time_offsets = [t for t, _ in offset_hits]
        for idx, (_, span) in enumerate(offset_hits):
            provenance[f"time_offsets[{idx}]"] = span

        latency_ms = (time.perf_counter() - started) * 1000.0

        extraction = CircularExtraction(
            circular_id=circular.circular_id,
            event=event,
            follow_up=follow_up,
            localization=localization,
            time_offsets=time_offsets,
            photometry=photometry,
            classification=classification,
            redshift=redshift,
            retraction=is_retraction(circular.subject),
            provenance=provenance,
            extraction_meta=ExtractionMeta(
                extractor=REGEX_EXTRACTOR_ID,
                latency_ms=latency_ms,
                notes=bound_notes,
            ),
        )
        # A datetime stated in the body is exact, so it binds first; T0 + a single
        # relative offset is rounded ("~4.5 days post-burst") and only fills rows
        # still untimed. Both are no-ops when the circular states neither.
        _apply_stated_mag_system(extraction, body)
        # Per-object first: a candidate's own paragraph times its rows, and only
        # what that leaves untimed falls back to the circular's single epoch.
        per_object = resolve_object_epochs(extraction, body)
        if per_object:
            extraction.extraction_meta.notes.append(
                "observation epochs taken per object from the paragraph naming each"
            )
        resolve_observation_epoch(
            extraction, body, circular.trigger_time, timed_per_object=per_object
        )
        resolve_inline_offsets(extraction, circular.body, circular.trigger_time)
        resolve_relative_epochs(extraction, circular.trigger_time)
        _apply_telescope(extraction, body, provenance)
        return extraction
