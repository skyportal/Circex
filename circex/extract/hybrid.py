"""HybridExtractor — per-field routing between regex and a grammar-constrained LLM.

Each field is served by whichever extractor's *failure mode that field can
tolerate*. This is the production system the four-way eval points to: a regular
expression is precise on lexically regular / structured fields and silent on
semantically mediated ones, while a constrained language model is the reverse.
Routing each field to its stronger extractor dominates either one alone.

The routing table below is grounded in a measured bake-off on 805 optical
circulars:

  - event names       -> regex   (regex emits fewer false-positive designations)
  - coordinates       -> regex   (the LLM emits 0% localizations as prompted;
                                   regex converts sexagesimal -> degrees)
  - photometry        -> LLM     (2,648 rows vs 964 — reads prose + odd tables)
  - classification    -> LLM     (regex classification is event-type defaults
                                   plus star-name contaminants — precision-poor)
  - redshift          -> LLM     (constrained F1 0.935 > regex 0.862)

For fields where both can contribute, the routing is primary -> secondary: the
secondary fills in only when the primary produced nothing. Provenance spans are
carried over from whichever extractor supplied each field, so the merged result
round-trips through ``to_label_fields`` for snippet-level human validation.
"""

from __future__ import annotations

import re

from circex.data.telescopes import canonicalize_telescope
from circex.extract.protocol import Circular, Extractor
from circex.extract.regex.mag_table import infer_bandpass_for
from circex.extract.regex.redshift import disclaimed_value
from circex.extract.regex.telescope import parse_telescope_with_span
from circex.extract.timing import parse_acquisition_epoch, parse_observation_epoch
from circex.schema import (
    CircularExtraction,
    Classification,
    ExtractionMeta,
    PhotometryExt,
    Redshift,
)
from circex.schema.span import Span

# field name -> (primary source, secondary source | None). Source is "regex" | "llm".
# A field taken from a source also inherits that source's provenance spans.
_ROUTING: dict[str, tuple[str, str | None]] = {
    "event": ("regex", "llm"),  # fewer false-positive designations
    "localization": ("regex", None),  # LLM emits no coordinates as prompted
    "follow_up": ("llm", "regex"),
    "datetime_": ("llm", "regex"),
    "time_offsets": ("llm", None),  # only the LLM produces these
    "photometry": ("llm", "regex"),  # 2,648 vs 964 rows
    "spectroscopy": ("llm", "regex"),
    "classification": ("llm", None),  # regex classification is defaults + contaminants
    "redshift": ("llm", "regex"),  # constrained F1 0.935 > regex 0.862
    "reporter": ("llm", "regex"),
}


# The LLM writes an X-ray band or a radio frequency into `filter` as prose
# ("0.5-10 keV"), which carries no bandpass and routes to no instrument. Regex
# parses those into energy_band_kev/frequency_ghz, so its structured rows stand
# in for the LLM's prose ones.
_BAND_AS_FILTER_RE = re.compile(
    r"\d\s*(?:-|–|to)\s*\d.*\b(?:keV|MeV|GHz|MHz)\b|\b(?:keV|GHz|MHz)\b", re.I
)


def _settle_redshift(
    fields: dict[str, object], body: str, regex_source: CircularExtraction
) -> str | None:
    """Reconcile the two readings of a redshift. Returns a note, if there is one.

    The LLM is the better reader of the number, but it answers the question it
    was asked and reports whatever redshift the circular contains. A catalogue
    redshift for a galaxy the authors offer as a *candidate* host -- hedged with
    a chance-coincidence probability, an offset, an "if associated" -- is not
    the transient's, and the regex already knows how to spot that context.
    """
    redshift = fields.get("redshift")
    if not isinstance(redshift, Redshift) or redshift.redshift is None:
        return None

    if (cue := disclaimed_value(body, redshift.redshift)) is not None:
        fields["redshift"] = None
        return (
            f"redshift {redshift.redshift:g} dropped: stated only for a "
            f"conditionally associated object ({cue!r})"
        )

    # Same number, two readings of what kind it is: the regex read the words
    # beside it, so its answer stands where it has one.
    from_regex = regex_source.redshift
    if (
        from_regex is not None
        and from_regex.redshift == redshift.redshift
        and from_regex.redshift_type is not None
        and from_regex.redshift_type != redshift.redshift_type
    ):
        fields["redshift"] = redshift.model_copy(update={"redshift_type": from_regex.redshift_type})
    return None


def _regex_epoch(regex_source: CircularExtraction) -> tuple[float, str] | None:
    """The single epoch regex read, when its rows agree on one.

    A row parsed from a table carries the epoch that row states, which is more
    specific than a time recovered from the prose: a table giving the
    mid-exposure beats "starting at 13:44 UT" by half the exposure. Rows
    disagreeing means several epochs, and no one of them speaks for the rest.
    """
    epochs = {
        (row.obs_mjd, row.obs_time)
        for row in regex_source.photometry
        if row.obs_mjd is not None and row.obs_time is not None
    }
    return epochs.pop() if len(epochs) == 1 else None


def _prefer_stated_epoch(
    fields: dict[str, object], body: str, regex_source: CircularExtraction
) -> None:
    """Replace generated epochs with the one the circular states, in place.

    A grammar constrains the model's syntax, not its arithmetic: it will emit a
    well-formed MJD that appears nowhere in the text. An epoch parsed from a
    character span is the more trustworthy of the two, so where the body states
    an observation time it overrides what the model supplied.
    """
    rows = fields.get("photometry")
    if not isinstance(rows, list) or not rows:
        return
    # An acquisition-image time is stated *for the photometry*, so it outranks
    # the epoch of whatever the circular went on to observe.
    stated = (
        parse_acquisition_epoch(body) or _regex_epoch(regex_source) or parse_observation_epoch(body)
    )
    if stated is None:
        return
    mjd, iso = stated
    for row in rows:
        if row.obs_mjd == mjd:
            continue
        row.obs_mjd = mjd
        row.obs_time = iso


# A small model asked for a telescope name sometimes degenerates into a run-on
# concatenation of every name in the circular. Real names are short and say a
# thing once.
_MAX_TELESCOPE_CHARS = 64


def _is_plausible_telescope(name: str) -> bool:
    if len(name) > _MAX_TELESCOPE_CHARS:
        return False
    words = name.lower().split()
    return len(set(words)) == len(words)


def _carry_telescope(
    fields: dict[str, object], regex_source: CircularExtraction, body: str
) -> None:
    """Name the telescope on LLM rows that omit it, in place.

    The LLM owns photometry, so a telescope regex found in the prose is otherwise
    discarded along with the rows regex built.
    """
    rows = fields.get("photometry")
    if not isinstance(rows, list) or not rows:
        return
    for row in rows:
        if row.telescope and not _is_plausible_telescope(row.telescope):
            row.telescope = None
            row.telescope_canonical = None

    named = next((r for r in regex_source.photometry if r.telescope), None)
    if named is not None:
        telescope, canonical = named.telescope, named.telescope_canonical
    else:
        # Regex found no rows of its own to hang a telescope on, so read the
        # prose directly rather than leaving the LLM's rows anonymous.
        found = parse_telescope_with_span(body)
        if found is None:
            return
        telescope = found[0]
        canonical = canonicalize_telescope(telescope)

    for row in rows:
        if row.telescope:
            continue
        row.telescope = telescope
        row.telescope_canonical = canonical


def _carry_xrf_subtype(fields: dict[str, object], regex_source: CircularExtraction) -> None:
    """Keep what regex read from the prose about the class, in place.

    The LLM owns classification, so a subtype it does not emit is otherwise
    dropped; a stellar flare, which regex states outright, replaces it.
    """
    regex_cls = regex_source.classification
    if regex_cls is None:
        return
    if not regex_cls.subtype and regex_cls.classification != "UV Ceti":
        return
    chosen = fields.get("classification")
    if regex_cls.classification == "UV Ceti":
        # A circular saying the trigger was a flaring star is settling what it
        # was; the model's guess from the same text does not override that.
        fields["classification"] = regex_cls
    elif isinstance(chosen, Classification):
        if not chosen.subtype:
            chosen.subtype = regex_cls.subtype
    elif chosen is None:
        fields["classification"] = regex_cls


def _is_structured(row: PhotometryExt) -> bool:
    return row.energy_band_kev is not None or row.frequency_ghz is not None


def _rescue_structured_photometry(
    fields: dict[str, object], regex_source: CircularExtraction
) -> None:
    """Restore regex X-ray/radio rows the LLM returned as prose, in place."""
    rows = fields.get("photometry")
    if not isinstance(rows, list) or not rows:
        return
    if any(_is_structured(r) for r in rows):
        return
    structured = [r for r in regex_source.photometry if _is_structured(r)]
    if not structured:
        return
    kept = [
        r
        for r in rows
        if not (r.bandpass is None and r.filter and _BAND_AS_FILTER_RE.search(r.filter))
    ]
    fields["photometry"] = kept + structured


def _fill_bandpasses(fields: dict[str, object]) -> None:
    """Name the bandpass for any row that states a filter, in place.

    The LLM owns photometry and writes the filter it read, not a canonical
    bandpass: rows arrive as `filter="J"` with `bandpass=None`. The regex
    row-builders crosswalk every filter they recognise, so the mapping exists
    and is simply never reached for an LLM row. A magnitude with no bandpass
    cannot be plotted or fitted, which makes the row not worth having.
    """
    rows = fields.get("photometry")
    if not isinstance(rows, list):
        return
    for row in rows:
        if getattr(row, "bandpass", None) is not None:
            continue
        band = infer_bandpass_for(getattr(row, "filter", None), getattr(row, "telescope", None))
        if band is not None:
            row.bandpass = band


def _present(value: object) -> bool:
    """A field counts as populated when it is neither None nor an empty list."""
    return value is not None and value != []


def _provenance_for(source: CircularExtraction, field: str) -> dict[str, Span]:
    """Provenance entries from ``source`` that belong to ``field``.

    Keys are dotted/indexed paths (``localization.ra``, ``photometry[0]``); the
    base segment before the first ``.`` or ``[`` names the top-level field. The
    ``datetime_`` field is stored under the ``datetime`` provenance base.
    """
    accepted = {field, field.rstrip("_")}
    out: dict[str, Span] = {}
    for key, span in source.provenance.items():
        base = key.split(".", 1)[0].split("[", 1)[0]
        if base in accepted:
            out[key] = span
    return out


class HybridExtractor(Extractor):
    """Merge a regex extractor and an LLM extractor by the per-field routing table.

    `routing_overrides` replaces individual entries of the default table — e.g. a
    consumer whose regex side carries a trained SN-type classifier passes
    ``{"classification": ("llm", "regex")}`` so that classifier output survives
    when the LLM abstains (the default drops regex classification as noise).
    """

    def __init__(
        self,
        regex: Extractor,
        llm: Extractor,
        routing_overrides: dict[str, tuple[str, str | None]] | None = None,
    ) -> None:
        self._regex = regex
        self._llm = llm
        self._routing = {**_ROUTING, **(routing_overrides or {})}

    @property
    def extractor_id(self) -> str:
        return f"hybrid:{self._regex.extractor_id}+{self._llm.extractor_id}"

    def extract(self, circular: Circular) -> CircularExtraction:
        sources = {"regex": self._regex.extract(circular), "llm": self._llm.extract(circular)}
        fields: dict[str, object] = {"circular_id": circular.circular_id}
        provenance: dict[str, Span] = {}

        for field, (primary, secondary) in self._routing.items():
            value = getattr(sources[primary], field)
            chosen = primary if _present(value) else None
            if chosen is None and secondary is not None:
                alt = getattr(sources[secondary], field)
                if _present(alt):
                    value, chosen = alt, secondary
            fields[field] = value
            if chosen is not None:
                provenance.update(_provenance_for(sources[chosen], field))

        _rescue_structured_photometry(fields, sources["regex"])
        _fill_bandpasses(fields)
        _carry_telescope(fields, sources["regex"], circular.body)
        _prefer_stated_epoch(fields, circular.body, sources["regex"])
        _carry_xrf_subtype(fields, sources["regex"])
        redshift_note = _settle_redshift(fields, circular.body, sources["regex"])

        # Retraction is read from the subject line, so it is the same either way
        # and never routed; without this the merged result always reads False.
        fields["retraction"] = sources["regex"].retraction

        fields["provenance"] = provenance
        fields["extraction_meta"] = ExtractionMeta(
            extractor=self.extractor_id,
            model_id=getattr(self._llm, "model_id", None),
            prompt_version=getattr(self._llm, "prompt_version", None),
            notes=[redshift_note] if redshift_note else [],
        )
        return CircularExtraction(**fields)  # type: ignore[arg-type]
