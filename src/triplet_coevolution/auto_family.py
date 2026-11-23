"""Determine scoring representations and coordinate representatives for family runs."""

import re

from triplet_coevolution.core.alphabets import AMBIGUOUS, CANONICAL, GAPS, MISSING
from triplet_coevolution.evidence.identifiers import accession

_PROFILE = re.compile(r"(PF\d{5}|RF\d{5})(?:\.\d+)?")
_TRANSCRIPT = re.compile(r"(?:NR|XR|NM|XM)_\d+(?:\.\d+)?|URS[\dA-F]{10}(?:_[1-9]\d*)?")
_GENOMIC = re.compile(r"(?:NC|NG|NT|NW)_\d+(?:\.\d+)?|NZ_[A-Z]{0,6}\d+(?:\.\d+)?")


def _identity_types(tag, value):
    token = value.upper()
    if tag.upper() in ("MOLECULE", "TP"):
        return {token.lower()} & {"protein", "rna", "dna"}
    if tag.upper() == "DR":
        fields = [part.strip() for part in token.split(";")]
        if len(fields) < 2:
            return set()
        namespace, token = fields[:2]
        if namespace in {"UNIPROT", "UNIPROTKB", "SWISS-PROT"} and accession(token):
            return {"protein"}
        if namespace not in {"REFSEQ", "RNACENTRAL"}:
            return set()
    parts = token.split("|")
    if len(parts) == 3 and parts[0] in {"SP", "TR"} and accession(parts[1]):
        return {"protein"}
    if _TRANSCRIPT.fullmatch(token):
        return {"rna"}
    if _GENOMIC.fullmatch(token):
        # A genomic interval can represent a transcribed feature on either strand.
        return {"nucleotide"}
    if re.fullmatch(r"(?:NP|XP|YP|WP)_\d+(?:\.\d+)?", token):
        return {"protein"}
    return set()


def infer_molecule(inventory, configured=None):
    """Audit explicit declarations, typed identifiers, then the observed scoring alphabet."""
    if configured is not None and configured not in CANONICAL:
        raise ValueError(f"unknown configured molecule {configured!r}")
    match = _PROFILE.fullmatch(inventory.accession or "")
    profile = match[1] if match else None
    profile_type = ("protein" if profile.startswith("PF") else "rna") if profile else None
    stated = {v for _, v in inventory.annotations}
    for tag, value in inventory.identity_metadata():
        stated.update(_identity_types(tag, value))
    if profile_type:
        stated.add(profile_type)

    symbols = set(inventory.symbols) - GAPS - MISSING
    biological = stated - {"nucleotide"}
    conflicting = (
        len(biological) > 1
        or "protein" in stated
        and ("nucleotide" in stated or inventory.has_reverse)
    )
    warnings = []
    source, molecule = "ambiguous_alphabet", None
    if configured:
        molecule, source = configured, "configured"
        if (
            conflicting
            or biological - {configured}
            or (configured == "protein" and ("nucleotide" in stated or inventory.has_reverse))
        ):
            warnings.append("configured molecule conflicts with source metadata")
    elif conflicting:
        source = "conflicting_molecule_metadata"
    elif biological:
        molecule = next(iter(biological))
        source = "family_accession" if profile else "typed_metadata"
    else:
        nucleotide = set(CANONICAL["dna"] + CANONICAL["rna"] + AMBIGUOUS["dna"])
        protein = set(CANONICAL["protein"] + AMBIGUOUS["protein"])
        protein_specific = set(CANONICAL["protein"]) - nucleotide
        if symbols <= protein and symbols & protein_specific:
            if "nucleotide" not in stated and not inventory.has_reverse:
                molecule, source = "protein", "protein_specific_residues"
            else:
                source = "conflicting_molecule_metadata"
        elif symbols <= nucleotide and {"A", "C", "G"} <= symbols:
            # T/U are also admissible protein symbols. Only infer their scoring
            # representation for an informative alignment or typed genomic records.
            genomic = "nucleotide" in stated or inventory.has_reverse
            informative = genomic or (
                inventory.max_nucleotide_coverage >= 20 and symbols <= set("ACGTUN")
            )
            if informative and ("T" in symbols) != ("U" in symbols):
                molecule = "dna" if "T" in symbols else "rna"
                source = "genomic_alphabet" if genomic else "nucleotide_alphabet_heuristic"
                warnings.append(
                    "alphabet identifies a scoring representation; biological molecule is inferred"
                )
    transcribe = molecule == "rna" and "rna" in biological and not conflicting and "T" in symbols
    if molecule:
        allowed = set(CANONICAL[molecule] + AMBIGUOUS[molecule]) | ({"T"} if transcribe else set())
        if symbols - allowed or molecule == "protein" and inventory.has_reverse:
            warnings.append(
                "source symbols or domain orientation disagree with the scoring representation"
            )
            if not configured:
                source, molecule, transcribe = "incompatible_source_alignment", None, False

    # A declared override can still score selected rows through the normal
    # validator, but cannot promote a conflicting biological profile into evidence.
    usable_profile = profile if molecule == profile_type and not conflicting else None
    return {
        "molecule": molecule,
        "source": source,
        "profile": usable_profile,
        "family_accession": inventory.accession,
        "metadata_types": sorted(stated),
        "symbols": sorted(inventory.symbols),
        "transcribe_t": transcribe,
        "warnings": warnings,
        "reason": source,
    }


def choose_reference(inventory, row_ids, molecule, *, transcribe_t=None):
    """Choose among selected, uniquely placed rows by canonical coverage, interval and ID."""
    chosen = best = None
    n_candidates = 0
    if molecule:
        if transcribe_t is None:
            transcribe_t = (
                molecule == "rna"
                and "T" in inventory.symbols
                and infer_molecule(inventory, molecule)["transcribe_t"]
            )
        for row_id in row_ids:
            row = inventory.row(row_id)
            if row is None:
                raise ValueError(f"matched row {row_id!r} does not occur in its source alignment")
            coverage = inventory.canonical_coverage(row_id, molecule, transcribe_t=transcribe_t)
            if not coverage or molecule == "protein" and row.start and row.start > row.end:
                continue
            n_candidates += 1
            key = (-coverage, row.start is None, row.row_id)
            if best is None or key < best:
                chosen, best = row, key
    return chosen, {
        "source": "matched_coordinate_representative" if chosen else "no_usable_matched_reference",
        "rank": ["most_canonical_residues", "explicit_interval", "lexical_exact_row_id"],
        "eligible_rows": n_candidates,
        "canonical_residues": -best[0] if best else 0,
        "alignment_columns": inventory.n_columns,
        "reference_row_id": chosen.row_id if chosen else None,
        "numbering": ("domain_interval" if chosen.start else "row_relative") if chosen else None,
    }
