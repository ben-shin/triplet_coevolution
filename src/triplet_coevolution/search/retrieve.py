"""Retrieve local homolog domains and retain the evidence for their reference coordinates."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.io.profiles import check_profile, read_sequences
from triplet_coevolution.msa.builders import (
    Placement,
    choose_locus,
    comparison,
    from_trace,
    oriented,
    tied_placements,
)
from triplet_coevolution.search import hmmer, infernal, psiblast
from triplet_coevolution.search.shared import Hit


def _support(reference, placement, molecule, possible):
    alphabet = CANONICAL[molecule]
    both = same = 0
    for residue, symbol in zip(reference[placement.first - 1 : placement.last], placement.aligned):
        if residue in alphabet and symbol in alphabet:
            both += 1
            same += residue == symbol
    return (same / both if both else 0.0), both / possible


def _reference_hits(comp, reference, possible):
    molecule = comp["molecule"]
    strands = ("+",) if molecule == "protein" else ("+", "-")
    for seq_id, seq in read_sequences(comp["database"], molecule):
        for strand in strands:
            target = oriented(seq, molecule, strand)
            alignment = comparison(reference, target, molecule, "local")
            if alignment is None:
                continue
            placement = from_trace(alignment, target, strand)
            identity, coverage = _support(reference, placement, molecule, possible)
            if coverage < comp["min_coverage"] or identity < comp["min_identity"]:
                continue
            start, end = map(int, alignment.coordinates[1, [0, -1]])
            first, last = (
                (start + 1, end) if strand == "+" else (len(seq) - end + 1, len(seq) - start)
            )
            yield Hit(
                seq_id,
                first,
                last,
                strand,
                alignment.score,
                identity=identity,
                coverage=coverage,
                placement=placement,
            )


def _external_inputs(comp, reference, work):
    check_profile(comp)
    normalized = work / "database.fasta"
    ids = {}
    with normalized.open("w", encoding="ascii") as handle:
        for idx, (seq_id, seq) in enumerate(read_sequences(comp["database"], comp["molecule"]), 1):
            key = f"s{idx:09d}"
            ids[key] = seq_id
            handle.write(f">{key}\n{seq}\n")
    backend = {"hmmer": hmmer, "infernal": infernal, "psiblast": psiblast}[comp["backend"]]
    loci = []
    if "profile" in comp:
        # Keep *all* matching reference intervals. A repeated domain can yield several equally
        # plausible loci. Search the reference separately to preserve target database E-values.
        probe = work / "reference-search"
        probe.mkdir()
        path = probe / "reference.fasta"
        path.write_text(f">reference\n{reference}\n", encoding="ascii")
        loci = [h for h in backend.search(comp, path, probe) if h.evalue <= comp["evalue"]]

    def hits():
        for hit in backend.search(comp, normalized, work):
            if hit.seq_id not in ids:
                raise ValueError(f"unknown search result ID {hit.seq_id!r}")
            if hit.evalue <= comp["evalue"]:
                yield replace(hit, seq_id=ids[hit.seq_id], model_strand=hit.strand)

    return hits(), loci


def _portable_placements(reference, fragment, hit, molecule):
    first = hit.placement
    target = oriented(fragment, molecule, hit.strand)
    yield from tied_placements(reference, target, molecule, first, 0, len(reference))
    if molecule != "protein":
        opposite = "-" if hit.strand == "+" else "+"
        other = oriented(fragment, molecule, opposite)
        if other != target:
            aln = comparison(reference, other, molecule, "fitting")
            if aln is not None and aln.score >= first.score:
                alt = from_trace(aln, other, opposite)
                yield from tied_placements(reference, other, molecule, alt, 0, len(reference))


def _profile_placements(reference, fragment, hit, loci, molecule):
    # A reference hit for an unrelated portion of a profile cannot anchor this target domain.
    overlaps = [
        min(hit.model_last, h.model_last) - max(hit.model_first, h.model_first) + 1 for h in loci
    ]
    best_overlap = max(overlaps, default=0)
    if best_overlap <= 0:
        return
    for locus, overlap in zip(loci, overlaps):
        if overlap < best_overlap:
            continue
        strand = "+" if hit.model_strand == locus.strand else "-"
        target = oriented(fragment, molecule, strand)
        start, end = locus.first - 1, locus.last
        alignment = comparison(reference[start:end], target, molecule, "fitting")
        placement = from_trace(alignment, target, strand, offset=start)
        if placement is not None:
            yield from tied_placements(reference, target, molecule, placement, start, end)


def _place_hit(comp, reference, seq, hit, loci, possible):
    molecule = comp["molecule"]
    if hit.last > len(seq):
        raise ValueError(f"search hit outside database record {hit.seq_id!r}")
    fragment = seq[hit.first - 1 : hit.last]
    if seq == reference:
        return Placement(hit.first, fragment, "+", hit.score), "source_interval"
    if molecule != "protein" and oriented(seq, molecule, "-") == reference:
        return Placement(
            len(seq) - hit.last + 1, oriented(fragment, molecule, "-"), "-", hit.score
        ), "source_interval"
    if hit.seq_id == comp["reference_id"]:
        raise ValueError(f"database record {hit.seq_id!r} conflicts with reference sequence")
    if comp["backend"] == "reference":

        def qualifying_traces():
            for candidate in _portable_placements(reference, fragment, hit, molecule):
                identity, coverage = _support(reference, candidate, molecule, possible)
                if identity >= comp["min_identity"] and coverage >= comp["min_coverage"]:
                    yield candidate

        placements = qualifying_traces()
        method = "query_trace"
    else:
        # A sequence-initiated PSI-BLAST HSP has direct query coordinates.
        if "profile" not in comp:
            loci = [
                Hit(
                    "reference",
                    hit.model_first,
                    hit.model_last,
                    "+",
                    hit.score,
                    model_first=hit.model_first,
                    model_last=hit.model_last,
                )
            ]
        placements = _profile_placements(reference, fragment, hit, loci, molecule)
        method = "profile_locus" if "profile" in comp else "query_interval"
    return choose_locus(
        placements, reference, seq, hit.seq_id, (hit.first, hit.last), molecule
    ), method


def retrieve(comp, reference):
    possible = sum(c in CANONICAL[comp["molecule"]] for c in reference)
    best = {}
    loci = []
    with TemporaryDirectory(prefix="triplet-search-") as tmp:
        if comp["backend"] == "reference":
            hits = _reference_hits(comp, reference, possible)
        else:
            hits, loci = _external_inputs(comp, reference, Path(tmp))
        for hit in hits:
            # Source-interval selection precedes placement on the coordinate reference.
            rank = (-hit.score, hit.evalue or 0, hit.strand, hit.first, hit.last)
            if hit.seq_id not in best or rank < best[hit.seq_id][0]:
                best[hit.seq_id] = rank, hit
    selected = {}
    sequences = {}
    for seq_id, seq in read_sequences(comp["database"], comp["molecule"]):
        if seq_id in best:
            hit = best[seq_id][1]
            placement, method = _place_hit(comp, reference, seq, hit, loci, possible)
            if comp["backend"] == "reference":
                identity, coverage = _support(reference, placement, comp["molecule"], possible)
                hit = replace(hit, identity=identity, coverage=coverage)
            selected[seq_id] = replace(
                hit, strand=placement.strand, placement=placement, projection_method=method
            )
            sequences[seq_id] = oriented(
                seq[hit.first - 1 : hit.last], comp["molecule"], placement.strand
            )
        elif (
            seq_id == comp["reference_id"]
            and seq != reference
            and (comp["molecule"] == "protein" or oriented(seq, comp["molecule"], "-") != reference)
        ):
            raise ValueError(f"database record {seq_id!r} conflicts with reference sequence")
    if set(selected) != set(best):
        raise ValueError("database changed during retrieval")
    return selected, sequences
