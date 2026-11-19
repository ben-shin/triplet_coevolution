"""Find local homolog domains, retrieve exact IDs, and orient them to the reference."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.io.profiles import check_profile, read_sequences
from triplet_coevolution.msa.builders import comparison, oriented
from triplet_coevolution.search import hmmer, infernal, psiblast
from triplet_coevolution.search.shared import Hit


def _reference_hits(comp, reference):
    molecule = comp["molecule"]
    alphabet = CANONICAL[molecule]
    possible = sum(c in alphabet for c in reference)
    strands = ("+",) if molecule == "protein" else ("+", "-")
    for seq_id, seq in read_sequences(comp["database"], molecule):
        for strand in strands:
            target = oriented(seq, molecule, strand)
            alignment = comparison(reference, target, molecule, "local")
            if alignment is None:
                continue
            both = same = 0
            for i, j in alignment.indices.T:
                if i >= 0 and j >= 0 and reference[i] in alphabet and target[j] in alphabet:
                    both += 1
                    same += reference[i] == target[j]
            if not both:
                continue
            coverage, identity = both / possible, same / both
            if coverage < comp["min_coverage"] or identity < comp["min_identity"]:
                continue
            start, end = map(int, alignment.coordinates[1, [0, -1]])
            first, last = (
                (start + 1, end) if strand == "+" else (len(seq) - end + 1, len(seq) - start)
            )
            yield Hit(
                seq_id, first, last, strand, alignment.score, identity=identity, coverage=coverage
            )


def _model_strand(comp, backend, reference, work):
    if comp["molecule"] == "protein":
        return "+"
    # Score the reference separately so its addition cannot alter the database E-values.
    orientation_work = work / "orientation"
    orientation_work.mkdir()
    target = orientation_work / "reference.fasta"
    target.write_text(f">reference\n{reference}\n", encoding="ascii")
    hits = [h for h in backend.search(comp, target, orientation_work) if h.evalue <= comp["evalue"]]
    if not hits:
        raise ValueError(
            "nucleotide profile did not match its coordinate reference at the configured E-value"
        )
    best = max((h.score, -h.evalue) for h in hits)
    strands = {h.strand for h in hits if (h.score, -h.evalue) == best}
    if len(strands) != 1:
        raise ValueError("nucleotide profile has equally strong hits on both reference strands")
    return strands.pop()


def _external_hits(comp, reference, work):
    check_profile(comp)
    normalized = work / "database.fasta"
    ids = {}
    with normalized.open("w", encoding="ascii") as handle:
        for idx, (seq_id, seq) in enumerate(read_sequences(comp["database"], comp["molecule"]), 1):
            key = f"s{idx:09d}"
            ids[key] = seq_id
            handle.write(f">{key}\n{seq}\n")
    backend = {"hmmer": hmmer, "infernal": infernal, "psiblast": psiblast}[comp["backend"]]
    ref_strand = _model_strand(comp, backend, reference, work)
    for hit in backend.search(comp, normalized, work):
        if hit.seq_id not in ids:
            raise ValueError(f"unknown search result ID {hit.seq_id!r}")
        if hit.evalue <= comp["evalue"]:
            strand = hit.strand if ref_strand == "+" else {"+": "-", "-": "+"}[hit.strand]
            yield replace(hit, seq_id=ids[hit.seq_id], strand=strand, model_strand=hit.strand)


def retrieve(comp, reference):
    best = {}
    with TemporaryDirectory(prefix="triplet-search-") as tmp:
        hits = (
            _reference_hits(comp, reference)
            if comp["backend"] == "reference"
            else _external_hits(comp, reference, Path(tmp))
        )
        for hit in hits:
            # Resolve multiple domains on one database record deterministically.
            rank = (-hit.score, hit.evalue or 0, hit.strand, hit.first, hit.last)
            if hit.seq_id not in best or rank < best[hit.seq_id][0]:
                best[hit.seq_id] = rank, hit
    selected = {key: value[1] for key, value in best.items()}
    sequences = {}
    for seq_id, seq in read_sequences(comp["database"], comp["molecule"]):
        if seq_id == comp["reference_id"] and seq != reference:
            raise ValueError(f"database record {seq_id!r} conflicts with reference sequence")
        if seq_id in selected:
            hit = selected[seq_id]
            if hit.last > len(seq):
                raise ValueError(f"search hit outside database record {seq_id!r}")
            # The supplied full reference determines coordinates when it also appears in the database.
            sequences[seq_id] = (
                reference
                if seq_id == comp["reference_id"]
                else oriented(seq[hit.first - 1 : hit.last], comp["molecule"], hit.strand)
            )
    if set(sequences) != set(selected):
        raise ValueError("database changed during retrieval")
    return selected, sequences
