"""Reference-anchored alignments from independently aligned, oriented domains."""

from Bio import Align
from Bio.Seq import Seq

from triplet_coevolution.core.alignment import ComponentMSA
from triplet_coevolution.core.alphabets import CANONICAL


def oriented(sequence, molecule, strand):
    if strand == "+":
        return sequence
    if molecule == "rna":
        return str(Seq(sequence).reverse_complement_rna())
    if molecule == "dna":
        return str(Seq(sequence).reverse_complement())
    raise ValueError("a protein sequence cannot have reverse orientation")


def comparison(reference, target, molecule, mode):
    if len(reference) * len(target) > 10_000_000:
        raise ValueError("pairwise alignment exceeds 10 million cells; use domain-sized records")
    alphabet = CANONICAL[molecule]

    def canonical(seq):
        return "".join(letter if letter in alphabet else "?" for letter in seq)

    aligner = Align.PairwiseAligner(
        mode=mode,
        match_score=2,
        mismatch_score=-1,
        open_gap_score=-4,
        extend_gap_score=-1,
        wildcard="?",
    )
    # next() avoids counting optimal alignments, which may overflow for repetitive sequences.
    return next(iter(aligner.align(canonical(reference), canonical(target))), None)


def anchored_msa(name, molecule, ref_id, reference, hits):
    """Keep every reference column. Omit unanchored insertions from the reference projection."""
    ids = [ref_id]
    rows = [reference]
    for seq_id, sequence in sorted(hits.items()):
        if seq_id == ref_id:
            if sequence != reference:
                raise ValueError(f"{name}: database hit conflicts with reference ID {ref_id!r}")
            continue
        alignment = comparison(reference, sequence, molecule, "global")
        row = ["-"] * len(reference)
        for i, j in alignment.indices.T:
            if i >= 0 and j >= 0:
                row[i] = sequence[j]
        ids.append(seq_id)
        rows.append("".join(row))
    return ComponentMSA(name, molecule, ids, rows)
