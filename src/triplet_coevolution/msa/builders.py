"""Project source-coordinate intervals and search tracebacks onto reference columns."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices
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


def _check_size(reference, target):
    if len(reference) * len(target) > 10_000_000:
        raise ValueError("pairwise alignment exceeds 10 million cells; use domain-sized records")


def _canonical(seq, molecule):
    return "".join(letter if letter in CANONICAL[molecule] else "?" for letter in seq)


def comparison(reference, target, molecule, mode):
    _check_size(reference, target)
    aligner = Align.PairwiseAligner(
        mode="global" if mode == "fitting" else mode,
        match_score=2,
        mismatch_score=-1,
        open_gap_score=-4,
        extend_gap_score=-1,
        wildcard="?",
    )
    if mode == "fitting":
        # Fit the whole selected target domain anywhere inside the known reference locus.
        aligner.left_deletion_score = 0
        aligner.right_deletion_score = 0
    return next(
        iter(aligner.align(_canonical(reference, molecule), _canonical(target, molecule))), None
    )


@dataclass(frozen=True)
class Placement:
    """A gapped reference projection beginning at a one-based reference column."""

    first: int
    aligned: str
    strand: str
    score: float

    @property
    def last(self):
        return self.first + len(self.aligned) - 1

    def row(self, length):
        return "-" * (self.first - 1) + self.aligned + "-" * (length - self.last)


def from_trace(alignment, target, strand, offset=0):
    reference_idx, target_idx = alignment.indices
    both = (reference_idx >= 0) & (target_idx >= 0)
    ref_cols, target_cols = reference_idx[both], target_idx[both]
    if not len(ref_cols):
        return None
    first, last = int(ref_cols[0]), int(ref_cols[-1])
    row = ["-"] * (last - first + 1)
    for i, j in zip(ref_cols, target_cols):
        row[i - first] = target[j]
    return Placement(offset + first + 1, "".join(row), strand, alignment.score)


def _fitting_ends(reference, target, molecule, score):
    """All reference end columns that attain score for the complete target.

    These are affine-gap fitting scores with a free reference prefix. Keeping a
    row of end scores avoids enumerating combinatorial internal gap tracebacks.
    """
    ref = np.frombuffer(_canonical(reference, molecule).encode("ascii"), dtype=np.uint8)
    query = _canonical(target, molecule).encode("ascii")
    size = len(ref) + 1
    floor = -100_000_000
    best = np.zeros(size, dtype=np.int32)
    insert = np.full(size, floor, dtype=np.int32)
    diagonal = np.full(size, floor, dtype=np.int32)
    offset = np.arange(len(ref), dtype=np.int32)
    for symbol in query:
        sub = np.where((ref == 63) | (symbol == 63), 0, np.where(ref == symbol, 2, -1))
        diagonal[1:] = best[:-1] + sub
        insert = np.maximum(best - 4, insert - 1)
        occupied = np.maximum(diagonal, insert)
        # A deletion may open after any previous occupied column, then extend to j.
        deletion = np.maximum.accumulate(occupied[:-1] - 4 + offset) - offset
        best = occupied.copy()
        best[1:] = np.maximum(occupied[1:], deletion)
    return np.flatnonzero(occupied == score).tolist()


def tied_placements(reference, target, molecule, first, lo, hi):
    """Yield one traceback per equally scoring reference span, including overlaps."""
    yield first
    window = reference[lo:hi]
    if first.score == 2 * len(target) and all(c in CANONICAL[molecule] for c in target):
        start = window.find(target)
        while start >= 0:
            if lo + start + 1 != first.first:
                yield Placement(lo + start + 1, target, first.strand, first.score)
            start = window.find(target, start + 1)
        return
    ends = _fitting_ends(window, target, molecule, first.score)
    starts = [
        len(window) - j for j in _fitting_ends(window[::-1], target[::-1], molecule, first.score)
    ]
    # Even a perfect match loses at least one point per excess gap position.
    min_length = max(1, int(np.ceil((first.score + len(target)) / 3)))
    max_length = int(3 * len(target) - first.score)
    for start in starts:
        left = bisect_left(ends, start + min_length)
        right = bisect_right(ends, start + max_length)
        for end in ends[left:right]:
            if (lo + start + 1, lo + end) == (first.first, first.last):
                continue
            # Start and end scores are marginal. Verify each candidate interval as a pair.
            aln = comparison(window[start:end], target, molecule, "global")
            if aln.score == first.score:
                placement = from_trace(aln, target, first.strand, offset=lo + start)
                if placement and (placement.first, placement.last) == (lo + start + 1, lo + end):
                    yield placement


# A high-scoring boundary symbol forces local flank comparison to start at the domain.
# It exceeds every attainable unanchored score under the pairwise cell limit above.
_ANCHOR_SCORE = 1_000_000


@lru_cache(maxsize=3)
def _flank_aligner(molecule):
    alphabet = CANONICAL[molecule] + "?!"
    matrix = substitution_matrices.Array(alphabet, dims=2)
    for a in alphabet:
        for b in alphabet:
            matrix[a, b] = 0 if "?" in (a, b) else (2 if a == b else -1)
    matrix["!", "!"] = _ANCHOR_SCORE
    return Align.PairwiseAligner(
        mode="local", substitution_matrix=matrix, open_gap_score=-4, extend_gap_score=-1
    )


def _flank_score(ref, seq, molecule):
    if not ref or not seq:
        return 0
    _check_size(ref, seq)
    return (
        _flank_aligner(molecule).score(
            "!" + _canonical(ref, molecule), "!" + _canonical(seq, molecule)
        )
        - _ANCHOR_SCORE
    )


def choose_locus(candidates, reference, raw, seq_id, source_interval, molecule):
    """Use flanking support only when domain scores tie at different reference loci."""

    first, last = source_interval

    def context(p):
        left, right = raw[: first - 1], raw[last:]
        if p.strand == "-":
            left, right = oriented(right, molecule, "-"), oriented(left, molecule, "-")
        return _flank_score(reference[: p.first - 1][::-1], left[::-1], molecule) + _flank_score(
            reference[p.last :], right, molecule
        )

    winner = other = None
    flank_best = None
    for p in candidates:
        if winner is None or p.score > winner.score:
            winner, other, flank_best = p, None, None
            continue
        if p.score < winner.score or (p.first, p.last, p.strand) == (
            winner.first,
            winner.last,
            winner.strand,
        ):
            continue
        if flank_best is None:
            flank_best = context(winner)
        flank = context(p)
        if flank > flank_best:
            winner, other, flank_best = p, None, flank
        elif flank == flank_best:
            other = p
            if first == 1 and last == len(raw) and p.score == 2 * len(raw):
                break  # No flank is available and the domain score is already maximal.
    if winner is None:
        raise ValueError(
            f"{seq_id}: no compatible reference locus; check the profile and reference"
        )
    if other is not None:
        spans = f"{winner.first}..{winner.last} ({winner.strand}), {other.first}..{other.last} ({other.strand})"
        raise ValueError(
            f"{seq_id}: ambiguous reference locus among {spans}; "
            "supply a domain-specific reference or a curated alignment"
        )
    return winner


def assemble_msa(name, molecule, ref_id, reference, rows):
    """Add the full coordinate seed to already-projected observation rows."""
    if any(len(row) != len(reference) for row in rows.values()):
        raise ValueError("projected rows must have the reference length")
    ref_row = ref_id
    if ref_id in rows and rows[ref_id] != reference:
        ref_row = f"{ref_id}.coordinate"
        suffix = 1
        while ref_row in rows:
            ref_row = f"{ref_id}.coordinate.{suffix}"
            suffix += 1
    ids = [ref_row, *(seq_id for seq_id in sorted(rows) if seq_id != ref_row)]
    return ComponentMSA(name, molecule, ids, [reference, *(rows[seq_id] for seq_id in ids[1:])])


def anchored_msa(name, molecule, ref_id, reference, hits):
    """Project ungapped standalone fragments without assuming an unknown repeat location."""
    rows = {}
    for seq_id, seq in hits.items():
        first = from_trace(comparison(reference, seq, molecule, "fitting"), seq, "+")
        options = (
            tied_placements(reference, seq, molecule, first, 0, len(reference)) if first else []
        )
        anchor = choose_locus(options, reference, seq, seq_id, (1, len(seq)), molecule)
        rows[seq_id] = anchor.row(len(reference))
    return assemble_msa(name, molecule, ref_id, reference, rows)
