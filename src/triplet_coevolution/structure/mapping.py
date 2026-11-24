"""Map complete reference sequences to deposited polymer sequences and assembly copies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from Bio.Align import substitution_matrices

if TYPE_CHECKING:
    from triplet_coevolution.structure.types import Chain, Residue, Structure


@dataclass(frozen=True)
class ReferenceSequence:
    component: str
    reference_id: str
    sequence: str
    molecule: str
    start: int = 1
    strand: str = "+"
    organism: str | None = None

    def __post_init__(self) -> None:
        if (
            self.component not in ("A", "B", "C")
            or not self.reference_id
            or self.molecule not in ("protein", "rna", "dna")
        ):
            raise ValueError("a reference needs a component, stable identifier and molecule type")
        seq = self.sequence.upper().replace("-", "").replace(".", "")
        alphabet = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ?" if self.molecule == "protein" else "ACGTURYSWKMBDHVNX?"
        )
        if not seq or any(c not in alphabet for c in seq):
            raise ValueError(f"invalid {self.molecule} reference sequence for {self.component}")
        if (
            type(self.start) is not int
            or self.start < 1
            or self.strand not in ("+", "-")
            or (self.strand == "-" and self.start < len(seq))
        ):
            raise ValueError("invalid reference coordinate start/strand")
        object.__setattr__(self, "sequence", seq)

    @property
    def positions(self) -> range:
        step = 1 if self.strand == "+" else -1
        return range(self.start, self.start + step * len(self.sequence), step)


@dataclass(frozen=True)
class Candidate:
    copy_id: str
    entity_id: str
    organism: str
    taxonomy_id: str
    same_organism_name: bool | None
    score: float
    identity: float
    coverage: float
    ambiguous_positions: int
    eligible: bool


@dataclass(frozen=True)
class MappedSite:
    reference_position: int
    reference_symbol: str
    status: str
    residue: Residue | None
    alternatives: tuple[int, ...]
    possible_gap: bool = False
    explicit: bool = False
    alternative_copies: tuple[tuple[str, tuple[int, ...], bool], ...] = ()


@dataclass(frozen=True)
class RoleMapping:
    status: str
    chain: Chain | None
    identity: float | None
    coverage: float | None
    sites: tuple[MappedSite, ...]
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class MappingResult:
    structure: Structure
    references: dict[str, ReferenceSequence]
    roles: dict[str, RoleMapping]
    settings: dict = field(default_factory=dict)


@lru_cache(maxsize=2)
def _substitution(molecule):
    if molecule == "protein":
        matrix = substitution_matrices.load("BLOSUM62")
        return matrix.alphabet, np.asarray(matrix), -10.0, -0.5
    # T and U are normalized only within the chosen nucleic-acid polymer type.
    alphabet = "ACGTN"
    matrix = np.full((5, 5), -2.0)
    np.fill_diagonal(matrix, 2)
    matrix[4, :] = matrix[:, 4] = 0
    return alphabet, matrix, -4.0, -1.0


def _encoded(sequence, alphabet):
    unknown = "X" if "X" in alphabet else "N"
    return np.array([alphabet.index(c if c in alphabet else unknown) for c in sequence], dtype=int)


def _forward(sub, opened, extended):
    """Global affine DP. States consume a paired site, a reference-only site or a chain-only site."""
    n, m = sub.shape
    val = np.full((3, n + 1, m + 1), -np.inf)
    val[0, 0, 0] = 0
    val[1, 1:, 0] = opened + extended * np.arange(n)
    val[2, 0, 1:] = opened + extended * np.arange(m)
    for diagonal in range(2, n + m + 1):
        i = np.arange(max(1, diagonal - m), min(n, diagonal - 1) + 1)
        j = diagonal - i
        val[0, i, j] = np.max(val[:, i - 1, j - 1], axis=0) + sub[i - 1, j - 1]
        val[1, i, j] = np.maximum.reduce(
            (val[0, i - 1, j] + opened, val[1, i - 1, j] + extended, val[2, i - 1, j] + opened)
        )
        val[2, i, j] = np.maximum.reduce(
            (val[0, i, j - 1] + opened, val[1, i, j - 1] + opened, val[2, i, j - 1] + extended)
        )
    return val


def _optimal_segment(sub, opened, extended):
    """For every reference site, find all target coordinates and gaps on any optimal path."""
    n, m = sub.shape
    if not n:
        return float(opened + extended * (m - 1) if m else 0), []
    if not m:
        return float(opened + extended * (n - 1)), [((), True)] * n
    forward = _forward(sub, opened, extended)
    reverse = _forward(sub[::-1, ::-1], opened, extended)[:, ::-1, ::-1]
    optimum = float(forward[:, -1, -1].max())
    # At a match, the suffix starts a new run. A gap followed by another gap of the
    # same kind replaces its opening penalty with an extension in the reversed suffix.
    continuation = reverse.max(axis=0)
    through_match = (forward[0] + continuation) == optimum
    gap_continuation = np.maximum(continuation, reverse[1] + extended - opened)
    through_gap = np.any(forward[1] + gap_continuation == optimum, axis=1)
    options = [
        (tuple(int(k) for k in np.flatnonzero(through_match[i, 1:]) + 1), bool(through_gap[i]))
        for i in range(1, n + 1)
    ]
    return optimum, options


@lru_cache(maxsize=24)
def _align(reference, polymer, molecule, anchors=()):
    if (len(reference) + 1) * (len(polymer) + 1) > 4_000_000:
        raise ValueError(
            "full polymer alignment exceeds four million cells; supply a domain-specific reference and structure"
        )
    alphabet, table, opened, extended = _substitution(molecule)
    if molecule != "protein":
        reference = reference.replace("U", "T")
        polymer = polymer.replace("U", "T")
    x = _encoded(reference, alphabet)
    y = _encoded(polymer, alphabet)
    sub = table[np.ix_(x, y)]
    score, options = _optimal_segment(sub, opened, extended)
    if anchors:
        constrained = []
        score_with_anchors = 0.0
        a = b = 0
        for i, j in (*anchors, (len(x) + 1, len(y) + 1)):
            if not (a < i <= len(x) + 1 and b < j <= len(y) + 1):
                raise ValueError(
                    "explicit residue anchors must follow polymer order without overlap"
                )
            segment, choices = _optimal_segment(sub[a : i - 1, b : j - 1], opened, extended)
            score_with_anchors += segment
            constrained.extend((tuple(t + b for t in options), gap) for options, gap in choices)
            if i <= len(x):
                constrained.append(((j,), False))
                score_with_anchors += sub[i - 1, j - 1]
            a, b = i, j
        if score_with_anchors != score:
            raise ValueError(
                "explicit residue anchors conflict with the optimal full-polymer sequence correspondence"
            )
        options = constrained
    canonical = set("ACDEFGHIKLMNPQRSTVWY" if molecule == "protein" else "ACGT")
    comparisons = [
        (reference[i], polymer[js[0] - 1])
        for i, (js, gap) in enumerate(options)
        if not gap
        and len(js) == 1
        and reference[i] in canonical
        and polymer[js[0] - 1] in canonical
    ]
    identity = sum(a == b for a, b in comparisons) / len(comparisons) if comparisons else 0.0
    coverage = len(comparisons) / len(reference)
    return score, tuple(options), identity, coverage


def map_components(
    references: Sequence[ReferenceSequence] | Mapping[str, ReferenceSequence],
    structure: Structure,
    *,
    chain_ids: Mapping[str, str] | None = None,
    explicit_map: Mapping[str, Mapping[int, int]] | None = None,
    min_identity: float = 0.3,
    min_coverage: float = 0.7,
) -> MappingResult:
    """Choose a unique supported copy per role, retaining exact optimal correspondence ambiguity."""
    items = tuple(references.values() if isinstance(references, Mapping) else references)
    refs = {r.component: r for r in items}
    if len(items) != 3 or set(refs) != set("ABC"):
        raise ValueError("supply exactly one complete reference for each of A, B, C")
    if not (0 <= min_identity <= 1 and 0 < min_coverage <= 1):
        raise ValueError("invalid identity or coverage threshold")
    chain_ids = dict(chain_ids or {})
    explicit_map = dict(explicit_map or {})
    if (set(chain_ids) | set(explicit_map)) - set(refs) or len(set(chain_ids.values())) != len(
        chain_ids
    ):
        raise ValueError("invalid or duplicate component chain selection")
    by_id = {c.copy_id: c for c in structure.chains}
    roles = {}
    for comp, ref in refs.items():
        wanted = chain_ids.get(comp)
        if wanted is not None and (wanted not in by_id or by_id[wanted].molecule != ref.molecule):
            raise ValueError(
                f"{comp} selected an absent or molecule-incompatible chain copy: {wanted}"
            )
        by_pos = {p: i for i, p in enumerate(ref.positions, start=1)}
        supplied = explicit_map.get(comp, {})
        if supplied and not wanted:
            raise ValueError("explicit residue anchors require an explicit chain copy")
        if any(
            type(p) is not int
            or p not in by_pos
            or type(j) is not int
            or not 1 <= j <= len(by_id[wanted].residues)
            for p, j in supplied.items()
        ):
            raise ValueError("explicit residue correspondence outside reference or polymer bounds")
        anchors = tuple(sorted((by_pos[p], j) for p, j in supplied.items()))
        candidates = []
        per_chain = {}
        for chain in structure.chains:
            if chain.molecule != ref.molecule:
                continue
            score, options, identity, coverage = _align(
                ref.sequence,
                chain.sequence,
                ref.molecule,
                anchors if chain.copy_id == wanted else (),
            )
            candidate = Candidate(
                chain.copy_id,
                chain.entity_id,
                chain.organism,
                chain.taxonomy_id,
                ref.organism.casefold() == chain.organism.casefold()
                if ref.organism and chain.organism
                else None,
                score,
                identity,
                coverage,
                sum(len(vals) > 1 or bool(vals and gap) for vals, gap in options),
                score > 0 and identity >= min_identity and coverage >= min_coverage,
            )
            candidates.append(candidate)
            per_chain[chain.copy_id] = options
        candidates.sort(key=lambda c: (-c.score, c.copy_id))
        good = [c for c in candidates if c.eligible and (not wanted or c.copy_id == wanted)]
        best = [c for c in good if c.score == good[0].score] if good else []
        choice = best[0] if len(best) == 1 else None
        chain = by_id[choice.copy_id] if choice else None
        status = (
            "mapped" if chain else ("ambiguous_chain" if best else "insufficient_sequence_evidence")
        )
        sites = []
        for i, (p, letter) in enumerate(zip(ref.positions, ref.sequence, strict=True)):
            opts, gap = per_chain[chain.copy_id][i] if chain else ((), False)
            residue = chain.residues[opts[0] - 1] if chain and len(opts) == 1 and not gap else None
            if not chain:
                local_status = status
            elif residue is None:
                local_status = "ambiguous_alignment" if opts else "reference_insertion"
            elif not residue.atoms:
                local_status = "missing_coordinates"
            else:
                local_status = "mapped"
            alternative_copies = (
                tuple((candidate.copy_id, *per_chain[candidate.copy_id][i]) for candidate in best)
                if len(best) > 1
                else ()
            )
            sites.append(
                MappedSite(
                    p, letter, local_status, residue, opts, gap, p in supplied, alternative_copies
                )
            )
        roles[comp] = RoleMapping(
            status,
            chain,
            choice.identity if choice else None,
            choice.coverage if choice else None,
            tuple(sites),
            tuple(candidates),
        )
    for comp, role in list(roles.items()):
        if role.chain and sum(other.chain is role.chain for other in roles.values()) > 1:
            # A single physical chain cannot satisfy two components. Withhold all implicated roles.
            for name, other in list(roles.items()):
                if other.chain is role.chain:
                    sites = tuple(
                        MappedSite(
                            s.reference_position, s.reference_symbol, "shared_chain", None, ()
                        )
                        for s in other.sites
                    )
                    roles[name] = RoleMapping(
                        "shared_chain", None, None, None, sites, other.candidates
                    )
    return MappingResult(
        structure,
        refs,
        roles,
        {
            "protein_substitution": "BLOSUM62",
            "protein_gap_open": -10,
            "protein_gap_extend": -0.5,
            "nucleotide_match": 2,
            "nucleotide_mismatch": -2,
            "nucleotide_unknown": 0,
            "nucleotide_gap_open": -4,
            "nucleotide_gap_extend": -1,
            "mode": "global affine on full declared polymers",
            "selection": "Positive global score plus both support thresholds; exactly tied best eligible copies require explicit selection.",
            "identity": "fraction identical among uniquely aligned canonical residue pairs",
            "coverage": "uniquely aligned canonical pairs / full reference length",
            "min_identity": min_identity,
            "min_coverage": min_coverage,
            "chain_ids": chain_ids,
            "explicit_anchors": explicit_map,
        },
    )
