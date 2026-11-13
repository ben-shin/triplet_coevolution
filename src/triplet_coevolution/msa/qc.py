# qc metrics for single and matched triplet msas

from dataclasses import dataclass
from re import T
from turtle import st

import numpy as np

from triplet_coevolution.core.alignment import (
    ComponentMSA,
    MatchedTripletMSA,
)
from triplet_coevolution.core.alphabets import CANONICAL, UNKNOWN_CODE, MoleculeType


@dataclass(frozen=True)
class ComponentQC:
    """descriptive qc results for one comp aln
    per site arrays follow original msa col order
    shannon entropy and conservation exclude gaps and ambigs"""

    name: str
    molecule: MoleculeType
    n_sequences: int
    n_sites: int
    n_canonical: np.ndarray
    gap_fraction: np.ndarray
    missing_fraction: np.ndarray
    coverage: np.ndarray
    entropy_bits: np.ndarray
    conservation: np.ndarray

    @property
    def mean_coverage(self) -> float:
        """mean fraction of canon res across msa cols"""
        return float(np.mean(self.coverage))

    @property
    def mean_gap_fraction(self) -> float:
        return float(np.mean(self.gap_fraction))

    @property
    def mean_missing_fraction(self) -> float:
        return float(np.mean(self.missing_fraction))

@dataclass(frozen=True)
class TripletQC:
    """qc results for the 3 synced aln comps"""

    a: ComponentQC
    b: ComponentQC
    c: ComponentQC
    n_observations: int

    @property
    def components(self) -> tuple[ComponentQC, ComponentQC, ComponentQC]:
        return (self.a, self.b, self.c)

    @property
    def n_sites(self) -> int:
        """total number of aln pos across a b c"""
        return self.a_n_sites + self.b_n_sites + self.c_n_sites

def component_qc(msa: ComponentMSA) -> ComponentQC:
    """calculate unweighted qc stats for each aln col
    canonical res: 0-19 for prot, 0-3 for rna/dna
    gaps: 20 for prot, 4 for rna/dna
    unknown/ambig: -1
    gap_fraction, missing_fraction, coverage use all seqs as denom
    entropy_bits and conservations ONLY use canonical res
    cols w 0 canonical res get np.nan"""

    if not isinstance(msa, ComponentMSA):
        raise TypeError("component_qc expects a ComponentMSA")

    matrix = msa.matrix
    n_sequences = msa.n_sequences
    n_sites = msa.n_sites
    q = len(CANONICAL[msa.molecule])

    # cound each canonical state at every aln col
    counts = np.stack(
        [np.count_nonzero(matrix == state, axis=0) for state in range(q)]
    )

    n_canonical = counts.sum(axis=0)
    n_gaps = np.count_nonzero(matrix == q, axis=0)
    n_missing = np.count_nonzero(matrix == UNKNOWN_CODE, axis = 0)

    # coverage, gaps, and unknown states parition each col
    coverage = n_canonical / n_sequences
    gap_fraction = n_gaps / n_sequences
    missing_fraction = n_missing / n_sequences

    # frequencies are conditional on a canon res being obs
    probabilities = np.zeros_like(counts, dtype=np.float64)
    np.divide(
        counts,
        n_canonical,
        out=probabilities,
        where=n_canonical > 0,
    )

    # define 0 * log2(0) as 0. no need to calcualte log2 on 0
    log_probabilities = np.zeros_like(probabilities)
    np.log2(
        probabilities,
        out=log_probabilities,
        where=probabilities > 0,
    )

    entropy_bits = -(probabilities * log_probabilites).sum(axis=0)
    entropy_bits[n_canonical == 0] = np.nan

    #fraction of canon obs carrying most freq state
    conservation = np.full(n_sites, np.nan, dtype=np.float64)
