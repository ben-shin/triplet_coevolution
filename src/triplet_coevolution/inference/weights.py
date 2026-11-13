# reweight redundant homs using canon seq id

from dataclasses import dataclass

import numpy as np

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA
from triplet_coevolution.core.alphabets import CANONICAL


@dataclass(frozen=True)
class SequenceWeightResult:
    """weights and eff depth for aln evol obs"""

    weights: np.ndarray
    neighbor_counts: np.ndarray
    n_eff: float
    n_observations: int
    components: tuple[str, ...]
    identity_threshold: float
    min_overlap: float

def _check_fraction(value: float, name: str) -> float:
    """check thresholds are finite fractions from 0 to 1"""

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a finite fraction between 0 and 1")

    return float(value)

def sequence_weights(
    msa: ComponentMSA | MatchedTripletMSA,
    identity_threshold: float = 0.8,
    min_overlap: float = 0.5,
) -> SequenceWeightResult:
    """"calculate identity based weights for one msa or matched triplets
    compare canon res only, ignore gaps/ambig
    each comp contributes equally to triplet identity
    each comp must hav eneough jointly observed canon cols
    a seq counts as its owen neightbor even if its coverage is low
    n_eff is a sum of weights not phylogenetic independence
    """

    identity_threshold = _check_fraction(identity_threshold, "identity_threshold")
    min_overlap = _check_fraction(min_overlap, "min_overlap")

    if isinstance(msa, MatchedTripletMSA):
        components = msa.components
        n = msa.n_observations
    elif isinstance(msa, ComponentMSA):
        components = (msa,)
        n = msa.n_sequences
    else:
        raise TypeError("sequence_weights expects a ComponentMSa or MatchedTripletMSA")

    # require at least one idable res per seq in every comp
    arrays = []
    for component in components:
        matrix = component.matrix
        q = len(CANONICAL[component.molecule])
        valid = (matrix >= 0) & (matrix < q)

        if np.any(valid.sum(axis=1) == 0):
            raise ValueError(
                f"component {component.name} has seqs w no canon res"
            )

        # joint overlap is a fraction of the full filtered msa length
        required = max(1, int(np.ceil(min_overlap * component.n_sites)))
        arrays.append((matrix, valid, required))

    neighbor_counts = np.zeros(n, dtype=np.int64)

    # compare one obs w all others at a time to avoid massive NxN storage
    for i in range(n):
        mean_identity = np.zeros(n, dtype=np.float64)
        enough_overlap = np.ones(n, dtype=bool)

        for matrix, valid, required in arrays:
            both = valid & valid[i]
            overlap = both.sum(axis=1)
            matched = np.count_nonzero(both & (matrix == matrix[i]), axis=1)

            identity = np.zeros(n, dtype=np.float64)
            np.divide(matched, overlap, out=identity, where=overlap > 0)

            mean_identity += identity
            enough_overlap &= overlap >= required

        mean_identity /= len(arrays)
        neighbors = enough_overlap & (mean_identity >= identity_threshold)

        # a seq always belongs to its own redundant cluster
        neighbors[i] = True
        neighbor_counts[i] = np.count_nonzero(neighbors)

    weights = 1.0 / neighbor_counts.astype(np.float64)
    n_eff = float(weights.sum())

    weights.setflags(write=False)
    neighbor_counts.setflags(write=False)

    return SequenceWeightResult(
        weights=weights,
        neighbor_counts=neighbor_counts,
        n_eff=n_eff,
        n_observations=n,
        components=tuple(component.name for component in components),
        identity_threshold=identity_threshold,
        min_overlap=min_overlap,
    )
