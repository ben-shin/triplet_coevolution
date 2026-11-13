# weighted mi and apc for matched msa comps

from dataclasses import dataclass

import numpy as np
from numpy.ma.extras import isin
from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.weights import SequenceWeightResult


@dataclass(frozen=True)
class MIBlock:
    """mi and apc scores for one pair of aligned comps"""

    component_x: str
    component_y: str
    mi_bits: np.ndarray
    mi_apc: np.ndarray
    n_valid: np.ndarray
    valid_weight: np.ndarray


@dataclass(frozen=True)
class TripletMI:
    """ab/ac/bc scores from the same matched triplet msa"""

    ab: MIBlock
    ac: MIBlock
    bc: MIBlock
    n_observations: int
    weight_sum: float
    include_gaps: bool
    min_valid: int


def _check_min_valid(min_valid: int) -> int:
    if isinstance(min_valid, (bool, np.bool_)) or not isinstance(min_valid, (int, np.integer)):
        raise ValueError("min_valid must be a positive integer")
    if min_valid < 1:
        raise ValueError("min_valid must be a positive integer")
    return int(min_valid)


def _get_weights(
    weights: SequenceWeightResult | np.ndarray | list[float] | None,
    n: int,
) -> np.ndarray:
    """validate one weight per aligned obs; zero weights exclude rows"""

    if isinstance(weights, SequenceWeightResult):
        if weights.n_observations != n:
            raise ValueError("weights must matched the number of aln obs")
        weights = weights.weights

    if weights is None:
        return np.ones(n, dtype=np.float64)

    values = np.asarray(weights)
    if values.ndim != 1 or values.shape[0] != n:
        raise ValueError("weights must match the number of aln obs")

    if values.dtype.kind not in "iuf":
        raise TypeError("weights must be numeric and not boolean")

    result = values.astype(np.float64)
    if not np.isfinite(result).all() or np.any(result < 0) or not np.any (result > 0):
        raise ValueError("weights must be finite, nonnegative, and not all zero")

    return result


def _site_mi(
    x: np.ndarray,
    y: np.ndarray,
    q_x: int,
    q_y: int,
    weights: np.ndarray,
    min_valid: int,
) -> tuple[float, int, float]:
    """calculate mi from joint obs only, renormalizing their weights"""

    # missing and excl gaps codes have no contribution
    valid = (
        (x >= 0) & (x < q_x)
        & (y >= 0) & (y < q_y)
        & (weights > 0)
    )
    n_valid = int(np.count_nonzero(valid))
    valid_weight = float(weights[valid].sum())

    if n_valid < min_valid:
        return np.nan, n_valid, valid_weight

    # count weighted join states and normalize by pair specific weight
    indices = x[valid].astype(np.int64) * q_y + y[valid]
    counts = np.bincount(
        indices,
        weights=weights[valid],
        minlength=q_x * q_y,
    ).reshape(q_x, q_y)

    joint = counts / valid_weight
    expected = np.outer(joint.sum(axis=1), joint.sum(axis=0))
    observed = joint > 0
    mi = float(np.sum(joint[observed] * np.log2(joint[observed] / expected[observed])))

    return max(mi, 0.0), n_valid, valid_weight


def weighted_mi(
    x: np.ndarray,
    y: np.ndarray,
    n_states_x: int,
    n_states_y: int,
    weights: SequenceWeightResult | np.ndarray | list[float] | None = None,
    min_valid: int = 2
) -> float:
    """calculate weighted mi in bits for 2 encoded aln cols
    states >= n_states_x or n_states_y are excluded
    caller defines whether gap states are incl in those counts
    unknown/ambig states always excl"""

    x = np.asarray(x)
    y = np.asarray(y)
    min_valid = _check_min_valid(min_valid)

    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("encoded cols must be 1d and equal in length")
    if x.dtype.kind not in "iu" or y.dtype.kind not in "iu":
        raise TypeError("encoded cols must contain int states")

    for name, count in (("n_states_x", n_states_x), ("n_states_y", n_states_y)):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError(f"{name} must be a positive int")

    row_weights = _get_weights(weights, x.size)
    return _site_mi(x, y, n_states_x, n_states_y, row_weights, min_valid)[0]


def apc(scores: np.ndarray) -> np.ndarray:
    """average product correction for a single mi matrix
    missing score cells stay NaN, zero global mean gives no correction
    apc can be negative. not directly p-val interpretable"""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or 0 in scores.shape:
        raise ValueError("apc expects nonempty 2d matrix")
    if np.isinf(scores).any() or np.any(scores[np.isfinite(scores)] < 0):
        raise ValueError("apc expects nonneg mi scores or nan")

    valid = np.isfinite(scores)
    result = np.full_like(scores, np.nan)
    if not valid.any():
        result.setflags(write=False)
        return result

    values = np.where(valid, scores, 0.0)
    n_row = valid.sum(axis=1)
    n_col = valid.sum(axis=0)

    row_mean = np.divide(
        values.sum(axis=1), n_row,
        out=np.full(scores.shape[0], np.nan), where=n_row > 0,
    )
    col_mean = np.divide(
        values.sum(axis=0), n_col,
        out=np.full(scores.shape[1], np.nan), where=n_col > 0,
    )
    global_mean = float(values.sum() / valid.sum())

    if global_mean == 0:
        result[valid] = 0.0
    else:
        correction = np.outer(row_mean, col_mean) / global_mean
        result[valid] = (scores - correction)[valid]

    result.setflags(write=False)
    return result

def mutual_information_block(
    a: ComponentMSA,
    b: ComponentMSA,
    weights: SequenceWeightResult | np.ndarray | list[float] | None = None,
    *,
    include_gaps: bool = False,
    min_valid: int = 2,
) -> MIBlock:
    """calculate per col mi for two already row matched msas
    caller must ensure the two msas contain corresp obs in row order
    gaps excluded, use include_gaps=True to count them
    all neg/ambig symbols always treated as missing"""

    if not isinstance(a, ComponentMSA) or not isinstance(b, ComponentMSA):
        raise TypeError("mutual_information_block expects two ComponentMSA objs")
    if a.n_sequences != b.n_sequences:
        raise ValueError("both msas must have the same number of matched obs")
    if not isinstance(include_gaps, (bool, np.bool_)):
        raise TypeError("include_gaps must be boolean")

    min_valid = _check_min_valid(min_valid)
    row_weights = _get_weights(weights, a.n_sequences)
    q_a = len(CANONICAL[a.molecule] + int(include_gaps))
    q_b = len(CANONICAL[b.molecule] + int(include_gaps))

    mi = np.full((a.n_sites, b.n_sites), np.nan, dtype=np.float64)
    n_valid = np.zeros(mi.shape, dtype=np.int64)
    valid_weight = np.zeros(mi.shape, dtype=np.float64)

    for i in range(a.n_sites):
        for j in range(b.n_sites):
            mi[i, j], n_valid[i, j], valid_weight[i, j] = _site_mi(
                a.matrix[:, i], b.matrix[:, j], q_a, q_b, row_weights, min_valid,
            )

    mi_apc = apc(mi)

    for array in (mi, n_valid, valid_weight):
        array.setflags(write=False)
    return MIBlock(
        component_x=a.name,
        component_y=b.name,
        mi_bits=mi,
        mi_apc=mi_apc,
        n_valid=n_valid,
        valid_weight=valid_weight,
    )

def triplet_mi(
    msa: MatchedTripletMSA,
    weights: SequenceWeightResult | np.ndarray | list[float] | None = None,
    *,
    include_gaps: bool = False,
    min_valid: int = 2,
) -> TripletMI:
    """calculate ab/ac/bc mi on the same synced triplet obs"""

    if not isinstance(msa, MatchedTripletMSA):
        raise TypeError("triplet_mi expects a MatchedTripletMSA")
    if isinstance(weights, SequenceWeightResult) and weights.components != ("A", "B", "C"):
        raise ValueError("triplet weights must come from all three comps")

    row_weights = _get_weights(weights, msa.n_observations)
    ab = mutual_information_block(
        msa.a, msa.b, row_weights, include_gaps=include_gaps, min_valid=min_valid
    )
    ac = mutual_information_block(
        msa.a, msa.c, row_weights, include_gaps=include_gaps, min_valid=min_valid
    )
    bc = mutual_information_block(
        msa.b, msa.c, row_weights, include_gaps=include_gaps, min_valid=min_valid
    )

    return TripletMI(
        ab=ab,
        ac=ac,
        bc=bc,
        n_observations=msa.n_observations,
        weight_sum=float(row_weights.sum()),
        include_gaps=include_gaps,
        min_valid=min_valid,
    )
