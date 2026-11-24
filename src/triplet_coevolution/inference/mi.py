# weighted mi and apc for matched msa comps

from dataclasses import dataclass
from math import frexp, fsum, isqrt
from typing import Literal

import numpy as np

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.weights import SequenceWeightResult, verify_weight_alignment

MIKernel = Literal["batched", "v1_scalar"]


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
    apc_universe: str = "retained"
    kernel: MIKernel = "batched"
    min_valid_weight: float = 0.0


def _check_min_valid(min_valid: int) -> int:
    if (
        isinstance(min_valid, (bool, np.bool_))
        or not isinstance(min_valid, (int, np.integer))
        or min_valid < 1
    ):
        raise ValueError("min_valid must be a positive integer")
    return int(min_valid)


def check_min_valid_weight(value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (float, int, np.floating, np.integer))
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError("min_valid_weight must be finite and nonnegative")
    return float(value)


def canonical_support_at_cutoff(support, valid, weights, threshold):
    """Use the correctly rounded sum near an inclusive cutoff across reduction kernels."""
    if threshold > 0 and abs(support - threshold) <= len(weights) * np.finfo(float).eps * threshold:
        return fsum(weights[valid])
    return support


def _pair_support_at_cutoff(support, valid_a, valid_b, weights, threshold):
    if threshold <= 0 or (weights.sum() <= 2**53 and np.all(weights == np.floor(weights))):
        return
    # BLAS dot and NumPy reductions can differ by an ulp on a threshold tie.
    near = np.abs(support - threshold) <= len(weights) * np.finfo(float).eps * threshold
    if not near.any():
        return
    active = weights > 0
    complete = valid_a[active].all(axis=0)[:, None] & valid_b[active].all(axis=0)[None, :] & near
    support[complete] = fsum(weights)
    near[complete] = False
    for i, j in zip(*np.nonzero(near)):
        support[i, j] = fsum(weights[valid_a[:, i] & valid_b[:, j]])


def _get_weights(
    weights: SequenceWeightResult | np.ndarray | list[float] | None,
    n: int,
    msa=None,
) -> np.ndarray:
    """validate one weight per aligned obs; zero weights exclude rows"""

    if isinstance(weights, SequenceWeightResult):
        if weights.n_observations != n:
            raise ValueError("weights must matched the number of aln obs")
        if msa is not None:
            verify_weight_alignment(msa,weights)
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

    with np.errstate(over="ignore"):
        if not np.isfinite(result.sum()):
            raise ValueError("sum of weights exceeds the numeric range")
    return result


def _site_mi(
    x: np.ndarray,
    y: np.ndarray,
    q_x: int,
    q_y: int,
    weights: np.ndarray,
    min_valid: int,
    min_valid_weight: float=0.0,
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

    valid_weight = canonical_support_at_cutoff(valid_weight, valid, weights, min_valid_weight)
    if n_valid < min_valid or valid_weight < min_valid_weight:
        return np.nan, n_valid, valid_weight

    # count weighted join states and normalize by pair specific weight
    indices = x[valid].astype(np.int64) * q_y + y[valid].astype(np.int64)
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
    min_valid: int = 2,
    *,
    min_valid_weight: float = 0.0,
) -> float:
    """calculate weighted mi in bits for 2 encoded aln cols
    states >= n_states_x or n_states_y are excluded
    caller defines whether gap states are incl in those counts
    unknown/ambig states always excl"""

    x = np.asarray(x)
    y = np.asarray(y)
    min_valid = _check_min_valid(min_valid)
    min_valid_weight = check_min_valid_weight(min_valid_weight)

    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("encoded cols must be 1d and equal in length")
    if x.dtype.kind not in "iu" or y.dtype.kind not in "iu":
        raise TypeError("encoded cols must contain int states")

    for name, count in (("n_states_x", n_states_x), ("n_states_y", n_states_y)):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError(f"{name} must be a positive int")

    row_weights = _get_weights(weights, x.size)
    return _site_mi(x, y, int(n_states_x), int(n_states_y), row_weights, min_valid,min_valid_weight)[0]


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

def _mi_from_counts(counts, weight):
    joint = counts / np.where(weight, weight, 1)[:, None, None]
    expected = joint.sum(axis=1, keepdims=True) * joint.sum(axis=2, keepdims=True)
    occupied = joint > 0
    terms = np.zeros_like(joint)
    np.divide(joint, expected, out=terms, where=occupied)
    np.log2(terms, out=terms, where=occupied)
    return np.maximum((joint * terms).sum(axis=(1, 2)), 0.0)


def _packed_unit_mi(a, b, q_a, q_b, row_weights, min_valid, mi, n_valid, valid_weight):
    """Exact low-alphabet contingency counts with a common power-of-two positive weight."""
    n = len(row_weights)
    words = (n + 63) // 64
    pair_words = words * q_a * q_b
    if (
        not hasattr(np, "bitwise_count")
        or n < 512
        or q_a * q_b > 25
        or pair_words > 131072
        or a.n_sites * b.n_sites < 64
    ):
        return False
    positive = row_weights > 0
    scale = float(row_weights.max())
    if frexp(scale)[0] != 0.5 or not np.all((row_weights == scale) | ~positive):
        return False
    # A power-of-two weight can be multiplied by the integer count exactly.
    # Source weight checks exclude overflow. Uniform non-dyadic weights keep
    # the original per-observation accumulation to preserve its rounding.
    tile = max(1, isqrt(131072 // pair_words))

    def planes(matrix, cols, q):
        values = matrix[:, cols].T
        packed = np.zeros((len(values), q, words), dtype=np.uint64)
        octets = packed.view(np.uint8)
        for state in range(q):
            bits = np.packbits((values == state) & positive, axis=1)
            octets[:, state, : bits.shape[1]] = bits
        return packed

    for i in range(0, a.n_sites, tile):
        x = slice(i, i + tile)
        left = planes(a.matrix, x, q_a)
        for j in range(0, b.n_sites, tile):
            y = slice(j, j + tile)
            right = planes(b.matrix, y, q_b)
            shared = left[:, None, :, None] & right[None, :, None, :]
            count = np.bitwise_count(shared).sum(axis=-1, dtype=np.int64)
            support = count.sum(axis=(-1, -2))
            weight = support * scale
            cells = count.reshape(-1, q_a, q_b) * scale
            score = _mi_from_counts(cells, weight.ravel()).reshape(support.shape)
            n_valid[x, y], valid_weight[x, y] = support, weight
            mi[x, y] = np.where(support >= min_valid, score, np.nan)
    return True


def _batched_mi(a, b, q_a, q_b, row_weights, min_valid, mi, n_valid, valid_weight):
    if _packed_unit_mi(a, b, q_a, q_b, row_weights, min_valid, mi, n_valid, valid_weight):
        return
    # Several column pairs share each bincount call. Each receives a separate
    # range of joint-state bins and one discarded bin for excluded observations.
    n = a.n_sequences
    states = q_a * q_b
    batch = max(1, min(b.n_sites, 65536 // max(n, states)))
    y = np.ascontiguousarray(b.matrix.T)
    y_valid = (y >= 0) & (y < q_b) & (row_weights > 0)
    offsets = (np.arange(batch, dtype=np.int64) * (states + 1))[:, None]
    tiled_weights = np.tile(row_weights, batch)
    for i, x in enumerate(a.matrix.T):
        x_valid = (x >= 0) & (x < q_a)
        x_index = x * q_b
        for start in range(0, b.n_sites, batch):
            stop = min(start + batch, b.n_sites)
            cols = slice(start, stop)
            size = stop - start
            valid = x_valid & y_valid[cols]
            n_valid[i, cols] = valid.sum(axis=1)
            weight = np.where(valid, row_weights, 0).sum(axis=1)
            valid_weight[i, cols] = weight
            indices = offsets[:size] + np.where(valid, x_index + y[cols], states)
            counts = np.bincount(
                indices.ravel(), weights=tiled_weights[: size * n], minlength=size * (states + 1)
            ).reshape(size, states + 1)
            bits = _mi_from_counts(counts[:, :states].reshape(size, q_a, q_b), weight)
            mi[i, cols] = np.where(n_valid[i, cols] >= min_valid, bits, np.nan)


def mutual_information_block(
    a: ComponentMSA,
    b: ComponentMSA,
    weights: SequenceWeightResult | np.ndarray | list[float] | None = None,
    *,
    include_gaps: bool = False,
    min_valid: int = 2,
    min_valid_weight: float = 0.0,
    kernel: MIKernel = "batched",
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

    if kernel not in ("batched", "v1_scalar"):
        raise ValueError("MI kernel must be batched or v1_scalar")
    min_valid = _check_min_valid(min_valid)
    min_valid_weight = check_min_valid_weight(min_valid_weight)
    row_weights = _get_weights(weights, a.n_sequences)
    q_a = len(CANONICAL[a.molecule]) + int(include_gaps)
    q_b = len(CANONICAL[b.molecule]) + int(include_gaps)

    mi = np.full((a.n_sites, b.n_sites), np.nan, dtype=np.float64)
    n_valid = np.zeros(mi.shape, dtype=np.int64)
    valid_weight = np.zeros(mi.shape, dtype=np.float64)

    if kernel == "v1_scalar":
        # v1 sums only occupied states and compacts the positive, valid weights
        # before reduction. Reordering those sums can change ranks near zero.
        for i in range(a.n_sites):
            for j in range(b.n_sites):
                mi[i, j], n_valid[i, j], valid_weight[i, j] = _site_mi(
                    a.matrix[:, i], b.matrix[:, j], q_a, q_b, row_weights, min_valid,min_valid_weight,
                )
    else:
        _batched_mi(a, b, q_a, q_b, row_weights, min_valid, mi, n_valid, valid_weight)

        if min_valid_weight:
            _pair_support_at_cutoff(
                valid_weight,
                (a.matrix >= 0) & (a.matrix < q_a),
                (b.matrix >= 0) & (b.matrix < q_b),
                row_weights,
                min_valid_weight,
            )
            mi[valid_weight < min_valid_weight] = np.nan

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
    min_valid_weight: float = 0.0,
    kernel: MIKernel = "batched",
) -> TripletMI:
    """calculate ab/ac/bc mi on the same synced triplet obs"""

    if not isinstance(msa, MatchedTripletMSA):
        raise TypeError("triplet_mi expects a MatchedTripletMSA")
    if isinstance(weights, SequenceWeightResult) and weights.components != ("A", "B", "C"):
        raise ValueError("triplet weights must come from all three comps")

    row_weights = _get_weights(weights, msa.n_observations,msa)
    ab = mutual_information_block(
        msa.a, msa.b, row_weights, include_gaps=include_gaps, min_valid=min_valid, min_valid_weight=min_valid_weight, kernel=kernel
    )
    ac = mutual_information_block(
        msa.a, msa.c, row_weights, include_gaps=include_gaps, min_valid=min_valid, min_valid_weight=min_valid_weight, kernel=kernel
    )
    bc = mutual_information_block(
        msa.b, msa.c, row_weights, include_gaps=include_gaps, min_valid=min_valid, min_valid_weight=min_valid_weight, kernel=kernel
    )

    return TripletMI(
        ab=ab,
        ac=ac,
        bc=bc,
        n_observations=msa.n_observations,
        weight_sum=float(row_weights.sum()),
        include_gaps=include_gaps,
        min_valid=min_valid,
        kernel=kernel,
        min_valid_weight=check_min_valid_weight(min_valid_weight),
    )
