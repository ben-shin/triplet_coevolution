"""Three-site categorical models and score-independent candidate selection."""

from dataclasses import dataclass
from math import prod
from random import Random

import numpy as np

from triplet_coevolution.core.alignment import MatchedTripletMSA
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.mi import _check_min_valid, _get_weights


@dataclass(frozen=True)
class TripleScore:
    threebody_rms: float
    co_information_bits: float
    n_valid: int
    valid_weight: float
    observed_cells: int
    possible_cells: int


def sample_candidates(axes, maximum: int, seed: int):
    """Sample unique triples uniformly without building the Cartesian product."""
    if maximum < 1:
        raise ValueError("max_triplets must be positive")
    lengths = tuple(len(axis) for axis in axes)
    total = prod(lengths)
    indices = (
        range(total) if total <= maximum else sorted(Random(seed).sample(range(total), maximum))
    )
    triples = []
    for index in indices:
        ab, c = divmod(index, lengths[2])
        a, b = divmod(ab, lengths[1])
        triples.append((axes[0][a], axes[1][b], axes[2][c]))
    return triples, total


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def fit_triples(
    msa: MatchedTripletMSA,
    candidates,
    weights=None,
    *,
    pseudocount: float = 0.5,
    min_valid: int = 2,
) -> list[TripleScore]:
    """Fit a smoothed saturated log-linear model to each selected ABC marginal."""
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("three-body pseudocount must be finite and positive")
    min_valid = _check_min_valid(min_valid)
    w = _get_weights(weights, msa.n_observations)
    matrices = tuple(c.matrix for c in msa.components)
    alphabets = tuple(len(CANONICAL[c.molecule]) for c in msa.components)
    results = []
    for triple in candidates:
        if len(triple) != 3 or any(
            not isinstance(j, (int, np.integer))
            or isinstance(j, (bool, np.bool_))
            or not 0 <= j < matrix.shape[1]
            for j, matrix in zip(triple, matrices)
        ):
            raise ValueError("triple candidates must be three valid zero-based column indices")
        columns = [matrix[:, j] for j, matrix in zip(triple, matrices)]
        valid = w > 0
        for x, q in zip(columns, alphabets):
            valid &= (x >= 0) & (x < q)
        n = int(valid.sum())
        weight = float(w[valid].sum())
        if n == 0:
            results.append(TripleScore(np.nan, np.nan, 0, 0.0, 0, 0))
            continue
        recoded = [np.unique(x[valid], return_inverse=True)[1] for x in columns]
        shape = tuple(int(x.max()) + 1 for x in recoded)
        index = np.ravel_multi_index(recoded, shape)
        counts = np.bincount(index, weights=w[valid], minlength=prod(shape)).reshape(shape)
        observed = int(np.count_nonzero(counts))
        if n < min_valid:
            results.append(TripleScore(np.nan, np.nan, n, weight, observed, counts.size))
            continue

        # The three commuting zero-sum projections remove every intercept,
        # main effect and pair term from log p(a,b,c).
        interaction = np.log(counts + pseudocount)
        for axis in range(3):
            interaction -= interaction.mean(axis=axis, keepdims=True)
        rms = float(np.sqrt(np.mean(interaction**2)))
        p = counts / weight
        singles = sum(_entropy(p.sum(axis=axes)) for axes in ((0, 1), (0, 2), (1, 2)))
        pairs = sum(_entropy(p.sum(axis=axis)) for axis in range(3))
        co_info = singles - pairs + _entropy(p)
        results.append(TripleScore(rms, co_info, n, weight, observed, counts.size))
    return results
