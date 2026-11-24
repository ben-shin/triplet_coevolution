"""Three-site categorical models and score-independent candidate selection."""

from dataclasses import dataclass
from functools import lru_cache
from math import prod
from random import Random

import numpy as np

from triplet_coevolution.core.alignment import MatchedTripletMSA
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.mi import (
    _check_min_valid,
    _get_weights,
    canonical_support_at_cutoff,
    check_min_valid_weight,
)


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


def _cache_sites(n):
    return min(4096, (2 * 1024**2) // (3 * n))


def triple_row_workspace(n, widths, candidates):
    """Bound cached int16 codes, canonical masks and per-candidate row temporaries in bytes."""
    if n <= 0 or candidates <= 0:
        return 0
    stored = min(_cache_sites(n), sum(widths), 3 * candidates)
    return n * (3 * stored + 80)


def fit_triples(
    msa: MatchedTripletMSA,
    candidates,
    weights=None,
    *,
    pseudocount: float = 0.5,
    min_valid: int = 2,
    min_valid_weight: float = 0.0,
) -> list[TripleScore]:
    """Fit a smoothed saturated log-linear model to each selected ABC marginal."""
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("three-body pseudocount must be finite and positive")
    min_valid = _check_min_valid(min_valid)
    min_valid_weight = check_min_valid_weight(min_valid_weight)
    w = _get_weights(weights, msa.n_observations, msa)
    matrices = tuple(c.matrix for c in msa.components)
    alphabets = tuple(len(CANONICAL[c.molecule]) for c in msa.components)
    full_cells = prod(alphabets)
    strides = (alphabets[1] * alphabets[2], alphabets[2], 1)
    # Cached int16 mixed-radix contributions and Boolean canonical masks only.
    # Cap both data bytes and Python cache entries, even for a generator of all triples.
    max_sites = _cache_sites(msa.n_observations)
    positive = w > 0

    @lru_cache(maxsize=max_sites)
    def site(role, column):
        values = matrices[role][:, column]
        return values * strides[role], (values >= 0) & (values < alphabets[role])

    results = []
    for triple in candidates:
        if len(triple) != 3 or any(
            not isinstance(j, (int, np.integer))
            or isinstance(j, (bool, np.bool_))
            or not 0 <= j < matrix.shape[1]
            for j, matrix in zip(triple, matrices)
        ):
            raise ValueError("triple candidates must be three valid zero-based column indices")
        a, b, c = (site(role, int(column)) for role, column in enumerate(triple))
        valid = positive & a[1] & b[1] & c[1]
        n = int(valid.sum())
        valid_weights = w[valid]
        weight = canonical_support_at_cutoff(float(valid_weights.sum()), valid, w, min_valid_weight)
        if n == 0:
            results.append(TripleScore(np.nan, np.nan, 0, 0.0, 0, 0))
            continue
        if full_cells <= max(512, 4 * n):
            # A bincount in canonical state order accumulates every occupied cell
            # in its original row order. Remove absent marginal states *after*
            # counting to preserve the local observed-state prior and gauge.
            index = (a[0] + b[0] + c[0])[valid]
            counts = np.bincount(index, weights=valid_weights, minlength=full_cells).reshape(
                alphabets
            )
            present = tuple(counts.any(axis=axes) for axes in ((1, 2), (0, 2), (0, 1)))
            if not all(mask.all() for mask in present):
                counts = counts[np.ix_(*present)]
        else:
            # Shallow, high-alphabet tables can have very sparse effective support.
            columns = [matrix[:, j] for j, matrix in zip(triple, matrices)]
            recoded = []
            for x, q in zip(columns, alphabets, strict=True):
                values = x[valid]
                if n < 64:
                    codes = np.unique(values, return_inverse=True)[1]
                else:
                    present = np.bincount(values, minlength=q) > 0
                    codes = (np.cumsum(present) - 1)[values]
                recoded.append(codes)
            shape = tuple(int(x.max()) + 1 for x in recoded)
            index = np.ravel_multi_index(recoded, shape)
            counts = np.bincount(index, weights=valid_weights, minlength=prod(shape)).reshape(shape)
        observed = int(np.count_nonzero(counts))
        if n < min_valid or weight < min_valid_weight:
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
