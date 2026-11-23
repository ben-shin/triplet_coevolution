# reweight redundant homs using canon seq id

from dataclasses import dataclass
from math import isqrt, prod

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


_CACHE_BYTES = 32 * 1024**2
_BYTE_COUNTS = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def _pack(flags):
    """One bit per aligned column, stored as word-by-row planes."""
    packed = np.packbits(flags, axis=1)
    extra = -packed.shape[1] % 8
    if extra:
        packed = np.pad(packed, ((0, 0), (0, extra)))
    return np.ascontiguousarray(packed.view(np.uint64).T)


def _counts(words):
    if hasattr(np, "bitwise_count"):
        counts = np.bitwise_count(words)
    else:  # NumPy 1.26–1.x
        octets = np.ascontiguousarray(words).view(np.uint8).reshape(*words.shape, 8)
        counts = _BYTE_COUNTS[octets].sum(axis=-1, dtype=np.uint8)
    return counts[0] if counts.shape[0] == 1 else counts.sum(axis=0, dtype=np.int64)


class _ComponentComparisons:
    def __init__(self, component):
        matrix = component.matrix
        q = len(CANONICAL[component.molecule])
        valid = (matrix >= 0) & (matrix < q)
        if not np.all(valid.any(axis=1)):
            raise ValueError(f"component {component.name} has seqs w no canonical res")
        self.valid = _pack(valid)
        self.bits = [_pack((matrix & (1 << k)) != 0) for k in range((q - 1).bit_length())]
        self.length = component.n_sites
        self.dtype = np.min_scalar_type(self.length)
        self.cache = None

    def compare(self, left, right):
        """Exact number of matching and jointly canonical positions per row pair."""
        both = self.valid[:, left, None] & self.valid[:, None, right]
        equal = both.copy()
        for bit in self.bits:
            equal &= ~(bit[:, left, None] ^ bit[:, None, right])
        return _counts(equal), _counts(both)


class _WeightComparisons:
    """Reuse per-component identities across rematches, subject to a memory budget."""

    def __init__(self, components, *, reuse=False):
        self.parts = [_ComponentComparisons(c) for c in components]
        self.n = components[0].n_sequences
        self.names = tuple(c.name for c in components)
        self.indices = (
            [{name: i for i, name in enumerate(c.ids)} for c in components] if reuse else []
        )
        self.n_components = len(components)
        max_denominator = self.n_components * prod(c.n_sites for c in components)
        self.dtype = np.int64 if max_denominator <= 2**53 else object
        # Limit each 3-D bitwise temporary to 131072 words (or a single pair
        # for an exceptionally long alignment). No N-by-N workspace is required.
        words = max(part.valid.shape[0] for part in self.parts)
        self.block = max(1, min(256, isqrt(131072 // words)))
        cache_bytes = 2 * self.n**2 * sum(part.dtype.itemsize for part in self.parts)
        if reuse and cache_bytes <= _CACHE_BYTES:
            for part in self.parts:
                part.cache = np.empty((2, self.n, self.n), dtype=part.dtype)
                for left, right in self.blocks():
                    matches, overlap = part.compare(left, right)
                    part.cache[0, left, right] = matches
                    part.cache[1, left, right] = overlap
                    if left.start != right.start:
                        part.cache[0, right, left] = matches.T
                        part.cache[1, right, left] = overlap.T

    def blocks(self):
        for start in range(0, self.n, self.block):
            left = slice(start, min(start + self.block, self.n))
            for other in range(start, self.n, self.block):
                yield left, slice(other, min(other + self.block, self.n))

    def weights(self, identity_threshold, min_overlap, components=None):
        orders = (
            [
                np.fromiter((lookup[key] for key in c.ids), dtype=np.intp, count=self.n)
                for lookup, c in zip(self.indices, components, strict=True)
            ]
            if components is not None
            else [None] * self.n_components
        )
        neighbors = np.zeros(self.n, dtype=np.int64)
        for left, right in self.blocks():
            shape = (left.stop - left.start, right.stop - right.start)
            numerator = np.zeros(shape, dtype=self.dtype)
            denominator = np.ones(shape, dtype=self.dtype)
            eligible = np.ones(shape, dtype=bool)
            for part, order in zip(self.parts, orders, strict=True):
                x, y = (left, right) if order is None else (order[left], order[right])
                if part.cache is None:
                    matched, overlap = part.compare(x, y)
                else:
                    ix = np.ix_(x, y) if order is not None else (x, y)
                    matched, overlap = part.cache[0][ix], part.cache[1][ix]
                # Add exact fractions. Rounding each component before taking
                # the mean changes classifications at inclusive thresholds.
                numerator = numerator * overlap + matched * denominator
                denominator *= overlap
                eligible &= (overlap > 0) & (overlap / part.length >= min_overlap)
            eligible[eligible] = (
                numerator[eligible] / (denominator[eligible] * self.n_components)
                >= identity_threshold
            )
            if left.start == right.start:
                np.fill_diagonal(eligible, True)
            else:
                neighbors[right] += eligible.sum(axis=0)
            neighbors[left] += eligible.sum(axis=1)
        weights = 1.0 / neighbors
        weights.setflags(write=False)
        neighbors.setflags(write=False)
        return SequenceWeightResult(
            weights,
            neighbors,
            float(weights.sum()),
            self.n,
            self.names,
            identity_threshold,
            min_overlap,
        )


def sequence_weights(
    msa: ComponentMSA | MatchedTripletMSA,
    identity_threshold: float = 0.8,
    min_overlap: float = 0.5,
) -> SequenceWeightResult:
    """Weight each observation by the reciprocal count of identity neighbors.

    Every component must meet the overlap threshold on jointly canonical sites.
    Component identities contribute equally and each row counts as its own neighbor.
    """
    identity_threshold = _check_fraction(identity_threshold, "identity_threshold")
    min_overlap = _check_fraction(min_overlap, "min_overlap")
    if isinstance(msa, MatchedTripletMSA):
        components = msa.components
    elif isinstance(msa, ComponentMSA):
        components = (msa,)
    else:
        raise TypeError("sequence_weights expects a ComponentMSA or MatchedTripletMSA")
    return _WeightComparisons(components).weights(identity_threshold, min_overlap)
