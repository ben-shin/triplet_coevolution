"""Exact redundancy weights and observation-keyed sampling designs for matched alignments."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import isqrt, prod
from typing import Literal

import numpy as np

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.msa.support import MissingRowSupport

WeightPolicy = Literal[
    "equal_component",
    "component",
    "concatenated",
    "cluster_representatives",
    "clade_balanced",
    "supplied",
    "uniform",
]
POLICIES = (
    "equal_component",
    "component",
    "concatenated",
    "cluster_representatives",
    "clade_balanced",
    "supplied",
    "uniform",
)


@dataclass(frozen=True)
class SequenceWeightResult:
    """Weights in cohort order, their absolute sum, and policy-specific evidence."""

    weights: np.ndarray
    neighbor_counts: np.ndarray | None
    n_eff: float
    n_observations: int
    components: tuple[str, ...]
    identity_threshold: float
    min_overlap: float
    observation_ids: tuple[str, ...] = ()
    policy: WeightPolicy = "equal_component"
    component: str | None = None
    cluster_ids: tuple[str, ...] = ()
    clades: tuple[str, ...] = ()
    preflight: dict = field(default_factory=dict)

    @property
    def kish(self) -> float:
        """Kish count is invariant to the absolute weight scale."""
        scaled = self.weights / self.weights.max()
        return float(scaled.sum() ** 2 / np.dot(scaled, scaled))

    def table_rows(self):
        """Emit stable observation identities and actual policy-specific support."""
        for i, key in enumerate(self.observation_ids):
            yield {
                "observation_id": key,
                "weight": float(self.weights[i]),
                "neighbor_count": int(self.neighbor_counts[i])
                if self.neighbor_counts is not None
                else None,
                "cluster_id": self.cluster_ids[i] if self.cluster_ids else None,
                "clade": self.clades[i] if self.clades else None,
            }

    def summary(self) -> dict:
        """JSON-ready global support, including the scale and computational definition."""
        neighbor_distribution = (
            Counter(map(int, self.neighbor_counts)) if self.neighbor_counts is not None else None
        )
        return {
            "policy": self.policy,
            "component": self.component,
            "sum_weights": self.n_eff,
            "kish": self.kish,
            "observations": self.n_observations,
            "positive_weight_rows": int(np.count_nonzero(self.weights)),
            "identity_threshold": self.identity_threshold
            if self.neighbor_counts is not None
            else None,
            "min_overlap": self.min_overlap if self.neighbor_counts is not None else None,
            "neighbor_count_distribution": {
                str(k): v for k, v in sorted(neighbor_distribution.items())
            }
            if neighbor_distribution is not None
            else None,
            "cluster_sizes": dict(sorted(Counter(self.cluster_ids).items()))
            if self.cluster_ids
            else None,
            "clade_sizes": dict(sorted(Counter(self.clades).items())) if self.clades else None,
            "definition": _DEFINITIONS[self.policy],
            "workspace": self.preflight,
        }


_DEFINITIONS = {
    "equal_component": "Reciprocal self-inclusive neighbor count. Every component passes canonical overlap; arithmetic mean of component identities meets the inclusive cutoff.",
    "component": "Reciprocal self-inclusive neighbor count using the specified component's canonical overlap and identity.",
    "concatenated": "Reciprocal self-inclusive neighbor count. Pool matched and jointly canonical site counts before taking their ratio; overlap is relative to the total concatenated length.",
    "cluster_representatives": "Connected components of the equal-component threshold graph (single linkage); lexicographically smallest observation ID has weight one, other members zero.",
    "clade_balanced": "Each supplied clade has total weight one, split equally among its observations.",
    "uniform": "Each matched observation has fixed unit weight.",
    "supplied": "Nonnegative finite weights supplied by observation ID; the original design scale is retained.",
}


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
    else:  # NumPy 1.26â€“1.x
        octets = np.ascontiguousarray(words).view(np.uint8).reshape(*words.shape, 8)
        counts = _BYTE_COUNTS[octets].sum(axis=-1, dtype=np.uint8)
    return counts[0] if counts.shape[0] == 1 else counts.sum(axis=0, dtype=np.int64)


class _ComponentComparisons:
    def __init__(self, component, *, allow_empty=False):
        matrix = component.matrix
        q = len(CANONICAL[component.molecule])
        valid = (matrix >= 0) & (matrix < q)
        if not allow_empty and not np.all(valid.any(axis=1)):
            raise MissingRowSupport(f"component {component.name} has seqs w no canonical res")
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


def _parts(msa, policy, component):
    if policy not in POLICIES:
        raise ValueError(f"unknown weighting policy: {policy!r}")
    if isinstance(msa, MatchedTripletMSA):
        parts = msa.components
    elif isinstance(msa, ComponentMSA):
        parts = (msa,)
    else:
        raise TypeError("weighting expects a ComponentMSA or MatchedTripletMSA")
    if policy == "component":
        selected = tuple(c for c in parts if c.name == component)
        if not selected:
            raise ValueError("component weighting requires a present component name A, B, or C")
        return selected
    if component is not None:
        raise ValueError("component selection applies only to the component weighting policy")
    return parts


def _workspace(parts, policy, memory_budget_bytes, reuse):
    if memory_budget_bytes is not None and (
        isinstance(memory_budget_bytes, (bool, np.bool_))
        or not isinstance(memory_budget_bytes, (int, np.integer))
        or memory_budget_bytes < 1
    ):
        raise ValueError("weight memory budget must be a positive integer in bytes")
    if memory_budget_bytes is not None:
        memory_budget_bytes = int(memory_budget_bytes)
    n = parts[0].n_sequences
    block = cache = comparisons = pair_slots = word_slots = 0
    fixed = 64 * n
    peak = fixed
    if policy not in ("supplied", "clade_balanced", "uniform"):
        widths = [(c.n_sites + 63) // 64 for c in parts]
        packed = sum(
            8 * n * w * (1 + (len(CANONICAL[c.molecule]) - 1).bit_length())
            for c, w in zip(parts, widths)
        )
        # Includes building encoded arrays from strings plus packed/boolean temporaries.
        fixed += packed + 8 * n * sum(c.n_sites for c in parts)
        per_pair = 256 + 96 * max(widths)
        block = max(1, min(n, 256, isqrt(131072 // max(widths))))
        requested_cache = (
            2 * n * n * sum(np.dtype(np.min_scalar_type(c.n_sites)).itemsize for c in parts)
        )
        if (
            reuse
            and requested_cache <= _CACHE_BYTES
            and (
                memory_budget_bytes is None
                or fixed + requested_cache + per_pair <= memory_budget_bytes
            )
        ):
            cache = requested_cache
        if memory_budget_bytes is not None:
            available = memory_budget_bytes - fixed - cache
            if available < per_pair:
                raise ValueError("weight memory budget cannot hold one exact comparison block")
            block = min(block, isqrt(available // per_pair))
        peak = fixed + cache + block * block * per_pair
        comparisons = n * (n - 1) // 2
        complete_tiles, fringe = divmod(n, block)
        # Distinct off-diagonal blocks are evaluated once. Each diagonal block
        # computes a full square, including both symmetric entries and self.
        pair_slots = (n * n + complete_tiles * block * block + fringe * fringe) // 2
        word_slots = pair_slots * sum(widths)
    if memory_budget_bytes is not None and peak > memory_budget_bytes:
        raise ValueError("weight memory budget cannot hold the output and working arrays")
    return {
        "policy": policy,
        "observations": n,
        "block_rows": block,
        "cache_bytes": cache,
        "peak_bytes": int(peak),
        "memory_budget_bytes": memory_budget_bytes,
        "unordered_distinct_row_pairs": comparisons,
        "component_count": len(parts),
        "row_pair_slots_per_pass": pair_slots,
        "component_word_slots_per_uncached_pass": word_slots,
        "scope": "Conservative numerical array estimate for one weighting calculation; Python objects, existing alignment strings and other workflow stages excluded.",
    }


def weighting_preflight(
    msa,
    *,
    policy: WeightPolicy = "equal_component",
    component=None,
    memory_budget_bytes=None,
    reuse=False,
):
    """Plan the exact weight calculation from dimensions, before materializing matrices."""
    return _workspace(_parts(msa, policy, component), policy, memory_budget_bytes, reuse)


def _observation_ids(msa):
    ids = (
        tuple(m.observation_id for m in msa.matches)
        if isinstance(msa, MatchedTripletMSA)
        else msa.ids
    )
    if any(not isinstance(key, str) or not key.strip() for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("every weight requires a unique nonempty observation ID")
    return ids


def verify_weight_alignment(msa, weights):
    if (
        isinstance(weights, SequenceWeightResult)
        and weights.observation_ids
        and weights.observation_ids != _observation_ids(msa)
    ):
        raise ValueError("weight observation IDs or order differ from the matched alignment")


class _Clusters:
    """Union-find keeps threshold-graph membership in linear space."""

    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size

    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def add(self, i, j):
        a, b = self.root(i), self.root(j)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]

    def representatives(self, ids):
        roots = [self.root(i) for i in range(len(ids))]
        named = {}
        for root, key in zip(roots, ids):
            if root not in named or key < named[root]:
                named[root] = key
        return tuple(named[root] for root in roots)


class _WeightComparisons:
    """Reuse per-component identities across rematches, subject to a memory budget."""

    def __init__(self, components, *, reuse=False, policy="equal_component", plan=None):
        self.policy = policy
        self.plan = plan or _workspace(components, policy, None, reuse)

        self.parts = [
            _ComponentComparisons(c, allow_empty=policy == "concatenated") for c in components
        ]
        if (
            policy == "concatenated"
            and not np.logical_or.reduce([p.valid.any(axis=0) for p in self.parts]).all()
        ):
            raise MissingRowSupport("concatenated sequence has no canonical residues")
        self.n = components[0].n_sequences
        self.names = tuple(c.name for c in components)
        self.indices = (
            [{name: i for i, name in enumerate(c.ids)} for c in components] if reuse else []
        )
        self.n_components = len(components)
        max_denominator = self.n_components * prod(c.n_sites for c in components)
        self.dtype = np.int64 if max_denominator <= 2**53 else object
        self.block = self.plan["block_rows"]
        if self.plan["cache_bytes"]:
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

    def weights(self, identity_threshold, min_overlap, components=None, *, observation_ids=()):
        clusters = _Clusters(self.n) if self.policy == "cluster_representatives" else None
        pooled = self.policy == "concatenated"

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
            if pooled:
                denominator.fill(0)
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
                if pooled:
                    numerator += matched
                    denominator += overlap
                else:
                    numerator = numerator * overlap + matched * denominator
                    denominator *= overlap
                    eligible &= (overlap > 0) & (overlap / part.length >= min_overlap)
            if pooled:
                eligible &= (denominator > 0) & (
                    denominator / sum(p.length for p in self.parts) >= min_overlap
                )
                scale = 1
            else:
                scale = self.n_components
            eligible[eligible] = (
                numerator[eligible] / (denominator[eligible] * scale) >= identity_threshold
            )
            if left.start == right.start:
                np.fill_diagonal(eligible, True)
            else:
                neighbors[right] += eligible.sum(axis=0)
            neighbors[left] += eligible.sum(axis=1)
            if clusters is not None:
                edges = np.triu(eligible, k=1) if left.start == right.start else eligible
                for i, j in zip(*np.nonzero(edges)):
                    clusters.add(left.start + int(i), right.start + int(j))
        cluster_ids = clusters.representatives(observation_ids) if clusters is not None else ()
        if cluster_ids:
            weights = np.array(
                [key == rep for key, rep in zip(observation_ids, cluster_ids)], dtype=float
            )
        else:
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
            observation_ids=observation_ids,
            policy=self.policy,
            cluster_ids=cluster_ids,
            preflight=self.plan,
        )


def sequence_weights(
    msa: ComponentMSA | MatchedTripletMSA,
    identity_threshold: float = 0.8,
    min_overlap: float = 0.5,
    *,
    policy: WeightPolicy = "equal_component",
    component: str | None = None,
    clades: Mapping[str, str] | None = None,
    supplied_weights: Mapping[str, float] | None = None,
    memory_budget_bytes: int | None = None,
) -> SequenceWeightResult:
    """Compute one explicitly defined weight per observation, in matched row order."""
    identity_threshold = _check_fraction(identity_threshold, "identity_threshold")
    min_overlap = _check_fraction(min_overlap, "min_overlap")
    parts = _parts(msa, policy, component)
    plan = _workspace(parts, policy, memory_budget_bytes, False)
    ids = _observation_ids(msa)
    names = tuple(
        c.name for c in (msa.components if isinstance(msa, MatchedTripletMSA) else (msa,))
    )
    if (clades is not None) != (policy == "clade_balanced"):
        raise ValueError("clades must be supplied exactly for the clade_balanced policy")
    if (supplied_weights is not None) != (policy == "supplied"):
        raise ValueError("supplied_weights must be supplied exactly for the supplied policy")
    if policy == "uniform":
        weights = np.ones(len(ids), dtype=float)
        weights.setflags(write=False)
        return SequenceWeightResult(
            weights,
            None,
            float(len(ids)),
            len(ids),
            names,
            identity_threshold,
            min_overlap,
            ids,
            policy,
            preflight=plan,
        )
    if policy in ("clade_balanced", "supplied"):
        mapping = clades if policy == "clade_balanced" else supplied_weights
        if not isinstance(mapping, Mapping) or set(mapping) != set(ids):
            raise ValueError(
                "weight or clade mapping must cover exactly the matched observation IDs"
            )
        labels = ()
        if policy == "clade_balanced":
            labels = tuple(mapping[key] for key in ids)
            if any(not isinstance(v, str) or not v.strip() for v in labels):
                raise ValueError("each matched observation requires a nonempty clade")
            counts = Counter(labels)
            weights = np.array([1.0 / counts[v] for v in labels])
        else:
            raw = [mapping[key] for key in ids]
            if any(
                isinstance(v, (bool, np.bool_))
                or not isinstance(v, (float, int, np.floating, np.integer))
                for v in raw
            ):
                raise ValueError("each supplied weight must be numeric")
            weights = np.array(raw, dtype=float)
            if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
                raise ValueError("supplied weights must be finite, nonnegative, and not all zero")
        with np.errstate(over="ignore"):
            total = float(weights.sum())
        if not np.isfinite(total):
            raise ValueError("the sum of supplied weights exceeds the numeric range")
        weights.setflags(write=False)
        return SequenceWeightResult(
            weights,
            None,
            total,
            len(ids),
            names,
            identity_threshold,
            min_overlap,
            ids,
            policy,
            clades=labels,
            preflight=plan,
        )
    result = _WeightComparisons(parts, policy=policy, plan=plan).weights(
        identity_threshold, min_overlap, observation_ids=ids
    )
    return replace(result, components=names, component=component)


def rematch_evaluator(msa, weights, strata, comparison_type=_WeightComparisons):
    """Bind a validated null policy to the anchored observation IDs before shuffling."""
    if not isinstance(weights, SequenceWeightResult):
        raise TypeError("rematching requires declared sequence or observation-keyed weights")
    verify_weight_alignment(msa, weights)
    if weights.policy in ("supplied", "clade_balanced", "uniform"):
        if weights.policy == "clade_balanced":
            from triplet_coevolution.validation.phylogeny import exchangeability_groups

            for group in exchangeability_groups(msa.n_observations, strata):
                if len({weights.clades[i] for i in group}) > 1:
                    raise ValueError(
                        "clade-balanced null permutations require strata nested within weighting clades"
                    )

        def fixed(shuffled):
            verify_weight_alignment(shuffled, weights)
            return weights

        return fixed
    parts = _parts(msa, weights.policy, weights.component)
    plan = _workspace(parts, weights.policy, weights.preflight.get("memory_budget_bytes"), True)
    comparisons = comparison_type(parts, reuse=True, policy=weights.policy, plan=plan)

    def evaluate(shuffled):
        selected = _parts(shuffled, weights.policy, weights.component)
        if weights.policy == "equal_component":
            # The public scoring interface also accepts positional numerical weights.
            result = comparisons.weights(weights.identity_threshold, weights.min_overlap, selected)
            return (
                replace(result, observation_ids=_observation_ids(shuffled))
                if isinstance(result, SequenceWeightResult)
                else result
            )
        result = comparisons.weights(
            weights.identity_threshold,
            weights.min_overlap,
            selected,
            observation_ids=_observation_ids(shuffled),
        )
        return replace(result, components=weights.components, component=weights.component)

    return evaluate
