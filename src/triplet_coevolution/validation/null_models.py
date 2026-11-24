"""Random matching controls and the fitted iid local pairwise bootstrap."""

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import lgamma, log, prod

import numpy as np

from triplet_coevolution.core.alignment import MatchedTripletMSA, TripletMatch
from triplet_coevolution.inference.maxent import FrozenSupport, MaxEntOptions, fit_joint
from triplet_coevolution.inference.mi import (
    _check_min_valid,
    _get_weights,
    canonical_support_at_cutoff,
    check_min_valid_weight,
)
from triplet_coevolution.inference.scores import get_score
from triplet_coevolution.inference.weights import SequenceWeightResult
from triplet_coevolution.validation.phylogeny import exchangeability_groups

MOVING = {
    "whole_component": (1, 2),
    "within_stratum": (1, 2),
    "preserve_ab": (2,),
    "preserve_ac": (1,),
    "preserve_bc": (0,),
}
INDEPENDENT_SELECTION = {"prespecified", "independent", "uniform", "held_out"}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class MatchingNull:
    name: str
    msa: MatchedTripletMSA = field(repr=False)
    groups: tuple[np.ndarray, ...] = field(repr=False)
    labels: tuple[str, ...]

    def row_orders(self, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        """Source row indices in destination-observation order, one permutation per role."""
        orders = tuple(np.arange(self.msa.n_observations) for _ in "ABC")
        for index in MOVING[self.name]:
            for group in self.groups:
                orders[index][group] = rng.permutation(group)
        return orders

    def draw(self, rng: np.random.Generator) -> MatchedTripletMSA:
        """Move complete, identified component sequence rows in allowed groups."""
        orders = self.row_orders(rng)
        components = list(self.msa.components)
        for index in MOVING[self.name]:
            comp, order = components[index], orders[index]
            components[index] = comp.take_ids([comp.ids[i] for i in order])
        matches = tuple(
            TripletMatch(old.observation_id, *ids)
            for old, ids in zip(
                self.msa.matches, zip(*(comp.ids for comp in components)), strict=True
            )
        )
        return MatchedTripletMSA(*components, matches)

    def metadata(self) -> dict:
        sizes = dict(Counter(self.labels))
        singletons = sum(n == 1 for n in sizes.values())
        n = self.msa.n_observations
        changing = ["ABC"[i] for i in MOVING[self.name]]
        sequence_entropy = 0.0
        changeable = 0
        for group in self.groups:
            group_changes = False
            for i in MOVING[self.name]:
                seqs = self.msa.components[i].sequences
                copies = Counter(seqs[j] for j in group)
                if len(copies) > 1:
                    group_changes = True
                    sequence_entropy += (
                        lgamma(len(group) + 1) - sum(lgamma(k + 1) for k in copies.values())
                    ) / log(2)
            if group_changes:
                changeable += len(group)
        return {
            "model": self.name,
            "shuffle_unit": "one intact identified component row",
            "shuffled_components": changing,
            "preserved_components": [name for name in "ABC" if name not in changing],
            "n_observations": n,
            "group_sizes": sizes,
            "singleton_count": singletons,
            "movable_fraction": (n - singletons) / n,
            "permutation_entropy_bits": len(changing)
            * sum(lgamma(k + 1) / log(2) for k in sizes.values()),
            "entropy_unit": "bits for uniform labeled row assignments, including identity",
            "sequence_permutation_entropy_bits": sequence_entropy,
            "sequence_movable_fraction": changeable / n,
            "warning": "all movable component rows have identical sequences within their strata"
            if not changeable
            else None,
            "strata_sha256": _digest(
                list(zip((m.observation_id for m in self.msa.matches), self.labels))
            ),
        }


def matching_null(
    msa: MatchedTripletMSA, name: str, *, strata: Mapping[str, str] | Sequence[str] | None = None
) -> MatchingNull:
    """Preserve the named pair if requested, optionally restricting moves to blocks."""
    if name not in MOVING:
        raise ValueError(f"unknown matching control {name!r}")
    if name == "whole_component" and strata is not None:
        raise ValueError("whole_component is unrestricted; choose within_stratum")
    if name == "within_stratum" and strata is None:
        raise ValueError("within_stratum requires explicit stratum labels")
    ids = [m.observation_id for m in msa.matches]
    if isinstance(strata, Mapping):
        if set(ids) != set(strata):
            raise ValueError("strata must identify exactly the matched observation IDs")
        strata = [strata[key] for key in ids]
    if isinstance(strata, (str, bytes)):
        raise TypeError("strata must give a separate label for every observation")
    groups = exchangeability_groups(len(ids), strata)
    for group in groups:
        group.setflags(write=False)
    return MatchingNull(
        name, msa, groups, tuple(strata) if strata is not None else ("all",) * len(ids)
    )


def _design_weights(weights, n, policy, msa=None):
    if policy not in {"uniform", "fixed_design"}:
        raise ValueError("iid bootstrap requires uniform or independently fixed_design weights")
    if isinstance(weights, SequenceWeightResult) and weights.policy not in {"uniform", "supplied"}:
        raise ValueError(
            "sequence-derived or clade-based weighting is outside the iid sampling design"
        )
    w = _get_weights(weights, n, msa)
    if policy == "uniform" and np.any(w != 1):
        raise ValueError("uniform iid bootstrap requires unit row weights")
    return w


def _contrast_witness(active: np.ndarray) -> bool:
    """A feasible 2×2×2 alternating move proves at least one estimable triple contrast."""
    for left, right in combinations(active, 2):
        common = (left & right).astype(int)
        if np.any(np.triu(common @ common.T, 1) >= 2):
            return True
    return False


def _m2_statistic(fit, score):
    if fit.score.status != "ok":
        return fit.score.status, float("nan")
    value = float(getattr(fit.score, score))
    if not np.isfinite(value):
        return "undefined_observed_statistic", float("nan")
    if fit.score.information_status == "constant_empirical_axis":
        return "constant_empirical_axis", float("nan")
    if not _contrast_witness(fit.p2 > 0):
        return "no_verified_triple_contrast", float("nan")
    return "ok", value


@dataclass(frozen=True)
class M2Bootstrap:
    status: str
    observed: float
    candidate: tuple[int, int, int]
    shape: tuple[int, int, int]
    score: str
    options: MaxEntOptions
    valid_indices: tuple[int, ...]
    row_weights: np.ndarray = field(repr=False)
    distribution: np.ndarray | None = field(repr=False)
    allowed: np.ndarray | None = field(repr=False)
    selection: str
    weight_policy: str
    n_observations: int
    observation_sha256: str

    def metadata(self) -> dict:
        return {
            "model": "local_m2",
            "status": self.status,
            "score": self.score,
            "direction": get_score(self.score).direction,
            "shape": list(self.shape),
            "candidate": [int(i) for i in self.candidate],
            "options": asdict(self.options),
            "selection": self.selection,
            "weight_policy": self.weight_policy,
            "n_observations": self.n_observations,
            "n_valid": len(self.valid_indices),
            "valid_weight": float(self.row_weights.sum()),
            "complete_observation_ids_sha256": self.observation_sha256,
            "design": "iid complete rows conditional on the fixed missingness and row-weight design",
        }

    def draw_counts(self, rng: np.random.Generator) -> np.ndarray:
        """Sample once per eligible observation, retaining its observed weight."""
        if self.status != "ok":
            raise ValueError(f"bootstrap unavailable: {self.status}")
        p = self.distribution.ravel()
        n = len(self.row_weights)
        if np.all(self.row_weights == self.row_weights[0]):
            # Multinomial counts are exactly the sum of n iid categorical draws.
            # Its last bin receives the remainder. Keep that bin inside the fitted face.
            if p[-1] == 0:
                positive = p > 0
                counts = np.zeros(p.size, dtype=float)
                counts[positive] = rng.multinomial(n, p[positive])
            else:
                counts = rng.multinomial(n, p)
            return (counts * self.row_weights[0]).reshape(self.shape)
        states = rng.choice(len(p), size=n, p=p)
        return np.bincount(states, weights=self.row_weights, minlength=len(p)).reshape(self.shape)

    def draw(self, rng: np.random.Generator) -> float:
        """Return a freshly refitted statistic; unresolved draws remain NaN."""
        if self.status != "ok":
            return float("nan")
        fit = fit_joint(
            self.draw_counts(rng),
            options=self.options,
            allowed=self.allowed,
            n_valid=len(self.row_weights),
        )
        return _m2_statistic(fit, self.score)[1]


def prepare_m2_bootstrap(
    msa: MatchedTripletMSA,
    candidate: tuple[int, int, int],
    *,
    support: FrozenSupport,
    options: MaxEntOptions,
    weights=None,
    weight_policy: str = "uniform",
    selection: str = "prespecified",
    score: str = "connected_information_bits",
    min_valid: int = 2,
    min_valid_weight: float = 0.0,
    allowed: np.ndarray | None = None,
) -> M2Bootstrap:
    """Fit the coherent local pair null on a fixed, score-independent test design."""
    if selection not in INDEPENDENT_SELECTION:
        raise ValueError(
            "local M2 calibration requires prespecified or independently selected candidates"
        )
    ids = tuple(m.observation_id for m in msa.matches)
    if set(ids).intersection(support.training_observations):
        raise ValueError("training-derived support requires disjoint calibration observations")
    get_score(score, "threebody")
    n_required = _check_min_valid(min_valid)
    min_weight = check_min_valid_weight(min_valid_weight)
    w = _design_weights(weights, len(ids), weight_policy, msa)
    if support.molecules != tuple(c.molecule.value for c in msa.components) or tuple(
        len(sites) for sites in support.maps
    ) != tuple(c.n_sites for c in msa.components):
        raise ValueError("frozen state support must match the retained alphabet and site axes")
    candidate = tuple(candidate)
    shape = support.shape_for(candidate)
    enc = [np.array(maps[i]) for maps, i in zip(support.maps, candidate)]
    raw = [c.matrix[:, i] for c, i in zip(msa.components, candidate)]
    valid = w > 0
    for col, lookup in zip(raw, enc):
        valid &= (col >= 0) & (col < len(lookup))
    indices = tuple(map(int, np.flatnonzero(valid)))
    row_weights = w[valid]
    row_weights.setflags(write=False)
    weight = canonical_support_at_cutoff(float(row_weights.sum()), valid, w, min_weight)
    status, observed, probability = "ok", float("nan"), None
    if prod(shape) > options.max_cells:
        status = "resource_limit"
    elif len(indices) < n_required:
        status = "insufficient_rows"
    elif weight < min_weight:
        status = "insufficient_weight"
    else:
        cell = np.ravel_multi_index(tuple(m[c[valid]] for m, c in zip(enc, raw)), shape)
        counts = np.bincount(cell, weights=row_weights, minlength=prod(shape)).reshape(shape)
        if allowed is not None:
            allowed = np.array(allowed, copy=True)
            allowed.setflags(write=False)
        fit = fit_joint(counts, options=options, allowed=allowed, n_valid=len(indices))
        status, observed = _m2_statistic(fit, score)
        if status == "ok":
            probability = fit.p2
    return M2Bootstrap(
        status,
        observed,
        candidate,
        shape,
        score,
        options,
        indices,
        row_weights,
        probability,
        allowed,
        selection,
        weight_policy,
        len(ids),
        _digest([ids[i] for i in indices]),
    )
