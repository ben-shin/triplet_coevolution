"""Fit and rank held-out statistics on a fold's frozen coordinate and state support."""

from dataclasses import dataclass

import numpy as np

from triplet_coevolution.inference.candidates import candidate_id, select_candidates
from triplet_coevolution.inference.dca_models import DCADesign
from triplet_coevolution.inference.maxent import fit_maxent_triples
from triplet_coevolution.inference.maxent_support import freeze_support
from triplet_coevolution.inference.mi import triplet_mi
from triplet_coevolution.inference.potts import direct_couplings
from triplet_coevolution.inference.scores import (
    coordinate_axes,
    dca_with_apc_policy,
    get_score,
    mi_with_apc_policy,
)
from triplet_coevolution.inference.weights import sequence_weights
from triplet_coevolution.validation.clade_splits import identity

PAIRS = {"AB": (0, 1), "AC": (0, 2), "BC": (1, 2)}


def fit_weights(msa, groups, options, original_ids=None):
    options = dict(options)
    ids = tuple(r.observation_id for r in msa.matches)
    if options["policy"] == "clade_balanced":
        options["clades"] = {key: groups[key] for key in ids}
    if options["policy"] == "supplied":
        supplied = options["supplied_weights"]
        options["supplied_weights"] = {
            key: supplied[source]
            for key, source in zip(ids, ids if original_ids is None else original_ids, strict=True)
        }
        if not any(options["supplied_weights"].values()):
            return None
    return sequence_weights(msa, **options)


@dataclass(frozen=True)
class FoldScores:
    maps: tuple
    axes: tuple
    statistics: tuple
    min_valid: int
    min_valid_weight: float
    include_gaps: bool
    ridge: float
    apc_universe: str
    triples: object
    support: object
    maxent: object
    dca_design: dict | None = None

    def coordinates(self, interface, index):
        if interface == "ABC":
            indices = self.triples.triples[index]
            roles = (0, 1, 2)
            key = candidate_id(self.maps, indices)
        else:
            roles = PAIRS[interface]
            shape = tuple(len(self.axes[r]) for r in roles)
            indices = tuple(
                int(self.axes[r][i])
                for r, i in zip(roles, np.unravel_index(index, shape), strict=True)
            )
            key = identity(
                [
                    [
                        self.maps[r].component,
                        self.maps[r].reference_id,
                        self.maps[r].original_columns[i] + 1,
                    ]
                    for r, i in zip(roles, indices, strict=True)
                ]
            )
        result = {"candidate_id": key}
        for r, i in zip(roles, indices, strict=True):
            m = self.maps[r]
            result.update(
                {
                    m.component + "_column": m.original_columns[i] + 1,
                    m.component + "_reference": m.reference_id,
                    m.component + "_position": m.reference_positions[i],
                }
            )
        return result

    def fit(self, msa, weights):
        block = {}
        methods = {get_score(v).method for v in self.statistics}
        cut = {"min_valid": self.min_valid, "min_valid_weight": self.min_valid_weight}
        if weights is not None and "mi" in methods:
            mi = mi_with_apc_policy(
                triplet_mi(msa, weights, include_gaps=self.include_gaps, **cut),
                self.maps,
                self.apc_universe,
            )
            block["mi"] = {p: getattr(mi, p.lower()) for p in PAIRS}
        if weights is not None and "dca" in methods:
            design = DCADesign.parse(
                self.dca_design, ridge=self.ridge if self.dca_design is None else None
            )
            dca = design.fit(msa, weights, mf=direct_couplings, **cut)
            block["dca"] = dca_with_apc_policy(dca, self.maps, self.apc_universe)
        triples = None
        if weights is not None and "threebody" in methods:
            triples = fit_maxent_triples(
                msa, self.triples.triples, weights, support=self.support, options=self.maxent, **cut
            )
        result = {}
        for name in self.statistics:
            spec = get_score(name)
            if spec.method == "threebody":
                result[name, "ABC"] = (
                    np.array([getattr(v, name) for v in triples])
                    if triples is not None
                    else np.full(len(self.triples.triples), np.nan)
                )
                continue
            for pair, (a, b) in PAIRS.items():
                shape = len(self.axes[a]), len(self.axes[b])
                result[name, pair] = (
                    spec.values(block[spec.method][pair])[
                        np.ix_(self.axes[a], self.axes[b])
                    ].ravel()
                    if weights is not None
                    else np.full(np.prod(shape), np.nan)
                )
        return result


def fold_scorer(train, maps, weights, settings, candidate_options, maxent):
    axes = coordinate_axes(maps, settings["require_reference"])
    support = candidates = None
    if any(get_score(s).method == "threebody" for s in settings["statistics"]):
        opts = dict(candidate_options)
        wants_evidence = opts["mode"] in ("triangle", "wedge", "entropy", "hybrid")
        candidates = select_candidates(
            maps,
            axes,
            **opts,
            msa=train if wants_evidence else None,
            weights=weights if wants_evidence else None,
        )
        ids = tuple(r.observation_id for r in train.matches)
        support = freeze_support(
            train,
            include_gaps=settings["include_gaps"],
            min_state_count=settings["min_state_count"],
            training_observations=ids if settings["min_state_count"] else None,
        )
    return FoldScores(
        maps,
        axes,
        settings["statistics"],
        settings["min_valid"],
        settings["min_valid_weight"],
        settings["include_gaps"],
        settings["ridge"],
        settings["apc_universe"],
        candidates,
        support,
        maxent,
        settings.get("dca_design"),
    )


@dataclass(frozen=True)
class Ranks:
    display: np.ndarray
    mid: np.ndarray
    credit: np.ndarray
    top: np.ndarray
    n_finite: int
    cutoff_tie_size: int
    cutoff_tie_selected: int


def rank_scores(values, statistic, top_k):
    """Canonical display ties, average ranks for correlation, fractional credit at K."""
    values = np.asarray(values)
    valid = np.flatnonzero(np.isfinite(values))
    sign = -1 if get_score(statistic).direction == "higher" else 1
    order = valid[np.argsort(sign * values[valid], kind="stable")]
    display = np.zeros(len(values), dtype=int)
    mid = np.full(len(values), np.nan)
    credit = np.zeros(len(values), dtype=float)
    count = min(len(order), top_k)
    if not count:
        return Ranks(display, mid, credit, order, 0, 0, 0)
    display[order] = np.arange(1, len(order) + 1)
    runs = np.r_[0, np.flatnonzero(np.diff(values[order]) != 0) + 1, len(order)]
    sizes = np.diff(runs)
    mid[order] = np.repeat((runs[:-1] + runs[1:] + 1) / 2, sizes)
    cutoff_run = int(np.searchsorted(runs[1:], count))
    begin, end = runs[cutoff_run : cutoff_run + 2]
    credit[order[:begin]] = 1.0
    credit[order[begin:end]] = (count - begin) / (end - begin)
    return Ranks(
        display, mid, credit, order[:count], len(order), int(end - begin), int(count - begin)
    )


def common_spearman(train, other, statistic):
    mask = np.isfinite(train) & np.isfinite(other)
    if mask.sum() < 2:
        return None
    a = rank_scores(train[mask], statistic, 1).mid
    b = rank_scores(other[mask], statistic, 1).mid
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])
