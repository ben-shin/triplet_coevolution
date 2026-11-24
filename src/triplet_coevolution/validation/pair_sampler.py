"""Exact finite-state pair-only panel fitting and iid simulation for small panels."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import prod
from numbers import Integral

import numpy as np

from triplet_coevolution.inference.maxent import MaxEntOptions, _ipf, fit_joint
from triplet_coevolution.inference.mi import (
    _check_min_valid,
    canonical_support_at_cutoff,
    check_min_valid_weight,
)
from triplet_coevolution.inference.scores import get_score
from triplet_coevolution.validation.null_models import _design_weights, _m2_statistic

MAX_CONFIGURATIONS = 65_536


def _shape(values, limit):
    shape = tuple(values)
    if not 3 <= len(shape) <= 8 or any(
        isinstance(q, bool) or not isinstance(q, Integral) or q < 1 for q in shape
    ):
        raise ValueError(
            "an exact panel needs three to eight sites with fixed positive integer alphabet sizes"
        )
    shape = tuple(map(int, shape))
    if prod(shape) > min(limit, MAX_CONFIGURATIONS):
        raise ValueError(f"exact panel has too many configurations: {prod(shape)}")
    return shape


def enumerate_pair_law(
    fields: Sequence[np.ndarray], couplings: Mapping[tuple[int, int], np.ndarray]
) -> np.ndarray:
    """Normalize exp(sum(fields) + sum(pair couplings)) over a bounded Cartesian panel."""
    fields = tuple(np.asarray(h, dtype=float) for h in fields)
    if any(h.ndim != 1 or not np.isfinite(h).all() for h in fields):
        raise ValueError("each field must be a finite vector")
    shape = _shape([h.size for h in fields], MAX_CONFIGURATIONS)
    dims = range(len(shape))
    log_mass = np.zeros(shape)
    for i, h in enumerate(fields):
        log_mass += h.reshape(tuple(shape[i] if d == i else 1 for d in dims))
    for (i, j), value in couplings.items():
        if type(i) is not int or type(j) is not int or not 0 <= i < j < len(shape):
            raise ValueError("couplings must have distinct ascending zero-based site indices")
        table = np.asarray(value, dtype=float)
        if table.shape != (shape[i], shape[j]) or not np.isfinite(table).all():
            raise ValueError("coupling shape or coefficients disagree with the declared alphabets")
        log_mass += table.reshape(tuple(shape[d] if d in (i, j) else 1 for d in dims))
    with np.errstate(under="ignore", invalid="ignore"):
        mass = np.exp(log_mass - log_mass.max())
    if not np.isfinite(mass).all() or np.any(mass == 0):
        raise ValueError("given finite pair-only coefficients exceed numerical probability range")
    mass /= mass.sum()
    mass.setflags(write=False)
    return mass


def _rows(rows, shape):
    values = np.asarray(rows)
    if (
        values.ndim != 2
        or values.shape[1] != len(shape)
        or values.dtype.kind not in "iu"
        or len(values) < 1
    ):
        raise ValueError(
            "panel rows must be a nonempty integer matrix with the declared number of sites"
        )
    if np.any(values < -1) or np.any(values >= np.asarray(shape)):
        raise ValueError("panel states must use -1 for missing or a valid declared code")
    return values


@dataclass(frozen=True)
class PairPanel:
    shape: tuple[int, ...]
    options: MaxEntOptions
    status: str
    probabilities: np.ndarray | None = field(repr=False)
    missing_mask: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)
    weight_policy: str
    min_valid: int
    min_valid_weight: float
    n_valid: int
    valid_weight: float
    iterations: int
    residual: float

    def metadata(self) -> dict:
        return {
            "model": "exact_pair_panel",
            "status": self.status,
            "shape": list(self.shape),
            "configurations": prod(self.shape),
            "fit_unit": "complete positive-weight panel rows",
            "sampling": "iid exact categorical on the full state space followed by fixed row masking",
            "n_observations": len(self.weights),
            "n_valid": self.n_valid,
            "valid_weight": self.valid_weight,
            "weight_policy": self.weight_policy,
            "iterations": self.iterations,
            "pair_margin_residual": self.residual if np.isfinite(self.residual) else None,
            "options": asdict(self.options),
            "min_valid": self.min_valid,
            "min_valid_weight": self.min_valid_weight,
        }

    def draw(
        self, rng: np.random.Generator, *, missing_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Generate independent full rows exactly and restore the specified observation design."""
        if self.status != "ok":
            raise ValueError(f"exact panel unavailable: {self.status}")
        mask = self.missing_mask if missing_mask is None else np.asarray(missing_mask)
        if mask.dtype != bool or mask.shape != self.missing_mask.shape:
            raise ValueError("missing mask must match the full fitted row and column design")
        codes = rng.choice(prod(self.shape), size=len(self.weights), p=self.probabilities.ravel())
        rows = np.stack(np.unravel_index(codes, self.shape), axis=1)
        rows[mask] = -1
        return rows

    def draw_refit(self, rng: np.random.Generator) -> tuple[np.ndarray, "PairPanel"]:
        """Draw under the fitted null and reestimate the full pair-only panel with identical options."""
        rows = self.draw(rng)
        fitted = fit_pair_panel(
            rows,
            self.shape,
            options=self.options,
            weights=self.weights,
            weight_policy=self.weight_policy,
            min_valid=self.min_valid,
            min_valid_weight=self.min_valid_weight,
        )
        return rows, fitted

    def local_score(
        self,
        rows: np.ndarray,
        columns: tuple[int, int, int],
        *,
        score: str = "connected_information_bits",
    ) -> float:
        """Refit the local M2/M3 statistic on its own complete rows inside the same generating panel."""
        get_score(score, "threebody")
        values = _rows(rows, self.shape)
        if len(values) != len(self.weights):
            raise ValueError("scoring rows must match observation weights")
        if (
            len(columns) != 3
            or any(
                isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < len(self.shape)
                for i in columns
            )
            or len(set(columns)) != 3
        ):
            raise ValueError("local statistic needs three distinct valid panel indices")
        sub = values[:, columns]
        valid = (sub >= 0).all(axis=1) & (self.weights > 0)
        n = int(valid.sum())
        w = canonical_support_at_cutoff(
            float(self.weights[valid].sum()), valid, self.weights, self.min_valid_weight
        )
        if self.status != "ok" or n < self.min_valid or w < self.min_valid_weight:
            return float("nan")
        shape = tuple(self.shape[i] for i in columns)
        cell = np.ravel_multi_index(sub[valid].T, shape)
        counts = np.bincount(cell, weights=self.weights[valid], minlength=prod(shape)).reshape(
            shape
        )
        fit = fit_joint(counts, options=self.options, n_valid=n)
        return _m2_statistic(fit, score)[1]


def fit_pair_panel(
    rows: np.ndarray,
    shape: Sequence[int],
    *,
    weights=None,
    options: MaxEntOptions | None = None,
    min_valid: int = 2,
    min_valid_weight: float = 0.0,
    weight_policy: str = "uniform",
) -> PairPanel:
    """Fit a normalized maximum-entropy law to all pair marginals of the complete-panel target."""
    opts = options or MaxEntOptions()
    shape = _shape(shape, opts.max_cells)
    values = _rows(rows, shape)
    w = _design_weights(weights, len(values), weight_policy).copy()
    threshold = _check_min_valid(min_valid)
    min_weight = check_min_valid_weight(min_valid_weight)
    missing = values == -1
    valid = ~missing.any(axis=1) & (w > 0)
    n = int(valid.sum())
    weight = canonical_support_at_cutoff(float(w[valid].sum()), valid, w, min_weight)
    w.setflags(write=False)
    missing.setflags(write=False)

    def result(status, p=None, it=0, residual=float("nan")):
        return PairPanel(
            shape,
            opts,
            status,
            p,
            missing,
            w,
            weight_policy,
            threshold,
            min_weight,
            n,
            weight,
            it,
            residual,
        )

    if n < threshold or weight < min_weight:
        return result("insufficient_rows" if n < threshold else "insufficient_weight")
    nu = prod(shape)
    code = np.ravel_multi_index(values[valid].T, shape)
    counts = np.bincount(code, weights=w[valid], minlength=nu).reshape(shape)
    prior = float(opts.prior_mass) * (nu if opts.prior_mode == "per_cell" else 1)
    if not np.isfinite(weight + prior):
        raise ValueError("total panel weight plus prior exceeds float range")
    target = counts / (weight + prior) + (prior / (weight + prior)) / nu
    if np.any((counts > 0) & (target == 0)) or (prior > 0 and np.any(target == 0)):
        return result("numerical_underflow")
    target /= target.sum()
    axes = tuple(
        tuple(k for k in range(len(shape)) if k not in pair)
        for pair in combinations(range(len(shape)), 2)
    )
    pair_margins = tuple(target.sum(axis=reduce) for reduce in axes)
    fit = _ipf(shape, np.ones(shape, dtype=bool), pair_margins, axes, opts, joint=target)
    return result(fit.status, fit.distribution, fit.iterations, fit.residual)
