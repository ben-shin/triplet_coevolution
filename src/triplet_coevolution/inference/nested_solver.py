"""Shared categorical conditional objective, exact finite-panel law and bounded optimizers."""

from dataclasses import dataclass
from itertools import combinations
from math import isfinite, prod
from numbers import Integral, Real

import numpy as np


def _integer(value, name, low=1, high=2**63 - 1):
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return int(value)


def _number(value, name, low, *, inclusive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or (value < low if inclusive else value <= low)
    ):
        raise ValueError(f"invalid {name}")
    return float(value)


@dataclass(frozen=True)
class NestedOptions:
    l2: float = 0.1
    field_l2: float | None = None
    pair_l2: float | None = None
    triple_l2: float | None = None
    min_valid: int = 2
    min_valid_weight: float = 0.0
    include_gaps: bool = False
    min_state_count: int = 0
    optimizer: str = "gradient"
    tolerance: float = 1e-6
    max_iterations: int = 500
    max_coefficients: int = 250_000
    max_workspace_bytes: int = 256 * 1024**2
    max_joint_states: int = 65_536

    def __post_init__(self):
        for name in ("l2", "tolerance", "field_l2", "pair_l2", "triple_l2"):
            value = getattr(self, name)
            if value is not None or name in ("l2", "tolerance"):
                _number(value, name, 0)
        for name, low, high in (
            ("min_valid", 1, 2**63 - 1),
            ("min_state_count", 0, 2**63 - 1),
            ("max_iterations", 1, 100_000),
            ("max_coefficients", 1, 2_000_000),
            ("max_workspace_bytes", 1, 2**63 - 1),
            ("max_joint_states", 1, 65_536),
        ):
            _integer(getattr(self, name), name, low, high)
        _number(self.min_valid_weight, "min_valid_weight", 0, inclusive=True)
        if self.optimizer not in ("gradient", "lbfgs") or type(self.include_gaps) is not bool:
            raise ValueError("invalid nested optimizer or gap policy")

    def penalty(self, order):
        return (self.field_l2, self.pair_l2, self.triple_l2)[order - 1] or self.l2


def _center(arr):
    for axis in range(arr.ndim):
        arr -= arr.mean(axis=axis, keepdims=True)
    return arr


def canonical_terms(dimensions, tensors):
    """ANOVA decomposition preserving complete configuration energies including the intercept."""
    result = {(): np.asarray(0.0)}
    for term, raw in tensors.items():
        term = tuple(term)
        raw = np.asarray(raw, dtype=float)
        if (
            term != tuple(sorted(set(term)))
            or any(type(i) is not int or not 0 <= i < len(dimensions) for i in term)
            or raw.shape != tuple(dimensions[i] for i in term)
            or not np.isfinite(raw).all()
        ):
            raise ValueError("invalid energy tensor")
        for order in range(len(term) + 1):
            for kept in combinations(range(len(term)), order):
                missing = tuple(i for i in range(len(term)) if i not in kept)
                piece = np.array(raw.mean(axis=missing), copy=True) if missing else raw.copy()
                subset = tuple(term[i] for i in kept)
                result[subset] = result.get(subset, 0.0) + _center(piece)
    return result


class Layout:
    def __init__(self, dimensions, edges, triples, options):
        self.dimensions = tuple(_integer(q, "alphabet size", 1, 32) for q in dimensions)
        n = len(self.dimensions)
        if not 1 <= n <= 256:
            raise ValueError("nested panel requires 1 to 256 sites")
        terms = [(i,) for i in range(n)]
        for order, values in ((2, edges), (3, triples)):
            parsed = set()
            for term in values:
                term = tuple(_integer(i, "term site", 0, n - 1) for i in term)
                if len(term) != order or tuple(sorted(set(term))) != term or term in parsed:
                    raise ValueError("terms must be distinct strictly increasing site tuples")
                parsed.add(term)
            terms.extend(sorted(parsed))
        self.terms = tuple(terms)
        self.shapes = tuple(tuple(self.dimensions[i] for i in term) for term in terms)
        counts = [prod(shape) for shape in self.shapes]
        self.size = sum(counts)
        self.degrees = sum(prod(q - 1 for q in shape) for shape in self.shapes)
        if self.size > options.max_coefficients:
            raise ValueError(
                f"nested panel requires {self.size} coefficients exceeding max_coefficients"
            )
        self.offsets = np.cumsum([0, *counts])
        self.options = options

    def arrays(self, theta):
        return [
            theta[self.offsets[i] : self.offsets[i + 1]].reshape(shape)
            for i, shape in enumerate(self.shapes)
        ]

    def projected(self, theta):
        v = np.array(theta, dtype=float, copy=True)
        if v.shape != (self.size,) or not np.isfinite(v).all():
            raise ValueError("invalid nested coefficient vector")
        for arr in self.arrays(v):
            _center(arr)
        return v

    def guard(self, nrows):
        width = sum(self.dimensions)
        slots = sum(sum(shape) for shape in self.shapes)
        # Every term owns an integer row-by-target-state index. Include dense conditional
        # arrays, row compaction and up to ten optimizer correction-vector pairs.
        estimate = 1024 * len(self.terms) + 8 * (
            nrows * (slots + 5 * width + 4 * len(self.dimensions))
            + self.size * (36 if self.options.optimizer == "lbfgs" else 12)
        )
        if estimate > self.options.max_workspace_bytes:
            raise ValueError(f"nested fit workspace {estimate} bytes exceeds max_workspace_bytes")
        return estimate


class Conditionals:
    def __init__(self, layout, rows):
        self.layout = layout
        x = np.asarray(rows)
        if x.ndim != 2 or x.shape[1] != len(layout.dimensions) or x.dtype.kind not in "iu":
            raise ValueError("encoded rows require a matching integer panel")
        if np.any(x < 0) or np.any(x >= np.asarray(layout.dimensions)):
            raise ValueError("conditional loss requires complete rows on the frozen bins")
        self.rows = x
        self.peak = layout.guard(len(x))
        self.lookups = []
        for term, shape in zip(layout.terms, layout.shapes, strict=True):
            strides = np.array([prod(shape[i + 1 :]) for i in range(len(shape))])
            full_index = x[:, term] @ strides
            contexts = []
            for k, site in enumerate(term):
                contexts.append(
                    (full_index - x[:, site] * strides[k])[:, None]
                    + strides[k] * np.arange(shape[k])
                )
            self.lookups.append(contexts)

    def logits(self, theta):
        logits = [np.zeros((len(self.rows), q)) for q in self.layout.dimensions]
        for term, look, tensor in zip(
            self.layout.terms, self.lookups, self.layout.arrays(theta), strict=True
        ):
            for site, indices in zip(term, look, strict=True):
                logits[site] += tensor.ravel()[indices]
        return logits

    def losses(self, theta):
        result = []
        for col, x in enumerate(self.logits(theta)):
            x -= x.max(axis=1, keepdims=True)
            result.append(np.log(np.exp(x).sum(axis=1)) - x[np.arange(len(x)), self.rows[:, col]])
        return np.stack(result, axis=1)


class ConditionalObjective:
    """Mean weighted per-site negative log conditional probability plus zero-sum L2 penalties."""

    def __init__(self, layout, rows, weights):
        w = np.asarray(weights, dtype=float)
        if (
            w.shape != (len(rows),)
            or not np.isfinite(w).all()
            or np.any(w < 0)
            or not np.any(w > 0)
            or not np.isfinite(w.sum())
        ):
            raise ValueError(
                "objective needs finite nonnegative weights with positive finite total"
            )
        self.layout = layout
        self.peak = layout.guard(len(rows))
        patterns, inverse = np.unique(np.asarray(rows)[w > 0], axis=0, return_inverse=True)
        self.conditionals = Conditionals(layout, patterns)
        self.mass = np.bincount(inverse, weights=w[w > 0] / w.sum()) / len(layout.dimensions)
        self.penalties = np.repeat(
            [layout.options.penalty(len(t)) for t in layout.terms], np.diff(layout.offsets)
        )

    def evaluate(self, vector, gradient=True):
        theta = self.layout.projected(vector)
        residuals = self.conditionals.logits(theta)
        rows = self.conditionals.rows
        penalty = self.penalties * theta
        loss = 0.5 * np.dot(theta, penalty)
        for i, x in enumerate(residuals):
            x -= x.max(axis=1, keepdims=True)
            chosen = x[np.arange(len(x)), rows[:, i]]
            np.exp(x, out=x)
            partition = x.sum(axis=1)
            loss += self.mass @ (np.log(partition) - chosen)
            if gradient:
                x /= partition[:, None]
                x[np.arange(len(x)), rows[:, i]] -= 1
                x *= self.mass[:, None]
        if not gradient:
            return float(loss), None
        grad = penalty
        for term, look, arr in zip(
            self.layout.terms, self.conditionals.lookups, self.layout.arrays(grad), strict=True
        ):
            for site, indices in zip(term, look, strict=True):
                arr.ravel()[:] += np.bincount(
                    indices.ravel(), weights=residuals[site].ravel(), minlength=arr.size
                )
            _center(arr)
        return float(loss), grad


def optimize(layout, rows, weights):
    objective = ConditionalObjective(layout, rows, weights)
    opt = layout.options
    theta = np.zeros(layout.size)
    if opt.optimizer == "lbfgs":
        try:
            from scipy.optimize import minimize
        except ImportError as exc:
            raise ValueError(
                "nested lbfgs requires optional SciPy; choose gradient or install scipy"
            ) from exc
        result = minimize(
            objective.evaluate,
            theta,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": opt.max_iterations,
                "gtol": opt.tolerance / 2,
                "ftol": 0,
                "maxcor": 8,
            },
        )
        theta = layout.projected(result.x)
        iterations = int(result.nit)
        status = "optimizer_stopped"
    else:
        step = 1.0
        status = "max_iterations"
        for iterations in range(opt.max_iterations + 1):
            loss, grad = objective.evaluate(theta)
            if np.max(abs(grad)) <= opt.tolerance or iterations == opt.max_iterations:
                break
            slope = float(grad @ grad)
            for _ in range(50):
                trial = theta - step * grad
                if objective.evaluate(trial, False)[0] <= loss - 1e-4 * step * slope:
                    theta = trial
                    step *= 1.5
                    break
                step *= 0.5
            else:
                status = "line_search_failed"
                break
    loss, grad = objective.evaluate(theta)
    err = float(np.max(abs(grad)))
    if err <= opt.tolerance:
        status = "converged"
    theta = layout.projected(theta)
    theta.setflags(write=False)
    return theta, {
        "status": status,
        "iterations": iterations,
        "objective": loss,
        "gradient_max": err,
        "optimizer": opt.optimizer,
        "coefficients": layout.size,
        "zero_sum_degrees": layout.degrees,
        "unique_training_patterns": len(objective.conditionals.rows),
        "workspace_estimate_bytes": objective.peak,
    }


def exact_law(layout, theta):
    """Normalize the declared finite undirected shared-tensor model exactly."""
    shape = layout.dimensions
    if prod(shape) > layout.options.max_joint_states or len(shape) > 32:
        raise ValueError("exact nested law exceeds max_joint_states")
    mass = np.zeros(shape)
    for term, arr in zip(layout.terms, layout.arrays(theta), strict=True):
        mass += arr.reshape(tuple(shape[i] if i in term else 1 for i in range(len(shape))))
    offset = float(mass.max())
    np.exp(mass - offset, out=mass)
    if np.any(mass == 0) or not np.isfinite(mass).all():
        raise ValueError("exact nested law underflow")
    z = float(mass.sum())
    mass /= z
    return mass, offset + float(np.log(z))
