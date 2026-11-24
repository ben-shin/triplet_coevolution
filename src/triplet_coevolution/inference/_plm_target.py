"""One masked, weighted categorical conditional likelihood and its L-BFGS fit."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from triplet_coevolution.inference.mi import canonical_support_at_cutoff


def zero_sum(matrix: np.ndarray) -> np.ndarray:
    """Project a finite pair coefficient table to zero mean across both axes."""
    return matrix - matrix.mean(axis=0) - matrix.mean(axis=1, keepdims=True) + matrix.mean()


@dataclass(frozen=True)
class ConditionalFit:
    fields: np.ndarray | None = field(repr=False)
    coefficients: dict[int, np.ndarray] = field(repr=False)
    predictor_frequencies: dict[int, np.ndarray] = field(repr=False)
    diagnostics: dict


class TargetObjective:
    """Mean conditional negative log likelihood in nats plus two quadratic penalties.

    Retained rows contain the target and every predictor on the prespecified panel.
    """

    def __init__(self, codes, dimensions, target, weights, options, min_valid_weight=0.0):
        self.options = options
        self.target = target
        self.qt = dimensions[target]
        self.predictors = tuple(j for j in range(len(dimensions)) if j != target)
        valid = (weights > 0) & np.all(codes >= 0, axis=1)
        self.rows = np.flatnonzero(valid)
        self.y = codes[valid, target]
        self.weight_sum = canonical_support_at_cutoff(
            float(weights[valid].sum()), valid, weights, min_valid_weight
        )
        self.w = weights[valid] / self.weight_sum if self.weight_sum else np.zeros(0)
        self.columns = {}
        self.means = {}
        self.slices = {}
        offset = 1
        rank_bound = 0
        for j in self.predictors:
            col = codes[valid, j]
            freq = np.bincount(col, weights=self.w, minlength=dimensions[j])
            if freq.sum() > 0:
                freq /= freq.sum()
            self.columns[j] = col
            self.means[j] = freq
            self.slices[j] = slice(offset, offset + dimensions[j])
            offset += dimensions[j]
            rank_bound += max(0, int(np.count_nonzero(freq)) - 1)
        self.size = offset * self.qt
        self.predictor_rank_upper_bound = min(max(0, len(self.rows) - 1), rank_bound)

    def capacity(self):
        columns = self.size // self.qt - 1
        trace = 1.0
        for p in self.means.values():
            trace += max(0.0, 1 - float(p @ p))
        lo, hi = (
            min(self.options.lambda_h, self.options.lambda_j),
            max(self.options.lambda_h, self.options.lambda_j),
        )
        rank = None
        if columns == 0 or self.predictor_rank_upper_bound == 0:
            rank = 0
        elif columns <= 128 and len(self.rows) * columns <= 65536:
            design = np.zeros((len(self.rows), columns))
            for j, sl in self.slices.items():
                block = design[:, sl.start - 1 : sl.stop - 1]
                col = self.columns[j]
                block[:] = -self.means[j]
                block[np.arange(len(col)), col] += 1
            design *= np.sqrt(self.w[:, None])
            rank = int(np.linalg.matrix_rank(design))
        return {
            "weighted_centered_design_rank": rank,
            "centered_predictor_rank_upper_bound": self.predictor_rank_upper_bound,
            "declared_predictor_features": columns,
            "penalized_hessian_condition_upper_bound": (hi + 0.5 * trace) / lo
            if hi + 0.5 * trace <= np.finfo(float).max * min(lo, 1.0)
            else None,
        }

    def loss_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        table = theta.reshape(-1, self.qt)
        gh = self.options.lambda_h
        gj = self.options.lambda_j
        grad = gj * table
        grad[0] = gh * table[0]
        loss = 0.5 * float(np.einsum("ij,ij->", table, grad))
        centers = {
            j: np.einsum("p,pk->k", self.means[j], table[sl]) for j, sl in self.slices.items()
        }
        step = self.options.batch_rows
        for start in range(0, len(self.y), step):
            end = min(len(self.y), start + step)
            ix = slice(start, end)
            n = end - start
            logits = np.broadcast_to(table[0], (n, self.qt)).copy()
            for j, sl in self.slices.items():
                col = self.columns[j][ix]
                logits += table[sl][col] - centers[j]
            logits -= logits.max(axis=1, keepdims=True)
            actual = logits[np.arange(n), self.y[ix]]
            np.exp(logits, out=logits)
            norm = logits.sum(axis=1)
            loss += float(np.sum(self.w[ix] * (np.log(norm) - actual)))
            logits /= norm[:, None]
            logits[np.arange(n), self.y[ix]] -= 1
            logits *= self.w[ix, None]
            row_sum = logits.sum(axis=0)
            grad[0] += row_sum
            for j, sl in self.slices.items():
                col = self.columns[j][ix]
                np.add.at(grad[sl], col, logits)
                grad[sl] -= self.means[j][:, None] * row_sum
        return loss, grad.ravel()


def _dot(x, y):
    return float(np.einsum("i,i->", x, y))


def minimize(objective):
    """Limited-memory BFGS with safeguarded descent and Armijo backtracking."""
    opt = objective.options
    x = np.zeros(objective.size)
    loss, g = objective.loss_grad(x)
    history = deque(maxlen=8)
    evaluations = 1
    for iteration in range(opt.max_iterations + 1):
        residual = float(np.max(np.abs(g)))
        if not np.isfinite(loss) or not np.isfinite(residual):
            status = "nonfinite_objective"
            break
        if residual <= opt.tolerance:
            status = "converged"
            break
        if iteration == opt.max_iterations:
            status = "iteration_limit"
            break
        q = g.copy()
        alphas = []
        for s, y, rho in reversed(history):
            alpha = rho * _dot(s, q)
            alphas.append(alpha)
            q -= alpha * y
        if history:
            s, y, rho = history[-1]
            q *= 1 / (rho * _dot(y, y))
        for (s, y, rho), alpha in zip(history, reversed(alphas)):
            q += s * (alpha - rho * _dot(y, q))
        slope = _dot(q, g)
        if not np.isfinite(slope) or slope <= 0:
            history.clear()
            q = g
            slope = _dot(g, g)
        step = 1.0
        for _ in range(45):
            new = x - step * q
            value, newg = objective.loss_grad(new)
            evaluations += 1
            if np.isfinite(value) and value <= loss - 1e-4 * step * slope:
                break
            step *= 0.5
        else:
            status = "line_search_failed"
            break
        s, y = new - x, newg - g
        curvature = _dot(s, y)
        if curvature > 1e-14 * np.sqrt(_dot(s, s) * _dot(y, y)):
            history.append((s, y, 1 / curvature))
        x, loss, g = new, value, newg
    return x, {
        "status": status,
        "iterations": iteration,
        "evaluations": evaluations,
        "objective_nats": loss if np.isfinite(loss) else None,
        "gradient_max": residual if np.isfinite(residual) else None,
    }


def solve_target(codes, dimensions, target, weights, options, min_valid, min_valid_weight):
    obj = TargetObjective(codes, dimensions, target, weights, options, min_valid_weight)
    count = len(obj.rows)
    observed = len(np.unique(obj.y))
    counts = np.bincount(obj.y, weights=obj.w, minlength=obj.qt)
    p = counts[counts > 0]
    diag = {
        "target": target,
        "target_states_observed": observed,
        "target_states_declared": obj.qt,
        "rows": count,
        "positive_rows_excluded_incomplete": int(np.count_nonzero(weights) - count),
        "zero_weight_rows_excluded": int(len(weights) - np.count_nonzero(weights)),
        "weight_sum": obj.weight_sum,
        "parameters": obj.size,
        "weighted_target_entropy_bits": float(-np.sum(p * np.log2(p))) if count else None,
        **obj.capacity(),
        "iterations": 0,
        "evaluations": 0,
        "objective_nats": None,
        "gradient_max": None,
    }
    if count < min_valid or obj.weight_sum < min_valid_weight or observed < 2:
        diag["status"] = (
            "insufficient_rows"
            if count < min_valid
            else "insufficient_weight"
            if obj.weight_sum < min_valid_weight
            else "constant_target"
        )
        return ConditionalFit(None, {}, obj.means, diag)
    x, report = minimize(obj)
    diag.update(report)
    if report["status"] != "converged":
        return ConditionalFit(None, {}, obj.means, diag)
    table = x.reshape(-1, obj.qt)
    return ConditionalFit(table[0], {j: table[sl] for j, sl in obj.slices.items()}, obj.means, diag)
