"""Global Potts model with pair and selected three-site terms, fitted by pseudolikelihood."""

from dataclasses import replace
from itertools import combinations
from math import prod

import numpy as np

from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.mi import (
    _check_min_valid,
    _get_weights,
    canonical_support_at_cutoff,
    check_min_valid_weight,
)
from triplet_coevolution.inference.nested import (
    NestedOptions,
    compare_nested,
    evaluate_model,
    fit_nested,
    load_model,
)
from triplet_coevolution.inference.threebody import TripleScore, fit_triples

__all__ = ["NestedOptions", "compare_nested", "evaluate_model", "fit_nested", "global_triples", "load_model"]


def _zero_sum(array):
    for axis in range(array.ndim):
        array -= array.mean(axis=axis, keepdims=True)
    return array


def _fit(design, dimensions, terms, weights, l2):
    """Fit shared field, pair and triple tensors on complete categorical rows."""
    n = len(weights)
    weights = weights / weights.sum()
    shapes = [tuple(dimensions[i] for i in term) for term in terms]
    size = sum(prod(shape) for shape in shapes)
    lookups = n * sum(len(term) for term in terms)
    if size > 250_000 or lookups > 2_000_000:
        raise ValueError(
            f"global Potts panel needs {size} coefficients and {lookups} row lookups "
            "(limits 250000 and 2000000). Reduce sites, observations, or selected triples."
        )

    offsets = np.cumsum([0, *(prod(shape) for shape in shapes)])
    # For each conditional target, this index selects the observed states of
    # the other terms' sites in every row. No joint partition sum is needed.
    caches = []
    for term, shape in zip(terms, shapes):
        contexts = []
        for axis in range(len(term)):
            others = term[:axis] + term[axis + 1 :]
            other_shape = shape[:axis] + shape[axis + 1 :]
            index = (
                np.ravel_multi_index([design[i] for i in others], other_shape)
                if others
                else np.zeros(n, dtype=int)
            )
            contexts.append((index, other_shape))
        caches.append(contexts)
    obs = np.arange(n)

    def objective(theta, gradient):
        logits = [np.zeros((n, q)) for q in dimensions]
        for t, term in enumerate(terms):
            tensor = theta[offsets[t] : offsets[t + 1]].reshape(shapes[t])
            for axis, site in enumerate(term):
                index, _ = caches[t][axis]
                logits[site] += np.moveaxis(tensor, axis, -1).reshape(-1, dimensions[site])[index]
        loss = l2 / 2 * (theta @ theta)
        for site, values in enumerate(logits):
            values -= values.max(axis=1, keepdims=True)
            actual = values[obs, design[site]]
            np.exp(values, out=values)
            denom = values.sum(axis=1)
            loss += weights @ (np.log(denom) - actual)
            if gradient:
                values /= denom[:, None]
                values[obs, design[site]] -= 1
                values *= weights[:, None]
        if not gradient:
            return loss, None
        grad = l2 * theta
        for t, term in enumerate(terms):
            block = grad[offsets[t] : offsets[t + 1]].reshape(shapes[t])
            for axis, site in enumerate(term):
                index, other_shape = caches[t][axis]
                grouped = np.zeros((prod(other_shape), dimensions[site]))
                np.add.at(grouped, index, logits[site])
                block += np.moveaxis(grouped.reshape(*other_shape, dimensions[site]), -1, axis)
            _zero_sum(block)
        return loss, grad

    theta = np.zeros(size)
    step = 1.0
    for iteration in range(1, 501):
        loss, grad = objective(theta, True)
        error = float(np.max(np.abs(grad), initial=0))
        if error <= 1e-6:
            info = {
                "iterations": iteration - 1,
                "objective": float(loss),
                "gradient_max": error,
                "coefficients": size,
                "variable_sites": len(dimensions),
            }
            return {
                term: theta[offsets[t] : offsets[t + 1]].reshape(shapes[t])
                for t, term in enumerate(terms)
            }, info
        slope = grad @ grad
        while step > 1e-12:
            trial = theta - step * grad
            if objective(trial, False)[0] <= loss - 1e-4 * step * slope:
                break
            step *= 0.5
        else:
            raise ValueError("global Potts fit failed its descent check; increase global_l2")
        theta = trial
        step *= 1.5
    raise ValueError("global Potts fit did not converge in 500 iterations; increase global_l2")


def global_triples(msa, candidates, weights=None, *, l2=0.1, min_valid=2, min_valid_weight=0.0):
    """Jointly fit selected ABC terms conditional on all retained variable sites."""
    if not np.isfinite(l2) or l2 <= 0:
        raise ValueError("global_l2 must be finite and positive")
    min_valid = _check_min_valid(min_valid)
    min_valid_weight = check_min_valid_weight(min_valid_weight)
    w = _get_weights(weights, msa.n_observations,msa)
    valid = w > 0
    for comp in msa.components:
        q = len(CANONICAL[comp.molecule])
        valid &= np.all((comp.matrix >= 0) & (comp.matrix < q), axis=1)
    n = int(valid.sum())
    weight = canonical_support_at_cutoff(float(w[valid].sum()),valid,w,min_valid_weight)
    info = {"complete_observations": n, "complete_weight": weight}
    if n < min_valid or weight < min_valid_weight:
        unavailable = TripleScore(np.nan, np.nan, n, weight, 0, 0)
        return [unavailable] * len(candidates), {**info, "status": "insufficient complete rows" if n < min_valid else "insufficient complete weight"}

    site_index = {}
    design = []
    dimensions = []
    for c, comp in enumerate(msa.components):
        for j in range(comp.n_sites):
            states, codes = np.unique(comp.matrix[valid, j], return_inverse=True)
            if len(states) == 1:
                continue
            site_index[c, j] = len(design)
            design.append(codes)
            dimensions.append(len(states))
            if len(design) > 64:
                raise ValueError(
                    "global Potts fit supports at most 64 variable sites; reduce the panel"
                )
    terms = [(i,) for i in range(len(design))] + list(combinations(range(len(design)), 2))
    triple_terms = []
    for columns in candidates:
        variables = tuple(site_index.get((c, j)) for c, j in enumerate(columns))
        triple_terms.append(None if None in variables else variables)
    terms.extend(t for t in triple_terms if t is not None)
    fitted, convergence = _fit(design, dimensions, terms, w[valid], l2)
    # Descriptive co-information and support use the same complete rows as the fit.
    counts = fit_triples(msa, candidates, np.where(valid, w, 0), min_valid=min_valid, min_valid_weight=min_valid_weight)
    scores = [
        replace(s, threebody_rms=float(np.sqrt(np.mean(fitted[t] ** 2))) if t else 0.0)
        for t, s in zip(triple_terms, counts)
    ]
    return scores, {**info, **convergence, "status": "converged"}
