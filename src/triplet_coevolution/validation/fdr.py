"""Benjamini-Hochberg adjustment on a predeclared family of hypotheses."""

import numpy as np


def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("BH expects a vector of finite probabilities")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    scaled = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted[order] = np.minimum(1, np.minimum.accumulate(scaled[::-1])[::-1])
    return adjusted
