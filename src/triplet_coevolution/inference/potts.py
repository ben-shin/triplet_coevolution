"""Ridge-regularized mean-field direct coupling analysis for mixed alphabets."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from triplet_coevolution.core.alignment import MatchedTripletMSA
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.mi import _check_min_valid, _get_weights, apc
from triplet_coevolution.inference.weights import SequenceWeightResult

PAIRS = tuple(combinations(range(3), 2))
MAX_SOLVE_DIMENSION = 4096
MAX_DUAL_ELEMENTS = 16_000_000


@dataclass(frozen=True)
class DCABlock:
    frobenius: np.ndarray
    corrected: np.ndarray
    n_valid: np.ndarray
    valid_weight: np.ndarray


def direct_couplings(
    msa: MatchedTripletMSA,
    weights: SequenceWeightResult | np.ndarray | None = None,
    *,
    ridge: float = 0.1,
    min_valid: int = 2,
) -> dict[str, DCABlock]:
    """Fit one inverse covariance model across all retained sites of A, B and C."""
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("DCA ridge must be finite and positive")
    min_valid = _check_min_valid(min_valid)
    w = _get_weights(weights, msa.n_observations)
    sites = []
    slices = []
    valid = []
    width = 0
    for comp in msa.components:
        q = len(CANONICAL[comp.molecule])
        is_canonical = (comp.matrix >= 0) & (comp.matrix < q)
        valid.append(is_canonical)
        blocks = []
        for column in range(comp.n_sites):
            x = comp.matrix[:, column]
            rows = is_canonical[:, column] & (w > 0)
            counts = np.bincount(x[rows], weights=w[rows], minlength=q)
            states = np.flatnonzero(counts)
            # A constant/empty site has no estimable contrasts.
            if len(states) < 2:
                states = states[:0]
            block = slice(width, width + len(states))
            width = block.stop
            blocks.append(block)
            if len(states):
                sites.append((x, q, states, counts[states] / counts.sum(), block))
        slices.append(blocks)
    n = len(w)
    dual = n < width and n <= MAX_SOLVE_DIMENSION and n * width <= MAX_DUAL_ELEMENTS
    if not dual and width > MAX_SOLVE_DIMENSION:
        raise ValueError(
            f"DCA panel with {n} observations and {width} observed-state features exceeds "
            f"both solver budgets (dimension {MAX_SOLVE_DIMENSION}, dual elements {MAX_DUAL_ELEMENTS}). "
            "Use a smaller panel or stricter column masks."
        )

    scaled_weights = np.sqrt(w / w.sum())

    def weighted_design(rows):
        design = np.zeros((len(w[rows]), width))
        for x, q, states, frequencies, block in sites:
            values = x[rows, None]
            centered = (values == states) - frequencies
            design[:, block] = np.where((values >= 0) & (values < q), centered, 0)
        design *= scaled_weights[rows, None]
        return design

    if dual:
        # For disjoint site blocks, -precision_ij = U_i.T @ U_j. This is
        # exactly the same ridge inverse via Woodbury in observation space.
        design = weighted_design(slice(None))
        gram = design @ design.T
        gram.flat[:: n + 1] += ridge
        features = np.linalg.solve(np.linalg.cholesky(gram), design) / np.sqrt(ridge)
        del design, gram
    else:
        covariance = np.zeros((width, width))
        batch = max(1, min(1024, 1_000_000 // max(1, width)))
        for start in range(0, n, batch):
            design = weighted_design(slice(start, start + batch))
            covariance += design.T @ design
        if width:
            covariance.flat[:: width + 1] += ridge
        precision = np.linalg.inv(covariance) if width else covariance

    results = {}
    for a, b in PAIRS:
        rows_a = valid[a] & (w > 0)[:, None]
        n_valid = rows_a.astype(np.int64).T @ valid[b]
        valid_weight = (w[:, None] * rows_a).T @ valid[b]
        scores = np.full(n_valid.shape, np.nan)
        for i, sx in enumerate(slices[a]):
            for j, sy in enumerate(slices[b]):
                if n_valid[i, j] < min_valid:
                    continue
                coupling = features[:, sx].T @ features[:, sy] if dual else -precision[sx, sy]
                if coupling.size:
                    coupling = (
                        coupling
                        - coupling.mean(0)
                        - coupling.mean(1, keepdims=True)
                        + coupling.mean()
                    )
                scores[i, j] = np.linalg.norm(coupling)
        results["ABC"[a] + "ABC"[b]] = DCABlock(scores, apc(scores), n_valid, valid_weight)
    return results
