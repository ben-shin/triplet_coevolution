"""Ridge-regularized mean-field direct coupling analysis for mixed alphabets."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from triplet_coevolution.core.alignment import MatchedTripletMSA
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.mi import (
    _check_min_valid,
    _get_weights,
    _pair_support_at_cutoff,
    apc,
    check_min_valid_weight,
)
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


def factor_backend(factor_solver):
    if factor_solver == "numpy_legacy":
        return None
    if factor_solver != "cholesky":
        raise ValueError("mf factor_solver must be numpy_legacy or cholesky")
    try:
        from scipy import linalg
    except ImportError as exc:
        raise ValueError("mf factor_solver=cholesky requires the optimize extra (SciPy)") from exc
    return linalg


def direct_couplings(
    msa: MatchedTripletMSA,
    weights: SequenceWeightResult | np.ndarray | None = None,
    *,
    ridge: float = 0.1,
    min_valid: int = 2,
    min_valid_weight: float = 0.0,
    conditioning: str = "joint",
    include_gaps: bool = False,
    alphabet_support: str = "observed",
    shrinkage: float = 0.0,
    diagnostics: dict | None = None,
    factor_solver: str = "numpy_legacy",
) -> dict[str, DCABlock]:
    """Fit ridge inverse covariance jointly or on each two-component conditioning panel."""
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("DCA ridge must be finite and positive")
    min_valid = _check_min_valid(min_valid)
    min_valid_weight = check_min_valid_weight(min_valid_weight)
    w = _get_weights(weights, msa.n_observations, msa)
    if conditioning not in {"joint", "pairwise"} or alphabet_support not in {"observed", "fixed"}:
        raise ValueError(
            "DCA conditioning must be joint or pairwise, and alphabet_support observed or fixed"
        )
    if (
        type(include_gaps) is not bool
        or type(shrinkage) not in (float, int)
        or not np.isfinite(shrinkage)
        or not 0 <= shrinkage <= 1
    ):
        raise ValueError("include_gaps must be Boolean and shrinkage a fraction in [0,1]")
    sets = [(0, 1, 2)] if conditioning == "joint" else PAIRS
    factors = factor_backend(factor_solver)
    results = {}
    for indices in sets:
        subset = [msa.components[i] for i in indices]
        info = {} if diagnostics is not None else None
        results.update(
            _ridge_fit(
                subset,
                w,
                ridge,
                min_valid,
                min_valid_weight,
                include_gaps,
                alphabet_support,
                shrinkage,
                info,
                factors,
            )
        )
        if info is not None:
            diagnostics["".join(c.name for c in subset)] = info
    return results


def _ridge_fit(
    components,
    w,
    ridge,
    min_valid,
    min_valid_weight,
    include_gaps,
    alphabet_support,
    shrinkage,
    info,
    factors,
):
    sites = []
    contrast_indices = []
    total_variance = 0.0
    slices = []
    valid = []
    width = 0
    for comp in components:
        native_q = len(CANONICAL[comp.molecule])
        q = native_q + include_gaps
        canonical_contrasts = []
        is_canonical = (comp.matrix >= 0) & (comp.matrix < q)
        valid.append(is_canonical)
        blocks = []
        for column in range(comp.n_sites):
            x = comp.matrix[:, column]
            rows = is_canonical[:, column] & (w > 0)
            counts = np.bincount(x[rows], weights=w[rows], minlength=q)
            states = np.flatnonzero(counts)
            # A constant/empty site has no estimable contrasts.
            if alphabet_support == "fixed":
                states = np.arange(q)
            elif len(states) < 2:
                states = states[:0]
            canonical_contrasts.append(np.flatnonzero(states < native_q))
            if shrinkage and counts.sum():
                total_variance += (counts.sum() - np.dot(counts, counts) / counts.sum()) / w.sum()
            block = slice(width, width + len(states))
            width = block.stop
            blocks.append(block)
            if len(states):
                sites.append(
                    (
                        x,
                        q,
                        states,
                        counts[states] / counts.sum() if counts.sum() else np.zeros(len(states)),
                        block,
                    )
                )
        slices.append(blocks)
        contrast_indices.append(canonical_contrasts)
    n = len(w)
    dual = n < width and n <= MAX_SOLVE_DIMENSION and n * width <= MAX_DUAL_ELEMENTS
    if not dual and width > MAX_SOLVE_DIMENSION:
        raise ValueError(
            f"DCA panel with {n} observations and {width} observed-state features exceeds "
            f"both solver budgets (dimension {MAX_SOLVE_DIMENSION}, dual elements {MAX_DUAL_ELEMENTS}). "
            "Use a smaller panel or stricter column masks."
        )

    # Spherical shrinkage toward trace(C)/D * I, followed by an additive covariance ridge.
    penalty = ridge + (shrinkage * total_variance / width if shrinkage and width else 0.0)
    scaled_weights = np.sqrt(w / w.sum())
    if shrinkage:
        scaled_weights = scaled_weights * np.sqrt(1.0 - shrinkage)

    def spectrum(matrix):
        if info is None:
            return
        info.update(
            solver="dual" if dual else "primal",
            feature_dimension=width,
            observations=len(w),
            positive_weight_observations=int(np.count_nonzero(w)),
            centered_covariance_rank_upper_bound=min(width, int(np.count_nonzero(w))),
            ridge=ridge,
            shrinkage=shrinkage,
            effective_diagonal_penalty=penalty,
            rank_scope="spectrum of scaled covariance before diagonal penalty",
            factor_solver="cholesky" if factors is not None else "numpy_legacy",
        )
        if matrix.shape[0] > 512:
            info["spectrum_status"] = "not_computed_over_512_dimensions"
            return
        eigen = np.linalg.eigvalsh(matrix)
        top = float(eigen[-1]) if len(eigen) else 0.0
        smallest = (
            0.0 if dual and width > len(w) else max(float(eigen[0]), 0.0) if len(eigen) else 0.0
        )
        tol = max(width, len(w)) * np.finfo(float).eps * top
        info.update(
            spectrum_status="computed",
            rank=int(np.count_nonzero(eigen > tol)),
            max_scaled_eigenvalue=top,
            min_scaled_eigenvalue=smallest,
            regularized_condition_number=(top + penalty) / (smallest + penalty),
        )

    def weighted_design(rows):
        design = np.zeros((len(w[rows]), width))
        for x, q, states, frequencies, block in sites:
            values = x[rows, None]
            centered = (values == states) - frequencies
            design[:, block] = np.where((values >= 0) & (values < q), centered, 0)
        design *= scaled_weights[rows, None]
        return design

    def probe_columns():
        # First, last, and (for a wider panel) two spaced interior right-hand sides.
        return np.linspace(0, width - 1, min(4, width), dtype=int)

    def record_residual(indices, actual, expected):
        denominator = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
        info["factor_probe_columns_zero_based"] = indices.tolist()
        info["factor_probe_relative_frobenius_residual"] = float(
            np.linalg.norm(actual - expected) / denominator
        )

    if dual:
        # For disjoint site blocks, -precision_ij = U_i.T @ U_j. This is
        # exactly the same ridge inverse via Woodbury in observation space.
        design = weighted_design(slice(None))
        gram = design @ design.T
        spectrum(gram)
        gram.flat[:: n + 1] += penalty
        if factors is None:
            features = np.linalg.solve(np.linalg.cholesky(gram), design) / np.sqrt(penalty)
        else:
            lower, _ = factors.cho_factor(gram, lower=True, check_finite=False)
            features = factors.solve_triangular(lower, design, lower=True, check_finite=False)
            if info is not None:
                indices = probe_columns()
                actual = factors.blas.dtrmm(1.0, lower, features[:, indices], lower=1)
                record_residual(indices, actual, design[:, indices])
            features /= np.sqrt(penalty)
            del lower
        del design, gram
    else:
        covariance = np.zeros((width, width))
        batch = max(1, min(1024, 1_000_000 // max(1, width)))
        for start in range(0, n, batch):
            design = weighted_design(slice(start, start + batch))
            covariance += design.T @ design
        spectrum(covariance)
        if width:
            covariance.flat[:: width + 1] += penalty
        if width and factors is not None:
            cholesky = factors.cho_factor(covariance, lower=True, check_finite=False)
            precision = factors.cho_solve(cholesky, np.eye(width), check_finite=False)
            if info is not None:
                indices = probe_columns()
                actual = covariance @ precision[:, indices]
                expected = np.zeros_like(actual)
                expected[indices, np.arange(len(indices))] = 1
                record_residual(indices, actual, expected)
            del cholesky
        else:
            precision = np.linalg.inv(covariance) if width else covariance

    results = {}
    for a, b in combinations(range(len(components)), 2):
        rows_a = valid[a] & (w > 0)[:, None]
        n_valid = rows_a.astype(np.int64).T @ valid[b]
        valid_weight = (w[:, None] * rows_a).T @ valid[b]
        _pair_support_at_cutoff(valid_weight, rows_a, valid[b], w, min_valid_weight)
        scores = np.full(n_valid.shape, np.nan)
        for i, sx in enumerate(slices[a]):
            for j, sy in enumerate(slices[b]):
                if n_valid[i, j] < min_valid or valid_weight[i, j] < min_valid_weight:
                    continue
                coupling = features[:, sx].T @ features[:, sy] if dual else -precision[sx, sy]
                if include_gaps:
                    coupling = coupling[np.ix_(contrast_indices[a][i], contrast_indices[b][j])]
                if coupling.size:
                    coupling = (
                        coupling
                        - coupling.mean(0)
                        - coupling.mean(1, keepdims=True)
                        + coupling.mean()
                    )
                scores[i, j] = np.linalg.norm(coupling)
        results[components[a].name + components[b].name] = DCABlock(
            scores, apc(scores), n_valid, valid_weight
        )
    return results
