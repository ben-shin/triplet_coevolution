"""Asymmetric categorical pseudolikelihood DCA on fixed mixed-alphabet site panels."""

from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral, Real
from tempfile import TemporaryFile

import numpy as np

from triplet_coevolution.core.alignment import MatchedTripletMSA
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference._plm_target import ConditionalFit, solve_target, zero_sum
from triplet_coevolution.inference.mi import (
    _check_min_valid,
    _get_weights,
    _pair_support_at_cutoff,
    apc,
    check_min_valid_weight,
)
from triplet_coevolution.inference.potts import PAIRS, DCABlock


@dataclass(frozen=True)
class PLMOptions:
    lambda_h: float = 0.01
    lambda_j: float = 0.01
    include_gaps: bool = False
    missing_policy: str = "complete"
    max_iterations: int = 250
    tolerance: float = 1e-6
    workers: int = 1
    batch_rows: int = 256
    memory_mib: int = 512
    scratch_mib: int = 512

    def __post_init__(self):
        for name in ("lambda_h", "lambda_j", "tolerance"):
            v = getattr(self, name)
            if not isinstance(v, Real) or isinstance(v, bool) or not isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")
            object.__setattr__(self, name, float(v))
        for name in ("workers", "batch_rows", "max_iterations", "memory_mib", "scratch_mib"):
            v = getattr(self, name)
            if not isinstance(v, Integral) or isinstance(v, bool) or v < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(v))
        if type(self.include_gaps) is not bool or self.missing_policy != "complete":
            raise ValueError(
                "PLM requires a Boolean include_gaps and missing_policy='complete' across the conditioning panel"
            )


@dataclass(frozen=True)
class PLMResult:
    blocks: dict[str, DCABlock]
    diagnostics: dict


def _groups(conditioning):
    if conditioning not in {"joint", "pairwise"}:
        raise ValueError("PLM conditioning must be joint or pairwise")
    return ((0, 1, 2),) if conditioning == "joint" else PAIRS


def _target_bytes(n, dimensions, options):
    qmax = max(dimensions)
    width = sum(dimensions)
    params = max(q * (1 + width - q) for q in dimensions)
    # Eight s/y history pairs, direction, trial, objective/gradient and recent vectors.
    byte_count = (
        8 * (32 * params + 8 * min(n, options.batch_rows) * qmax)
        + 32 * n
        + 16 * n * len(dimensions)
        + 32 * min(n * width, 65536)
    )
    return params, byte_count


def plm_preflight(
    msa: MatchedTripletMSA, options: PLMOptions | None = None, *, conditioning: str = "joint"
) -> dict:
    """Budget all conditioned fits, reporting worst simultaneous RAM and temporary file size."""
    opt = options or PLMOptions()
    widths = [c.n_sites for c in msa.components]
    qs = [len(CANONICAL[c.molecule]) for c in msa.components]
    n = msa.n_observations
    cells = sum(widths[a] * widths[b] for a, b in PAIRS)
    peaks = []
    disk = []
    max_params = workers = work = 0
    for group in _groups(conditioning):
        dims = [qs[i] + opt.include_gaps for i in group for _ in range(widths[i])]
        parallel = min(len(dims), opt.workers)
        params, target = _target_bytes(n, dims, opt)
        max_params = max(params, max_params)
        workers = max(workers, parallel)
        peaks.append(parallel * target + n * sum(widths) * 40 + 128 * cells + 65536)
        disk.append(
            sum(
                widths[a] * widths[b] * qs[a] * qs[b] * 8
                for a, b in PAIRS
                if a in group and b in group
            )
        )
        work += n * sum(dims) * (len(dims) - 1)
    peak = max(peaks)
    scratch = max(disk)
    if peak > opt.memory_mib * 1024**2:
        raise ValueError(
            f"PLM estimated numeric workspace {peak} bytes exceeds memory_mib={opt.memory_mib}"
        )
    if scratch > opt.scratch_mib * 1024**2:
        raise ValueError(
            f"PLM directional coefficient scratch {scratch} bytes exceeds scratch_mib={opt.scratch_mib}"
        )
    return {
        "status": "ready",
        "conditioning": conditioning,
        "estimated_numeric_bytes": peak,
        "scratch_bytes": scratch,
        "max_target_parameters": max_params,
        "workers": workers,
        "one_gradient_work_estimate": work,
        "one_gradient_work_unit": "row/predictor/target-state contributions across fitted targets (one full sweep)",
        "optimizer": "L-BFGS(8), safeguarded Armijo backtracking, gradient infinity-norm stop",
        "memory_scope": "numeric work, concurrent target fits and output blocks; excludes Python objects and supplied input strings",
    }


def fit_conditional(
    codes: np.ndarray,
    alphabet_sizes: Sequence[int],
    target: int,
    weights=None,
    *,
    options: PLMOptions | None = None,
    min_valid: int = 2,
    min_valid_weight: float = 0.0,
) -> ConditionalFit:
    """Fit one target in a pre-encoded panel. Negative one denotes a masked native state."""
    opt = options or PLMOptions()
    x = np.asarray(codes)
    dims = tuple(alphabet_sizes)
    if x.ndim != 2 or x.dtype.kind not in "iu" or len(dims) != x.shape[1] or not len(x) or not dims:
        raise ValueError(
            "codes must be a nonempty row-by-site integer matrix aligned with alphabets"
        )
    if any(not isinstance(q, Integral) or isinstance(q, bool) or q < 2 for q in dims):
        raise ValueError("every fixed categorical alphabet must contain at least two states")
    if not isinstance(target, Integral) or isinstance(target, bool) or not 0 <= target < len(dims):
        raise ValueError("target must be an in-panel integer column")
    dims = tuple(map(int, dims))
    _, size = _target_bytes(len(x), dims, opt)
    if size > opt.memory_mib * 1024**2:
        raise ValueError("PLM conditional optimization exceeds the memory budget")
    if np.any(x < -1) or np.any(x >= np.asarray(dims)):
        raise ValueError("categorical states must be within their declared support or -1")
    w = _get_weights(weights, len(x))
    return solve_target(
        x,
        dims,
        int(target),
        w,
        opt,
        _check_min_valid(min_valid),
        check_min_valid_weight(min_valid_weight),
    )


def _fit_panel(msa, group, opt, w, minimum, min_weight, sink, diagnostics):
    components = msa.components
    widths = [c.n_sites for c in components]
    qs = [len(CANONICAL[c.molecule]) for c in components]
    sites = [(role, i) for role in group for i in range(widths[role])]
    dimensions = tuple(qs[r] + opt.include_gaps for r, _ in sites)
    codes = np.column_stack([components[r].matrix[:, i] for r, i in sites])
    codes[codes >= np.asarray(dimensions)] = -1
    design = (w > 0) & (codes >= 0).all(axis=1)
    native = {r: (components[r].matrix >= 0) & (components[r].matrix < qs[r]) for r in group}
    arrays = {}
    base = {}
    offset = 0
    for a, b in PAIRS:
        if a not in group or b not in group:
            continue
        keep = native[a] & design[:, None]
        counts = keep.astype(np.int64).T @ native[b]
        valid_weight = (w[:, None] * keep).T @ native[b]
        _pair_support_at_cutoff(valid_weight, keep, native[b], w, min_weight)
        arrays[a, b] = (np.full(counts.shape, np.nan), counts, valid_weight)
        base[a, b] = offset
        offset += widths[a] * widths[b] * qs[a] * qs[b] * 8

    def fit(i):
        return solve_target(codes, dimensions, i, w, opt, minimum, min_weight)

    success = []
    label = "".join("ABC"[r] for r in group)
    parallel = min(len(sites), opt.workers)
    with (
        TemporaryFile() as scratch,
        ThreadPoolExecutor(parallel) if parallel > 1 else nullcontext() as pool,
    ):
        for start in range(0, len(sites), parallel):
            targets = range(start, min(len(sites), start + parallel))
            fits = map(fit, targets) if pool is None else pool.map(fit, targets)
            for t, result in zip(targets, fits, strict=True):
                r, site = sites[t]
                diagnostics.append(
                    {**result.diagnostics, "panel": label, "role": "ABC"[r], "retained_index": site}
                )
                success.append(result.fields is not None)
                if result.fields is None:
                    continue
                for j, coeff in result.coefficients.items():
                    other, other_site = sites[j]
                    if other == r:
                        continue
                    if r < other:
                        a, b, i, k = r, other, site, other_site
                    else:
                        a, b, i, k = other, r, other_site, site
                    scores, counts, weights = arrays[a, b]
                    if counts[i, k] < minimum or weights[i, k] < min_weight:
                        continue
                    block = coeff.T if r < other else coeff
                    block = zero_sum(block[: qs[a], : qs[b]])
                    size = qs[a] * qs[b] * 8
                    scratch.seek(base[a, b] + (i * widths[b] + k) * size)
                    if r < other:
                        scratch.write(block.tobytes())
                    elif success[j]:
                        rows = design & native[a][:, i] & native[b][:, k]
                        left = components[a].matrix[rows, i]
                        right = components[b].matrix[rows, k]
                        if np.all(left == left[0]) or np.all(right == right[0]):
                            continue
                        reverse = np.frombuffer(scratch.read(size), dtype=np.float64).reshape(
                            block.shape
                        )
                        combined = (block + reverse) / 2
                        scores[i, k] = np.linalg.norm(combined)
                        if sink is not None:
                            sink("ABC"[a] + "ABC"[b], i, k, combined)
    return {
        "ABC"[a] + "ABC"[b]: DCABlock(s, apc(s), n, weights)
        for (a, b), (s, n, weights) in arrays.items()
    }


def fit_plm(
    msa: MatchedTripletMSA,
    weights=None,
    *,
    options: PLMOptions | None = None,
    min_valid: int = 2,
    min_valid_weight: float = 0,
    conditioning: str = "joint",
    coupling_sink: Callable[[str, int, int, np.ndarray], None] | None = None,
) -> PLMResult:
    """Fit asymmetric fields and intra/inter pairs, then reconcile canonical inter-role scores."""
    opt = options or PLMOptions()
    plan = plm_preflight(msa, opt, conditioning=conditioning)
    minimum = _check_min_valid(min_valid)
    mass = check_min_valid_weight(min_valid_weight)
    w = _get_weights(weights, msa.n_observations, msa)
    targets = []
    blocks = {}
    for group in _groups(conditioning):
        blocks.update(_fit_panel(msa, group, opt, w, minimum, mass, coupling_sink, targets))
    statuses = dict(Counter(t["status"] for t in targets))
    total = sum(np.isfinite(b.frobenius).sum() for b in blocks.values())
    unresolved = any(
        k in statuses for k in ("iteration_limit", "line_search_failed", "nonfinite_objective")
    )
    return PLMResult(
        blocks,
        {
            "backend": "plm",
            "solver": "independent asymmetric weighted categorical conditionals",
            "status": "partial_nonconvergence"
            if unresolved
            else "ok"
            if total
            else "no_estimable_pairs",
            "options": asdict(opt),
            "conditioning": conditioning,
            "preflight": plan,
            "objective": "per-target weighted mean negative log conditional probability in nats plus lambda_h/2 ||h||² and lambda_j/2 sum ||J||²",
            "predictor_coding": "one-hot minus weighted observed frequency of the predictor in complete positive-weight panel rows",
            "gauge": "center both canonical block axes, transpose reverse and average before Frobenius norm; gap excluded",
            "support": "positive eligible panel rows with both endpoints canonical; constant canonical endpoints withheld",
            "finite_scores": int(total),
            "target_status_counts": statuses,
            "targets": targets,
        },
    )
