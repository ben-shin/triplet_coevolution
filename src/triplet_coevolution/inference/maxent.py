"""Finite-support maximum-entropy hierarchy for three categorical variables."""

from dataclasses import dataclass, fields, replace
from math import isfinite, prod, sqrt
from numbers import Integral, Real
from typing import Literal

import numpy as np

from triplet_coevolution.core.alignment import MatchedTripletMSA
from triplet_coevolution.inference.maxent_support import FrozenSupport, freeze_support
from triplet_coevolution.inference.mi import (
    _check_min_valid,
    _get_weights,
    canonical_support_at_cutoff,
    check_min_valid_weight,
)

NAN = float("nan")
PAIR_AXES = ((2,), (1,), (0,))
SINGLE_AXES = ((1, 2), (0, 2), (0, 1))


@dataclass(frozen=True)
class MaxEntOptions:
    prior_mode: Literal["total", "per_cell"] = "total"
    prior_mass: float = 0.0
    tolerance: float = 1e-9
    max_iterations: int = 1000
    max_cells: int = 100_000

    def __post_init__(self):
        if self.prior_mode not in ("total", "per_cell"):
            raise ValueError("prior_mode must be total or per_cell")
        if (
            isinstance(self.prior_mass, bool)
            or not isinstance(self.prior_mass, Real)
            or not isfinite(self.prior_mass)
            or self.prior_mass < 0
        ):
            raise ValueError("prior_mass must be finite and nonnegative")
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, Real)
            or not 0 < self.tolerance < 1
        ):
            raise ValueError("tolerance must be finite and between zero and one")
        for name, maximum in (("max_iterations", 100_000), ("max_cells", 1_000_000)):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, Integral) or not 1 <= val <= maximum:
                raise ValueError(f"{name} must be an integer between 1 and {maximum}")


@dataclass(frozen=True)
class MaxEntScore:
    status: str
    n_valid: int | None
    valid_weight: float
    observed_cells: int
    possible_cells: int
    connected_information_bits: float = NAN
    target_kl_bits: float = NAN
    co_information_bits: float = NAN
    target_co_information_bits: float = NAN
    threebody_rms: float = NAN
    hx_bits: float = NAN
    hy_bits: float = NAN
    hz_bits: float = NAN
    h1_bits: float = NAN
    h2_bits: float = NAN
    h3_bits: float = NAN
    entropy_difference_bits: float = NAN
    identity_error_bits: float = NAN
    m1_log_likelihood_nats: float = NAN
    m2_log_likelihood_nats: float = NAN
    m3_log_likelihood_nats: float = NAN
    deviance_nats: float = NAN
    target_deviance_nats: float = NAN
    prior_total: float = NAN
    prior_fraction: float = NAN
    prior_data_ratio: float = NAN
    expected_occupied_m2: float = NAN
    m1_kind: str = "unavailable"
    ipf_iterations: int = 0
    ipf_residual: float = NAN
    ipf_relative_residual: float = NAN
    m1_iterations: int = 0
    m1_residual: float = NAN
    active_pair_face_cells: int = 0
    information_status: str = "unavailable"


MAXENT_FIELDS = tuple(f.name for f in fields(MaxEntScore))


@dataclass(frozen=True)
class JointFit:
    score: MaxEntScore
    p1: np.ndarray | None = None
    p2: np.ndarray | None = None
    p3: np.ndarray | None = None


@dataclass(frozen=True)
class IPFFit:
    status: str
    distribution: np.ndarray | None
    iterations: int
    residual: float
    relative_residual: float
    active_cells: int


def _validate_shape(shape, options):
    if len(shape) != 3 or any(n < 1 for n in shape):
        raise ValueError("joint support must have three nonempty axes")
    if prod(shape) > options.max_cells:
        raise ValueError(f"joint support {tuple(shape)} exceeds max_cells={options.max_cells}")


def _allowed_mask(shape, allowed):
    if allowed is None:
        return np.ones(shape, dtype=bool)
    result = np.asarray(allowed)
    if result.shape != tuple(shape) or result.dtype != bool or not result.any():
        raise ValueError(
            "structural support must be a nonempty boolean mask matching the joint shape"
        )
    return result


def _ipf(shape, support, targets, axes, options, *, joint=None):
    active = support.copy()
    for margin, reduced in zip(targets, axes, strict=True):
        active &= np.expand_dims(margin > 0, axis=reduced)
    size = int(active.sum())

    def failed(status, iteration=0, residual=NAN, relative=NAN):
        return IPFFit(status, None, iteration, residual, relative, size)

    if size == 0 or any(
        np.any((margin > 0) & ~active.any(axis=reduced))
        for margin, reduced in zip(targets, axes, strict=True)
    ):
        return failed("incompatible_support")
    # A uniform starting measure on *all* feasible cells selects maximum entropy.
    # Unseen cells remain eligible unless explicitly forbidden or forced to zero.
    p = active.astype(float) / size
    target_entropy = _entropy(joint) if joint is not None else None
    identity_atol = max(options.tolerance, 128 * np.finfo(float).eps * max(1, np.log2(size)))
    limit_status = "iteration_limit"
    for iteration in range(1, options.max_iterations + 1):
        for target, reduced in zip(targets, axes, strict=True):
            current = p.sum(axis=reduced, keepdims=True)
            desired = np.expand_dims(target, axis=reduced)
            if np.any((current == 0) & (desired > 0)):
                return failed("numerical_underflow", iteration)
            # Normalize conditionally first so no huge target/current ratio forms.
            np.divide(p, current, out=p, where=current > 0)
            p *= desired
        if np.any(active & (p == 0)) or not np.isfinite(p).all():
            return failed("numerical_underflow", iteration)
        residual = relative = 0.0
        for target, reduced in zip(targets, axes, strict=True):
            err = np.abs(p.sum(axis=reduced) - target)
            residual = max(residual, float(err.max()))
            relative = max(
                relative,
                float(np.divide(err, target, out=np.zeros_like(err), where=target > 0).max()),
            )
        if residual <= options.tolerance and relative <= sqrt(options.tolerance):
            if (
                joint is not None
                and abs(_kl(joint, p) - (_entropy(p) - target_entropy)) > identity_atol
            ):
                limit_status = "entropy_identity_unresolved"
                continue
            p.setflags(write=False)
            return IPFFit("ok", p, iteration, residual, relative, size)
        limit_status = "iteration_limit"
    return failed(limit_status, options.max_iterations, residual, relative)


def fit_pair_marginals(
    xy: np.ndarray,
    xz: np.ndarray,
    yz: np.ndarray,
    *,
    options: MaxEntOptions | None = None,
    allowed: np.ndarray | None = None,
) -> IPFFit:
    """Fit M2 to three supplied probability margins, with explicit feasibility diagnostics."""
    opts = options or MaxEntOptions()
    pair = tuple(np.asarray(v, dtype=float) for v in (xy, xz, yz))
    if any(v.ndim != 2 for v in pair):
        raise ValueError("XY, XZ and YZ targets must be two-dimensional")
    shape = (pair[0].shape[0], pair[0].shape[1], pair[1].shape[1])
    _validate_shape(shape, opts)
    if pair[1].shape != (shape[0], shape[2]) or pair[2].shape != (shape[1], shape[2]):
        raise ValueError("XY, XZ and YZ marginal axes disagree")
    if any(not np.isfinite(v).all() or np.any(v < 0) for v in pair):
        raise ValueError("pair targets must be finite nonnegative probabilities")
    support = _allowed_mask(shape, allowed)
    # Normalization/compatibility is an input property, independent of a caller's
    # chosen convergence tolerance.
    tol = 64 * np.finfo(float).eps * max(shape)
    if any(abs(v.sum() - 1) > tol for v in pair):
        raise ValueError("pair marginals must each sum to one")
    overlap = (
        (pair[0].sum(1), pair[1].sum(1)),
        (pair[0].sum(0), pair[2].sum(1)),
        (pair[1].sum(0), pair[2].sum(0)),
    )
    if any(np.max(np.abs(a - b)) > tol for a, b in overlap):
        return IPFFit("inconsistent_margins", None, 0, NAN, NAN, 0)
    return _ipf(shape, support, pair, PAIR_AXES, opts)


def _entropy(p):
    positive = p[p > 0]
    return float(-np.dot(positive, np.log2(positive)))


def _coinformation(p):
    singles = tuple(_entropy(p.sum(axis=a)) for a in SINGLE_AXES)
    return sum(singles) - sum(_entropy(p.sum(axis=a)) for a in PAIR_AXES) + _entropy(p), singles


def _kl(p, q):
    # Sum generalized KL cell contributions. The series avoids cancellation
    # for two almost identical tables; both probability tables have unit mass.
    positive = p > 0
    pv, qv = p[positive], q[positive]
    near = np.abs(pv - qv) < 1e-4 * qv
    delta = (pv[near] - qv[near]) / qv[near]
    small = qv[near] * delta**2 * (0.5 - delta / 6 + delta**2 / 12)
    far = ~near
    large = pv[far] * (np.log(pv[far]) - np.log(qv[far])) - pv[far] + qv[far]
    return float((small.sum() + large.sum() + q[~positive].sum()) / np.log(2))


def _likelihood(counts, p):
    observed = counts > 0
    return float(np.dot(counts[observed], np.log(p[observed])))


def _target_rms(p):
    if np.any(p <= 0):
        return NAN
    terms = np.log(p)
    for axis in range(3):
        terms -= terms.mean(axis=axis, keepdims=True)
    return float(np.sqrt(np.mean(terms**2)))


def fit_joint(
    counts: np.ndarray,
    *,
    options: MaxEntOptions | None = None,
    allowed: np.ndarray | None = None,
    n_valid: int | None = None,
) -> JointFit:
    """Compare the M1/M2/M3 distributions on one prespecified support.

    Counts can be fractional observation weights. n_valid, when given, is the
    number of actual complete positive-weight rows for the occupancy diagnostic.
    """
    opts = options or MaxEntOptions()
    arr = np.asarray(counts)
    _validate_shape(arr.shape, opts)
    if arr.dtype.kind not in "iuf":
        raise TypeError("joint counts must be numeric and nonboolean")
    arr = arr.astype(float, copy=False)
    if not np.isfinite(arr).all() or np.any(arr < 0):
        raise ValueError("joint counts must be finite and nonnegative")
    support = _allowed_mask(arr.shape, allowed)
    if np.any((arr > 0) & ~support):
        raise ValueError("positive data count in a declared structural zero")
    present = int(np.count_nonzero(arr))
    if n_valid is not None and (
        isinstance(n_valid, bool)
        or not isinstance(n_valid, Integral)
        or n_valid < present
        or (n_valid > 0 and not present)
    ):
        raise ValueError("n_valid must be an integer compatible with observed joint cells")
    n_valid = int(n_valid) if n_valid is not None else None
    cells = int(support.sum())
    weight = float(arr.sum())
    prior_total = float(opts.prior_mass) * (cells if opts.prior_mode == "per_cell" else 1)
    if not np.isfinite(weight + prior_total):
        raise ValueError("total count plus prior exceeds the floating-point range")
    base = MaxEntScore("no_data", n_valid, weight, present, cells, prior_total=prior_total)
    if not present:
        return JointFit(base)
    # Scale separately before adding so individual low counts are preserved.
    p3 = arr / (weight + prior_total)
    p3[support] += (prior_total / (weight + prior_total)) / cells
    if np.any((arr > 0) & (p3 == 0)) or (prior_total > 0 and np.any(support & (p3 == 0))):
        return JointFit(replace(base, status="numerical_underflow"))
    p3 /= p3.sum()
    p3.setflags(write=False)
    empirical = arr / weight
    signed, _ = _coinformation(empirical)
    target_signed, (hx, hy, hz) = _coinformation(p3)
    h3 = _entropy(p3)
    active_counts = [np.count_nonzero(empirical.sum(axis=axis)) for axis in SINGLE_AXES]
    info_status = (
        "constant_empirical_axis"
        if min(active_counts) < 2
        else "sparse_observed_support"
        if present < cells
        else "all_cells_observed"
    )
    base = replace(
        base,
        co_information_bits=signed,
        target_co_information_bits=target_signed,
        threebody_rms=_target_rms(p3),
        h3_bits=h3,
        hx_bits=hx,
        hy_bits=hy,
        hz_bits=hz,
        m3_log_likelihood_nats=_likelihood(arr, p3),
        prior_fraction=prior_total / (weight + prior_total),
        prior_data_ratio=prior_total / weight,
        information_status=info_status,
    )
    singles = tuple(p3.sum(axis=a) for a in SINGLE_AXES)
    p1 = singles[0][:, None, None] * singles[1][None, :, None] * singles[2][None, None, :]
    if np.any(p1[~support] > 0):
        m1 = _ipf(arr.shape, support, singles, SINGLE_AXES, opts)
        base = replace(
            base,
            m1_kind="support_constrained_singles",
            m1_iterations=m1.iterations,
            m1_residual=m1.residual,
        )
        if m1.status != "ok":
            return JointFit(replace(base, status="m1_" + m1.status), p3=p3)
        p1 = m1.distribution
    else:
        base = replace(base, m1_kind="independent_product", m1_residual=0.0)
    p1.setflags(write=False)
    if np.any((arr > 0) & (p1 == 0)):
        return JointFit(replace(base, status="numerical_underflow"), p3=p3)
    base = replace(base, h1_bits=_entropy(p1), m1_log_likelihood_nats=_likelihood(arr, p1))
    m2 = _ipf(
        arr.shape, support, tuple(p3.sum(axis=a) for a in PAIR_AXES), PAIR_AXES, opts, joint=p3
    )
    base = replace(
        base,
        status=m2.status,
        ipf_iterations=m2.iterations,
        ipf_residual=m2.residual,
        ipf_relative_residual=m2.relative_residual,
        active_pair_face_cells=m2.active_cells,
    )
    if m2.status != "ok":
        return JointFit(base, p1=p1, p3=p3)
    p2 = m2.distribution
    h2 = _entropy(p2)
    kl = _kl(p3, p2)
    gap = h2 - h3
    err = abs(kl - gap)
    base = replace(base, h2_bits=h2, entropy_difference_bits=gap, identity_error_bits=err)
    occupied = NAN
    if n_valid is not None:
        some = p2[(p2 > 0) & (p2 < 1)]
        occupied = float(np.count_nonzero(p2 == 1) - np.expm1(n_valid * np.log1p(-some)).sum())
    ll2 = _likelihood(arr, p2)
    prior_only_axis = prior_total > 0 and any(
        n < 2 and np.count_nonzero(s) > 1 for n, s in zip(active_counts, singles)
    )
    return JointFit(
        replace(
            base,
            status="prior_only_axis" if prior_only_axis else "ok",
            connected_information_bits=NAN if prior_only_axis else kl,
            target_kl_bits=kl,
            m2_log_likelihood_nats=ll2,
            deviance_nats=2 * (base.m3_log_likelihood_nats - ll2),
            target_deviance_nats=2 * (weight + prior_total) * np.log(2) * kl,
            expected_occupied_m2=occupied,
        ),
        p1,
        p2,
        p3,
    )


def fit_maxent_grid(
    msa: MatchedTripletMSA,
    candidates,
    weights=None,
    *,
    options_sequence,
    support: FrozenSupport | None = None,
    min_valid: int = 2,
    min_valid_weight: float = 0.0,
    structural_support=None,
) -> list[list[MaxEntScore]]:
    """Count each triple once and fit one model per distinct prior/solver setting.

    structural_support optionally maps candidate tuples to boolean masks on their
    fixed bin axes. Weights are never used as multiplicities for the valid-row count.
    """
    settings = tuple(options_sequence)
    if not settings or any(not isinstance(o, MaxEntOptions) for o in settings):
        raise ValueError("supply at least one typed MaxEntOptions setting")
    declared = support if support is not None else freeze_support(msa)
    if tuple(c.molecule.value for c in msa.components) != declared.molecules or tuple(
        c.n_sites for c in msa.components
    ) != tuple(map(len, declared.maps)):
        raise ValueError(
            "frozen state map must match the retained alignment alphabets and dimensions"
        )
    threshold = _check_min_valid(min_valid)
    weight_threshold = check_min_valid_weight(min_valid_weight)
    w = _get_weights(weights, msa.n_observations, msa)
    unique = dict.fromkeys(settings)
    result = [[] for _ in settings]
    previous = {}
    for candidate in candidates:
        candidate = tuple(candidate)
        shape = declared.shape_for(candidate)
        if candidate in previous:
            for series in result:
                series.append(series[previous[candidate]])
            continue
        previous[candidate] = len(result[0])
        maps = [np.array(sites[i]) for i, sites in zip(candidate, declared.maps)]
        columns = [c.matrix[:, i] for c, i in zip(msa.components, candidate)]
        valid = w > 0
        for col, mapping in zip(columns, maps):
            valid &= (col >= 0) & (col < len(mapping))
        n = int(valid.sum())
        weight = canonical_support_at_cutoff(float(w[valid].sum()), valid, w, weight_threshold)
        indices = np.ravel_multi_index(tuple(m[c[valid]] for m, c in zip(maps, columns)), shape)
        mask = structural_support.get(candidate) if structural_support is not None else None
        if mask is not None:
            mask = _allowed_mask(shape, mask)
            if not mask.ravel()[indices].all():
                raise ValueError(
                    f"positive observation in structural zero for candidate {candidate}"
                )
        cells = prod(shape) if mask is None else int(mask.sum())
        scores = {}
        counts = None
        for opt in unique:
            reason = (
                "resource_limit"
                if prod(shape) > opt.max_cells
                else "no_data"
                if not n
                else "insufficient_rows"
                if n < threshold
                else "insufficient_weight"
                if weight < weight_threshold
                else ""
            )
            if reason:
                scores[opt] = MaxEntScore(reason, n, weight, len(np.unique(indices)), cells)
                continue
            if counts is None:
                counts = np.bincount(indices, weights=w[valid], minlength=prod(shape)).reshape(
                    shape
                )
            fit = fit_joint(counts, options=opt, allowed=mask, n_valid=n)
            scores[opt] = replace(fit.score, valid_weight=weight)
        for series, opt in zip(result, settings):
            series.append(scores[opt])
    return result


def fit_maxent_triples(
    msa: MatchedTripletMSA,
    candidates,
    weights=None,
    *,
    support: FrozenSupport | None = None,
    options: MaxEntOptions | None = None,
    min_valid: int = 2,
    min_valid_weight: float = 0.0,
    structural_support=None,
) -> list[MaxEntScore]:
    """Fit local fixed-support maximum-entropy hierarchies on selected ABC site triples."""
    return fit_maxent_grid(
        msa,
        candidates,
        weights,
        support=support,
        options_sequence=(options or MaxEntOptions(),),
        min_valid=min_valid,
        min_valid_weight=min_valid_weight,
        structural_support=structural_support,
    )[0]
