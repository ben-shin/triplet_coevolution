"""Bounded joint pair-model bootstrap. Draw independent endpoints after checked burn-in."""

from dataclasses import asdict, dataclass, fields
from math import isfinite

import numpy as np

from triplet_coevolution.inference.nested_solver import _integer


@dataclass(frozen=True)
class GibbsOptions:
    burn_sweeps: int = 32
    chains: int = 4
    diagnostic_sweeps: int = 256
    max_rhat: float = 1.05
    min_ess: int = 100
    max_total_tv: float = 0.001
    local_i3: bool = False

    def __post_init__(self):
        for name, lower, upper in (
            ("burn_sweeps", 1, 4096),
            ("chains", 4, 16),
            ("diagnostic_sweeps", 64, 8192),
            ("min_ess", 20, 100000),
        ):
            _integer(getattr(self, name), name, lower, upper)
        if self.diagnostic_sweeps % 2 or self.min_ess > self.chains * self.diagnostic_sweeps:
            raise ValueError("diagnostic_sweeps must be even and the pilot must cover min_ess")
        for name, lower, upper in (("max_rhat", 1.0, 1.1), ("max_total_tv", 0.0, 0.01)):
            v = getattr(self, name)
            if type(v) not in (int, float) or not isfinite(v) or not lower < v <= upper:
                raise ValueError(f"{name} must be in ({lower}, {upper}]")
        if type(self.local_i3) is not bool:
            raise TypeError("local_i3 must be boolean")

    def metadata(self):
        return {"method": "gibbs", **asdict(self)}


def parse_sampler(value):
    if value is None or isinstance(value, GibbsOptions):
        return value
    if not isinstance(value, dict):
        raise TypeError("null_sampler must be an object with method exact or gibbs")
    if value == {"method": "exact"}:
        return None
    if value.get("method") != "gibbs" or set(value) - {
        "method",
        *(f.name for f in fields(GibbsOptions)),
    }:
        raise ValueError("unknown null_sampler method or settings")
    return GibbsOptions(**{k: v for k, v in value.items() if k != "method"})


def _diagnostic(values):
    """Classical split-Rhat and multichain initial-positive monotone-sequence ESS."""
    chains, steps = values.shape
    x = values.reshape(2 * chains, steps // 2).astype(float)
    m, n = x.shape
    means = x.mean(axis=1)
    x -= means[:, None]
    within = float(np.sum(x * x) / (m * (n - 1)))
    if within == 0:
        return None, 0.0
    variance = (n - 1) / n * within + float(np.var(means, ddof=1))
    rhat = max(1.0, float(np.sqrt(variance / within)))
    transformed = np.fft.rfft(x, n=2 * n, axis=1)
    auto = (
        np.fft.irfft((transformed * transformed.conj()).real, n=2 * n, axis=1)[:, :n].mean(axis=0)
        / n
    )
    rho = 1 - (within - auto) / variance
    rho[0] = 1.0
    pairs = rho[: n // 2 * 2].reshape(-1, 2).sum(axis=1)
    nonpositive = np.flatnonzero(pairs <= 0)
    positive = pairs[: int(nonpositive[0])] if nonpositive.size else pairs
    tau = max(1.0, 2 * np.minimum.accumulate(positive).sum() - 1.0)
    return rhat, float(m * n / tau)


def _workspace(layout, opt, rows, draws, folds):
    dims = layout.dimensions
    if any(len(term) > 2 for term in layout.terms):
        raise ValueError("Gibbs bootstrap requires a shared pair-only Hamiltonian")
    pilot = opt.chains * opt.diagnostic_sweeps
    conditional_cells = sum(sum(shape) for shape in layout.shapes)
    sweeps = opt.chains * (opt.burn_sweeps + opt.diagnostic_sweeps) + rows * draws * opt.burn_sweeps
    # One FFT time series at a time, plus the pilot trajectory, bounded endpoint
    # batches, restored row arrays and field/adjacency views.
    memory = (
        16 * 1024
        + pilot * (len(dims) + 144)
        + rows * len(dims) * 16
        + min(rows, 256) * (max(dims) * 64 + 8 * len(dims))
        + layout.size * 32
        + 8 * len(dims) ** 2
        + 1536 * len(layout.terms)
        + 512 * sum(dims)
    )
    indicators = sum(dims) + len(layout.terms) + 1
    certificate_work = opt.burn_sweeps * len(dims) ** 2 + sum(
        shape[0] * shape[1] ** 2 for shape in layout.shapes if len(shape) == 2
    )
    work = folds * (
        sweeps * conditional_cells
        + pilot * indicators * int(np.ceil(np.log2(opt.diagnostic_sweeps)))
        + certificate_work
    )
    if work > 300_000_000:
        raise ValueError(
            f"Gibbs planned updates/diagnostics exceed the 300 million work limit ({work})"
        )
    if memory > min(256 * 1024**2, layout.options.max_workspace_bytes):
        raise ValueError(
            f"Gibbs sampler workspace {memory} bytes exceeds the configured memory limit"
        )
    return {
        "sampler_workspace_bytes": memory,
        "planned_work_units_all_folds": work,
        "endpoint_batch": 256,
    }


class _PairChains:
    """Systematic scan of shared zero-sum fields and pair tensors in log probability units."""

    def __init__(self, layout, theta):
        self.dims = layout.dimensions
        self.fields = []
        self.neighbors = [[] for _ in self.dims]
        self.edges = []
        self.influence = np.zeros((len(self.dims), len(self.dims)))
        for term, tensor in zip(layout.terms, layout.arrays(theta), strict=True):
            if len(term) == 1:
                self.fields.append(tensor)
            else:
                i, j = term
                self.neighbors[i].append((j, tensor.T))
                self.neighbors[j].append((i, tensor))
                self.edges.append((i, j, tensor))
                # Largest odds-ratio cross difference over two states at each endpoint.
                oscillation = max(
                    float(np.ptp(tensor - tensor[:, b, None], axis=0).max())
                    for b in range(tensor.shape[1])
                )
                self.influence[i, j] = self.influence[j, i] = np.tanh(oscillation / 4)

    def probabilities(self, rows, site):
        logits = np.broadcast_to(self.fields[site], (len(rows), self.dims[site])).copy()
        for j, neighbor in self.neighbors[site]:
            logits += neighbor[rows[:, j]]
        logits -= logits.max(axis=1, keepdims=True)
        np.exp(logits, out=logits)
        return logits / logits.sum(axis=1, keepdims=True)

    def sweep(self, rows, uniforms):
        for i in range(len(self.dims)):
            cdf = np.cumsum(self.probabilities(rows, i), axis=1)
            rows[:, i] = (uniforms[i, :, None] >= cdf[:, :-1]).sum(axis=1)

    def burn_bound(self, sweeps):
        # Couple to a stationary chain. After replacing site i, its mismatch
        # probability is bounded by sum_j Cij*dj, using updated sites in scan order.
        delta = np.ones(len(self.dims))
        for _ in range(sweeps):
            for i, c in enumerate(self.influence):
                delta[i] = min(1.0, float(c @ delta))
        return min(1.0, float(delta.sum()))

    def endpoints(self, n, opt, rng):
        result = np.empty((n, len(self.dims)), dtype=np.int16)
        for start in range(0, n, 256):
            block = result[start : start + 256]
            block[:] = rng.integers(self.dims, size=block.shape)
            for _ in range(opt.burn_sweeps):
                self.sweep(block, rng.random((len(self.dims), len(block))))
        return result

    def pilot(self, opt, rng):
        n = len(self.dims)
        seed_words = rng.integers(2**32, size=(opt.chains, 4), dtype=np.uint32)
        streams = [
            np.random.Generator(np.random.PCG64(np.random.SeedSequence(s))) for s in seed_words
        ]
        rows = np.stack([r.integers(self.dims, size=n) for r in streams])
        rows[0] = 0
        rows[1] = np.asarray(self.dims) - 1
        paths = np.empty((opt.chains, opt.diagnostic_sweeps, n), dtype=np.uint8)
        for _ in range(opt.burn_sweeps):
            self.sweep(rows, np.stack([r.random(n) for r in streams], axis=1))
        for k in range(opt.diagnostic_sweeps):
            self.sweep(rows, np.stack([r.random(n) for r in streams], axis=1))
            paths[:, k] = rows
        checks = []

        def check(name, series):
            rhat, ess = _diagnostic(series)
            ok = rhat is not None and rhat <= opt.max_rhat and ess >= opt.min_ess
            checks.append({"observable": name, "split_rhat": rhat, "ess": ess, "pass": bool(ok)})

        for i, q in enumerate(self.dims):
            if q > 1:
                for state in range(q):
                    check(f"state[{i}]={state}", paths[:, :, i] == state)
        energy = np.zeros(paths.shape[:2])
        for i, field in enumerate(self.fields):
            energy += field[paths[:, :, i]]
        for i, j, edge in self.edges:
            contribution = edge[paths[:, :, i], paths[:, :, j]]
            energy += contribution
            if np.ptp(edge) > 0:
                check(f"pair_energy[{i},{j}]", contribution)
        if any(np.ptp(f) > 0 for f in self.fields) or any(np.ptp(e) > 0 for _, _, e in self.edges):
            check("joint_log_potential", energy)
        return checks, seed_words.tolist()


def prepare_gibbs(layout, theta, opt, *, rows, draws, folds, rng):
    """Preflight and independently check mixing once per fixed fitted pair law."""
    rows = _integer(rows, "endpoint rows", 1)
    draws = _integer(draws, "draws", 1)
    folds = _integer(folds, "folds", 1)
    budget = _workspace(layout, opt, rows, draws, folds)
    theta = np.asarray(theta, dtype=float)
    if not np.allclose(theta, layout.projected(theta), rtol=1e-10, atol=1e-10):
        raise ValueError("Gibbs pair model requires centered zero-sum tensors")
    chains = _PairChains(layout, theta)
    checks, seed_words = chains.pilot(opt, rng)
    bound = chains.burn_bound(opt.burn_sweeps)
    run_bound = min(1.0, rows * draws * folds * bound)
    failed = [r["observable"] for r in checks if not r["pass"]]
    return chains, {
        **budget,
        "settings": opt.metadata(),
        "single_endpoint_tv_bound": bound,
        "endpoints_per_draw": rows,
        "requested_replicates": draws,
        "planned_folds": folds,
        "fold_endpoint_tv_bound": min(1.0, rows * draws * bound),
        "all_endpoint_tv_bound_at_this_fold_rate": run_bound,
        "endpoint_tv_budget_per_fold": opt.max_total_tv / folds,
        "mixing_status": "passed" if not failed and run_bound <= opt.max_total_tv else "withheld",
        "indicator_failures": failed,
        "indicator_diagnostics": checks,
        "burn_certificate_pass": run_bound <= opt.max_total_tv,
        "max_influence_row_sum": float(chains.influence.sum(axis=1).max()),
        "pilot_start": "all first states; all last states; remaining chains uniformly dispersed",
        "pilot_rng": "PCG64(SeedSequence(four uint32 words per chain))",
        "pilot_seed_words": seed_words,
        "scan": "ascending site index, one complete sweep updates every site",
        "endpoints": "one independent chain per synthetic observation; retain final state only",
    }
