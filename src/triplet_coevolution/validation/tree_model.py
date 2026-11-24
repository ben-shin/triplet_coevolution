"""Independent stationary equal-input sites evolving on fixed, supplied trees."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from math import isfinite
from numbers import Real

import numpy as np

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.mi import _get_weights
from triplet_coevolution.validation.tree_topology import ValidatedTree, compare_trees, load_tree

__all__ = [
    "IndependentSiteTree",
    "ValidatedTree",
    "compare_trees",
    "f81_transition",
    "load_tree",
    "prepare_tree_null",
    "tree_workspace",
]


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _probabilities(values):
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] < 2 or raw.dtype.kind not in "iuf":
        raise ValueError("stationary frequencies must be numeric probability vectors")
    p = raw.astype(float, copy=True)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(np.abs(p.sum(axis=1) - 1) > 1e-12):
        raise ValueError("stationary probabilities must be finite, nonnegative and sum to one")
    p /= p.sum(axis=1, keepdims=True)
    return p


def _heterozygosity(pi):
    # 2 sum_{i<j} p_i p_j avoids cancellation for nearly constant sites.
    return 2 * np.sum(pi[:, 1:] * np.cumsum(pi[:, :-1], axis=1), axis=1)


def _redraw_probability(lengths, rate, het):
    if not rate:
        return np.zeros((len(lengths), len(het)))
    with np.errstate(over="ignore", under="ignore"):
        scaled = lengths[:, None] * rate
        return -np.expm1(
            -np.divide(scaled, het, out=np.zeros((len(lengths), len(het))), where=het > 0)
        )


def f81_transition(
    frequencies: Sequence[float], length: float, *, rate_scale: float = 1.0
) -> np.ndarray:
    """F81 transition matrix at one site in expected substitutions per variable site."""
    raw = np.asarray(frequencies)
    if raw.ndim != 1:
        raise ValueError("one site's probability vector is required")
    pi = _probabilities(raw[None, :])
    t = _nonnegative(length, "branch length")
    rate = _nonnegative(rate_scale, "rate_scale")
    replace = _redraw_probability(np.array([t]), rate, _heterozygosity(pi))[0, 0]
    return (1 - replace) * np.eye(pi.size) + replace * np.broadcast_to(pi, (pi.size, pi.size))


def _draw_stationary(cdf, u):
    states = np.empty(u.shape, dtype=np.uint8)
    for site in range(u.shape[1]):
        states[:, site] = np.searchsorted(cdf[site], u[:, site], side="right")
    return states


def _block_size(nodes, width):
    # At most one million node/site slots apart from a single very large column.
    return min(width, 128, max(1, 1_048_576 // nodes))


def tree_workspace(nodes: int, observations: int, widths: Sequence[int]) -> int:
    """Conservative per-draw numeric array allowance excluding the downstream scorer and strings."""
    cols = max(widths)
    return int(
        64 * nodes * _block_size(nodes, cols)
        + 16 * observations * sum(widths)
        + 20 * 8 * sum(widths) * 4
    )


@dataclass(frozen=True)
class IndependentSiteTree:
    msa: MatchedTripletMSA = field(repr=False)
    trees: tuple[ValidatedTree, ...] = field(repr=False)
    frequencies: tuple[np.ndarray, ...] = field(repr=False)
    support_weight: tuple[np.ndarray, ...] = field(repr=False)
    frequency_policy: str
    prior_mass: float
    rate_scale: float

    def metadata(self) -> dict:
        tree_meta = {c.name: tree.metadata() for c, tree in zip(self.msa.components, self.trees)}
        sites = {}
        for c, pi, w in zip(self.msa.components, self.frequencies, self.support_weight):
            sites[c.name] = {
                "alphabet": CANONICAL[c.molecule],
                "frequencies": pi.tolist(),
                "canonical_support_weight": w.tolist(),
                "constant_sites_0based": np.flatnonzero(_heterozygosity(pi) == 0).tolist(),
                "no_positive_canonical_data_sites_0based": np.flatnonzero(w == 0).tolist(),
            }
        discord = {}
        for i, j in combinations(range(3), 2):
            if self.trees[i] is not self.trees[j]:
                discord["ABC"[i] + "ABC"[j]] = compare_trees(self.trees[i], self.trees[j])
        return {
            "model": "independent_site_tree",
            "process": "stationary reversible equal-input CTMC (F81)",
            "branch_units": "expected substitutions per nonconstant site at rate_scale=1",
            "frequency_policy": self.frequency_policy,
            "prior_mass_per_site": self.prior_mass,
            "rate_scale": self.rate_scale,
            "site_indexing": "zero-based retained columns in archive order",
            "mask_policy": "simulate canonical states at all tips, restore each original gap and ambiguous symbol",
            "trees": tree_meta,
            "component_tree_comparisons": discord,
            "sites": sites,
            "tree_uncertainty": "conditional on supplied topology, mapping, branch lengths and rate scale",
            "estimated_numeric_workspace_bytes": tree_workspace(
                max(len(t.parents) for t in self.trees),
                self.msa.n_observations,
                [c.n_sites for c in self.msa.components],
            ),
        }

    def draw_codes(self, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        """Draw latent canonical tip codes in matched row order before reapplying observed masks."""
        draws = []
        for comp, tree, pi in zip(self.msa.components, self.trees, self.frequencies):
            n = len(tree.parents)
            step = _block_size(n, comp.n_sites)
            result = np.empty((len(tree.tips), comp.n_sites), dtype=np.uint8)
            for start in range(0, comp.n_sites, step):
                stop = min(comp.n_sites, start + step)
                h = _heterozygosity(pi[start:stop])
                cdf = np.cumsum(pi[start:stop], axis=1)
                # Assign floating-point remainder to the last positive state. A
                # trailing zero-frequency symbol must be unreachable even at a rounding tie.
                last = pi.shape[1] - 1 - (pi[start:stop, ::-1] > 0).argmax(axis=1)
                cdf[np.arange(pi.shape[1])[None, :] >= last[:, None]] = 1
                state = np.empty((n, stop - start), dtype=np.uint8)
                state[:1] = _draw_stationary(cdf, rng.random((1, stop - start)))
                for level in tree.levels[1:]:
                    inherited = state[tree.parents[level]]
                    change = _redraw_probability(tree.lengths[level], self.rate_scale, h)
                    moves = rng.random(change.shape) < change
                    if moves.any():
                        reset = _draw_stationary(cdf, rng.random(moves.shape))
                        np.copyto(inherited, reset, where=moves)
                    state[level] = inherited
                result[:, start:stop] = state[tree.tips]
            draws.append(result)
        return tuple(draws)

    def draw(self, rng: np.random.Generator) -> MatchedTripletMSA:
        """Generate full matched component rows, preserving every ID and original missing/gap symbol."""
        parts = []
        for comp, states in zip(self.msa.components, self.draw_codes(rng)):
            symbols = CANONICAL[comp.molecule]
            canonical = (comp.matrix >= 0) & (comp.matrix < len(symbols))
            rendered = []
            for original, row, keep in zip(comp.sequences, states, canonical):
                result = bytearray(original.encode("ascii"))
                simulated = np.frombuffer(symbols.encode("ascii"), dtype=np.uint8)[row]
                out = np.frombuffer(result, dtype=np.uint8)
                out[keep] = simulated[keep]
                rendered.append(result.decode("ascii"))
            parts.append(ComponentMSA(comp.name, comp.molecule, comp.ids, tuple(rendered)))
        return MatchedTripletMSA(*parts, self.msa.matches)


def prepare_tree_null(
    msa: MatchedTripletMSA,
    tree: ValidatedTree | Mapping[str, ValidatedTree],
    *,
    weights=None,
    frequency_policy: str = "empirical",
    frequencies: Mapping | None = None,
    prior_mass: float = 0.0,
    rate_scale: float = 1.0,
) -> IndependentSiteTree:
    """Freeze the tree(s), frequencies and observed design before any simulated refits."""
    ids = tuple(m.observation_id for m in msa.matches)
    if isinstance(tree, ValidatedTree):
        trees = (tree,) * 3
    elif isinstance(tree, Mapping) and set(tree) == set("ABC"):
        trees = tuple(tree[c] for c in "ABC")
    else:
        raise ValueError("supply one validated tree or validated trees for roles A, B and C")
    if any(not isinstance(t, ValidatedTree) or t.observation_ids != ids for t in trees):
        raise ValueError("tree tips must match the archived observation ordering")
    rate = _nonnegative(rate_scale, "rate_scale")
    prior = _nonnegative(prior_mass, "prior_mass")
    if frequency_policy not in {"uniform", "empirical", "supplied"}:
        raise ValueError("frequency_policy must be uniform, empirical or supplied")
    if frequency_policy != "empirical" and prior:
        raise ValueError("prior_mass applies only to empirical frequencies")
    if frequency_policy == "supplied":
        if not isinstance(frequencies, Mapping) or set(frequencies) != set("ABC"):
            raise ValueError(
                "supplied frequencies require a probability array for each role A, B and C"
            )
    elif frequencies is not None:
        raise ValueError("explicit frequencies require the supplied policy")
    w = _get_weights(weights, msa.n_observations, msa)
    params, masses = [], []
    for comp in msa.components:
        width, q = comp.n_sites, len(CANONICAL[comp.molecule])
        counts = np.empty((width, q))
        for col in range(width):
            v = comp.matrix[:, col]
            valid = (v >= 0) & (v < q) & (w > 0)
            counts[col] = np.bincount(v[valid], weights=w[valid], minlength=q)
        mass = counts.sum(1)
        if frequency_policy == "empirical":
            if np.any(mass + prior == 0):
                raise ValueError(
                    f"no positive-weight canonical data or prior for {comp.name} retained columns {np.flatnonzero(mass == 0).tolist()}"
                )
            if not np.isfinite(mass + prior).all():
                raise ValueError("stationary data plus prior exceeds numerical range")
            denom = (mass + prior)[:, None]
            p = counts / denom + (prior / denom) / q
        elif frequency_policy == "uniform":
            p = np.full((width, q), 1 / q)
        else:
            supplied = np.asarray(frequencies[comp.name])
            if supplied.shape == (q,):
                supplied = np.broadcast_to(supplied, (width, q))
            if supplied.shape != (width, q):
                raise ValueError(
                    f"{comp.name} supplied frequencies must have shape {(width, q)} or {(q,)}"
                )
            p = _probabilities(supplied)
        p.setflags(write=False)
        mass.setflags(write=False)
        params.append(p)
        masses.append(mass)
    return IndependentSiteTree(
        msa, trees, tuple(params), tuple(masses), frequency_policy, prior, rate
    )
