"""Score definitions, complete-block corrections and deterministic ranking."""

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from numbers import Integral
from typing import Literal

import numpy as np

from triplet_coevolution.core.coordinates import ReferenceCoordinates
from triplet_coevolution.inference.mi import MIBlock, TripletMI, apc, check_min_valid_weight
from triplet_coevolution.inference.potts import PAIRS, DCABlock

Method = Literal["mi", "dca", "threebody"]
APCUniverse = Literal["retained", "reference"]
Direction = Literal["higher", "lower"]


@dataclass(frozen=True)
class ScoreSpec:
    """One numerical statistic, including its one-sided alternative."""

    name: str
    method: Method
    field: str
    backend: str
    direction: Direction
    units: str
    correction: str
    conditioning: str
    universe: str
    missing_policy: str
    interpretation: str

    def metadata(
        self, *, apc_universe: APCUniverse = "retained", threebody_model: str = "local"
    ) -> dict[str, str]:
        """Resolve method and support descriptions for a particular run."""
        check_apc_universe(apc_universe)
        if threebody_model not in ("local", "global"):
            raise ValueError("threebody_model must be local or global")
        data = asdict(self)
        if self.correction == "APC":
            data["universe"] = f"{apc_universe} axes per component pair"
        if self.method == "threebody" and threebody_model == "global":
            data["missing_policy"] = "positive-weight complete rows across the entire fitted panel"
            if self.name == "threebody_rms":
                data.update(
                    backend="L2 global categorical pseudolikelihood",
                    conditioning="all retained sites, with shared pair and selected triple terms",
                )
        return data

    def values(self, block: MIBlock | DCABlock) -> np.ndarray:
        """Select the actual numerical matrix from a matching estimator block."""
        if isinstance(block, MIBlock) and self.method == "mi":
            return getattr(block, self.field)
        if isinstance(block, DCABlock) and self.method == "dca":
            return getattr(block, self.field)
        raise ValueError(f"score {self.name!r} is incompatible with {type(block).__name__}")


_MI_MISSING = "jointly observed positive-weight rows, gap policy as configured, min_valid"
_DCA_MISSING = "centered mean imputation for fit, pairwise canonical overlap for min_valid"
_LOCAL_MISSING = "complete canonical positive-weight rows for each candidate, min_valid"
_MI = ScoreSpec(
    "mi_bits", "mi", "mi_bits", "weighted empirical joint frequencies", "higher", "bits",
    "none", "two-site marginal", "retained column pairs", _MI_MISSING,
    "Marginal statistical dependence.",
)
_DCA = ScoreSpec(
    "dca_frobenius", "dca", "frobenius", "ridge mean-field DCA", "higher",
    "coupling norm (dimensionless)", "none", "all retained sites of A, B and C",
    "retained column pairs", _DCA_MISSING, "Frobenius norm of a zero-sum coupling block.",
)
_TRIPLE = ScoreSpec(
    "threebody_rms", "threebody", "threebody_rms", "local smoothed log-linear model", "higher",
    "nats", "none", "three-site marginal", "prespecified or sampled ABC candidates", _LOCAL_MISSING,
    "RMS of a zero-sum fitted triple term.",
)
_COINFO = ScoreSpec(
    "co_information_bits", "threebody", "co_information_bits", "weighted empirical entropies",
    "lower", "bits", "none", "three-site marginal", _TRIPLE.universe, _LOCAL_MISSING,
    "Signed co-information. More negative values favor synergy in the net information balance.",
)
_REGISTRY = {s.name: s for s in (
    _MI, replace(_MI, name="mi_apc", field="mi_apc", correction="APC",
                 interpretation="Marginal MI after blockwise average-product correction."),
    _DCA, replace(_DCA, name="dca_apc", field="corrected", correction="APC",
                  interpretation="Coupling norm after blockwise average-product correction."),
    _TRIPLE, _COINFO,
)}


def get_score(name: str, method: Method | None = None, *, units: str | None = None) -> ScoreSpec:
    """Get a registered statistic and reject incompatible method or units."""
    if not isinstance(name, str) or name not in _REGISTRY:
        raise ValueError(f"unknown score {name!r}, choose from {', '.join(_REGISTRY)}")
    spec = _REGISTRY[name]
    if method is not None and spec.method != method:
        raise ValueError(f"score {name!r} belongs to {spec.method}, not {method}")
    if units is not None and units != spec.units:
        raise ValueError(f"score {name!r} requires units {spec.units!r}")
    return spec


def check_apc_universe(policy: str) -> None:
    if policy not in ("retained", "reference"):
        raise ValueError("apc_universe must be retained or reference")


def coordinate_axes(
    maps: Sequence[ReferenceCoordinates], require_reference: bool,
) -> tuple[np.ndarray, ...]:
    """Indices in canonical original-column order, independent of reference strand."""
    return tuple(
        np.flatnonzero([pos is not None for pos in m.reference_positions])
        if require_reference else np.arange(m.n_sites)
        for m in maps
    )


def _corrected(raw: np.ndarray, axes: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    result = np.full(raw.shape, np.nan)
    if all(axis.size for axis in axes):
        ix = np.ix_(*axes)
        result[ix] = apc(raw[ix])
    result.setflags(write=False)
    return result


def mi_with_apc_policy(
    result: TripletMI, maps: Sequence[ReferenceCoordinates], apc_universe: APCUniverse = "retained",
) -> TripletMI:
    """Return full-shaped MI blocks. Reference-only APC is NaN outside its fit axes."""
    check_apc_universe(apc_universe)
    if apc_universe == "retained" and result.apc_universe == "retained":
        return result
    axes = coordinate_axes(maps, apc_universe == "reference")
    updates = {}
    for a, b in PAIRS:
        pair = "abc"[a] + "abc"[b]
        block = getattr(result, pair)
        updates[pair] = replace(block, mi_apc=_corrected(block.mi_bits, (axes[a], axes[b])))
    return replace(result, **updates, apc_universe=apc_universe)


def dca_with_apc_policy(
    blocks: dict[str, DCABlock], maps: Sequence[ReferenceCoordinates], apc_universe: APCUniverse,
) -> dict[str, DCABlock]:
    """Choose the correction domain of a newly fitted complete DCA matrix."""
    check_apc_universe(apc_universe)
    if apc_universe == "retained":
        return blocks
    axes = coordinate_axes(maps, True)
    result = {}
    for a, b in PAIRS:
        pair = "ABC"[a] + "ABC"[b]
        block = blocks[pair]
        result[pair] = replace(block, corrected=_corrected(block.frobenius, (axes[a], axes[b])))
    return result


def ranked_indices(
    values: np.ndarray, score_name: str, limit: int | None = None,
) -> Iterator[tuple[int, int, int]]:
    """Yield flat C index, sequential display rank and exact-tie competition rank.

    Supply values in canonical coordinate order. Nonfinite scores have no rank.
    Top-K chooses earliest coordinates at the cutoff and only sorts the retained K.
    """
    spec = get_score(score_name)
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1
    ):
        raise ValueError("rank limit must be a positive integer or None")
    flat = np.asarray(values, dtype=float).ravel()
    if limit is None:
        finite = np.flatnonzero(np.isfinite(flat))
    else:
        # Keep only the best K across chunks. Equal scores retain their original
        # order even when an exact tie crosses a chunk boundary or the cutoff.
        finite = np.empty(0, dtype=np.intp)
        chunk_size = max(65536, limit)
        for start in range(0, flat.size, chunk_size):
            chunk = flat[start:start + chunk_size]
            indices = np.concatenate((finite, start + np.flatnonzero(np.isfinite(chunk))))
            if indices.size <= limit:
                finite = indices
                continue
            scores = flat[indices]
            if spec.direction == "higher":
                scores *= -1
            cutoff = np.partition(scores, limit - 1)[limit - 1]
            best = indices[scores < cutoff]
            equal = indices[scores == cutoff][:limit - best.size]
            finite = np.concatenate((best, equal))
    keys = flat[finite]
    if spec.direction == "higher":
        keys *= -1
    order = finite[np.argsort(keys, kind="stable")]
    competition = 0
    previous = None
    for rank, index in enumerate(order, 1):
        value = flat[index]
        if value != previous:
            competition = rank
            previous = value
        yield int(index), rank, competition


def pair_coverage(
    block: MIBlock | DCABlock, x_map: ReferenceCoordinates, y_map: ReferenceCoordinates,
    score: str, *, require_reference: bool = False, min_valid: int = 2,
    min_valid_weight: float = 0.0, max_pairs: int | None = 1000, apc_universe: APCUniverse = "retained",
) -> dict[str, int | bool]:
    """Report the successive coordinate, support, finite-score and display filters."""
    spec = get_score(score)
    check_apc_universe(apc_universe)
    values = spec.values(block)
    mapped_only = require_reference or (spec.correction == "APC" and apc_universe == "reference")
    ix = np.ix_(*coordinate_axes((x_map, y_map), mapped_only))
    selected = values[ix]
    count_ok = block.n_valid[ix] >= min_valid
    support = count_ok & (block.valid_weight[ix] >= check_min_valid_weight(min_valid_weight))
    finite = int(np.count_nonzero(np.isfinite(selected) & support))
    eligible = selected.size
    supported = int(support.sum())
    exported = finite if max_pairs is None else min(finite, max_pairs)
    return {
        "total": values.size,
        "eligible": eligible,
        "finite": finite,
        "excluded_coordinates": values.size - eligible,
        "below_min_valid": int(eligible - count_ok.sum()),
        **(
            {"below_min_valid_weight_after_count": int(count_ok.sum()) - supported}
            if min_valid_weight
            else {}
        ),
        "nonfinite_after_support": supported - finite,
        "exported": exported,
        "complete": exported == finite,
    }
