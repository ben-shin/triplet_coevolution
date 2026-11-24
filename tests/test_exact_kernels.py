from math import prod

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference import mi, weights
from triplet_coevolution.inference.dca_models import DCADesign
from triplet_coevolution.inference.threebody import TripleScore, fit_triples


def make_msa(n=519, types=("dna", "rna", "protein"), lengths=(11, 13, 17), *, boundary=False):
    rng = np.random.default_rng(819)
    components = []
    for role, kind, length in zip("ABC", types, lengths, strict=True):
        alphabet = CANONICAL[kind]
        states = rng.choice(list(alphabet + "-?"), size=(n, length))
        states[:, 0] = rng.choice(list(alphabet), size=n)
        if boundary:
            states[0] = "A"
            states[1] = "A"
            if role == "A":
                states[1, 4:] = "C"  # 4/10 matches on A, 100% on B and C -> mean=.8
            states[2] = states[0]
            states[2, 1:] = "-"  # weak joint overlap
        else:
            states[:, 1] = "?"
            states[:, 2] = "A"
        ids = tuple(f"{role}{i}" for i in range(n))
        components.append(ComponentMSA(role, kind, ids, tuple("".join(row) for row in states)))
    return MatchedTripletMSA(
        *components, tuple(TripletMatch(str(i), *(c.ids[i] for c in components)) for i in range(n))
    )


@pytest.mark.parametrize("threshold", [np.nextafter(0.8, 0), 0.8, np.nextafter(0.8, 1.0)])
@pytest.mark.parametrize("reuse", [False, True])
def test_integer_upper_bound_preserves_fraction_threshold_ties_and_rematching(
    threshold, reuse, monkeypatch
):
    msa = make_msa(261, lengths=(10, 64, 33), boundary=True)
    comp = weights._WeightComparisons(msa.components, reuse=reuse)
    source = comp._sparse_neighbors
    reduced = []

    def observed(*args):
        mask = source(*args)
        if mask is not None:
            reduced.append(mask.shape)
        return mask

    monkeypatch.setattr(comp, "_sparse_neighbors", observed)
    if reuse:
        # Nonmonotonic pair lookups, independent per-component permutations, preserve the tie pair.
        order = np.random.default_rng(37).permutation(np.arange(3, 261))
        reordered = [
            part.take_ids(tuple(part.ids[i] for i in [0, 1, 2, *np.roll(order, k)]))
            for k, part in enumerate(msa.components)
        ]
    else:
        reordered = None
    fitted = comp.weights(threshold, 0.7, reordered)
    monkeypatch.setattr(comp, "_sparse_neighbors", lambda *args: None)
    dense = comp.weights(threshold, 0.7, reordered)
    assert reduced
    np.testing.assert_array_equal(fitted.neighbor_counts, dense.neighbor_counts)
    np.testing.assert_array_equal(fitted.weights, dense.weights)
    assert fitted.neighbor_counts[0] == (2 if threshold <= 0.8 else 1)


@pytest.mark.parametrize("gaps", [False, True])
@pytest.mark.parametrize("scale", [0.125, 1.0, 0.3])
def test_packed_histogram_preserves_raw_apc_support_at_ulp_and_zero_rows(gaps, scale, monkeypatch):
    msa = make_msa()
    w = np.where(np.arange(msa.n_observations) % 11, scale, 0.0)
    orig = mi._packed_unit_mi
    used = []

    def check(*args):
        status = orig(*args)
        used.append(status)
        return status

    monkeypatch.setattr(mi, "_packed_unit_mi", check)
    cutoff = w.sum()
    for threshold in (np.nextafter(cutoff, 0.0), cutoff, np.nextafter(cutoff, np.inf)):
        with monkeypatch.context() as patch:
            patch.setattr(mi, "_packed_unit_mi", lambda *args: False)
            baseline = mi.mutual_information_block(
                msa.a, msa.b, w, include_gaps=gaps, min_valid_weight=threshold
            )
        actual = mi.mutual_information_block(
            msa.a, msa.b, w, include_gaps=gaps, min_valid_weight=threshold
        )
        for name in ("mi_bits", "mi_apc", "n_valid", "valid_weight"):
            np.testing.assert_array_equal(getattr(baseline, name), getattr(actual, name))
    assert all(status == (hasattr(np, "bitwise_count") and scale != 0.3) for status in used)


def scalar_triple(msa, candidate, w):
    cols = [m.matrix[:, j] for m, j in zip(msa.components, candidate)]
    keep = np.logical_and.reduce(
        [
            w > 0,
            *[(x >= 0) & (x < len(CANONICAL[c.molecule])) for x, c in zip(cols, msa.components)],
        ]
    )
    n = int(keep.sum())
    if n == 0:
        return TripleScore(np.nan, np.nan, 0, 0.0, 0, 0)
    indices = [np.unique(col[keep], return_inverse=True)[1] for col in cols]
    shape = tuple(int(i.max() + 1) for i in indices)
    counts = np.bincount(
        np.ravel_multi_index(indices, shape), weights=w[keep], minlength=prod(shape)
    ).reshape(shape)
    logp = np.log(counts + 0.5)
    for axis in range(3):
        logp -= logp.mean(axis=axis, keepdims=True)

    def ent(p):
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    p = counts / w[keep].sum()
    co = (
        sum(ent(p.sum(axis=axes)) for axes in ((0, 1), (0, 2), (1, 2)))
        - sum(ent(p.sum(axis=i)) for i in range(3))
        + ent(p)
    )
    return TripleScore(
        float(np.sqrt(np.mean(logp**2))),
        co,
        n,
        w[keep].sum(),
        np.count_nonzero(counts),
        counts.size,
    )


@pytest.mark.parametrize(
    ("types", "n"),
    [
        (("dna", "rna", "protein"), 519),
        (("protein", "dna", "rna"), 48),
        (("protein",) * 3, 27),
        (("protein",) * 3, 519),
    ],
)
def test_cached_conditional_triple_uses_observed_joint_support(types, n):
    msa = make_msa(n, types)
    w = np.where(np.arange(n) % 13, np.random.default_rng(71).uniform(size=n), 0.0)
    coords = [(2, 3, 5), (3, 3, 5), (1, 7, 4), (0, 0, 0), (3, 3, 5), (2, 2, 2)]
    actual = fit_triples(msa, iter(coords), w)
    for key, score in zip(coords, actual, strict=True):
        expected = scalar_triple(msa, key, w)
        np.testing.assert_array_equal(list(vars(score).values()), list(vars(expected).values()))


@pytest.mark.parametrize(
    ("n", "shape", "solver"), [(14, (9, 8, 7), "dual"), (150, (4, 4, 5), "primal")]
)
@pytest.mark.parametrize("conditioning", ["joint", "pairwise"])
def test_explicit_cholesky_fit_preserves_scores_and_probes(
    n, shape, solver, conditioning, monkeypatch
):
    pytest.importorskip("scipy.linalg")
    msa = make_msa(n, lengths=shape)
    w = np.random.default_rng(71).uniform(size=n)
    base = {
        "backend": "mf",
        "conditioning": conditioning,
        "include_gaps": True,
        "shrinkage": 0.12,
        "alphabet_support": "fixed",
    }
    baseline = DCADesign.parse(base).fit(msa, w, min_valid=2, min_valid_weight=0.7)
    design = DCADesign.parse({**base, "factor_solver": "cholesky"})
    assert (
        DCADesign.parse(design.metadata()) == design
        and "factor_solver" not in DCADesign().metadata()
    )
    if solver == "dual":
        monkeypatch.setattr(np, "eye", lambda *a, **k: pytest.fail("unexpected dual identity"))
        monkeypatch.setattr(
            np.linalg, "inv", lambda *a, **k: pytest.fail("unexpected feature inverse")
        )
    diagnostic = {}
    current = design.fit(msa, w, min_valid=2, min_valid_weight=0.7, diagnostics=diagnostic)
    assert all(
        s["solver"] == solver
        and s["factor_solver"] == "cholesky"
        and s["factor_probe_relative_frobenius_residual"] < 1.0e-12
        for s in diagnostic.values()
    )
    for pair in baseline:
        for name in ("frobenius", "corrected"):
            np.testing.assert_allclose(
                getattr(current[pair], name),
                getattr(baseline[pair], name),
                rtol=2.0e-12,
                atol=1.0e-13,
            )
        for name in ("n_valid", "valid_weight"):
            np.testing.assert_array_equal(
                getattr(current[pair], name), getattr(baseline[pair], name)
            )


def test_bad_solver_is_not_substituted():
    with pytest.raises(ValueError, match="factor_solver"):
        DCADesign(factor_solver="unknown")
    with pytest.raises(ValueError, match="only to mf"):
        DCADesign.parse({"backend": "plm", "factor_solver": "cholesky"})


def test_missing_optional_solver_fails_in_preflight_and_fit(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "scipy", None)
    msa = make_msa(12)
    design = DCADesign(factor_solver="cholesky")
    with pytest.raises(ValueError, match="optimize extra"):
        design.preflight(msa)
    with pytest.raises(ValueError, match="optimize extra"):
        design.fit(msa, None, min_valid=2, min_valid_weight=0)
