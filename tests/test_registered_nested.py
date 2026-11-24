"""Guard blinded six-site holdout scoring, control construction and published denominators."""

import importlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.coordinates import ReferenceCoordinates
from triplet_coevolution.inference.nested import NestedOptions
from triplet_coevolution.validation.clade_splits import make_split_plan

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"


def sample(flip=False):
    count = 40
    ids = tuple(f"o{i:02d}" for i in range(count))
    components = []
    for role in "ABC":
        seq = []
        for i in range(count):
            x, y = i // 2 % 2, i % 2
            states = {"A": (x, y), "B": (y, x), "C": (x ^ y, x ^ y)}
            bits = states[role]
            if flip and role == "C" and i >= 32:
                bits = tuple(1 - bit for bit in bits)
            seq.append("".join("AC"[z] for z in bits))
        components.append(ComponentMSA(role, "dna", tuple(role + k for k in ids), tuple(seq)))
    return MatchedTripletMSA(
        *components,
        tuple(TripletMatch(k, *[c.ids[i] for c in components]) for i, k in enumerate(ids)),
    )


def test_six_site_holdout_uses_only_training_splits(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(BENCH))
    study = importlib.import_module("independent_nested")
    source = sample()
    ids = [m.observation_id for m in source.matches]
    groups = {k: str(i // 8) for i, k in enumerate(ids)}
    plan = make_split_plan(ids, groups, test_groups=["4"])
    maps = tuple(ReferenceCoordinates(r, r + "ref", (0, 1), (1, 2), ("A", "C")) for r in "ABC")
    options = NestedOptions(min_valid=24, min_state_count=8, tolerance=1e-6)
    triples = list(itertools.product((1, 2), repeat=3))
    args = (maps, plan, dict.fromkeys(ids, 1.0), options, triples)
    original = study.loss_fold(source, *args, tmp_path / "a")
    changed = study.loss_fold(sample(True), *args, tmp_path / "b")
    assert original["heldout_complete"] == 8 and original["training_complete"] == 32
    assert original["gain"] > 0 and changed["gain"] < 0
    for field in ("training_design_sha256", "pair_only_sha256", "pair_plus_k_sha256"):
        assert changed[field] == original[field]


def test_negative_controls_keep_declared_units(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCH))
    controls = importlib.import_module("independent_nested").controls
    source = sample()
    ids = [m.observation_id for m in source.matches]
    groups = {k: str(i // 8) for i, k in enumerate(ids)}
    sets, info = controls(source, groups)
    assert info["unchanged_row_assignment_counts_by_component"] == {"A": 40, "B": 0, "C": 0}
    shuf = sets["broken_matching"]
    for i, r in enumerate("ABC"):
        original = source.components[i]
        moved = shuf.components[i]
        assert Counter(original.sequences) == Counter(moved.sequences)
        for row, record in zip(shuf.matches, moved.ids, strict=True):
            assert groups[row.observation_id] == groups[record[len(r) :]]
        per_site = sets["independent_sites"].components[i]
        for group in sorted(set(groups.values())):
            k = [j for j, id_ in enumerate(ids) if groups[id_] == group]
            for t in (0, 1):
                assert sorted(per_site.sequences[j][t] for j in k) == sorted(
                    original.sequences[j][t] for j in k
                )


def test_committed_fold_report_has_all_attempts_and_registered_interval():
    result = json.loads((BENCH / "independent_nested_metrics.json").read_bytes())
    assert result["shared_assemblies"] == 139
    for name, n in (("rnap", 22), ("ribosome", 9)):
        cohort = result["cohorts"][name]
        assert len(cohort["settings"]) == 4
        for setting in cohort["settings"].values():
            rows = setting["folds"]
            assert len(rows) == n and [r["phylum"] for r in rows] == cohort["eligible_phyla"]
            good = [r for r in rows if r["status"] == "evaluated"]
            assert sum(r["heldout_complete"] for r in good) == setting["scored_rows"]
            vals = np.array([r["pair_loss"] - r["plus_k_loss"] for r in good])
            assert np.mean(vals) == pytest.approx(setting["equal_phylum_mean_gain"], abs=1.0e-12)
            rng = np.random.Generator(np.random.PCG64(731903))
            ix = rng.integers(len(vals), size=(2000, len(vals)))
            lo, hi = np.quantile(vals[ix].mean(axis=1), [0.025, 0.975])
            assert setting["equal_phylum_interval"]["lower"] == pytest.approx(lo, abs=1.0e-12)
            assert setting["equal_phylum_interval"]["upper"] == pytest.approx(hi, abs=1.0e-12)
