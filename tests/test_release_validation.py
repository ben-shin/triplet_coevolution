"""Release regressions for source coordinates, calibration declarations and process replay."""

import json
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from triplet_coevolution.cli import app
from triplet_coevolution.core.alignment import ComponentMSA
from triplet_coevolution.core.coordinates import (
    ReferenceCoordinates,
    build_reference_map,
    reference_positions,
)
from triplet_coevolution.validation.benchmark import read_label_table
from triplet_coevolution.validation.clade_splits import identity, load_split_plan, make_split_plan
from triplet_coevolution.validation.null_policy import NullPolicy
from triplet_coevolution.validation.pair_sampler import enumerate_pair_law, fit_pair_panel
from triplet_coevolution.validation.shards_local import main as local_shards
from triplet_coevolution.validation.structure import read_contacts
from triplet_coevolution.validation.tree_design import TreeDesign


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"reference_id": ""}, "reference_id"),
        ({"component": "X"}, "component"),
        ({"original_columns": (0, 0, 2)}, "strictly increasing"),
        ({"original_columns": (-1, 1, 2)}, "nonnegative"),
        ({"symbols": ("AC", "-", "G")}, "single characters"),
        ({"reference_positions": (7, 8, 6)}, "gaps"),
        ({"reference_positions": (0, None, 5)}, "positive integers"),
        ({"strand": "?"}, "strand"),
        ({"reference_positions": (8, None, 6)}, "precede"),
        ({"reference_positions": (7, None, 8)}, "progress"),
    ],
)
def test_reloaded_reverse_reference_rejects_invalid_coordinate_records(change, reason):
    source = ReferenceCoordinates(
        "B", "reference", (0, 2, 3), (7, None, 6), ("A", "-", "G"), 7, "-"
    )
    with pytest.raises(ValueError, match=reason):
        replace(source, **change)


def test_reverse_reference_bounds_respect_biological_numbering():
    assert reference_positions("A-.?G", 3, strand="-") == (3, None, None, 2, 1)
    with pytest.raises(ValueError, match="below position 1"):
        reference_positions("A-.?G", 2, strand="-")
    with pytest.raises(ValueError, match="strand"):
        reference_positions("AC", strand="reverse")
    protein = ComponentMSA("A", "protein", ("r",), ("AC",))
    with pytest.raises(ValueError, match="nucleotide"):
        build_reference_map(protein, "r", strand="-")


@pytest.mark.parametrize(
    ("mod", "reason"),
    [
        ({"schema_version": 2}, "schema"),
        ({"null": "other"}, "unknown null"),
        ({"statistics": []}, "nonempty"),
        ({"statistics": ["mi_bits", "mi_bits"]}, "unique"),
        ({"draws": True}, "positive integer"),
        ({"seed": -1}, "nonnegative"),
        ({"planned_draws": 1}, "cover"),
        ({"family": "unregistered"}, "interface"),
        ({"alpha": float("nan")}, "alpha"),
        ({"max_t": 1}, "Boolean"),
        ({"strata_column": ""}, "match-table field"),
        ({"resume": True}, "checkpoint"),
        ({"strata_column": "clade"}, "strata"),
        ({"null": "within_stratum"}, "stratum"),
        ({"null": "independent_site_tree"}, "tree design"),
        ({"weight_design": "fixed_design"}, "independent"),
    ],
)
def test_misdeclared_calibration_cannot_reach_a_refit(mod, reason):
    config = {"null": "whole_component", "statistics": ["mi_bits"], "draws": 7}
    with pytest.raises((ValueError, TypeError), match=reason):
        NullPolicy.from_config({**config, **mod})


def test_recomputed_split_digest_cannot_hide_overlapping_train_test_rows(tmp_path):
    sets = {"one": "x", "two": "x", "three": "y"}
    p = make_split_plan(tuple(sets), sets, test_groups=["y"])
    data = json.loads(json.dumps(p.metadata()))
    data["folds"][0]["train_ids"].append("three")
    data["plan_sha256"] = identity({k: v for k, v in data.items() if k != "plan_sha256"})
    path = tmp_path / "split.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="disjoint partition"):
        load_split_plan(path)
    data = p.metadata()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="IDs/order"):
        load_split_plan(path, tuple(reversed(sets)))


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        ("A a 1 B b", "malformed"),
        ("A a 1 B wrong 2 1", "unknown"),
        ("A a 0 B b 2 0", "positive"),
        ("A a 1 A a 2 0", "different components|within-component"),
        ("A a 1 B b 2 2", "0 or 1|0, 1 or blank"),
        ("A a 1 B b 2 0\nB b 2 A a 1 0", "duplicate"),
    ],
)
def test_native_and_legacy_label_readers_refuse_invalid_shared_coordinates(
    tmp_path, record, reason
):
    path = tmp_path / "contacts.tsv"
    header = "x_component x_reference_id x_reference_pos y_component y_reference_id y_reference_pos contact"
    path.write_text((header + "\n" + record + "\n").replace(" ", "\t"))
    maps = [ReferenceCoordinates(role, role.lower(), (0,), (1,), ("A",)) for role in "AB"]
    with pytest.raises(ValueError, match=reason):
        read_contacts(path, maps)
    with pytest.raises(ValueError, match=reason):
        read_label_table(path, {"A": "a", "B": "b"})


@pytest.mark.parametrize(
    ("field", "val", "reason"),
    [
        ("status", "unresolved", "inconsistent"),
        ("distance_angstrom", "nan", "distance"),
        ("distance_angstrom", "-1", "distance"),
        ("split", "test-and-train", "split"),
    ],
)
def test_unknown_geometry_and_overlapping_split_labels_cannot_become_truth(
    tmp_path, field, val, reason
):
    path = tmp_path / "contacts.tsv"
    path.write_text(
        "x_component\tx_reference_id\tx_reference_pos\ty_component\ty_reference_id\ty_reference_pos\tcontact\t"
        + field
        + "\nA\ta\t1\tB\tb\t2\t1\t"
        + val
        + "\n"
    )
    with pytest.raises(ValueError, match=reason):
        read_label_table(path, {"A": "a", "B": "b"})


def test_panel_complete_case_support_and_ineligible_draws():
    states = np.array(list(product(range(2), repeat=3)))
    fit = fit_pair_panel(states, (2, 2, 2))
    assert fit.status == "ok" and np.allclose(fit.probabilities, np.full((2, 2, 2), 1 / 8))
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="missing mask"):
        fit.draw(rng, missing_mask=np.zeros((8, 1), dtype=bool))
    with pytest.raises(ValueError, match="observation weights"):
        fit.local_score(states[:3], (0, 1, 2))
    with pytest.raises(ValueError, match="distinct"):
        fit.local_score(states, (0, 0, 2))
    states[1:, 1] = -1
    insufficient = fit_pair_panel(states, (2, 2, 2), min_valid=2)
    assert insufficient.status == "insufficient_rows"
    with pytest.raises(ValueError, match="insufficient_rows"):
        insufficient.draw(rng)
    assert np.isnan(insufficient.local_score(states, (0, 1, 2)))


@pytest.mark.parametrize(
    ("sizes", "terms", "reason"),
    [
        ([2, 2], {}, "three to eight"),
        ([2, 2, 2], {(1, 0): np.zeros((2, 2))}, "ascending"),
        ([2, 2, 2], {(0, 2): np.zeros((2, 3))}, "shape"),
        ([2, 2, 2], {(0, 2): np.array([[0, 0], [0, 1000]])}, "probability range"),
    ],
)
def test_generative_pair_coefficients_require_a_normalizable_declared_grid(sizes, terms, reason):
    with pytest.raises(ValueError, match=reason):
        enumerate_pair_law([np.zeros(n) for n in sizes], terms)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"origin": "confirmed-by-scores"}, "origin"),
        ({"components": {"A": "a.nwk"}}, "exactly A, B, C"),
        ({"newick": ""}, "requires a Newick"),
        ({"source": "\n"}, "nonempty"),
        ({"frequency_policy": "supplied"}, "explicit frequencies"),
        ({"frequency_policy": "uniform", "prior_mass": 0.1}, "prior mass"),
        ({"alternatives": {}}, "alternatives|alternative"),
        (
            {"alternatives": [{"label": "PRIMARY", "source": "independent", "newick": "b.nwk"}]},
            "unique",
        ),
    ],
)
def test_tree_sensitivity_refuses_ambiguous_provenance_and_nuisance_design(changes, reason):
    with pytest.raises(ValueError, match=reason):
        TreeDesign.parse({"source": "synthetic", "newick": "a.nwk", **changes})


def test_local_launcher_scoped_budget_replay_and_verify(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    ex = Path(__file__).resolve().parents[1] / "examples/mixed"
    fit = tmp_path / "fit"
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--a",
            str(ex / "protein.fasta"),
            "--b",
            str(ex / "dna.fasta"),
            "--c",
            str(ex / "rna.fasta"),
            "--b-type",
            "dna",
            "--c-type",
            "rna",
            "--a-ref",
            "prot_ref",
            "--b-ref",
            "dna_ref",
            "--c-ref",
            "rna_ref",
            "--matches",
            str(ex / "matches.tsv"),
            "--out",
            str(fit),
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = tmp_path / "policy.json"
    cfg.write_text(
        json.dumps(
            {"draws": 5, "statistics": ["mi_bits"], "null": "whole_component", "memory_mib": 32}
        )
    )
    plan = tmp_path / "plan"
    base = ["--plan", str(plan), "--memory-mib", "640", "--jobs", "1"]
    new = ["--run", str(fit), "--config", str(cfg), "--shards", "2"]
    with pytest.raises(SystemExit):
        local_shards([*base, *new, "--memory-mib", "1"])
    local_shards([*base, *new, "--out", str(tmp_path / "first")])
    with pytest.raises(SystemExit):
        local_shards([*base, *new])
    with pytest.raises(SystemExit):
        local_shards([*base, "--memory-mib", "1"])
    local_shards([*base, "--resume", "--out", str(tmp_path / "next")])
    assert (tmp_path / "next/calibration.tsv").read_bytes() == (
        tmp_path / "first/calibration.tsv"
    ).read_bytes()
