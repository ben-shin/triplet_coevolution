import csv

import pytest

from triplet_coevolution.automatic_panel import _shape, make_scoring_panel
from triplet_coevolution.io.metadata import read_triplet_table
from triplet_coevolution.io.stockholm import open_stockholm


def _prepared(tmp_path, *, sizes=(8, 8, 8), total=24, reverse=False):
    prepared = tmp_path / "prepared"
    prepared.mkdir(parents=True)
    kwargs = {"matches": prepared / "matches.tsv"}
    header = ["observation_id", "a_id", "b_id", "c_id", "stratum", "role_evidence"]
    with kwargs["matches"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        for i in reversed(range(total)) if reverse else range(total):
            writer.writerow(
                [
                    f"g{i:03d}",
                    *[f"{c}{i}" for c in "ABC"],
                    "small" if i < 4 else "middle" if i < 12 else "large",
                    "orthogroup_supported",
                ]
            )
    components = {}
    for c, width, kind in zip("ABC", sizes, ("protein", "dna", "rna")):
        key = c.lower()
        components[c] = {"source_columns": width, "molecule": kind}
        path = prepared / f"{c}.sto"
        reference = f"{c}_coordinate/50-{50 + width - 2}"
        kwargs.update({key: path, f"{key}_ref": f"{c} display name", f"{key}_ref_row": reference})
        canonical = {"protein": "DELF", "dna": "AGCT", "rna": "AGCU"}[kind]
        with path.open("w", encoding="utf-8") as stream:
            stream.write("# STOCKHOLM 1.0\n")
            stream.write(f"#=GS {reference} DE original coordinate reference\n")
            stream.write(f"{reference} -{canonical[0] * (width - 1)}\n")
            for i in range(total):
                sequence = "".join(canonical[(i + j) % 4] for j in range(width))
                stream.write(f"{c}{i} {sequence}\n")
            stream.write("//\n")
    return prepared, kwargs, components


def _select(project, *, weight_limit=12 * 12 * 24, pair_limit=99_999, **kw):
    return make_scoring_panel(
        *project, {"pair_cells": pair_limit, "weight_cells": weight_limit}, **kw
    )


def test_stable_depth_subset_keeps_originals_and_true_extra_coordinates(tmp_path):
    project = _prepared(tmp_path / "project")
    originals = {p: p.read_bytes() for p in project[0].iterdir()}
    args, audit = _select(project)
    reordered = _prepared(tmp_path / "reordered", reverse=True)
    replay_args, repeated = _select(reordered)
    assert audit == repeated
    assert audit["n_input"] == 24 and audit["n_selected"] == 12
    assert audit["site_panel"] is None
    assert audit["site_method"] == "all_source_columns"
    assert args["matches"].read_bytes() == replay_args["matches"].read_bytes()
    assert len(set(audit["selected_observation_ids"])) == 12
    selected = read_triplet_table(args["matches"])
    assert [r.observation_id for r in selected.matches] == audit["selected_observation_ids"]
    assert all(m["role_evidence"] == "orthogroup_supported" for m in selected.annotations)
    for c in "abc":
        ref = project[1][f"{c}_ref_row"]
        with open_stockholm(args[c]) as inv:
            assert [row.row_id for row in inv.rows()] == [
                *(getattr(m, f"{c}_id") for m in selected.matches),
                ref,
            ]
            assert (inv.row(ref).start, inv.row(ref).end) == (50, 56)
            assert inv.row(ref).annotations == (("DE", "original coordinate reference"),)
    assert all(p.read_bytes() == content for p, content in originals.items())
    _, other = _select(project, seed=42)
    assert other["selected_observation_ids"] != audit["selected_observation_ids"]


def test_stratification_and_exact_sparse_original_columns(tmp_path):
    project = _prepared(tmp_path, sizes=(15, 21, 27))
    _, stratified = _select(project, weight_limit=12 * 12 * 63, strata_column="stratum")
    assert stratified["strata_counts"] == {
        "small": {"input": 4, "selected": 2},
        "middle": {"input": 8, "selected": 4},
        "large": {"input": 12, "selected": 6},
    }
    args, sparse = _select(project, weight_limit=100_000, pair_limit=75)
    assert sparse["selected_widths"] == dict.fromkeys("ABC", 5)
    assert sparse["source_widths"] == dict(zip("ABC", (15, 21, 27)))
    with args["site_panel"].open(encoding="utf-8") as stream:
        cols = list(csv.DictReader(stream, delimiter="\t"))
    assert [int(r["original_column"]) for r in cols if r["component"] == "A"] == [2, 5, 8, 11, 14]
    assert [int(r["original_column"]) for r in cols if r["component"] == "B"] == [3, 7, 11, 15, 19]
    assert [int(r["original_column"]) for r in cols if r["component"] == "C"] == [3, 9, 14, 19, 25]
    for c, width in sparse["source_widths"].items():
        with open_stockholm(args[c.lower()]) as inv:
            assert inv.n_columns == width


@pytest.mark.parametrize(
    "options",
    [
        {"pair_limit": 11},
        {"select_columns": False, "pair_limit": 40},
        {"min_valid": 25},
        {"weight_limit": 7 * 7 * 6},
    ],
)
def test_infeasible_panel_has_audited_outcome_without_creating_subset(tmp_path, options):
    project = _prepared(tmp_path)
    args, audit = _select(project, **options)
    assert not args and audit["status"] == "resource_limit" and audit["n_selected"] == 0
    assert not list(project[0].glob("panel*"))


def test_large_panel_fits_permutation_and_dca_planning_budgets():
    limits = {"pair_cells": 1_000_000, "weight_cells": 4_000_000_000}
    widths = [1500, 2500, 1800]
    n, sites = _shape(widths, ["protein", "protein", "rna"], 50_000, limits, 2, True)
    assert n == 128 and len(set(sites)) == 1
    pairs = sites[0] * sites[1] + sites[0] * sites[2] + sites[1] * sites[2]
    assert pairs * n <= 4_000_000 and n * n * sum(sites) <= 80_000_000
    assert n * (20 * sites[0] + 20 * sites[1] + 4 * sites[2]) <= 3_000_000
    assert _shape([1_000_000] * 3, ["protein"] * 3, 24, limits, 2, True)[0] == 0
