"""End-to-end mathematical controls and provenance for local three-site inference."""

import csv
import json
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from triplet_coevolution.cli import app


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def rows(path):
    with path.open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="," if path.suffix == ".csv" else "\t"))


def invoke(name, args, out, status=0):
    result = CliRunner().invoke(app, [name, *args, "--out", str(out)])
    assert result.exit_code == status, (result.output, result.exception)
    return result


def sources(directory, kind="xor"):
    if kind == "xor":
        cells = [(a, b, a ^ b) for a, b in product(range(2), repeat=2)] * 2
    else:
        cells = list(product(range(2), repeat=3))
    fasta = [directory / f"{k}.fa" for k in "ABC"]
    for i, (name, path) in enumerate(zip("ABC", fasta, strict=True)):
        seqs = [f">{name}-ref\nA-C"]
        for j, cell in enumerate(cells):
            seqs.append(f">{name}{j}\n{'AC'[cell[i]]}-C")
        path.write_text("\n".join(seqs) + "\n")
    matches = directory / "matches.tsv"
    matches.write_text(
        "observation_id\ta_id\tb_id\tc_id\tunit\n"
        + "".join(f"r{j}\tA{j}\tB{j}\tC{j}\t1\n" for j in range(len(cells)))
    )
    return [
        "--a",
        str(fasta[0]),
        "--b",
        str(fasta[1]),
        "--c",
        str(fasta[2]),
        "--a-ref",
        "A-ref",
        "--b-ref",
        "B-ref",
        "--c-ref",
        "C-ref",
        "--b-type",
        "dna",
        "--c-type",
        "rna",
        "--c-strand",
        "-",
        "--c-start",
        "11",
        "--matches",
        str(matches),
        "--weight-policy",
        "supplied",
        "--weight-column",
        "unit",
    ]


@pytest.mark.parametrize(("kind", "expected"), [("xor", 1), ("independent", 0)])
def test_exhaustive_local_hierarchy_and_exact_replay(tmp_path, kind, expected):
    args = sources(tmp_path, kind)
    path = tmp_path / "local.json"
    path.write_text(
        json.dumps(
            {
                "sensitivity": [
                    {"prior_mass": 1},
                    {"prior_mode": "per_cell", "prior_mass": 0.1},
                    {"prior_mass": 1},
                ]
            }
        )
    )
    args.extend(
        [
            "--local-policy",
            str(path),
            "--candidate-mode",
            "exhaustive",
            "--max-triplets",
            "8",
            "--permutations",
            "2",
            "--fmt",
            "csv",
        ]
    )
    out = tmp_path / "local"
    invoke("local", args, out)
    manifest = read(out / "candidate_manifest.json")
    assert manifest["eligible_triples"] == manifest["selected"] == 8 and manifest["exhaustive"]
    assert (
        not (out / "permutation_threebody.csv").exists() and (out / "permutation_mi.csv").is_file()
    )
    fits = rows(out / "local_model_fits.csv")
    assert len(fits) == 24
    active = next(
        r for r in fits if r["setting"] == "0" and all(r[f"{k}_original_col"] == "1" for k in "abc")
    )
    assert float(active["connected_information_bits"]) == pytest.approx(expected, abs=1e-9)
    assert float(active["entropy_difference_bits"]) == pytest.approx(expected, abs=1e-9)
    assert float(active["co_information_bits"]) == pytest.approx(-expected, abs=1e-9)
    assert active["possible_cells"] == "320" and active["n_valid"] == "8"
    assert float(active["deviance_nats"]) == pytest.approx(2 * 8 * np.log(2) * expected)
    joined = next(
        r
        for r in rows(out / "candidate_selection.csv")
        if r["candidate_id"] == active["candidate_id"]
    )
    assert joined["c_reference_pos"] == "11"
    for r in fits:
        if r["candidate_id"] != active["candidate_id"] or r["setting"] == "0":
            continue
        assert float(r["prior_total"]) == pytest.approx(1 if r["setting"] == "1" else 32)
    ranked = rows(out / "threebody_abc.csv")
    assert all(r["statistic"] == "connected_information_bits" for r in ranked)
    assert select_ids(out / "candidate_selection.csv") == select_ids(out / "threebody_support.csv")
    support = read(out / "local_state_support.json")
    assert support["coordinates"]["C"]["reference_positions"] == [11, 10]
    summary = read(out / "run_summary.json")
    assert summary["advanced"]["triple_calibration"]["status"] == "withheld"
    assert "Triple calibration withheld" in (out / "report.html").read_text(encoding="utf8")
    replay = read(out / "analysis_replay.json")["arguments"]
    again = tmp_path / "repeat"
    invoke(replay[0], replay[1:], again)
    assert (again / "local_model_fits.csv").read_bytes() == (
        out / "local_model_fits.csv"
    ).read_bytes()


def select_ids(path):
    return [r["candidate_id"] for r in rows(path)]


def test_structural_zeros_fail_for_observed_cells_and_are_recorded_on_success(tmp_path):
    args = sources(tmp_path)
    policy = tmp_path / "local.json"

    def write(cell):
        policy.write_text(
            json.dumps({"structural_zeros": [{"original_columns": [1, 1, 1], "bins": cell}]})
        )

    opts = [*args, "--local-policy", str(policy), "--candidate-mode", "exhaustive"]
    write([0, 0, 0])
    invalid = invoke("local", opts, tmp_path / "invalid", 1)
    assert "structural zero" in invalid.output and not (tmp_path / "invalid").exists()
    write([0, 0, 1])
    out = tmp_path / "valid"
    invoke("local", opts, out)
    fits = rows(out / "local_model_fits.tsv")
    top = next(r for r in fits if all(r[f"{s}_original_col"] == "1" for s in "abc"))
    assert top["possible_cells"] == "319"
    assert read(out / "local_state_support.json")["structural_zeros"][0]["bins"] == [0, 0, 1]


def test_hybrid_limit_and_misdeclared_prior_or_input(tmp_path):
    args = sources(tmp_path)
    spec = tmp_path / "policy.json"
    spec.write_text(json.dumps({"uniform_quota": 2, "screen_budget": 5}))
    opts = [
        *args,
        "--candidate-mode",
        "hybrid",
        "--candidate-policy",
        str(spec),
        "--max-triplets",
        "4",
        "--permutations",
        "2",
    ]
    out = tmp_path / "hybrid"
    invoke("analyze", [*opts, "--threebody"], out)
    manifest = read(out / "candidate_manifest.json")
    assert manifest["uniform_reserved"] == 2 and manifest["same_data_screen"]
    assert (
        manifest["screened_triples"] <= 5 and len(select_ids(out / "candidate_selection.tsv")) == 4
    )
    assert not (out / "permutation_threebody.tsv").exists()
    assert "exploratory" in (out / "report.html").read_text(encoding="utf8")
    plan = tmp_path / "plan"
    invoke("local", [*opts, "--dry-run"], plan)
    assert read(plan / "plan.json")["candidate_count_scope"] == "upper_bound_until_screen"
    invoke(
        "plan",
        [*args, "--threebody", "--candidate-mode", "exhaustive", "--max-triplets", "2"],
        tmp_path / "over",
        3,
    )
    for marker, extra in (
        ("prior", ["--candidate-mode", "prior"]),
        ("mixed", ["--legacy"]),
        ("axes", ["--threebody-model", "global"]),
    ):
        invoke("local", [*args, *extra], tmp_path / marker, 1)


def test_named_config_and_reference_candidates(tmp_path):
    root = Path(__file__).resolve().parents[1]
    work = tmp_path / "incoming"
    shutil.copytree(root / "examples/workflow", work)
    candidate = work / "candidate.tsv"
    candidate.write_text(
        "a_reference_id\ta_reference_pos\tb_reference_id\tb_reference_pos\tc_reference_id\tc_reference_pos\nprot_ref\t2\tdna_ref\t5\trna_ref\t3\n"
    )
    (work / "local.json").write_text('{"prior_mode":"total","prior_mass":0}')
    cfg = read(work / "run.json")
    cfg["analysis"] = {
        "threebody": True,
        "threebody_model": "maxent",
        "candidate_mode": "explicit",
        "triplet_candidates": "candidate.tsv",
        "local_policy": "local.json",
        "permutations": 2,
    }
    (work / "run.json").write_text(json.dumps(cfg))
    out = tmp_path / "full"
    invoke("run", ["--config", str(work / "run.json")], out)
    fit = rows(out / "analysis/local_model_fits.tsv")[0]
    assert float(fit["connected_information_bits"]) == pytest.approx(1) and float(
        fit["co_information_bits"]
    ) == pytest.approx(-1)
    assert "local_policy" in read(out / "analysis/run_summary.json")["inputs"]


def test_maxent_pooling_gap_support_and_insufficient_status(tmp_path):
    args = sources(tmp_path)
    data = tmp_path / "B.fa"
    data.write_text(
        data.read_text().replace(">B0\nA-C", ">B0\n--C").replace(">B4\nA-C", ">B4\nN-C")
    )
    candidate = tmp_path / "sites.tsv"
    candidate.write_text("a_original_col\tb_original_col\tc_original_col\n1\t1\t1\n")
    policy = tmp_path / "local.json"
    opts = [*args, "--triplet-candidates", str(candidate), "--local-policy", str(policy)]
    for gaps, depth in ((False, 6), (True, 7)):
        policy.write_text(
            json.dumps(
                {
                    "include_gaps": gaps,
                    "min_state_count": 1,
                    "training_ids": ["r0", "r1", "r2", "r3"],
                }
            )
        )
        out = tmp_path / f"gap-{gaps}"
        invoke("local", opts, out)
        row = rows(out / "local_model_fits.tsv")[0]
        assert int(row["n_valid"]) == depth
        frozen = read(out / "local_state_support.json")
        assert frozen["include_gaps"] is gaps and frozen["training_observations"] == [
            "r0",
            "r1",
            "r2",
            "r3",
        ]
        assert frozen["policy"] == "train_pooled"
    policy.write_text(json.dumps({"max_cells": 1}))
    out = tmp_path / "capped"
    invoke("local", opts, out)
    assert rows(out / "local_model_fits.tsv")[0]["status"] == "resource_limit"
    assert not rows(out / "threebody_abc.tsv")
    assert read(out / "run_summary.json")["advanced"]["local_fit"]["primary_fit_status"] == {
        "resource_limit": 1
    }
    assert "resource_limit" in (out / "report.html").read_text(encoding="utf8")


@pytest.mark.parametrize("origin", ["training", "same_data", "prespecified"])
def test_imported_file_origin_controls_legacy_null(tmp_path, origin):
    args = sources(tmp_path)
    candidate = tmp_path / "sites.tsv"
    candidate.write_text("a_original_col\tb_original_col\tc_original_col\n1\t1\t1\n")
    policy = tmp_path / "screen.json"
    policy.write_text(json.dumps({"file_origin": origin}))
    out = tmp_path / "scored"
    invoke(
        "analyze",
        [
            *args,
            "--threebody",
            "--triplet-candidates",
            str(candidate),
            "--candidate-policy",
            str(policy),
            "--permutations",
            "2",
        ],
        out,
    )
    manifest = read(out / "candidate_manifest.json")
    assert manifest["file_origin"] == origin and manifest["file_origin_provenance"] == "declared"
    assert (out / "permutation_threebody.tsv").is_file() == (origin == "prespecified")
    assert (out / "permutation_mi.tsv").is_file()


def test_pair_screen_reuses_native_dca_fit_once(tmp_path, monkeypatch):
    from triplet_coevolution.inference import advanced

    args = sources(tmp_path)
    policy = tmp_path / "screen.json"
    policy.write_text('{"pair_statistic":"dca_apc"}')
    options = [
        *args,
        "--candidate-mode",
        "triangle",
        "--candidate-policy",
        str(policy),
        "--max-triplets",
        "2",
    ]
    invoke("local", options, tmp_path / "invalid", 1)
    calls = []
    original = advanced.direct_couplings

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(advanced, "direct_couplings", tracked)
    out = tmp_path / "scores"
    invoke("local", [*options, "--dca"], out)
    assert len(calls) == 1
    assert read(out / "candidate_manifest.json")["ranking_screen"] == "dca_apc"
    assert set(read(out / "candidate_manifest.json")["pair_score_sha256"]) == {"AB", "AC", "BC"}
