"""Check paired tree calibration, clade holdouts and external-input provenance."""

import csv
import json
from itertools import product

import pytest
from typer.testing import CliRunner

from triplet_coevolution.cli import app


def invoke(args, code=0):
    r = CliRunner().invoke(app, list(map(str, args)))
    if r.exit_code != code:
        if r.exception and not isinstance(r.exception, SystemExit):
            raise r.exception
        pytest.fail(r.output)
    return r.output


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def rows(p):
    with p.open(encoding="utf8", newline="") as h:
        return list(csv.DictReader(h, delimiter="\t"))


def save(p, v):
    p.write_text(json.dumps(v), encoding="utf8")
    return p


@pytest.fixture
def example(tmp_path):
    cfg = tmp_path / "DesignInputs"
    cfg.mkdir()
    tips = [f"Obs{i}" for i in range(8)]
    fasta = []
    bits = list(product(range(2), repeat=3))
    for i, (c, mol) in enumerate(zip("ABC", ("protein", "dna", "rna"), strict=True)):
        p = cfg / f"{c}.fa"
        seqs = [f">{c}Ref\nG-G"]
        seqs.extend(
            f">{c}{j}\n{'AC'[pos[i]]}-{'GT'[pos[(i + 1) % 3]] if i != 2 else 'GU'[pos[(i + 1) % 3]]}"
            for j, pos in enumerate(bits)
        )
        p.write_text("\n".join(seqs) + "\n")
        fasta.extend(
            [
                "--" + c.lower(),
                p,
                "--" + c.lower() + "-ref",
                c + "Ref",
                "--" + c.lower() + "-type",
                mol,
            ]
        )
    t = cfg / "math.tsv"
    t.write_text(
        "observation_id\ta_id\tb_id\tc_id\tclade\n"
        + "".join(f"{x}\tA{i}\tB{i}\tC{i}\tGroup{i // 2}\n" for i, x in enumerate(tips))
    )
    fasta.extend(
        ["--matches", t, "--weight-policy", "uniform", "--c-start", "23", "--c-strand", "-"]
    )
    groups = ["(" + ",".join(f"T{i + j}:0.1" for j in (0, 1)) + "):0.4" for i in range(0, 8, 2)]
    (cfg / "Marker.nwk").write_text("(" + ",".join(groups) + ");\n")
    (cfg / "Alternative.nwk").write_text("(" + ",".join(f"T{i}:0.5" for i in range(8)) + ");\n")
    (cfg / "Tips.tsv").write_text(
        "tip_label\tobservation_id\n" + "".join(f"T{i}\t{x}\n" for i, x in enumerate(tips))
    )
    tree = {
        "newick": "Marker.nwk",
        "tip_map": "Tips.tsv",
        "source": "synthetic marker",
        "origin": "synthetic",
        "frequency_policy": "uniform",
    }
    return cfg, fasta, t, tree


def test_tree_preflight_inline_standalone_streams_and_taxon_ids(tmp_path, example):
    cfg, args, _, tree = example
    p = save(
        cfg / "tree.json",
        {
            "null": "independent_site_tree",
            "statistics": ["mi_bits", "dca_apc"],
            "seed": 12,
            "draws": 5,
            "max_t": True,
            "tree": tree,
        },
    )
    run = tmp_path / "fit"
    invoke(["analyze", *args, "--dca", "--null-policy", p, "--out", run])
    info = read(run / "calibration/calibration.json")
    assert info["tree_inference_scope"] == "conditional_on_declared_tree"
    assert info["diagnostics"]["topologies"]["shared"]["tips"] == 8
    assert info["diagnostics"]["evolution"]["frequency_policy"] == "uniform"
    assert (
        read(run / "plan.json")["calibration"]["tree_scenarios"][0][
            "numeric_draw_bytes_all_workers"
        ]
        > 0
    )
    assert any(r["c_reference_pos"] == "22" for r in rows(run / "calibration/calibration.tsv"))
    repeat = tmp_path / "parallel"
    invoke(["calibrate", "--run", run, "--config", p, "--workers", "2", "--out", repeat])
    assert (repeat / "calibration.tsv").read_bytes() == (
        run / "calibration/calibration.tsv"
    ).read_bytes()


def test_sensitivity_reports_all_scenarios_and_resume_identity(tmp_path, example):
    cfg, args, _, tree = example
    alt = {
        "label": "Branch <uncertainty>",
        "source": "alternative synthetic",
        "origin": "outcome_derived",
        "newick": "Alternative.nwk",
        "tip_map": "Tips.tsv",
    }
    policy = {
        "null": "independent_site_tree",
        "statistics": ["mi_apc"],
        "draws": 4,
        "seed": 8,
        "tree": {**tree, "alternatives": [alt]},
    }
    p = save(cfg / "null.json", policy)
    run = tmp_path / "fit"
    invoke(["analyze", *args, "--out", run])
    dest = tmp_path / "sens"
    cp = tmp_path / "cp"
    base = ["calibrate", "--run", run, "--config", p, "--checkpoint", cp]
    invoke([*base, "--draws", "2", "--out", tmp_path / "prev"])
    invoke([*base, "--resume", "--out", dest])
    info = read(dest / "calibration.json")
    assert info["runner"]["restored_draws"] == 2 and len(info["tree_sensitivity"]["scenarios"]) == 2
    alti = read(dest / info["tree_sensitivity"]["scenarios"][1]["manifest"])
    assert (
        alti["tree_inference_scope"] == "exploratory_tree_provenance"
        and alti["runner"]["restored_draws"] == 2
    )
    assert "&lt;uncertainty&gt;" in (dest / "report.html").read_text(encoding="utf8")
    assert all(r["scenarios"] == "2" for r in rows(dest / "tree_sensitivity.tsv"))
    (cfg / "Alternative.nwk").write_text(
        (cfg / "Alternative.nwk").read_text().replace(":0.5", ":0.8")
    )
    assert "identity mismatch" in invoke([*base, "--resume", "--out", tmp_path / "reject"], 1)


def test_component_gene_discordance_and_case_sensitive_mapping(tmp_path, example):
    cfg, args, _, _tree = example
    components = {
        c: {"newick": "Marker.nwk" if c != "B" else "Alternative.nwk", "tip_map": "Tips.tsv"}
        for c in "ABC"
    }
    policy = {
        "null": "independent_site_tree",
        "statistics": ["mi_bits"],
        "draws": 2,
        "tree": {
            "source": "synthetic per-component trees",
            "origin": "synthetic",
            "components": components,
        },
    }
    p = save(cfg / "null.json", policy)
    run = tmp_path / "fit"
    invoke(["analyze", *args, "--null-policy", p, "--out", run])
    assert set(read(run / "calibration/calibration.json")["diagnostics"]["discordance"]) == {
        "AB",
        "AC",
        "BC",
    }
    tip = cfg / "Tips.tsv"
    tip.write_text(tip.read_text().replace("Obs0", "obs0"))
    assert "bijection" in invoke(
        ["calibrate", "--run", run, "--config", p, "--out", tmp_path / "bad"], 1
    )


def test_split_and_stability_source_panel_boundaries(tmp_path, example):
    cfg, args, matches, _ = example
    p = save(
        cfg / "clades.json",
        {
            "clade_column": "clade",
            "source": "synthetic experimental groups",
            "test_groups": ["Group3"],
            "statistics": ["mi_bits"],
            "bootstrap_draws": 4,
            "top_k": 2,
            "seed": 6,
        },
    )
    cut = tmp_path / "split"
    invoke(["split", "--matches", matches, "--config", p, "--out", cut])
    members = rows(cut / "split_members.tsv")
    assert len(members) == 8 and {r["observation_id"] for r in members if r["heldout_fold"]} == {
        "Obs6",
        "Obs7",
    }
    run = tmp_path / "fit"
    invoke(["analyze", *args, "--stability-policy", p, "--out", run])
    assert read(run / "run_summary.json")["clade_stability"] == {
        "source_observations": 8,
        "groups": 4,
        "folds": 1,
        "bootstrap_draws": 4,
    }
    dest = tmp_path / "stability-repeat"
    invoke(["stability", "--run", run, "--config", p, "--out", dest])
    assert (dest / "stability_pairs.tsv").read_bytes() == (
        run / "stability/stability_pairs.tsv"
    ).read_bytes()
    assert "Held-out clade stability" in (run / "report.html").read_text(encoding="utf8")
    assert "report.html" in read(dest / "stability_summary.json")["artifact_sha256"]


def test_predeclared_split_replay_without_global_cohort_qc(tmp_path, example):
    cfg, args, matches, _ = example
    split_config = save(
        cfg / "holdout.json",
        {
            "clade_column": "clade",
            "source": "prospective field design",
            "test_groups": ["Group2", "Group3"],
        },
    )
    save_dir = tmp_path / "frozen"
    invoke(["split", "--matches", matches, "--config", split_config, "--out", save_dir])
    p = save(
        cfg / "study.json",
        {
            "split_manifest": str(save_dir / "split_manifest.json"),
            "statistics": ["mi_bits"],
            "top_k": 1,
        },
    )
    run = tmp_path / "fit"
    invoke(["analyze", *args, "--out", run])
    study = tmp_path / "study"
    invoke(["stability", "--run", run, "--config", p, "--out", study])
    saved = read(save_dir / "split_manifest.json")
    checked = read(study / "split_manifest.json")
    assert (
        saved["folds"] == checked["folds"]
        and checked["source"]["group_source"]["split_plan_sha256"] == saved["plan_sha256"]
    )
    assert checked["source"]["inputs"]["matches"] == saved["source"]["matches_sha256"]


def test_topological_depth_split_and_uninformative_tree_controls(tmp_path, example):
    cfg, args, matches, tree = example
    p = save(
        cfg / "tree-groups.json",
        {
            "clade_tree": {"newick": "Marker.nwk", "tip_map": "Tips.tsv"},
            "tree_depth": 1,
            "source": "synthetic grouping",
        },
    )
    invoke(["split", "--matches", matches, "--config", p, "--out", tmp_path / "groups"])
    assert sorted(read(tmp_path / "groups/split_manifest.json")["block_sizes"].values()) == [2] * 4
    tree["alternatives"] = [
        {
            "label": "same model different label",
            "source": "alias",
            "origin": "synthetic",
            "newick": "Marker.nwk",
            "tip_map": "Tips.tsv",
        }
    ]
    p = save(
        cfg / "compare.json",
        {"null": "independent_site_tree", "statistics": ["mi_bits"], "draws": 3, "tree": tree},
    )
    run = tmp_path / "fit"
    invoke(["analyze", *args, "--null-policy", p, "--out", run])
    info = read(run / "calibration/calibration.json")
    assert info["tree_sensitivity"]["different_tail_estimates"] == 0
    assert (
        info["tree_sensitivity"]["scenarios"][1]["comparison_to_primary"]["A"]["unrooted_rf"] == 0
    )
