"""Native inspection, resource ceilings and replay of row/column/weight decisions."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from triplet_coevolution.cli import app
from triplet_coevolution.io.score_store import load_score_store

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/mixed"


def rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fixture(tmp):
    data = tmp / "in"
    shutil.copytree(EXAMPLE, data)
    prot = data / "protein.fasta"
    prot.write_text(prot.read_text().replace(">prot08\nV-RMT", ">prot08\n?---T"))
    table = data / "matches.tsv"
    lines = table.read_text().splitlines()
    # Deliberately unequal clade sizes, positive and zero design weights.
    table.write_text(
        lines[0]
        + "\tlineage\tprior\ttaxon_id\n"
        + "\n".join(
            row
            + "\t"
            + ("s" if i < 5 else "t")
            + "\t"
            + str([1, 0.5, 2, 0, 1, 1, 0.5, 10][i])
            + "\t"
            + str(i % 3)
            for i, row in enumerate(lines[1:])
        )
        + "\n"
    )
    policy = data / "qc.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "iteration": "fixed_point",
                "rows": {"A": {"min_canonical": 2}},
                "clade_column": "lineage",
            }
        )
    )
    base = [
        "--a",
        str(prot),
        "--b",
        str(data / "dna.fasta"),
        "--c",
        str(data / "rna.fasta"),
        "--matches",
        str(table),
        "--a-ref",
        "prot_ref",
        "--b-ref",
        "dna_ref",
        "--c-ref",
        "rna_ref",
        "--b-type",
        "dna",
        "--c-type",
        "rna",
        "--c-start",
        "8",
        "--c-strand",
        "-",
        "--qc-policy",
        str(policy),
    ]
    return base, data


def invoke(cmd, arguments, out, code=0):
    result = CliRunner().invoke(app, [cmd, *arguments, "--out", str(out)])
    assert result.exit_code == code, (result.output, result.exception)
    return result


def test_plan_only_matches_run_and_effective_support(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    args += [
        "--weight-policy",
        "supplied",
        "--weight-column",
        "prior",
        "--min-valid-weight",
        "6",
        "--dca",
        "--threebody",
        "--permutations",
        "2",
        "--seed",
        "2",
    ]
    with monkeypatch.context() as scoped:

        def unexpected(*args, **kwargs):
            pytest.fail("plan attempted an estimator or a neighbor calculation")

        scoped.setattr("triplet_coevolution.analysis_engine.triplet_mi", unexpected)
        scoped.setattr("triplet_coevolution.analysis_engine.sequence_weights", unexpected)
        invoke("plan", args, tmp_path / "plan")
    p = json.loads((tmp_path / "plan/plan.json").read_text())
    assert p["observations"] == 7 and p["retained_sites"] == {"A": 4, "B": 4, "C": 3}
    assert p["minimum_permutation_p"] == pytest.approx(1 / 3) and p["dca_solver"] == "dual"
    assert p["reference_pairs"] == {"AB": 16, "AC": 12, "BC": 12}
    assert not (tmp_path / "plan/scores").exists() and not (tmp_path / "plan/weights.tsv").exists()
    out = tmp_path / "run"
    invoke("analyze", args, out)
    s = json.loads((out / "run_summary.json").read_text())
    assert s["input_observations"] == 8 and s["matched_observations"] == 7
    assert s["weighting"]["sum_weights"] == 6 and s["weighting"][
        "kish_effective_count"
    ] == pytest.approx(36 / 7.5)
    assert s["qc_diagnostics"]["metadata"]["clade"]["retained"]["distinct"] == 2
    assert s["qc_diagnostics"]["metadata"]["taxon"]["retained"]["distinct"] == 3
    assert (out / "qc_rows.tsv").read_bytes() == (tmp_path / "plan/qc_rows.tsv").read_bytes()
    weights = rows(out / "weights.tsv")
    assert [r["observation_id"] for r in weights] == [f"obs0{i}" for i in range(1, 8)]
    assert all(r["neighbor_count"] == "" for r in weights)
    per_site = rows(out / "site_weight_support.tsv")
    assert any(
        r["component"] == "C"
        and r["reference_position"] == "8"
        and float(r["canonical_weight"]) == 6
        for r in per_site
    )
    store = load_score_store(out / "scores")
    assert store.provenance["observation_ids"] == [r["observation_id"] for r in weights]
    assert store.min_valid_weight == 6 and all(
        store.eligible(pair).all() for pair in ("AB", "AC", "BC")
    )
    for pair in ("ab", "ac", "bc"):
        assert all(float(r["valid_weight"]) >= 6 for r in rows(out / f"dca_{pair}.tsv"))
    assert s["advanced"]["settings"]["min_valid_weight"] == 6
    assert "Kish effective count" in (out / "report.html").read_text(encoding="utf8")
    replay = json.loads((out / "analysis_replay.json").read_text())["arguments"]
    invoke(replay[0], replay[1:], tmp_path / "replay")
    assert (out / "weights.tsv").read_bytes() == (tmp_path / "replay/weights.tsv").read_bytes()
    assert (out / "permutation_threebody.tsv").read_bytes() == (
        tmp_path / "replay/permutation_threebody.tsv"
    ).read_bytes()
    assert (out / "scores/score_manifest.json").is_file()
    for name in (
        "weights.tsv",
        "qc_summary.json",
        "site_weight_support.tsv",
        "analysis_replay.json",
    ):
        assert s["artifact_sha256"][name] == hashlib.sha256((out / name).read_bytes()).hexdigest()


def test_weight_threshold_just_above_available_support(tmp_path):
    args, _ = fixture(tmp_path)
    args += [
        "--weight-policy",
        "supplied",
        "--weight-column",
        "prior",
        "--min-valid-weight",
        "6.000001",
        "--dca",
        "--threebody",
        "--permutations",
        "2",
    ]
    out = tmp_path / "none"
    invoke("analyze", args, out)
    store = load_score_store(out / "scores")
    assert all(not store.eligible(p).any() for p in ("AB", "AC", "BC"))
    assert all(
        np.isnan(matrix).all() for blocks in store.scores.values() for matrix in blocks.values()
    )
    assert not rows(out / "threebody_abc.tsv") and not rows(out / "permutation_threebody.tsv")
    assert not rows(out / "mi_ab.tsv") and not rows(out / "dca_ab.tsv")
    assert (out / "report.html").is_file()


@pytest.mark.parametrize("kind", ["inputs", "qc", "workspace"])
def test_memory_guard_precedes_allocations_and_publishes_plan(tmp_path, monkeypatch, kind):
    args, data = fixture(tmp_path)
    if kind == "workspace":
        args[args.index("--c-start") + 1] = "100"
        for filename in ("protein.fasta", "dna.fasta", "rna.fasta"):
            p = data / filename
            p.write_text(
                "\n".join(s if s.startswith(">") else s * 20 for s in p.read_text().splitlines())
                + "\n"
            )
    budgets = {"inputs": "0.00001", "qc": "0.003", "workspace": "0.25"}
    args += ["--memory-mib", budgets[kind], "--dca", "--permutations", "5"]

    def unexpected(*args, **kwargs):
        pytest.fail("attempted scoring after a failed preflight")

    monkeypatch.setattr("triplet_coevolution.analysis_engine.sequence_weights", unexpected)
    out = tmp_path / "plan"
    invoke("plan", args, out, 3)
    p = json.loads((out / "plan.json").read_text())
    assert p["status"] == "blocked" and any(b["method"] == kind for b in p["blockers"])
    invoke("analyze", args, tmp_path / "score", 1)
    assert not (tmp_path / "score").exists()


def test_clade_balance_and_stratified_null(tmp_path):
    args, _ = fixture(tmp_path)
    args += [
        "--weight-policy",
        "clade_balanced",
        "--clade-column",
        "lineage",
        "--permutations",
        "2",
    ]
    out = tmp_path / "clades"
    invoke("analyze", args, out)
    s = json.loads((out / "run_summary.json").read_text())
    assert s["weighting"]["sum_weights"] == pytest.approx(2)
    assert s["advanced"]["settings"]["strata_column"] == "lineage"
    assert (
        json.loads((out / "plan.json").read_text())["exchangeability"]["movable_observations"] == 7
    )
    result = invoke("plan", [*args, "--strata-column", "taxon_id"], tmp_path / "bad", 1)
    assert "nested" in result.output


def test_cli_dry_run_global_checks_and_explicit_reference_sites(tmp_path):
    args, data = fixture(tmp_path)
    panel = data / "sites.tsv"
    panel.write_text(
        "component\treference_id\treference_position\nA\tprot_ref\t1\nB\tdna_ref\t2\nC\trna_ref\t7\n"
    )
    # A has a one-column panel, so replace the fixture minimum of two.
    (data / "qc.json").write_text(json.dumps({"rows": {c: {"min_canonical": 1} for c in "ABC"}}))
    opts = ["--site-panel", str(panel), "--threebody", "--threebody-model", "global", "--dry-run"]
    out = tmp_path / "dry"
    invoke("analyze", [*args, *opts], out)
    plan = json.loads((out / "plan.json").read_text())
    assert plan["retained_sites"] == dict.fromkeys("ABC", 1)
    assert plan["global"]["candidate_counting"] == "exact"
    assert plan["estimates"]["tested_triple_support_rows"] == 1
    assert plan["settings"]["threebody_model"] == "global"
    retained = [
        r for r in rows(out / "qc_sites.tsv") if r["component"] == "C" and r["retained"] == "True"
    ]
    assert len(retained) == 1 and retained[0]["reference_position"] == "7"
    assert sorted(p.name for p in out.iterdir()) == [
        "plan.json",
        "qc_rows.tsv",
        "qc_sites.tsv",
        "qc_summary.json",
    ]


def test_invalid_labels_and_weight_scale_block_preflight(tmp_path):
    args, data = fixture(tmp_path)
    label = data / "bad.tsv"
    label.write_text("unknown\tcolumn\n")
    invalid = invoke("plan", [*args, "--contacts", str(label)], tmp_path / "bad", 1)
    assert "header" in invalid.output
    path = data / "matches.tsv"
    lines = path.read_text().splitlines()
    for i in (1, 2):
        cols = lines[i].split("\t")
        cols[-2] = "1e308"
        lines[i] = "\t".join(cols)
    path.write_text("\n".join(lines) + "\n")
    invalid = invoke(
        "plan",
        [*args, "--weight-policy", "supplied", "--weight-column", "prior"],
        tmp_path / "large",
        1,
    )
    assert "finite" in invalid.output
