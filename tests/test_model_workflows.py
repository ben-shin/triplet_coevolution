"""Integrated backend and conditioning provenance, support, null refits and held-out commands."""

import csv
import json
from itertools import product

import numpy as np
import pytest
from typer.testing import CliRunner

from triplet_coevolution.cli import app
from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.inference.potts import direct_couplings
from triplet_coevolution.io.score_store import load_score_store


def call(args, expected=0):
    result = CliRunner().invoke(app, list(map(str, args)))
    if result.exit_code != expected:
        if result.exception and not isinstance(result.exception, SystemExit):
            raise result.exception
        pytest.fail(result.output)
    return result.output


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


@pytest.fixture
def cohort(tmp_path):
    args = []
    ab = list(product(range(2), repeat=2)) * 10
    for i, (name, kind) in enumerate(zip("ABC", ("protein", "dna", "rna"), strict=True)):
        path = tmp_path / (name + ".fa")
        s = [
            "AC"[(a, b, a ^ b)[i]] + ("?" if i == 2 and j % 7 == 0 else "CA"[j % 2])
            for j, (a, b) in enumerate(ab)
        ]
        path.write_text(
            f">{name}ref\nAC\n" + "".join(f">{name}{j}\n{seq}\n" for j, seq in enumerate(s))
        )
        args.extend(
            [
                "--" + name.lower(),
                path,
                "--" + name.lower() + "-type",
                kind,
                "--" + name.lower() + "-ref",
                name + "ref",
            ]
        )
    table = tmp_path / "matches.tsv"
    table.write_text(
        "observation_id\ta_id\tb_id\tc_id\tclade\n"
        + "".join(f"obs{j}\tA{j}\tB{j}\tC{j}\tgroup{j // 8}\n" for j in range(len(ab)))
    )
    return [
        *args,
        "--matches",
        table,
        "--weight-policy",
        "uniform",
        "--c-strand",
        "-",
        "--c-start",
        "52",
    ]


def test_plm_support_native_refits_views_and_dca_grid(tmp_path, cohort, monkeypatch):
    source = tmp_path / "source"
    config = tmp_path / "null.json"
    config.write_text(
        json.dumps(
            {
                "null": "whole_component",
                "statistics": ["dca_frobenius", "dca_apc"],
                "draws": 3,
                "seed": 50,
            }
        )
    )
    from triplet_coevolution.inference import plm

    original = plm.fit_plm
    recorded = []

    def tracked(*a, **kw):
        recorded.append(kw["conditioning"])
        return original(*a, **kw)

    monkeypatch.setattr(plm, "fit_plm", tracked)
    call(
        [
            "analyze",
            *cohort,
            "--out",
            source,
            "--dca",
            "--dca-backend",
            "plm",
            "--null-policy",
            config,
        ]
    )
    assert len(recorded) >= 4 and set(recorded) == {"joint"}
    archive = load_score_store(source / "scores")
    assert np.all(archive.dca_counts["AB"] == 34)
    assert np.all(archive.n_valid["AB"] == 40)
    assert archive.maps[2].reference_positions == (52, 51)
    summary = read(source / "run_summary.json")
    assert summary["advanced"]["dca_diagnostics"]["target_status_counts"]["converged"] >= 3
    assert summary["advanced"]["settings"]["dca_design"]["backend"] == "plm"
    assert (
        "asymmetric"
        in read(source / "scores" / "score_manifest.json")["scores"]["dca_apc"]["backend"]
    )
    assert "Direct coupling fit" in (source / "report.html").read_text(encoding="utf8")
    output = tmp_path / "compare"
    config.write_text(
        json.dumps(
            {
                "models": {
                    "base": {"backend": "mf"},
                    "interfaces": {"backend": "plm", "conditioning": "pairwise"},
                }
            }
        )
    )
    call(["dca-compare", "--run", source, "--config", config, "--out", output])
    sensitivity = read(output / "dca_comparison.json")
    assert sensitivity["score_rows"] == 24
    assert sensitivity["models"][1]["design"]["conditioning"] == "pairwise"
    assert "score_covariate_spearman" in sensitivity["models"][0]["interfaces"]["AB"]["dca_apc"]
    with (output / "dca_sensitivity.tsv").open(encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert {r["n_valid"] for r in rows if r["model"] == "interfaces" and r["pair"] == "AB"} == {
        "40"
    }
    invalid = tmp_path / "invalid"
    assert "--legacy" in call(
        ["analyze", *cohort, "--out", invalid, "--legacy", "--dca", "--dca-backend", "plm"], 1
    )
    assert not invalid.exists()


def test_holdout_manifest_grid_and_reload(tmp_path, cohort):
    from triplet_coevolution.inference.nested import load_model

    config = tmp_path / "model.json"
    source = tmp_path / "source"
    call(["analyze", *cohort, "--out", source])
    config.write_text(
        json.dumps(
            {
                "clade_column": "clade",
                "source": "synthetic prespecified blocks",
                "test_groups": ["group3", "group4"],
            }
        )
    )
    call(
        [
            "split",
            "--matches",
            tmp_path / "matches.tsv",
            "--config",
            config,
            "--out",
            tmp_path / "partition",
        ]
    )
    config.write_text(
        json.dumps(
            {
                "clades": {"split_manifest": "partition/split_manifest.json"},
                "panel": {"A": [1], "B": [1], "C": [1]},
                "options": {"min_state_count": 1},
                "triples": [[1, 1, 1]],
                "l2_grid": [0.01, 0.1],
                "bootstrap_draws": 20,
            }
        )
    )
    out = tmp_path / "heldout"
    call(["conditional", "--run", source, "--config", config, "--out", out])
    native = tmp_path / "native"
    call(["analyze", *cohort, "--out", native, "--conditional-policy", config])
    assert "Held-out conditional prediction" in (native / "report.html").read_text(encoding="utf8")
    value = read(out / "conditional_summary.json")
    assert read(native / "conditional" / "conditional_summary.json")["grid"] == value["grid"]
    assert (
        value["split_sha256"] == read(tmp_path / "partition" / "split_manifest.json")["plan_sha256"]
    )
    assert len(value["grid"]) == 2
    for row in value["grid"]:
        fit = read(out / row["summary"])
        assert fit["folds"][0]["training_rows"] == 24
        assert fit["folds"][0]["test_rows"] == 16
        assert row["clade_macro_gain_nats_per_site"] > 0
        models = next(iter(fit["model_artifacts"].values()))
        model = load_model((out / row["summary"]).parent / models["pair_plus_k"]["path"])
        assert model.sites[2].reference_position == 52
        assert model.metadata["split_sha256"] == value["split_sha256"]


@pytest.mark.parametrize("n", (8, 68))
@pytest.mark.parametrize("scope", ("joint", "pairwise"))
def test_fixed_alphabet_gap_shrinkage_matches_explicit_inverse(n, scope):
    rng = np.random.default_rng(481)
    seq = [rng.choice(list(s), size=n).tolist() for s in ("ACDE-?", "ACGT-?", "ACGU-?")]
    parts = [
        ComponentMSA(c, m, tuple(f"{c}{i}" for i in range(n)), tuple(vals))
        for c, m, vals in zip("ABC", ("protein", "dna", "rna"), seq, strict=True)
    ]
    msa = MatchedTripletMSA(
        *parts, tuple(TripletMatch(str(i), *(p.ids[i] for p in parts)) for i in range(n))
    )
    w = np.arange(1, n + 1, dtype=float)
    diag = {}
    blocks = direct_couplings(
        msa,
        w,
        include_gaps=True,
        conditioning=scope,
        alphabet_support="fixed",
        ridge=0.17,
        shrinkage=0.27,
        min_valid=1,
        diagnostics=diag,
    )
    for pair, b in blocks.items():
        roles = list("ABC") if scope == "joint" else list(pair)
        x = []
        slices = {}
        for c in roles:
            j = "ABC".index(c)
            vals = parts[j].matrix[:, 0]
            q = (20 if j == 0 else 4) + 1
            known = vals >= 0
            one = np.zeros((n, q))
            for i in np.flatnonzero(known):
                one[i, vals[i]] = 1
            avg = np.average(one[known], axis=0, weights=w[known])
            one[known] -= avg
            start = sum(v.shape[1] for v in x)
            slices[c] = slice(start, start + q - 1)
            x.append(one)
        x = np.column_stack(x)
        cov = (x * w[:, None]).T @ x / w.sum()
        prec = np.linalg.inv(
            0.73 * cov + np.eye(len(cov)) * (0.17 + 0.27 * np.trace(cov) / len(cov))
        )
        mat = -prec[slices[pair[0]], slices[pair[1]]]
        norm = np.linalg.norm(mat - mat.mean(0) - mat.mean(1, keepdims=True) + mat.mean())
        assert b.frobenius[0, 0] == pytest.approx(norm, abs=2e-12)
        assert diag["".join(roles)]["regularized_condition_number"] >= 1
