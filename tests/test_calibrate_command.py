"""Named calibration through archive, native analysis, and public CLI."""

import csv
import json
from itertools import product

import pytest
from typer.testing import CliRunner

from triplet_coevolution.cli import app
from triplet_coevolution.validation.null_policy import adjusted, mc_interval, resolution


def command(args, code=0):
    result = CliRunner().invoke(app, list(map(str, args)))
    if result.exit_code != code:
        if result.exception and not isinstance(result.exception, SystemExit):
            raise result.exception
        pytest.fail(str((result.output, result.exception)))
    return result.output


def table(path):
    with path.open(encoding="utf8", newline="") as h:
        return list(csv.DictReader(h, delimiter="\t"))


def policy(path, **kwargs):
    path.write_text(
        json.dumps(
            {
                "null": "whole_component",
                "statistics": ["mi_bits"],
                "draws": 5,
                "seed": 123,
                **kwargs,
            }
        )
    )
    return path


@pytest.fixture
def args(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    args = []
    bits = list(product(range(2), repeat=2)) * 8
    for i, (name, mol) in enumerate(zip("ABC", ("protein", "dna", "rna"), strict=True)):
        path = source / f"{name}.fa"
        path.write_text(
            f">{name}ref\nA\n"
            + "".join(f">{name}{j}\n{'AC'[(a, b, a ^ b)[i]]}\n" for j, (a, b) in enumerate(bits))
        )
        args.extend(
            [
                f"--{name.lower()}",
                path,
                f"--{name.lower()}-type",
                mol,
                f"--{name.lower()}-ref",
                name + "ref",
            ]
        )
    matches = source / "matches.tsv"
    matches.write_text(
        "observation_id\ta_id\tb_id\tc_id\tclade\tunit\n"
        + "".join(f"obs{j}\tA{j}\tB{j}\tC{j}\ts{j % 3}\t1\n" for j in range(len(bits)))
    )
    return [
        *args,
        "--matches",
        matches,
        "--weight-policy",
        "uniform",
        "--c-strand",
        "-",
        "--c-start",
        "55",
    ]


def test_inline_archive_same_seed_parallel_and_preserved_pair(tmp_path, args):
    cfg = policy(
        tmp_path / "null.json",
        null="preserve_ab",
        statistics=["mi_apc", "dca_apc", "co_information_bits"],
        strata_column="clade",
        max_t=True,
        family="interface",
        draws=7,
    )
    run = tmp_path / "source"
    command(["local", *args, "--dca", "--null-policy", cfg, "--out", run])
    prior = run / "calibration"
    out = tmp_path / "parallel"
    command(
        [
            "calibrate",
            "--scores",
            run / "scores" / "score_manifest.json",
            "--config",
            cfg,
            "--workers",
            "2",
            "--out",
            out,
        ]
    )
    assert (out / "calibration.tsv").read_bytes() == (prior / "calibration.tsv").read_bytes()
    info = json.loads((out / "calibration.json").read_text())
    assert info["diagnostics"]["preserved_components"] == ["A", "B"]
    rows = table(out / "calibration.tsv")
    triple = next(r for r in rows if r["interface"] == "ABC")
    assert (
        triple["c_reference_pos"] == "55"
        and float(triple["observed"]) == -1.0
        and triple["direction"] == "lower"
    )
    assert len(info["families"]) == 7
    assert "Named null calibration" in (run / "report.html").read_text(encoding="utf8")


@pytest.mark.parametrize("null", ["m2_bootstrap", "pair_panel"])
def test_generative_uses_matching_model_and_actual_fitted_score(tmp_path, args, null):
    run = tmp_path / "fit"
    cfg = policy(
        tmp_path / "design.json",
        null=null,
        statistics=["connected_information_bits"],
        max_t=null == "pair_panel",
        draws=11,
    )
    command(["local", *args, "--null-policy", cfg, "--out", run])
    row = table(run / "calibration/calibration.tsv")[0]
    assert float(row["observed"]) == pytest.approx(1.0)
    assert float(row["p"]) == (1 + int(row["conservative_exceedances"])) / 12
    assert float(row["p"]) < 0.4 and row["candidate_id"]
    assert bool(row["max_t_p"]) == (null == "pair_panel")


def test_stats_archive_design_validation_and_relocation(tmp_path, args):
    run = tmp_path / "analysis"
    command(["local", *args, "--weight-policy", "equal_component", "--out", run])
    cfg = policy(
        tmp_path / "plan.json", statistics=["connected_information_bits"], null="m2_bootstrap"
    )
    assert "weight-policy uniform" in command(
        ["calibrate", "--run", run, "--config", cfg, "--out", tmp_path / "bad"], 1
    )
    cfg = policy(cfg, statistics=["mi_bits"])
    (tmp_path / "inputs").rename(tmp_path / "moved")
    assert "checksum" in command(
        ["calibrate", "--run", run, "--config", cfg, "--out", tmp_path / "no"], 1
    )
    move = tmp_path / "relocated.json"
    move.write_text(
        json.dumps({**{c: f"moved/{c}.fa" for c in "ABC"}, "matches": "moved/matches.tsv"})
    )
    command(
        ["calibrate", "--run", run, "--config", cfg, "--out", tmp_path / "good", "--inputs", move]
    )


def test_prefix_resume_identity_and_interim(tmp_path, args):
    run = tmp_path / "fit"
    command(["analyze", *args, "--out", run])
    cfg = policy(tmp_path / "plan.json", planned_draws=9, draws=9, max_t=True)
    cp = tmp_path / "checkpoint"

    def cli(out, *extra):
        return ["calibrate", "--run", run, "--config", cfg, "--out", out, *extra]

    command(cli(tmp_path / "part", "--draws", "3", "--checkpoint", cp))
    full = tmp_path / "full"
    command(cli(full))
    resumed = tmp_path / "resumed"
    command(cli(resumed, "--checkpoint", cp, "--resume", "--workers", "2"))
    assert (resumed / "calibration.tsv").read_bytes() == (full / "calibration.tsv").read_bytes()
    prev = json.loads((tmp_path / "part/calibration.json").read_text())["runner"]
    final = json.loads((resumed / "calibration.json").read_text())["runner"]
    assert prev["status"] == "interim_exploratory" and final["restored_draws"] == 3
    policy(cfg, planned_draws=9, draws=9, max_t=True, seed=333)
    assert command(cli(tmp_path / "stale", "--checkpoint", cp, "--resume"), 1)


def test_resolution_ties_and_wilson_bounds():
    assert resolution(315492, 999, 0.05)["draws_for_bh_rank_one"] == 6309839
    bh, by = adjusted([0.025, 0.025, 1])
    assert list(bh) == pytest.approx([0.0375, 0.0375, 1])
    assert by[0] == pytest.approx(0.06875)
    assert mc_interval(0, 0, 100)[0] == 0
    assert mc_interval(0, 4, 100)[1] > mc_interval(0, 0, 100)[1]


def test_plan_apc_families_and_explicit_declared_refusals(tmp_path, args):
    cfg = policy(
        tmp_path / "policy.json",
        null="m2_bootstrap",
        statistics=["connected_information_bits"],
        draws=2,
    )
    command(
        [
            "plan",
            *args,
            "--threebody",
            "--threebody-model",
            "maxent",
            "--null-policy",
            cfg,
            "--out",
            tmp_path / "plan",
        ]
    )
    fam = json.loads((tmp_path / "plan/plan.json").read_text())["calibration"]["families"]
    assert fam["connected_information_bits"]["tests"] == 1 and fam["connected_information_bits"][
        "minimum_p"
    ] == pytest.approx(1 / 3)
    policy(cfg, null="m2_bootstrap", statistics=["connected_information_bits"], max_t=True)
    assert "joint draws" in command(
        ["local", *args, "--null-policy", cfg, "--out", tmp_path / "bad"], 1
    )
    cfg = policy(cfg, statistics=["co_information_bits"])
    assert "independent" in command(
        [
            "local",
            *args,
            "--candidate-mode",
            "triangle",
            "--null-policy",
            cfg,
            "--out",
            tmp_path / "screen",
        ],
        1,
    )


def test_standalone_refits_only_requested_statistic(tmp_path, args, monkeypatch):
    run = tmp_path / "dca"
    command(["analyze", *args, "--dca", "--out", run])

    def unexpected(*args, **kwargs):
        pytest.fail("unrequested MI fit")

    monkeypatch.setattr("triplet_coevolution.validation.calibration_models.triplet_mi", unexpected)
    cfg = policy(tmp_path / "policy.json", statistics=["dca_apc"], draws=2)
    command(["calibrate", "--run", run, "--config", cfg, "--out", tmp_path / "test"])
