import errno
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from triplet_coevolution.cli import app
from triplet_coevolution.diagnostics import doctor_report, validate_config
from triplet_coevolution.local_pipeline import run_pipeline
from triplet_coevolution.operations import Journal, lease, read_json, stage
from triplet_coevolution.policy_inputs import dependencies

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
run = CliRunner()


def check(args, code=0):
    result = run.invoke(app, list(map(str, args)))
    assert result.exit_code == code, (result.output, result.exception)
    return result


def test_diagnose_and_validate_without_search(tmp_path, monkeypatch):
    monkeypatch.setattr("triplet_coevolution.diagnostics.shutil.which", lambda n: None)
    report = doctor_report(tmp_path, required=["nhmmer"])
    assert report["status"] == "blocked" and report["resources"]["free_disk_bytes"] > 0
    assert any("nhmmer" in r for r in report["blockers"])
    assert doctor_report(tmp_path, workers=1000000)["status"] == "blocked"
    result = validate_config(EXAMPLES / "workflow" / "prepare.json")
    assert (result["matched_groups"], result["excluded_groups"]) == (8, 1)
    assert all(row["rows"] > 0 and row["columns"] > 0 for row in result["components"].values())
    assert validate_config(EXAMPLES / "workflow" / "run.json")["matched_groups"] is None
    workload = tmp_path / "unused"
    dry = check(
        ["run", "--config", EXAMPLES / "workflow" / "run.json", "--dry-run", "--work-dir", workload]
    )
    assert json.loads(dry.stdout)["status"] == "dry_run" and not workload.exists()
    reserved = tmp_path / "existing"
    reserved.mkdir()
    (reserved / "keep").touch()
    monkeypatch.setattr(
        "triplet_coevolution.pipeline_commands.validate_config",
        lambda c: pytest.fail("checked output too late"),
    )
    check(["validate-inputs", "--config", EXAMPLES / "workflow" / "run.json", "--out", reserved], 1)
    quiet = check(["--quiet", "doctor", "--scratch", tmp_path])
    assert not quiet.stdout
    verbose = check(["--json-log", "doctor", "--scratch", tmp_path])
    assert json.loads(verbose.stdout)["status"] == "ready"
    assert [json.loads(line)["event"] for line in verbose.stderr.splitlines()] == [
        "command_started",
        "command_ended",
    ]


def test_failed_or_dirty_attempt_not_reused(tmp_path):
    journal = Journal(tmp_path)
    state = {"version": 1}
    fp = lambda: dict(state)

    def broken(p):
        p.mkdir()
        (p / "half").touch()
        raise OSError(errno.ENOSPC, "test disk full")

    with pytest.raises(OSError, match="disk full"):
        stage(tmp_path, journal, "x", fp, broken)
    assert not (tmp_path / "objects").exists()

    def changed(p):
        p.mkdir()
        (p / "output").touch()
        state["version"] += 1

    with pytest.raises(ValueError, match="changed during"):
        stage(tmp_path, Journal(tmp_path), "x", fp, changed)

    def write(p):
        p.mkdir()
        (p / "foo").write_text("yes")

    result = stage(tmp_path, Journal(tmp_path), "x", fp, write)
    stage(tmp_path, Journal(tmp_path), "x", fp, lambda p: pytest.fail("reran"))
    (result / "foo").write_text("bad")
    with pytest.raises(ValueError, match="failed verification"):
        stage(tmp_path, Journal(tmp_path), "x", fp, write)
    with lease(tmp_path), pytest.raises(ValueError, match="owns"), lease(tmp_path):
        pass


@pytest.mark.parametrize("suffix", ["", ".policy", ".csv", ".json"])
def test_semantic_policy_fingerprints_ignore_nondependency_strings(tmp_path, suffix):
    tree = tmp_path / "tree.rows"
    tree.write_text("(a:1,b:2);")
    tips = tmp_path / "tip-map"
    tips.touch()
    alt = tmp_path / "alt.data"
    alt.touch()
    frequencies = tmp_path / "freq.tab"
    frequencies.write_text("{}")
    source = tmp_path / "description"
    source.touch()
    policy = tmp_path / ("null" + suffix)
    policy.write_text(
        json.dumps(
            {
                "null": "independent_site_tree",
                "statistics": ["mi_bits"],
                "draws": 3,
                "tree": {
                    "newick": tree.name,
                    "tip_map": tips.name,
                    "origin": "synthetic",
                    "source": source.name,
                    "frequency_policy": "supplied",
                    "frequencies": frequencies.name,
                    "alternatives": [
                        {
                            "label": "other",
                            "source": "independent",
                            "components": {"A": alt.name, "B": alt.name, "C": alt.name},
                        }
                    ],
                },
            }
        )
    )
    before = dependencies("null_policy", policy)
    assert set(before) == {str(x) for x in (policy, tree, tips, alt, frequencies)}
    source.write_text("changed")
    assert dependencies("null_policy", policy) == before
    tree.write_text("(a:2,b:1);")
    assert dependencies("null_policy", policy)[str(tree)] != before[str(tree)]
    tips.unlink()
    with pytest.raises(FileNotFoundError):
        dependencies("null_policy", policy)


def test_nested_clade_screen_and_panel_file_inventory(tmp_path):
    from triplet_coevolution.config import ANALYSIS_TYPES
    from triplet_coevolution.policy_inputs import SELF_CONTAINED

    assert {k for k, v in ANALYSIS_TYPES.items() if v is Path} == SELF_CONTAINED | {
        "null_policy",
        "stability_policy",
        "conditional_policy",
    }
    panel = tmp_path / "panel.json"
    panel.write_text("component\\toriginal_column\\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    split = sub / "split"
    split.touch()
    clade = sub / "group.csv"
    clade.write_text(json.dumps({"split_manifest": split.name}))
    nested = tmp_path / "nested"
    nested.write_text(json.dumps({"clades": "sub/group.csv", "panel_file": panel.name}))
    assert set(dependencies("conditional_policy", nested)) == {
        str(p) for p in (nested, panel, clade, split)
    }
    clade.write_text(
        json.dumps(
            {
                "clade_tree": {"newick": "tree", "tip_map": "tips"},
                "tree_depth": 1,
                "source": "local",
            }
        )
    )
    for name in ("tree", "tips"):
        (sub / name).touch()
    assert set(dependencies("conditional_policy", nested)) == {
        str(p) for p in (nested, panel, clade, sub / "tree", sub / "tips")
    }
    # A .json extension alone does not transform a biological data table into a policy.
    assert set(dependencies("site_panel", panel)) == {str(panel)}


def test_stage_reuse_and_independent_invalidation(tmp_path):
    project = tmp_path / "workflow"
    shutil.copytree(EXAMPLES / "workflow", project)
    cfg = project / "run.json"
    work = tmp_path / "work"
    settings = read_json(cfg)
    settings["analysis"] = {"dca": True, "contacts": "synthetic_labels.tsv", "seed": 5}
    cfg.write_text(json.dumps(settings))
    first = run_pipeline(cfg, work)
    assert first["status"] == "complete"
    stages = {s["stage"]: s for s in first["stages"]}
    assert all(s["action"] == "executed" for s in stages.values())
    assert all(s["measured"]["process_peak_rss_bytes"] > 0 for s in stages.values())
    assert read_json(Path(stages["score"]["output"]) / "plan.json")["observations"] == 8
    assert (Path(first["manifest"]).parent / "index.html").is_file()
    second = run_pipeline(cfg, work)
    assert all(s["action"] == "reused" for s in second["stages"])
    label = project / "synthetic_labels.tsv"
    label.write_bytes(label.read_bytes().replace(b"\t0", b"\t1", 1))
    third = run_pipeline(cfg, work)
    actions = {s["stage"]: s["action"] for s in third["stages"]}
    assert [k for k, v in actions.items() if v == "executed"] == ["benchmark"]
    settings["analysis"]["ridge"] = 0.2
    cfg.write_text(json.dumps(settings))
    fourth = run_pipeline(cfg, work, until="score")
    assert [s["action"] for s in fourth["stages"]] == ["reused", "executed", "executed"]
    source = project / "A_database.fasta"
    source.write_bytes(source.read_bytes() + b"\n")
    fifth = run_pipeline(cfg, work, until="prepare")
    assert fifth["stages"][0]["action"] == "executed"


def test_automatic_stage_replay_and_archived_selection(tmp_path):
    from triplet_coevolution.validation.null_archive import RefitData

    root = tmp_path / "workspace"
    args = ["run", "--input", EXAMPLES / "families", "--offline", "--work-dir", root]
    first = check(args)
    manifest = Path(first.stdout.split("Workflow: ")[1].strip())
    out = Path(read_json(manifest)["stages"][1]["output"])
    summary = read_json(out / "workflow_summary.json")
    assert summary["status"] == "complete" and summary["analysis_summary"]
    archive = RefitData.load(out / "analysis")
    assert archive.msa.n_observations == summary["scored_scope"]["observations"]
    assert "family_prepare: reused" in check(args).output
    recovered = tmp_path / "screen"
    check(["select", "--run", out / "analysis", "--out", recovered])
    assert read_json(recovered / "candidate_manifest.json")["selected"] > 0


def test_killed_supervisor_recovers_unpublished_stage(tmp_path):
    import subprocess
    import sys
    import time

    work = tmp_path / "recovery"
    cfg = EXAMPLES / "workflow" / "prepare.json"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "triplet_coevolution.cli",
            "pipeline",
            "--config",
            str(cfg),
            "--work-dir",
            str(work),
            "--until",
            "score",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        start = time.monotonic()
        while time.monotonic() - start < 40 and proc.poll() is None:
            log = next((work / "runs").glob("*/events.jsonl"), None)
            lines = log.read_text(encoding="utf-8").splitlines(keepends=True) if log else ()
            if any(
                record.get("event") == "stage_started" and record.get("stage") == "plan"
                for record in (json.loads(line) for line in lines if line.endswith("\n"))
            ):
                import os
                import signal

                pid = json.loads(log.read_text(encoding="utf-8").splitlines()[0])["supervisor_pid"]
                os.kill(pid, signal.SIGTERM)
                break
            time.sleep(0.02)
        assert proc.wait(timeout=10) != 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert not list((work / "objects" / "plan").glob("*/receipt.json"))
    recovered = run_pipeline(cfg, work, until="score")
    assert [s["action"] for s in recovered["stages"]] == ["reused", "executed", "executed"]
    assert len(list((work / "runs").glob("*/pipeline.json"))) == 1


def test_refuses_foreign_workspace_and_preserves_blocked_plan(tmp_path):
    user = tmp_path / "my-results"
    user.mkdir()
    (user / "private").write_text("retained")
    with pytest.raises(ValueError, match="unrecognized"):
        run_pipeline(EXAMPLES / "workflow" / "prepare.json", user, until="prepare")
    assert list(user.iterdir()) == [user / "private"]
    blocked = run_pipeline(
        EXAMPLES / "workflow" / "prepare.json",
        tmp_path / "workspace",
        until="score",
        memory_mib=0.0001,
    )
    assert blocked["status"] == "blocked" and [s["stage"] for s in blocked["stages"]] == [
        "prepare",
        "plan",
    ]
    plan = read_json(Path(blocked["stages"][-1]["output"]) / "plan.json")
    assert plan["blockers"]


def test_pipeline_calibration_recovers_committed_draws(tmp_path):
    work = tmp_path / "workspace"
    policy = tmp_path / "null.json"
    policy.write_text(
        json.dumps({"null": "whole_component", "statistics": ["mi_bits"], "draws": 5, "seed": 14})
    )
    cfg = EXAMPLES / "workflow" / "prepare.json"

    def stop(record):
        if record["event"] == "artifact_verification" and record["stage"] == "calibrate":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(cfg, work, until="calibrate", null_config=policy, emit=stop)
    assert len(list((work / "checkpoints").glob("*/null.sqlite3"))) == 1
    assert not list((work / "objects" / "calibrate").glob("*/receipt.json"))
    resumed = run_pipeline(cfg, work, until="calibrate", null_config=policy, workers=2)
    calibration = read_json(Path(resumed["stages"][-1]["output"]) / "calibration.json")
    assert calibration["runner"]["restored_draws"] == 5
    assert calibration["runner"]["new_draws"] == 0


def test_legacy_declines_different_factorization(tmp_path):
    from triplet_coevolution.workflow import prepare

    src = tmp_path / "prepared"
    prepare(EXAMPLES / "workflow" / "prepare.json", src)
    opts = read_json(src / "analyze_args.json")
    for name in ("--a", "--b", "--c", "--matches"):
        i = opts.index(name) + 1
        opts[i] = str(src / opts[i])
    path = tmp_path / "solver.json"
    path.write_text('{"backend":"mf","factor_solver":"cholesky"}')
    res = check([*opts, "--dca", "--dca-policy", path, "--legacy", "--out", tmp_path / "fit"], 1)
    assert "legacy" in res.output.lower() and not (tmp_path / "fit").exists()


def test_renamed_tree_policy_reuses_only_matching_law(tmp_path):
    source = EXAMPLES / "workflow" / "prepare.json"
    policy = tmp_path / "phylogenetic.policy"
    tree = tmp_path / "branches"
    policy.write_text(
        json.dumps(
            {
                "null": "independent_site_tree",
                "statistics": ["mi_bits"],
                "draws": 19,
                "seed": 8,
                "tree": {
                    "newick": tree.name,
                    "source": "fixture",
                    "origin": "synthetic",
                    "frequency_policy": "uniform",
                },
            }
        )
    )
    work = tmp_path / "workspace"

    def run_at(root, length):
        tree.write_text("(" + ",".join(f"obs{i:02}:{length}" for i in range(1, 9)) + ");")
        result = run_pipeline(source, root, until="calibrate", null_config=policy)
        row = result["stages"][-1]
        calibration = read_json(Path(row["output"]) / "calibration.json")
        return (
            row["action"],
            calibration["runner"]["count_sha256"],
            (Path(row["output"]) / "calibration.tsv").read_bytes(),
        )

    old = run_at(work, 0.01)
    assert run_at(work, 0.01) == ("reused", *old[1:])
    changed = run_at(work, 3.51)
    assert changed[0] == "executed" and changed[1] != old[1]
    assert run_at(tmp_path / "fresh", 3.51) == changed
    assert len(list((work / "checkpoints").glob("*/null.sqlite3"))) == 2
