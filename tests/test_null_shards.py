import json
import multiprocessing
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from triplet_coevolution.validation.null_runner import run_null
from triplet_coevolution.validation.shards import (
    merge_shards,
    prepare_shard_plan,
    run_shard,
    shard_ids,
    shard_progress,
)


def evaluate(i, s):
    x = s.generator("score").integers(-2, 3, 5).astype(float)
    if i % 4 == 1:
        x[3] = np.nan
    return {"upper": x, "lower": -x[:3]}


def design(seed=135):
    obs = {"upper": np.array([0, 1.0, np.nan, -1, 2]), "lower": np.array([1.0, 0, -2])}
    opt = {
        "draws": 19,
        "seed": seed,
        "null_name": "test",
        "identity": {"coordinates": ["u", "l"], "inputs": {"A": "mock"}},
        "directions": {"upper": "higher", "lower": "lower"},
        "joint_draws": True,
        "max_t": True,
    }
    return obs, opt


def test_disjoint_scattered_uneven_matches_serial_and_progress(tmp_path):
    obs, opt = design()
    plan = prepare_shard_plan(obs, evaluate, **opt)
    serial = run_null(obs, evaluate, workers=1, **opt)
    indices = [list(reversed(range(0, 19, 3))), range(1, 19, 3), range(2, 19, 3)]
    parts = []
    for i, indices_ in enumerate(indices):
        p = tmp_path / f"s{i}"
        parts.append(p)
        receipt = run_shard(plan, evaluate, p, replicate_ids=indices_, workers=2)
        assert receipt["completed_draws"] == len(indices_)
        assert receipt == run_shard(plan, evaluate, p, replicate_ids=indices_, resume=True)
    combined = merge_shards(plan, reversed(parts))
    for k in ("plan_sha256", "replicate_evidence_sha256", "count_sha256", "resolution"):
        assert combined.manifest[k] == serial.manifest[k]
    for key, f in serial.families.items():
        for field in vars(f):
            np.testing.assert_equal(getattr(combined.families[key], field), getattr(f, field))
    progress = shard_progress(plan.manifest, [*parts, tmp_path / "missing"])
    assert progress["completed"] == 19 and progress["shards"][-1]["status"] == "unstarted"
    assert [list(shard_ids(8, i, 3)) for i in range(3)] == [[0, 1, 2], [3, 4, 5], [6, 7]]
    for invalid in ([0, 2, 2], [-1], [19], [], [1.5], [True]):
        with pytest.raises(ValueError):
            run_shard(plan, evaluate, tmp_path / "bad", replicate_ids=invalid)
    with pytest.raises(ValueError, match="overlap"):
        merge_shards(plan, [*parts, parts[0]])
    with pytest.raises(ValueError, match="missing replicate IDs"):
        merge_shards(plan, parts[:2])
    with pytest.raises(ValueError, match="budget"):
        run_shard(
            plan,
            evaluate,
            tmp_path / "limited",
            replicate_ids=range(5),
            checkpoint_budget_bytes=100,
        )


def test_foreign_mismatched_inputs_mask_seed_code_and_corruption(tmp_path):
    obs, opt = design()
    plan = prepare_shard_plan(obs, evaluate, **opt)
    path = tmp_path / "s"
    run_shard(plan, evaluate, path, replicate_ids=range(19))
    altered = [
        (obs, {**opt, "seed": 56}),
        (obs, {**opt, "identity": {"inputs": "foreign"}}),
        (obs, {**opt, "draws": 20}),
        ({**obs, "upper": np.ones(5)}, opt),
    ]
    for data, opts in altered:
        p = prepare_shard_plan(data, evaluate, **opts)
        with pytest.raises(ValueError, match="foreign"):
            merge_shards(p, [path])

    def other(i, s):
        return evaluate(i, s)

    with pytest.raises(ValueError, match="source"):
        run_shard(plan, other, tmp_path / "other", replicate_ids=range(19))
    with pytest.raises(ValueError, match="identity mismatch"):
        run_shard(plan, evaluate, path, replicate_ids=range(18), resume=True)
    receipt = (path / "receipt.json").read_text()
    obj = json.loads(receipt)
    obj["completed_draws"] -= 1
    (path / "receipt.json").write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="checksum"):
        merge_shards(plan, [path])
    (path / "receipt.json").write_text(receipt)
    with sqlite3.connect(path / "null.sqlite3") as conn:
        conn.execute("UPDATE draws SET payload=zeroblob(length(payload)) WHERE replicate_id=4")
    with pytest.raises(ValueError, match="checksum"):
        merge_shards(plan, [path])
    with pytest.raises(ValueError, match="checksum"):
        run_shard(plan, evaluate, path, replicate_ids=range(19), resume=True)


def test_native_and_shard_progress_measure_only_new_committed_draws(tmp_path, monkeypatch):
    # On Windows 3.11 a fast checkpoint commit can finish inside one GetTickCount64 tick.
    monkeypatch.setattr("time.monotonic", lambda: 1.0)
    obs, opts = design()
    plan = prepare_shard_plan(obs, evaluate, **opts)

    def halt(e):
        if e["event"] == "replicate_completed" and e["completed"] == 2:
            raise RuntimeError("interrupt")

    for name in ("serial", "shard"):
        path = tmp_path / name

        def invoke(resume, progress, name=name, path=path):
            if name == "serial":
                return run_null(
                    obs, evaluate, checkpoint=path, resume=resume, progress=progress, **opts
                )
            return run_shard(
                plan, evaluate, path, replicate_ids=range(19), resume=resume, progress=progress
            )

        with pytest.raises(RuntimeError, match="interrupt"):
            invoke(False, halt)
        events = []
        invoke(True, events.append)
        assert events[0]["restored_draws"] == 2 and events[0]["new_draws"] == 0
        assert (
            events[0]["draws_per_second"] is None
            and events[0]["estimated_remaining_seconds"] is None
        )
        first = next(e for e in events if e["event"] == "replicate_completed")
        assert first["new_draws"] == 1 and first["draws_per_second"] > 0
        assert first["estimated_remaining_seconds"] == pytest.approx(16 / first["draws_per_second"])
        assert events[-1]["new_draws"] == 17 and events[-1]["estimated_remaining_seconds"] == 0


def test_scheduler_thread_guard_and_partition_bounds(tmp_path, monkeypatch):
    obs, opts = design()
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        monkeypatch.setenv(key, "2")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    plan = prepare_shard_plan(obs, evaluate, **opts)
    with pytest.raises(ValueError, match="exceeds SLURM"):
        run_shard(plan, evaluate, tmp_path / "partition", replicate_ids=range(5), workers=2)
    for i, n in ((3, 3), (0, 20), (-1, 3)):
        with pytest.raises(ValueError):
            shard_ids(19, i, n)


def test_filesystem_receipt_failure_keeps_all_transactions(tmp_path):
    obs, opt = design()
    plan = prepare_shard_plan(obs, evaluate, **opt)
    root = tmp_path / "capacity"

    def outage(event):
        if event["event"] == "replicate_completed" and event["completed"] == 19:
            (root / "receipt.json").mkdir()

    with pytest.raises(OSError):
        run_shard(plan, evaluate, root, replicate_ids=range(19), progress=outage)
    assert not (root / "receipt.json").is_file()
    with pytest.raises(ValueError, match="missing complete"):
        merge_shards(plan, [root])
    (root / "receipt.json").rmdir()
    events = []
    run_shard(plan, evaluate, root, replicate_ids=range(19), resume=True, progress=events.append)
    assert events[0]["restored_draws"] == 19 and all(
        e["event"] != "replicate_started" for e in events
    )
    assert (
        merge_shards(plan, [root]).manifest["count_sha256"]
        == run_null(obs, evaluate, **opt).manifest["count_sha256"]
    )


def _native_inputs():
    from triplet_coevolution.inference.mi import triplet_mi
    from triplet_coevolution.inference.weights import sequence_weights
    from triplet_coevolution.io.alignments import read_msa
    from triplet_coevolution.io.metadata import read_triplet_table
    from triplet_coevolution.io.score_store import file_digest
    from triplet_coevolution.msa.synchronize import synchronize
    from triplet_coevolution.validation.null_models import matching_null

    root = Path(__file__).resolve().parents[1] / "examples/mixed"
    files = [root / f"{m}.fasta" for m in ("protein", "dna", "rna")]
    components = [
        read_msa(p, k, m) for p, k, m in zip(files, "ABC", ("protein", "dna", "rna"), strict=True)
    ]
    msa = synchronize(*components, read_triplet_table(root / "matches.tsv"))
    model = matching_null(msa, "whole_component")

    def score(data):
        pairs = triplet_mi(data, sequence_weights(data))
        return {p: getattr(pairs, p.lower()).mi_bits.ravel() for p in ("AB", "AC", "BC")}

    def callback(i, streams):
        return score(model.draw(streams.generator("matching")))

    opts = {
        "draws": 17,
        "seed": 2346,
        "null_name": "matched",
        "identity": {
            "inputs": {p.name: file_digest(p) for p in [*files, root / "matches.tsv"]},
            "model": model.metadata(),
        },
        "max_t": True,
        "joint_draws": True,
    }
    return score(msa), callback, opts


def test_actual_sqlite_capacity_limit_in_middle_of_shard(tmp_path, monkeypatch):
    from triplet_coevolution.validation import _null_checkpoint

    obs = {"big": np.arange(100000, dtype=np.float64)}

    def callback(i, s):
        return obs

    options = {
        "draws": 3,
        "seed": 62,
        "identity": {"fixture": "database pages limited"},
        "null_name": "capacity",
    }
    plan = prepare_shard_plan(obs, callback, **options)

    def pause(event):
        if event["event"] == "replicate_completed":
            raise RuntimeError("stop after one commit")

    root = tmp_path / "limited-db"
    with pytest.raises(RuntimeError, match="one commit"):
        run_shard(plan, callback, root, replicate_ids=range(3), progress=pause)
    configure = _null_checkpoint._configure

    def impose_db_quota(conn):
        configure(conn)
        pages = conn.execute("PRAGMA page_count").fetchone()[0]
        conn.execute(f"PRAGMA max_page_count={int(pages)}")

    with monkeypatch.context() as context:
        context.setattr(_null_checkpoint, "_configure", impose_db_quota)
        with pytest.raises(ValueError, match="database or disk is full"):
            run_shard(plan, callback, root, replicate_ids=range(3), resume=True)
    assert not (root / "receipt.json").exists()
    current = shard_progress(plan.manifest, [root])
    assert current["completed"] == 1 and current["shards"][0]["status"] == "partial_or_running"
    run_shard(plan, callback, root, replicate_ids=range(3), resume=True)
    np.testing.assert_array_equal(
        merge_shards(plan, [root]).families["big"].exceedances, np.full(100000, 3)
    )


def _process(base, i, hold=None):
    obs, callback, opt = _native_inputs()
    plan = prepare_shard_plan(obs, callback, **opt)

    def event(data):
        if hold and data["event"] == "replicate_started" and data["replicate_id"] == 2:
            hold[0].set()
            hold[1].wait(40)

    run_shard(plan, callback, base / f"s{i}", replicate_ids=shard_ids(17, i, 3), progress=event)


def test_native_multiprocess_kinetic_kill_resume_and_exact_merge(tmp_path):
    mp = multiprocessing.get_context("spawn")
    hold = (mp.Event(), mp.Event())
    processes = [
        mp.Process(target=_process, args=(tmp_path, i, hold if i == 0 else None)) for i in range(3)
    ]
    for p in processes:
        p.start()
    try:
        assert hold[0].wait(30)
        obs, callback, opt = _native_inputs()
        plan = prepare_shard_plan(obs, callback, **opt)
        with pytest.raises(ValueError, match="writer lock"):
            run_shard(
                plan, callback, tmp_path / "s0", replicate_ids=shard_ids(17, 0, 3), resume=True
            )
        processes[0].kill()
        for p in processes:
            p.join(30)
        assert processes[0].exitcode != 0 and all(p.exitcode == 0 for p in processes[1:])
        with pytest.raises(ValueError, match="receipt"):
            merge_shards(plan, [tmp_path / "s0"])
        events = []
        run_shard(
            plan,
            callback,
            tmp_path / "s0",
            replicate_ids=shard_ids(17, 0, 3),
            resume=True,
            workers=2,
            progress=events.append,
        )
        assert events[0]["restored_draws"] == 2
        merged = merge_shards(plan, [tmp_path / f"s{i}" for i in (2, 0, 1)])
        serial = run_null(obs, callback, **opt)
        assert (
            merged.manifest["replicate_evidence_sha256"]
            == serial.manifest["replicate_evidence_sha256"]
        )
        for k, v in serial.families.items():
            np.testing.assert_array_equal(v.max_t_p, merged.families[k].max_t_p)
            np.testing.assert_array_equal(v.p, merged.families[k].p)
    finally:
        for p in processes:
            if p.is_alive():
                p.kill()
                p.join(15)


def test_native_commands_publish_same_coordinate_bounds_and_fdr_after_complete_merge(tmp_path):
    from typer.testing import CliRunner

    from triplet_coevolution.cli import app

    def call(*args, ok=True):
        result = CliRunner().invoke(app, list(map(str, args)))
        if result.exit_code and ok:
            if result.exception and not isinstance(result.exception, SystemExit):
                raise result.exception
            pytest.fail(result.output)
        assert (result.exit_code == 0) == ok, result.output
        return result.output

    example = Path(__file__).resolve().parents[1] / "examples/mixed"
    source, plan, merged, serial = (tmp_path / k for k in ("native", "plan", "merged", "serial"))
    cfg = tmp_path / "null.json"
    cfg.write_text(
        json.dumps(
            {
                "draws": 7,
                "null": "whole_component",
                "seed": 7,
                "max_t": True,
                "statistics": ["mi_bits", "mi_apc"],
                "family": "interface",
            }
        )
    )
    call(
        "analyze",
        "--a",
        example / "protein.fasta",
        "--b",
        example / "dna.fasta",
        "--c",
        example / "rna.fasta",
        "--b-type",
        "dna",
        "--c-type",
        "rna",
        "--matches",
        example / "matches.tsv",
        "--a-ref",
        "prot_ref",
        "--b-ref",
        "dna_ref",
        "--c-ref",
        "rna_ref",
        "--out",
        source,
    )
    call("shard-plan", "--run", source, "--config", cfg, "--out", plan, "--shards", 3)
    before = json.loads(call("shard-status", "--plan", plan))
    assert before["completed"] == 0 and before["planned_draws"] == 7
    for i in (2, 0):
        call("null-shard", "--plan", plan, "--shard-id", i)
    assert "receipt" in call("merge-null", "--plan", plan, "--out", merged, ok=False)
    assert not (merged / "calibration.tsv").exists()
    scattered = tmp_path / "middle"
    chosen = tmp_path / "ids.json"
    chosen.write_text("[4,3]")
    call("null-shard", "--plan", plan, "--ids", chosen, "--out", scattered, "--workers", 2)
    call(
        "merge-null",
        "--plan",
        plan,
        "--out",
        merged,
        "--shard",
        scattered,
        "--shard",
        plan / "shards/shard-00002",
        "--shard",
        plan / "shards/shard-00000",
    )
    call("calibrate", "--run", source, "--config", cfg, "--out", serial)
    assert (merged / "calibration.tsv").read_bytes() == (serial / "calibration.tsv").read_bytes()
    cfg.write_text(cfg.read_text() + " ")
    assert "checksum" in call("null-shard", "--plan", plan, "--shard-id", 1, ok=False)
