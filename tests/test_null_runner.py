"""Checked replicate identities, conservative tails, memory bounds and durable execution."""

import copy
import hashlib
import sqlite3
from contextlib import closing
from threading import Event, Thread

import numpy as np
import pytest

from triplet_coevolution.validation.null_runner import ReplicateStreams, null_workspace, run_null


def observed():
    return {"up": np.array([0.0, 2.0, np.nan]), "down": np.array([-1.0, 1.0])}


def simulation(i, streams):
    g = streams.generator("matching")
    upper = g.integers(-2, 4, size=3).astype(float)
    lower = g.integers(-2, 3, size=2).astype(float)
    if i % 3 == 1:
        upper[1] = np.nan
    if i % 5 == 0:
        lower[0] = np.inf
    return {"up": upper, "down": lower}


def base(**kw):
    return {
        "draws": 17,
        "seed": 1,
        "null_name": "fixed_test",
        "joint_draws": True,
        "max_t": True,
        "directions": {"up": "higher", "down": "lower"},
        "identity": {
            "inputs": {"synthetic": "v1"},
            "axis_order": ["x", "y", "z"],
            "statistic": "score",
        },
        **kw,
    }


def test_deterministic_cross_order_and_independent_tail_oracle():
    opts = base()
    one = run_null(observed(), simulation, **opts)
    flipped = run_null(
        dict(reversed(list(observed().items()))),
        simulation,
        workers=3,
        replicate_order=list(reversed(range(17))),
        **opts,
    )
    for key in ("replicate_evidence_sha256", "count_sha256", "plan_sha256"):
        assert one.manifest[key] == flipped.manifest[key]
    assert flipped.manifest["is_final"] and flipped.manifest["status"] == "complete_prespecified"
    for family, v in observed().items():
        f = one.families[family]
        q = flipped.families[family]
        for field in vars(f):
            np.testing.assert_equal(getattr(f, field), getattr(q, field))
        data = np.stack(
            [simulation(i, ReplicateStreams(1, "fixed_test", i))[family] for i in range(17)]
        )
        signs = 1 if family == "up" else -1
        maxima = []
        for row in data:
            maxima.append(np.max(signs * row) if np.isfinite(row).all() else np.nan)
        np.testing.assert_equal(f.max_t_null_maxima, maxima)
        for j, actual in enumerate(v):
            finite = np.isfinite(data[:, j])
            finite_exceed = sum(z * signs >= actual * signs for z in data[finite, j])
            conservative = finite_exceed + int((~finite).sum())
            assert f.valid_draws[j] == finite.sum()
            assert f.invalid_draws[j] == 17 - f.valid_draws[j]
            assert f.exceedances[j] == finite_exceed
            assert f.conservative_exceedances[j] == conservative
            assert f.p[j] == ((1 + conservative) / 18 if np.isfinite(actual) else 1)
            max_b = sum(not np.isfinite(z) or z >= actual * signs for z in maxima)
            assert f.max_t_p[j] == ((1 + max_b) / 18 if np.isfinite(actual) else 1)
            assert f.max_t_p[j] >= f.p[j]


def test_ties_and_coordinate_substreams_known_bytes():
    g = ReplicateStreams(4, "within_clade", 9)
    assert g.generator("matching").bit_generator.random_raw(3).tolist() == [
        4631811824538631101,
        11147775206224429422,
        15837804145722626375,
    ]

    def draw_by(order):
        return {k: g.generator("candidate", k).integers(0, 2**62) for k in order}

    assert draw_by(("a", "b")) == draw_by(("b", "a"))
    assert (
        g.generator("candidate", "ab", "c").random() != g.generator("candidate", "a", "bc").random()
    )

    def near(i, s):
        return {"x": np.array([np.nextafter(1.0, -np.inf), 1.0 - 1e-10])}

    result = run_null(
        {"x": [1.0, 1.0]}, near, draws=5, seed=1, null_name="ties", identity={"test": "fixed"}
    )
    np.testing.assert_allclose(result.families["x"].p, [1, 1 / 6])


def test_checkpoint_extension_failure_and_order_validation(tmp_path):
    path = tmp_path / "journal"
    opts = base(checkpoint=path, planned_draws=17)
    events = []

    def interrupt(e):
        events.append(e)
        if e["event"] == "replicate_completed" and e["completed"] == 3:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_null(observed(), simulation, progress=interrupt, **opts)
    assert events[0]["event"] == "run_started" and events[-1]["event"] == "run_failed"
    assert events[-1]["completed"] == 3
    with pytest.raises(ValueError, match="already exists"):
        run_null(observed(), simulation, **opts)
    interim = run_null(observed(), simulation, **{**opts, "draws": 6}, resume=True)
    assert (
        interim.manifest["status"] == "interim_exploratory"
        and interim.manifest["restored_draws"] == 3
    )
    resume = run_null(
        observed(),
        simulation,
        **opts,
        resume=True,
        workers=4,
        replicate_order=list(reversed(range(17))),
    )
    complete = run_null(observed(), simulation, **base())
    assert resume.manifest["count_sha256"] == complete.manifest["count_sha256"]
    assert (
        resume.manifest["replicate_evidence_sha256"]
        == complete.manifest["replicate_evidence_sha256"]
    )
    assert not run_null(observed(), simulation, **opts, resume=True).manifest["new_draws"]
    with pytest.raises(ValueError, match="identity mismatch"):
        run_null(observed(), simulation, **{**opts, "planned_draws": 18, "draws": 18}, resume=True)
    with pytest.raises(ValueError, match="outside this requested prefix"):
        run_null(observed(), simulation, **{**opts, "draws": 5}, resume=True)
    with pytest.raises(ValueError, match="duplicate|missing"):
        run_null(observed(), simulation, **base(replicate_order=[0] * 17))


@pytest.mark.parametrize(
    "changed", ["observed", "seed", "source", "statistic", "coordinates", "direction", "null"]
)
def test_resume_identity_rejects_changed_spec(tmp_path, changed):
    path = tmp_path / "checkpoint"
    opts = base(checkpoint=path, draws=2, planned_draws=5)
    run_null(observed(), simulation, **opts)
    values = observed()
    opts = copy.deepcopy(opts)
    if changed == "observed":
        values["up"][0] = 1e-9
    if changed == "seed":
        opts["seed"] += 1
    if changed == "source":
        opts["identity"]["inputs"]["synthetic"] = "revised"
    if changed == "coordinates":
        opts["identity"]["axis_order"].reverse()
    if changed == "statistic":
        opts["identity"]["statistic"] = "different"
    if changed == "direction":
        opts["directions"]["up"] = "lower"
    if changed == "null":
        opts["null_name"] = "other"
    with pytest.raises(ValueError, match="identity mismatch"):
        run_null(values, simulation, **opts, resume=True)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM draws WHERE step=1",
        "UPDATE draws SET payload=zeroblob(length(payload)) WHERE step=1",
        "UPDATE draws SET replicate_id=99 WHERE step=0",
        "UPDATE draws SET replicate_id=8 WHERE step=0",
        "UPDATE state SET n=2",
        "UPDATE header SET sha256='1234567'",
    ],
)
def test_journal_gap_corruption_and_truncation_rejected(tmp_path, sql):
    path = tmp_path / "checkpoint"
    opts = base(checkpoint=path, draws=4, planned_draws=17)
    run_null(observed(), simulation, **opts)
    with closing(sqlite3.connect(path / "null.sqlite3")) as db, db:
        db.execute(sql)
    with pytest.raises(ValueError, match="checksum|missing|gap|mismatch|malformed"):
        run_null(observed(), simulation, **opts, resume=True)


def test_duplicate_writer_and_disk_failure_leave_retryable_ids(tmp_path):
    path = tmp_path / "checkpoint"
    started = Event()
    advance = Event()
    failures = []

    def blocking_progress(event):
        if event["event"] == "replicate_completed" and event["completed"] == 1:
            started.set()
            assert advance.wait(10)
            raise RuntimeError("stop here")

    def runner():
        try:
            run_null(observed(), simulation, **base(checkpoint=path), progress=blocking_progress)
        except RuntimeError as exc:
            failures.append(str(exc))

    thread = Thread(target=runner)
    thread.start()
    try:
        assert started.wait(10)
        with pytest.raises(ValueError, match="writer lock"):
            run_null(observed(), simulation, **base(checkpoint=path), resume=True)
    finally:
        advance.set()
        thread.join(10)
    assert not thread.is_alive() and failures == ["stop here"]
    # SQLite fires this statement inside the two-write transaction. The replicate
    # INSERT itself must roll back when a state update fails, just like disk full.
    with closing(sqlite3.connect(path / "null.sqlite3")) as db, db:
        db.execute(
            "CREATE TRIGGER injected_full BEFORE UPDATE ON state BEGIN SELECT RAISE(ABORT,'database or disk is full'); END"
        )
    with pytest.raises(ValueError, match="disk is full"):
        run_null(observed(), simulation, **base(checkpoint=path), resume=True)
    with closing(sqlite3.connect(path / "null.sqlite3")) as db, db:
        assert db.execute("SELECT count(*) FROM draws").fetchone() == (1,)
        db.execute("DROP TRIGGER injected_full")
    restored = run_null(observed(), simulation, **base(checkpoint=path), resume=True)
    assert restored.manifest["restored_draws"] == 1 and restored.manifest["new_draws"] == 16


def test_thread_exception_preserves_checkpoint_correctness(tmp_path):
    fail = {3}

    def evaluator(i, streams):
        if i in fail:
            raise ArithmeticError("injected failure")
        return simulation(i, streams)

    opts = base(checkpoint=tmp_path / "checkpoint", workers=3)
    with pytest.raises(ArithmeticError, match="injected failure"):
        run_null(observed(), evaluator, **opts)
    fail.clear()
    result = run_null(observed(), evaluator, **opts, resume=True)
    assert (
        result.manifest["replicate_evidence_sha256"]
        == run_null(observed(), evaluator, **base()).manifest["replicate_evidence_sha256"]
    )


def test_guards():
    for kwargs, reason in [
        ({"joint_draws": False}, "coherent"),
        ({"directions": {"up": "lower"}}, "direction"),
        ({"workers": 0}, "workers"),
        ({"memory_budget_bytes": 100}, "memory budget"),
    ]:
        with pytest.raises(ValueError, match=reason):
            run_null(observed(), simulation, **base(**kwargs))
    assert null_workspace([9, 8], draws=5, max_t=True)["packed_bytes_per_draw"] == 22
    with pytest.raises(ValueError, match="checkpoint budget"):
        run_null(observed(), simulation, **base(checkpoint="unused", checkpoint_budget_bytes=1))
    for wrong in ({"up": np.array([1]), "down": np.ones(2)}, {"bad": [1]}):
        with pytest.raises(ValueError, match="family|families|shape"):
            run_null(observed(), lambda i, s, wrong=wrong: wrong, **base())


def _native_job(root, checkpoint, workers, resume=False, hold=None):
    from triplet_coevolution.inference.mi import triplet_mi
    from triplet_coevolution.inference.weights import sequence_weights
    from triplet_coevolution.io.alignments import read_msa
    from triplet_coevolution.io.metadata import read_triplet_table
    from triplet_coevolution.msa.synchronize import synchronize
    from triplet_coevolution.validation.null_models import matching_null

    parts = [
        read_msa(root / f"{m}.fasta", k, m)
        for m, k in zip(("protein", "dna", "rna"), "ABC", strict=True)
    ]
    msa = synchronize(*parts, read_triplet_table(root / "matches.tsv"))
    model = matching_null(msa, "whole_component")

    def score(data):
        result = triplet_mi(data, sequence_weights(data))
        return {p: getattr(result, p.lower()).mi_bits.ravel() for p in ("AB", "AC", "BC")}

    def evaluate(rid, streams):
        permuted = model.draw(streams.generator("matching"))
        if hold is not None and rid == 5:
            started, release = hold
            started.set()
            release.wait(20)
        return score(permuted)

    inputs = {
        k: hashlib.sha256((root / k).read_bytes()).hexdigest()
        for k in ("protein.fasta", "dna.fasta", "rna.fasta", "matches.tsv")
    }
    return run_null(
        score(msa),
        evaluate,
        draws=21,
        seed=9235,
        null_name="whole_component",
        identity={
            "inputs": inputs,
            "null": model.metadata(),
            "score": "mi_bits",
            "weighting": "equal_component",
            "axes": "all input columns in alignment order",
            "mask": "canonical joint observations",
        },
        checkpoint=checkpoint,
        resume=resume,
        workers=workers,
        joint_draws=True,
        max_t=True,
    )


def test_force_kill_in_the_middle_of_native_matched_mi_replicate(tmp_path):
    import multiprocessing
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "examples/mixed"
    mp = multiprocessing.get_context("spawn")
    started, release = mp.Event(), mp.Event()
    cp = tmp_path / "checkpoint"
    process = mp.Process(target=_native_job, args=(root, cp, 1, False, (started, release)))
    process.start()
    try:
        assert started.wait(20)
        process.kill()
        process.join(10)
        assert not process.is_alive() and process.exitcode != 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
    resume = _native_job(root, cp, 3, True)
    fresh = _native_job(root, None, 1)
    assert resume.manifest["restored_draws"] == 5 and resume.manifest["new_draws"] == 16
    for key in ("count_sha256", "replicate_evidence_sha256", "plan_sha256"):
        assert resume.manifest[key] == fresh.manifest[key]
    for name in ("AB", "AC", "BC"):
        np.testing.assert_array_equal(resume.families[name].p, fresh.families[name].p)
        np.testing.assert_array_equal(resume.families[name].max_t_p, fresh.families[name].max_t_p)


def test_real_sqlite_full_quota_preserves_previous_transaction(tmp_path, monkeypatch):
    from triplet_coevolution.validation import _null_checkpoint

    # Constrain only the test database. This triggers SQLite's actual SQLITE_FULL
    # path while the host filesystem remains available.
    path = tmp_path / "capacity"
    x = np.arange(100000, dtype=float)

    def evaluate(i, streams):
        return {"family": x}

    opts = {
        "draws": 1,
        "planned_draws": 2,
        "seed": 9,
        "identity": {"fixture": "quota"},
        "null_name": "database_test",
        "checkpoint": path,
    }
    run_null({"family": x}, evaluate, **opts)
    configure = _null_checkpoint._configure

    def limit(conn):
        configure(conn)
        pages = int(conn.execute("PRAGMA page_count").fetchone()[0])
        conn.execute(f"PRAGMA max_page_count={pages}")

    with monkeypatch.context() as context:
        context.setattr(_null_checkpoint, "_configure", limit)
        with pytest.raises(ValueError, match="database or disk is full"):
            run_null({"family": x}, evaluate, **{**opts, "draws": 2}, resume=True)
    complete = run_null({"family": x}, evaluate, **{**opts, "draws": 2}, resume=True)
    assert complete.manifest["restored_draws"] == 1 and complete.manifest["new_draws"] == 1
    np.testing.assert_array_equal(complete.families["family"].exceedances, np.full(len(x), 2))
