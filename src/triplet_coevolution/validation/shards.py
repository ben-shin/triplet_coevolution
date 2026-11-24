"""Partition and recombine a fixed native null plan using the serial runner's evidence."""

import contextlib
import hashlib
import itertools
import json
import os
import platform
import sqlite3
import sys
import tempfile
import threading
import time
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from triplet_coevolution.validation import null_runner as engine
from triplet_coevolution.validation._null_checkpoint import (
    DATABASE,
    Journal,
    canonical,
    digest,
    open_journal,
)

PLAN = "triplet.null_shard_plan.v1"
RECEIPT = "triplet.null_shard_receipt.v1"
THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
)


def _runtime():
    deps = getattr(np.__config__, "CONFIG", {}).get("Build Dependencies", {})
    return {
        "machine": platform.machine(),
        "platform": sys.platform,
        "byteorder": sys.byteorder,
        "linear_algebra": {
            k: {v: d[v] for v in ("name", "version", "openblas configuration") if v in d}
            for k, d in deps.items()
            if k in ("blas", "lapack")
        },
        "thread_environment": {k: os.environ[k] for k in THREAD_ENV if k in os.environ},
    }


def _check_cpus(workers):
    cpu_limit = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpu_limit is None:
        return
    declared = _runtime()["thread_environment"]
    if not all(
        k in declared for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    ):
        raise ValueError("set OpenBLAS, MKL and OpenMP thread counts before starting a Slurm task")
    try:
        threads = [int(v) for v in declared.values()]
        available = int(cpu_limit)
    except ValueError as exc:
        raise ValueError(
            "Slurm and numerical thread declarations must be positive integers"
        ) from exc
    if min(threads) < 1 or workers * max(threads) > available:
        raise ValueError("workers times numerical thread limit exceeds SLURM_CPUS_PER_TASK")


@dataclass(frozen=True)
class ShardPlan:
    manifest: dict
    actual: dict
    threshold: dict
    evaluator: object

    @property
    def draws(self):
        return self.manifest["runner"]["planned_draws"]

    @property
    def sha256(self):
        return self.manifest["plan_sha256"]

    def verify(self, evaluate=None):
        runner = self.manifest["runner"]
        if digest(canonical(runner)) != self.sha256 or self.manifest["runtime"] != _runtime():
            raise ValueError("shard plan or numerical runtime identity mismatch")
        if runner["source"] != engine._source_identity(
            self.evaluator if evaluate is None else evaluate
        ):
            raise ValueError("shard evaluator or package source identity mismatch")
        if runner["observed"] != engine._data_identity(self.actual):
            raise ValueError("shard observed vectors or validity mask changed")


def prepare_shard_plan(
    observed,
    evaluate,
    *,
    draws,
    seed,
    null_name,
    identity,
    directions=None,
    max_t=False,
    joint_draws=False,
    memory_budget_bytes=None,
):
    """Prepare one final ensemble. The runner header and draws match an ordinary run_null."""
    b, final, seed, raw, directions = engine._validate_design(
        observed, identity, null_name, draws, None, seed, directions, max_t, joint_draws
    )
    _resources(raw, b, b, 1, max_t, memory_budget_bytes)
    actual, threshold, header = engine._prepare_evidence(
        raw, evaluate, identity, null_name, final, seed, directions, max_t, joint_draws, True
    )

    def fixed(arrays):
        return MappingProxyType(
            {k: np.frombuffer(v.tobytes(), dtype=v.dtype) for k, v in arrays.items()}
        )

    return ShardPlan(
        {
            "schema": PLAN,
            "plan_sha256": digest(canonical(header)),
            "runner": header,
            "runtime": _runtime(),
        },
        fixed(actual),
        fixed(threshold),
        evaluate,
    )


def shard_ids(draws, shard_id, shards):
    """Balanced contiguous partitions, first remainder partitions receive one extra draw."""
    b = engine._integer(draws, "draws", 1)
    n = engine._integer(shards, "shards", 1, b)
    i = engine._integer(shard_id, "shard_id", 0, n - 1)
    size, extra = divmod(b, n)
    first = i * size + min(i, extra)
    return range(first, first + size + (i < extra))


def _selection(ids, draws, limit):
    if isinstance(ids, range) and ids.step == 1:
        if not 0 <= ids.start < ids.stop <= draws:
            raise ValueError("invalid or empty shard interval")
        return ids, {"start": ids.start, "stop": ids.stop}
    if not isinstance(ids, Sized) or not 1 <= len(ids) <= draws:
        raise ValueError("explicit shard IDs require a nonempty bounded sequence")
    if len(ids) * 160 > limit:
        raise ValueError("explicit shard ID list exceeds memory budget; use a contiguous range")
    ids = sorted(engine._integer(i, "replicate ID", 0, draws - 1) for i in ids)
    if any(i == j for i, j in itertools.pairwise(ids)):
        raise ValueError("duplicate explicit replicate IDs")
    if ids[-1] - ids[0] + 1 == len(ids):
        return range(ids[0], ids[-1] + 1), {"start": ids[0], "stop": ids[-1] + 1}
    return ids, {"ids": ids}


def _from_assignment(a, b, limit):
    if (
        isinstance(a, dict)
        and set(a) == {"start", "stop"}
        and all(type(v) is int for v in a.values())
    ):
        entries = range(a["start"], a["stop"])
    elif isinstance(a, dict) and set(a) == {"ids"} and isinstance(a["ids"], list):
        entries = a["ids"]
    else:
        raise ValueError("invalid shard assignment")
    ids, normal = _selection(entries, b, limit)
    if a != normal:
        raise ValueError("shard assignment is not canonical")
    return ids


def _resources(actual, count, final, workers, max_t, limit, explicit=0):
    res = engine.null_workspace(
        [len(v) for v in actual.values()],
        draws=count,
        planned_draws=final,
        workers=workers,
        max_t=max_t,
    )
    # At most two total-B masks at once, already included in the engine estimate.
    # Explicit scattered assignments also involve a bounded Python ID/metadata list.
    res["runner_workspace_bytes"] += 160 * explicit
    limit = (
        engine.DEFAULT_WORKSPACE if limit is None else engine._integer(limit, "memory budget", 1)
    )
    if res["runner_workspace_bytes"] > limit:
        raise ValueError("shard runner workspace exceeds memory budget")
    width = res["packed_bytes_per_draw"]
    res["estimated_checkpoint_bytes"] = (
        262144 + 3 * count * (width + 256) + 32 * (4096 + width) + 160 * explicit
    )
    return res


def _journal_header(plan, assignment):
    return {
        **plan.manifest["runner"],
        "distributed": {
            "plan_sha256": plan.sha256,
            "runtime_sha256": digest(canonical(plan.manifest["runtime"])),
            "assignment": assignment,
        },
    }


def _atomic_json(path, obj):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".shard-", suffix=".partial")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(canonical(obj) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def _record_digests(journal):
    h = hashlib.sha256()
    for rid, packed in journal.conn.execute(
        "SELECT replicate_id,payload FROM draws ORDER BY replicate_id"
    ):
        h.update(rid.to_bytes(8, "big"))
        h.update(hashlib.sha256(packed).digest())
    return h.hexdigest()


def _receipt(plan, assignment, journal):
    obj = {
        "schema": RECEIPT,
        "plan_sha256": plan.sha256,
        "runtime_sha256": digest(canonical(plan.manifest["runtime"])),
        "assignment": assignment,
        "completed_draws": journal.count,
        "journal_header_sha256": journal.hash,
        "evidence_sha256": _record_digests(journal),
    }
    return {**obj, "receipt_sha256": digest(canonical(obj))}


def _load_json(path, limit):
    if path.stat().st_size > limit:
        raise ValueError("shard metadata exceeds memory budget")

    def unique(items):
        mapping = dict(items)
        if len(mapping) != len(items):
            raise ValueError("duplicate shard metadata key")
        return mapping

    with path.open(encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=unique)


def _load_receipt(root, plan, limit):
    path = root / "receipt.json"
    if not path.is_file():
        raise ValueError(f"missing complete shard receipt: {root}")
    rec = _load_json(path, limit)
    if not isinstance(rec, dict):
        raise TypeError("invalid shard receipt")
    values = {k: v for k, v in rec.items() if k != "receipt_sha256"}
    if rec.get("schema") != RECEIPT or digest(canonical(values)) != rec.get("receipt_sha256"):
        raise ValueError("shard receipt checksum mismatch")
    if rec["plan_sha256"] != plan.sha256 or rec["runtime_sha256"] != digest(
        canonical(plan.manifest["runtime"])
    ):
        raise ValueError("foreign shard scientific plan or runtime identity")
    return rec


def run_shard(
    plan,
    evaluate,
    directory,
    *,
    replicate_ids,
    workers=1,
    resume=False,
    memory_budget_bytes=None,
    checkpoint_budget_bytes=None,
    progress=None,
):
    """Checkpoint logical IDs independently; publish only a receipt after completing this subset."""
    plan.verify(evaluate)
    if type(resume) is not bool:
        raise TypeError("resume must be boolean")
    workers = engine._integer(workers, "workers", 1, engine.MAX_WORKERS)
    _check_cpus(workers)
    limit = (
        engine.DEFAULT_WORKSPACE
        if memory_budget_bytes is None
        else engine._integer(memory_budget_bytes, "memory budget", 1)
    )
    ids, assignment = _selection(replicate_ids, plan.draws, limit)
    scattered = not isinstance(ids, range)
    h = plan.manifest["runner"]
    res = _resources(
        plan.actual, len(ids), plan.draws, workers, h["max_t"], limit, len(ids) if scattered else 0
    )
    if checkpoint_budget_bytes is not None and res["estimated_checkpoint_bytes"] > engine._integer(
        checkpoint_budget_bytes, "checkpoint budget", 1
    ):
        raise ValueError("shard journal exceeds checkpoint budget")
    root = Path(directory)
    db_exists = (root / DATABASE).is_file()
    if (
        root.is_dir()
        and not db_exists
        and any(
            p.name != "null.writer.lock"
            and not (p.is_file() and p.name.startswith(".null-init-") and p.suffix == ".sqlite3")
            for p in root.iterdir()
        )
    ):
        raise ValueError("new shard output must be empty or a checkpoint created by this runner")
    if not resume and db_exists:
        raise ValueError("new shard output must be empty; explicitly resume its checkpoint")
    seen = bytearray(plan.draws)
    for rid in ids:
        seen[rid] = 1

    def accept(rid):
        if not 0 <= rid < len(seen) or seen[rid] != 1:
            raise ValueError("duplicate or unassigned replicate in shard")
        seen[rid] = 2

    start = time.perf_counter()
    done = restored = 0
    latest_heartbeat = start
    work_start = start

    def heartbeat():
        nonlocal latest_heartbeat
        latest_heartbeat = time.perf_counter()
        _atomic_json(
            root / "timing.json",
            {
                "plan_sha256": plan.sha256,
                "last_update_unix": time.time(),
                "completed": done,
                "restored_draws": restored,
                "elapsed_seconds": latest_heartbeat - work_start,
            },
        )

    lock = threading.Lock()

    def emit(event, rid=None, **kwargs):
        if progress:
            with lock:
                elapsed = time.perf_counter() - start
                progress(
                    {
                        "event": event,
                        "replicate_id": rid,
                        "completed": done,
                        "requested_draws": len(ids),
                        "planned_draws": plan.draws,
                        "elapsed_seconds": elapsed,
                        **engine._progress_rate(done, restored, len(ids), work_start),
                        **kwargs,
                    }
                )

    def compute(rid):
        emit("replicate_started", rid)
        sim = evaluate(rid, engine.ReplicateStreams(h["seed"], h["null_name"], rid))
        return engine._pack_result(plan.actual, plan.threshold, h["directions"], sim, h["max_t"])

    try:
        with open_journal(
            root,
            _journal_header(plan, assignment),
            res["packed_bytes_per_draw"],
            resume and db_exists,
        ) as journal:
            for rid, _ in journal.records():
                accept(rid)
            done = restored = journal.count
            work_start = time.perf_counter()
            existing = (
                _load_receipt(root, plan, limit) if (root / "receipt.json").exists() else None
            )
            if existing is not None and (
                done != len(ids) or existing != _receipt(plan, assignment, journal)
            ):
                raise ValueError("complete shard receipt disagrees with durable evidence")
            emit("shard_started", restored_draws=restored, workers=workers)
            if existing is None:
                heartbeat()
            for rid, packed in engine._compute((i for i in ids if seen[i] == 1), workers, compute):
                journal.append(rid, packed)
                accept(rid)
                done += 1
                if time.perf_counter() - latest_heartbeat >= 5:
                    heartbeat()
                emit("replicate_completed", rid)
            if done != len(ids):
                raise ValueError("shard contains missing replicate IDs")
            plan.verify(evaluate)
            rec = existing or _receipt(plan, assignment, journal)
            if existing is None:
                heartbeat()
                _atomic_json(root / "receipt.json", rec)
            emit("shard_completed", receipt_sha256=rec["receipt_sha256"])
            return rec
    except BaseException as exc:
        emit("shard_failed", exception_type=type(exc).__name__)
        raise


@contextlib.contextmanager
def _reader(root):
    path = (root / DATABASE).resolve()
    if not path.is_file():
        raise ValueError(f"missing shard journal: {root}")
    try:
        with contextlib.closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            conn.execute("PRAGMA query_only=ON")
            conn.execute("BEGIN")
            yield conn
    except sqlite3.Error as exc:
        raise ValueError(f"shard journal database failure: {exc}") from exc


def merge_shards(plan, directories, *, memory_budget_bytes=None, progress=None):
    """Return a normal final NullRun only after verifying exact global coverage and all evidence."""
    plan.verify()
    start = time.perf_counter()
    limit = (
        engine.DEFAULT_WORKSPACE
        if memory_budget_bytes is None
        else engine._integer(memory_budget_bytes, "memory budget", 1)
    )
    h = plan.manifest["runner"]
    res = _resources(plan.actual, plan.draws, plan.draws, 1, h["max_t"], limit)
    acc = engine._Counts(plan.actual, plan.draws, h["max_t"])
    receipt_tags = []
    for directory in directories:
        receipt_bytes = 192 * (len(receipt_tags) + 1)
        if res["runner_workspace_bytes"] + receipt_bytes > limit:
            raise ValueError("merge receipt inventory exceeds memory budget; use fewer shards")
        root = Path(directory)
        rec = _load_receipt(root, plan, limit)
        ids = _from_assignment(rec["assignment"], plan.draws, limit)
        if any(acc.complete[rid] for rid in ids):
            raise ValueError("overlapping or duplicate shard replicate IDs")
        if not isinstance(ids, range):
            _resources(
                plan.actual, plan.draws, plan.draws, 1, h["max_t"], limit - receipt_bytes, len(ids)
            )
        member = ids if isinstance(ids, range) else set(ids)
        with _reader(root) as conn:
            j = Journal(
                conn, _journal_header(plan, rec["assignment"]), res["packed_bytes_per_draw"]
            )
            if j.count != len(ids):
                raise ValueError("incomplete shard journal")
            for rid, packed in j.records():
                if rid not in member:
                    raise ValueError("unassigned replicate in shard evidence")
                acc.add(rid, packed)
            if _receipt(plan, rec["assignment"], j) != rec:
                raise ValueError("shard receipt disagrees with verified evidence")
        receipt_tags.append(rec["receipt_sha256"])
        if progress:
            elapsed = time.perf_counter() - start
            progress(
                {
                    "event": "shard_merged",
                    "completed": acc.n,
                    "planned_draws": plan.draws,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": (plan.draws - acc.n) * elapsed / acc.n,
                }
            )
    if acc.n != plan.draws:
        raise ValueError(
            f"missing replicate IDs: received {acc.n} of {plan.draws}; no inferential result published"
        )
    plan.verify()
    result = engine._finish_run(
        h,
        acc,
        plan.actual,
        plan.threshold,
        resources=res,
        resumed=0,
        workers=1,
        elapsed=time.perf_counter() - start,
    )
    result.manifest["distributed"] = {
        "shards": len(receipt_tags),
        "sorted_receipt_sha256": sorted(receipt_tags),
        "runtime": plan.manifest["runtime"],
    }
    return result


def shard_progress(manifest, directories, *, memory_budget_bytes=engine.DEFAULT_WORKSPACE):
    """Read consistent completion counters. Full verification belongs to merge_shards."""
    hdr = manifest["runner"]
    if manifest["schema"] != PLAN or manifest["plan_sha256"] != digest(canonical(hdr)):
        raise ValueError("shard plan checksum mismatch")
    b = engine._integer(hdr["planned_draws"], "planned draws", 1, memory_budget_bytes)
    seen = bytearray(b)
    total = 0
    pending = 0
    estimates = []
    details = []
    for directory in directories:
        if b + (len(details) + 1) * 2048 > memory_budget_bytes:
            raise ValueError("shard status inventory exceeds memory budget")
        root = Path(directory)
        if not (root / DATABASE).is_file():
            details.append({"directory": str(root), "status": "unstarted", "completed": 0})
            pending += 1
            continue
        # Publication is monotonic. Read the marker before the SQLite snapshot so a
        # shard completing mid-poll cannot pair an old count with a newer receipt.
        complete = (root / "receipt.json").is_file()
        with _reader(root) as conn:
            row = conn.execute("SELECT data FROM header").fetchone()
            if not row or len(row[0]) > memory_budget_bytes:
                raise ValueError("missing or oversized shard journal header")
            claimed = json.loads(row[0])
            assignment = claimed.get("distributed", {}).get("assignment")
            ids = _from_assignment(assignment, b, memory_budget_bytes)
            expected = {
                **hdr,
                "distributed": {
                    "plan_sha256": manifest["plan_sha256"],
                    "runtime_sha256": digest(canonical(manifest["runtime"])),
                    "assignment": assignment,
                },
            }
            journal = Journal(conn, expected, 0)
            if journal.count > len(ids):
                raise ValueError("shard progress exceeds assignment")
            for rid in ids:
                if seen[rid]:
                    raise ValueError("duplicate or overlapping shard assignments")
                seen[rid] = 1
            n = journal.count
        if complete and n != len(ids):
            raise ValueError("published shard has incomplete journal state")
        status = (
            "receipt_present"
            if complete
            else "awaiting_receipt"
            if n == len(ids)
            else "partial_or_running"
        )
        detail = {"directory": str(root), "completed": n, "assigned": len(ids), "status": status}
        total += n
        if n < len(ids):
            timing = root / "timing.json"
            remaining = None
            if timing.is_file():
                t = _load_json(timing, memory_budget_bytes)
                new = t["completed"] - t["restored_draws"]
                if t["plan_sha256"] == manifest["plan_sha256"] and 0 < new <= n:
                    remaining = (len(ids) - n) * t["elapsed_seconds"] / new
                    detail["last_update_unix"] = t["last_update_unix"]
            detail["estimated_remaining_seconds"] = remaining
            if remaining is None:
                pending += 1
            else:
                estimates.append(remaining)
        details.append(detail)
    return {
        "schema": "triplet.null_shard_progress.v1",
        "plan_sha256": manifest["plan_sha256"],
        "completed": total,
        "planned_draws": b,
        "unassigned_or_unstarted_draws": b - sum(seen),
        "estimated_remaining_seconds": max(estimates, default=0.0)
        if not pending and all(seen)
        else None,
        "estimate_assumption": "Each sampled shard continues concurrently at its measured rate. Scheduler queues and pauses are unknown.",
        "evidence_status": "provisional durable counters; full chain and receipt checks occur at merge",
        "shards": details,
    }
