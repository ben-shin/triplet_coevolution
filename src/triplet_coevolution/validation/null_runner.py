"""Deterministic finite-ensemble Monte Carlo tests with optional crash-resumable journals."""

import hashlib
import inspect
import json
import platform
import threading
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Integral
from pathlib import Path

import numpy as np

from triplet_coevolution import __version__
from triplet_coevolution.validation._null_checkpoint import SCHEMA, canonical, digest, open_journal

RNG = "triplet.logical-pcg64-sha256.v1"
MAX_WORKERS = 64
DEFAULT_WORKSPACE = 512 * 1024**2
TIE_EPS = 64 * np.finfo(np.float64).eps


def _integer(value, name, lower, upper=2**63 - 2):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not lower <= value <= upper
    ):
        raise ValueError(f"{name} must be an integer between {lower} and {upper}")
    return int(value)


@dataclass(frozen=True)
class ReplicateStreams:
    seed: int
    null_name: str
    replicate_id: int

    def generator(self, *logical_key):
        """A fresh PCG64 stream. Reuse its generator object to advance within a key."""
        material = canonical([RNG, self.seed, self.null_name, self.replicate_id, list(logical_key)])
        entropy = np.frombuffer(hashlib.sha256(material).digest(), dtype="<u4").tolist()
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


@dataclass(frozen=True)
class NullFamily:
    p: np.ndarray
    exceedances: np.ndarray
    valid_draws: np.ndarray
    invalid_draws: np.ndarray
    conservative_exceedances: np.ndarray
    observed_valid: np.ndarray
    max_t_p: np.ndarray | None
    max_t_exceedances: np.ndarray | None
    max_t_invalid_draws: int | None
    max_t_null_maxima: np.ndarray | None


@dataclass(frozen=True)
class NullRun:
    families: dict[str, NullFamily]
    manifest: dict


def null_workspace(
    family_sizes, *, draws, workers=1, max_t=False, planned_draws=None, custom_order=False
):
    """Estimated runner workspace and conservative local checkpoint storage, in bytes."""
    b = _integer(draws, "draws", 1)
    final = _integer(planned_draws if planned_draws is not None else b, "planned_draws", b)
    workers = _integer(workers, "workers", 1, MAX_WORKERS)
    if isinstance(family_sizes, Mapping):
        family_sizes = family_sizes.values()
    sizes = [_integer(v, "family size", 0) for v in family_sizes]
    if not sizes or not sum(sizes):
        raise ValueError("a null run requires at least one hypothesis")
    # Two bit vectors per family and one binary64 maximum when requested. SQLite
    # stores these masks rather than an unbounded set of dense simulated scores.
    width = sum(2 * ((n + 7) // 8) + 8 * bool(max_t) for n in sizes)
    n = sum(sizes)
    working = (
        112 * n
        + workers * (28 * n + 3 * width)
        + 32 * b
        + 2048 * len(sizes)
        + 2 * final
        + (8 * b if custom_order else 0)
        + (32 * b * len(sizes) if max_t else 0)
        + 2**16
    )
    disk = 262144 + 3 * final * (width + 256) + 32 * (4096 + width)
    return {
        "runner_workspace_bytes": working,
        "estimated_checkpoint_bytes": disk,
        "packed_bytes_per_draw": width,
        "fitting_workspace": "separate; include evaluator and BLAS workspace per concurrent draw",
    }


def _source_identity(evaluate):
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(
            canonical(
                [p.relative_to(root).as_posix(), digest(p.read_text(encoding="utf-8").encode())]
            )
        )
    module_hash = None
    try:
        source_file = inspect.getsourcefile(evaluate)
        if source_file and Path(source_file).is_file():
            module_hash = digest(Path(source_file).read_text(encoding="utf-8").encode())
        source = inspect.getsource(evaluate)
    except (OSError, TypeError):
        # User callbacks developed interactively must declare their exact model
        # identity alongside input hashes. Native CLI callbacks have source text.
        source = None
    return {
        "package_version": __version__,
        "package_python_sha256": h.hexdigest(),
        "numpy_version": np.__version__,
        "biopython_version": version("biopython"),
        "python_version": platform.python_version(),
        "evaluator_source_sha256": digest(source.encode()) if source else None,
        "evaluator_module_sha256": module_hash,
    }


def _arrays(observed):
    if (
        not isinstance(observed, Mapping)
        or not observed
        or any(not isinstance(k, str) or not k for k in observed)
    ):
        raise ValueError(
            "observed must map nonempty family names to one-dimensional numeric arrays"
        )
    raw = {name: np.asarray(observed[name]) for name in sorted(observed)}
    if any(v.ndim != 1 or v.dtype.kind not in "iuf" for v in raw.values()):
        raise ValueError("observed null statistics require one-dimensional numeric arrays")
    return raw


def _data_identity(actual):
    result = {}
    for k, v in actual.items():
        mask = np.isfinite(v)
        # Every nonfinite observation is ineligible at p=1. Canonicalize NaN payloads.
        encoded = np.where(mask, v, np.nan).astype("<f8", copy=False)
        result[k] = {
            "length": len(v),
            "finite_observations": int(mask.sum()),
            "score_sha256": digest(encoded.tobytes()),
            "finite_mask_sha256": digest(np.packbits(mask, bitorder="little").tobytes()),
        }
    return result


def _pack_result(actual, threshold, directions, simulated, max_t):
    if not isinstance(simulated, Mapping) or set(simulated) != set(actual):
        raise ValueError(
            "null replicate families differ from the declared observed family universe"
        )
    chunks = []
    for name, obs in actual.items():
        v = np.asarray(simulated[name])
        if v.shape != obs.shape or v.dtype.kind not in "iuf":
            raise ValueError(
                f"null replicate scores for {name} require the declared shape and numeric dtype"
            )
        values = v.astype(np.float64, copy=False)
        finite = np.isfinite(values)
        oriented = values if directions[name] == "higher" else -values
        hit = finite & (oriented >= threshold[name])
        chunks.append(np.packbits(hit, bitorder="little").tobytes())
        chunks.append(np.packbits(~finite, bitorder="little").tobytes())
        if max_t:
            maximum = float(oriented.max()) if len(oriented) and finite.all() else float("nan")
            chunks.append(np.asarray(maximum, dtype="<f8").tobytes())
    return b"".join(chunks)


def _schedule(order, draws):
    if order is None:
        return range(draws)
    if len(order) != draws:
        raise ValueError("replicate_order must be a permutation of all requested replicate IDs")
    seen = bytearray(draws)
    schedule = np.empty(draws, dtype=np.int64)
    for i, rid in enumerate(order):
        rid = _integer(rid, "replicate ID", 0, draws - 1)
        if seen[rid]:
            raise ValueError("duplicate or missing ID in requested replicate order")
        seen[rid] = 1
        schedule[i] = rid
    return schedule


def _compute(todo, workers, compute):
    """Only the coordinator writes the checkpoint. Live results are bounded by workers."""
    if workers == 1:
        for rid in todo:
            yield rid, compute(rid)
        return
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="triplet-null")
    pending = {}
    try:
        for _ in range(workers):
            rid = next(todo, None)
            if rid is not None:
                pending[executor.submit(compute, rid)] = rid
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                rid = pending.pop(future)
                yield rid, future.result()
                following = next(todo, None)
                if following is not None:
                    pending[executor.submit(compute, following)] = following
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


class _Counts:
    def __init__(self, actual, draws, max_t):
        self.hit = {k: np.zeros(len(v), dtype=np.int64) for k, v in actual.items()}
        self.bad = {k: np.zeros(len(v), dtype=np.int64) for k, v in actual.items()}
        self.maxima = {k: np.full(draws, np.nan) for k in actual} if max_t else None
        self.complete = bytearray(draws)
        self.record_hashes = np.empty((draws, 32), dtype=np.uint8)
        self.n = 0

    def add(self, rid, payload):
        if rid < 0 or rid >= len(self.complete):
            raise ValueError("checkpoint includes replicate IDs outside this requested prefix")
        if self.complete[rid]:
            raise ValueError(f"duplicate null replicate ID {rid}")
        offset = 0
        for name, h in self.hit.items():
            n = len(h)
            byte_count = (n + 7) // 8
            flags = []
            for target in (h, self.bad[name]):
                block = np.frombuffer(payload, dtype=np.uint8, count=byte_count, offset=offset)
                offset += byte_count
                mask = np.unpackbits(block, bitorder="little")[:n]
                target += mask
                flags.append(mask)
            if (flags[0] & flags[1]).any():
                raise ValueError(
                    "invalid null replicate flags: a finite exceedance is marked invalid"
                )
            if self.maxima is not None:
                self.maxima[name][rid] = np.frombuffer(
                    payload, dtype="<f8", count=1, offset=offset
                )[0]
                offset += 8
        if offset != len(payload):
            raise ValueError("malformed null replicate payload")
        self.record_hashes[rid] = np.frombuffer(hashlib.sha256(payload).digest(), dtype=np.uint8)
        self.complete[rid] = 1
        self.n += 1

    def finish(self, actual, threshold):
        b = self.n
        if b != len(self.complete) or not all(self.complete):
            raise ValueError("null results have missing replicate IDs")
        result = {}
        for name, vals in actual.items():
            valid = np.isfinite(vals)
            hit, bad = self.hit[name], self.bad[name]
            conservative = hit + bad
            p = (1.0 + conservative) / (1.0 + b)
            p[~valid] = 1.0
            max_p = max_hit = max_bad = maxima = None
            if self.maxima is not None:
                maxima = self.maxima[name]
                ordered = np.sort(maxima[np.isfinite(maxima)])
                max_bad = b - len(ordered)
                # One invalid score makes its family's shared extremum unknown.
                # Count that whole draw as an exceedance for every threshold.
                max_hit = b - np.searchsorted(ordered, threshold[name], side="left")
                max_p = (1.0 + max_hit) / (1.0 + b)
                max_p[~valid] = 1.0
            result[name] = NullFamily(
                p, hit, b - bad, bad, conservative, valid, max_p, max_hit, max_bad, maxima
            )
        return result


def _validate_design(
    observed, identity, null_name, draws, planned_draws, seed, directions, max_t, joint_draws
):
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("null identity must declare data, model settings and coordinate universe")
    if not isinstance(null_name, str) or not null_name.strip():
        raise ValueError("null_name must be a nonempty scientific null identity")
    for option in (max_t, joint_draws):
        if not isinstance(option, bool):
            raise TypeError("max_t and joint_draws must be booleans")
    if max_t and not joint_draws:
        raise ValueError("maxT requires a declared coherent joint draw for each family")
    b = _integer(draws, "draws", 1)
    final = _integer(b if planned_draws is None else planned_draws, "planned_draws", b)
    seed = _integer(seed, "seed", 0, 2**256 - 1)
    raw = _arrays(observed)
    keys = tuple(raw)
    if directions is None:
        directions = dict.fromkeys(keys, "higher")
    if (
        not isinstance(directions, Mapping)
        or set(directions) != set(keys)
        or any(k not in ("higher", "lower") for k in directions.values())
    ):
        raise ValueError("supply one higher/lower direction per observed null family")
    return b, final, seed, raw, directions


def _prepare_evidence(
    raw, evaluate, identity, null_name, final, seed, directions, max_t, joint_draws, require_source
):
    actual = {k: np.array(v, dtype=np.float64, copy=True) for k, v in raw.items()}
    threshold = {}
    for name, v in actual.items():
        oriented = v if directions[name] == "higher" else -v
        limit = np.full(v.shape, np.inf)
        finite = np.isfinite(v)
        with np.errstate(over="ignore"):
            limit[finite] = oriented[finite] - TIE_EPS * np.maximum(1.0, np.abs(v[finite]))
        threshold[name] = limit
    science = json.loads(canonical(identity))
    header = {
        "schema": SCHEMA,
        "rng": RNG,
        "seed": seed,
        "null_name": null_name,
        "identity": science,
        "science_sha256": digest(canonical(science)),
        "observed": _data_identity(actual),
        "directions": dict(directions),
        "planned_draws": final,
        "max_t": max_t,
        "joint_draws": joint_draws,
        "source": _source_identity(evaluate),
        "tie_relative_tolerance": TIE_EPS,
        "invalid_draw_policy": "worst_case_exceedance",
    }
    if (
        require_source
        and header["source"]["evaluator_source_sha256"] is None
        and not (
            isinstance(science.get("evaluator_sha256"), str)
            and len(science["evaluator_sha256"]) == 64
            and all(c in "0123456789abcdef" for c in science["evaluator_sha256"])
        )
    ):
        raise ValueError(
            "checkpointed callback without inspectable source requires evaluator_sha256 in identity"
        )
    return actual, threshold, header


def _finish_run(header, acc, actual, threshold, *, resources, resumed, workers, elapsed):
    b = len(acc.complete)
    final = header["planned_draws"]
    max_t = header["max_t"]
    families = acc.finish(actual, threshold)
    manifest = {
        **header,
        "plan_sha256": digest(canonical(header)),
        "replicate_evidence_sha256": hashlib.sha256(memoryview(acc.record_hashes)).hexdigest(),
        "count_sha256": digest(
            canonical(
                {
                    k: [
                        digest(f.exceedances.astype("<i8", copy=False).tobytes()),
                        digest(f.invalid_draws.astype("<i8", copy=False).tobytes()),
                    ]
                    for k, f in families.items()
                }
            )
        ),
        "requested_draws": b,
        "completed_draws": acc.n,
        "restored_draws": resumed,
        "new_draws": acc.n - resumed,
        "workers": workers,
        "status": "complete_prespecified" if b == final else "interim_exploratory",
        "is_final": b == final,
        "resolution": 1 / (b + 1),
        "completed_id_range": [0, b - 1],
        "workspace": resources,
        "elapsed_seconds": elapsed,
        "max_t_definition": "maximum oriented statistic in each joint replicate within family; any nonfinite member makes the maximum unknown and conservative"
        if max_t
        else None,
    }
    return NullRun(families, manifest)


def _progress_rate(done, restored, requested, work_start):
    new = max(0, done - restored) if work_start is not None else 0
    elapsed = time.monotonic() - work_start if work_start is not None else 0
    rate = new / elapsed if new and elapsed > 0 else None
    return {
        "new_draws": new,
        "draws_per_second": rate,
        "estimated_remaining_seconds": (requested - done) / rate if rate else None,
    }


def run_null(
    observed,
    evaluate,
    *,
    draws,
    seed,
    null_name,
    identity,
    directions=None,
    workers=1,
    checkpoint=None,
    resume=False,
    planned_draws=None,
    max_t=False,
    joint_draws=False,
    memory_budget_bytes=None,
    checkpoint_budget_bytes=None,
    progress=None,
    replicate_order=None,
):
    """Evaluate fixed-family Monte Carlo statistics using independently addressed draws.

    Pass numeric scores in native units and their upper/lower directions. evaluate(id,streams)
    must be deterministic given the supplied streams, thread-safe when workers>1, and
    independent of draw completion order. Include all inputs, scoring settings, masks
    and coordinate identities in identity. Each family defines its own coherent maxT.
    """
    start = time.monotonic()
    if not isinstance(resume, bool):
        raise TypeError("resume must be boolean")
    workers = _integer(workers, "workers", 1, MAX_WORKERS)
    b, final, seed, raw, directions = _validate_design(
        observed, identity, null_name, draws, planned_draws, seed, directions, max_t, joint_draws
    )
    resources = null_workspace(
        (len(v) for v in raw.values()),
        draws=b,
        workers=workers,
        max_t=max_t,
        planned_draws=final,
        custom_order=replicate_order is not None,
    )
    memory_limit = (
        DEFAULT_WORKSPACE
        if memory_budget_bytes is None
        else _integer(memory_budget_bytes, "memory budget", 1)
    )
    if resources["runner_workspace_bytes"] > memory_limit:
        raise ValueError(
            "null runner workspace exceeds memory budget; reduce workers, draw count, or hypotheses"
        )
    if checkpoint_budget_bytes is not None:
        limit = _integer(checkpoint_budget_bytes, "checkpoint budget", 1)
        if checkpoint is not None and resources["estimated_checkpoint_bytes"] > limit:
            raise ValueError("planned null journal exceeds checkpoint budget")
    actual, threshold, header = _prepare_evidence(
        raw,
        evaluate,
        identity,
        null_name,
        final,
        seed,
        directions,
        max_t,
        joint_draws,
        bool(checkpoint),
    )
    schedule = _schedule(replicate_order, b)
    acc = _Counts(actual, b, max_t)
    guard = threading.RLock()
    work_start = None

    def emit(event, rid=None, **details):
        if progress is None:
            return
        with guard:
            progress(
                {
                    "event": event,
                    "replicate_id": rid,
                    "completed": acc.n,
                    "requested_draws": b,
                    "planned_draws": final,
                    "elapsed_seconds": time.monotonic() - start,
                    **_progress_rate(acc.n, resumed, b, work_start),
                    **details,
                }
            )

    def compute(rid):
        emit("replicate_started", rid)
        streams = ReplicateStreams(seed, null_name, rid)
        simulation = evaluate(rid, streams)
        return _pack_result(actual, threshold, directions, simulation, max_t)

    resumed = 0
    try:
        with open_journal(
            checkpoint, header, resources["packed_bytes_per_draw"], resume
        ) as journal:
            if journal is not None:
                for rid, packed in journal.records():
                    acc.add(rid, packed)
            resumed = acc.n
            work_start = time.monotonic()
            emit("run_started", restored_draws=resumed, workers=workers)
            todo = (int(i) for i in schedule if not acc.complete[i])
            for rid, packed in _compute(todo, workers, compute):
                if journal is not None:
                    journal.append(rid, packed)
                acc.add(rid, packed)
                emit("replicate_completed", rid)
            result = _finish_run(
                header,
                acc,
                actual,
                threshold,
                resources=resources,
                resumed=resumed,
                workers=workers,
                elapsed=time.monotonic() - start,
            )
    except BaseException as exc:
        emit("run_failed", exception_type=type(exc).__name__)
        raise
    emit("run_completed")
    return result
