"""Content-addressed local stages with an OS-released writer lease and verifiable receipts."""

import hashlib
import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_json(path):
    from triplet_coevolution.config import _unique

    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique)


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def file_id(path, on_bytes=None):
    path = Path(path)
    with path.open("rb") as f:
        before = os.fstat(f.fileno())
        if on_bytes is None:
            checksum = hashlib.file_digest(f, "sha256").hexdigest()
        else:
            hasher = hashlib.sha256()
            while block := f.read(1024 * 1024):
                hasher.update(block)
                on_bytes(len(block))
            checksum = hasher.hexdigest()
        after = os.fstat(f.fileno())
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if stamp(before) != stamp(after) or stamp(after) != stamp(path.stat()):
        raise ValueError(f"input changed during checksum: {path}")
    return {"sha256": checksum, "bytes": before.st_size}


def inventory(directory, progress=None):
    paths = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError(f"stage contains a linked path: {path}")
        if path.is_file():
            paths.append(path)
        elif not path.is_dir():
            raise ValueError(f"stage contains a special object: {path}")
    total = sum(p.stat().st_size for p in paths)
    processed = 0
    start = last = time.monotonic()

    def advance(size):
        nonlocal processed, last
        processed += size
        now = time.monotonic()
        if progress and (now - last >= 5 or processed == total) and now > start:
            rate = processed / (now - start)
            progress(
                {
                    "verified_bytes": processed,
                    "total_bytes": total,
                    "bytes_per_second": rate,
                    "estimated_remaining_seconds": max(0, total - processed) / rate,
                }
            )
            last = now

    return {p.relative_to(directory).as_posix(): file_id(p, advance) for p in paths}


@contextmanager
def pipeline_workspace(directory):
    """Claim only an empty directory or a recognized workspace; never adopt other data."""
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "workspace.json"
    contract = {
        "schema_version": 1,
        "application": "triplet-coevolution",
        "kind": "stage_workspace",
    }
    if not marker.is_file() and any(p.name != "pipeline.lock" for p in directory.iterdir()):
        raise ValueError(f"nonempty unrecognized pipeline workspace: {directory}")
    with lease(directory):
        if marker.is_file():
            if marker.is_symlink() or read_json(marker) != contract:
                raise ValueError(f"invalid workspace receipt: {directory}")
        else:
            put_json(marker, contract)
        for name in ("objects", "runs", "specs", "provider-cache", "checkpoints"):
            p = directory / name
            if p.is_symlink() or getattr(p, "is_junction", lambda: False)():
                raise ValueError(f"linked workspace directory: {p}")
        yield


@contextmanager
def lease(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "pipeline.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            acquire = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise ValueError(f"another process owns the pipeline workspace: {directory}") from exc
        try:
            yield
        finally:
            release()


class Journal:
    def __init__(self, root, emit=None):
        self.root = root / "runs" / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.emit = emit
        self.start = time.monotonic()
        self.stages = []
        self.verified_outputs = {}

    def remember(self, path, files):
        self.verified_outputs[path] = {
            name: (self._stamp(path / name), attrs) for name, attrs in files.items()
        }

    @staticmethod
    def _stamp(path):
        st = path.stat()
        return (st.st_dev, st.st_ino, st.st_size, st.st_ctime_ns, st.st_mtime_ns)

    def check_output(self, path):
        # The entire directory was hashed while acquiring its receipt. Detect later changes
        # before/after its consumers without rereading unchanged large arrays for every view.
        known = self.verified_outputs[path]
        actual = {f.relative_to(path).as_posix() for f in path.rglob("*") if f.is_file()}
        if actual != known.keys():
            raise ValueError(f"published artifact inventory changed during pipeline: {path}")
        for name, (stamp, identity) in known.items():
            f = path / name
            if f.is_symlink() or self._stamp(f) != stamp and file_id(f) != identity:
                raise ValueError(f"published artifact changed during pipeline: {f}")

    def event(self, event, **details):
        record = {
            "time_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - self.start,
            "event": event,
            "supervisor_pid": os.getpid(),
            **details,
        }
        with (self.root / "events.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if self.emit:
            self.emit(record)

    def finish(self, status, **details):
        summary = {"schema_version": 1, "status": status, "stages": self.stages, **details}
        self.event("run_finished", status=status)
        put_json(self.root / "pipeline.json", summary)
        return self.root / "pipeline.json"


def stage(root, journal, name, inputs, produce, minimum_disk=0, *, public_location=False):
    """Fingerprint sources before and after work and publish a complete artifact directory."""
    sources = inputs()
    identity = {"schema_version": 1, "stage": name, "inputs": sources}
    key = digest(identity)
    obj = root / "objects" / name / key
    journal.event("stage_started", stage=name, key=key)
    started = time.monotonic()

    def verified(row):
        journal.event("artifact_verification", stage=name, **row)

    if obj.exists():
        if any(
            p.is_symlink() or getattr(p, "is_junction", lambda: False)()
            for p in (obj, obj / "output", obj / "receipt.json")
        ):
            raise ValueError(f"cached {name} contains a linked object")
        receipt = read_json(obj / "receipt.json")
        files = inventory(obj / "output", verified)
        if (
            {p.name for p in obj.iterdir()} != {"receipt.json", "output"}
            or not (obj / "output").is_dir()
            or receipt.get("identity") != identity
            or receipt.get("artifacts") != files
            or not files
            or receipt.get("receipt_sha256") != digest({"key": key, "artifacts": files})
        ):
            raise ValueError(
                f"cached {name} failed verification: {obj}. Preserve it and use a fresh workspace."
            )
        if inputs() != sources:
            raise ValueError(f"{name} inputs changed during verification")
        action = "reused"
    else:
        attempt = journal.root / f"{name}-attempt"
        attempt.mkdir()
        if shutil.disk_usage(root).free < minimum_disk:
            raise OSError(f"insufficient free storage for {name}: need {minimum_disk} bytes")
        output = attempt / "output"
        if public_location:
            produce(output, obj / "output")
        else:
            produce(output)
        files = inventory(output, verified)
        if not files:
            raise ValueError(f"{name} produced no artifacts")
        if inputs() != sources:
            raise ValueError(f"{name} inputs changed during execution")
        receipt = {
            "identity": identity,
            "artifacts": files,
            "receipt_sha256": digest({"key": key, "artifacts": files}),
        }
        put_json(attempt / "receipt.json", receipt)
        obj.parent.mkdir(parents=True, exist_ok=True)
        attempt.rename(obj)
        action = "executed"
    result = {
        "stage": name,
        "action": action,
        "key": key,
        "output": str(obj / "output"),
        "receipt_sha256": receipt["receipt_sha256"],
        "files": len(files),
        "artifact_bytes": sum(f["bytes"] for f in files.values()),
        "elapsed_seconds": time.monotonic() - started,
    }
    measured = journal.root / (name + ".resources.json")
    if action == "executed" and measured.is_file():
        result["measured"] = read_json(measured)
    result["artifact_bytes_per_stage_second"] = result["artifact_bytes"] / max(
        result["elapsed_seconds"], 1e-9
    )
    journal.remember(obj / "output", files)
    journal.stages.append(result)
    journal.event("stage_finished", **result)
    return obj / "output"
