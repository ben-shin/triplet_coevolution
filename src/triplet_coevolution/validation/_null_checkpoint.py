"""Atomic append-only null replicate journals and process-lifetime writer locks."""

import contextlib
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

DATABASE = "null.sqlite3"
SCHEMA = "triplet.null_checkpoint.v1"


def canonical(value):
    """Require a portable, unambiguous JSON representation for persisted identities."""

    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate keys in null identity after JSON conversion")
        return result

    normalized = json.loads(json.dumps(value, allow_nan=False), object_pairs_hook=unique)
    return json.dumps(
        normalized, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


@contextlib.contextmanager
def _exclusive(root):
    # Keep this file in place. Replacing a locked file would create a second inode
    # on POSIX. The OS releases the lock after SIGKILL, TerminateProcess, or exit.
    lock_path = root / "null.writer.lock"
    with lock_path.open("a+b") as lock:
        if not lock.seek(0, os.SEEK_END):
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError("another process owns this null checkpoint writer lock") from exc
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("another process owns this null checkpoint writer lock") from exc
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _state_digest(header, count, head):
    return digest(canonical([header, count, head]))


def _record_digest(header, step, replicate_id, packed):
    h = hashlib.sha256(bytes.fromhex(header))
    h.update(step.to_bytes(8, "big"))
    h.update(replicate_id.to_bytes(8, "big"))
    h.update(packed)
    return h.hexdigest()


def _chain(previous, record):
    return digest(bytes.fromhex(previous) + bytes.fromhex(record))


def _configure(connection):
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=2500")
    # Default rollback journal mode is local-file safe and doesn't leave a
    # long-lived WAL containing another full copy of every completed replicate.
    connection.execute("PRAGMA journal_mode=DELETE")


def _initialize(root, path, header, header_hash):
    fd, name = tempfile.mkstemp(prefix=".null-init-", suffix=".sqlite3", dir=root)
    os.close(fd)
    temp = Path(name)
    try:
        with contextlib.closing(sqlite3.connect(temp)) as conn:
            _configure(conn)
            with conn:
                conn.execute("CREATE TABLE header (data BLOB NOT NULL, sha256 TEXT NOT NULL)")
                conn.execute(
                    "CREATE TABLE state (n INTEGER NOT NULL, head TEXT NOT NULL, sha256 TEXT NOT NULL)"
                )
                conn.execute(
                    "CREATE TABLE draws (step INTEGER PRIMARY KEY, replicate_id INTEGER UNIQUE NOT NULL, payload BLOB NOT NULL, sha256 TEXT NOT NULL)"
                )
                conn.execute("INSERT INTO header VALUES (?,?)", (header, header_hash))
                conn.execute(
                    "INSERT INTO state VALUES (?,?,?)",
                    (0, header_hash, _state_digest(header_hash, 0, header_hash)),
                )
        os.replace(temp, path)
        if os.name != "nt":
            directory = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def _only(conn, sql):
    rows = conn.execute(sql).fetchmany(2)
    if len(rows) != 1:
        raise ValueError("null checkpoint has missing or duplicate metadata")
    return rows[0]


class Journal:
    def __init__(self, conn, header, expected_bytes):
        self.conn = conn
        self.expected_bytes = expected_bytes
        raw, tag = _only(conn, "SELECT data,sha256 FROM header")
        if not isinstance(raw, bytes) or digest(raw) != tag:
            raise ValueError("null checkpoint header checksum mismatch")
        self.header = json.loads(raw)
        if self.header.get("schema") != SCHEMA or canonical(header) != raw:
            actual = {
                k for k in set(self.header) | set(header) if self.header.get(k) != header.get(k)
            }
            raise ValueError("null checkpoint identity mismatch: " + ", ".join(sorted(actual)))
        self.hash = tag
        self.count, self.head, state_hash = _only(conn, "SELECT n,head,sha256 FROM state")
        if (
            type(self.count) is not int
            or not 0 <= self.count <= header["planned_draws"]
            or _state_digest(tag, self.count, self.head) != state_hash
        ):
            raise ValueError("null checkpoint completion metadata checksum mismatch")

    def records(self):
        """Verify immutable IDs, order, row bytes and final chain, including removed rows."""
        head = self.hash
        expected = 0
        ids = bytearray(self.header["planned_draws"])
        for step, rid, packed, tag in self.conn.execute(
            "SELECT step,replicate_id,payload,sha256 FROM draws ORDER BY step"
        ):
            if (
                step != expected
                or type(rid) is not int
                or not 0 <= rid < self.header["planned_draws"]
                or ids[rid]
                or not isinstance(packed, bytes)
                or len(packed) != self.expected_bytes
            ):
                raise ValueError(
                    "invalid, duplicate, missing or malformed null checkpoint replicate"
                )
            if _record_digest(self.hash, step, rid, packed) != tag:
                raise ValueError(f"null checkpoint replicate {rid} checksum mismatch")
            ids[rid] = 1
            expected += 1
            head = _chain(head, tag)
            yield rid, packed
        if expected != self.count or head != self.head:
            raise ValueError("null checkpoint has a gap or chain mismatch")

    def append(self, rid, packed):
        if len(packed) != self.expected_bytes:
            raise ValueError("invalid null checkpoint replicate payload length")
        record = _record_digest(self.hash, self.count, rid, packed)
        head = _chain(self.head, record)
        count = self.count + 1
        # Both SQL writes commit or roll back together, including disk-full cases.
        with self.conn:
            self.conn.execute(
                "INSERT INTO draws VALUES (?,?,?,?)", (self.count, rid, packed, record)
            )
            self.conn.execute(
                "UPDATE state SET n=?,head=?,sha256=?",
                (count, head, _state_digest(self.hash, count, head)),
            )
        self.count, self.head = count, head


@contextlib.contextmanager
def open_journal(checkpoint, header, expected_bytes, resume):
    if checkpoint is None:
        if resume:
            raise ValueError("resume requires a checkpoint directory")
        yield None
        return
    root = Path(checkpoint)
    if not resume:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError("checkpoint directory is missing")
    path = root / DATABASE
    with _exclusive(root):
        if resume and not path.is_file():
            raise ValueError("no null checkpoint exists to resume")
        if not resume:
            if path.exists():
                raise ValueError(
                    "null checkpoint already exists; resume explicitly or choose another directory"
                )
            raw = canonical(header)
            _initialize(root, path, raw, digest(raw))
        try:
            with contextlib.closing(sqlite3.connect(path)) as conn:
                _configure(conn)
                yield Journal(conn, header, expected_bytes)
        except sqlite3.Error as exc:
            raise ValueError(f"null checkpoint database failure: {exc}") from exc
