"""Source selection and reusable lookup snapshots for unattended family projects."""

import gzip
import hashlib
import re
import sqlite3
from contextlib import closing

from triplet_coevolution.io.stockholm import open_stockholm


def _contents(path):
    with path.open("rb") as source:
        compressed = source.read(2) == b"\x1f\x8b"
        source.seek(0)
        if compressed:
            with gzip.open(source, "rb") as data:
                return hashlib.file_digest(data, "sha256").hexdigest()
        return hashlib.file_digest(source, "sha256").hexdigest()


def choose_source(candidates, family):
    """Collapse byte-identical encodings, then distinguish matching full/seed releases."""
    copies = {}
    for path in candidates:
        copies.setdefault(_contents(path), []).append(path)
    variants = []
    detail = []
    for paths in copies.values():
        path = min(paths, key=lambda p: (p.suffix.lower() == ".gz", p.name))
        kinds = {
            token
            for p in paths
            for token in re.split(r"[^a-z0-9]+", p.name.lower())
            if token in {"full", "seed"}
        }
        with open_stockholm(path, family=family) as inv:
            record = {
                "family": inv.accession or inv.family_id,
                "rows": inv.n_rows,
                "columns": inv.n_columns,
                "files": [str(p) for p in paths],
            }
        detail.append(record)
        variants.append(
            (path, "full" if "full" in kinds else "seed" if "seed" in kinds else "", record)
        )
    if len(variants) == 1:
        winner, reason = variants[0][0], "identical decompressed contents"
    else:
        identifiers = {r["family"] for _, _, r in variants}
        full = [(p, r) for p, kind, r in variants if kind == "full"]
        if not (
            len(identifiers) == 1
            and all(identifiers)
            and len(full) == 1
            and all(kind for _, kind, _ in variants)
            and all(full[0][1]["rows"] >= r["rows"] for _, _, r in variants)
        ):
            raise ValueError(
                f"{candidates[0].parent}: competing Stockholm inputs "
                f"({', '.join(p.name for p in candidates)}); supply one family or set "
                "alignment in component.json before a future run"
            )
        winner, reason = full[0][0], "full alignment of the same identified family"
    return winner, {"path": str(winner), "reason": reason, "candidates": detail}


def _backup(source, destination):
    with (
        closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as src,
        closing(sqlite3.connect(destination)) as dest,
    ):
        src.backup(dest)


def lookup_snapshot(configured, cache_directory, metadata, filename, online):
    shared = cache_directory / filename
    source = configured or (shared if shared.is_file() else None)
    target = metadata / filename
    if source is not None:
        _backup(source, target)
        if configured and shared.is_file():
            _merge(shared, target, filename)
    return target if source is not None or online else None


def _merge(source, target, filename, keys=None):
    """Add missing immutable lookup records with concurrent writers serialized by SQLite."""
    table, key = ("lookups", "id") if filename == "uniprot.sqlite" else ("annotations", "key")
    with closing(sqlite3.connect(target, timeout=30, uri=True)) as db, db:
        db.execute("PRAGMA trusted_schema=OFF")
        db.execute("ATTACH DATABASE ? AS snapshot", (source.resolve().as_uri() + "?mode=ro",))
        db.execute("BEGIN IMMEDIATE")
        version = db.execute("PRAGMA snapshot.user_version").fetchone()[0]
        app_id = db.execute("PRAGMA snapshot.application_id").fetchone()[0]
        if not db.execute("SELECT 1 FROM main.sqlite_master").fetchone():
            db.execute(f"CREATE TABLE main.{table} ({key} TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            db.execute(f"PRAGMA main.user_version={version}")
            db.execute(f"PRAGMA main.application_id={app_id}")
        if (
            db.execute("PRAGMA main.application_id").fetchone()[0] != app_id
            or db.execute("PRAGMA main.user_version").fetchone()[0] != version
        ):
            raise ValueError(f"incompatible schemas for {filename}")
        insert = (
            f"INSERT OR IGNORE INTO main.{table} ({key}, payload) "
            f"SELECT {key}, payload FROM snapshot.{table}"
        )
        if keys is None:
            db.execute(insert)
        else:
            db.executemany(insert + f" WHERE {key}=?", ((k,) for k in keys))


def save_lookup_batch(metadata, cache_directory, filename, keys):
    """Durably publish each validated provider batch before fetching the next one."""
    cache_directory.mkdir(parents=True, exist_ok=True)
    _merge(metadata / filename, cache_directory / filename, filename, keys)


def save_snapshots(metadata, cache_directory):
    """Each completed lookup remains in the run snapshot and in the reusable project cache."""
    for name in ("uniprot.sqlite", "nucleotide.sqlite"):
        source = metadata / name
        if source.is_file():
            cache_directory.mkdir(parents=True, exist_ok=True)
            _merge(source, cache_directory / name, name)
