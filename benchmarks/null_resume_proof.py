"""Kill a native calibration process after committed draws and verify a bit-identical resume."""

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path


def _completed(path):
    if not path.is_file():
        return 0
    try:
        with closing(sqlite3.connect(path, timeout=0.15)) as conn:
            row = conn.execute("SELECT n FROM state").fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc) or "no such table" in str(exc):
            return 0
        raise


def _fingerprint(directory):
    def find(value):
        if isinstance(value, dict):
            if "replicate_evidence_sha256" in value and "count_sha256" in value:
                return value
            for sub in value.values():
                found = find(sub)
                if found:
                    return found
        return None

    for f in sorted(directory.rglob("*.json")):
        found = find(json.loads(f.read_text(encoding="utf-8")))
        if found:
            return found
    raise ValueError("published calibration manifest lacks runner evidence digests")


def reproduce(fixture, draws=97):
    if draws < 32:
        raise ValueError("use at least 32 draws so the worker can be terminated mid-run")
    fixture = Path(fixture).resolve()
    prefix = [sys.executable, "-B", "-m", "triplet_coevolution.cli"]
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NO_COLOR": "1",
        "COLUMNS": "180",
    }

    def call(args, *, must_succeed=True):
        p = subprocess.run(
            [*prefix, *map(str, args)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        if must_succeed and p.returncode:
            raise RuntimeError(p.stdout[-4000:])
        return p

    with tempfile.TemporaryDirectory(prefix="triplet-null-recovery-") as td:
        root = Path(td)
        run = root / "native"
        source = []
        for role, name, molecule in zip(
            "abc", ("protein", "dna", "rna"), ("protein", "dna", "rna"), strict=True
        ):
            source.extend(
                [
                    f"--{role}",
                    fixture / f"{name}.fasta",
                    f"--{role}-type",
                    molecule,
                    f"--{role}-ref",
                    f"{'prot' if name == 'protein' else name}_ref",
                ]
            )
        call(["analyze", *source, "--matches", fixture / "matches.tsv", "--no-plots", "--out", run])
        config = root / "policy.json"
        policy = {
            "null": "whole_component",
            "statistics": ["mi_bits", "mi_apc"],
            "family": "interface",
            "draws": draws,
            "seed": 17829,
            "max_t": True,
        }
        config.write_text(json.dumps(policy), encoding="utf-8")
        base = ["calibrate", "--run", str(run), "--config", str(config)]
        cp = root / "durable"
        db = cp / "null.sqlite3"
        interrupted_out = root / "interrupted"
        logfile = root / "interrupt.log"
        with logfile.open("wb") as log:
            child = subprocess.Popen(
                [
                    *prefix,
                    *base,
                    "--out",
                    str(interrupted_out),
                    "--checkpoint",
                    str(cp),
                    "--workers",
                    "2",
                    "--progress",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 40
                while _completed(db) < 5:
                    if child.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(
                            "worker ended before interruption: "
                            + logfile.read_text(errors="replace")[-3000:]
                        )
                    time.sleep(0.008)
                child.kill()
                exit_code = child.wait(timeout=15)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=15)
        committed = _completed(db)
        if not 5 <= committed < draws:
            raise RuntimeError(f"test failed to kill mid-run; committed={committed}")
        if interrupted_out.exists() and list(interrupted_out.glob("*manifest*.json")):
            raise RuntimeError("the interrupted worker published a completed output manifest")
        resumed = root / "resumed"
        normal = root / "uninterrupted"
        call([*base, "--out", resumed, "--checkpoint", cp, "--resume", "--workers", "3"])
        call([*base, "--out", normal, "--workers", "1"])
        a, b = _fingerprint(resumed), _fingerprint(normal)
        for key in ("plan_sha256", "replicate_evidence_sha256", "count_sha256"):
            if a[key] != b[key]:
                raise ValueError("resumed and uninterrupted runs differ in " + key)
        tables = {}
        for f in sorted(normal.rglob("*.tsv")):
            rel = f.relative_to(normal)
            data = f.read_bytes()
            if (resumed / rel).read_bytes() != data:
                raise ValueError(f"resumed table disagrees: {rel}")
            tables[rel.as_posix()] = hashlib.sha256(data).hexdigest()
        if not tables or a["restored_draws"] != committed or a["completed_draws"] != draws:
            raise ValueError("incomplete recovery evidence")
        config.write_text(json.dumps({**policy, "seed": policy["seed"] + 1}), encoding="utf-8")
        mismatch = call(
            [*base, "--out", root / "refused", "--checkpoint", cp, "--resume"], must_succeed=False
        )
        if mismatch.returncode == 0 or "identity mismatch" not in mismatch.stdout.lower():
            raise ValueError(
                "changed seed did not trigger the checkpoint identity guard: "
                + mismatch.stdout[-2000:]
            )
        return {
            "fixture": "bundled mixed protein/DNA/RNA native matched alignment",
            "draws": draws,
            "killed_exit_code": exit_code,
            "restored_draws": committed,
            "additional_draws": draws - committed,
            "workers_interrupted_resumed_uninterrupted": [2, 3, 1],
            "changed_seed_rejected": True,
            "byte_identical_tables_sha256": tables,
            "count_sha256": a["count_sha256"],
            "replicate_evidence_sha256": a["replicate_evidence_sha256"],
            "checkpoint_database_bytes": db.stat().st_size,
        }


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument(
        "--fixture", type=Path, default=Path(__file__).resolve().parents[1] / "examples/mixed"
    )
    args.add_argument("--draws", type=int, default=97)
    options = args.parse_args()
    print(json.dumps(reproduce(options.fixture, options.draws), indent=2))
