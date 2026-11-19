"""Common representation of hits and execution of local profile programs."""

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Hit:
    seq_id: str
    first: int
    last: int
    strand: str
    score: float
    evalue: float | None = None
    identity: float | None = None
    coverage: float | None = None
    model_strand: str | None = None


def execute(args, cwd=None):
    executable = shutil.which(args[0])
    if executable is None:
        raise ValueError(
            f"missing search executable {args[0]!r}; install it or use backend reference"
        )
    result = subprocess.run(
        [executable, *(str(v) for v in args[1:])],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        check=False,
        cwd=cwd,
        env={
            **os.environ,
            "BLAST_USAGE_REPORT": "false",
            "NCBI_CONFIG__BLAST__DATA_LOADERS": "none",
        },
    )
    if result.returncode:
        raise ValueError(
            f"{args[0]} exited with {result.returncode}: {result.stderr.strip()[-1600:]}"
        )


def table_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                yield line.split()


def external_hit(fields, seq_id, start, end, score, evalue, strand="+"):
    a, b = int(fields[start]), int(fields[end])
    value, bits = float(fields[evalue]), float(fields[score])
    if not math.isfinite(value) or value < 0 or not math.isfinite(bits) or min(a, b) < 1:
        raise ValueError("invalid reported search score, E-value, or coordinate")
    if strand not in ("+", "-") or (strand == "+" and b < a) or (strand == "-" and a < b):
        raise ValueError("invalid reported hit orientation")
    return Hit(seq_id, min(a, b), max(a, b), strand, bits, value)
