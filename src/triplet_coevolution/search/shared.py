"""Common representation of hits and execution of local profile programs."""

import math
import os
import re
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


def execute(args, cwd=None, *, capture_stdout=False):
    executable = shutil.which(args[0])
    if executable is None:
        raise ValueError(
            f"missing search executable {args[0]!r}; install it or use backend reference"
        )
    result = subprocess.run(
        [executable, *(str(v) for v in args[1:])],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
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
    return str(Path(executable).absolute()), result.stdout


def program_versions(comp):
    names = {
        "reference": (),
        "hmmer": ("hmmsearch" if comp["molecule"] == "protein" else "nhmmer",),
        "infernal": ("cmsearch",),
        "psiblast": ("makeblastdb", "psiblast"),
    }[comp["backend"]]
    programs = []
    for name in names:
        is_blast = name in ("psiblast", "makeblastdb")
        path, banner = execute([name, "-version" if is_blast else "-h"], capture_stdout=True)
        prefix = f"{name}:" if is_blast else (r"# INFERNAL" if name == "cmsearch" else r"# HMMER")
        match = re.search(rf"^{prefix}\s+(\S+)", banner, flags=re.MULTILINE)
        if match is None:
            raise ValueError(f"could not identify {name} version from its banner: {path}")
        programs.append({"name": name, "path": path, "version": match.group(1)})
    return programs


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
