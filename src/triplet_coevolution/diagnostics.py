"""Offline workflow inventory and host resource checks."""

import ctypes
import os
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

from triplet_coevolution.config import COMPONENTS, load_config
from triplet_coevolution.operations import file_id, read_json
from triplet_coevolution.provenance import software_identity

MIB = 1024**2
THREAD_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
PROGRAMS = (
    "jackhmmer",
    "hmmsearch",
    "nhmmer",
    "hmmalign",
    "cmsearch",
    "cmalign",
    "psiblast",
    "makeblastdb",
)


def nearest_directory(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    if not path.is_dir():
        raise ValueError(f"expected directory for scratch/output: {path}")
    return path


def available_memory():
    if os.name == "nt":

        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (f"size{i}", ctypes.c_uint64) for i in range(7)
            ]

        mem = Memory()
        mem.length = ctypes.sizeof(mem)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
            return mem.size1
    info = Path("/proc/meminfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                # Respect the common cgroup-v2 container memory ceiling.
                root = Path("/sys/fs/cgroup")
                maximum, current = root / "memory.max", root / "memory.current"
                if maximum.is_file() and current.is_file():
                    limit = maximum.read_text().strip()
                    if limit.isdecimal():
                        available = min(available, max(0, int(limit) - int(current.read_text())))
                return available
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None


def resources(scratch):
    cpus = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    if hasattr(os, "sched_getaffinity"):
        cpus = min(cpus, len(os.sched_getaffinity(0)))
    allocation = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if allocation.isdecimal() and int(allocation) > 0:
        cpus = min(cpus, int(allocation))
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.is_file():
        maximum, period = quota.read_text().split()
        if maximum.isdecimal() and int(period) > 0:
            cpus = min(cpus, max(1, int(maximum) // int(period)))
    return {
        "logical_cpus": cpus,
        "available_memory_bytes": available_memory(),
        "free_disk_bytes": shutil.disk_usage(nearest_directory(scratch)).free,
        "filesystem": str(nearest_directory(scratch)),
        "thread_environment": {k: os.environ[k] for k in THREAD_VARS if k in os.environ},
    }


def resource_blockers(host, workers=1, blas_threads=1, memory_mib=None, disk_mib=0):
    if (
        workers < 1
        or blas_threads < 1
        or disk_mib < 0
        or memory_mib is not None
        and memory_mib <= 0
    ):
        raise ValueError(
            "workers, BLAS threads and memory must be positive, disk reserve nonnegative"
        )
    reasons = []
    if workers * blas_threads > host["logical_cpus"]:
        reasons.append(
            f"requested {workers} workers × {blas_threads} BLAS threads exceeds CPU allocation"
        )
    if memory_mib and (
        host["available_memory_bytes"] is None or memory_mib * MIB > host["available_memory_bytes"]
    ):
        reasons.append("requested memory exceeds the available or detectable host allocation")
    if disk_mib * MIB > host["free_disk_bytes"]:
        reasons.append("requested disk reserve exceeds free disk space")
    return reasons


def doctor_report(scratch, *, required=(), workers=1, blas_threads=1, memory_mib=None, disk_mib=0):
    host = resources(scratch)
    blockers = resource_blockers(host, workers, blas_threads, memory_mib, disk_mib)
    if set(required) - set(PROGRAMS):
        raise ValueError("unknown required executable; supported: " + ", ".join(PROGRAMS))
    tools = {}
    for name in PROGRAMS:
        path = shutil.which(name)
        tools[name] = {"path": path, "version_banner": None}
        if path:
            flag = "-version" if name in ("makeblastdb", "psiblast") else "-h"
            try:
                p = subprocess.run(
                    [path, flag],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=8,
                    text=True,
                    errors="replace",
                    env={**os.environ, "BLAST_USAGE_REPORT": "false"},
                    check=False,
                )
                banner = next(
                    (
                        line
                        for line in p.stdout.splitlines()
                        if "HMMER " in line or "INFERNAL " in line or line.startswith(name + ":")
                    ),
                    None,
                )
                if p.returncode == 0:
                    tools[name]["version_banner"] = banner
            except (OSError, subprocess.TimeoutExpired):
                pass
        if name in required and not tools[name]["version_banner"]:
            blockers.append(f"required executable unavailable or unrecognized: {name}")
    extras = {}
    for name in ("matplotlib", "scipy"):
        try:
            extras[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            extras[name] = None
    return {
        "schema_version": 1,
        "status": "blocked" if blockers else "ready",
        "blockers": blockers,
        "resources": host,
        "software": software_identity(),
        "optional_packages": extras,
        "tools": tools,
    }


def config_mode(path):
    raw = read_json(path)
    comps = raw.get("components", {}) if isinstance(raw, dict) else {}
    return (
        "prepare"
        if isinstance(comps, dict)
        and any(isinstance(c, dict) and "alignment" in c for c in comps.values())
        else "run"
    )


def validate_config(path):
    from triplet_coevolution.io.alignments import open_msa
    from triplet_coevolution.io.profiles import check_profile, read_reference, read_sequences
    from triplet_coevolution.matching.orthology import read_evidence
    from triplet_coevolution.matching.triplets import match_groups
    from triplet_coevolution.search.local_tools import tools_for
    from triplet_coevolution.workflow import _inputs

    mode = config_mode(path)
    cfg = load_config(path, mode)
    sources = {k: {"path": str(v), **file_id(v)} for k, v in _inputs(cfg, path).items()}
    groups, _, _ = read_evidence(cfg["evidence"])
    components, available, blockers = {}, {}, []
    for name, c in cfg["components"].items():
        ref = c.get("reference_row_id", c["reference_id"])
        if mode == "prepare":
            with open_msa(c["alignment"], name, c["molecule"], c.get("format")) as inv:
                if not inv.contains(ref):
                    raise ValueError(f"{name} missing coordinate reference row {ref!r}")
                wanted = {
                    r.sequence_id for rows in groups.values() for r in rows if r.component == name
                }
                present = {i for i in wanted if inv.contains(i)}
                if ref != c["reference_id"]:
                    present.discard(ref)
                available[name] = present
                components[name] = {
                    "format": inv.format,
                    "rows": inv.n_rows,
                    "columns": inv.n_columns,
                    "reference_row": ref,
                    "present_evidence_ids": len(present),
                }
        else:
            reference = read_reference(c)
            check_profile(c)
            rows = letters = 0
            for _, seq in read_sequences(c["database"], c["molecule"]):
                rows += 1
                letters += len(seq)
            try:
                tools = tools_for(c)
            except ValueError as exc:
                blockers.append(str(exc))
                tools = []
            components[name] = {
                "database_rows": rows,
                "database_residues": letters,
                "reference_length": len(reference),
                "backend": c["backend"],
                "executables": tools,
            }
    count = exclusions = None
    if mode == "prepare":
        matches, qc = match_groups(available, groups, cfg["paralogs"])
        count, exclusions = len(matches), qc["excluded_groups"]
        if not count:
            blockers.append("no complete resolved matched groups")
    if any(file_id(Path(s["path"]))["sha256"] != s["sha256"] for s in sources.values()):
        raise ValueError("input changed during validation")
    return {
        "schema_version": 1,
        "status": "blocked" if blockers else "ready",
        "blockers": blockers,
        "mode": mode,
        "inputs": sources,
        "components": components,
        "evidence_groups": len(groups),
        "matched_groups": count,
        "excluded_groups": exclusions,
        "next": "prepare then plan for retained QC axes, operation count, workspace and output estimates",
    }


def validate_folders(directory):
    from triplet_coevolution.auto_family import infer_molecule
    from triplet_coevolution.automatic_inputs import choose_source
    from triplet_coevolution.family_config import family_config
    from triplet_coevolution.io.stockholm import open_stockholm

    cfg, inputs, order = family_config(None, directory, uniprot=False, automatic=True)
    components = {}
    for k in COMPONENTS:
        c = cfg["components"][k]
        alternatives = [p for n, p in inputs.items() if n.startswith(k + ".candidate.")]
        source, choice = choose_source(alternatives or [c["alignment"]], c.get("family"))
        with open_stockholm(source, family=c.get("family"), molecule=c["molecule"]) as inv:
            components[k] = {
                "name": c["name"],
                "rows": inv.n_rows,
                "columns": inv.n_columns,
                "source_selection": choice,
                "molecule_decision": infer_molecule(inv, c["molecule"]),
            }
    return {
        "schema_version": 1,
        "status": "ready",
        "mode": "family_inventory",
        "components": components,
        "folder_order": order,
        "network_requests": 0,
        "inputs": {n: {"path": str(p), **file_id(p)} for n, p in inputs.items()},
        "matched_groups": None,
        "next": "run performs evidence resolution before QC and scoring",
    }
