"""Time native serial, three-process and replayed shards on the packaged synthetic mixed MSA."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def sample_process_tree(pid):
    """Sum sampled RSS and OS threads. Shared pages can be counted by several processes."""
    if not Path("/proc/self/status").is_file():
        return None, None, None
    stack = [pid]
    rss = threads = processes = 0
    while stack:
        p = stack.pop()
        try:
            proc = Path("/proc") / str(p)
            rows = dict(
                line.split(":", 1)
                for line in (proc / "status").read_text().splitlines()
                if ":" in line
            )
            rss += int(rows.get("VmRSS", "0").split()[0]) * 1024
            threads += int(rows["Threads"])
            processes += 1
            stack.extend(map(int, (proc / f"task/{p}/children").read_text().split()))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rss, threads, processes


def benchmark(repo, root, draws):
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for k in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "GOTO_NUM_THREADS",
    ):
        env[k] = "1"
    cli = [sys.executable, "-m", "triplet_coevolution.cli"]
    observation = {}

    def run(label, args, extra_env=None):
        begin = time.perf_counter()
        peaks = [None, None, None]
        with (
            (root / f"{label}.log").open("wb") as log,
            subprocess.Popen(
                list(map(str, args)),
                env={**env, **(extra_env or {})},
                stdout=log,
                stderr=subprocess.STDOUT,
            ) as p,
        ):
            while p.poll() is None:
                for i, v in enumerate(sample_process_tree(p.pid)):
                    if v is not None:
                        peaks[i] = max(peaks[i] or 0, v)
                time.sleep(0.02)
            if p.returncode:
                raise RuntimeError(f"{label} failed: {(root / f'{label}.log').read_text()[-1600:]}")
        observation[label] = {
            "elapsed_seconds": round(time.perf_counter() - begin, 4),
            "sampled_tree_peak_rss_bytes": peaks[0],
            "max_OS_threads": peaks[1],
            "max_processes": peaks[2],
        }

    ex = repo / "examples/mixed"
    fit, cfg, plan = root / "fit", root / "policy.json", root / "plan"
    cfg.write_text(
        json.dumps(
            {
                "draws": draws,
                "null": "whole_component",
                "statistics": ["mi_bits", "mi_apc"],
                "family": "interface",
                "seed": 942,
                "max_t": True,
                "memory_mib": 128,
            }
        )
    )
    run(
        "source",
        [
            *cli,
            "analyze",
            "--a",
            ex / "protein.fasta",
            "--b",
            ex / "dna.fasta",
            "--c",
            ex / "rna.fasta",
            "--b-type",
            "dna",
            "--c-type",
            "rna",
            "--matches",
            ex / "matches.tsv",
            "--a-ref",
            "prot_ref",
            "--b-ref",
            "dna_ref",
            "--c-ref",
            "rna_ref",
            "--out",
            fit,
        ],
    )
    run("serial", [*cli, "calibrate", "--run", fit, "--config", cfg, "--out", root / "serial"])
    run("plan", [*cli, "shard-plan", "--run", fit, "--config", cfg, "--shards", 3, "--out", plan])
    local = [
        sys.executable,
        "-m",
        "triplet_coevolution.validation.shards_local",
        "--plan",
        plan,
        "--jobs",
        3,
        "--workers",
        1,
        "--memory-mib",
        1024,
    ]
    run("three_concurrent_processes", local)
    shards = [plan / "shards" / f"shard-{i:05d}" for i in range(3)]
    run(
        "merge",
        [
            *cli,
            "merge-null",
            "--plan",
            plan,
            "--out",
            root / "merged",
            "--shard",
            shards[2],
            "--shard",
            shards[0],
            "--shard",
            shards[1],
        ],
    )
    run("completed_resume", [*local, "--resume"])
    if os.name == "posix":
        # Exercise the unmodified array body locally using the current Python environment.
        shim = root / "triplet-host"
        import shlex

        shim.write_text(
            "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' -m triplet_coevolution.cli "$@"\n'
        )
        shim.chmod(0o700)
        run(
            "array_body_local_replay",
            ["bash", repo / "examples/hpc/null-array.sh"],
            {
                "TRIPLET_BIN": str(shim),
                "TRIPLET_PLAN": str(plan),
                "SLURM_ARRAY_JOB_ID": "123",
                "SLURM_ARRAY_TASK_ID": "1",
                "SLURM_CPUS_PER_TASK": "1",
            },
        )
    metric = lambda p: hashlib.sha256((p / "calibration.tsv").read_bytes()).hexdigest()
    serial, merged = root / "serial", root / "merged"
    table_hash = metric(serial)
    assert table_hash == metric(merged)
    s = json.loads((serial / "calibration.json").read_text())
    m = json.loads((merged / "calibration.json").read_text())
    for k in ("plan_sha256", "replicate_evidence_sha256", "count_sha256"):
        assert s["runner"][k] == m["runner"][k]
    return {
        "draws": draws,
        "source": "packaged synthetic examples/mixed",
        "table_sha256": table_hash,
        "tests": sum(f["tests"] for f in m["families"].values()),
        "families": len(m["families"]),
        "replicate_evidence_sha256": m["runner"]["replicate_evidence_sha256"],
        "equal_native_calibration_tsv": True,
        "shard_db_bytes": sum((p / "null.sqlite3").stat().st_size for p in shards),
        "resources": observation,
        "sample_interval_seconds": 0.02,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--draws", type=int, default=127)
    opt = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="triplet-null-shards-") as tmp:
        print(json.dumps(benchmark(opt.repo, Path(tmp), opt.draws), indent=2))
