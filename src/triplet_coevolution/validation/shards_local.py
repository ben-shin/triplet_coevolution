"""Launch native shard processes on one machine with declared aggregate CPU/memory bounds."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, required=True, help="Existing or new shard plan directory"
    )
    parser.add_argument("--run", type=Path, help="Native source when preparing a new plan")
    parser.add_argument("--config", type=Path, help="Null policy when preparing a new plan")
    parser.add_argument("--shards", type=int, help="Partition count when preparing a new plan")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--memory-mib", type=int, required=True, help="Total job memory allowance")
    parser.add_argument(
        "--overhead-mib",
        type=int,
        default=128,
        help="Reserve per interpreter including this launcher",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, help="Publish final calibration if all shards succeed")
    opts = parser.parse_args(argv)
    if min(opts.jobs, opts.workers, opts.memory_mib, opts.overhead_mib) < 1:
        parser.error("resource budgets and parallelism must be positive")
    available = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    )
    if "SLURM_CPUS_PER_TASK" in os.environ:
        available = min(available, int(os.environ["SLURM_CPUS_PER_TASK"]))
    if opts.jobs * opts.workers > available:
        parser.error("jobs times workers exceeds available logical CPUs")
    cli = [sys.executable, "-m", "triplet_coevolution.cli"]
    thread_names = (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "GOTO_NUM_THREADS",
    )
    env = {**os.environ, **dict.fromkeys(thread_names, "1")}
    plan = opts.plan.resolve()
    document = plan / "shard_plan.json"
    if not document.is_file():
        if None in (opts.run, opts.config, opts.shards):
            parser.error("new plans require --run, --config and --shards")
        # Refuse an oversized single-process plan before loading its scientific inputs.
        policy = json.loads(opts.config.read_text(encoding="utf-8-sig"))
        if policy.get("memory_mib", 512) + 2 * opts.overhead_mib > opts.memory_mib:
            parser.error(
                "plan preparation exceeds total memory allowance including interpreter reserves"
            )
        subprocess.run(
            [
                *cli,
                "shard-plan",
                "--run",
                str(opts.run),
                "--config",
                str(opts.config),
                "--shards",
                str(opts.shards),
                "--out",
                str(plan),
            ],
            env=env,
            check=True,
        )
    elif any(v is not None for v in (opts.run, opts.config, opts.shards)):
        parser.error(
            "existing plan already fixes the source, config and partitions; supply --plan alone"
        )
    from triplet_coevolution.shard_commands import load_native_plan

    _, spec = load_native_plan(plan)
    memory = opts.jobs * (spec["process_memory_mib"] + opts.overhead_mib) + opts.overhead_mib
    if memory > opts.memory_mib:
        parser.error(
            f"planned concurrent processes need {memory} MiB including interpreter reserves"
        )
    if spec["numerical"]["runtime"]["thread_environment"] != dict.fromkeys(thread_names, "1"):
        parser.error(
            "this local launcher requires a plan prepared with all six numerical thread settings at one; create it with --run/--config/--shards"
        )
    logs = plan / "logs"
    logs.mkdir(exist_ok=True)
    pending = iter(range(spec["shards"]))
    running = {}

    def event(name, **values):
        print(json.dumps({"event": name, **values}), flush=True)

    try:
        while True:
            while len(running) < opts.jobs:
                i = next(pending, None)
                if i is None:
                    break
                with (logs / f"shard-{i:05d}.jsonl").open("ab") as f:
                    cmd = [
                        *cli,
                        "null-shard",
                        "--plan",
                        str(plan),
                        "--shard-id",
                        str(i),
                        "--workers",
                        str(opts.workers),
                        "--progress",
                    ]
                    if opts.resume:
                        cmd.append("--resume")
                    proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
                running[i] = proc
                event("local_shard_started", shard_id=i, pid=proc.pid)
            if not running:
                break
            for i, proc in list(running.items()):
                code = proc.poll()
                if code is not None:
                    del running[i]
                    event("local_shard_exited", shard_id=i, returncode=code)
                    if code:
                        raise RuntimeError(
                            f"shard {i} failed; inspect its log and retry with --resume"
                        )
            if running:
                time.sleep(0.05)
    finally:
        for proc in running.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in running.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    if opts.out:
        subprocess.run(
            [*cli, "merge-null", "--plan", str(plan), "--out", str(opts.out), "--progress"],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
