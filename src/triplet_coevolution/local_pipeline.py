"""Resume configured local scientific stages using receipts over exact inputs and outputs."""

import html
import os
import subprocess
import sys
import time
from pathlib import Path

from triplet_coevolution.config import load_config
from triplet_coevolution.diagnostics import (
    MIB,
    THREAD_VARS,
    config_mode,
    resource_blockers,
    resources,
    validate_config,
)
from triplet_coevolution.operations import (
    Journal,
    digest,
    file_id,
    pipeline_workspace,
    put_json,
    read_json,
    stage,
)
from triplet_coevolution.policy_inputs import dependencies
from triplet_coevolution.provenance import package_code_digest, software_identity
from triplet_coevolution.search.local_tools import tools_for
from triplet_coevolution.workflow import _inputs

STAGES = ("prepare", "plan", "score", "select", "calibrate", "benchmark", "report")


def tokens(settings):
    args = []
    for k, v in settings.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            args.append(flag)
        else:
            for item in v if isinstance(v, list) else [v]:
                args.extend([flag, str(item)])
    return args


def _snapshot(raw, cfg):
    """Keep only declared preparation fields, resolving all paths before moving configuration."""
    public = {k: v for k, v in raw.items() if k not in ("components", "analysis")}
    public["evidence"] = str(cfg["evidence"])
    public["components"] = {
        name: {
            k: str(cfg["components"][name][k])
            if isinstance(cfg["components"][name][k], Path)
            else value
            for k, value in c.items()
        }
        for name, c in raw["components"].items()
    }
    return public


def _run(args, output, journal, env, name, cwd=None):
    cmd = [
        sys.executable,
        "-m",
        "triplet_coevolution.stage_worker",
        str(journal.root / (name + ".resources.json")),
        *map(str, args),
        "--out",
        str(output),
    ]
    start = time.monotonic()
    with (
        (journal.root / (name + ".stdout.log")).open("wb") as stdout,
        (journal.root / (name + ".stderr.log")).open("wb") as stderr,
    ):
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.PIPE,
            env={
                **env,
                "TRIPLET_SUPERVISED_STAGE": "1",
                "TRIPLET_SUPERVISOR_PID": str(os.getpid()),
            },
            start_new_session=os.name != "nt",
        )
        try:
            while True:
                try:
                    code = process.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    journal.event(
                        "stage_running", stage=name, stage_seconds=time.monotonic() - start
                    )
        finally:
            process.stdin.close()
            if process.poll() is None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait()
    if (
        code == 3
        and name == "plan"
        and (output / "plan.json").is_file()
        and read_json(output / "plan.json").get("status") == "blocked"
    ):
        return
    if code:
        tail = (
            (journal.root / (name + ".stderr.log"))
            .read_bytes()[-1400:]
            .decode("utf-8", errors="replace")
        )
        raise ValueError(f"{name} exited {code}: {tail.strip() or 'see stage logs'}")


def run_pipeline(
    config,
    work_dir,
    *,
    until="report",
    labels=None,
    null_config=None,
    workers=1,
    blas_threads=1,
    memory_mib=None,
    disk_mib=0,
    dry_run=False,
    emit=None,
):
    config = Path(config).resolve()
    root = Path(work_dir).resolve()
    if until not in STAGES:
        raise ValueError("unknown stage; choose " + ", ".join(STAGES))
    mode = config_mode(config)
    cfg = load_config(config, mode)
    raw = read_json(config)
    settings = dict(cfg.get("analysis", {}))
    original_config_id = file_id(config)
    if config.is_relative_to(root):
        raise ValueError("use a work directory separate from the source configuration")
    labels = Path(labels).resolve() if labels else settings.get("contacts")
    null_config = Path(null_config).resolve() if null_config else settings.get("null_policy")
    if until == "calibrate" and not null_config or until == "benchmark" and not labels:
        raise ValueError(
            "the requested final stage requires --null-config or --labels respectively"
        )
    preview = {
        "schema_version": 1,
        "mode": mode,
        "requested_final_stage": until,
        "stages": [
            s
            for s in STAGES[: STAGES.index(until) + 1]
            if (s != "benchmark" or labels) and (s != "calibrate" or null_config)
        ],
        "resources": resources(root),
    }
    blockers = resource_blockers(preview["resources"], workers, blas_threads, memory_mib, disk_mib)
    for name, c in cfg["components"].items():
        if c.get("cpu", 1) > workers:
            blockers.append(f"{name} search cpu exceeds --workers")
    if settings.get("dca_policy"):
        from triplet_coevolution.inference.dca_models import DCADesign

        design = DCADesign.parse(settings["dca_policy"])
        if design.backend == "plm" and design.plm_options().workers > workers:
            blockers.append("PLM target workers exceed --workers; adjust --workers or dca_policy")
    if blockers:
        raise ValueError("; ".join(blockers))
    if dry_run:
        preview.update(status="dry_run", validation=validate_config(config))
        return preview
    runtime = software_identity()
    runtime.pop("git", None)  # A documentation-only commit has no numerical effect.
    from importlib.metadata import PackageNotFoundError, version

    for name in ("matplotlib", "scipy"):
        try:
            runtime["dependencies"][name] = version(name)
        except PackageNotFoundError:
            runtime["dependencies"][name] = None
    runtime["blas_threads"] = blas_threads
    env = {
        **os.environ,
        **dict.fromkeys(THREAD_VARS, str(blas_threads)),
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "BLAST_USAGE_REPORT": "false",
        "PYTHONUTF8": "1",
    }
    with pipeline_workspace(root):
        journal = Journal(root, emit)
        journal.event("run_started", requested_final_stage=until, stages=preview["stages"])

        def finish(status, reason=None):
            links = []
            for result in journal.stages:
                path = Path(result["output"])
                target = next(
                    (
                        path / n
                        for n in (
                            "report.html",
                            "benchmark.json",
                            "calibration.json",
                            "candidate_manifest.json",
                            "run_summary.json",
                            "plan.json",
                            "workflow_summary.json",
                        )
                        if (path / n).is_file()
                    ),
                    path,
                )
                from urllib.parse import quote

                links.append(
                    '<li><a href="'
                    + quote(os.path.relpath(target, journal.root).replace(os.sep, "/"))
                    + '">'
                    + html.escape(result["stage"])
                    + "</a> ("
                    + result["action"]
                    + ")</li>"
                )
            (journal.root / "index.html").write_text(
                "<!doctype html><meta charset=utf-8><title>Pipeline results</title><h1>Pipeline results</h1><p>"
                + html.escape(status)
                + "</p><ul>"
                + "".join(links)
                + "</ul>",
                encoding="utf-8",
            )
            manifest = journal.finish(
                status,
                reason=reason,
                report="index.html",
                config_sha256=original_config_id["sha256"],
            )
            return {**read_json(manifest), "manifest": str(manifest)}

        try:
            spec = _snapshot(raw, cfg)
            specfile = root / "specs" / (digest(spec) + ".json")
            if specfile.exists() and read_json(specfile) != spec:
                raise ValueError("workflow snapshot checksum mismatch")
            if not specfile.exists():
                put_json(specfile, spec)

            def prep_identity():
                if read_json(specfile) != spec:
                    raise ValueError("workflow snapshot changed during preparation")
                sources = {
                    k: {"path": str(p), **file_id(p)}
                    for k, p in _inputs(cfg, config).items()
                    if k != "config" and not k.startswith("analysis.")
                }
                tools = (
                    {n: tools_for(c) for n, c in cfg["components"].items()} if mode == "run" else {}
                )
                return {
                    "settings": spec,
                    "sources": sources,
                    "executables": tools,
                    "environment": runtime,
                }

            def execute(name, args, inputs, *, cwd=None, space=0):
                def snapshot():
                    if package_code_digest() != runtime["code_sha256"]:
                        raise ValueError(
                            "installed source changed during pipeline; retry on a stable installation"
                        )
                    if file_id(config) != original_config_id:
                        raise ValueError("workflow configuration changed during the run")
                    return inputs()

                return stage(
                    root,
                    journal,
                    name,
                    snapshot,
                    lambda p: _run(args, p, journal, env, name, cwd),
                    minimum_disk=int(disk_mib * MIB + space),
                )

            preparation = execute(
                "prepare",
                ["prepare", "--config", specfile, *(["--raw"] if mode == "run" else [])],
                prep_identity,
            )
            if until == "prepare":
                return finish("complete")
            summary = read_json(preparation / "workflow_summary.json")
            if not summary["matched_groups"]:
                return finish("insufficient_data", "No resolved matches for scoring")
            replay = preparation / summary["analyze_args"]
            ready = replay.parent
            base = read_json(replay)[1:]
            prep_receipt = journal.stages[-1]["receipt_sha256"]
            # --legacy contacts.json belongs to the historical scoring interface.
            if not settings.get("legacy"):
                settings.pop("contacts", None)
            for key in ("null_policy", "benchmark_bootstrap", "benchmark_k"):
                settings.pop(key, None)
            if memory_mib:
                settings["memory_mib"] = min(memory_mib, settings.get("memory_mib", memory_mib))

            def score_identity():
                journal.check_output(preparation)
                deps = {}
                for kind, opt in settings.items():
                    if isinstance(opt, Path):
                        deps.update(dependencies(kind, opt))
                return {
                    "prepared": prep_receipt,
                    "settings": {
                        k: str(v) if isinstance(v, Path) else v for k, v in settings.items()
                    },
                    "dependencies": deps,
                    "environment": runtime,
                }

            argv = base + tokens(settings)
            planned = execute("plan", ["plan", *argv], score_identity, cwd=ready)
            forecast = read_json(planned / "plan.json")
            if forecast["status"] == "blocked":
                return finish("blocked", "see plan.json for model or resource blockers")
            if until == "plan":
                return finish("complete")
            estimates = forecast.get("estimates", {})
            output_estimate = estimates.get("score_text_bytes_approx", 0) + estimates.get(
                "matrix_payload_before_compression_bytes", 0
            )
            scored = execute(
                "score", ["analyze", *argv], score_identity, cwd=ready, space=output_estimate
            )
            score_receipt = journal.stages[-1]["receipt_sha256"]
            if until == "score":
                return finish("complete")
            run = read_json(scored / "run_summary.json")

            def downstream(extra=None):
                journal.check_output(scored)
                return {"score_receipt": score_receipt, "environment": runtime, **(extra or {})}

            # Candidate planning can also be used on a pair-only archive. When the source
            # fitted triples, export its exact checksummed plan including exclusions/scope.
            select_args = (
                []
                if "candidate_manifest.json" in run["artifact_sha256"]
                else [
                    "--mode",
                    "uniform",
                    "--seed",
                    str(settings.get("seed", 0)),
                    "--maximum",
                    str(settings.get("max_triplets", 1000)),
                ]
            )
            execute(
                "select",
                ["select", "--run", scored, *select_args],
                lambda: downstream(
                    {"seed": settings.get("seed", 0), "maximum": settings.get("max_triplets", 1000)}
                ),
            )
            if until == "select":
                return finish("complete")
            if null_config:
                from triplet_coevolution.validation._null_checkpoint import DATABASE
                from triplet_coevolution.validation.null_policy import NullPolicy

                policy = NullPolicy.from_config(null_config)
                null_inputs = lambda: downstream(
                    {"policy": dependencies("null_policy", null_config)}
                )
                checkpoint = (policy.checkpoint or root / "checkpoints") / digest(null_inputs())
                primary = (
                    checkpoint / "scenario-01"
                    if policy.tree and policy.tree.alternatives
                    else checkpoint
                )
                flags = [
                    "--checkpoint",
                    checkpoint,
                    "--resume" if (primary / DATABASE).exists() else "--fresh",
                ]
                if memory_mib is not None:
                    if memory_mib < 1:
                        raise ValueError(
                            "standalone calibration requires a memory budget of at least one MiB"
                        )
                    flags += ["--memory-mib", str(min(int(memory_mib), policy.memory_mib))]
                execute(
                    "calibrate",
                    [
                        "calibrate",
                        "--config",
                        null_config,
                        "--run",
                        scored,
                        "--workers",
                        str(workers),
                        "--progress",
                        *flags,
                    ],
                    null_inputs,
                )
            if until == "calibrate":
                return finish("complete")
            if labels:
                checks = {
                    k: v
                    for k, v in cfg.get("analysis", {}).items()
                    if k in ("benchmark_k", "benchmark_bootstrap", "seed")
                }
                benchmark_opts = {
                    "top_k": checks.get("benchmark_k"),
                    "bootstrap": checks.get("benchmark_bootstrap"),
                    "seed": checks.get("seed"),
                }
                execute(
                    "benchmark",
                    [
                        "benchmark",
                        "--scores",
                        scored / run["outputs"]["score_manifest"],
                        "--labels",
                        labels,
                        *tokens(benchmark_opts),
                    ],
                    lambda: downstream(
                        {"labels": str(labels), "file": file_id(labels), "options": checks}
                    ),
                )
            if until == "benchmark":
                return finish("complete")
            execute("report", ["report", "--run", scored], lambda: downstream())
            return finish("complete")
        except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
            # A full volume can also prevent a failure log; no unpublished attempt is reusable.
            try:
                journal.event("run_failed", error=str(exc))
                finish("failed", str(exc))
            except OSError:
                pass
            raise
