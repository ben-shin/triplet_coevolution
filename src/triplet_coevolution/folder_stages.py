"""Resume automatic folder workflows while preserving their matching and scoring decisions."""

import os
import shutil
from pathlib import Path
from uuid import uuid4

from triplet_coevolution.diagnostics import THREAD_VARS
from triplet_coevolution.family_config import family_config
from triplet_coevolution.local_pipeline import _run, dependencies
from triplet_coevolution.operations import (
    Journal,
    file_id,
    pipeline_workspace,
    put_json,
    read_json,
    stage,
)
from triplet_coevolution.provenance import package_code_digest, software_identity


def worker(action, job_file, flag, output):
    """Private child operation. The public input is always a typed folder configuration."""
    from triplet_coevolution.automatic_run import finish_run
    from triplet_coevolution.cli import _stage_run
    from triplet_coevolution.family_workflow import _prepare

    if flag != "--out":
        raise ValueError("missing output directory")
    job = read_json(job_file)
    folder = Path(job["input"])
    output = Path(output)
    if action == "prepare":
        _prepare(
            None,
            folder,
            output,
            False,
            job["online"],
            None,
            automatic=True,
            cache_directory=Path(job["cache"]),
            defer_scoring=True,
        )
    elif action == "finish":
        prior = Path(job["prepared"])
        cfg, _, _ = family_config(None, folder, uniprot=job["online"], automatic=True)
        with _stage_run(output) as tmp:
            shutil.copytree(prior, tmp, dirs_exist_ok=True)
            summary = read_json(tmp / "workflow_summary.json")
            kwargs = summary.pop("deferred_analysis_inputs")
            for key in ("matches", "a", "b", "c"):
                if key in kwargs:
                    kwargs[key] = tmp / kwargs[key]
            summary["analysis_requested"] = True
            finish_run(
                tmp,
                Path(job["public"]),
                summary,
                {**kwargs, **cfg["analysis"]},
                cfg["analysis"],
                cfg["analysis_limits"],
                job["plots"],
            )
            put_json(tmp / "workflow_summary.json", summary)
    else:
        raise ValueError("unknown family stage")


def run_folder_stages(directory, work_dir, online=None, cache_dir=None, plots=None, emit=None):
    folder = directory.resolve()
    root = work_dir.resolve()
    cache = (cache_dir or root / "provider-cache").resolve()
    if root.is_relative_to(folder) or cache.is_relative_to(folder) or folder.is_relative_to(root):
        raise ValueError(
            "keep the input project separate from the workspace and writable provider cache"
        )
    if cache_dir and (cache.is_relative_to(root) or root.is_relative_to(cache)):
        raise ValueError("choose a separate external cache or omit --cache-dir")
    env = {
        **os.environ,
        **dict.fromkeys(THREAD_VARS, "1"),
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "PYTHONUTF8": "1",
    }
    runtime = software_identity()
    runtime.pop("git", None)
    with pipeline_workspace(root):
        journal = Journal(root, emit)
        cfg, _, _ = family_config(None, folder, uniprot=online, automatic=True)
        allow = cfg["uniprot"]
        request = uuid4().hex if allow else None
        journal.event("run_started", mode="family", online=allow)

        def prepare_sources():
            if package_code_digest() != runtime["code_sha256"]:
                raise ValueError("installed source changed during pipeline")
            active, inputs, order = family_config(None, folder, uniprot=online, automatic=True)
            files = {
                k: {"path": str(p), **file_id(p)}
                for k, p in inputs.items()
                if not k.startswith("analysis.")
            }
            snapshots = {}
            if not allow:
                stable = cache
                generation = active["providers"]["cache_generation"]
                if generation != "default":
                    stable = cache / generation
                for name in ("uniprot.sqlite", "nucleotide.sqlite"):
                    p = stable / name
                    if any(
                        Path(str(p) + s).exists() and Path(str(p) + s).stat().st_size
                        for s in ("-wal", "-journal")
                    ):
                        raise ValueError(
                            "offline workspace reuse requires closed provider snapshots"
                        )
                    if p.is_file():
                        snapshots[name] = file_id(p)
            return {
                "files": files,
                "order": order,
                "input": str(folder),
                "runtime": runtime,
                "online": allow,
                "online_request": request,
                "cache": str(cache),
                "closed_snapshots": snapshots,
            }

        def child(name, action, output, extra):
            jobfile = journal.root / (name + ".json")
            put_json(jobfile, {"input": str(folder), "online": allow, "cache": str(cache), **extra})
            _run(["_folder-stage", action, jobfile], output, journal, env, name)

        try:
            prior = stage(
                root,
                journal,
                "family_prepare",
                prepare_sources,
                lambda p: child("family_prepare", "prepare", p, {}),
            )
            prep_hash = journal.stages[-1]["receipt_sha256"]

            def finish_sources():
                journal.check_output(prior)
                if package_code_digest() != runtime["code_sha256"]:
                    raise ValueError("installed source changed during pipeline")
                current, _, _ = family_config(None, folder, uniprot=online, automatic=True)
                cfg_files = {}
                for v in current["analysis"].values():
                    if isinstance(v, Path):
                        cfg_files.update(dependencies(v))
                return {
                    "preparation": prep_hash,
                    "runtime": runtime,
                    "analysis": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in current["analysis"].items()
                    },
                    "limits": current["analysis_limits"],
                    "policy_inputs": cfg_files,
                    "plots": plots,
                }

            result = stage(
                root,
                journal,
                "family_score",
                finish_sources,
                lambda p, public: child(
                    "family_score",
                    "finish",
                    p,
                    {"prepared": str(prior), "plots": plots, "public": str(public)},
                ),
                public_location=True,
            )
            summary = read_json(result / "workflow_summary.json")
            receipt = journal.finish(
                summary["status"],
                scientific_exit_code=summary["exit_code"],
                report=str(result / "report.html"),
            )
            return summary["exit_code"], receipt, result / "report.html"
        except (ValueError, OSError, TypeError) as exc:
            try:
                journal.finish("failed", reason=str(exc))
            except OSError:
                pass
            raise
