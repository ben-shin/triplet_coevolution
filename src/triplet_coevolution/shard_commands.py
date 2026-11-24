"""Native calibration stage entry points for local processes and scheduler job arrays."""

import csv
import json
import os
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer

from triplet_coevolution.calibration import _calibration_design, _publish_calibration
from triplet_coevolution.io.score_store import file_digest
from triplet_coevolution.validation._null_checkpoint import canonical, digest
from triplet_coevolution.validation.null_archive import RefitData
from triplet_coevolution.validation.null_policy import NullPolicy
from triplet_coevolution.validation.shards import (
    _atomic_json,
    _load_json,
    merge_shards,
    prepare_shard_plan,
    run_shard,
    shard_ids,
    shard_progress,
)

SCHEMA = "triplet.native_shard_design.v1"
DEFAULT_META_LIMIT = 512 * 1024**2
Run = Annotated[Path, typer.Option("--run", help="Native analysis with verified row-level replay.")]
Plan = Annotated[Path, typer.Option("--plan", help="shard-plan directory or shard_plan.json.")]
Out = Annotated[Path, typer.Option("--out")]
Config = Annotated[Path, typer.Option("--config")]
Inputs = Annotated[
    Path | None,
    typer.Option("--inputs", help="Relocate source A/B/C/matches by their original checksums."),
]
ShardDirs = Annotated[
    list[Path] | None,
    typer.Option(
        "--shard", help="Completed shard directory; repeat to override the planned paths."
    ),
]


def _command(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except (ValueError, KeyError, TypeError, OSError, csv.Error) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from exc

    return wrapped


def _paths(src):
    if src is None:
        return None
    table = _load_json(src, DEFAULT_META_LIMIT)
    if not isinstance(table, dict):
        raise TypeError("input relocation must be a JSON object")
    return {k: src.resolve().parent / v for k, v in table.items()}


def _born(source, config, inputs, workers=None):
    policy = NullPolicy.from_config(config, workers=workers)
    if policy.draws != (policy.planned_draws or policy.draws):
        raise ValueError("distributed calibration requires draws equal to the final planned_draws")
    if policy.resume or policy.checkpoint:
        raise ValueError(
            "shards own separate durable journals; remove the serial checkpoint policy"
        )
    if policy.tree and policy.tree.alternatives:
        raise ValueError("prepare one shard plan for each declared tree scenario")
    data = RefitData.load(source, inputs=_paths(inputs))
    model, definitions, policy_data, identity, directions = _calibration_design(
        source, policy, data
    )
    plan = prepare_shard_plan(
        model.observed,
        model.evaluate,
        draws=policy.draws,
        seed=policy.seed,
        null_name=policy.null,
        identity=identity,
        directions=directions,
        max_t=policy.max_t,
        joint_draws=policy.null != "m2_bootstrap",
        memory_budget_bytes=model.runner_budget,
    )
    return plan, model, definitions, policy_data, identity


def _relative(path, root):
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), root.resolve())
    except ValueError:  # Windows volumes may differ.
        return str(path.resolve())


def _separate(output, source):
    output, source = output.resolve(), source.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("destination and protected input must be disjoint")


def load_native_plan(path):
    """Load and authenticate the declarative design without executing callbacks or refits."""
    path = Path(path).resolve()
    if path.is_dir():
        path /= "shard_plan.json"
    obj = _load_json(path, DEFAULT_META_LIMIT)
    if not isinstance(obj, dict) or obj.get("schema") != SCHEMA:
        raise ValueError("unsupported native shard plan")
    if obj.get("document_sha256") != digest(
        canonical({k: v for k, v in obj.items() if k != "document_sha256"})
    ):
        raise ValueError("native shard plan document checksum mismatch")
    return path.parent, obj


def _rebuild(root, spec, workers=None, inputs=None):
    source = (root / spec["run"]).resolve()
    cfg = root / spec["config"]
    if (
        file_digest(cfg) != spec["config_sha256"]
        or file_digest(source / "run_summary.json") != spec["run_summary_sha256"]
    ):
        raise ValueError("source run or null config checksum differs from the persisted design")
    relocation = inputs or (root / spec["inputs"] if spec["inputs"] else None)
    rebuilt = _born(source, cfg, relocation, workers)
    if rebuilt[0].manifest != spec["numerical"]:
        raise ValueError(
            "native shard design mismatch in data, candidates, masks, model, code or numerical runtime"
        )
    return rebuilt


def shard_directory(root, i):
    return root / "shards" / f"shard-{i:05d}"


def _shard_dirs(root, spec, overrides):
    return (
        overrides
        if overrides is not None
        else (shard_directory(root, i) for i in range(spec["shards"]))
    )


@_command
def shard_plan(
    run: Run,
    config: Config,
    out: Out,
    shards: Annotated[int, typer.Option("--shards")],
    inputs: Inputs = None,
):
    """Prepare a checksummed final calibration design and balanced logical replicate assignments."""
    from triplet_coevolution.cli import _stage_run

    run, out = run.resolve(), out.resolve()
    _separate(out, run)
    cfg_hash = file_digest(config)
    run_hash = file_digest(run / "run_summary.json")
    plan, model, *_ = _born(run, config, inputs)
    shard_ids(plan.draws, 0, shards)
    if file_digest(config) != cfg_hash or file_digest(run / "run_summary.json") != run_hash:
        raise ValueError("run or policy changed while preparing the shard design")
    obj = {
        "schema": SCHEMA,
        "numerical": plan.manifest,
        "shards": shards,
        "assignment": "balanced_contiguous_zero_based",
        "run": _relative(run, out),
        "config": _relative(config, out),
        "inputs": _relative(inputs, out),
        "config_sha256": cfg_hash,
        "run_summary_sha256": run_hash,
        "workspace": model.workspace,
        "process_memory_mib": model.policy.memory_mib,
        "checkpoint_mib": model.policy.checkpoint_mib,
    }
    obj["document_sha256"] = digest(canonical(obj))
    with _stage_run(out) as stage:
        _atomic_json(stage / "shard_plan.json", obj)
    typer.echo(str(out / "shard_plan.json"))


@_command
def null_shard(
    plan: Plan,
    shard_id: Annotated[int | None, typer.Option("--shard-id")] = None,
    ids: Annotated[
        Path | None, typer.Option("--ids", help="JSON array of explicit global replicate IDs.")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out")] = None,
    inputs: Inputs = None,
    workers: Annotated[int | None, typer.Option("--workers")] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    progress: Annotated[bool, typer.Option("--progress")] = False,
):
    """Calculate one disjoint subset. A completed receipt contains numerical evidence for merging."""
    root, spec = load_native_plan(plan)
    if (shard_id is None) == (ids is None) or ids is not None and out is None:
        raise ValueError("choose --shard-id or both --ids and --out")
    numerical, model, *_ = _rebuild(root, spec, workers, inputs)
    if ids is None:
        selected = shard_ids(numerical.draws, shard_id, spec["shards"])
        out = out or shard_directory(root, shard_id)
    else:
        selected = _load_json(ids, min(DEFAULT_META_LIMIT, model.runner_budget))
        if not isinstance(selected, list):
            raise ValueError("--ids requires a JSON array")
    _separate(out, root / spec["run"])
    if root.is_relative_to(out.resolve()):
        raise ValueError("shard output cannot contain its plan")
    receipt = run_shard(
        numerical,
        model.evaluate,
        out,
        replicate_ids=selected,
        workers=model.policy.workers,
        resume=resume,
        memory_budget_bytes=model.runner_budget,
        checkpoint_budget_bytes=model.policy.checkpoint_mib * 1024**2,
        progress=(lambda event: typer.echo(json.dumps(event), err=True)) if progress else None,
    )
    typer.echo(
        json.dumps(
            {
                "receipt": str(out / "receipt.json"),
                "completed_draws": receipt["completed_draws"],
                "receipt_sha256": receipt["receipt_sha256"],
            }
        )
    )


@_command
def merge_null(
    plan: Plan,
    out: Out,
    shard: ShardDirs = None,
    inputs: Inputs = None,
    progress: Annotated[bool, typer.Option("--progress")] = False,
):
    """Verify every assigned draw, then publish the complete native calibration, bounds and FDR."""
    from triplet_coevolution.cli import _stage_run

    root, spec = load_native_plan(plan)
    _separate(out, root / spec["run"])
    if root.is_relative_to(out.resolve()):
        raise ValueError("merge output cannot contain its plan")
    numerical, model, definitions, scientific, identity = _rebuild(root, spec, 1, inputs)
    dirs = _shard_dirs(root, spec, shard)

    def verified_paths():
        for p in dirs:
            _separate(out, p)
            yield p

    with (
        _stage_run(out.resolve()) as stage,
        (stage / "progress.jsonl").open("w", encoding="utf8") as f,
    ):

        def report(event):
            encoded = json.dumps(event, separators=(",", ":"))
            f.write(encoded + "\n")
            f.flush()
            if progress:
                typer.echo(encoded, err=True)

        job = merge_shards(
            numerical, verified_paths(), memory_budget_bytes=model.runner_budget, progress=report
        )
        _publish_calibration(stage, model, job, definitions, scientific, identity)
    typer.echo(str(out / "calibration.json"))


@_command
def shard_status(plan: Plan, shard: ShardDirs = None):
    """Print provisional per-shard and aggregate counts as one JSON object."""
    root, spec = load_native_plan(plan)
    typer.echo(json.dumps(shard_progress(spec["numerical"], _shard_dirs(root, spec, shard))))
