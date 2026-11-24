"""Calibration from a native analysis: stable coordinates, refits and explicit test families."""

import csv
import hashlib
import html
import json
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Annotated

import numpy as np
import typer

from triplet_coevolution.inference.scores import get_score
from triplet_coevolution.io.score_store import file_digest
from triplet_coevolution.provenance import file_hashes
from triplet_coevolution.score_views import write_json
from triplet_coevolution.validation.calibration_models import CalibrationModel
from triplet_coevolution.validation.null_archive import RefitData, read_json
from triplet_coevolution.validation.null_policy import NullPolicy, adjusted, mc_interval, resolution
from triplet_coevolution.validation.null_runner import run_null

QUESTIONS = {
    "whole_component": "Does the association exceed random matching of intact components?",
    "within_stratum": "Does the association exceed random matching within the supplied strata?",
    "preserve_ab": "Does the statistic exceed random assignment of C to intact AB rows?",
    "preserve_ac": "Does the statistic exceed random assignment of B to intact AC rows?",
    "preserve_bc": "Does the statistic exceed random assignment of A to intact BC rows?",
    "m2_bootstrap": "Does the local statistic exceed iid samples from its fitted three-site pairwise maximum-entropy law?",
    "pair_panel": "Does the local statistic exceed iid samples from the pair-only law on the entire retained conditioning panel?",
}


def _write_results(directory, model, run):
    p = model.policy
    metrics = [
        "status",
        "observed",
        "finite_exceedances",
        "valid_draws",
        "invalid_draws",
        "conservative_exceedances",
        "p",
        "bh_q",
        "by_q",
        "mc95_lower",
        "mc95_upper",
        "max_t_p",
        "max_t_exceedances",
        "max_t_invalid_draws",
    ]
    coords = [f"{c}_{k}" for c in "abc" for k in ("reference_id", "original_col", "reference_pos")]
    families = {}
    with (directory / "calibration.tsv").open("w", encoding="utf8", newline="") as h:
        writer = csv.DictWriter(
            h,
            fieldnames=[
                "family",
                "statistic",
                "direction",
                "interface",
                "candidate_id",
                *coords,
                *metrics,
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for name, result in run.families.items():
            values = model.observed[name]
            bh, by = adjusted(result.p)
            info = resolution(len(values), p.draws, p.alpha)
            info["planned_resolution"] = resolution(
                len(values), p.planned_draws or p.draws, p.alpha
            )
            info.update(
                finite_observed=int(np.isfinite(values).sum()),
                invalid_refits=int(result.invalid_draws.sum()),
                max_t_invalid_draws=result.max_t_invalid_draws,
            )
            families[name] = info
            for i, (row, x) in enumerate(zip(model.coordinate_rows(name), values, strict=True)):
                ok = np.isfinite(x)
                hit, invalid = int(result.exceedances[i]), int(result.invalid_draws[i])
                low, high = mc_interval(hit, invalid, p.draws) if ok else (None, None)
                status = "tested" if ok else "ineligible_observed"
                if ok and invalid:
                    status = "conservative_invalid_refits"
                writer.writerow(
                    {
                        **row,
                        "family": name,
                        "direction": get_score(row["statistic"]).direction,
                        "status": status,
                        "observed": float(x) if ok else None,
                        "finite_exceedances": hit,
                        "valid_draws": int(result.valid_draws[i]),
                        "invalid_draws": invalid,
                        "conservative_exceedances": int(result.conservative_exceedances[i]),
                        "p": float(result.p[i]),
                        "bh_q": float(bh[i]),
                        "by_q": float(by[i]),
                        "mc95_lower": low,
                        "mc95_upper": high,
                        "max_t_p": float(result.max_t_p[i]) if p.max_t else None,
                        "max_t_exceedances": int(result.max_t_exceedances[i]) if p.max_t else None,
                        "max_t_invalid_draws": result.max_t_invalid_draws,
                    }
                )
    return families


def _html(info):
    cells = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{f['tests']}</td><td>{f['finite_observed']}</td><td>{f['minimum_p']:.5g}</td><td>{f['draws_for_bh_rank_one']}</td><td>{f['invalid_refits']}</td></tr>"
        for name, f in info["families"].items()
    )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Null calibration</title>
<style>body{{max-width:70rem;margin:3rem auto;padding:0 1rem;font:16px/1.5 system-ui;color:#1d2935}}table{{border-collapse:collapse}}td,th{{padding:.5rem;border:1px solid #b9c4ce;text-align:left}}</style>
<h1>Null calibration</h1><p>{html.escape(info["question"])}</p><p>{info["runner"]["status"]}. Completed {info["runner"]["completed_draws"]} of {info["runner"]["planned_draws"]} planned draws. {html.escape(info["sampling"])}</p>
<table><thead><tr><th>Test family</th><th>Hypotheses</th><th>Finite observed</th><th>Minimum p</th><th>Draws for first-rank BH</th><th>Invalid refits</th></tr></thead><tbody>{cells}</tbody></table>
<p>Monte Carlo intervals quantify simulation uncertainty. Each row reports its observed score and units are defined in the manifest. Unresolved refits count against rejection.</p>
<p><a href="calibration.tsv">All hypotheses and corrections</a> · <a href="calibration.json">Design, exchangeability and replay information</a></p></html>"""


def _check_checkpoint(policy, paths):
    cp = policy.checkpoint
    if cp and any(
        cp.is_relative_to(Path(path).resolve()) or Path(path).resolve().is_relative_to(cp)
        for path in paths
    ):
        raise ValueError(
            "checkpoint must be separate from the source run and published output directories"
        )


def _write_calibration(
    directory, source, policy, *, summary=None, live=None, inputs=None, progress=None
):
    """Fit and export one declared calibration. Caller stages output atomically."""
    directory, source = Path(directory), Path(source)
    _check_checkpoint(policy, [source, directory])
    data = RefitData.load(source, summary, inputs=inputs, live=live)
    model = CalibrationModel(data, policy)
    definitions = {
        s.name: s.metadata(apc_universe=model.apc, threebody_model=model.method)
        for s in model.specs
    }
    scientific_policy = {
        k: v
        for k, v in asdict(policy).items()
        if k
        not in {
            "checkpoint",
            "resume",
            "workers",
            "draws",
            "planned_draws",
            "memory_mib",
            "checkpoint_mib",
        }
    }
    prov = data.store.provenance
    identity = {
        "policy": scientific_policy,
        "score_manifest_sha256": file_digest(source / data.summary["outputs"]["score_manifest"]),
        "inputs": prov["inputs"],
        "row_ids": list(data.weights.observation_ids),
        "weight_table_sha256": prov["weight_table_sha256"],
        "definitions": definitions,
        "coordinates": {m.component: list(m.original_positions) for m in data.store.maps},
        "candidate_manifest": model.candidate_scope,
        "diagnostics_sha256": hashlib.sha256(
            json.dumps(
                model.diagnostics, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
    }
    if model.support:
        identity["frozen_support_sha256"] = file_digest(
            data.artifact(data.summary["advanced"]["local_fit"]["frozen_support"])
        )
        identity["local_primary_setting"] = asdict(model.options)
    directions = {g: get_score(members[0][0]).direction for g, members in model.groups.items()}
    directory.mkdir(parents=True, exist_ok=True)
    lock = Lock()
    with (directory / "progress.jsonl").open("w", encoding="utf8") as log:

        def report(event):
            with lock:
                log.write(json.dumps(event, separators=(",", ":")) + "\n")
                log.flush()
                if progress:
                    progress(event)

        job = run_null(
            model.observed,
            model.evaluate,
            draws=policy.draws,
            planned_draws=policy.planned_draws,
            seed=policy.seed,
            null_name=policy.null,
            identity=identity,
            directions=directions,
            workers=policy.workers,
            checkpoint=policy.checkpoint,
            resume=policy.resume,
            max_t=policy.max_t,
            joint_draws=policy.null != "m2_bootstrap",
            memory_budget_bytes=model.runner_budget,
            checkpoint_budget_bytes=policy.checkpoint_mib * 1024**2,
            progress=report,
        )
    families = _write_results(directory, model, job)
    info = {
        "schema_version": 1,
        "question": QUESTIONS[policy.null],
        "policy": scientific_policy,
        "runner": job.manifest,
        "workspace": model.workspace,
        "families": families,
        "score_definitions": definitions,
        "diagnostics": model.diagnostics,
        "sampling": "IID model draws condition on the recorded missingness and exogenous weight design."
        if policy.null in {"pair_panel", "m2_bootstrap"}
        else "Component rows move intact. Label exchangeability is assumed within the declared groups.",
        "test_design": "Original row cohort, column masks, candidate IDs and statistic definitions are fixed. Identity-derived weights are recomputed on each rematching. Fixed observation weights retain their declared scale.",
        "multiplicity": "Each statistic forms its own family, optionally split by interface. BH assumes independence or suitable positive dependence. BY controls FDR under arbitrary dependence. Joint single-step maxT describes the joint null; strong control additionally requires subset pivotality.",
        "monte_carlo": "Inclusive one-sided tail. (1 + finite exceedances + unresolved draws)/(1 + requested draws). Ineligible observations stay in family with p=1. Wilson 95% bounds bracket both assignments of unresolved draws. Interim runs are exploratory.",
        "provenance": identity,
        "outputs": {
            "tests": "calibration.tsv",
            "report": "report.html",
            "progress": "progress.jsonl",
        },
    }
    (directory / "report.html").write_text(_html(info), encoding="utf8")
    info["artifact_sha256"] = file_hashes(directory, exclude=("calibration.json",))
    write_json(directory / "calibration.json", info)
    return info


def calibrate_run(source, out, policy, *, inputs=None, progress=None):
    """Calibrate a native score archive and atomically publish a new result directory.

    Inputs can relocate A/B/C/matches by their original digests. A progress callback
    receives dictionaries of replicate IDs, counts and times. An interrupted job
    retains durable draws in the independently specified checkpoint.
    """
    from triplet_coevolution.cli import _stage_run

    source, out = Path(source).resolve(), Path(out).resolve()
    policy = policy if isinstance(policy, NullPolicy) else NullPolicy.from_config(policy)
    if source.is_relative_to(out) or out.is_relative_to(source):
        raise ValueError("calibration output and source analysis must be separate")
    _check_checkpoint(policy, [source, out])
    with _stage_run(out) as stage:
        return _write_calibration(stage, source, policy, inputs=inputs, progress=progress)


def calibrate(
    config: Annotated[
        Path, typer.Option("--config", help="Versioned named null, statistics and simulation plan.")
    ],
    out: Annotated[Path, typer.Option("--out", help="New or empty calibration directory.")],
    run: Annotated[
        Path | None, typer.Option("--run", help="Native analysis directory with row-level replay.")
    ] = None,
    scores: Annotated[
        Path | None,
        typer.Option(
            "--scores",
            help="Native scores/ directory or score_manifest.json with its companion analysis.",
        ),
    ] = None,
    inputs: Annotated[
        Path | None,
        typer.Option(
            "--inputs", help="Optional JSON file relocating checksummed source paths A/B/C/matches."
        ),
    ] = None,
    workers: Annotated[int | None, typer.Option("--workers")] = None,
    draws: Annotated[
        int | None, typer.Option("--draws", help="Execute a prefix of the fixed plan.")
    ] = None,
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
    resume: Annotated[bool | None, typer.Option("--resume/--fresh")] = None,
    progress: Annotated[
        bool, typer.Option("--progress", help="Emit JSON-line progress on stderr.")
    ] = False,
):
    """Calibrate chosen archived statistics using the original matched observations."""
    try:
        if (run is None) == (scores is None):
            raise ValueError("choose exactly one of --run and --scores")
        if scores is not None:
            manifest = scores / "score_manifest.json" if scores.is_dir() else scores
            if (
                manifest.name != "score_manifest.json"
                or manifest.parent.name != "scores"
                or not manifest.is_file()
            ):
                raise ValueError("--scores requires the native scores/score_manifest.json")
            run = manifest.parent.parent
        policy = NullPolicy.from_config(
            config, workers=workers, draws=draws, checkpoint=checkpoint, resume=resume
        )
        relocated = None
        if inputs:
            paths = read_json(inputs)
            if not isinstance(paths, dict):
                raise TypeError("input relocation must be a JSON object")
            relocated = {k: inputs.resolve().parent / v for k, v in paths.items()}
        info = calibrate_run(
            run,
            out,
            policy,
            inputs=relocated,
            progress=(lambda e: typer.echo(json.dumps(e), err=True)) if progress else None,
        )
    except (ValueError, KeyError, TypeError, OSError, csv.Error) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{info['runner']['status']}: {out / 'calibration.json'}")
