"""Resource estimates for a native analysis using its actual synchronized, masked coordinates."""

import csv
import math
import os
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np

from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.potts import MAX_DUAL_ELEMENTS, MAX_SOLVE_DIMENSION
from triplet_coevolution.io.alignments import FILE_FORMATS
from triplet_coevolution.msa.support import MissingRowSupport


class AnalysisBudgetError(ValueError):
    """An explicit allocation budget or a model limit prevents the planned computation."""


class QCBlocked(MissingRowSupport):
    """Carry a completed diagnostic inspection to an outer workflow for publication."""

    def __init__(self, audit, inputs):
        super().__init__(" ".join(audit.blockers))
        self.audit = audit
        self.inputs = inputs


def input_cells(paths):
    """Count raw sequence symbols. For matched working sets use the public plan command."""
    import gzip

    counts = {}
    for name in "ABC":
        path = Path(paths[name])
        name_path = path.with_suffix("") if path.suffix.lower() == ".gz" else path
        form = FILE_FORMATS.get(name_path.suffix.lower())
        if form is None:
            raise ValueError(f"unsupported alignment format for {path}")
        total = 0
        with path.open("rb") as binary:
            opener = gzip.open if binary.read(2) == b"\x1f\x8b" else open
        display = False
        with opener(path, "rt", encoding="utf-8-sig") as stream:
            for raw in stream:
                line = raw.strip()
                if form in ("fasta", "a3m"):
                    if line.startswith(">"):
                        identifier = line[1:].split(maxsplit=1)
                        display = (
                            form == "a3m"
                            and bool(identifier)
                            and identifier[0]
                            in {
                                "ss_pred",
                                "ss_conf",
                                "ss_dssp",
                                "sa_dssp",
                                "ss_cons",
                            }
                        )
                    if line and not display and not line.startswith((">", ";", "#", "!")):
                        total += len("".join(line.split()))
                elif line and not line.startswith("#") and line != "//":
                    tokens = line.split()
                    if len(tokens) == 2:
                        total += len(tokens[1])
        counts[name] = total
    return counts


def match_metadata(table, indices, column, *, numeric=False):
    if column is None:
        return None
    if column not in table.extra_columns:
        raise ValueError(f"unknown match-table column: {column}")
    values = {}
    for i in indices:
        key = table.matches[i].observation_id
        v = table.annotations[i][column]
        if not v.strip():
            raise ValueError(f"empty {column} for observation {key}")
        if numeric:
            v = float(v)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"invalid nonnegative weight for observation {key}")
        values[key] = v
    if numeric and values and not any(values.values()):
        raise ValueError("supplied weights must include a positive active observation")
    return values


def counts_by_site(msa, positive=None):
    counts = {}
    complete = np.ones(msa.n_observations, dtype=bool) if positive is None else positive.copy()
    for c in msa.components:
        q = len(CANONICAL[c.molecule])
        matrix = c.matrix
        complete &= np.all((matrix >= 0) & (matrix < q), axis=1)
        counted = matrix if positive is None else matrix[positive]
        values = np.zeros(c.n_sites, dtype=int)
        for k in range(q):
            values += np.any(counted == k, axis=0)
        counts[c.name] = values
    return counts, complete


def estimate_model(
    msa,
    maps,
    *,
    dca,
    threebody,
    threebody_model,
    require_reference,
    max_triplets,
    triplet_candidates,
    permutations,
    min_valid,
    min_valid_weight,
    all_pairs,
    max_pairs,
    weight_policy,
    weight_component,
    supplied_weights,
    memory_budget_bytes,
    seed,
    identity_threshold,
    min_overlap,
    candidate_mode=None,
    candidate_policy=None,
    local_policy=None,
    dca_design=None,
):
    from triplet_coevolution.inference.weights import weighting_preflight

    n = msa.n_observations
    supplied = (
        tuple(supplied_weights[m.observation_id] for m in msa.matches) if supplied_weights else None
    )
    positive = np.array(supplied) > 0 if supplied else None
    states, complete = counts_by_site(msa, positive)
    widths = {m.component: m.n_sites for m in maps}
    numbered = {m.component: sum(p is not None for p in m.reference_positions) for m in maps}

    def pair(widths):
        return {a + b: widths[a] * widths[b] for a, b in combinations("ABC", 2)}

    raw = pair(widths)
    mapped = pair(numbered)
    eligible = mapped if require_reference else raw
    pairs = sum(raw.values())
    dimensions = sum(int(v[v > 1].sum()) for v in states.values())
    variable = sum(int((v > 1).sum()) for v in states.values())
    try:
        wplan = weighting_preflight(
            msa,
            policy=weight_policy,
            component=weight_component,
            memory_budget_bytes=memory_budget_bytes,
            reuse=bool(permutations),
        )
    except ValueError as exc:
        if memory_budget_bytes is None or "memory budget" not in str(exc):
            raise
        wplan = weighting_preflight(
            msa, policy=weight_policy, component=weight_component, reuse=bool(permutations)
        )
        wplan["budget_blocker"] = str(exc)
    # Estimate simultaneous large numeric buffers. Python objects, BLAS workspaces and
    # full benchmark JSON are outside this workspace estimate.
    peak_weights = wplan["peak_bytes"]
    array_bytes = 32 * pairs  # MI raw, APC, n_valid and valid_weight.
    peak_pair = 2 * array_bytes + min(65536, max(1, n) * max(widths.values())) * 64
    eligible_triples = math.prod((numbered if require_reference else widths).values())
    candidates = min(max_triplets, eligible_triples) if threebody else 0
    blockers = (
        [{"method": "weighting", "reason": wplan["budget_blocker"]}]
        if "budget_blocker" in wplan
        else []
    )
    warnings = []
    if (
        not math.isfinite(identity_threshold)
        or not 0 <= identity_threshold <= 1
        or not math.isfinite(min_overlap)
        or not 0 <= min_overlap <= 1
    ):
        raise ValueError("weighting identity_threshold and min_overlap must be between 0 and 1")
    if n < min_valid:
        warnings.append(
            "Fewer retained observations than the required pair/triple count threshold."
        )
    if supplied and not math.isfinite(sum(supplied)):
        raise ValueError("total supplied weight exceeds finite floating-point range")
    if supplied and sum(supplied) < min_valid_weight:
        warnings.append("Total supplied positive weight falls below the minimum valid weight.")
    from triplet_coevolution.inference.candidates import select_candidates
    from triplet_coevolution.inference.local_options import (
        SCREENED,
        LocalSettings,
        candidate_options,
    )
    from triplet_coevolution.inference.scores import coordinate_axes

    mode, screen = candidate_options(
        candidate_mode, candidate_policy, triplet_candidates, max_triplets
    )
    if screen.get("pair_statistic", "mi_bits").startswith("dca") and not dca:
        raise ValueError("DCA candidate screening requires --dca")
    ids = {m.observation_id for m in msa.matches}
    if screen.get("training_ids") and not set(screen["training_ids"]) < ids:
        raise ValueError(
            "candidate training_ids must be a proper subset of the retained observations"
        )
    axes = coordinate_axes(maps, require_reference)
    verified = None
    if threebody and mode == "exhaustive" and candidates < eligible_triples:
        blockers.append(
            {
                "method": "threebody",
                "reason": f"exhaustive universe of {eligible_triples} exceeds max_triplets={max_triplets}",
            }
        )
    if threebody and mode not in {"prior", "explicit"} and candidates > 100_000:
        blockers.append(
            {
                "method": "threebody",
                "reason": "ABC candidate count exceeds 100000. Reduce --max-triplets or the site panel.",
            }
        )
    if (
        threebody
        and mode in SCREENED
        and min(eligible_triples, screen.get("screen_budget", 100_000)) > 1_000_000
    ):
        blockers.append(
            {
                "method": "threebody",
                "reason": "ABC screen exceeds 1000000 probes. Reduce screen_budget.",
            }
        )
    if triplet_candidates is not None:
        file_mode = "prior" if mode == "hybrid" else mode
        file_opts = {key: screen[key] for key in ("prior_source", "file_origin") if key in screen}
        file_set = select_candidates(
            maps,
            axes,
            mode=file_mode,
            maximum=screen.get("screen_budget", 100_000) if mode == "hybrid" else max_triplets,
            candidate_file=triplet_candidates,
            seed=seed,
            **file_opts,
        )
        if mode != "hybrid":
            verified = file_set.triples
            candidates = len(verified)
    local_details = None
    screened_probes = (
        min(eligible_triples, screen.get("screen_budget", 100_000))
        if threebody and mode in SCREENED
        else 0
    )
    settings_count = 1
    maxent_cells = 0
    if threebody_model == "maxent":
        local = LocalSettings.parse(local_policy)
        support = local.support(msa)
        zero_bins = local.zero_bins(maps, support, axes)
        if verified is not None and zero_bins.keys() - set(verified):
            raise ValueError("structural zero specified for an unselected ABC candidate")
        # A screen does not run during preflight. Its sites therefore use an upper bound.
        if verified is not None:
            maxent_cells = max((math.prod(support.shape_for(t)) for t in verified), default=0)
        else:
            maxent_cells = math.prod(
                max((max(sites[j]) + 1 for j in axis), default=0)
                for sites, axis in zip(support.maps, axes, strict=True)
            )
        settings_count = len(local.options)
        local_details = {
            "prior_settings": settings_count,
            "max_cells_per_candidate": maxent_cells,
            "cell_limit": local.options[0].max_cells,
            "ipf_iterations_per_setting_limit": local.options[0].max_iterations,
            "declared_structural_zeros": len(local.structural_zeros),
            "include_gaps": local.include_gaps,
            "frozen_state_mapping": "train_pooled" if local.min_state_count else "declared",
            "settings": [asdict(setting) for setting in local.options],
            "diagnostic_rows_upper": settings_count * candidates,
            "ipf_cell_updates_upper": 3
            * min(maxent_cells, local.options[0].max_cells)
            * local.options[0].max_iterations
            * settings_count
            * candidates,
        }
        if maxent_cells > local.options[0].max_cells:
            warnings.append(
                "Candidates exceeding the configured max_cells will have resource_limit status and undefined connected information."
            )
    choice = None
    dca_bytes = 0
    gram_flops = 0
    dca_plan = None
    if dca and dca_design is not None:
        try:
            dca_plan = dca_design.preflight(msa, positive)
            choice = dca_design.backend + "_" + dca_design.conditioning
            dca_bytes = dca_plan["estimated_numeric_bytes"]
            gram_flops = None
        except ValueError as exc:
            blockers.append({"method": "dca", "reason": str(exc)})
    elif dca:
        dual = n < dimensions and n <= MAX_SOLVE_DIMENSION and n * dimensions <= MAX_DUAL_ELEMENTS
        choice = "dual" if dual else "primal"
        if not dual and dimensions > MAX_SOLVE_DIMENSION:
            message = f"DCA panel has {n} observations and {dimensions} observed-state features; both solver limits are exceeded."
            if weight_policy == "cluster_representatives":
                warnings.append(message + " Positive representative rows may reduce feature count.")
            else:
                blockers.append({"method": "dca", "reason": message})
        if dual:
            gram_flops = 2 * n**2 * dimensions
            dca_bytes = 8 * (3 * n * dimensions + 3 * n * n)
        else:
            gram_flops = 2 * n * dimensions**2
            batch = max(1, min(n, 1024, 1_000_000 // max(1, dimensions)))
            dca_bytes = 8 * (4 * dimensions**2 + batch * dimensions)
    peak_dca = dca_bytes + 2 * array_bytes if dca else 0
    max_states = math.prod(max(1, int(states[c].max())) for c in "ABC")
    peak_triples = (candidates * 256 + max_states * 96) if threebody else 0
    if threebody and threebody_model == "local":
        from triplet_coevolution.inference.threebody import triple_row_workspace

        peak_triples += triple_row_workspace(n, tuple(widths.values()), candidates)
    if threebody:
        peak_triples += screened_probes * 80
        if local_details:
            peak_triples += (
                min(maxent_cells, local.options[0].max_cells) * 128
                + min(candidates, 32) * settings_count * 1024
                + candidates * 1024
            )
    global_details = None
    if threebody and threebody_model == "global":
        cn = int(complete.sum())
        cm, _ = counts_by_site(msa, complete)
        dims = [int(v) for c in "ABC" for v in cm[c] if v > 1]
        count = len(dims)
        # Exact field/pair size; triple candidates only add parameters on variable sites.
        pair_coeff = (sum(dims) ** 2 - sum(k * k for k in dims)) // 2
        lower = sum(dims) + pair_coeff
        lookups = cn * count * count
        from triplet_coevolution.inference.scores import coordinate_axes
        from triplet_coevolution.inference.threebody import sample_candidates

        axes = coordinate_axes(maps, require_reference)
        is_exact = True
        if (
            mode in {"uniform", "exhaustive"}
            and not triplet_candidates
            and candidates == eligible_triples
        ):
            # Product of the sums equals the sum of all triple tensor sizes.
            active = [
                [int(cm[c][j]) for j in axis if cm[c][j] > 1]
                for c, axis in zip("ABC", axes, strict=True)
            ]
            kterms = math.prod(len(items) for items in active)
            triple_params = math.prod(sum(items) for items in active)
        elif (
            verified is not None
            or mode in {"uniform", "exhaustive"}
            and candidates <= min(50_000, (memory_budget_bytes or 16_000_000) // 256)
        ):
            chosen = (
                verified if verified is not None else sample_candidates(axes, max_triplets, seed)[0]
            )
            sizes = (tuple(int(cm[c][j]) for c, j in zip("ABC", t, strict=True)) for t in chosen)
            kterms = triple_params = 0
            for size in sizes:
                if min(size) > 1:
                    kterms += 1
                    triple_params += math.prod(size)
        else:
            # Inspecting an enormous random sample would itself allocate the tensor-index list.
            is_exact = False
            kterms = candidates
            triple_params = candidates * max_states
        global_details = {
            "complete_rows": cn,
            "variable_sites_on_complete_rows": count,
            "field_pair_parameters": lower,
            "candidate_terms": kterms,
            "triple_parameters": triple_params,
            "lookup_upper_bound": lookups + 3 * cn * kterms,
            "candidate_counting": "exact" if is_exact else "upper_bound",
        }
        limits_exceeded = (
            count > 64
            or lower > 250_000
            or lookups > 2_000_000
            or is_exact
            and (lower + triple_params > 250_000 or lookups + 3 * cn * kterms > 2_000_000)
        )
        complete_weight_upper = (
            math.fsum(w for w, keep in zip(supplied, complete, strict=True) if keep)
            if supplied
            else cn
        )
        if cn >= min_valid and complete_weight_upper >= min_valid_weight and limits_exceeded:
            reason = f"Global Potts fit needs {count} variable sites, {lower + triple_params} coefficients and {lookups + 3 * cn * kterms} row lookups (limits 64, 250000 and 2000000). Reduce the analysis panel."
            if weight_policy == "cluster_representatives":
                warnings.append(
                    reason + " Positive representative rows are determined during weighting."
                )
            else:
                blockers.append({"method": "threebody", "reason": reason})
        peak_triples += 48 * (lower + triple_params) + 8 * global_details["lookup_upper_bound"]
    withhold = (
        threebody_model == "maxent"
        or mode in SCREENED
        or screen.get("file_origin") in {"same_data", "training"}
    )
    nulls = (pairs * (2 if dca else 1) + (0 if withhold else candidates)) if permutations else 0
    peak_null = (
        (2 * array_bytes if dca else array_bytes)
        + 96 * nulls
        + max(peak_weights, wplan["cache_bytes"] + max(peak_pair, peak_dca, peak_triples))
        if permutations
        else 0
    )
    peak = max(peak_weights, peak_pair, peak_dca, peak_triples + array_bytes, peak_null)
    if memory_budget_bytes is not None and peak > memory_budget_bytes:
        blockers.append(
            {
                "method": "workspace",
                "reason": f"Estimated numeric workspace {peak} bytes exceeds --memory-mib budget {memory_budget_bytes} bytes. Reduce rows or sites, permutations or enabled models.",
            }
        )
    ranked_rows = (
        sum(eligible.values()) if all_pairs else sum(min(v, max_pairs) for v in eligible.values())
    )
    return {
        "schema_version": 1,
        "status": "blocked" if blockers else "ready",
        "blockers": blockers,
        "warnings": warnings,
        "observations": n,
        "execution": {
            "analysis_processes": 1,
            "blas_thread_environment": {
                key: os.environ[key]
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
                if key in os.environ
            },
        },
        "retained_sites": widths,
        "numbered_sites": numbered,
        "retained_pairs": raw,
        "reference_pairs": mapped,
        "eligible_pair_coordinates": eligible,
        "retained_triplets": math.prod(widths.values()),
        "reference_triplets": math.prod(numbered.values()),
        "candidate_universe": eligible_triples,
        "candidate_count": candidates,
        "candidate_mode": mode if threebody else None,
        "candidate_count_scope": "upper_bound_until_screen"
        if threebody and mode in SCREENED
        else "requested_coordinates",
        "candidate_screen_probes_upper": screened_probes,
        "local_hierarchy": local_details,
        "triple_null": "withheld"
        if withhold
        else "whole_component_shuffle"
        if threebody and permutations
        else None,
        "count_threshold": min_valid,
        "weight_threshold": min_valid_weight,
        "variable_sites": variable,
        "observed_state_features": dimensions,
        "feature_scope": "upper bound before choosing positive representatives"
        if weight_policy == "cluster_representatives"
        else "positive-weight observations",
        "weighting": {
            **wplan,
            "identity_threshold": identity_threshold,
            "min_overlap": min_overlap,
        },
        "dca_solver": choice,
        "dca_preflight": dca_plan,
        "dca_limits": {
            "max_solve_dimension": MAX_SOLVE_DIMENSION,
            "max_dual_elements": MAX_DUAL_ELEMENTS,
        },
        "global": global_details,
        "permutations": permutations,
        "minimum_permutation_p": 1 / (permutations + 1) if permutations else None,
        "estimates": {
            "numeric_workspace_bytes": peak,
            "budget_bytes": memory_budget_bytes,
            "mi_workspace_bytes": peak_pair,
            "dca_workspace_bytes": peak_dca,
            "dca_gram_formation_flops_per_observed_fit": gram_flops,
            "permutation_workspace_bytes": peak_null,
            "mi_observation_pair_visits": n * pairs * (1 + permutations),
            "matrix_payload_before_compression_bytes": array_bytes + (16 * pairs if dca else 0),
            "ranked_pair_rows_upper": ranked_rows * (1 + int(dca)),
            "tested_triple_support_rows": candidates,
            "score_text_bytes_approx": 256
            * (
                ranked_rows * (1 + int(dca))
                + nulls
                + 2 * candidates
                + (settings_count * candidates if local_details else 0)
            ),
            "scope": "Approximate numeric workspace and text size. Excludes interpreter, file parsing, BLAS-internal memory, benchmarks and HTML; no wall-clock or process-RSS guarantee.",
        },
    }


def publish_plan(out, plan, inputs, audit=None):
    """Publish a non-scoring plan on both successful and blocked dry runs."""
    from triplet_coevolution.cli import _stage_run
    from triplet_coevolution.msa.diagnostics import write_qc
    from triplet_coevolution.provenance import file_hashes, software_identity
    from triplet_coevolution.score_views import write_json

    records = {
        k: {**info, "path": replay_input_path(Path(info["path"]), out)}
        for k, info in inputs.items()
    }
    plan = dict(plan, inputs=records, software=software_identity())
    with _stage_run(out) as stage:
        if audit is not None:
            plan["diagnostics"] = write_qc(stage, audit)
        plan["output_sha256"] = file_hashes(stage)
        write_json(stage / "plan.json", plan)


def write_weights(path, weights):
    rows = iter(weights.table_rows())
    first = next(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def replay_input_path(path, out):
    """Reference generated siblings after an outer workflow atomically moves its stage."""
    path = path.expanduser().absolute()
    parent = next((p for p in out.resolve().parents if p.name == ".triplet-in-progress"), None)
    return (
        os.path.relpath(path, out.absolute())
        if parent and path.is_relative_to(parent)
        else str(path)
    )


def replay_arguments(options, out):
    """Replay native options from the analysis directory, including generated sibling inputs."""
    argv = ["analyze"]
    for key, value in options.items():
        if key in {"dry_run", "out"} or value is None or value is False:
            continue
        name = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            argv.append(name)
            continue
        if isinstance(value, Path):
            value = replay_input_path(value, out)
        for item in value if isinstance(value, list) else (value,):
            argv.extend([name, str(item)])
    return {
        "schema_version": 1,
        "working_directory": "analysis output directory",
        "arguments": argv,
        "output": "Append --out with a new or empty path.",
    }


def write_weighted_sites(path, matched, audit, weights):
    """Summarize final observation weights on every original site, including excluded columns."""
    keys = (
        "component",
        "original_column",
        "reference_id",
        "reference_position",
        "retained",
        "positive_canonical_rows",
        "canonical_weight",
        "gap_weight",
        "missing_weight",
        "canonical_kish",
    )
    weight = weights.weights
    ids = np.array(audit.row_indices, dtype=int)
    audit_sites = iter(audit.sites)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(keys)
        for comp in matched.components:
            q = len(CANONICAL[comp.molecule])
            # Column-at-a-time arrays keep this pass O(N) in temporary memory.
            for j in range(comp.n_sites):
                site = next(audit_sites)
                col = comp.matrix[ids, j]
                valid = (col >= 0) & (col < q)
                positive = weight[valid & (weight > 0)]
                mass = float(positive.sum())
                if len(positive):
                    relative = positive / positive.max()
                    kish = float(relative.sum() ** 2 / (relative @ relative))
                else:
                    kish = None
                writer.writerow(
                    (
                        comp.name,
                        site["source_column"],
                        site["reference_id"],
                        site["reference_position"],
                        site["retained"],
                        len(positive),
                        mass,
                        float(weight[col == q].sum()),
                        float(weight[col < 0].sum()),
                        kish,
                    )
                )
