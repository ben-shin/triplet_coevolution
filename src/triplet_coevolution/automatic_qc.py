"""Recover estimable triplets after quality filtering removes a row's last canonical residue."""

import csv

import numpy as np

from triplet_coevolution.config import COMPONENTS
from triplet_coevolution.io.metadata import (
    REQUIRED_COLUMNS,
    TripletTable,
    read_triplet_table,
    write_triplet_table,
)
from triplet_coevolution.io.selected_sources import matched_sources
from triplet_coevolution.msa.diagnostics import QCPolicy, inspect_qc
from triplet_coevolution.msa.support import site_masks
from triplet_coevolution.msa.synchronize import synchronize


def filter_observations(prepared, kwargs):
    """Recompute column and row support until the synchronized scoring table stabilizes."""
    original = read_triplet_table(kwargs["matches"])
    source = matched_sources(
        {c: kwargs[c.lower()] for c in COMPONENTS},
        original.matches,
        {c: kwargs[c.lower() + "_type"] for c in COMPONENTS},
        {c: kwargs[c.lower() + "_ref"] for c in COMPONENTS},
        row_references={c: kwargs.get(c.lower() + "_ref_row") for c in COMPONENTS},
        starts={c: kwargs.get(c.lower() + "_start", 1) for c in COMPONENTS},
        strands={c: kwargs.get(c.lower() + "_strand", "+") for c in COMPONENTS},
    )
    components, maps = source.components, source.maps
    panel = site_masks(kwargs["site_panel"], components, maps) if kwargs.get("site_panel") else None
    thresholds = {
        k: kwargs.get(k, 0.5) for k in ("min_coverage", "max_gap_fraction", "max_missing_fraction")
    }
    supported = inspect_qc(
        synchronize(*components, original),
        maps,
        original,
        column_options=thresholds,
        panel_masks=panel,
        policy=QCPolicy.from_config(
            {
                "iteration": "fixed_point",
                "rows": {c: {"min_canonical": 1} for c in COMPONENTS},
            }
        ),
    )
    masks = supported.masks
    pass_number = len(supported.summary["passes"])
    remaining = supported.row_indices
    table = (
        TripletTable(
            tuple(original.matches[i] for i in remaining),
            original.extra_columns,
            tuple(original.annotations[i] for i in remaining),
        )
        if remaining
        else None
    )
    preview = []
    exclusions = prepared / "observation_qc.tsv"
    matches = prepared / "scored_matches.tsv"
    columns = prepared / "scored_columns.tsv"
    failures = {}
    for row in supported.rows:
        if row["exclusion_reasons"]:
            failure = failures.setdefault(row["observation_id"], (row["exclusion_pass"], []))
            failure[1].append(row["component"])
    with exclusions.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow((*REQUIRED_COLUMNS, "qc_pass", "components_without_canonical_residues"))
        failed = sorted(
            (m for m in original.matches if m.observation_id in failures),
            key=lambda m: failures[m.observation_id][0],
        )
        for m in failed:
            turn, identifiers = failures[m.observation_id]
            absent = ",".join(identifiers)
            writer.writerow((m.observation_id, m.a_id, m.b_id, m.c_id, turn, absent))
            if len(preview) < 10:
                preview.append(
                    {"observation_id": m.observation_id, "components": absent, "pass": turn}
                )
    if table:
        write_triplet_table(matches, table)
    else:
        matches.write_text(
            "\t".join((*REQUIRED_COLUMNS, *original.extra_columns)) + "\n", encoding="utf-8"
        )
    with columns.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("component", "original_column"))
        if table:
            for name, keep in zip(COMPONENTS, masks, strict=True):
                writer.writerows((name, i + 1) for i in np.flatnonzero(keep))
    n = table.n_observations if table else 0
    audit = {
        "input_observations": original.n_observations,
        "eligible_observations": n,
        "excluded_observations": original.n_observations - n,
        "passes": pass_number,
        "reason": "No canonical residue in one or more components after column quality filtering.",
        "table": "prepared/" + exclusions.name,
        "matches": "prepared/" + matches.name,
        "columns": "prepared/" + columns.name,
        "examples": preview,
    }
    return {"matches": matches, "site_panel": columns}, audit
