"""Recover estimable triplets after quality filtering removes a row's last canonical residue."""

import csv

import numpy as np

from triplet_coevolution.config import COMPONENTS
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.io.alignments import read_msa
from triplet_coevolution.io.metadata import (
    REQUIRED_COLUMNS,
    TripletTable,
    read_triplet_table,
    write_triplet_table,
)
from triplet_coevolution.msa.qc import triplet_qc
from triplet_coevolution.msa.support import retained_sites, site_masks
from triplet_coevolution.msa.synchronize import synchronize


def filter_observations(prepared, kwargs):
    """Recompute column and row support until the synchronized scoring table stabilizes."""
    components = tuple(
        read_msa(kwargs[c.lower()], c, kwargs[f"{c.lower()}_type"]) for c in COMPONENTS
    )
    original = read_triplet_table(kwargs["matches"])
    panel = site_masks(kwargs["site_panel"], components) if kwargs.get("site_panel") else None
    thresholds = {
        k: kwargs.get(k, 0.5) for k in ("min_coverage", "max_gap_fraction", "max_missing_fraction")
    }
    table, preview, pass_number = original, [], 0
    exclusions = prepared / "observation_qc.tsv"
    matches = prepared / "scored_matches.tsv"
    columns = prepared / "scored_columns.tsv"
    with exclusions.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow((*REQUIRED_COLUMNS, "qc_pass", "components_without_canonical_residues"))
        while table is not None:
            pass_number += 1
            aligned = synchronize(*components, table)
            masks = retained_sites(triplet_qc(aligned), thresholds, panel)
            present = []
            for part, keep in zip(aligned.components, masks, strict=True):
                values = part.matrix[:, keep]
                present.append(
                    ((values >= 0) & (values < len(CANONICAL[part.molecule]))).any(axis=1)
                )
            supported = np.logical_and.reduce(present)
            if supported.all():
                break
            for idx in np.flatnonzero(~supported):
                match = table.matches[idx]
                absent = ",".join(
                    c for c, ok in zip(COMPONENTS, present, strict=True) if not ok[idx]
                )
                writer.writerow(
                    (match.observation_id, match.a_id, match.b_id, match.c_id, pass_number, absent)
                )
                if len(preview) < 10:
                    preview.append(
                        {
                            "observation_id": match.observation_id,
                            "components": absent,
                            "pass": pass_number,
                        }
                    )
            remaining = np.flatnonzero(supported)
            table = (
                TripletTable(
                    tuple(table.matches[i] for i in remaining),
                    table.extra_columns,
                    tuple(table.annotations[i] for i in remaining),
                )
                if len(remaining)
                else None
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
