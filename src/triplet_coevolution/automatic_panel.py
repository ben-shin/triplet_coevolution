"""Build a replayable, score-independent subset of a prepared three-component panel."""

import csv
import hashlib
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from triplet_coevolution.io.metadata import REQUIRED_COLUMNS
from triplet_coevolution.io.stockholm import open_stockholm

_COMPONENTS = ("A", "B", "C")
_TARGETS = {
    "max_depth": 128,
    "minimum_depth": 8,
    "minimum_sites_per_component": 2,
    "pair_cells": 80_000,
    "depth_times_pair_cells": 4_000_000,
    "weight_cells": 80_000_000,
    "depth_times_state_width": 3_000_000,
    "raw_alignment_cells": 8_000_000,
}


def _rank(seed, value):
    return hashlib.sha256(f"triplet-panel-v1\0{seed}\0{value}".encode()).digest()


def _index_matches(db, path, seed, strata_column):
    """Spool the match annotations to disk and count strata independently of row order."""
    db.execute("CREATE TABLE matches (id TEXT PRIMARY KEY, stratum TEXT, rank BLOB, fields TEXT)")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", strict=True)
        header = reader.fieldnames or []
        if not set(REQUIRED_COLUMNS) <= set(header) or len(set(header)) != len(header):
            raise ValueError(
                "the prepared match table requires unique observation and component columns"
            )
        if strata_column is not None and strata_column not in header:
            raise ValueError(f"unknown panel stratification column {strata_column!r}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"malformed prepared match table at line {reader.line_num}")
            identifier = row["observation_id"]
            stratum = row[strata_column] if strata_column else ""
            if not identifier or strata_column and not stratum:
                raise ValueError(f"empty observation or stratum at line {reader.line_num}")
            db.execute(
                "INSERT INTO matches VALUES (?,?,?,?)",
                (
                    identifier,
                    stratum,
                    _rank(seed, identifier),
                    json.dumps(row),
                ),
            )
    db.execute("CREATE INDEX by_stratum ON matches(stratum,rank,id)")
    return header, db.execute("SELECT count(*) FROM matches").fetchone()[0]


def _shape(widths, molecules, total, limits, min_valid, select_columns):
    targets = _TARGETS
    # The native reader constructs maps and initial QC on full-length scoring rows.
    # A reference outside the row sample can add one row to each component.
    maximum = min(total, targets["max_depth"], targets["raw_alignment_cells"] // sum(widths) - 1)
    minimum = max(min_valid, targets["minimum_depth"])
    min_sites = [min(w, targets["minimum_sites_per_component"]) for w in widths]

    def fits(n, sites):
        a, b, c = sites
        pairs = a * b + a * c + b * c
        state_width = sum(
            w * min(n, 20 if mol == "protein" else 4) for w, mol in zip(sites, molecules)
        )
        return (
            pairs <= min(limits["pair_cells"], targets["pair_cells"])
            and n * pairs <= targets["depth_times_pair_cells"]
            and n * n * sum(sites) <= min(limits["weight_cells"], targets["weight_cells"])
            and n * state_width <= targets["depth_times_state_width"]
        )

    for n in range(maximum, minimum - 1, -1):
        if fits(n, widths):
            return n, widths
    if select_columns:
        for n in range(maximum, minimum - 1, -1):
            if not fits(n, min_sites):
                continue
            low, high = targets["minimum_sites_per_component"], max(widths)
            while low < high:
                cap = (low + high + 1) // 2
                if fits(n, [min(w, cap) for w in widths]):
                    low = cap
                else:
                    high = cap - 1
            return n, [min(w, low) for w in widths]
    return 0, [0, 0, 0]


def _observations(db, total, n, seed, stratify):
    if not stratify:
        return [
            json.loads(row)
            for (row,) in db.execute("SELECT fields FROM matches ORDER BY rank,id LIMIT ?", (n,))
        ], {}
    # Largest-remainder allocation by stratum size. Seeded hash breaks ties.
    db.create_function("stratum_rank", 1, lambda value: _rank(seed, value), deterministic=True)
    db.execute(
        """
        CREATE TEMP TABLE quotas AS
        WITH sizes AS (SELECT stratum,count(*) AS size FROM matches GROUP BY stratum),
        allocation AS (SELECT *,size*:n/:total AS base,size*:n%:total AS rem FROM sizes)
        SELECT stratum,size,base + (row_number() OVER (ORDER BY rem DESC,stratum_rank(stratum),stratum)
            <= :n-sum(base) OVER ()) AS wanted FROM allocation
    """,
        {"n": n, "total": total},
    )
    rows, strata = [], {}
    for stratum, count, wanted in db.execute(
        "SELECT stratum,size,wanted FROM quotas WHERE wanted>0 ORDER BY stratum"
    ):
        strata[stratum] = {"input": count, "selected": wanted}
        rows.extend(
            json.loads(row)
            for (row,) in db.execute(
                "SELECT fields FROM matches WHERE stratum=? ORDER BY rank,id LIMIT ?",
                (stratum, wanted),
            )
        )
    return rows, strata


def make_scoring_panel(
    prepared,
    kwargs,
    components,
    limits,
    *,
    min_valid=2,
    seed=0,
    strata_column=None,
    select_columns=True,
):
    """Build a reduced panel in a fresh prepared directory, retaining the originals for replay.

    Return native analysis path overrides and an audit. Infeasible budgets return an
    empty overrides dict and a resource_limit audit before installing any panel artifacts.
    """
    prepared = Path(prepared)
    widths = [components[c]["source_columns"] for c in _COMPONENTS]
    molecules = [components[c]["molecule"] for c in _COMPONENTS]
    with tempfile.TemporaryDirectory(dir=prepared) as scratch:
        staging = Path(scratch)
        with closing(sqlite3.connect(staging / "ranking.sqlite")) as db:
            db.execute("PRAGMA temp_store=FILE")
            header, total = _index_matches(db, kwargs["matches"], seed, strata_column)
            n, sites = _shape(widths, molecules, total, limits, min_valid, select_columns)
            audit = {
                "n_input": total,
                "n_selected": n,
                "source_widths": dict(zip(_COMPONENTS, widths)),
                "selected_widths": dict(zip(_COMPONENTS, sites)),
                "method": "sha256_observation_id_v1",
                "site_method": "all_source_columns"
                if widths == sites
                else "equal_width_bin_midpoints",
                "seed": seed,
                "targets": dict(_TARGETS),
                "strata_column": strata_column,
                "matches": None,
                "site_panel": None,
            }
            if not n:
                audit.update(
                    status="resource_limit",
                    reason=(
                        f"No scoring panel fits the budgets at a minimum of {max(_TARGETS['minimum_depth'], min_valid)} observations."
                    ),
                )
                return {}, audit
            rows, strata = _observations(db, total, n, seed, strata_column is not None)
        rows.sort(key=lambda row: row["observation_id"])
        names = {"matches": "panel_matches.tsv"}
        with (staging / names["matches"]).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=header, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        if widths != sites:
            names["site_panel"] = "panel_columns.tsv"
            with (staging / names["site_panel"]).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(("component", "original_column"))
                for component, width, keep in zip(_COMPONENTS, widths, sites):
                    # One midpoint per equal-width source interval, in original 1-based coordinates.
                    writer.writerows(
                        (component, (2 * j + 1) * width // (2 * keep) + 1) for j in range(keep)
                    )
            audit["site_panel"] = "prepared/" + names["site_panel"]
        for component, molecule in zip(_COMPONENTS, molecules):
            key = component.lower()
            names[key] = f"panel_{key}.sto"
            ids = [row[f"{key}_id"] for row in rows]
            reference = kwargs.get(f"{key}_ref_row") or kwargs[f"{key}_ref"]
            if reference not in ids:
                ids.append(reference)
            with open_stockholm(kwargs[key], molecule=molecule, temp_dir=staging) as src:
                src.extract(ids, staging / names[key])
        for name in names.values():
            (staging / name).replace(prepared / name)
        audit.update(
            status="selected",
            reason="Selected a reproducible panel within the scoring and benchmark budgets.",
            matches="prepared/" + names["matches"],
            strata_counts=strata,
            selected_observation_ids=[row["observation_id"] for row in rows],
        )
    return {key: prepared / name for key, name in names.items()}, audit
