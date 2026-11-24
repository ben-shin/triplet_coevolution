"""Check sampled triplet geometry against original atoms and summarize the two follow-ups."""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from cohort_sources import digest
from followup_geometry import CALIPERS, select_matched
from followup_plm import DEST, METHODS, PAIRS, REG_SHA, raw_geometry
from independent_pair_scores import OUT


def verify(system):
    stage = DEST / system
    geom = json.loads((stage / "geometry.json").read_text())
    plm = json.loads((stage / "plm.json").read_text())
    sel = stage / "geometry-blind-selection.json"
    if digest(sel) != geom["selection_sha256"] or plm["registration"] != REG_SHA:
        raise ValueError("Derived inputs do not match the study")
    selected = json.loads(sel.read_text())
    x = np.asarray(selected["features"])
    cases, pool, pairs = select_matched(x, selected["ci"], selected["ids"], len(x) // 10)
    assert pairs.tolist() == selected["matched"]
    assert len(set(pairs[:, 1])) == len(pairs) and np.all(
        np.abs(x[pairs[:, 0]] - x[pairs[:, 1]]) <= CALIPERS
    )
    assert not set(cases).intersection(pool)
    structure = json.loads((OUT / system / "structure.json").read_text())
    with gzip.open(OUT / "sources" / (structure["entry"] + ".cif.gz"), "rt") as h:
        cif = MMCIF2Dict(h)
    cols = (
        "label_asym_id",
        "label_seq_id",
        "pdbx_PDB_model_num",
        "label_comp_id",
        "label_alt_id",
        "type_symbol",
        "occupancy",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
    )
    raw = raw_geometry(system)
    known = unknown = 0
    # Physical coordinates selected by evenly spaced matched-pair registration order alone.
    sample = pairs[np.linspace(0, len(pairs) - 1, 17, dtype=int)].ravel()
    pos = np.asarray(selected["positions"])[sample]
    for copy, old in raw.items():
        lookup = {}
        atoms = defaultdict(list)
        requested = {(role, int(v)) for i, role in enumerate("ABC") for v in pos[:, i]}
        with (OUT / system / (copy + "-mapping.tsv")).open(newline="") as h:
            for r in csv.DictReader(h, delimiter="\t"):
                key = r["component"], int(r["reference_pos"])
                if key in requested and r["mapping_status"] == "mapped":
                    lookup[r["label_asym_id"], r["polymer_index"]] = (key, r["mon_id"], r["altloc"])
        for chain, idx, model, mon, alt, e, occ, a, b, c in zip(
            *(cif["_atom_site." + col] for col in cols), strict=True
        ):
            hit = lookup.get((chain, idx))
            if (
                hit is None
                or model != "1"
                or e in ("H", "D", "T")
                or occ in ("?", ".")
                or float(occ) <= 0
            ):
                continue
            key, chosen, altid = hit
            if chosen == mon and alt in ("?", ".", altid):
                atoms[key].append([float(a), float(b), float(c)])
        for trip in pos:
            for edge in PAIRS:
                a, b = [int(trip["ABC".index(role)]) for role in edge]
                xx, yy = atoms[edge[0], a], atoms[edge[1], b]
                value = old[edge][a - 1, b - 1]
                if xx and yy:
                    observed = np.linalg.norm(
                        np.asarray(xx)[:, None, :] - np.asarray(yy), axis=2
                    ).min()
                    assert abs(value - observed) < 1e-9
                    known += 1
                else:
                    assert np.isnan(value)
                    unknown += 1
    panels = []
    residuals = []
    iterations = []
    for idx, expected in plm["panel_digests"].items():
        path = stage / "panels" / (idx + ".json")
        assert digest(path) == expected
        item = json.loads(path.read_text())
        targets = [
            {
                k: t[k]
                for k in (
                    "role",
                    "retained_index",
                    "status",
                    "iterations",
                    "gradient_max",
                    "rows",
                    "parameters",
                )
            }
            for t in item.get("targets", [])
        ]
        for t in targets:
            if t["status"] == "converged":
                residuals.append(t["gradient_max"])
                iterations.append(t["iterations"])
        panels.append(
            {
                **{
                    k: item[k]
                    for k in ("index", "row_sha", "eligible", "input_n", "status", "positions")
                },
                "targets": targets,
                "native_preflight": item.get("preflight"),
            }
        )
    # Reconstruct one full panel's six score families' APC directly.
    first = json.loads((stage / "panels/0-0.json").read_text())
    for p in PAIRS:
        for m in METHODS[::2]:
            z = np.asarray(first["scores"][p][m], dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = z - np.nanmean(z, axis=1)[:, None] * np.nanmean(z, axis=0)[
                    None, :
                ] / np.nanmean(z)
            if np.nanmean(z) == 0:
                v = z
            assert np.allclose(
                v,
                np.asarray(first["scores"][p][m + "_apc"], dtype=float),
                equal_nan=True,
                rtol=1e-10,
                atol=1e-12,
            )
    return {
        "plm": plm,
        "geometry": geom,
        "panel_diagnostics": panels,
        "checks": {
            "raw_atom_distances_observed": known,
            "raw_atom_distances_unknown": unknown,
            "independent_apc_blocks": 9,
            "converged_targets": len(residuals),
            "max_gradient": max(residuals),
            "iteration_range": [min(iterations), max(iterations)],
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "output", type=Path, help="Compressed aggregate and compact target diagnostics JSON"
    )
    dest = p.parse_args().output
    result = {s: verify(s) for s in ("rnap", "ribosome")}
    payload = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    dest.write_bytes(gzip.compress(payload, mtime=0))
    print(sha256(payload).hexdigest(), {s: r["checks"] for s, r in result.items()}, flush=True)
