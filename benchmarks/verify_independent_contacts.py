"""Verify sampled distances directly from mmCIF and each primary pooled AUROC by rank sums."""

import csv
import gzip
import json
from collections import defaultdict

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from independent_pair_scores import OUT

from triplet_coevolution.io.score_store import PAIRS, load_score_store


def verify(name):
    out = OUT / name
    report = json.loads((out / "evaluation.json").read_text())
    entry = report["structure"]["entry"]
    with gzip.open(OUT / "sources" / (entry + ".cif.gz"), "rt") as f:
        cif = MMCIF2Dict(f)
    cols = [
        "label_asym_id",
        "pdbx_PDB_model_num",
        "label_seq_id",
        "label_comp_id",
        "label_alt_id",
        "type_symbol",
        "occupancy",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
    ]
    store = {w: load_score_store(out / w) for w in report["fits"]}
    sampled = known = rank_checks = 0
    for copy, summary in report["copies"].items():
        with (out / (copy + "-mapping.tsv")).open(newline="") as f:
            axis = defaultdict(list)
            for row in csv.DictReader(f, delimiter="\t"):
                axis[row["component"]].append(row)
        selected = {}
        xyz = defaultdict(list)
        for role, rows in axis.items():
            for i in np.linspace(0, len(rows) - 1, 25, dtype=int):
                r = rows[i]
                if r["mapping_status"] == "mapped":
                    selected[r["label_asym_id"], r["polymer_index"]] = (
                        role,
                        i,
                        r["mon_id"],
                        r["altloc"],
                    )
        for label, model, idx, mon, alt, element, occupancy, x, y, z in zip(
            *(cif["_atom_site." + c] for c in cols), strict=True
        ):
            key = (label, idx)
            if (
                key not in selected
                or model != "1"
                or element in ("H", "D", "T")
                or occupancy in ("?", ".")
                or float(occupancy) <= 0
            ):
                continue
            role, i, selected_mon, selected_alt = selected[key]
            if mon == selected_mon and alt in (".", "?", selected_alt):
                xyz[role, i].append((float(x), float(y), float(z)))
        with np.load(out / (copy + "-distance.npz")) as blocks:
            for p in PAIRS:
                dist = blocks[p]
                for i in np.linspace(0, dist.shape[0] - 1, 25, dtype=int):
                    for j in np.linspace(0, dist.shape[1] - 1, 25, dtype=int):
                        a, b = xyz[p[0], i], xyz[p[1], j]
                        if a and b:
                            observed = np.linalg.norm(
                                np.asarray(a)[:, None, :] - np.asarray(b), axis=2
                            ).min()
                            assert np.isclose(observed, dist[i, j], rtol=0, atol=1e-9), (
                                name,
                                copy,
                                p,
                            )
                            known += 1
                        else:
                            assert np.isnan(dist[i, j]), (name, copy, p)
                        sampled += 1
            for weight, s in store.items():
                axes = {c.component: np.asarray(c.reference_positions) - 1 for c in s.maps}
                d = np.concatenate(
                    [blocks[p][np.ix_(axes[p[0]], axes[p[1]])].ravel() for p in PAIRS]
                )
                metrics = summary["weights"][weight]["views"]["natural"]["pooled"]["strata"]["8"]
                values = {
                    m: np.concatenate([s.scores[m][p].ravel() for p in PAIRS]) for m in s.scores
                }
                common = np.isfinite(d) & np.logical_and.reduce(
                    [np.isfinite(v) for v in values.values()]
                )
                truth = d[common] <= 8
                pos = int(truth.sum())
                assert (truth.size, pos) == (metrics["n"], metrics["positive"])
                for method, v in values.items():
                    x = v[common]
                    _, inverse, frequency = np.unique(x, return_inverse=True, return_counts=True)
                    last_rank = np.cumsum(frequency)
                    ranks = (last_rank - (frequency - 1) / 2)[inverse]
                    auc = (ranks[truth].sum() - pos * (pos + 1) / 2) / (pos * (len(ranks) - pos))
                    assert np.isclose(auc, metrics["methods"][method]["auroc"], rtol=0, atol=1e-12)
                    rank_checks += 1
    print(
        name,
        "sampled distances",
        sampled,
        "observed",
        known,
        "pooled AUROC checks",
        rank_checks,
        flush=True,
    )


if __name__ == "__main__":
    verify("rnap")
    verify("ribosome")
