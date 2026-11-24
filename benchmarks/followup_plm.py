"""Fit score-independent PLM panels and compare common-support real contact rankings."""

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from math import ceil, fsum, sqrt
from time import perf_counter

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"]:
    os.environ[variable] = "1"

import numpy as np
from cohort_sources import HERE, STAGE, atomic_json, digest
from independent_contact_metrics import top
from independent_pair_scores import OUT, checked_json, prepare

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.mi import triplet_mi
from triplet_coevolution.inference.plm import PLMOptions, fit_plm, plm_preflight
from triplet_coevolution.inference.potts import direct_couplings
from triplet_coevolution.validation.benchmark import _metrics

REG_SHA = "b8275f331bcd1e1caa4c6e6e8a7a635885e096d765b98c735e3fde3d1059a7aa"
RULE = checked_json(HERE / "plm_geometry_followup_policy.json", REG_SHA)
DEST = STAGE.parent / "plm-geometry-followup"
PAIRS = ("AB", "AC", "BC")
METHODS = ("mi", "mi_apc", "mf", "mf_apc", "plm", "plm_apc")
DATA = None


def raw_geometry(system):
    with gzip.open(HERE / "independent_contact_metrics.json.gz", "rt") as f:
        report = json.load(f)[system]["structure"]
    result = {}
    for copy in RULE["copies"][system]:
        path = OUT / system / (copy + "-distance.npz")
        if digest(path) != report["copies"][copy]["distance_sha256"]:
            raise ValueError("historical distance data changed")
        with np.load(path) as data:
            result[copy] = {p: data[p] for p in PAIRS}
    return result


def init(system):
    global DATA
    msa, maps, _, _ = prepare(system)
    groups = []
    for c, m in zip(msa.components, maps, strict=True):
        q = len(CANONICAL[c.molecule])
        count = ((c.matrix >= 0) & (c.matrix < q)).sum(axis=0)
        eligible = np.flatnonzero(count * 100 >= 98 * msa.n_observations)
        selected = sorted(
            eligible,
            key=lambda j: hashlib.sha256(
                f"followup-plm-v1/{system}/{c.name}/{m.reference_positions[j]}".encode("ascii")
            ).digest(),
        )[:64]
        if len(selected) != 64:
            raise ValueError("Insufficient eligible high-coverage panel")
        groups.append(np.array(selected).reshape(8, 8))
    DATA = (system, msa, maps, groups)


def fit_panel(index):
    system, msa, maps, groups = DATA
    i, j = index
    dest = DEST / system / "panels" / f"{i}-{j}.json"
    if dest.exists():
        cached = json.loads(dest.read_text())
        if cached["registration"] != REG_SHA:
            raise ValueError("panel registry mismatch")
        return cached
    indices = [groups[0][i], groups[1][j], groups[2][(i + j) % 8]]
    positions = {
        c.name: [m.reference_positions[k] for k in ix]
        for c, m, ix in zip(msa.components, maps, indices, strict=True)
    }
    keep = np.ones(msa.n_observations, dtype=bool)
    for c, ix in zip(msa.components, indices, strict=True):
        sub = c.matrix[:, ix]
        keep &= ((sub >= 0) & (sub < len(CANONICAL[c.molecule]))).all(axis=1)
    rows = np.flatnonzero(keep)
    selected = tuple(msa.matches[r] for r in rows)
    minimum = ceil(msa.n_observations / 2)
    res = {
        "registration": REG_SHA,
        "index": [i, j],
        "positions": positions,
        "eligible": int(keep.sum()),
        "input_n": msa.n_observations,
        "row_sha": hashlib.sha256(
            "\n".join(m.observation_id for m in selected).encode()
        ).hexdigest(),
        "scores": {},
    }
    if len(rows) >= minimum:
        pieces = []
        for c, ix in zip(msa.components, indices, strict=True):
            pieces.append(
                ComponentMSA(
                    c.name,
                    c.molecule,
                    tuple(c.ids[r] for r in rows),
                    tuple("".join(c.sequences[r][k] for k in ix) for r in rows),
                )
            )
        subset = MatchedTripletMSA(*pieces, selected)
        options = PLMOptions(**RULE["plm"]["native"])
        res["preflight"] = plm_preflight(subset, options)
        clock = perf_counter()
        check = {}

        def sink(pair, a, b, block):
            if pair not in check:
                norm = sqrt(fsum(float(x) ** 2 for x in block.ravel()))
                check[pair] = (a, b, norm)

        fit = fit_plm(subset, options=options, min_valid=minimum, coupling_sink=sink)
        mi = triplet_mi(subset, min_valid=minimum)
        mf = direct_couplings(subset, min_valid=minimum)
        for p in PAIRS:
            pm = fit.blocks[p]
            mm = getattr(mi, p.lower())
            if p in check:
                a, b, norm = check[p]
                assert np.isclose(pm.frobenius[a, b], norm, rtol=1e-12)
            for support in (pm.n_valid, mm.n_valid, mf[p].n_valid):
                assert np.all(support == len(rows))
            arrays = (
                mm.mi_bits,
                mm.mi_apc,
                mf[p].frobenius,
                mf[p].corrected,
                pm.frobenius,
                pm.corrected,
            )
            res["scores"][p] = {
                m: [[float(v) if np.isfinite(v) else None for v in r] for r in a]
                for m, a in zip(METHODS, arrays, strict=True)
            }
        res.update(
            status=fit.diagnostics["status"],
            targets=fit.diagnostics["targets"],
            independent_norm_checks=len(check),
            fit_seconds=perf_counter() - clock,
        )
    else:
        res["status"] = "insufficient_complete_rows"
    atomic_json(dest, res)
    return res


def stats(values, d):
    common = np.isfinite(d) & np.logical_and.reduce([np.isfinite(v) for v in values.values()])
    out = {
        "n": len(d),
        "observable": int(np.isfinite(d).sum()),
        "unknown": int(np.isnan(d).sum()),
        "common": int(common.sum()),
        "finite": {m: int(np.isfinite(v).sum()) for m, v in values.items()},
        "cuts": {},
        "top": {m: top(v, d) for m, v in values.items()},
    }
    for k in (5, 8, 10):
        y = d[common] <= k
        pos = int(y.sum())
        metrics = {}
        for m, v in values.items():
            res = _metrics(y, v[common], curves=False)
            metrics[m] = {
                "ap": None if pos in (0, len(y)) else res["average_precision"],
                "auc": res["auroc"],
            }
            if res["auroc"] is not None:
                _, inv, n = np.unique(v[common], return_inverse=True, return_counts=True)
                ranks = (np.cumsum(n) - (n - 1) / 2)[inv]
                auc = (ranks[y].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos))
                assert np.isclose(auc, res["auroc"], atol=1e-12, rtol=0)
        out["cuts"][str(k)] = {
            "observable_positives": int(np.count_nonzero(d <= k)),
            "positives": pos,
            "prevalence": pos / len(y) if len(y) else None,
            "methods": metrics,
        }
    return out


def evaluate(system, panels):
    pairs = {p: {} for p in PAIRS}
    for r in panels:
        for p in PAIRS:
            for u, a in enumerate(r["positions"][p[0]]):
                for v, b in enumerate(r["positions"][p[1]]):
                    if (a, b) in pairs[p]:
                        raise ValueError("panel pair repeated")
                    pairs[p][a, b] = [
                        r["scores"].get(p, {}).get(m, [[None] * 8] * 8)[u][v] for m in METHODS
                    ]
    report = {}
    for copy, labels in raw_geometry(system).items():
        results = {}
        dists = []
        combined = []
        for p in PAIRS:
            coords = sorted(pairs[p])
            assert len(coords) == 4096
            vals = np.array([pairs[p][k] for k in coords], dtype=float)
            d = np.array([labels[p][a - 1, b - 1] for a, b in coords])
            results[p] = stats(dict(zip(METHODS, vals.T, strict=True)), d)
            combined.append(vals)
            dists.append(d)
        results["pooled"] = stats(
            dict(zip(METHODS, np.concatenate(combined).T, strict=True)), np.concatenate(dists)
        )
        results["macro"] = {}
        for k in ("5", "8", "10"):
            results["macro"][k] = {}
            for m in METHODS:
                summary = {}
                for key in ("ap", "auc"):
                    finite = [
                        results[p]["cuts"][k]["methods"][m][key]
                        for p in PAIRS
                        if results[p]["cuts"][k]["methods"][m][key] is not None
                    ]
                    summary[key] = float(np.mean(finite)) if finite else None
                    summary["interfaces"] = len(finite)
                results["macro"][k][m] = summary
        report[copy] = results
    statuses = Counter(t["status"] for r in panels for t in r.get("targets", []))
    summary = {
        "system": system,
        "registration": REG_SHA,
        "panels": len(panels),
        "panel_status": dict(Counter(r["status"] for r in panels)),
        "target_status": dict(statuses),
        "complete_row_range": [
            min(r["eligible"] for r in panels),
            max(r["eligible"] for r in panels),
        ],
        "norm_checks": sum(r.get("independent_norm_checks", 0) for r in panels),
        "copies": report,
        "panel_digests": {
            f"{r['index'][0]}-{r['index'][1]}": digest(
                DEST / system / "panels" / f"{r['index'][0]}-{r['index'][1]}.json"
            )
            for r in panels
        },
    }
    atomic_json(DEST / system / "plm.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=["rnap", "ribosome"])
    args = parser.parse_args()
    (DEST / args.system / "panels").mkdir(exist_ok=True, parents=True)
    with ProcessPoolExecutor(max_workers=4, initializer=init, initargs=(args.system,)) as pool:
        panels = []
        for r in pool.map(fit_panel, product(range(8), repeat=2)):
            panels.append(r)
            if len(panels) % 8 == 0:
                print(args.system, len(panels), "panels", flush=True)
    print("Evaluated", evaluate(args.system, panels)["panel_status"], flush=True)
