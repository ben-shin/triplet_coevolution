"""Summarize registered structural rankings with common supports, unknowns and baselines."""

import argparse
import csv
import gzip
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from cohort_sources import atomic_json, digest
from independent_pair_scores import OUT, POLICY, POLICY_SHA

from triplet_coevolution.io.score_store import PAIRS, load_score_store
from triplet_coevolution.validation.benchmark import _metrics

METHODS = POLICY["pair_scores"]["outputs"]
WEIGHTS = POLICY["pair_scores"]["weights"]
CUTOFFS = POLICY["labels"]["positive_at_angstrom"]


def counts(d):
    valid = np.isfinite(d)
    return {
        "n": int(d.size),
        "observable": int(valid.sum()),
        "unknown": int((~valid).sum()),
        "positive": {str(k): int(np.count_nonzero(d <= k)) for k in CUTOFFS},
        "distant_25_or_more": int(np.count_nonzero(d >= 25)),
    }


def top(scores, distances):
    finite = np.flatnonzero(np.isfinite(scores))
    # The caller concatenates in PAIRS order and each C-order array follows reference order.
    order = finite[np.argsort(-scores[finite], kind="stable")]
    out = {}
    for k in POLICY["labels"]["top_k"]:
        chosen = distances[order[:k]]
        n = len(chosen)
        pos = int((chosen <= 8).sum())
        unknown = int(np.isnan(chosen).sum())
        out[str(k)] = {
            "n": n,
            "positive": pos,
            "negative": n - pos - unknown,
            "unknown": unknown,
            "lower": pos / n if n else None,
            "upper": (pos + unknown) / n if n else None,
        }
    return out


def rankings(d, values):
    both = np.isfinite(d)
    for v in values.values():
        both &= np.isfinite(v)
    y = d[both]
    result = {"common": counts(y), "strata": {}}
    for label, k, include in [(str(t), t, np.ones(len(y), dtype=bool)) for t in CUTOFFS] + [
        ("8_vs_25", 8, (y <= 8) | (y >= 25))
    ]:
        truth = y[include] <= k
        pos = int(truth.sum())
        n = len(truth)
        prev = pos / n if n else None
        metrics = {}
        for method, all_scores in values.items():
            m = _metrics(truth, all_scores[both][include], curves=False)
            metrics[method] = {
                "ap": m["average_precision"] if 0 < pos < n else None,
                "auroc": m["auroc"] if 0 < pos < n else None,
            }
        differences = {}
        for a, b in [("mi_apc", "mi_bits"), ("dca_frobenius", "mi_bits"), ("dca_apc", "mi_apc")]:
            differences[a + " minus " + b] = {
                k: None if metrics[a][k] is None else metrics[a][k] - metrics[b][k]
                for k in ("ap", "auroc")
            }
        result["strata"][label] = {
            "n": n,
            "positive": pos,
            "negative": n - pos,
            "prevalence": prev,
            "methods": metrics,
            "paired_differences": differences,
        }
    return result


def macro(groups):
    result = {}
    for t in [*(str(i) for i in CUTOFFS), "8_vs_25"]:
        result[t] = {}
        for m in METHODS:
            result[t][m] = {}
            for k in ["ap", "auroc"]:
                vals = [groups[p]["strata"][t]["methods"][m][k] for p in PAIRS]
                valid = [x for x in vals if x is not None]
                result[t][m][k] = {
                    "mean": float(np.mean(valid)) if valid else None,
                    "interfaces": len(valid),
                }
    return result


def evaluate(name):
    out = OUT / name
    structure = json.loads((out / "structure.json").read_text())
    scores = {w: load_score_store(out / w) for w in WEIGHTS}
    axes = {c.component: np.array(c.reference_positions) - 1 for c in scores["uniform"].maps}
    for store in scores.values():
        if any(
            not np.array_equal(axes[c.component], np.array(c.reference_positions) - 1)
            for c in store.maps
        ):
            raise ValueError("Score axes differ across weights")
        if store.provenance["evaluation_policy_sha256"] != POLICY_SHA:
            raise ValueError("Store does not bind registered policy")
    finite_by_weight = {
        w: {p: np.logical_and.reduce([np.isfinite(s.scores[m][p]) for m in METHODS]) for p in PAIRS}
        for w, s in scores.items()
    }
    crossweight = {
        p: np.logical_and.reduce([finite_by_weight[w][p] for w in WEIGHTS]) for p in PAIRS
    }
    same_weight = all(
        np.array_equal(finite_by_weight[a][p], finite_by_weight[b][p])
        for a, b in combinations(WEIGHTS, 2)
        for p in PAIRS
    )
    full = {}
    for copy, cert in structure["copies"].items():
        file = out / (copy + "-distance.npz")
        if digest(file) != cert["distance_sha256"]:
            raise ValueError("Distance data differ from frozen slice")
        with np.load(file) as arrays:
            full[copy] = {p: arrays[p] for p in PAIRS}
    shared = {p: np.logical_and.reduce([np.isfinite(v[p]) for v in full.values()]) for p in PAIRS}
    report = {
        "policy_sha256": POLICY_SHA,
        "structure": structure,
        "fits": {w: json.loads((out / w / "fit.json").read_text()) for w in WEIGHTS},
        "same_weight_finite_masks": same_weight,
        "copies": {},
    }
    for copy, distances in full.items():
        rep = report["copies"][copy] = {
            "all_reference": {p: counts(d) for p, d in distances.items()},
            "weights": {},
        }
        rep["all_reference"]["pooled"] = counts(
            np.concatenate([distances[p].ravel() for p in PAIRS])
        )
        distance_retain = {p: distances[p][np.ix_(axes[p[0]], axes[p[1]])] for p in PAIRS}
        rep["retained"] = {p: counts(d) for p, d in distance_retain.items()}
        for w, store in scores.items():
            info = rep["weights"][w] = {"views": {}, "top_before_mask": {}}
            poolscores = {
                m: np.concatenate([store.scores[m][p].ravel() for p in PAIRS]) for m in METHODS
            }
            pooldist = np.concatenate([distance_retain[p].ravel() for p in PAIRS])
            for p in (*PAIRS, "pooled"):
                d = pooldist if p == "pooled" else distance_retain[p].ravel()
                info["top_before_mask"][p] = {
                    m: top(poolscores[m] if p == "pooled" else store.scores[m][p].ravel(), d)
                    for m in METHODS
                }
            views = {
                "natural": {p: np.ones_like(v, dtype=bool) for p, v in distance_retain.items()},
                "common_across_copies_and_weights": {
                    p: shared[p][np.ix_(axes[p[0]], axes[p[1]])] & crossweight[p] for p in PAIRS
                },
            }
            for view, masks in views.items():
                ds = [np.where(masks[p], distance_retain[p], np.nan).ravel() for p in PAIRS]
                if view != "natural" and all(
                    np.array_equal(np.isfinite(d), np.isfinite(distance_retain[p].ravel()))
                    for p, d in zip(PAIRS, ds, strict=True)
                ):
                    info["views"][view] = {"same_as": "natural"}
                    continue
                groups = {}
                for p, d in zip(PAIRS, ds, strict=True):
                    groups[p] = rankings(d, {m: store.scores[m][p].ravel() for m in METHODS})
                groups["pooled"] = rankings(np.concatenate(ds), poolscores)
                groups["macro"] = macro(groups)
                info["views"][view] = groups
        print("Evaluated", name, copy, flush=True)
    atomic_json(out / "evaluation.json", report)
    print("Result", name, digest(out / "evaluation.json"), flush=True)
    return report


def export_reports(destination):
    reports = {
        s: json.loads((OUT / s / "evaluation.json").read_text()) for s in ("rnap", "ribosome")
    }
    matrix = []
    for name, report in reports.items():
        for copy, detail in report["copies"].items():
            for weight, view in detail["weights"].items():
                for p in (*PAIRS, "pooled"):
                    result = view["views"]["natural"][p]["strata"]["8"]
                    for method in METHODS:
                        m = result["methods"][method]
                        topk = view["top_before_mask"][p][method]
                        prev = result["prevalence"]
                        matrix.append(
                            {
                                "system": name,
                                "copy": copy,
                                "weight": weight,
                                "interface": p,
                                "method": method,
                                "full_reference": detail["all_reference"][p]["n"],
                                "full_observable": detail["all_reference"][p]["observable"],
                                "full_observable_positive_8": detail["all_reference"][p][
                                    "positive"
                                ]["8"],
                                "common": result["n"],
                                "positives": result["positive"],
                                "prevalence": prev,
                                "ap": m["ap"],
                                "ap_minus_prevalence": (m["ap"] - prev)
                                if m["ap"] is not None
                                else None,
                                "auroc": m["auroc"],
                                "auroc_minus_chance": (m["auroc"] - 0.5)
                                if m["auroc"] is not None
                                else None,
                                **{
                                    f"top{k}_{field}": topk[str(k)][field]
                                    for k in (10, 50)
                                    for field in (
                                        "positive",
                                        "negative",
                                        "unknown",
                                        "lower",
                                        "upper",
                                    )
                                },
                            }
                        )
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "independent_contact_metrics.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(matrix)
    raw = (
        json.dumps(reports, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    (destination / "independent_contact_metrics.json.gz").write_bytes(gzip.compress(raw, mtime=0))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--export-only",
        type=Path,
        help="Export existing complete results to a compact JSON archive and a primary-cutoff TSV.",
    )
    args = p.parse_args()
    if args.export_only:
        export_reports(args.export_only)
    else:
        for system in ("rnap", "ribosome"):
            evaluate(system)
