"""Benchmark full pair-score matrices on explicit reference-coordinate labels."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from triplet_coevolution.io.score_store import PAIRS, ScoreStore, file_digest


@dataclass(frozen=True)
class Label:
    contact: int | None
    distance: float | None
    cluster: str
    source: str
    split: str
    status: str


def read_benchmark_labels(path: Path, store: ScoreStore) -> dict:
    """Read binary or unknown labels. Repeated coordinate pairs must be adjudicated upstream."""
    return read_label_table(path, {m.component: m.reference_id for m in store.maps})


def read_label_table(path: Path, references: dict[str, str]) -> dict:
    """Parse labels against declared physical reference IDs before scores exist."""
    required = {
        f"{s}_{k}" for s in "xy" for k in ("component", "reference_id", "reference_pos")
    } | {"contact"}
    allowed = required | {"distance_angstrom", "cluster_id", "source", "split", "status"}
    labels = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        header = reader.fieldnames or []
        if (
            not required.issubset(header)
            or set(header) - allowed
            or len(header) != len(set(header))
        ):
            raise ValueError(
                "invalid benchmark label header; use v1 contact coordinates and supported annotation fields"
            )
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"malformed label at line {reader.line_num}")
            sites = []
            for side in "xy":
                c = row[f"{side}_component"]
                if c not in references or row[f"{side}_reference_id"] != references[c]:
                    raise ValueError(f"unknown reference identity at label line {reader.line_num}")
                pos = int(row[f"{side}_reference_pos"])
                if pos < 1:
                    raise ValueError("reference positions must be positive integers")
                sites.append((c, pos))
            (a, x), (b, y) = sorted(sites)
            key = (a + b, x, y)
            if a == b or key in labels:
                raise ValueError(f"duplicate or within-component label at {key}")
            value = row["contact"]
            if value not in {"", "0", "1"}:
                raise ValueError(f"expected 0, 1 or blank contact at {key}")
            status = row.get("status") or ("resolved" if value else "unknown")
            if status not in {"resolved", "unknown", "unresolved", "ambiguous", "excluded"} or (
                value and status != "resolved"
            ):
                raise ValueError(f"inconsistent structural label status at {key}")
            distance = float(row["distance_angstrom"]) if row.get("distance_angstrom") else None
            if distance is not None and (not np.isfinite(distance) or distance < 0):
                raise ValueError(f"invalid structural distance at {key}")
            split = row.get("split") or "unspecified"
            if split not in {"unspecified", "test", "train", "validation"}:
                raise ValueError(f"unknown evaluation split at {key}")
            labels[key] = Label(
                int(value) if value else None,
                distance,
                row.get("cluster_id", ""),
                row.get("source", ""),
                split,
                status,
            )
    return labels


def _metrics(truth, scores, *, curves=True):
    """Descending score thresholds; a tie is consumed in one step for both curves."""
    n = len(truth)
    order = np.argsort(-scores, kind="stable")
    x, y = scores[order], truth[order]
    ends = np.flatnonzero(np.r_[x[:-1] != x[1:], True]) if n else np.empty(0, dtype=int)
    tp = np.cumsum(y, dtype=np.int64)[ends]
    fp = ends + 1 - tp
    pos = int(tp[-1]) if n else 0
    neg = n - pos
    hits = np.diff(tp, prepend=0)
    misses = np.diff(fp, prepend=0)
    precision = tp / (ends + 1)
    auc = float(np.sum(misses * (tp - hits / 2)) / (pos * neg)) if pos and neg else None
    ap = float(np.sum(hits * precision) / pos) if pos else None
    result = {
        "auroc": auc,
        "average_precision": ap,
        "scored": n,
        "positives_scored": pos,
        "negatives_scored": neg,
        "class_support": "both_classes"
        if pos and neg
        else (
            "only_positives" if pos else ("only_negatives" if neg else "no_common_labeled_scores")
        ),
    }
    if curves:
        thresholds = x[ends].tolist()
        result["curves"] = {
            "thresholds": thresholds,
            "true_positives": tp.tolist(),
            "false_positives": fp.tolist(),
            "precision": precision.tolist(),
            "recall": (tp / pos).tolist() if pos else None,
            "false_positive_rate": (fp / neg).tolist() if neg else None,
            "initial": {"true_positives": 0, "false_positives": 0, "precision": None},
        }
    return result


def _intervals(truth, values, clusters, draws, seed):
    units = sorted(set(clusters) - {""})
    out = {
        "method": "cluster percentile bootstrap",
        "units": len(units),
        "requested_draws": draws,
        "seed": seed,
        "confidence": 0.95,
    }
    if not draws or len(units) < 2 or "" in clusters:
        return {
            **out,
            "status": "unavailable",
            "reason": "not requested"
            if not draws
            else "at least two explicitly independent clusters covering all common labels are required",
        }
    groups = [np.flatnonzero(clusters == u) for u in units]
    rng = np.random.default_rng(seed)
    samples = {"auroc": [], "average_precision": []}
    for _ in range(draws):
        idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        vals = _metrics(truth[idx], values[idx], curves=False)
        for name, sample in samples.items():
            if vals[name] is not None:
                sample.append(vals[name])
    for key, sample in samples.items():
        out[key] = {
            "valid_draws": len(sample),
            "interval": np.quantile(sample, [0.025, 0.975]).tolist() if len(sample) >= 2 else None,
        }
    out["status"] = "estimated" if any(len(x) >= 2 for x in samples.values()) else "unavailable"
    return out


def _distance_summary(values):
    values = values[np.isfinite(values)]
    edges = [0, 4, 8, 12, 20, 25, 40]
    return {
        "n": len(values),
        "unit": "angstrom",
        "min_median_max": np.quantile(values, [0, 0.5, 1]).tolist() if len(values) else None,
        "histogram": {
            "lower_edges_inclusive": edges,
            "counts": np.histogram(values, [*edges, np.inf])[0].tolist(),
        },
    }


def _top(order, truth, distances, ks, positive_total, prevalence):
    result = {}
    for name, k in ks.items():
        idx = order[:k]
        hit = int(np.sum(truth[idx] == 1))
        neg = int(np.sum(truth[idx] == 0))
        known = hit + neg
        result[name] = {
            "requested": k,
            "returned": len(idx),
            "positive": hit,
            "negative": neg,
            "unknown": len(idx) - known,
            "verified_positive_lower_bound": hit / len(idx) if len(idx) else None,
            "precision_among_labeled": hit / known if known else None,
            "precision": hit / len(idx) if len(idx) and known == len(idx) else None,
            "recall_of_known_positives": hit / positive_total if positive_total else None,
            "enrichment": (hit / len(idx)) / prevalence if prevalence and len(idx) else None,
            "distance": _distance_summary(distances[idx]),
        }
    return result


def _axes(store, labels):
    """Index only retained, numbered axes. Preserve genomic reference numbering and strand."""
    maps = {m.component: m for m in store.maps}
    prepared = {}
    for pair in PAIRS:
        x, y = [m.reference_positions for m in (maps[pair[0]], maps[pair[1]])]
        axis = [{v: i for i, v in enumerate(side) if v is not None} for side in (x, y)]
        keep = store.eligible(pair, reference_only=True).ravel()
        truth = np.full(keep.shape, -1, dtype=np.int8)
        distance = np.full(keep.shape, np.nan)
        positions = {}
        for key, label in labels.items():
            p, i, j = key
            if p != pair or i not in axis[0] or j not in axis[1]:
                continue
            index = axis[0][i] * len(y) + axis[1][j]
            positions[key] = index
            if label.split in {"train", "validation"}:
                continue
            if label.contact is not None:
                truth[index] = label.contact
            if label.distance is not None:
                distance[index] = label.distance
        prepared[pair] = (keep, truth, distance, positions, min(len(axis[0]), len(axis[1])))
    return prepared


def benchmark_scores(
    store: ScoreStore,
    labels_path: Path,
    *,
    top_k: Sequence[int] = (10, 25, 50, 100),
    bootstrap: int = 0,
    seed: int = 0,
) -> dict:
    """Compare all stored statistics on identical finite labeled pairs and report coverage."""
    if (
        any(type(k) is not int or k < 1 for k in top_k)
        or type(bootstrap) is not int
        or bootstrap < 0
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("top K must be positive integers and bootstrap/seed nonnegative integers")
    digest, labels = _stable_labels(labels_path, store)
    axes = _axes(store, labels)
    comparison = {p: axes[p][0] & (axes[p][1] >= 0) for p in PAIRS}
    for fields in store.scores.values():
        for p in PAIRS:
            comparison[p] &= np.isfinite(fields[p].ravel())
    excluded = []
    for key, label in labels.items():
        pair = key[0]
        i = axes[pair][3].get(key)
        if i is None:
            reason = "outside_retained_reference_axes"
        elif label.split in {"train", "validation"}:
            reason = "non_test_split"
        elif label.contact is None:
            reason = label.status
        elif store.n_valid[pair].ravel()[i] < 0:
            reason = "unreported_legacy_support"
        elif not axes[pair][0][i]:
            reason = "insufficient_pair_support"
        elif not comparison[pair][i]:
            reason = "missing_score_in_comparison"
        else:
            continue
        excluded.append({"pair": pair, "positions": list(key[1:]), "reason": reason})
    stats = {}
    for field, blocks in store.scores.items():
        scope = {}
        all_values = []
        all_truth = []
        all_dist = []
        all_masks = []
        common_values = []
        common_truth = []
        all_clusters = []
        for pair in PAIRS:
            keep, truth, dist, positions, length = axes[pair]
            maps = [store.maps["ABC".index(c)] for c in pair]
            numbered = [
                np.array([v is not None for v in m.reference_positions], dtype=bool) for m in maps
            ]
            mapped = numbered[0][:, None] & numbered[1][None, :]
            values = blocks[pair].ravel()
            finite = keep & np.isfinite(values)
            mask = comparison[pair]
            # Boolean selection is row-major; associate bootstrap units with exactly those rows.
            clusters_by_index = {i: labels[key].cluster for key, i in positions.items() if mask[i]}
            clusters = np.asarray([clusters_by_index[i] for i in np.flatnonzero(mask)], dtype=str)
            result = _metrics(truth[mask], values[mask])
            idx = np.flatnonzero(finite)
            order = idx[np.argsort(-values[idx], kind="stable")]
            ks = {str(k): k for k in top_k}
            ks.update({"L": length, "L/2": max(1, length // 2), "L/5": max(1, length // 5)})
            labeled = int(np.count_nonzero(keep & (truth >= 0)))
            positives = int(np.count_nonzero(keep & (truth == 1)))
            source_axes = store.provenance.get("source_retained_columns", {})
            known_universe = not np.any(mapped & (store.n_valid[pair] < 0)) and all(
                m.component not in source_axes or len(source_axes[m.component]) == m.n_sites
                for m in maps
            )
            prevalence = (
                positives / int(keep.sum())
                if known_universe and keep.any() and labeled == keep.sum()
                else None
            )
            result.update(
                coverage={
                    "eligible_pairs": int(keep.sum()),
                    "support_and_axes_complete": known_universe,
                    "mapped_reference_pairs": int(mapped.sum()),
                    "unreported_support": int(np.count_nonzero(mapped & (store.n_valid[pair] < 0))),
                    "finite": len(idx),
                    "unscored_eligible": int(keep.sum()) - len(idx),
                    "labeled_eligible": labeled,
                    "labeled_finite": int((finite & (truth >= 0)).sum()),
                    "labeled_unscored": int(np.sum(keep & ~finite & (truth >= 0))),
                    "common_labeled": int(mask.sum()),
                    "unlabeled_eligible": int(keep.sum()) - labeled,
                    "eligible_prevalence": prevalence,
                    "observed_labeled_prevalence": positives / labeled if labeled else None,
                },
                top_k=_top(order, truth, dist, ks, positives, prevalence),
                distance=_distance_summary(dist[finite]),
                uncertainty=_intervals(truth[mask], values[mask], clusters, bootstrap, seed),
            )
            scope[pair] = result
            all_values.append(values)
            all_truth.append(truth)
            all_dist.append(dist)
            all_masks.append(keep)
            common_values.append(values[mask])
            common_truth.append(truth[mask])
            all_clusters.append(clusters)
        pooled_truth = np.concatenate(common_truth)
        pooled_values = np.concatenate(common_values)
        pooled = _metrics(pooled_truth, pooled_values)
        vals = np.concatenate(all_values)
        y = np.concatenate(all_truth)
        ds = np.concatenate(all_dist)
        keep = np.concatenate(all_masks)
        idx = np.flatnonzero(keep & np.isfinite(vals))
        order = idx[np.argsort(-vals[idx], kind="stable")]
        labeled = int(np.sum(keep & (y >= 0)))
        positives = int(np.sum(keep & (y == 1)))
        known_universe = all(v["coverage"]["support_and_axes_complete"] for v in scope.values())
        prevalence = (
            positives / int(keep.sum())
            if known_universe and keep.any() and labeled == keep.sum()
            else None
        )
        pooled["top_k"] = _top(order, y, ds, {str(k): k for k in top_k}, positives, prevalence)
        pooled["coverage"] = {
            k: sum(scope[p]["coverage"][k] for p in PAIRS)
            for k in (
                "eligible_pairs",
                "mapped_reference_pairs",
                "unreported_support",
                "finite",
                "unscored_eligible",
                "labeled_eligible",
                "labeled_finite",
                "labeled_unscored",
                "common_labeled",
                "unlabeled_eligible",
            )
        }
        pooled["coverage"].update(
            eligible_prevalence=prevalence,
            support_and_axes_complete=known_universe,
            observed_labeled_prevalence=positives / labeled if labeled else None,
        )
        pooled["uncertainty"] = _intervals(
            pooled_truth, pooled_values, np.concatenate(all_clusters), bootstrap, seed
        )
        macro = {}
        for k in ("auroc", "average_precision"):
            eligible = [scope[p][k] for p in PAIRS if scope[p][k] is not None]
            macro[k] = float(np.mean(eligible)) if eligible else None
            macro[k + "_interfaces"] = len(eligible)
        stats[field] = {"interfaces": scope, "pooled": pooled, "macro": macro}
    return {
        "schema_version": 2,
        "statistics": stats,
        "label_sha256": digest,
        "label_records": len(labels),
        "source_universe": store.provenance.get(
            "universe", "Complete retained model axes and pair support."
        ),
        "excluded_labels": excluded,
        "eligibility": "Numbered reference-coordinate pairs meeting minimum pair support. Classification metrics use the common finite labeled set; top-K uses every finite eligible score for each method.",
        "tie_policy": "AUROC gives half credit, AP groups equal score thresholds. Top-K breaks ties by AB/AC/BC then original alignment coordinates.",
        "length_definition": "L is the smaller number of represented numbered sites of either component. Unmapped legacy columns do not contribute to L. Fractional L cutoffs round down with a minimum of one.",
        "macro_definition": "Unweighted mean over interfaces with a defined metric; each metric includes the number of contributing interfaces.",
        "degenerate_metrics": "AUROC requires both classes. AP requires at least one positive and equals one on an all-positive labeled set.",
        "calibration": {
            "status": "unavailable",
            "reason": "MI and coupling norms are association scores, not fitted probabilities.",
        },
        "evaluation_split": "Unspecified or explicit test labels; train/validation annotations are withheld.",
        "uncertainty_units": "Cluster resampling requires caller-supplied independent biological units; no independent-pair interval is reported.",
    }


def _stable_labels(path, store):
    path = Path(path)
    digest = file_digest(path)
    labels = read_benchmark_labels(path, store)
    if file_digest(path) != digest:
        raise ValueError("benchmark label input changed while reading")
    return digest, labels


def write_merged_scores(path: Path, store: ScoreStore, labels_path: Path) -> None:
    """Export every numbered reference pair, all fitted scores, support and explicit label evidence."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"merged score destination already exists: {path}")
    digest, labels = _stable_labels(labels_path, store)
    axes = _axes(store, labels)
    fields = [
        "pair",
        "x_reference_id",
        "x_current_col",
        "x_original_col",
        "x_reference_pos",
        "y_reference_id",
        "y_current_col",
        "y_original_col",
        "y_reference_pos",
        "eligible",
        "support_status",
        "n_valid",
        "valid_weight",
        "contact",
        "label_status",
        "label_source",
        "label_sha256",
        "distance_angstrom",
        "cluster_id",
        "split",
        "score_uncertainty",
    ]
    for stat in store.scores:
        fields.extend((stat, stat + "_competition_rank", stat + "_rank_percentile"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
        temp = Path(scratch) / "scores.tsv"
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for pair in PAIRS:
                keep = axes[pair][0]
                ranks = {}
                sizes = {}
                for stat, blocks in store.scores.items():
                    values = blocks[pair].ravel()
                    idx = np.flatnonzero(keep & np.isfinite(values))
                    ascending = np.sort(values[idx])
                    rank = np.zeros(len(values), dtype=np.int64)
                    rank[idx] = len(idx) + 1 - np.searchsorted(ascending, values[idx], side="right")
                    ranks[stat], sizes[stat] = rank, len(idx)
                a, b = [store.maps["ABC".index(c)] for c in pair]
                cols = store.provenance.get("legacy_current_columns", {})
                for i, x in enumerate(a.reference_positions):
                    if x is None:
                        continue
                    for j, y in enumerate(b.reference_positions):
                        if y is None:
                            continue
                        index = i * b.n_sites + j
                        label = labels.get((pair, x, y))
                        row = {
                            "pair": pair,
                            "x_reference_id": a.reference_id,
                            "x_current_col": cols[a.component][i] if a.component in cols else i + 1,
                            "x_original_col": a.original_columns[i] + 1,
                            "x_reference_pos": x,
                            "y_reference_id": b.reference_id,
                            "y_current_col": cols[b.component][j] if b.component in cols else j + 1,
                            "y_original_col": b.original_columns[j] + 1,
                            "y_reference_pos": y,
                            "eligible": int(keep[index])
                            if store.n_valid[pair][i, j] >= 0
                            else None,
                            "support_status": "known"
                            if store.n_valid[pair][i, j] >= 0
                            else "unreported_legacy_support",
                            "n_valid": store.n_valid[pair][i, j]
                            if store.n_valid[pair][i, j] >= 0
                            else None,
                            "valid_weight": store.valid_weight[pair][i, j]
                            if store.n_valid[pair][i, j] >= 0
                            else None,
                            "contact": label.contact if label else None,
                            "label_status": label.status if label else "unlabeled",
                            "label_source": label.source if label else "",
                            "label_sha256": digest,
                            "distance_angstrom": label.distance if label else None,
                            "cluster_id": label.cluster if label else "",
                            "split": label.split if label else "",
                            "score_uncertainty": "unavailable",
                        }
                        for stat, blocks in store.scores.items():
                            value = blocks[pair][i, j]
                            if np.isfinite(value):
                                row[stat] = float(value)
                            rank = int(ranks[stat][index])
                            if rank:
                                row[stat + "_competition_rank"] = rank
                                row[stat + "_rank_percentile"] = 1 - (rank - 1) / max(
                                    1, sizes[stat] - 1
                                )
                        writer.writerow(row)
        os.link(temp, path)
