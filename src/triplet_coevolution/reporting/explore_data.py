"""Bounded summaries of archived scores. No inference or model fitting occurs here."""

import csv
import json
from pathlib import Path, PureWindowsPath

import numpy as np

PAIRS = ("AB", "AC", "BC")
GRID = 28


def artifact(directory, relative):
    """Resolve only local children, including when rendering on another platform."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("expected a relative report artifact path")
    path = Path(relative)
    if (
        path.is_absolute()
        or PureWindowsPath(relative).drive
        or ".." in path.parts
        or ":" in relative
        or path.as_posix() != relative
    ):
        raise ValueError(f"invalid report artifact path {relative!r}")
    resolved = (directory / path).resolve()
    if not resolved.is_relative_to(directory.resolve()):
        raise ValueError(f"report artifact outside output directory: {relative!r}")
    return resolved


def records(directory, relative):
    path = artifact(directory, relative)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="," if path.suffix.lower() == ".csv" else "\t")


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def write_label_scores(directory, labels, store):
    """Stream one row per supplied label, preserving even masked or unmapped positions.

    The source is the checksummed label snapshot already validated by benchmark_scores.
    Join by role, exact reference identity and position and respect method-specific support.
    """
    relative = "report_assets/labeled_scores.tsv"
    path = artifact(directory, relative)
    path.parent.mkdir(exist_ok=True)
    axes = {
        m.component: {
            (m.reference_id, p): (i, int(c) + 1)
            for i, (p, c) in enumerate(zip(m.reference_positions, m.original_columns, strict=True))
            if p is not None
        }
        for m in store.maps
    }
    stats = store.scores.keys()
    fields = [
        "pair",
        "x_reference_id",
        "x_reference_pos",
        "x_original_col",
        "y_reference_id",
        "y_reference_pos",
        "y_original_col",
        "contact",
        "label_status",
        "split",
        "label_source",
        "distance_angstrom",
        "n_valid",
        "valid_weight",
    ]
    if store.dca_counts is not None:
        fields += ["dca_n_valid", "dca_valid_weight"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[*fields, *stats], delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in records(directory, labels):
            x, y = sorted("xy", key=lambda k: row[k + "_component"])
            pair = row[x + "_component"] + row[y + "_component"]
            if pair not in PAIRS:
                raise ValueError("invalid benchmark label interface")
            out = {
                "pair": pair,
                "contact": row["contact"],
                "label_status": row.get("status") or ("resolved" if row["contact"] else "unknown"),
                "split": row.get("split") or "unspecified",
                "label_source": row.get("source", ""),
                "distance_angstrom": row.get("distance_angstrom", ""),
            }
            indices = []
            for role, side, source in zip(pair, "xy", (x, y), strict=True):
                ref = row[source + "_reference_id"]
                pos = int(row[source + "_reference_pos"])
                out[side + "_reference_id"] = ref
                out[side + "_reference_pos"] = pos
                if coord := axes[role].get((ref, pos)):
                    indices.append(coord[0])
                    out[side + "_original_col"] = coord[1]
            if len(indices) == 2:
                ij = tuple(indices)
                for stat in stats:
                    ns, ws = store.support(pair, stat)
                    n = int(ns[ij])
                    w = finite(ws[ij])
                    prefix = (
                        "dca_" if store.dca_counts is not None and stat.startswith("dca_") else ""
                    )
                    out[prefix + "n_valid"] = n if n >= 0 else None
                    out[prefix + "valid_weight"] = w
                    if n >= store.min_valid and w is not None and w >= store.min_valid_weight:
                        out[stat] = finite(store.scores[stat][pair][ij])
            writer.writerow(out)
    return relative


def _slices(shape):
    rows = max(1, 65536 // shape[1])
    for start in range(0, shape[0], rows):
        yield slice(start, min(shape[0], start + rows))


def _axes(store, pair):
    x, y = (store.maps["ABC".index(role)] for role in pair)
    return x, y


def _gate(store, name, pair, block, axes):
    count, weight = store.support(pair, name)
    keep = (count[block] >= store.min_valid) & (weight[block] >= store.min_valid_weight)
    if store.require_reference:
        keep &= axes[0][block, None] & axes[1][None, :]
    return keep


def axis_bins(axis):
    n = len(axis.original_columns)
    bins = min(n, GRID)
    index = np.arange(n, dtype=np.int64) * bins // n
    labels = []
    for start, stop in zip(
        np.searchsorted(index, np.arange(bins)),
        np.searchsorted(index, np.arange(1, bins + 1)),
        strict=True,
    ):
        lo, hi = int(axis.original_columns[start]) + 1, int(axis.original_columns[stop - 1]) + 1
        labels.append(str(lo) if lo == hi else f"{lo}–{hi}")
    return index, labels


def score_grid(store, pair, name):
    matrix = store.scores[name][pair]
    x, y = _axes(store, pair)
    rx, ry = (np.array([p is not None for p in axis.reference_positions]) for axis in (x, y))
    ix, xlabels = axis_bins(x)
    iy, ylabels = axis_bins(y)
    shape = len(xlabels), len(ylabels)
    total = np.zeros(np.prod(shape), dtype=float)
    counts = np.zeros(total.shape, dtype=np.int64)
    high = 0.0
    support_lo = support_hi = None
    for block in _slices(matrix.shape):
        values = matrix[block]
        keep = _gate(store, name, pair, block, (rx, ry)) & np.isfinite(values)
        if not keep.any():
            continue
        a, b = np.nonzero(keep)
        bins = ix[a + block.start] * shape[1] + iy[b]
        v = values[a, b]
        high = max(high, float(np.abs(v).max()))
        total += np.bincount(bins, weights=v, minlength=len(total))
        counts += np.bincount(bins, minlength=len(total))
        n = store.support(pair, name)[0][block][a, b]
        lo, hi = int(n.min()), int(n.max())
        support_lo = lo if support_lo is None else min(support_lo, lo)
        support_hi = hi if support_hi is None else max(support_hi, hi)
    grid = np.divide(total, counts, out=np.full_like(total, np.nan), where=counts > 0).reshape(
        shape
    )
    return {
        "values": grid,
        "counts": counts.reshape(shape),
        "x": xlabels,
        "y": ylabels,
        "total": matrix.size,
        "finite": int(counts.sum()),
        "max_abs": high,
        "shape": matrix.shape,
        "support": (support_lo, support_hi),
    }


def common_ranks(store, pair, mi_name, dca_name, limit=192):
    """Rank a systematic display sample against *all* common eligible finite cells.

    Equal scores receive competition rank 1 + number strictly greater. The bounded
    per-tile sorts never allocate a ranking vector of the whole interface.
    """
    mx, dx = (store.scores[k][pair] for k in (mi_name, dca_name))
    x, y = _axes(store, pair)
    numbered = tuple(np.array([p is not None for p in axis.reference_positions]) for axis in (x, y))

    def chunks():
        for part in _slices(mx.shape):
            ok = _gate(store, mi_name, pair, part, numbered) & _gate(
                store, dca_name, pair, part, numbered
            )
            ok &= np.isfinite(mx[part]) & np.isfinite(dx[part])
            yield part, ok

    total = sum(int(ok.sum()) for _, ok in chunks())
    if not total:
        return {"common": 0, "points": []}
    selected = np.linspace(0, total - 1, min(limit, total), dtype=np.int64)
    values = [[], []]
    positions = []
    offset = 0
    for part, ok in chunks():
        a, b = np.nonzero(ok)
        chosen = selected[(selected >= offset) & (selected < offset + len(a))] - offset
        a, b = a[chosen] + part.start, b[chosen]
        positions.extend(
            (int(x.original_columns[i]) + 1, int(y.original_columns[j]) + 1)
            for i, j in zip(a, b, strict=True)
        )
        values[0].extend(mx[a, b])
        values[1].extend(dx[a, b])
        offset += int(ok.sum())
    ranks = np.ones((2, len(selected)), dtype=np.int64)
    for part, ok in chunks():
        for k, matrix in enumerate((mx, dx)):
            ordered = np.sort(matrix[part][ok])
            ranks[k] += len(ordered) - np.searchsorted(ordered, values[k], side="right")
    return {
        "common": total,
        "points": [(*pos, int(rm), int(rd)) for pos, rm, rd in zip(positions, *ranks, strict=True)],
    }


def benchmark_preview(result):
    """Summarize already measured discrimination curves without recomputing the benchmark."""
    items = []
    for statistic, method in result.get("statistics", {}).items():
        for scope, metrics in {
            **method.get("interfaces", {}),
            "Pooled": method.get("pooled", {}),
        }.items():
            curves = metrics.get("curves", {})
            fpr, recall, precision = (
                curves.get(k) for k in ("false_positive_rate", "recall", "precision")
            )
            n = len(recall or [])
            take = np.linspace(0, n - 1, min(128, n), dtype=int) if n else []
            roc = [[float(fpr[i]), float(recall[i])] for i in take] if fpr is not None else []
            pr = (
                [[float(recall[i]), float(precision[i])] for i in take]
                if precision is not None
                else []
            )
            items.append(
                {
                    "statistic": statistic,
                    "scope": scope,
                    "status": metrics.get("class_support", metrics.get("status")),
                    "auroc": metrics.get("auroc"),
                    "ap": metrics.get("average_precision"),
                    "positive": metrics.get("positives_scored"),
                    "negative": metrics.get("negatives_scored"),
                    "thresholds": len(curves.get("thresholds", [])),
                    "roc": roc,
                    "pr": pr,
                    "top_k": metrics.get("top_k", {}),
                    "distance": metrics.get("distance", {}),
                    "uncertainty": metrics.get("uncertainty", {}),
                }
            )
    return {"schema_version": 1, "methods": items}


def saved_benchmark(directory, outputs):
    if path := outputs.get("benchmark_display"):
        return json.loads(artifact(directory, path).read_text(encoding="utf-8"))
    if path := outputs.get("benchmark"):
        source = artifact(directory, path)
        if source.stat().st_size <= 16 * 1024**2:
            return benchmark_preview(json.loads(source.read_text(encoding="utf-8")))
        return {
            "methods": [],
            "reason": "This older benchmark stores its curves in the complete JSON download. The report preview reader is limited to 16 MiB.",
        }
    return {"methods": [], "reason": "This report has no saved label benchmark."}
