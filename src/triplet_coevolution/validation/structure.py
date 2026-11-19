"""Benchmark site scores against explicitly labelled reference-coordinate contacts."""

import csv
from itertools import groupby
from pathlib import Path

import numpy as np


def read_contacts(path, maps):
    references = {mapping.component: mapping.reference_id for mapping in maps}
    fields = {
        f"{side}_{field}"
        for side in "xy"
        for field in ("component", "reference_id", "reference_pos")
    } | {"contact"}
    labels = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        if not reader.fieldnames or set(reader.fieldnames) != fields or len(reader.fieldnames) != 7:
            raise ValueError("contact TSV requires " + ", ".join(sorted(fields)))
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"malformed contact row at line {reader.line_num}")
            sides = []
            for side in "xy":
                name = row[f"{side}_component"]
                if name not in references or row[f"{side}_reference_id"] != references[name]:
                    raise ValueError(
                        f"unknown contact component/reference at line {reader.line_num}"
                    )
                pos = int(row[f"{side}_reference_pos"])
                if pos < 1:
                    raise ValueError("contact reference positions must be positive")
                sides.append((name, pos))
            (a, x), (b, y) = sorted(sides)
            if a == b or row["contact"] not in {"0", "1"}:
                raise ValueError("contact rows require different components and contact 0 or 1")
            key = (a + b, x, y)
            if key in labels:
                raise ValueError(f"duplicate labelled contact {key}")
            labels[key] = int(row["contact"])
    if not labels:
        raise ValueError("contact table has no labelled pairs")
    return labels


def contact_metrics(labels, scores):
    """Compute tie-aware AUROC and noninterpolated average precision on labelled pairs."""
    pairs = [
        (scores[key], label)
        for key, label in labels.items()
        if key in scores and np.isfinite(scores[key])
    ]
    pairs.sort(reverse=True)
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    tp = fp = 0
    auc = ap = 0.0
    for _, tied in groupby(pairs, key=lambda item: item[0]):
        group = [label for _, label in tied]
        hit = sum(group)
        miss = len(group) - hit
        auc += miss * (tp + 0.5 * hit)
        tp += hit
        fp += miss
        ap += hit * tp / (tp + fp)
    return {
        "labels": len(labels),
        "scored": len(pairs),
        "unscored": len(labels) - len(pairs),
        "positives_scored": positives,
        "negatives_scored": negatives,
        "auroc": auc / (positives * negatives) if positives and negatives else None,
        "average_precision": ap / positives if positives else None,
    }
