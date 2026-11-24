"""Replay the owner-supplied GatABC metrics on six complete legacy score tables."""

import argparse
import csv
import hashlib
import json
import math
from bisect import bisect_right
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

from triplet_coevolution.validation.structure import contact_metrics, read_contacts

MANIFEST = Path(__file__).with_name("gatabc_manifest.json")
INTERFACES = ("AB", "AC", "BC")
SCORES = {"mi": ("mi_bits", "mi_apc"), "dca": ("dca_frobenius", "dca_apc")}
SITE_FIELDS = {
    f"{side}_{field}"
    for side in "xy"
    for field in ("component", "reference_id", "reference_pos", "symbol")
}


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check(actual, expected, name):
    if actual != expected:
        raise ValueError(f"{name}: expected {expected!r}, observed {actual!r}")


class MetricDisagreement(ValueError):
    """The tables passed identity and rank checks but differed from the audit metrics."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("; ".join(report["metric_disagreements"]))


def _compare_metric(actual, expected, tolerance, name, disagreements):
    if actual is None or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        disagreements.append(f"{name}: expected {expected} +/- {tolerance}, observed {actual}")


def _read_table(path, pair, fields, manifest, labels, known_axes, labeled_scores):
    targets = {
        tuple(item["positions"]): item
        for item in manifest["reference_pairs"]
        if item["pair"] == pair
    }
    target_values = {}
    score_values = {field: [] for field in fields}
    seen = set()
    axes = {name: set() for name in pair}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        headers = reader.fieldnames or []
        needed = SITE_FIELDS | {"pair", *fields}
        if len(headers) != len(set(headers)) or not needed.issubset(headers):
            raise ValueError(
                f"{path.name}: invalid score header; required fields: {sorted(needed)}"
            )
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"{path.name}:{reader.line_num}: malformed score row")
            _check(row["pair"], pair, f"{path.name} interface")
            positions = []
            for side, component in zip("xy", pair, strict=True):
                _check(row[f"{side}_component"], component, f"{path.name} component")
                _check(
                    row[f"{side}_reference_id"],
                    manifest["references"][component],
                    f"{path.name} reference",
                )
                pos = int(row[f"{side}_reference_pos"])
                if pos < 1:
                    raise ValueError(f"{path.name}: invalid reference position {pos}")
                positions.append(pos)
                axes[component].add(pos)
            coord = tuple(positions)
            if coord in seen:
                raise ValueError(f"{path.name}: duplicate reference pair {coord}")
            seen.add(coord)
            key = (pair, *coord)
            for field in fields:
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError(f"{path.name}: nonfinite {field} at {coord}")
                score_values[field].append(value)
                if key in labels:
                    labeled_scores[field][key] = value
                if coord in targets:
                    target_values[coord, field] = value
            if coord in targets:
                _check(
                    [row["x_symbol"], row["y_symbol"]],
                    targets[coord]["symbols"],
                    f"{pair} {coord} reference symbols",
                )

    cohort = manifest["cohort"]
    _check(len(seen), cohort["reference_eligible_pairs"][pair], f"{path.name} eligible pairs")
    for component, axis in axes.items():
        _check(
            len(axis),
            cohort["reference_eligible_columns"][component],
            f"{path.name} eligible axis {component}",
        )
        if known_axes.setdefault(component, axis) != axis:
            raise ValueError(f"{path.name}: inconsistent reference-coordinate axis {component}")
    _check(len(seen), len(axes[pair[0]]) * len(axes[pair[1]]), f"{path.name} full rectangle")
    ranks = {}
    for field, values in score_values.items():
        values.sort()
        for coord, item in targets.items():
            if field not in item["ranks"]:
                continue
            if (coord, field) not in target_values:
                raise ValueError(f"{path.name}: missing regression pair {coord}")
            rank = 1 + len(values) - bisect_right(values, target_values[coord, field])
            _check(rank, item["ranks"][field], f"{pair} {coord} {field} competition rank")
            ranks.setdefault(f"{pair}:{coord[0]}-{coord[1]}", {})[field] = rank
    return {"rows": len(seen), "sha256": _digest(path), "checked_ranks": ranks}


def _checked_labels(root, manifest):
    paths = manifest["paths"]
    label_path = root / paths["labels"]
    digest = _digest(label_path)
    _check(digest, manifest["supplied_sha256"][paths["labels"]], "contact label SHA-256")
    maps = [SimpleNamespace(component=c, reference_id=r) for c, r in manifest["references"].items()]
    labels = read_contacts(label_path, maps)
    partition = {p: {k: v for k, v in labels.items() if k[0] == p} for p in INTERFACES}
    for pair, subset in partition.items():
        counts = manifest["labels"]["counts"][pair]
        _check(sum(subset.values()), counts["positive"], f"{pair} positive labels")
        _check(len(subset) - sum(subset.values()), counts["negative"], f"{pair} negative labels")
    for item in manifest["reference_pairs"]:
        key = (item["pair"], *item["positions"])
        label = {None: "unlabeled", 0: "negative", 1: "positive"}[labels.get(key)]
        _check(label, item["label"], f"{key} reference label")
    return digest, labels, partition


def _compact_scores(root, manifest, labels, scores):
    name = "regression/labeled_pairwise_scores.tsv"
    path = root / name
    digest = _digest(path)
    _check(digest, manifest["supplied_sha256"][name], "compact labeled score SHA-256")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        needed = {
            "pair",
            "x_reference_pos",
            "y_reference_pos",
            "contact",
            "method_family",
            "raw_score",
            "apc_score",
        }
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError("invalid compact labeled score header")
        for row in reader:
            key = (row["pair"], int(row["x_reference_pos"]), int(row["y_reference_pos"]))
            fields = SCORES.get(row["method_family"])
            if fields is None or key not in labels:
                raise ValueError(f"unrecognized method or coordinate in compact table at {key}")
            _check(int(row["contact"]), labels[key], f"{key} compact contact label")
            for field, column in zip(fields, ("raw_score", "apc_score"), strict=True):
                if key in scores[field]:
                    raise ValueError(f"duplicate compact score for {field} at {key}")
                value = float(row[column])
                if not math.isfinite(value):
                    raise ValueError(f"nonfinite compact score for {field} at {key}")
                scores[field][key] = value
    return {name: {"sha256": digest, "rows": sum(map(len, scores.values())) // 2}}


def validate_legacy_tables(
    root: Path, manifest: Mapping, *, recomputed_scores: Path | None = None, compact: bool = False
) -> dict:
    """Compare archived or recomputed score tables with the original labeled benchmark."""
    if compact and recomputed_scores is not None:
        raise ValueError("choose compact or recomputed scores")
    paths = manifest["paths"]
    digest, labels, partition = _checked_labels(root, manifest)
    scores = {f: {} for fields in SCORES.values() for f in fields}
    axes, tables = {}, {}
    if compact:
        tables = _compact_scores(root, manifest, labels, scores)
    else:
        for family, fields in SCORES.items():
            for pair in INTERFACES:
                name = f"{family}_{pair.lower()}.tsv"
                relative = Path(paths["full_score_directory"]) / name
                path = (
                    recomputed_scores / name if recomputed_scores is not None else root / relative
                )
                tables[name] = _read_table(path, pair, fields, manifest, labels, axes, scores)
                expected_digest = manifest["supplied_sha256"][relative.as_posix()]
                tables[name]["matches_archived_sha256"] = tables[name]["sha256"] == expected_digest
                if recomputed_scores is None:
                    _check(tables[name]["sha256"], expected_digest, f"{name} archived SHA-256")
    results = {}
    disagreements = []
    tolerances = manifest["metric_definitions"]["absolute_tolerances"]
    for field, labeled in scores.items():
        expected = manifest["metrics"][field]
        results[field] = {}
        for scope, subset in {**partition, "pooled": labels}.items():
            result = contact_metrics(subset, labeled)
            _check(result["scored"], len(subset), f"{field} {scope} finite labeled scores")
            tolerance = tolerances["pooled_or_macro" if scope == "pooled" else "interface"]
            for key, target in zip(("auroc", "average_precision"), expected[scope], strict=True):
                _compare_metric(
                    result[key], target, tolerance, f"{field} {scope} {key}", disagreements
                )
            results[field][scope] = result
        macro = sum(results[field][pair]["auroc"] for pair in INTERFACES) / len(INTERFACES)
        _compare_metric(
            macro,
            expected["macro_auroc"],
            tolerances["pooled_or_macro"],
            f"{field} macro AUROC",
            disagreements,
        )
        results[field]["macro_auroc"] = macro
    report = {
        "status": "metric_mismatch" if disagreements else "passed",
        "score_source": "compact_labeled"
        if compact
        else ("recomputed" if recomputed_scores is not None else "archived"),
        "label_sha256": digest,
        "score_tables": tables,
        "metrics": results,
        "metric_disagreements": disagreements,
    }
    if disagreements:
        raise MetricDisagreement(report)
    return report


def main() -> None:
    """Print a JSON replay result for an explicitly supplied private data root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing metadata/ and results/")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--recomputed-scores",
        type=Path,
        metavar="DIRECTORY",
        help="Check regenerated score tables; record byte differences from archived tables.",
    )
    source.add_argument(
        "--compact",
        action="store_true",
        help="Use the compact bundle's original labeled scores to verify AUROC/AP.",
    )
    args = parser.parse_args()
    try:
        result = validate_legacy_tables(
            args.root,
            json.loads(MANIFEST.read_text(encoding="utf-8")),
            recomputed_scores=args.recomputed_scores,
            compact=args.compact,
        )
    except MetricDisagreement as exc:
        print(json.dumps(exc.report, indent=2, allow_nan=False))
        parser.exit(1, f"GatABC numerical metric disagreement: {exc}\n")
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"GatABC score-table replay failed: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
