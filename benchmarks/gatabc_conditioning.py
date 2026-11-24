"""Prespecified retrospective DCA sensitivity on a bounded private GatABC panel."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from triplet_coevolution.cli import analyze
from triplet_coevolution.dca_compare import compare_dca
from triplet_coevolution.io.score_store import file_digest


def benchmark(root, out, panel="availability"):
    root, out = Path(root), Path(out)
    if out.exists():
        raise ValueError("choose a new output path")
    manifest = json.loads(
        Path(__file__).with_name("gatabc_manifest.json").read_text(encoding="utf8")
    )
    prep = root / manifest["paths"]["prepared_directory"]
    labels = root / manifest["paths"]["labels"]
    source = {c: prep / f"{c}.fasta" for c in "ABC"}
    matches = prep / "matches.tsv"
    for file in (*source.values(), labels, matches):
        if file_digest(file) != manifest["supplied_sha256"][file.relative_to(root).as_posix()]:
            raise ValueError(f"historical file digest differs for {file.name}")
    counts = {c: Counter() for c in "ABC"}
    classes = {}
    with labels.open(encoding="utf8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row["x_component"], row["y_component"], row["contact"])
            classes.setdefault(key, []).append(
                (
                    row["x_reference_id"],
                    int(row["x_reference_pos"]),
                    row["y_reference_id"],
                    int(row["y_reference_pos"]),
                )
            )
            for p in ("x", "y"):
                c = row[p + "_component"]
                counts[c][row[p + "_reference_id"], int(row[p + "_reference_pos"])] += 1
    # Ascertain on availability alone. Neither the class nor sequence score enters the rule.
    selected = {c: sorted(v, key=lambda k: (-v[k], k))[:6] for c, v in counts.items()}
    if panel == "balanced":
        by_role = {c: set() for c in "ABC"}
        for (x, y, _), rows in classes.items():
            for xi, xp, yi, yp in sorted(
                rows, key=lambda row: hashlib.sha256(json.dumps(row).encode()).digest()
            )[:2]:
                by_role[x].add((xi, xp))
                by_role[y].add((yi, yp))
        selected = {c: sorted(v) for c, v in by_role.items()}
    out.mkdir(parents=True)
    with (out / "panel.tsv").open("w", encoding="utf8", newline="") as f:
        wr = csv.writer(f, delimiter="\t", lineterminator="\n")
        wr.writerow(("component", "reference_id", "reference_position"))
        for c, sites in selected.items():
            for ref, pos in sites:
                wr.writerow((c, ref, pos))
    refs = {c: v[0][0] for c, v in selected.items()}
    (out / "qc.json").write_text(
        json.dumps(
            {
                "order": "columns_first",
                "iteration": "fixed_point",
                "rows": {c: {"min_canonical": 1} for c in "ABC"},
            }
        ),
        encoding="utf8",
    )
    analyze(
        a=source["A"],
        b=source["B"],
        c=source["C"],
        matches=matches,
        a_ref=refs["A"],
        b_ref=refs["B"],
        c_ref=refs["C"],
        site_panel=out / "panel.tsv",
        out=out / "source",
        weight_policy="uniform",
        qc_policy=out / "qc.json",
        max_gap_fraction=1.0,
        max_missing_fraction=1.0,
        min_coverage=0.0,
    )
    models = {
        f"mf-{scope}-{v}": {"backend": "mf", "conditioning": scope, "ridge": v}
        for scope in ["joint", "pairwise"]
        for v in [0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
    }
    models.update(
        {
            f"plm-{scope}": {
                "backend": "plm",
                "conditioning": scope,
                "plm": {"lambda_h": 0.01, "lambda_j": 0.01, "workers": 1},
            }
            for scope in ["joint", "pairwise"]
        }
    )
    return compare_dca(out / "source", out / "comparison", {"models": models}, labels=labels)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("out", type=Path)
    cli.add_argument("--panel", choices=("availability", "balanced"), default="availability")
    args = cli.parse_args()
    benchmark(args.root, args.out, args.panel)
