"""Replay the registered Poelwijk 2019 assay. Output contains aggregate results only."""

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

DOI = "10.1038/s41467-019-12130-8"
SOURCE = "https://pmc-oa-opendata.s3.amazonaws.com/PMC6746860.1/"
SOURCES = {
    "41467_2019_12130_MOESM7_ESM.xlsx": "d893d36645d10334b326696e98edbb7222b15bbeb946176482ed83637120904b",
    "41467_2019_12130_MOESM8_ESM.xlsx": "5674c3caecf4f777c7b203b135ac4e4607dad2e87433aec16c91835fbbbb85cb",
}
REGISTRATIONS = {
    "experimental_protocol.json": "14d58d27f6b6f4730b493a66ac3c3df1fac25ae01b5c1a4ab733b943d3f42b24",
    "independent_evaluation_policy.json": "5e415125959c925e8408994e638220eb9a559d6a0ded3c69a8336780a85186d3",
}
PRIMARY = {
    "A": ("binary", "genotype"),
    "C": ("counts", "input"),
    "D": ("counts", "red"),
    "E": ("counts", "blue"),
    "J": ("brightness", "combined"),
}
REPEATS = {
    "A": ("binary", "genotype"),
    "C": ("brightness", "trial1"),
    "E": ("brightness", "trial2"),
}


def source_files(directory, fetch):
    directory = Path(directory)
    if fetch:
        directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, digest in SOURCES.items():
        path = directory / name
        if not path.exists() and fetch:
            with urllib.request.urlopen(SOURCE + name, timeout=90) as response:
                content = response.read(2_000_001)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"download hash mismatch for {name}")
            with path.open("xb") as out:
                out.write(content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"source hash mismatch for {name}")
        paths.append(path)
    return paths


def read_workbook(path, specs, nbits=13):
    """Read specified columns only. Leave missing numeric cells as NaN."""
    size = 2**nbits
    values = {key: np.full(size, np.nan) for key in specs if key != "A"}
    seen = set()
    with zipfile.ZipFile(path) as z:
        # These are fixed publisher files. Cell formulas are never executed.
        strings = ["".join(s.itertext()) for s in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        header = {}
        for row in sheet.findall(".//{*}row"):
            index = int(row.get("r"))
            cells = {}
            for cell in row:
                column = re.sub(r"\d+$", "", cell.get("r"))
                if column not in specs:
                    continue
                value = cell.findtext("{*}v")
                cells[column] = strings[int(value)] if cell.get("t") == "s" else value
            if index in (1, 2):
                header[index] = cells
                continue
            if not any(x is not None for x in cells.values()):
                continue
            bits = cells.get("A", "")
            if not isinstance(bits, str) or not re.fullmatch(f"'[01]{{{nbits}}}'", bits):
                raise ValueError(f"missing or malformed fixed-width genotype at row {index}")
            genotype = int(bits[1:-1], 2)
            if genotype in seen:
                raise ValueError(f"duplicate genotype at row {index}")
            seen.add(genotype)
            for key, array in values.items():
                value = cells.get(key)
                if value is not None:
                    array[genotype] = float(value)
        if any(
            (header.get(1, {}).get(k), header.get(2, {}).get(k)) != pair
            for k, pair in specs.items()
        ):
            raise ValueError("unexpected source columns")
    if len(seen) != size:
        raise ValueError(f"incomplete genotype cube: {len(seen)}/{size}")
    return values


def cube_design(nbits, max_degree):
    """Rows follow printed fixed-width binary order. A zero allele codes as +1."""
    if not 1 <= max_degree <= nbits <= 13:
        raise ValueError("invalid cube dimensions")
    strings = [format(i, f"0{nbits}b") for i in range(2**nbits)]
    bits = np.array([[int(x) for x in s] for s in strings])
    z = 1.0 - 2.0 * bits
    terms = [idx for k in range(max_degree + 1) for idx in itertools.combinations(range(nbits), k)]
    x = np.column_stack([z[:, idx].prod(axis=1) for idx in terms])
    folds = np.array(
        [int.from_bytes(hashlib.sha256(s.encode("ascii")).digest(), "big") % 5 for s in strings]
    )
    return x, terms, folds, bits.sum(axis=1)


def walsh(y):
    """Uniform factorial E[y * product(1 - 2b_j)], indexed by bit subset mask."""
    means = np.array(y, dtype=np.float64, copy=True)
    size = len(means)
    if means.ndim != 1 or size < 2 or size & (size - 1) or not np.isfinite(means).all():
        raise ValueError("Walsh analysis requires a complete finite binary cube")
    step = 1
    while step < size:
        pairs = means.reshape(-1, 2, step)
        first, second = pairs[:, 0, :].copy(), pairs[:, 1, :].copy()
        pairs[:, 0, :], pairs[:, 1, :] = first + second, first - second
        step *= 2
    return means / size


def order_variance(y):
    coef = walsh(y)
    squared = np.bincount([i.bit_count() for i in range(len(y))], weights=coef**2)
    total = float(np.var(y))
    if not np.isclose(squared[1:].sum(), total, rtol=1e-12, atol=1e-14):
        raise ArithmeticError("orthogonal variance totals disagree")
    return {
        "mean": float(coef[0]),
        "variance": total,
        "squared_coefficients_by_order": squared[1:].tolist(),
        "variance_fractions_by_order": (squared[1:] / total).tolist() if total else None,
    }


def metrics(y, prediction):
    residual = float(np.mean((y - prediction) ** 2))
    variance = float(np.var(y))
    return {
        "n": len(y),
        "rmse": math.sqrt(residual),
        "r_squared": 1 - residual / variance if variance else None,
    }


def heldout_predictions(x, y, folds):
    """Fit all requested response columns independently, sharing only the fixed design."""
    if len(x) != len(y) or len(x) != len(folds) or set(folds) != set(range(5)):
        raise ValueError("five nonempty matching folds required")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("incomplete response or design")
    result = np.empty_like(y, dtype=float)
    for fold in range(5):
        test = folds == fold
        coef, _, rank, _ = np.linalg.lstsq(x[~test], y[~test], rcond=None)
        if rank != x.shape[1]:
            raise ValueError(f"training design for fold {fold} lacks full column rank")
        result[test] = x[test] @ coef
    return result


def evaluate(primary, trials, nbits=13):
    x, terms, folds, strata = cube_design(nbits, 3)
    outcomes = {"primary": primary["J"], "data4_trial1": trials["C"], "data4_trial2": trials["E"]}
    data = np.column_stack(list(outcomes.values()))
    statistics = {
        name: {
            "missing_or_nonfinite": int((~np.isfinite(v)).sum()),
            "exact_zero": int((v == 0).sum()),
            "minimum": float(np.min(v[np.isfinite(v)])) if np.isfinite(v).any() else None,
            "maximum": float(np.max(v[np.isfinite(v)])) if np.isfinite(v).any() else None,
            "at_minimum": int((v == v[np.isfinite(v)].min()).sum()) if np.isfinite(v).any() else 0,
        }
        for name, v in outcomes.items()
    }
    counts = np.column_stack([primary[k] for k in ("C", "D", "E")])
    if (
        not np.isfinite(counts).all()
        or (counts < 0).any()
        or not (counts == np.floor(counts)).all()
    ):
        raise ValueError("assay count fields must be finite nonnegative integers")
    summary = {
        "genotypes": len(folds),
        "fold_counts_by_hamming_weight": [
            np.bincount(folds[strata == k], minlength=5).tolist() for k in range(nbits + 1)
        ],
        "fold_sha256": hashlib.sha256(bytes(folds.astype(np.uint8))).hexdigest(),
        "outcome_support": statistics,
        "read_count_diagnostics": {
            "zero_input": int((counts[:, 0] == 0).sum()),
            "zero_red": int((counts[:, 1] == 0).sum()),
            "zero_blue": int((counts[:, 2] == 0).sum()),
            "zero_both_colors": int((counts[:, 1:] == 0).all(axis=1).sum()),
            "zero_all_three": int((counts == 0).all(axis=1).sum()),
        },
        "unavailable": "Per-genotype censor and pseudocount indicators were not supplied in the scored workbooks. Zero sequencing counts and a minimum processed response are separate observations.",
    }
    complete = np.isfinite(data).all(axis=0)
    if not complete.any():
        summary["status"] = "unavailable_no_complete_response"
        return summary
    labels = [name for name, finite in zip(outcomes, complete, strict=True) if finite]
    complete_y = data[:, complete]
    summary["analysis"] = {
        name: {"walsh": order_variance(outcomes[name]), "held_out_by_degree": {}} for name in labels
    }
    for degree in (1, 2, 3):
        prefix = sum(len(t) <= degree for t in terms)
        predicted = heldout_predictions(x[:, :prefix], complete_y, folds)
        for col, name in enumerate(labels):
            y = complete_y[:, col]
            p = predicted[:, col]
            summary["analysis"][name]["held_out_by_degree"][str(degree)] = {
                "coefficients_per_fit": prefix,
                **metrics(y, p),
                "fold_metrics": [metrics(y[folds == k], p[folds == k]) for k in range(5)],
            }
    for response in summary["analysis"].values():
        fits = response["held_out_by_degree"]
        paired = [
            (fits["2"], fits["3"]),
            *zip(fits["2"]["fold_metrics"], fits["3"]["fold_metrics"], strict=True),
        ]
        changes = [
            {
                "r_squared_gain": b["r_squared"] - a["r_squared"]
                if a["r_squared"] is not None and b["r_squared"] is not None
                else None,
                "rmse_reduction": a["rmse"] - b["rmse"],
            }
            for a, b in paired
        ]
        response["degree3_vs_2"] = {"pooled": changes[0], "folds": changes[1:]}
    if complete.all():
        summary["trial_comparison"] = {
            "trial1_vs_primary_max_absolute_difference": float(
                np.max(np.abs(trials["C"] - primary["J"]))
            ),
            "trial2_vs_trial1_rmse": metrics(trials["C"], trials["E"])["rmse"],
            "trial2_vs_trial1_pearson": float(np.corrcoef(trials["C"], trials["E"])[0, 1])
            if np.std(trials["C"]) and np.std(trials["E"])
            else None,
        }
    summary["status"] = "complete" if complete.all() else "partial_incomplete_response"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--download", action="store_true", help="Retrieve missing public measurement tables"
    )
    args = parser.parse_args()
    for name, expected in REGISTRATIONS.items():
        canonical = Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(canonical).hexdigest() != expected:
            raise ValueError(f"protocol hash mismatch for {name}")
    first, second = source_files(args.input_dir, args.download)
    report = evaluate(read_workbook(first, PRIMARY), read_workbook(second, REPEATS))
    report.update(
        doi=DOI,
        source_sha256=SOURCES,
        source_base_url=SOURCE,
        registrations_sha256_lf=REGISTRATIONS,
        numpy=np.__version__,
        python=sys.version.split()[0],
    )
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
