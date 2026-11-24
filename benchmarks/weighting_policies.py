"""Weight policies under biased replication and an optional held-out GatABC taxon split.

The fixed analysis plan and interpretation are in weighting-policies.md. Prints only aggregates.
"""

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from _resources import peak_resident_bytes

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.weights import sequence_weights
from triplet_coevolution.io.alignments import read_msa
from triplet_coevolution.io.metadata import read_triplet_table
from triplet_coevolution.msa.masking import apply_triplet_masks, column_mask
from triplet_coevolution.msa.qc import triplet_qc
from triplet_coevolution.msa.synchronize import synchronize


def strategies(ids, labels, supplied):
    yield "unit", {"policy": "supplied", "supplied_weights": dict.fromkeys(ids, 1.0)}
    yield "equal_component", {}
    yield "concatenated", {"policy": "concatenated"}
    for part in "ABC":
        yield "component_" + part, {"policy": "component", "component": part}
    yield "representatives", {"policy": "cluster_representatives"}
    yield (
        "group_balanced",
        {"policy": "clade_balanced", "clades": dict(zip(ids, labels, strict=True))},
    )
    if supplied is not None:
        yield (
            "design",
            {"policy": "supplied", "supplied_weights": dict(zip(ids, supplied, strict=True))},
        )


def interval(x, draws):
    x = np.asarray(x)
    return [float(x.mean()), *map(float, np.quantile(x[draws].mean(axis=1), [0.025, 0.975]))]


def simulated():
    summary = defaultdict(list)
    units = []
    n_rows = []
    for seed in range(8201, 8233):
        rng = np.random.default_rng(seed)
        founders = 40
        outcome = rng.binomial(1, 0.35, size=founders)
        replicates = np.where(outcome == 1, 8, 2)
        group = np.repeat(np.arange(founders), replicates)
        n = len(group)
        n_rows.append(n)
        ids = tuple(f"obs{i:04}" for i in range(n))
        comps = []
        for role, molecule, length, rate in zip(
            "ABC", ("protein", "dna", "rna"), (40, 8, 6), (0.05, 0.12, 0.18)
        ):
            letters = np.array(list(CANONICAL[molecule]))
            templates = rng.choice(letters, (founders, length))
            mat = templates[group]
            change = rng.random(mat.shape) < rate
            mat[change] = rng.choice(letters, change.sum())
            comps.append(ComponentMSA(role, molecule, ids, tuple("".join(row) for row in mat)))
        rows = tuple(TripletMatch(key, key, key, key) for key in ids)
        msa = MatchedTripletMSA(*comps, rows)
        target = rng.binomial(1, 0.35, size=512).mean()
        for name, opts in strategies(ids, list(map(str, group)), 1.0 / replicates[group]):
            start = perf_counter()
            w = sequence_weights(msa, **opts)
            elapsed = perf_counter() - start
            estimate = np.average(outcome[group], weights=w.weights)
            summary[name].append(
                [abs(estimate - target), w.n_eff, w.kish, elapsed, w.preflight["peak_bytes"]]
            )
            if name == "unit":
                units.append(abs(estimate - target))
    draws = np.random.default_rng(48027).integers(32, size=(4000, 32))
    results = {}
    for name, data in summary.items():
        e, weight, kish, seconds, peak = np.array(data).T
        results[name] = {
            "absolute_error": interval(e, draws),
            "delta_from_unit": interval(e - units, draws),
            "mean_sum_weight": weight.mean(),
            "mean_kish": kish.mean(),
            "median_weight_seconds": np.median(seconds),
            "max_plan_bytes": int(peak.max()),
        }
    return {"founders": 40, "replicates": 32, "mean_n": np.mean(n_rows), "policies": results}


def taxon_test(root):
    manifest = json.loads(Path(__file__).with_name("gatabc_manifest.json").read_text())
    prep = manifest["paths"]["prepared_directory"]
    for rel in ["A.fasta", "B.fasta", "C.fasta", "analyze_args.json", "matches.tsv"]:
        path = root / prep / rel
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != manifest["supplied_sha256"][f"{prep}/{rel}"]:
            raise ValueError(f"input checksum mismatch: {rel}")
    table = read_triplet_table(root / prep / "matches.tsv")
    tx = [r["taxon_id"] for r in table.annotations]
    if any(not key.strip() for key in tx):
        raise ValueError("every observation requires a supplied taxon ID")
    parts = [read_msa(root / prep / f"{c}.fasta", c, "protein") for c in "ABC"]

    def subset(indices):
        return synchronize(*parts, tuple(table.matches[i] for i in indices))

    def masks(msa):
        return [column_mask(q) for q in triplet_qc(msa).components]

    full = subset(range(len(tx)))
    full_masked = apply_triplet_masks(full, *masks(full)).msa
    started = perf_counter()
    baseline = sequence_weights(full_masked)
    baseline_time = perf_counter() - started
    if baseline.n_eff != manifest["cohort"]["effective_observations"]:
        raise ValueError("default GatABC weight sum differs from the legacy baseline")
    # Assignment uses only group identity and a fixed pre-fit salt.
    held = [
        i
        for i, t in enumerate(tx)
        if int.from_bytes(hashlib.sha256(("gatabc-phase3-v1/" + t).encode()).digest()[:8], "big")
        % 5
        == 0
    ]
    held_set = set(held)
    train = [i for i in range(len(tx)) if i not in held_set]
    training = subset(train)
    keep = masks(training)
    train_msa = apply_triplet_masks(training, *keep).msa
    test = apply_triplet_masks(subset(held), *keep).msa
    ids = [m.observation_id for m in train_msa.matches]
    labels = [tx[i] for i in train]
    test_taxa = [tx[i] for i in held]
    taxa = sorted(set(test_taxa))
    assert not set(labels).intersection(taxa)
    draws = np.random.default_rng(38427).integers(len(taxa), size=(4000, len(taxa)))
    results = {}
    units = None
    for name, opts in strategies(ids, labels, None):
        started = perf_counter()
        w = sequence_weights(train_msa, **opts)
        seconds = perf_counter() - started
        # Equalize total scale before adding a common prior to each state. Same
        # test sites, smoothing and assessment rows across competing estimators.
        design = w.weights / w.n_eff * len(ids)
        sums = np.zeros(len(held))
        counts = np.zeros(len(held), dtype=int)
        for tr, te in zip(train_msa.components, test.components):
            q = len(CANONICAL[tr.molecule])
            for j, col in enumerate(tr.matrix.T):
                ok = (col >= 0) & (col < q)
                freq = np.bincount(col[ok], weights=design[ok], minlength=q) + 0.5
                losses = -np.log2(freq / freq.sum())
                symbols = te.matrix[:, j]
                valid = (symbols >= 0) & (symbols < q)
                sums[valid] += losses[symbols[valid]]
                counts += valid
        if np.any(counts == 0):
            raise ValueError("no held-out canonical observations for at least one row")
        by_row = sums / counts
        by_taxon = np.array([np.mean(by_row[np.array(test_taxa) == t]) for t in taxa])
        if name == "unit":
            units = by_taxon.copy()
        results[name] = {
            "bits_per_symbol": interval(by_taxon, draws),
            "paired_delta": interval(by_taxon - units, draws),
            "train_weight_sum": w.n_eff,
            "train_kish": w.kish,
            "weight_seconds": seconds,
            "max_plan_bytes": w.preflight["peak_bytes"],
        }
    return {
        "baseline_sum": baseline.n_eff,
        "baseline_kish": baseline.kish,
        "baseline_times": baseline_time,
        "baseline_workspace": baseline.preflight,
        "n": len(tx),
        "taxa": len(set(tx)),
        "train_n": len(train),
        "held_n": len(held),
        "held_taxa": len(taxa),
        "train_lengths": [c.n_sites for c in train_msa.components],
        "held_canonical_symbols": int(sum(counts)),
        "policies": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gatabc-root", type=Path)
    args = parser.parse_args()
    started = perf_counter()
    result = {"simulation": simulated()}
    if args.gatabc_root:
        result["gatabc"] = taxon_test(args.gatabc_root)
    result["runtime"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "wall_seconds": perf_counter() - started,
        "peak_process_bytes": peak_resident_bytes(),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
