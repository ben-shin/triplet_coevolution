"""Time weighting and permutation calibration on nested synthetic or prepared matched alignments."""

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def fixture(n, lengths, seed):
    from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch

    seeds = np.random.SeedSequence(seed).spawn(4)
    families = np.random.default_rng(seeds[0]).integers(64, size=n)
    comps = []
    for index, (molecule, symbols, length) in enumerate(
        zip(
            ("protein", "dna", "rna"),
            ("ACDEFGHIKLMNPQRSTVWY", "ACGT", "ACGU"),
            lengths,
            strict=True,
        )
    ):
        template_rng, residue_rng, change_rng = (
            np.random.default_rng(s) for s in seeds[index + 1].spawn(3)
        )
        letters = np.array(list(symbols + "-?"))
        template = template_rng.integers(len(symbols), size=(64, length))
        states = template[families]
        substitutions = residue_rng.integers(len(symbols), size=states.shape)
        action = change_rng.random(states.shape)
        states = np.where(action < 0.10, substitutions, states)
        states[(action >= 0.10) & (action < 0.16)] = len(symbols)
        states[(action >= 0.16) & (action < 0.18)] = len(symbols) + 1
        # Guarantee one canonical site while keeping the other columns variable.
        states[:, 0] = template[families, 0]
        name = "ABC"[index]
        ids = tuple(f"{name}{i}" for i in range(n))
        comps.append(
            ComponentMSA(name, molecule, ids, tuple("".join(row) for row in letters[states]))
        )
    matches = tuple(TripletMatch(str(i), *(comp.ids[i] for comp in comps)) for i in range(n))
    return MatchedTripletMSA(*comps, matches), [str(i % 8) for i in families]


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def worker(args):
    from _resources import peak_resident_bytes

    from triplet_coevolution.core.coordinates import build_reference_map
    from triplet_coevolution.inference.advanced import write_advanced_results
    from triplet_coevolution.inference.mi import triplet_mi
    from triplet_coevolution.inference.weights import sequence_weights

    n = args.worker
    if args.prepared:
        from _matched_inputs import prepared_subset

        msa, maps, strata, settings = prepared_subset(
            args.prepared, n, max(args.sizes), args.seed, args.strata_column
        )
    else:
        msa, strata = fixture(n, args.lengths, args.seed)
        maps = tuple(build_reference_map(c, c.ids[0]) for c in msa.components)
        settings = {
            "identity_threshold": 0.8,
            "min_overlap": 0.5,
            "min_valid": 2,
            "score": "mi_apc",
            "require_reference": False,
            "strata_column": "synthetic_group",
        }
    for comp in msa.components:
        _ = comp.matrix
    weight_options = {name: settings[name] for name in ("identity_threshold", "min_overlap")}
    score_options = {name: settings[name] for name in ("min_valid", "score", "require_reference")}
    w = sequence_weights(msa, **weight_options)
    mi = triplet_mi(msa, w, min_valid=settings["min_valid"]) if args.job == "permutation" else None
    times = []
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(args.repeats):
            start = time.perf_counter()
            if args.job == "weights":
                w = sequence_weights(msa, **weight_options)
            else:
                manifest = write_advanced_results(
                    tmp,
                    msa,
                    maps,
                    w,
                    mi_result=mi,
                    permutations=args.permutations,
                    threebody=True,
                    max_triplets=16,
                    seed=args.seed,
                    strata=strata,
                    max_pairs=1000,
                    **score_options,
                )
            times.append(time.perf_counter() - start)
        signature = {
            "counts_sha256": digest(w.neighbor_counts.astype("<i8")),
            "weights_sha256": digest(w.weights.astype("<f8")),
            "settings": settings,
            "retained_sites": [c.n_sites for c in msa.components],
            "effective_depth": w.n_eff,
        }
        if args.job == "permutation":
            signature["tests"] = manifest["hypotheses"]
            for name in ("mi", "threebody"):
                with (Path(tmp) / f"permutation_{name}.tsv").open(
                    encoding="utf-8", newline=""
                ) as f:
                    rows = list(csv.DictReader(f, delimiter="\t"))
                values = np.array(
                    [[float(row[k]) for k in ("score", "permutation_p", "bh_q")] for row in rows],
                    dtype="<f8",
                ).reshape(-1, 3)
                np.save(Path(args.artifacts) / f"{args.variant}_{n}_{name}.npy", values)
                signature[name + "_finite"] = len(rows)
                signature[name + "_p_q_sha256"] = digest(values[:, 1:])
    weights_path = Path(sequence_weights.__code__.co_filename).resolve()
    print(
        json.dumps(
            {
                "n": n,
                "job": args.job,
                "weights_source": str(weights_path),
                "weights_source_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                "seconds": times,
                "median_seconds": statistics.median(times),
                "peak_rss_bytes": peak_resident_bytes(),
                "signature": signature,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 768, 2048])
    parser.add_argument("--lengths", nargs=3, type=int, default=[72, 30, 24])
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--prepared",
        type=Path,
        help="Prepared folder or analyze_args.json for a nested sweep of its matched observations",
    )
    parser.add_argument("--strata-column", help="Override the prepared match-table group column")
    parser.add_argument(
        "--baseline-src", type=Path, help="Optional src/ directory of a comparison revision"
    )
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--job", choices=("weights", "permutation"), help=argparse.SUPPRESS)
    parser.add_argument("--variant", help=argparse.SUPPRESS)
    parser.add_argument("--artifacts", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.sizes) < 2 or min(args.lengths) < 1 or min(args.repeats, args.permutations) < 1:
        parser.error("sizes must be >=2 and lengths, repeats, and permutations must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.strata_column and not args.prepared:
        parser.error("--strata-column requires --prepared")
    if args.worker is not None:
        try:
            worker(args)
        except (ValueError, OSError, csv.Error) as exc:
            parser.error(str(exc))
        return
    if (
        args.baseline_src
        and not (args.baseline_src / "triplet_coevolution/inference/weights.py").is_file()
    ):
        parser.error(
            "--baseline-src must contain the comparison package under triplet_coevolution/"
        )
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in args.sizes:
            for job in ("weights", "permutation"):
                per_variant = []
                for variant in ["baseline", "current"] if args.baseline_src else ["current"]:
                    env = os.environ.copy()
                    if variant == "baseline":
                        env["PYTHONPATH"] = os.pathsep.join(
                            [str(args.baseline_src.resolve()), env.get("PYTHONPATH", "")]
                        )
                    cmd = [
                        sys.executable,
                        __file__,
                        "--worker",
                        str(n),
                        "--job",
                        job,
                        "--variant",
                        variant,
                        "--artifacts",
                        tmp,
                        "--sizes",
                        *map(str, args.sizes),
                        "--lengths",
                        *map(str, args.lengths),
                        "--seed",
                        str(args.seed),
                        "--permutations",
                        str(args.permutations),
                        "--repeats",
                        str(args.repeats),
                    ]
                    if args.prepared:
                        cmd.extend(["--prepared", str(args.prepared.resolve())])
                    if args.strata_column:
                        cmd.extend(["--strata-column", args.strata_column])
                    try:
                        measured = json.loads(subprocess.check_output(cmd, env=env, text=True))
                    except subprocess.CalledProcessError as exc:
                        parser.exit(exc.returncode)
                    if variant == "baseline" and not Path(
                        measured["weights_source"]
                    ).is_relative_to(args.baseline_src.resolve()):
                        raise RuntimeError(
                            "baseline process imported weights outside --baseline-src"
                        )
                    print(
                        f"{variant:8s} {n:6d} {job:12s} {measured['median_seconds']:.4f} s",
                        file=sys.stderr,
                    )
                    per_variant.append(measured)
                    results.append({"version": variant, **measured})
                if len(per_variant) == 2:
                    if per_variant[0]["signature"] != per_variant[1]["signature"]:
                        raise RuntimeError(
                            f"statistical results differ for {job} at {n} observations"
                        )
                    if job == "permutation":
                        for name in ("mi", "threebody"):
                            old, new = (
                                np.load(Path(tmp) / f"{tag}_{n}_{name}.npy")
                                for tag in ("baseline", "current")
                            )
                            np.testing.assert_allclose(old[:, 0], new[:, 0], atol=1e-12, rtol=1e-12)
        print(
            json.dumps(
                {
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "platform": platform.platform(),
                    "prepared": str(args.prepared) if args.prepared else None,
                    "sizes": args.sizes,
                    "lengths": None if args.prepared else args.lengths,
                    "permutations": args.permutations,
                    "repeats": args.repeats,
                    "seed": args.seed,
                    "results": results,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
