"""Time native kernels in isolated processes and verify results against a frozen checkout."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np


def digest(arrays):
    h = hashlib.sha256()
    for values in arrays:
        a = np.ascontiguousarray(values)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def v1_manifest(source):
    path = source if source.is_file() else source / "analyze_args.json"
    if not path.exists() and source.is_dir():
        path = source / "prepared" / "analyze_args.json"
    words = json.loads(path.read_text(encoding="utf-8"))
    known = {
        "--matches",
        "--identity-threshold",
        "--min-overlap",
        "--min-valid",
        "--score",
        "--require-reference",
        "--min-coverage",
        "--max-gap-fraction",
        "--max-missing-fraction",
        "--strata-column",
        *(
            f"--{role}{suffix}"
            for role in "abc"
            for suffix in ("", "-ref", "-ref-row", "-start", "-strand", "-type")
        ),
    }
    if (
        not isinstance(words, list)
        or not all(isinstance(v, str) for v in words)
        or any(v.startswith("--") and v not in known for v in words)
    ):
        raise ValueError(
            "--prepared-v1 requires v1 matching, input and QC options; newer policies require native replay"
        )
    for role in "abc":
        flag = f"--{role}"
        if flag not in words or words.index(flag) + 1 == len(words):
            raise ValueError(f"prepared manifest is missing {flag}")
        name = Path(words[words.index(flag) + 1]).name.lower().removesuffix(".gz")
        if Path(name).suffix not in (".fa", ".fas", ".fasta", ".faa", ".fna", ".ffn"):
            raise ValueError("--prepared-v1 requires aligned FASTA inputs")
    return path


def worker(args):
    sys.path.insert(0, str(args.source.resolve()))
    from _resources import peak_resident_bytes
    from matched_scaling import fixture

    from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
    from triplet_coevolution.core.coordinates import build_reference_map
    from triplet_coevolution.inference.mi import mutual_information_block, triplet_mi
    from triplet_coevolution.inference.potts import direct_couplings
    from triplet_coevolution.inference.threebody import fit_triples, sample_candidates
    from triplet_coevolution.inference.weights import sequence_weights

    (n,) = args.rows
    if args.prepared_v1:
        from _matched_inputs import prepared_subset

        msa, mappings, _, config = prepared_subset(
            v1_manifest(args.prepared_v1), n, args.largest or n, args.seed
        )
    else:
        msa, _ = fixture(n, args.widths, args.seed)
        mappings = None
        config = {"identity_threshold": 0.8, "min_overlap": 0.5, "min_valid": 2}
    if args.tied_dca:
        rng = np.random.default_rng(args.seed)
        components = []
        for tag in "ABC":
            first, other = rng.integers(0, 2, size=(2, n))
            states = np.column_stack([first, 1 - first, first, other, 1 - other, other])
            ids = tuple(tag + str(i) for i in range(n))
            sequences = tuple("".join(row) for row in np.array(["A", "C"])[states])
            components.append(ComponentMSA(tag, "dna", ids, sequences))
        matches = tuple(TripletMatch(str(i), *(c.ids[i] for c in components)) for i in range(n))
        msa = MatchedTripletMSA(*components, matches)
    if args.layout == "4/4/20" or args.dense:
        components = []
        order = (1, 2, 0) if args.layout == "4/4/20" else (0, 1, 2)
        for tag, offset in zip("ABC", order, strict=True):
            comp = msa.components[offset]
            seqs = comp.sequences if not args.dense else (comp.sequences[0],) * n
            components.append(ComponentMSA(tag, comp.molecule, comp.ids, seqs))
        matches = tuple(TripletMatch(str(i), *(c.ids[i] for c in components)) for i in range(n))
        msa = MatchedTripletMSA(*components, matches)
    width = [c.n_sites for c in msa.components]
    for c in msa.components:
        _ = c.matrix
    candidates = sample_candidates([range(w) for w in width], args.triples, args.seed)[0]
    if mappings is None:
        mappings = [build_reference_map(c, c.ids[0]) for c in msa.components]
    weight_options = {k: config[k] for k in ("identity_threshold", "min_overlap")}
    score_options = {"min_valid": config["min_valid"]}
    w = sequence_weights(msa, **weight_options)
    dca_weights = None if args.tied_dca else w
    unit = (np.arange(n) % 13 != 0).astype(float)
    nucleotide_parts = [c for c in msa.components if c.molecule.value in ("rna", "dna")]
    if args.jobs == ["mi_nucleotides"] and len(nucleotide_parts) != 2:
        raise ValueError("mi_nucleotides requires exactly two nucleotide components")
    jobs = {
        "weights": lambda: sequence_weights(msa, **weight_options),
        "mi": lambda: triplet_mi(msa, w, **score_options),
        "mi_unit": lambda: triplet_mi(msa, unit, **score_options),
        "mi_nucleotides": lambda: mutual_information_block(
            *nucleotide_parts, unit, **score_options
        ),
        "mi_legacy": lambda: triplet_mi(msa, w, kernel="v1_scalar", **score_options),
        "triples": lambda: fit_triples(msa, candidates, w, **score_options),
        "dca": lambda: direct_couplings(msa, dca_weights, **score_options),
        "dca_cholesky": lambda: direct_couplings(
            msa, dca_weights, factor_solver="cholesky", **score_options
        ),
    }
    job = args.jobs[0]
    if args.preload_scipy:
        import scipy.linalg  # noqa: F401 -- equalize import cost in both DCA workers
    before = peak_resident_bytes()
    seconds = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        r = jobs[job]()
        seconds.append(time.perf_counter() - start)
    peak = peak_resident_bytes()
    if job.startswith("mi"):
        blocks = (r,) if job == "mi_nucleotides" else (r.ab, r.ac, r.bc)
        arrays = [
            getattr(b, name)
            for b in blocks
            for name in ("mi_bits", "mi_apc", "n_valid", "valid_weight")
        ]
        pairs = sum(b.mi_bits.size for b in blocks)
    elif job == "weights":
        arrays = [r.weights, r.neighbor_counts]
        pairs = n * (n - 1) // 2
    elif job == "triples":
        arrays = [
            np.array(
                [
                    [
                        s.threebody_rms,
                        s.co_information_bits,
                        s.n_valid,
                        s.valid_weight,
                        s.observed_cells,
                        s.possible_cells,
                    ]
                    for s in r
                ]
            )
        ]
        pairs = len(candidates)
    else:
        arrays = [
            getattr(b, name)
            for b in r.values()
            for name in ("frobenius", "corrected", "n_valid", "valid_weight")
        ]
        pairs = sum(b.frobenius.size for b in r.values())
        if args.arrays:
            np.savez(args.arrays, *arrays)
    coord = [
        np.asarray(values, dtype=np.int64)
        for m in mappings
        for values in (
            m.original_columns,
            [p if p is not None else -1 for p in m.reference_positions],
        )
    ]
    print(
        json.dumps(
            {
                "kernel_source_sha256": hashlib.sha256(
                    Path(
                        {"weights": sequence_weights, "triples": fit_triples}.get(
                            job, triplet_mi if job.startswith("mi") else direct_couplings
                        ).__code__.co_filename
                    ).read_bytes()
                ).hexdigest(),
                "seconds": seconds,
                "median_seconds": statistics.median(seconds),
                "peak_rss_bytes": peak,
                "peak_before_bytes": before,
                "units_per_second": pairs / statistics.median(seconds),
                "work_units": pairs,
                "full_result_sha256": digest(arrays),
                "candidate_sha256": digest([candidates]),
                "coordinates_sha256": digest(coord),
                "weight_sum": (
                    float(n)
                    if args.tied_dca
                    else float(unit.sum())
                    if job in ("mi_unit", "mi_nucleotides")
                    else w.n_eff
                ),
                "widths": width,
                "weight_workspace_bytes": w.preflight["peak_bytes"],
            }
        )
    )


def run_one(args, n, source, job, output=None):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--source",
        str(source),
        "--rows",
        str(n),
        "--widths",
        *map(str, args.widths),
        "--seed",
        str(args.seed),
        "--repeat",
        "1",
        "--triples",
        str(args.triples),
        "--layout",
        args.layout,
        "--jobs",
        job,
    ]
    if args.dense:
        cmd.append("--dense")
    if args.tied_dca:
        cmd.append("--tied-dca")
    if args.prepared_v1:
        cmd.extend(["--prepared-v1", str(args.prepared_v1), "--largest", str(max(args.rows))])
    if output:
        cmd.extend(["--arrays", str(output), "--preload-scipy"])
    env = {
        **os.environ,
        **dict.fromkeys(("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"), "1"),
    }
    return json.loads(subprocess.check_output(cmd, env=env, text=True))


def summarize(runs):
    first = runs[0].copy()
    keys = ("full_result_sha256", "candidate_sha256", "coordinates_sha256")
    if any(r[k] != first[k] for r in runs for k in keys):
        raise ValueError("result changed across repeated runs")
    first["seconds"] = [r["median_seconds"] for r in runs]
    first["median_seconds"] = statistics.median(first["seconds"])
    first["units_per_second"] = first["work_units"] / first["median_seconds"]
    first["peak_rss_bytes"] = statistics.median(r["peak_rss_bytes"] for r in runs)
    first["peak_before_bytes"] = statistics.median(r["peak_before_bytes"] for r in runs)
    return first


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, nargs="+", default=[256, 2048])
    p.add_argument("--widths", type=int, nargs=3, default=[96, 64, 48])
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--seed", type=int, default=93)
    p.add_argument("--triples", type=int, default=1024)
    p.add_argument("--layout", choices=["20/4/4", "4/4/20"], default="20/4/4")
    p.add_argument("--dense", action="store_true", help="Use repeated identical genotypes.")
    p.add_argument(
        "--tied-dca",
        action="store_true",
        help="Use six duplicated and complemented binary columns per role, with unit fit weights.",
    )
    p.add_argument(
        "--prepared-v1",
        type=Path,
        help="Benchmark a local native v1 prepared FASTA panel using the matched-scaling QC policy.",
    )
    p.add_argument("--largest", type=int, help=argparse.SUPPRESS)
    p.add_argument(
        "--jobs",
        choices=[
            "weights",
            "mi",
            "mi_unit",
            "mi_nucleotides",
            "mi_legacy",
            "triples",
            "dca",
            "dca_cholesky",
        ],
        nargs="+",
        default=["weights", "mi", "mi_unit", "triples"],
    )
    p.add_argument(
        "--revision", default="e0ad9ae", help="Reference Git revision in this repository."
    )
    p.add_argument(
        "--reference-src", type=Path, help="Reference package src/ in an existing checkout."
    )
    p.add_argument("--current-src", type=Path, help="Override current package src/.")
    p.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--arrays", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--preload-scipy", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if min(*args.rows, *args.widths, args.repeat, args.triples) < 1:
        p.error("sizes, repeats and number of candidates must be positive")
    if args.prepared_v1 and (args.dense or args.layout != "20/4/4"):
        p.error("synthetic genotype/layout overrides cannot be combined with a prepared panel")
    if args.tied_dca and (
        (not args.source and args.jobs != ["dca_cholesky"])
        or args.prepared_v1
        or args.dense
        or args.layout != "20/4/4"
    ):
        p.error("--tied-dca requires only --jobs dca_cholesky on synthetic data")
    if args.source:
        worker(args)
        return
    import triplet_coevolution

    current = args.current_src or Path(triplet_coevolution.__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        ref = args.reference_src
        if ref is None:
            root = Path(
                subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
            )
            archive = folder / "baseline.zip"
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "archive",
                    "--format=zip",
                    "--output",
                    str(archive),
                    args.revision,
                    "src",
                ],
                check=True,
            )
            with zipfile.ZipFile(archive) as z:
                z.extractall(folder)
            ref = folder / "src"
        for n in args.rows:
            for job in args.jobs:
                outputs = (
                    [folder / "baseline.npz", folder / "candidate.npz"]
                    if job == "dca_cholesky"
                    else [None, None]
                )
                runs = [[], []]
                variants = [(ref, "dca" if job == "dca_cholesky" else job), (current, job)]
                for repeat in range(args.repeat):
                    for variant in (0, 1) if repeat % 2 == 0 else (1, 0):
                        src, worker_job = variants[variant]
                        runs[variant].append(run_one(args, n, src, worker_job, outputs[variant]))
                base, now = map(summarize, runs)
                for k in (
                    "candidate_sha256",
                    "coordinates_sha256",
                    "weight_sum",
                    "widths",
                    "work_units",
                ):
                    if base[k] != now[k]:
                        raise ValueError(f"{job}: {k} changed")
                parity = base["full_result_sha256"] == now["full_result_sha256"]
                details = {}
                if outputs[0]:
                    with np.load(outputs[0]) as a, np.load(outputs[1]) as b:
                        diffs = []
                        reordered = []
                        for i, k in enumerate(a.files):
                            x, y = a[k], b[k]
                            if i % 4 in (2, 3):
                                np.testing.assert_array_equal(x, y)
                            else:
                                np.testing.assert_allclose(
                                    x, y, rtol=2e-11, atol=2e-12, equal_nan=True
                                )
                                diffs.append(
                                    float(np.max(np.abs(x - y), where=np.isfinite(x), initial=0))
                                )
                                reordered.append(
                                    int(
                                        np.count_nonzero(
                                            np.argsort(-x.ravel(), kind="stable")
                                            != np.argsort(-y.ravel(), kind="stable")
                                        )
                                    )
                                )
                        details = {
                            "max_abs_score_difference": max(diffs),
                            "changed_raw_or_apc_rank_positions": reordered,
                        }
                elif not parity:
                    raise ValueError(f"{job}: output bytes differ from {args.revision}")
                print(
                    json.dumps(
                        {
                            "rows": n,
                            "job": job,
                            "speedup": base["median_seconds"] / now["median_seconds"],
                            "bitwise_equal": parity,
                            "reference_revision": args.revision
                            if args.reference_src is None
                            else "external_checkout",
                            "baseline": base,
                            "candidate": now,
                            **details,
                            "python": platform.python_version(),
                            "numpy": np.__version__,
                            "threads": 1,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
