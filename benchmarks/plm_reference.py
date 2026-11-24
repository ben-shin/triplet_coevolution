"""Compare a synthetic gap-free, full-alphabet protein fit with an external plmc executable."""

import argparse
import hashlib
import json
import subprocess
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference._plm_target import zero_sum
from triplet_coevolution.inference.plm import PLMOptions, fit_plm


def synthetic(seed=716031):
    """Balanced residue frequencies by cyclic replication of each six-site observation."""
    alphabet = CANONICAL["protein"]
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, 20, (120, 6))
    for a, b, offset, rate in ((0, 2, 0, 0.75), (0, 4, 3, 0.55), (1, 5, 8, 0.45), (3, 4, 1, 0.30)):
        keep = rng.random(len(rows)) < rate
        rows[keep, b] = (rows[keep, a] + offset) % 20
    rows = ((rows[:, None, :] + np.arange(20)[None, :, None]) % 20).reshape(-1, 6)
    seqs = tuple("".join(alphabet[v] for v in row) for row in rows)
    ids = tuple(f"s{i}" for i in range(len(rows)))
    comps = tuple(
        ComponentMSA(
            c, "protein", tuple(c + i for i in ids), tuple(s[2 * j : 2 * j + 2] for s in seqs)
        )
        for j, c in enumerate("ABC")
    )
    msa = MatchedTripletMSA(*comps, tuple(TripletMatch(i, *(c + i for c in "ABC")) for i in ids))
    return msa, seqs


def _read_reference(path, n):
    data = path.read_bytes()
    L, q, nseq, skipped, _iterations = np.frombuffer(data, "<i4", 5)
    if (L, q, nseq, skipped) != (6, 20, n, 0) or data[40 : 40 + q].decode() != CANONICAL["protein"]:
        raise ValueError(
            "plmc dimensions, skipped sequence count, or alphabet disagree with the fixture"
        )
    params = np.frombuffer(data, "<f4", 5, offset=20)
    offset = 40 + q + 4 * (nseq + skipped) + L + 4 * L + 2 * 4 * L * q
    pairs = list(combinations(range(L), 2))
    n_pair = len(pairs) * q * q
    values = np.frombuffer(data, "<f4", offset=offset)
    if len(values) != 2 * n_pair:
        raise ValueError("unexpected external parameter array length")
    return {
        pair: zero_sum(m.astype(float)) for pair, m in zip(pairs, values[n_pair:].reshape(-1, q, q))
    }, params


def _rank(x):
    order = np.argsort(-np.asarray(x), kind="stable")
    return np.argsort(order) + 1


def compare(plmc: str, wsl: str | None = None) -> dict:
    msa, seqs = synthetic()
    n = len(seqs)
    opt = PLMOptions(lambda_h=0.008, lambda_j=0.004, tolerance=2e-8)
    native = {}

    def collect(pair, i, j, block):
        key = ("ABC".index(pair[0]) * 2 + i, "ABC".index(pair[1]) * 2 + j)
        native[key] = block

    fitted = fit_plm(msa, options=opt, coupling_sink=collect)
    if fitted.diagnostics["status"] != "ok":
        raise RuntimeError("native PLM optimization failed")
    with TemporaryDirectory(prefix="synthetic-plmc-") as folder:
        path = Path(folder)
        seqfile = path / "synthetic.fa"
        seqfile.write_bytes("".join(f">s{i}\n{s}\n" for i, s in enumerate(seqs)).encode("ascii"))
        fasta_sha = hashlib.sha256(seqfile.read_bytes()).hexdigest()
        (path / "weights.txt").write_bytes(b"1\n" * n)
        prefix = ["wsl", "-d", wsl, "--exec"] if wsl else []
        refdir = (
            subprocess.run(
                [*prefix, "wslpath", "-u", path.as_posix()],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            if wsl
            else path.as_posix()
        )
        # Native averages within each target. Shared pair penalty covers BOTH directional fits.
        args = [
            plmc,
            "-a",
            CANONICAL["protein"],
            "-w",
            refdir + "/weights.txt",
            "-lh",
            str(n * opt.lambda_h / 2),
            "-le",
            str(n * opt.lambda_j),
            "-lg",
            "0",
            "-m",
            "400",
            "-o",
            refdir + "/model.bin",
            refdir + "/synthetic.fa",
        ]
        proc = subprocess.run([*prefix, *args], capture_output=True, text=True, check=True)
        log = (proc.stdout + "\n" + proc.stderr).strip()
        if "Gradient optimization: Minimization success" not in log:
            raise RuntimeError(
                f"external plmc optimization did not report convergence: {log.splitlines()[-1:]}"
            )
        reference, external_options = _read_reference(path / "model.bin", n)
    keys = sorted(native)
    own = np.array([np.linalg.norm(native[key]) for key in keys])
    ext = np.array([np.linalg.norm(reference[key]) for key in keys])
    ours, theirs = _rank(own), _rank(ext)
    return {
        "fixture_sha256": fasta_sha,
        "rows": n,
        "alphabet": CANONICAL["protein"],
        "plmc_options": {
            "lambda_h": float(external_options[1]),
            "lambda_e": float(external_options[2]),
            "n_eff": float(external_options[4]),
        },
        "external_final_message": log.splitlines()[-1],
        "native_targets": fitted.diagnostics["target_status_counts"],
        "gradient_max": max(t["gradient_max"] for t in fitted.diagnostics["targets"]),
        "pairs": len(keys),
        "spearman": float(np.corrcoef(ours, theirs)[0, 1]),
        "pearson": float(np.corrcoef(own, ext)[0, 1]),
        "max_norm_difference": float(np.max(np.abs(own - ext))),
        "relative_norm_rmse": float(np.linalg.norm(own - ext) / np.linalg.norm(ext)),
        "max_coefficient_difference": float(
            max(np.max(np.abs(native[k] - reference[k])) for k in keys)
        ),
        "scores": [
            {"pair": [int(i + 1) for i in k], "asymmetric": float(a), "plmc": float(b)}
            for k, a, b in zip(keys, own, ext)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plmc", required=True, help="path to compiled plmc on the host or inside WSL"
    )
    parser.add_argument("--wsl", help="optional Windows WSL distribution name")
    print(json.dumps(compare(**vars(parser.parse_args())), indent=2))
