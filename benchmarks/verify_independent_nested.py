"""Independently reconstruct all primary held-out phylum losses from fitted energy tensors."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from Bio import SeqIO

ALPHABETS = {"protein": "ACDEFGHIKLMNPQRSTVWY", "rna": "ACGU", "dna": "ACGT"}


def energy_losses(theta, header, heldout, aligned):
    sites = header["sites"]
    size = [max(site["native_to_bin"]) + 1 for site in sites]
    terms = [(i,) for i in range(len(size))]
    terms.extend(tuple(t) for t in sorted(header["edges"]))
    terms.extend(tuple(t) for t in sorted(header["triples"]))
    tensors = []
    start = 0
    for t in terms:
        shape = tuple(size[j] for j in t)
        length = int(np.prod(shape))
        tensors.append(theta[start : start + length].reshape(shape))
        start += length
    if start != len(theta):
        raise ValueError("tensor shape mismatch")
    losses = []
    for row in heldout:
        state = []
        for site in sites:
            role = site["component"]
            symbol = aligned[role][row[role.lower() + "_id"]][site["original_column"] - 1]
            code = ALPHABETS[site["molecule"]].find(symbol)
            if code < 0:
                break
            state.append(site["native_to_bin"][code])
        if len(state) != len(sites):
            continue
        total = 0.0
        for j, q in enumerate(size):
            value = state[j]
            energy = np.empty(q)
            for alternative in range(q):
                state[j] = alternative
                energy[alternative] = sum(
                    tensor[tuple(state[k] for k in t)]
                    for t, tensor in zip(terms, tensors, strict=True)
                )
            state[j] = value
            hi = energy.max()
            total += hi + np.log(np.exp(energy - hi).sum()) - energy[value]
        losses.append(total / len(sites))
    return np.asarray(losses)


def verify(cohorts, results):
    summary = json.loads((results / "independent_nested_metrics.json").read_bytes())
    count = 0
    max_error = 0.0
    for system, data in summary["cohorts"].items():
        source = cohorts / system
        with (source / "matches.tsv").open(encoding="utf8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        aligned = {
            r: {
                rec.id: str(rec.seq).upper()
                for rec in SeqIO.parse(source / (r + ".fasta"), "fasta")
            }
            for r in "ABC"
        }
        for fold in data["settings"]["observed_uniform"]["folds"]:
            train = tuple(row["observation_id"] for row in rows if row["phylum"] != fold["phylum"])
            held = [row for row in rows if row["phylum"] == fold["phylum"]]
            for kind, metric in (("pair_only", "pair_loss"), ("pair_plus_k", "plus_k_loss")):
                path = (
                    results / system / "observed_uniform" / fold["fold_directory"] / (kind + ".npz")
                )
                with np.load(path, allow_pickle=False) as datafile:
                    h = json.loads(datafile["header"].tobytes())
                    theta = datafile["theta"]
                sha = h.pop("sha256")
                raw = json.dumps(h, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                actual = hashlib.sha256(raw + theta.astype("<f8").tobytes()).hexdigest()
                if (
                    sha != fold[kind + "_sha256"]
                    or actual != sha
                    or tuple(h["fit"]["training_ids"]) != train
                ):
                    raise ValueError("fitted identity or training cohort differs")
                values = energy_losses(theta, h, held, aligned)
                if len(values) != fold["heldout_complete"] or not np.isfinite(values).all():
                    raise ValueError("held-out complete cases differ")
                if fold["status"] == "evaluated":
                    diff = abs(values.mean() - fold[metric])
                    max_error = max(max_error, diff)
                    if diff > 1e-11:
                        raise ValueError("heldout score failed independent scalar energy check")
                    count += 1
    return {"model_fold_losses": count, "maximum_absolute_error_nats_per_site": float(max_error)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohorts", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.cohorts, args.results), indent=2))
