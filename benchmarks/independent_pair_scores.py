"""Generate the registered full matrix MI and joint mfDCA score archives."""

import argparse
import csv
import hashlib
import json
import math
import os
import time

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "4"

import numpy as np
from Bio import SeqIO
from cohort_sources import HERE, STAGE, atomic_json, digest

from triplet_coevolution import __version__
from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.coordinates import build_reference_map, subset_reference_map
from triplet_coevolution.inference.dca_models import DCADesign
from triplet_coevolution.inference.mi import triplet_mi
from triplet_coevolution.io.score_store import from_results, write_score_store

POLICY_SHA = "5e415125959c925e8408994e638220eb9a559d6a0ded3c69a8336780a85186d3"
OUT = STAGE.parent / "independent-evaluation"


def checked_json(path, sha):
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f"Registration digest mismatch: {path}")
    return json.loads(data)


POLICY = checked_json(HERE / "independent_evaluation_policy.json", POLICY_SHA)
FREEZE = checked_json(HERE / "independent_cohort_freeze.json", POLICY["cohort_freeze_sha256_lf"])


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h, delimiter="\t"))


def prepare(name):
    inputs = STAGE / name
    hashes = FREEZE["systems"][name]["files_sha256"]
    for f, expected in hashes.items():
        if digest(inputs / f) != expected:
            raise ValueError(f"Staged input hash mismatch: {f}")
    matched = read_rows(inputs / "matches.tsv")
    match_ids = tuple(
        TripletMatch(*(r[k] for k in ("observation_id", "a_id", "b_id", "c_id"))) for r in matched
    )
    weight_rows = read_rows(inputs / "prelabel-weights.tsv")
    if [w["observation_id"] for w in weight_rows] != [m.observation_id for m in match_ids]:
        raise ValueError("Weights do not follow registered assembly order")
    masks = json.loads((inputs / "prelabel-sites.json").read_text())
    axis_maps, components = [], []
    for comp in "ABC":
        seqs = list(SeqIO.parse(inputs / (comp + ".fasta"), "fasta"))
        molecule = "rna" if name == "ribosome" and comp == "A" else "protein"
        ref = seqs[0]
        coord = build_reference_map(
            ComponentMSA(comp, molecule, (ref.id,), (str(ref.seq),)), ref.id
        )
        indices = [j - 1 for j in masks[comp]["retained_reference_1based"]]
        axis_maps.append(subset_reference_map(coord, indices))
        trimmed = tuple("".join(str(r.seq)[j] for j in indices) for r in seqs[1:])
        components.append(ComponentMSA(comp, molecule, tuple(r.id for r in seqs[1:]), trimmed))
    data = MatchedTripletMSA(*components, match_ids)
    weights = {
        w: np.array([float(r[w]) for r in weight_rows]) for w in POLICY["pair_scores"]["weights"]
    }
    return data, tuple(axis_maps), weights, hashes


def fit(name, weight_name):
    out = OUT / name / weight_name
    out.mkdir(parents=True, exist_ok=True)
    data, maps, weights, hashes = prepare(name)
    weights = weights[weight_name]
    design = DCADesign.parse(POLICY["pair_scores"]["mf"])
    plan = design.preflight(data)
    expected = POLICY["resources"]["plan"][name]
    if (
        plan["estimated_numeric_bytes"] != expected["mf_native_bytes"]
        or plan["scopes"]["ABC"]["features"] != expected["D"]
    ):
        raise ValueError("Registered feature dimensions or native resource estimate changed")
    min_valid = math.ceil(len(weights) / 2)
    clock = time.perf_counter()
    print("Registered fit", name, weight_name, plan, flush=True)
    mi = triplet_mi(data, weights, min_valid=min_valid)
    seconds_mi = time.perf_counter() - clock
    clock = time.perf_counter()
    print("Full MI complete", name, weight_name, round(seconds_mi, 2), flush=True)
    info = {}
    dca = design.fit(data, weights, min_valid=min_valid, min_valid_weight=0.0, diagnostics=info)
    seconds_dca = time.perf_counter() - clock
    provenance = {
        "software_version": __version__,
        "inputs": {**{c: hashes[c + ".fasta"] for c in "ABC"}, "matches": hashes["matches.tsv"]},
        "methods": {"dca_design": design.metadata()},
        "weight_policy": weight_name,
        "evaluation_policy_sha256": POLICY_SHA,
        "weight_table_sha256": hashes["prelabel-weights.tsv"],
        "column_policy_sha256": hashes["prelabel-sites.json"],
    }
    store = from_results(maps, mi, dca, provenance=provenance, require_reference=True)
    write_score_store(out, store)
    summary = {
        "name": name,
        "weights": weight_name,
        "phylum_count": FREEZE["systems"][name]["distinct_phyla"],
        "n": len(weights),
        "weight_sum": float(weights.sum()),
        "kish": float(weights.sum() ** 2 / (weights @ weights)),
        "min_valid": min_valid,
        "plan": plan,
        "diagnostics": info,
        "seconds_mi": seconds_mi,
        "seconds_mf": seconds_dca,
        "policy_sha256": POLICY_SHA,
        "manifest_sha256": digest(out / "score_manifest.json"),
        "payload_sha256": digest(out / "pair_scores.npz"),
        "pair_support": {
            p: {
                "count_min": int(n.min()),
                "count_max": int(n.max()),
                "weight_min": float(store.valid_weight[p].min()),
                "weight_max": float(store.valid_weight[p].max()),
                "cells": n.size,
            }
            for p, n in store.n_valid.items()
        },
    }
    atomic_json(out / "fit.json", summary)
    print("Frozen", name, weight_name, summary["payload_sha256"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("system", choices=["rnap", "ribosome"])
    p.add_argument("weight", nargs="*", choices=POLICY["pair_scores"]["weights"])
    a = p.parse_args()
    for w in a.weight or POLICY["pair_scores"]["weights"]:
        fit(a.system, w)
