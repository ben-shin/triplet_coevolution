"""Run the registered synthetic screening, inference and predictive recovery matrix."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np

from triplet_coevolution.inference.maxent import MaxEntOptions, fit_joint
from triplet_coevolution.inference.mi import weighted_mi
from triplet_coevolution.validation.fdr import bh_adjust

PROTOCOL = Path(__file__).with_name("recovery_matrix_protocol.json")
HASH = "9ce24e2f68caf787cf1a48070393d19384bc84c6dc716d846ba93fc0493f6b9d"
OPTIONS = {p: MaxEntOptions(prior_mass=p, tolerance=1e-8, max_iterations=300) for p in (0.5, 1, 8)}
PAIR = tuple(product(range(2), range(2, 4)))
TRIPLE = tuple(product(range(2), range(2, 4), range(4, 6)))


@dataclass(frozen=True)
class Cell:
    id: int
    depth: int
    q: float
    missing: float
    wrong: float
    law: str
    pair_strength: float
    parity_strength: float


def grid():
    raw = PROTOCOL.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != HASH:
        raise ValueError("registered matrix protocol checksum changed")
    p = json.loads(raw)
    settings = product(
        p["train_and_test_depths_each"],
        p["base_minor_frequencies"],
        p["per_cell_missing_probabilities"],
        p["whole_C_wrong_partner_probabilities"],
        p["planted_laws"],
    )
    return [
        Cell(i, n, q, m, c, law, *p["planted_laws"][law])
        for i, (n, q, m, c, law) in enumerate(settings)
    ], p


def stream(seed, cell, rep, purpose):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, cell, rep, purpose])))


def generate(cell, target, n, rng):
    rows = (rng.random((n, 6)) < cell.q).astype(np.int8)
    a, b, c = target
    copy = rng.random(n) < cell.pair_strength
    rows[copy, b] = rows[copy, a]
    parity = rng.random(n) < cell.parity_strength
    rows[parity, c] = rows[parity, a] ^ rows[parity, b]
    wrong = rng.random(n) < cell.wrong
    donor = rng.random((wrong.sum(), 2)) < cell.q
    discord = (1 - cell.pair_strength) * 2 * cell.q * (1 - cell.q)
    pc = (1 - cell.parity_strength) * cell.q + cell.parity_strength * discord
    donor[:, c - 4] = rng.random(wrong.sum()) < pc
    rows[wrong, 4:] = donor
    rows[rng.random(rows.shape) < cell.missing] = -1
    return rows


def table(rows, sites):
    x = rows[:, sites]
    x = x[(x >= 0).all(axis=1)]
    idx = x @ (2 ** np.arange(len(sites) - 1, -1, -1))
    return np.bincount(idx, minlength=2 ** len(sites)).reshape((2,) * len(sites))


def fixed_margin_p(counts):
    """Exact probability-ordering on the binary fiber given all (d-1)-way margins."""
    counts = np.asarray(counts)
    if (
        counts.shape not in ((2, 2), (2, 2, 2))
        or counts.dtype.kind not in "iu"
        or (counts < 0).any()
    ):
        raise ValueError(
            "exact test requires a binary pair or triple table of nonnegative integer counts"
        )
    flat = counts.ravel().astype(np.int64)
    direction = np.array([(-1) ** i.bit_count() for i in range(len(flat))])
    lo, hi = -int(flat[direction == 1].min()), int(flat[direction == -1].min())
    if lo == hi:
        return 1.0
    shifts = np.arange(lo, hi + 1)[:, None]
    alternatives = flat + shifts * direction
    logfac = np.array([math.lgamma(i + 1) for i in range(int(alternatives.max()) + 1)])
    logmass = -logfac[alternatives].sum(axis=1)
    extreme = logmass <= logmass[-lo] + 1e-12
    weights = np.exp(logmass - logmass.max())
    return float(weights[extreme].sum() / weights.sum())


def screen(rows, candidates):
    best = None
    high = -math.inf
    reasons = Counter()
    for sites in candidates:
        ct = table(rows, sites)
        n = int(ct.sum())
        if n < 8:
            reasons["train_lt8"] += 1
            continue
        if any(
            (ct.sum(axis=tuple(j for j in range(ct.ndim) if j != k)) == 0).any()
            for k in range(ct.ndim)
        ):
            reasons["train_constant"] += 1
            continue
        if len(sites) == 2:
            val = weighted_mi(rows[:, sites[0]], rows[:, sites[1]], 2, 2, min_valid=8)
        else:
            fit = fit_joint(ct, options=OPTIONS[1], n_valid=n)
            val = fit.score.connected_information_bits
            if not math.isfinite(val):
                reasons["train_fit_" + fit.score.status] += 1
                continue
        reasons["eligible"] += 1
        if val > high:
            best, high = sites, val
    return best, reasons


def log_score(fitted, counts):
    observed = counts > 0
    return float(np.average(np.log(fitted[observed]), weights=counts[observed]))


def tune(train, valid, sites):
    a, b = table(train, sites), table(valid, sites)
    if b.sum() < 8:
        return {}, {"validation_lt8": 1}
    best, reasons = {}, Counter()
    for mass in (0.5, 8):
        fit = fit_joint(a, options=OPTIONS[mass], n_valid=int(a.sum()))
        for name, fitted in (("M2", fit.p2), ("M3", fit.p3)):
            if fitted is None or (fitted <= 0).any():
                reasons[name + "_" + fit.score.status] += 1
                continue
            score = log_score(fitted, b)
            if name not in best or score > best[name][0]:
                best[name] = (score, mass, fitted)
    return best, reasons


def one(cell, seed, rep):
    hidden = stream(seed, cell.id, rep, 0).integers(2, size=3) + np.array([0, 2, 4])
    true = tuple(hidden)
    train, valid, test = [
        generate(cell, true, n, stream(seed, cell.id, rep, j))
        for j, n in enumerate((cell.depth, cell.depth // 2, cell.depth), start=1)
    ]
    return evaluate_cohorts(train, valid, test, true, cell)


def evaluate_cohorts(train, valid, test, true, cell):
    chosen_pair, pair_reasons = screen(train, PAIR)
    chosen_triple, triple_reasons = screen(train, TRIPLE)
    chosen = {"pair": chosen_pair, "triple": chosen_triple}
    correct = {
        "pair": bool(cell.pair_strength > 0 and chosen_pair == true[:2]),
        "triple": bool(cell.parity_strength > 0 and chosen_triple == true),
    }
    record = {
        "selected": chosen,
        "correct": correct,
        "pair_screen_candidates": pair_reasons,
        "triple_screen_candidates": triple_reasons,
        "invalid": Counter(),
        "p": {},
        "counts": {},
        "tuning_choices": {},
        "gain": None,
    }
    for kind, sites in chosen.items():
        if sites is None:
            record["invalid"][kind + "_no_candidate"] += 1
            continue
        ct = table(test, sites)
        for partition, rows in (("train", train), ("validation", valid), ("test", test)):
            record["counts"][kind + "_" + partition] = int(table(rows, sites).sum())
        if ct.sum() < 8:
            record["invalid"][kind + "_test_lt8"] += 1
        else:
            record["p"][kind] = fixed_margin_p(ct)
        if kind == "triple":
            models, issues = tune(train, valid, sites)
            record["invalid"].update(issues)
            record["tuning_choices"] = {k: z[1] for k, z in models.items()}
            if len(models) == 2 and ct.sum() >= 8:
                record["gain"] = log_score(models["M3"][2], ct) - log_score(models["M2"][2], ct)
    return record


def rate(k, n):
    if not n:
        return {"successes": 0, "n": 0, "estimate": None, "wilson95": None}
    p, z = k / n, 1.959963984540054
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {
        "successes": int(k),
        "n": n,
        "estimate": p,
        "wilson95": [max(0, center - half), min(1, center + half)],
    }


def estimate(values, rng):
    if not values:
        return {"mean": None, "n": 0, "bootstrap95": None}
    v = np.asarray(values)
    ci = None
    if len(v) >= 2:
        ci = np.quantile(rng.choice(v, size=(1000, len(v))).mean(axis=1), [0.025, 0.975]).tolist()
    return {"mean": float(v.mean()), "n": len(v), "bootstrap95": ci}


def run_cell(job):
    cell, reps, seed = job
    counters, support = defaultdict(Counter), defaultdict(list)
    gains, fdp = [], []
    events = Counter()
    for rep in range(reps):
        row = one(cell, seed, rep)
        for key in ("pair_screen_candidates", "triple_screen_candidates", "invalid"):
            counters[key].update(row[key])
        for model, value in row["tuning_choices"].items():
            counters["smoothing"][f"{model}_mass_{value}"] += 1
        for k, v in row["counts"].items():
            support[k].append(v)
        if row["gain"] is not None:
            gains.append(row["gain"])
        keys = ("pair", "triple")
        bh = bh_adjust([row["p"].get(k, 1) for k in keys]) <= 0.05
        nfalse = 0
        for k, hit in zip(keys, bh, strict=True):
            correct = row["correct"][k]
            tested = k in row["p"]
            raw = row["p"].get(k, 1) <= 0.05
            nfalse += bool(hit and not correct)
            for event, flag in (
                ("selected_truth", correct),
                ("evaluable", tested),
                ("raw_any", raw),
                ("raw_correct", raw and correct),
                ("raw_wrong", raw and not correct),
                ("bh_any", hit),
                ("bh_correct", hit and correct),
                ("bh_wrong", hit and not correct),
            ):
                events[k + "_" + event] += int(flag)
        events["family_any"] += int(bh.any())
        events["family_wrong"] += int(nfalse > 0)
        fdp.append(nfalse / max(1, int(bh.sum())))
    rates = {}
    for k, v in events.items():
        planted = cell.pair_strength if k.startswith("pair_") else cell.parity_strength
        if ("selected_truth" in k or k.endswith("_correct")) and not planted:
            continue
        rates[k] = rate(v, reps)
        if k.endswith("_any") and not k.startswith("family"):
            rates[k + "_per_evaluable"] = rate(v, events[k.split("_")[0] + "_evaluable"])
    rng = stream(seed, cell.id, 0, 4)
    return {
        "setting": asdict(cell),
        "replicates": reps,
        "rates": rates,
        "screen_and_fit_counts": dict(counters),
        "mean_eligible_training_candidates": {
            k: counters[k + "_screen_candidates"]["eligible"] / reps for k in ("pair", "triple")
        },
        "support_means_when_selected": {
            k: {"n": len(v), "mean": float(np.mean(v))} for k, v in support.items()
        },
        "bh_false_discovery_proportion": estimate(fdp, rng),
        "test_M3_minus_M2_nats": estimate(gains, rng),
        "test_gain_positive_all": rate(sum(x > 0 for x in gains), reps),
        "test_gain_positive_evaluable": rate(sum(x > 0 for x in gains), len(gains)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    if not 1 <= a.jobs <= 16:
        parser.error("jobs must be between 1 and 16")
    cells, p = grid()
    jobs = [(cell, p["repetitions_per_cell"], p["root_seed"]) for cell in cells]
    if a.jobs == 1:
        rows = list(map(run_cell, jobs))
    else:
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            rows = list(pool.map(run_cell, jobs))
    result = {
        "registration_sha256_lf": HASH,
        "numpy": np.__version__,
        "settings": len(rows),
        "replicates": sum(r["replicates"] for r in rows),
        "cells": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
