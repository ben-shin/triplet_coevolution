"""Run the registered six-site cross-phylum comparison on frozen local public cohorts."""

import argparse
import csv
import hashlib
import itertools
import json
import platform
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from triplet_coevolution import __version__
from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.core.coordinates import subset_reference_map
from triplet_coevolution.inference.nested import (
    NestedOptions,
    evaluate_model,
    fit_nested,
    load_model,
)
from triplet_coevolution.inference.nested_model import save_model
from triplet_coevolution.io.metadata import read_triplet_table
from triplet_coevolution.io.selected_sources import matched_sources
from triplet_coevolution.msa.masking import apply_triplet_masks
from triplet_coevolution.msa.synchronize import synchronize
from triplet_coevolution.operations import file_id, put_json, read_json
from triplet_coevolution.validation.clade_splits import identity, make_split_plan, subset_rows

REGISTERED = "5e415125959c925e8408994e638220eb9a559d6a0ded3c69a8336780a85186d3"
HERE = Path(__file__).resolve().parent
SCENARIOS = (
    ("observed", "uniform"),
    ("observed", "equal_phylum"),
    ("broken_matching", "uniform"),
    ("independent_sites", "uniform"),
)


def registered(path, expected):
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"registration checksum differs for {path.name}")
    return json.loads(raw)


def cohort(root, name, freeze, panel):
    directory = root / name
    for key, digest in freeze["files_sha256"].items():
        if file_id(directory / key)["sha256"] != digest:
            raise ValueError(f"{name} source checksum mismatch for {key}")
    axes = read_json(directory / "prelabel-sites.json")
    paths = {r: directory / (r + ".fasta") for r in "ABC"}
    molecules = {r: "rna" if name == "ribosome" and r == "A" else "protein" for r in "ABC"}
    refs = {r: axes[r]["reference_id"] for r in "ABC"}
    table = read_triplet_table(directory / "matches.tsv")
    source = matched_sources(paths, table.matches, molecules, refs)
    full = synchronize(*source.components, table)
    groups = {
        m.observation_id: ann["phylum"]
        for m, ann in zip(table.matches, table.annotations, strict=True)
    }
    with (directory / "prelabel-weights.tsv").open(encoding="utf8", newline="") as handle:
        masses = {
            row["observation_id"]: float(row["equal_phylum"])
            for row in csv.DictReader(handle, delimiter="\t")
        }
    if set(masses) != set(groups):
        raise ValueError("predeclared weights do not cover exact assembly IDs")
    masks = []
    maps = []
    for m in source.maps:
        original = panel[m.component]
        if original != [m.n_sites // 3, 2 * m.n_sites // 3]:
            raise ValueError("registered reference widths or panel have changed")
        ix = [pos - 1 for pos in original]
        chosen = subset_reference_map(m, ix)
        if list(chosen.reference_positions) != original:
            raise ValueError("panel does not have the registered native numbering")
        maps.append(chosen)
        masks.append(np.isin(np.arange(m.n_sites), ix))
    return apply_triplet_masks(full, *masks).msa, tuple(maps), groups, masses


def controls(source, groups):
    ids = [m.observation_id for m in source.matches]
    idx = {key: i for i, key in enumerate(ids)}
    strata = {g: sorted(i for i in ids if groups[i] == g) for g in sorted(set(groups.values()))}
    order = [np.arange(len(ids)) for _ in range(3)]
    for labels in strata.values():
        ordered = sorted(
            labels, key=lambda k: hashlib.sha256(("matched-negative-v1/" + k).encode()).digest()
        )
        j = [idx[k] for k in ordered]
        for offset in (1, 2):
            order[offset][j] = np.roll(j, -offset)
    parts = [
        c.take_ids(tuple(c.ids[i] for i in perm))
        for c, perm in zip(source.components, order, strict=True)
    ]
    matches = tuple(TripletMatch(key, *(c.ids[i] for c in parts)) for i, key in enumerate(ids))
    mixed = MatchedTripletMSA(*parts, matches)
    seqs = [list(map(list, c.sequences)) for c in source.components]
    rng = np.random.Generator(np.random.PCG64(731904))
    for labels in strata.values():
        for comp, data in zip(source.components, seqs, strict=True):
            letters = CANONICAL[comp.molecule]
            for j in range(comp.n_sites):
                eligible = [idx[k] for k in labels if data[idx[k]][j].upper() in letters]
                values = rng.permutation([data[i][j] for i in eligible])
                for i, symbol in zip(eligible, values, strict=True):
                    data[i][j] = str(symbol)
    parts = [
        ComponentMSA(c.name, c.molecule, c.ids, tuple("".join(row) for row in data))
        for c, data in zip(source.components, seqs, strict=True)
    ]
    site_shuffle = MatchedTripletMSA(*parts, source.matches)
    info = {
        "single_member_phyla": sum(len(v) == 1 for v in strata.values()),
        "two_member_phyla": sum(len(v) == 2 for v in strata.values()),
        "unchanged_row_assignment_counts_by_component": {
            r: int(np.count_nonzero(o == np.arange(len(ids))))
            for r, o in zip("ABC", order, strict=True)
        },
        "original_assignments_sha256": identity([list(c.ids) for c in source.components]),
        "shuffled_assignments_sha256": identity([list(c.ids) for c in mixed.components]),
        "column_shuffle_sha256": identity([list(c.sequences) for c in site_shuffle.components]),
    }
    return {"observed": source, "broken_matching": mixed, "independent_sites": site_shuffle}, info


def interval(values):
    if len(values) < 3:
        return {
            "lower": None,
            "upper": None,
            "blocks": len(values),
            "status": "insufficient_evaluable_phyla",
        }
    rng = np.random.Generator(np.random.PCG64(731903))
    draws = rng.choice(values, size=(2000, len(values))).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "lower": float(lo),
        "upper": float(hi),
        "blocks": len(values),
        "draws": 2000,
        "status": "available",
    }


def loss_fold(msa, maps, plan, masses, options, triples, dest):
    (fold,) = plan.folds
    train = subset_rows(msa, fold.train_ids)
    test = subset_rows(msa, fold.test_ids)
    if set(fold.train_ids) & set(fold.test_ids):
        raise ValueError("overlap between train and test")
    group = fold.test_groups[0]
    put_json(dest / "split.json", plan.metadata())
    wtrain = {k: masses[k] for k in fold.train_ids}
    wtest = np.array([masses[k] for k in fold.test_ids])
    fit = fit_nested(
        train,
        maps,
        options=options,
        triples=triples,
        weight_options={"policy": "supplied", "supplied_weights": wtrain},
    )
    design = fit.design
    good = np.ones(test.n_observations, dtype=bool)
    for component in test.components:
        good &= (
            (component.matrix >= 0) & (component.matrix < len(CANONICAL[component.molecule]))
        ).all(axis=1)
    good &= wtest > 0
    m = wtest[good]
    row = {
        "phylum": group,
        "fold_id": fold.fold_id,
        "split_sha256": plan.metadata()["plan_sha256"],
        "training_rows": len(fold.train_ids),
        "training_complete": design["positive_complete_rows"],
        "training_mass": design["complete_weight"],
        "training_kish": design["complete_kish"],
        "heldout_rows": len(fold.test_ids),
        "heldout_complete": int(good.sum()),
        "heldout_mass": float(m.sum()),
        "heldout_kish": float(m.sum() ** 2 / (m @ m)) if len(m) else None,
        "training_design_sha256": design["training_design_sha256"],
        "status": design["status"],
        "pair_loss": None,
        "plus_k_loss": None,
        "gain": None,
    }
    if len(design["edges"]) != 15 or len(design["selected_triples"]) != 8:
        raise ValueError("fit did not retain the registered graph")
    for name, model in fit.models.items():
        path = dest / (name + ".npz")
        save_model(path, model)
        recovered = load_model(path)
        if recovered.digest() != model.digest():
            raise ValueError("model did not round trip")
        diagnostic = model.metadata["diagnostics"]
        row[name + "_sha256"] = model.digest()
        row[name + "_iterations"] = diagnostic["iterations"]
        row[name + "_parameters"] = diagnostic["coefficients"]
        row[name + "_gradient"] = diagnostic["gradient_max"]
    if row["status"] != "converged":
        return row
    if not len(m):
        row["status"] = "no_complete_heldout_rows"
        return row
    losses = []
    for name in ("pair_only", "pair_plus_k"):
        scored = evaluate_model(fit.models[name], test, maps, weights=wtest)
        if scored["row_ids"] != fold.test_ids or not np.array_equal(good, scored["complete_mask"]):
            raise ValueError("model evaluation changed the paired observation set")
        loss = scored["loss_per_row_site_nats"][good]
        if not np.isfinite(loss).all():
            row["status"] = "nonfinite_heldout_loss"
            return row
        losses.append(loss)
    row.update(
        status="evaluated",
        pair_loss=float(np.average(losses[0].mean(axis=1), weights=m)),
        plus_k_loss=float(np.average(losses[1].mean(axis=1), weights=m)),
        gain=float(np.average((losses[0] - losses[1]).mean(axis=1), weights=m)),
    )
    return row


def summarize(rows):
    ok = [r for r in rows if r["status"] == "evaluated"]
    mean = lambda field, key: (
        float(np.average([r[key] for r in ok], weights=[r[field] for r in ok])) if ok else None
    )
    return {
        "planned_folds": len(rows),
        "evaluable_folds": len(ok),
        "eligible_holdout_rows": sum(r["heldout_rows"] for r in rows),
        "positive_complete_rows": sum(r["heldout_complete"] for r in rows),
        "scored_rows": sum(r["heldout_complete"] for r in ok),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "equal_phylum_mean_gain": float(np.mean([r["gain"] for r in ok])) if ok else None,
        "weighted_pooled_gain": mean("heldout_mass", "gain"),
        "weighted_pooled_pair_loss": mean("heldout_mass", "pair_loss"),
        "weighted_pooled_plus_k_loss": mean("heldout_mass", "plus_k_loss"),
        "assembly_count_pooled_gain": mean("heldout_complete", "gain"),
        "positive_gain_folds": sum(r["gain"] > 0 for r in ok),
        "equal_phylum_interval": interval([r["gain"] for r in ok]),
        "folds": rows,
    }


def study(cohorts, out):
    try:
        from scipy import __version__ as scipy_version
    except ImportError as exc:
        raise ValueError("The registered L-BFGS estimator requires the optimize extra.") from exc
    policy = registered(HERE / "independent_evaluation_policy.json", REGISTERED)
    frozen = registered(HERE / "independent_cohort_freeze.json", policy["cohort_freeze_sha256_lf"])
    nested = policy["nested"]
    options = NestedOptions(**nested["options"])
    out.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": "registered-heldout-six-sites.v1",
        "policy_sha256_lf": REGISTERED,
        "cohorts": {},
    }
    memberships = []
    start = perf_counter()
    for system in ("rnap", "ribosome"):
        panel = nested["panel"][system]
        msa, maps, groups, masses = cohort(cohorts, system, frozen["systems"][system], panel)
        cohorts_by_condition, control_info = controls(msa, groups)
        ids = tuple(groups)
        memberships.append(set(ids))
        sizes = Counter(groups.values())
        eligible = sorted(g for g, n in sizes.items() if n >= 8 and len(ids) - n >= 24)
        data = {
            "admitted": len(ids),
            "phyla": len(sizes),
            "eligible_phyla": eligible,
            "small_or_no_training_phyla": {
                g: n for g, n in sorted(sizes.items()) if g not in eligible
            },
            "panel": panel,
            "controls": control_info,
            "settings": {},
        }
        summary["cohorts"][system] = data
        triples = tuple(itertools.product(*(panel[r] for r in "ABC")))
        for control, weight in SCENARIOS:
            key = control + "_" + weight
            values = dict.fromkeys(ids, 1.0) if weight == "uniform" else masses
            rows = []
            for index, group in enumerate(eligible, 1):
                plan = make_split_plan(
                    ids,
                    groups,
                    test_groups=[group],
                    source={"policy": REGISTERED, "system": system},
                )
                directory = f"fold-{index:02d}"
                destination = out / system / key / directory
                row = loss_fold(
                    cohorts_by_condition[control], maps, plan, values, options, triples, destination
                )
                row["fold_directory"] = directory
                rows.append(row)
            data["settings"][key] = summarize(rows)
            print(system, key, len(rows), "folds", flush=True)
    summary["shared_assemblies"] = len(set.intersection(*memberships))
    if summary["shared_assemblies"] != 139:
        raise ValueError("frozen cohort overlap changed")
    put_json(out / "independent_nested_metrics.json", summary)
    put_json(
        out / "execution.json",
        {
            "elapsed_seconds": perf_counter() - start,
            "python": platform.python_version(),
            "package": __version__,
            "numpy": np.__version__,
            "scipy": scipy_version,
            "code_sha256": file_id(Path(__file__))["sha256"],
        },
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohorts", type=Path)
    parser.add_argument(
        "out", type=Path, help="New directory for fold models and aggregate results."
    )
    args = parser.parse_args()
    study(args.cohorts, args.out)
