"""Advanced score tables and calibration on already synchronized, masked observations."""

import csv
import hashlib
import json
from math import prod
from pathlib import Path

import numpy as np

from triplet_coevolution.inference.global_potts import global_triples
from triplet_coevolution.inference.mi import triplet_mi
from triplet_coevolution.inference.potts import PAIRS, direct_couplings
from triplet_coevolution.inference.threebody import fit_triples, sample_candidates
from triplet_coevolution.inference.weights import _WeightComparisons
from triplet_coevolution.validation.nulls import permutation_test
from triplet_coevolution.validation.structure import contact_metrics, read_contacts

SITE_FIELDS = ("component", "reference_id", "col", "original_col", "reference_pos", "symbol")
PAIR_FIELDS = ["pair", *(f"{prefix}_{key}" for prefix in "xy" for key in SITE_FIELDS)]
TRIPLE_FIELDS = [f"{prefix}_{key}" for prefix in "abc" for key in SITE_FIELDS]
NULL_FIELDS = ["statistic", "score", "permutation_p", "bh_q", "null_draws"]


class ModelFitError(ValueError):
    """One optional estimator could not fit the supplied data."""

    def __init__(self, method, reason):
        super().__init__(str(reason))
        self.method = method


def _fit(method, estimator, *args, **kwargs):
    try:
        return estimator(*args, **kwargs)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ModelFitError(method, exc) from exc


def _site(mapping, index, prefix):
    s = mapping.site(int(index))
    return dict(
        zip(
            (f"{prefix}_{key}" for key in SITE_FIELDS),
            (
                mapping.component,
                mapping.reference_id,
                s.current_column,
                s.original_column,
                s.reference_position,
                s.reference_symbol,
            ),
        )
    )


def _identity(path):
    if path is None:
        return None
    path = Path(path)
    with path.open("rb") as handle:
        return {
            "path": str(path.absolute()),
            "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
        }


def _candidates(path, maps, axes, max_triplets, seed):
    if path is None:
        return sample_candidates(axes, max_triplets, seed)
    lookups = [
        {mapping.original_columns[i] + 1: i for i in indices}
        for mapping, indices in zip(maps, axes)
    ]
    selected = set()
    fields = [f"{x}_original_col" for x in "abc"]
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        if not reader.fieldnames or sorted(reader.fieldnames) != sorted(fields):
            raise ValueError("triplet candidate TSV requires " + ", ".join(fields))
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"malformed triplet candidate at line {reader.line_num}")
            try:
                triple = tuple(lookup[int(row[field])] for field, lookup in zip(fields, lookups))
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"candidate line {reader.line_num} refers to an ineligible original column"
                ) from exc
            if triple in selected:
                raise ValueError(f"duplicate candidate at line {reader.line_num}")
            selected.add(triple)
            if len(selected) > max_triplets:
                raise ValueError("explicit candidates exceed max_triplets")
    if not selected:
        raise ValueError("triplet candidate table is empty")
    return sorted(selected), prod(len(axis) for axis in axes)


def _table(path, fields, rows, fmt):
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="," if fmt == "csv" else "\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return {"path": path.name, "rows": count}


def write_advanced_results(
    out_dir,
    msa,
    maps,
    weights,
    *,
    dca=False,
    threebody=False,
    ridge=0.1,
    pseudocount=0.5,
    max_triplets=1000,
    triplet_candidates=None,
    permutations=0,
    seed=0,
    strata=None,
    contacts=None,
    min_valid=2,
    max_pairs=1000,
    require_reference=False,
    fmt="tsv",
    mi_result=None,
    score="mi_apc",
    threebody_model="local",
    global_l2=0.1,
):
    """Write within caller-managed staging. Return JSON-native settings, method and file inventory."""
    if fmt not in {"csv", "tsv"} or score not in {"mi_bits", "mi_apc"}:
        raise ValueError("advanced results require CSV/TSV and mi_bits or mi_apc")
    if permutations < 0 or seed < 0 or max_triplets < 1 or max_pairs < 1:
        raise ValueError("seed/permutations must be nonnegative and output limits positive")
    if triplet_candidates is not None and not threebody:
        raise ValueError("triplet candidates require threebody inference")
    if threebody_model not in {"local", "global"} or (
        threebody_model == "global" and not threebody
    ):
        raise ValueError("threebody_model must be local or global, and global requires threebody")
    if strata is not None and not permutations:
        raise ValueError("strata require permutations")
    if len(maps) != 3 or any(
        m.component != c.name or m.n_sites != c.n_sites for m, c in zip(maps, msa.components)
    ):
        raise ValueError("reference maps must match each filtered alignment")
    out = Path(out_dir)
    extra_inputs = {
        "triplet_candidates": _identity(triplet_candidates),
        "contacts": _identity(contacts),
    }
    axes = [
        np.asarray(
            [
                i
                for i, pos in enumerate(m.reference_positions)
                if not require_reference or pos is not None
            ],
            dtype=int,
        )
        for m in maps
    ]
    fixed = []
    eligible = 0
    selection = None
    if threebody:
        fixed, eligible = _candidates(triplet_candidates, maps, axes, max_triplets, seed)
        selection = "exhaustive" if len(fixed) == eligible else "uniform"
        if triplet_candidates is not None:
            selection = "file"
    labels = read_contacts(contacts, maps) if contacts is not None else {}
    if extra_inputs != {
        "triplet_candidates": _identity(triplet_candidates),
        "contacts": _identity(contacts),
    }:
        raise ValueError("advanced input changed while reading; retry with stable input files")
    outputs = {}
    settings = {
        "dca": dca,
        "threebody": threebody,
        "ridge": ridge,
        "pseudocount": pseudocount,
        "max_triplets": max_triplets,
        "seed": seed,
        "permutations": permutations,
        "stratified": strata is not None,
        "min_valid": min_valid,
        "max_pairs": max_pairs,
        "threebody_model": threebody_model,
        "global_l2": global_l2,
        "require_reference": require_reference,
        "fmt": fmt,
        "mi_statistic": score,
        **extra_inputs,
    }

    def score_data(data, row_weights, mi=None):
        pair_scores = {}
        if permutations or labels:
            mi = mi if mi is not None else triplet_mi(data, row_weights, min_valid=min_valid)
            pair_scores["mi"] = {
                pair: getattr(getattr(mi, pair.lower()), score) for pair in ("AB", "AC", "BC")
            }
        dca_blocks = (
            _fit("dca", direct_couplings, data, row_weights, ridge=ridge, min_valid=min_valid)
            if dca
            else {}
        )
        if dca:
            pair_scores["dca"] = {pair: block.frobenius for pair, block in dca_blocks.items()}
        diagnostics = {}
        triples = []
        if threebody and not fixed:
            diagnostics = {"status": "no eligible triples"}
        elif threebody and threebody_model == "global":
            triples, diagnostics = _fit(
                "threebody",
                global_triples,
                data,
                fixed,
                row_weights,
                l2=global_l2,
                min_valid=min_valid,
            )
        elif threebody:
            triples = _fit(
                "threebody",
                fit_triples,
                data,
                fixed,
                row_weights,
                pseudocount=pseudocount,
                min_valid=min_valid,
            )
        vectors = {
            key: np.concatenate(
                [blocks["ABC"[a] + "ABC"[b]][np.ix_(axes[a], axes[b])].ravel() for a, b in PAIRS]
            )
            for key, blocks in pair_scores.items()
        }
        if threebody:
            vectors["threebody"] = np.array([row.threebody_rms for row in triples])
        return pair_scores, dca_blocks, triples, vectors, diagnostics

    pair_scores, dca_blocks, triples, observed, diagnostics = score_data(msa, weights, mi_result)
    null = {}
    if permutations:
        comparisons = _WeightComparisons(msa.components, reuse=True)

        def refit(shuffled):
            w = comparisons.weights(
                weights.identity_threshold, weights.min_overlap, shuffled.components
            )
            return score_data(shuffled, w)[3]

        null = permutation_test(
            msa, observed, refit, permutations=permutations, seed=seed, strata=strata
        )
    out.mkdir(parents=True, exist_ok=True)

    def write(name, fields, rows):
        outputs[name] = _table(out / f"{name}.{fmt}", fields, rows, fmt)

    def pair_row(a, b, i, j):
        return {"pair": "ABC"[a] + "ABC"[b], **_site(maps[a], i, "x"), **_site(maps[b], j, "y")}

    def triple_row(index):
        return {
            key: value
            for prefix, m, j in zip("abc", maps, fixed[index])
            for key, value in _site(m, j, prefix).items()
        }

    def ranked_pair_rows(a, b):
        pair = "ABC"[a] + "ABC"[b]
        block = dca_blocks[pair]
        values = block.frobenius[np.ix_(axes[a], axes[b])]
        order = np.argsort(-values.ravel(), kind="stable")
        for rank, index in enumerate(order[:max_pairs], 1):
            x, y = np.unravel_index(index, values.shape)
            i, j = axes[a][x], axes[b][y]
            if not np.isfinite(values[x, y]):
                break
            yield {
                "rank": rank,
                **pair_row(a, b, i, j),
                "dca_frobenius": values[x, y],
                "dca_apc": block.corrected[i, j],
                "n_valid": block.n_valid[i, j],
                "valid_weight": block.valid_weight[i, j],
            }

    if dca:
        for a, b in PAIRS:
            pair = "ABC"[a] + "ABC"[b]
            write(
                f"dca_{pair.lower()}",
                ["rank", *PAIR_FIELDS, "dca_frobenius", "dca_apc", "n_valid", "valid_weight"],
                ranked_pair_rows(a, b),
            )
    if threebody:
        triple_fields = [
            "threebody_rms",
            "co_information_bits",
            "n_valid",
            "valid_weight",
            "observed_cells",
            "possible_cells",
        ]
        order = np.argsort(-observed["threebody"], kind="stable")
        write(
            "threebody_abc",
            ["rank", *TRIPLE_FIELDS, *triple_fields],
            (
                {"rank": rank, **triple_row(i), **vars(triples[i])}
                for rank, i in enumerate(order, 1)
                if np.isfinite(observed["threebody"][i])
            ),
        )

    def coordinate_rows(method):
        if method == "threebody":
            for i in range(len(fixed)):
                yield triple_row(i)
        else:
            for a, b in PAIRS:
                for i in axes[a]:
                    for j in axes[b]:
                        yield pair_row(a, b, i, j)

    for method, (p, q) in null.items():
        statistic = {"mi": score, "dca": "dca_frobenius", "threebody": "threebody_rms"}[method]
        write(
            "permutation_" + method,
            [*(TRIPLE_FIELDS if method == "threebody" else PAIR_FIELDS), *NULL_FIELDS],
            (
                {
                    **row,
                    "statistic": statistic,
                    "score": value,
                    "permutation_p": pv,
                    "bh_q": qv,
                    "null_draws": permutations,
                }
                for row, value, pv, qv in zip(coordinate_rows(method), observed[method], p, q)
                if np.isfinite(value)
            ),
        )
    if labels:
        methods = {}
        for method, blocks in pair_scores.items():
            labelled_scores = {}
            for a, b in PAIRS:
                pair = "ABC"[a] + "ABC"[b]
                x = {pos: i for i, pos in enumerate(maps[a].reference_positions) if pos is not None}
                y = {pos: i for i, pos in enumerate(maps[b].reference_positions) if pos is not None}
                for key in labels:
                    name, i, j = key
                    if name == pair and i in x and j in y:
                        labelled_scores[key] = blocks[pair][x[i], y[j]]
            methods[method] = contact_metrics(labels, labelled_scores)
        result = {"methods": methods, "mi_statistic": score}
        if dca:
            result["dca_statistic"] = "dca_frobenius"
        (out / "contacts.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        outputs["contacts"] = {"path": "contacts.json", "rows": len(labels)}
    return {
        "settings": settings,
        "outputs": outputs,
        "global_fit": diagnostics,
        "candidates": {
            "eligible": eligible,
            "tested": len(fixed),
            "finite": int(np.count_nonzero(np.isfinite(observed.get("threebody", [])))),
            "selection": selection,
        },
        "hypotheses": {k: len(v) for k, v in observed.items()},
        "null_model": "independent whole-component row permutations within strata"
        if permutations
        else None,
    }
