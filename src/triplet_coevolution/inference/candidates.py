"""Reproducible ABC candidate selection on a declared coordinate and evidence universe."""

import csv
import hashlib
import heapq
import itertools
import json
import math
import sys
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from random import Random
from typing import Literal, NamedTuple

import numpy as np

from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.mi import _get_weights, weighted_mi
from triplet_coevolution.inference.scores import get_score
from triplet_coevolution.inference.weights import (
    SequenceWeightResult,
    sequence_weights,
    verify_weight_alignment,
)
from triplet_coevolution.msa.synchronize import synchronize

MODES = ("uniform", "exhaustive", "explicit", "prior", "triangle", "wedge", "entropy", "hybrid")
SCREENED = {"triangle", "wedge", "entropy", "hybrid"}
PAIRS = (("AB", 0, 1), ("AC", 0, 2), ("BC", 1, 2))
ROLES = "abc"
SITE_FIELDS = ("col", "original_col", "reference_id", "reference_pos", "symbol")
ROW_FIELDS = [
    "candidate_id",
    *(f"{r}_{f}" for r in ROLES for f in SITE_FIELDS),
    "rationale",
    "screen_value",
    "selection_scope",
    "file_origin",
]


def _json(data):
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _digest(data):
    return hashlib.sha256(_json(data)).hexdigest()


def candidate_id(maps, triple):
    """Stable identifier for the three component/reference IDs and one-based source columns."""
    return _digest(
        [
            [m.component, m.reference_id, m.original_columns[i] + 1]
            for m, i in zip(maps, triple, strict=True)
        ]
    )


def _file_digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


CandidateMode = Literal[
    "uniform", "exhaustive", "explicit", "prior", "triangle", "wedge", "entropy", "hybrid"
]
Rationale = Literal[
    "uniform",
    "exhaustive",
    "explicit",
    "prior",
    "triangle",
    "wedge",
    "entropy",
    "uniform_reserved",
    "external_prior",
]


class CandidateReason(NamedTuple):
    mechanism: Rationale
    screen_value: float | None


@dataclass(frozen=True)
class CandidateSelection:
    triples: tuple[tuple[int, int, int], ...]
    reasons: tuple[CandidateReason, ...]
    maps: tuple
    summary: dict

    def batches(self, size=128):
        """Iterate the fixed order without allocating another full copy of the selection."""
        size = _integer(size, "batch size", 1)
        for i in range(0, len(self.triples), size):
            yield self.triples[i : i + size]

    def rows(self):
        for triple, (method, value) in zip(self.triples, self.reasons, strict=True):
            row = {}
            for role, m, index in zip(ROLES, self.maps, triple, strict=True):
                site = m.site(index)
                row.update(
                    zip(
                        (f"{role}_{k}" for k in SITE_FIELDS),
                        (
                            site.current_column,
                            site.original_column,
                            m.reference_id,
                            site.reference_position,
                            site.reference_symbol,
                        ),
                        strict=True,
                    )
                )
            yield {
                "candidate_id": candidate_id(self.maps, triple),
                **row,
                "rationale": method,
                "screen_value": value,
                "selection_scope": self.summary["scope"],
                "file_origin": self.summary["file_origin"],
            }


def write_candidate_manifest(directory, selection, fmt="tsv"):
    """Publish a machine-readable selection summary and the ordered coordinate table."""
    if fmt not in ("csv", "tsv"):
        raise ValueError("candidate selection format must be csv or tsv")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    table = root / f"candidate_selection.{fmt}"
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=ROW_FIELDS, delimiter="\t" if fmt == "tsv" else ","
        )
        writer.writeheader()
        writer.writerows(selection.rows())
    manifest = {
        **selection.summary,
        "coordinate_table": table.name,
        "coordinate_table_sha256": _file_digest(table),
    }
    (root / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


class CandidateInputError(ValueError):
    """Strict input failure with explicit duplicate, invalid and out-of-universe counts."""

    def __init__(self, message, counts):
        super().__init__(f"{message}; candidate input counts: {counts}")
        self.counts = dict(counts)


def _read_entries(path, maps, axes, maximum):
    originals = [
        {m.original_columns[i] + 1: i for i in axis} for m, axis in zip(maps, axes, strict=True)
    ]
    refs = [
        {m.reference_positions[i]: i for i in axis if m.reference_positions[i] is not None}
        for m, axis in zip(maps, axes, strict=True)
    ]
    src = {f"{r}_original_col" for r in ROLES}
    ref = {f"{r}_{key}" for r in ROLES for key in ("reference_id", "reference_pos")}
    counters = {"rows": 0, "duplicates": 0, "invalid": 0, "outside_axes": 0}
    seen = set()
    before = _file_digest(path)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", strict=True)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or set(fields) not in (src, ref):
            raise CandidateInputError(
                "triplet candidate TSV requires original columns or reference IDs and positions",
                counters,
            )
        by_original = set(fields) == src
        for entry in reader:
            counters["rows"] += 1
            if counters["rows"] > 1_000_000:
                raise CandidateInputError(
                    "candidate TSV exceeds the one-million-row intake ceiling", counters
                )
            if None in entry or None in entry.values():
                counters["invalid"] += 1
                continue
            try:
                if by_original:
                    triple = tuple(
                        lookup[int(entry[f"{r}_original_col"])]
                        for r, lookup in zip(ROLES, originals, strict=True)
                    )
                else:
                    if any(
                        entry[f"{r}_reference_id"] != m.reference_id
                        for r, m in zip(ROLES, maps, strict=True)
                    ):
                        raise KeyError("reference ID")
                    triple = tuple(
                        lookup[int(entry[f"{r}_reference_pos"])]
                        for r, lookup in zip(ROLES, refs, strict=True)
                    )
            except ValueError:
                counters["invalid"] += 1
                continue
            except KeyError:
                counters["outside_axes"] += 1
                continue
            if triple in seen:
                counters["duplicates"] += 1
            seen.add(triple)
            if len(seen) > maximum:
                raise CandidateInputError(
                    "explicit candidates exceed max_triplets or file intake budget", counters
                )
    if any(counters[k] for k in ("duplicates", "invalid", "outside_axes")):
        raise CandidateInputError(
            "duplicate candidate, malformed field, or ineligible original column/reference",
            counters,
        )
    if not seen:
        raise CandidateInputError("triplet candidate table is empty", counters)
    if before != _file_digest(path):
        raise ValueError("candidate TSV changed while being read")
    return sorted(seen), counters, before


def _axes(maps, axes):
    if tuple(m.component for m in maps) != tuple("ABC") or len(axes) != 3:
        raise ValueError("candidate coordinates require A, B, C axes and reference maps")
    result = []
    h = hashlib.sha256()
    for m, axis in zip(maps, axes, strict=True):
        values = tuple(_integer(i, "candidate site index", 0) for i in axis)
        if any(i >= m.n_sites for i in values) or any(
            a >= b for a, b in itertools.pairwise(values)
        ):
            raise ValueError("candidate axes require distinct increasing in-range indices")
        result.append(values)
        h.update(_json([m.component, m.reference_id, m.start, m.strand]))
        for i in values:
            site = m.site(i)
            h.update(
                _json(
                    [
                        i,
                        site.current_column,
                        site.original_column,
                        site.reference_position,
                        site.reference_symbol,
                    ]
                )
            )
    return tuple(result), h.hexdigest()


def _draw(total, n, seed):
    if n >= total:
        return range(total)
    rng = Random(seed)
    if total <= sys.maxsize:
        return sorted(rng.sample(range(total), n))
    # Floyd sampling for arbitrary integer products beyond Py_ssize_t.
    chosen = set()
    for j in range(total - n, total):
        k = rng.randrange(j + 1)
        chosen.add(j if k in chosen else k)
    return sorted(chosen)


def _decode(codes, axes):
    lb, lc = len(axes[1]), len(axes[2])
    for code in codes:
        ab, c = divmod(code, lc)
        a, b = divmod(ab, lb)
        yield axes[0][a], axes[1][b], axes[2][c]


def _screen_inputs(msa, weights, training_ids):
    if msa is None:
        if training_ids is not None or weights is not None:
            raise ValueError("candidate training or weights require a matched alignment")
        return None, None, None, None
    ids = tuple(m.observation_id for m in msa.matches)
    verify_weight_alignment(msa, weights)
    if training_ids is not None:
        if isinstance(training_ids, (str, bytes)):
            raise ValueError("training IDs must be a collection of observation identifiers")
        selected = tuple(training_ids)
        members = set(selected)
        if not selected or len(members) != len(selected) or not members < set(ids):
            raise ValueError(
                "training IDs must be a nonempty unique proper subset of the matched cohort"
            )
        records = tuple(m for m in msa.matches if m.observation_id in members)
        msa = synchronize(*msa.components, records)
        if weights is not None:
            if not isinstance(weights, SequenceWeightResult) or weights.observation_ids != ids:
                raise ValueError(
                    "training-only screening requires a declared observation-keyed weighting policy"
                )
            extra = {}
            if weights.policy == "supplied":
                extra["supplied_weights"] = {
                    k: float(w) for k, w in zip(ids, weights.weights) if k in members
                }
            if weights.policy == "clade_balanced":
                extra["clades"] = {
                    k: clade for k, clade in zip(ids, weights.clades) if k in members
                }
            weights = sequence_weights(
                msa,
                weights.identity_threshold,
                weights.min_overlap,
                policy=weights.policy,
                component=weights.component,
                memory_budget_bytes=weights.preflight.get("memory_budget_bytes"),
                **extra,
            )
        ids = tuple(m.observation_id for m in records)
    w = _get_weights(weights, len(ids), msa)
    h = hashlib.sha256()
    h.update(_json(ids))
    for part in msa.components:
        h.update(_json([part.name, part.molecule.value, part.ids]))
        for row in part.sequences:
            h.update(row.encode("ascii"))
    h.update(np.asarray(w, dtype="<f8").tobytes())
    return msa, w, _digest(ids), h.hexdigest()


class _Evidence:
    def __init__(
        self,
        msa,
        weights,
        pair_scores,
        pair_statistic,
        maps,
        axes,
        needs_pairs,
        needs_sites,
        min_entropy_bits,
        min_site_coverage,
    ):
        self.columns = msa.components if msa is not None else ()
        self.weights = weights
        self.blocks = {} if pair_scores is not None else None
        self.pairs = {}
        self.sites = None
        self.hashes = {}
        if msa is not None and any(
            c.name != m.component or c.n_sites != m.n_sites
            for c, m in zip(self.columns, maps, strict=True)
        ):
            raise ValueError("candidate screen alignment columns disagree with coordinate maps")
        if needs_pairs:
            spec = get_score(pair_statistic)
            if spec.method not in ("mi", "dca") or spec.direction != "higher":
                raise ValueError("candidate pair screen requires a higher-tail pair statistic")
            if pair_scores is None and (msa is None or pair_statistic != "mi_bits"):
                raise ValueError(
                    "pair screen requires supplied matrices or raw mi_bits from the screening cohort"
                )
            if pair_scores is not None:
                if set(pair_scores) != {"AB", "AC", "BC"}:
                    raise ValueError("candidate pair score matrices must cover AB, AC and BC")
                for name, a, b in PAIRS:
                    vals = np.asarray(pair_scores[name])
                    if (
                        vals.shape != (maps[a].n_sites, maps[b].n_sites)
                        or vals.dtype.kind not in "fiu"
                    ):
                        raise ValueError(
                            f"invalid retained coordinate shape or type in candidate {name} scores"
                        )
                    self.blocks[name] = vals
                    h = hashlib.sha256()
                    for row in vals:
                        if np.isinf(row).any():
                            raise ValueError("candidate pair matrices use NaN for missing scores")
                        h.update(np.asarray(row, dtype="<f8").tobytes())
                    self.hashes[name] = h.hexdigest()
        if needs_sites:
            if msa is None:
                raise ValueError(
                    "entropy/support screens require a matched alignment and its weights"
                )
            self.sites = []
            for c, axis in zip(self.columns, axes, strict=True):
                site_scores = {}
                q = len(CANONICAL[c.molecule])
                weight_sum = weights.sum()
                for j in axis:
                    col = c.matrix[:, j]
                    valid = (col >= 0) & (col < q) & (weights > 0)
                    freq = np.bincount(col[valid], weights=weights[valid], minlength=q)
                    mass = freq.sum()
                    if mass > 0:
                        prob = freq[freq > 0] / mass
                        entropy = float(-np.dot(prob, np.log2(prob)))
                        coverage = min(1.0, float(mass / weight_sum))
                        if entropy >= min_entropy_bits and coverage >= min_site_coverage:
                            site_scores[j] = float(entropy * coverage)
                self.sites.append(site_scores)

    def local_pair(self, name, a, b, i, j):
        if self.blocks is not None:
            return self.blocks[name][i, j]
        key = (name, i, j)
        if key not in self.pairs:
            if len(self.pairs) >= 8192:
                self.pairs.clear()
            x, y = self.columns[a], self.columns[b]
            self.pairs[key] = weighted_mi(
                x.matrix[:, i],
                y.matrix[:, j],
                len(CANONICAL[x.molecule]),
                len(CANONICAL[y.molecule]),
                self.weights,
            )
        return self.pairs[key]

    def score(self, method, triple):
        if method == "entropy":
            values = [s.get(j) for s, j in zip(self.sites, triple, strict=True)]
            return min(values) if all(x is not None for x in values) else None
        v = [self.local_pair(name, a, b, triple[a], triple[b]) for name, a, b in PAIRS]
        scores = sorted((float(x) for x in v if np.isfinite(x)), reverse=True)
        required = 3 if method == "triangle" else 2
        return scores[required - 1] if len(scores) >= required else None


def _push(heap, limit, score, triple):
    # For equal score prefer the earlier original-coordinate tuple.
    item = (score, tuple(-x for x in triple))
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _ranked(heap):
    return [(tuple(-x for x in neg), s) for s, neg in sorted(heap, reverse=True)]


def select_candidates(
    maps,
    axes,
    *,
    mode: CandidateMode = "uniform",
    maximum=1000,
    seed=0,
    candidate_file=None,
    prior_source=None,
    file_origin=None,
    pair_scores=None,
    pair_statistic="mi_bits",
    msa=None,
    weights=None,
    min_entropy_bits=0.0,
    min_site_coverage=0.0,
    uniform_quota=None,
    screen_budget=100000,
    training_ids=None,
):
    """Select a fixed source-mapped set, bounded by the requested fit and screen budgets.

    In data-dependent modes, a single seeded probability sample bounds the number of
    cheap screens if the coordinate product exceeds screen_budget. Hybrid reserves
    independent uniform coordinates first and alternates screened wedges and entropy.
    """
    maximum = _integer(maximum, "max_triplets", 1)
    screen_budget = _integer(screen_budget, "screen_budget", 1)
    seed = _integer(seed, "seed", 0)
    if mode not in MODES:
        raise ValueError(f"unknown candidate selection mode {mode!r}")
    maps = tuple(maps)
    axes, axis_hash = _axes(maps, axes)
    total = math.prod(map(len, axes))
    target = min(maximum, total)
    if mode == "exhaustive" and total > maximum:
        raise ValueError(f"exhaustive candidate universe {total} exceeds max_triplets {maximum}")
    if target > 100_000 and mode not in ("explicit", "prior"):
        raise ValueError(
            "candidate selection exceeds the supported 100000 fitted candidates; reduce max_triplets or the panel"
        )
    if mode in SCREENED and min(screen_budget, total) > 1_000_000:
        raise ValueError(
            "candidate screen exceeds the supported 1000000 probes; reduce screen_budget or the panel"
        )
    if mode == "hybrid":
        quota = (
            max(1, (target + 3) // 4)
            if uniform_quota is None
            else _integer(uniform_quota, "hybrid uniform quota", 1)
        )
        if quota > maximum:
            raise ValueError("hybrid uniform quota exceeds max_triplets")
        quota = min(target, quota)
    elif uniform_quota is not None:
        raise ValueError("uniform quota applies only to hybrid candidate selection")
    else:
        quota = 0
    if candidate_file is not None and mode not in ("explicit", "prior", "hybrid"):
        raise ValueError("candidate file requires explicit, prior or hybrid selection")
    if (mode in ("prior", "hybrid") and candidate_file is not None) != (prior_source is not None):
        raise ValueError("a prior candidate file requires an explicit evidence source")
    if prior_source is not None and (not isinstance(prior_source, str) or not prior_source.strip()):
        raise ValueError("candidate prior source must be nonempty")
    if mode in ("explicit", "prior") and candidate_file is None:
        raise ValueError("explicit and prior modes require a candidate TSV")
    if file_origin is not None and (
        mode != "explicit" or file_origin not in ("prespecified", "same_data", "training")
    ):
        raise ValueError(
            "file_origin applies to explicit candidate files and must be prespecified, same_data or training"
        )
    declared_origin = (
        file_origin if file_origin is not None else "prespecified" if mode == "explicit" else None
    )
    if training_ids is not None and (mode not in SCREENED or pair_scores is not None):
        raise ValueError(
            "training-only screening requires internal fits from the specified training rows"
        )
    needs_pairs = mode in ("triangle", "wedge", "hybrid")
    needs_sites = mode in ("entropy", "hybrid")
    if pair_scores is not None and not needs_pairs:
        raise ValueError("candidate pair matrices require a pair screen")
    for v, name, hi in (
        (min_entropy_bits, "minimum entropy", math.inf),
        (min_site_coverage, "minimum site coverage", 1),
    ):
        if (
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, Real)
            or not math.isfinite(v)
            or not 0 <= v <= hi
        ):
            raise ValueError(f"invalid candidate {name}")
    if not needs_sites and (min_entropy_bits or min_site_coverage):
        raise ValueError("candidate site thresholds require entropy or hybrid mode")
    entries = []
    inputs = None
    digest = None
    if candidate_file is not None:
        entries, inputs, digest = _read_entries(
            candidate_file, maps, axes, min(100_000, screen_budget if mode == "hybrid" else maximum)
        )
    chosen = {}
    duplicate_hits = 0
    used_hashes = {}
    screened = 0
    missing = {}
    cohort_hash = source_hash = cohort_count = None
    if mode in ("uniform", "exhaustive"):
        chosen = {t: (mode, None) for t in _decode(_draw(total, target, seed), axes)}
    elif mode in ("explicit", "prior"):
        chosen = {t: (mode, None) for t in entries}
    elif target:
        selected_msa, w, cohort_hash, source_hash = _screen_inputs(msa, weights, training_ids)
        cohort_count = selected_msa.n_observations if selected_msa is not None else None
        evidence = _Evidence(
            selected_msa,
            w,
            pair_scores,
            pair_statistic,
            maps,
            axes,
            needs_pairs,
            needs_sites,
            min_entropy_bits,
            min_site_coverage,
        )
        used_hashes = evidence.hashes
        if quota:
            chosen = {
                t: ("uniform_reserved", None) for t in _decode(_draw(total, quota, seed), axes)
            }
        for t in entries:
            if t in chosen:
                duplicate_hits += 1
            elif len(chosen) < target:
                chosen[t] = ("external_prior", None)
        capacity = target - len(chosen)
        if capacity:
            methods = ("wedge", "entropy") if mode == "hybrid" else (mode,)
            heaps = {m: [] for m in methods}
            missing = dict.fromkeys(methods, 0)
            pool_seed = int(_digest(["candidate_screen_v1", seed]), 16)
            for t in _decode(_draw(total, min(total, screen_budget), pool_seed), axes):
                screened += 1
                if t in chosen:
                    duplicate_hits += 1
                    continue
                for method in methods:
                    score = evidence.score(method, t)
                    if score is None:
                        missing[method] += 1
                    else:
                        _push(heaps[method], capacity, score, t)
            ordered = {method: _ranked(heaps[method]) for method in methods}
            for rank in range(capacity):
                for method in methods:
                    if rank >= len(ordered[method]) or len(chosen) >= target:
                        continue
                    triple, value = ordered[method][rank]
                    if triple in chosen:
                        duplicate_hits += 1
                    else:
                        chosen[triple] = (method, value)
    triples = tuple(sorted(chosen))
    reasons = tuple(CandidateReason(*chosen[t]) for t in triples)
    mapped = hashlib.sha256()
    for t in triples:
        mapped.update(_json([m.original_columns[i] + 1 for m, i in zip(maps, t, strict=True)]))
    same_data = (mode in SCREENED and training_ids is None) or declared_origin == "same_data"
    if mode in SCREENED:
        scope = "training_rows_only" if training_ids is not None else "same_data_exploratory"
    elif mode == "explicit":
        scope = {
            "prespecified": "user_supplied_file",
            "same_data": "same_data_exploratory",
            "training": "user_declared_training_file",
        }[declared_origin]
    elif mode == "prior":
        scope = "user_declared_external_prior"
    else:
        scope = "coordinate_only"
    summary = {
        "schema": "triplet.candidate_selection.v1",
        "mode": mode,
        "scope": scope,
        "file_origin": declared_origin or ("external_prior" if prior_source else None),
        "file_origin_provenance": "declared"
        if file_origin or prior_source
        else "assumed_legacy_default"
        if mode == "explicit"
        else None,
        "seed": seed,
        "retained_cartesian_triples": math.prod(m.n_sites for m in maps),
        "eligible_triples": total,
        "axes": {
            m.component: {
                "retained_sites": m.n_sites,
                "reference_numbered_sites": sum(v is not None for v in m.reference_positions),
                "eligible_sites": len(axis),
                "reference_id": m.reference_id,
            }
            for m, axis in zip(maps, axes, strict=True)
        },
        "coordinate_exclusions": math.prod(m.n_sites for m in maps) - total,
        "maximum": maximum,
        "selected": len(triples),
        "exhaustive": len(triples) == total,
        "coordinate_axes_sha256": axis_hash,
        "selected_original_coordinates_sha256": mapped.hexdigest(),
        "ranking_screen": pair_statistic if needs_pairs else None,
        "pair_screen_source": (
            "supplied_retained_matrices"
            if pair_scores is not None
            else "internally_fitted_canonical_MI_min_valid_2"
        )
        if needs_pairs
        else None,
        "probe_pool_seed_recipe": "SHA-256 of canonical JSON [candidate_screen_v1, seed], interpreted as an integer",
        "pair_score_sha256": used_hashes,
        "screen_budget": screen_budget,
        "screened_triples": screened,
        "outside_screen_pool": total - screened if mode in SCREENED else None,
        "screen_ineligible": missing,
        "min_site_coverage": float(min_site_coverage),
        "min_entropy_bits": float(min_entropy_bits),
        "uniform_reserved": quota,
        "rationale_counts": {
            name: sum(r[0] == name for r in reasons) for name in sorted({r[0] for r in reasons})
        },
        "file_sha256": digest,
        "file_counts": inputs,
        "prior_source": prior_source,
        "unselected_file_rows": sum(t not in chosen for t in entries),
        "duplicate_proposals": duplicate_hits,
        "screen_observation_ids_sha256": cohort_hash,
        "screen_observation_count": cohort_count,
        "reserved_inclusion_probability": (
            (target if mode in ("uniform", "exhaustive") else quota) / total
        )
        if total and mode in ("uniform", "exhaustive", "hybrid")
        else None,
        "screen_source_sha256": source_hash,
        "evaluated_without_independent_holdout": mode in SCREENED
        or declared_origin in ("training", "same_data"),
        "same_data_screen": same_data,
        "used_external_prior": prior_source is not None,
        "screen_rules": {
            "triangle": "weakest of three finite pair scores",
            "wedge": "second highest of at least two finite pair scores",
            "entropy": "minimum of site Shannon entropy in bits times fraction of positive weight with canonical observations",
            "hybrid": "reserve uniform first, then prior in source-coordinate order, then alternate wedge and entropy proposals",
        },
        "interpretation": "Ranks describe fitted candidates. Data-screened hypotheses need separate evaluation rows or selection-aware calibration. Prior-guided selection requires independent validation evidence.",
    }
    return CandidateSelection(triples, reasons, maps, summary)
