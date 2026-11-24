"""Inspect synchronized row/site decisions without fitting a coevolution model."""

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path

import numpy as np

from triplet_coevolution.core.alignment import MatchedTripletMSA
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.core.coordinates import ReferenceCoordinates, subset_reference_map
from triplet_coevolution.io.metadata import TripletTable
from triplet_coevolution.msa._qc_policy import COMPONENTS, QCPolicy, RowRule, SiteRule
from triplet_coevolution.msa.masking import MaskedTriplet, apply_triplet_masks, column_mask
from triplet_coevolution.msa.qc import TripletQC, triplet_qc
from triplet_coevolution.msa.support import MissingRowSupport

__all__ = ["QCPolicy", "QCReport", "inspect_qc", "write_qc"]


@dataclass
class QCReport:
    row_indices: tuple[int, ...]
    masks: tuple[np.ndarray, ...]
    filtered: MaskedTriplet | None
    maps: tuple[ReferenceCoordinates, ...]
    qc: TripletQC
    rows: list[dict]
    sites: list[dict]
    summary: dict
    blockers: tuple[str, ...]

    def require_scoring(self) -> None:
        if self.blockers:
            raise MissingRowSupport(" ".join(self.blockers))


def _subset(msa, indices):
    if len(indices) == msa.n_observations:
        return msa
    components = [comp.take_ids([comp.ids[i] for i in indices]) for comp in msa.components]
    return MatchedTripletMSA(*components, tuple(msa.matches[i] for i in indices))


def _ratio(num, den):
    return np.divide(num, den, out=np.full(np.shape(num), np.nan), where=np.asarray(den) > 0)


def _row_metrics(component, mapping, mask, full_lengths):
    n = component.n_sequences
    q = len(CANONICAL[component.molecule])
    x = component.matrix[:, mask]
    width = x.shape[1]
    valid = (x >= 0) & (x < q)
    gaps = x == q
    counts = np.stack([(x == i).sum(axis=1) for i in range(q)], axis=1)
    num = counts.sum(axis=1)
    prob = np.divide(counts, np.maximum(num, 1)[:, None])
    logs = np.log2(prob, out=np.zeros_like(prob), where=prob > 0)
    entropy = -(prob * logs).sum(axis=1)
    entropy[num == 0] = np.nan
    refs = np.array([pos is not None for pos in mapping.reference_positions])[mask]
    n_ref = valid[:, refs].sum(axis=1)
    left = np.where(num > 0, valid.argmax(axis=1), width) if width else np.zeros(n, dtype=int)
    right = (
        np.where(num > 0, valid[:, ::-1].argmax(axis=1), width) if width else np.zeros(n, dtype=int)
    )
    # Exclude both terminal stretches when measuring internal gap runs.
    internal = np.zeros(n, dtype=int)
    if width:
        cols = np.arange(width)
        internal = (gaps & (cols >= left[:, None]) & (cols < width - right[:, None])).sum(axis=1)
    med = np.median(full_lengths)
    return {
        "canonical_count": num,
        "site_count": np.full(n, width),
        "coverage": _ratio(num, width),
        "gap_count": gaps.sum(axis=1),
        "missing_count": (x < 0).sum(axis=1),
        "gap_fraction": _ratio(gaps.sum(axis=1), width),
        "missing_fraction": _ratio((x < 0).sum(axis=1), width),
        "reference_sites": np.full(n, int(refs.sum())),
        "reference_canonical": n_ref,
        "reference_coverage": _ratio(n_ref, refs.sum()),
        "entropy_bits": entropy,
        "state_count": (counts > 0).sum(axis=1),
        "left_noncanonical": left,
        "right_noncanonical": right,
        "internal_gap_count": internal,
        "full_canonical_length": full_lengths,
        "length_ratio": _ratio(full_lengths, med),
    }


def _plain(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def _row_reasons(rule, values):
    fields = {
        "min_coverage": "coverage",
        "min_canonical": "canonical_count",
        "max_canonical": "canonical_count",
        "min_reference_coverage": "reference_coverage",
        "min_entropy_bits": "entropy_bits",
        "min_length_ratio": "length_ratio",
        "max_length_ratio": "length_ratio",
    }
    failed = {}
    for name, limit in asdict(rule).items():
        if limit is not None:
            value = values[fields[name]]
            failed[name] = ~np.isfinite(value) | (
                value < limit if name.startswith("min") else value > limit
            )
    return failed


def _column_rules(options, policy):
    default = {"min_coverage": 0.5, "max_gap_fraction": 0.5, "max_missing_fraction": 0.5}
    base = default | dict(options)
    if any(base[name] is None for name in default):
        raise ValueError("column coverage and maximum gap/missing fractions must be numeric")
    if set(base) - set(asdict(SiteRule())):
        raise ValueError("unknown column QC threshold")
    validated = QCPolicy(
        sites={
            c: base
            | {k: v for k, v in asdict(policy.sites.get(c, SiteRule())).items() if v is not None}
            for c in COMPONENTS
        }
    )
    return {
        c: {k: v for k, v in asdict(r).items() if v is not None} for c, r in validated.sites.items()
    }


def _site_decisions(qc, options, panel):
    masks, reasons = [], []
    for comp, selected in zip(qc.components, panel, strict=True):
        opts = options[comp.name]
        ordinary = {k: v for k, v in opts.items() if k not in ("min_canonical", "min_states")}
        usable = column_mask(comp, **ordinary)
        flags = {
            "outside_panel": ~selected,
            "no_canonical": comp.n_canonical == 0,
            "min_coverage": comp.coverage < opts["min_coverage"],
            "max_gap_fraction": comp.gap_fraction > opts["max_gap_fraction"],
            "max_missing_fraction": comp.missing_fraction > opts["max_missing_fraction"],
        }
        if "min_entropy_bits" in opts:
            flags["min_entropy_bits"] = ~np.isfinite(comp.entropy_bits) | (
                comp.entropy_bits < opts["min_entropy_bits"]
            )
        counts = comp.state_counts
        if "min_canonical" in opts:
            flags["min_canonical"] = comp.n_canonical < opts["min_canonical"]
        if "min_states" in opts:
            flags["min_states"] = (counts > 0).sum(axis=0) < opts["min_states"]
        masks.append(usable & selected & ~np.logical_or.reduce(tuple(flags.values())))
        reasons.append(flags)
    return tuple(masks), reasons


def _dup_frequencies(msa, kept, masks):
    # Codes include the encoded gap and unknown states. This groups scoring-
    # equivalent sequences even when input aliases encode the same missing state.
    signatures = [
        tuple(comp.matrix[i, mask].tobytes() for i in kept)
        for comp, mask in zip(msa.components, masks, strict=True)
    ]
    signatures.append(tuple(zip(*signatures, strict=True)))
    summary, members = {}, {}
    for label, rows in zip((*COMPONENTS, "ABC"), signatures, strict=True):
        if not kept or any(not masks[COMPONENTS.index(c)].any() for c in label):
            members[label] = {}
            summary[label] = {"status": "no retained observations or sites"}
            continue
        groups = Counter(rows)
        first = {}
        assignments = {}
        for i, sig in zip(kept, rows, strict=True):
            first.setdefault(sig, msa.matches[i].observation_id)
            assignments[i] = (first[sig], groups[sig])
        members[label] = assignments
        summary[label] = {
            "status": "measured",
            "groups": len(groups),
            "duplicate_rows": len(kept) - len(groups),
            "largest_group": max(groups.values(), default=0),
            "group_size_histogram": dict(sorted(Counter(groups.values()).items())),
        }
    return summary, members


def _categories(table, kept, policy):
    result = {}
    for kind, explicit, conventional in (
        ("taxon", policy.taxon_column, "taxon_id"),
        ("clade", policy.clade_column, "clade"),
    ):
        key = explicit or conventional
        if key not in table.extra_columns:
            if explicit is not None:
                raise ValueError(f"unknown QC {kind} column {key!r} in match table")
            result[kind] = {"status": "unavailable", "column": None}
            continue

        def population(rows, key=key):
            counts = Counter(table.annotations[i][key] for i in rows)
            missing = counts.pop("", 0)
            return {
                "counts": dict(sorted(counts.items())),
                "missing": missing,
                "distinct": len(counts),
            }

        result[kind] = {
            "status": "supplied",
            "column": key,
            "input": population(range(table.n_observations)),
            "retained": population(kept),
        }
    return result


def _joint(metrics, kept):
    # For a given observation the fraction of fully canonical pairs on the
    # Cartesian grid factors into its two marginal fractions, and likewise for ABC.
    result = {}
    for label in ("AB", "AC", "BC", "ABC"):
        cols = [metrics[COMPONENTS.index(c)] for c in label]
        total = int(np.prod([int(v["site_count"][0]) for v in cols], dtype=object))
        den = len(kept) * total

        def n_cells(name, cols=cols):
            return sum(prod(int(v[name][i]) for v in cols) for i in kept)

        canonical, gap = int(n_cells("canonical_count")), int(n_cells("gap_count"))
        result[label] = {
            "site_combinations": total,
            "observation_cells": den,
            "joint_canonical_cells": canonical,
            "joint_gap_cells": gap,
            "joint_canonical_fraction": canonical / den if den else None,
            "joint_gap_fraction": gap / den if den else None,
        }
    return result


def inspect_qc(
    matched: MatchedTripletMSA,
    maps: tuple[ReferenceCoordinates, ...],
    table: TripletTable,
    *,
    column_options: dict | None = None,
    panel_masks: tuple[np.ndarray, ...] | None = None,
    policy: QCPolicy | None = None,
) -> QCReport:
    """Return row decisions, source-column masks, original coordinates and measurable support."""
    if table.matches != matched.matches:
        raise ValueError("QC metadata must correspond to synchronized observation order and IDs")
    if len(maps) != 3 or any(
        c.name != m.component or c.n_sites != m.n_sites for c, m in zip(matched.components, maps)
    ):
        raise ValueError("QC reference maps must match each input component")
    policy = policy or QCPolicy()
    opts = _column_rules(column_options or {}, policy)
    panel = (
        tuple(np.ones(c.n_sites, dtype=bool) for c in matched.components)
        if panel_masks is None
        else tuple(np.asarray(v) for v in panel_masks)
    )
    if len(panel) != 3 or any(
        v.dtype != bool or v.shape != (c.n_sites,) for v, c in zip(panel, matched.components)
    ):
        raise ValueError("QC panel requires one boolean mask per source component")
    all_idx = tuple(range(matched.n_observations))
    full_lengths = tuple(
        ((c.matrix >= 0) & (c.matrix < len(CANONICAL[c.molecule]))).sum(axis=1)
        for c in matched.components
    )

    def row_metrics(masks):
        return [
            _row_metrics(c, m, v, raw)
            for c, m, v, raw in zip(matched.components, maps, masks, full_lengths, strict=True)
        ]

    initial_qc = triplet_qc(matched)
    idx = all_idx
    masks = panel
    audits = {}
    decisions = {}
    history = []
    limit = policy.max_passes or matched.n_observations + 2
    settled = False
    for turn in range(1, limit + 1):
        before = idx
        old_masks = masks
        if policy.order == "columns_first":
            qc = initial_qc if idx == all_idx else triplet_qc(_subset(matched, idx))
            masks, mask_flags = _site_decisions(qc, opts, panel)
        decision_masks = masks
        vectors = row_metrics(decision_masks)
        reasons = [
            _row_reasons(policy.rows.get(c, RowRule()), values)
            for c, values in zip(COMPONENTS, vectors)
        ]
        rejected = []
        for i in idx:
            failures = [tuple(k for k, arr in flags.items() if arr[i]) for flags in reasons]
            if any(failures):
                rejected.append(i)
                for c, values, why in zip(COMPONENTS, vectors, failures):
                    audits[i, c] = {k: _plain(v[i]) for k, v in values.items()}
                    decisions[i, c] = turn, why
        excluded = set(rejected)
        idx = tuple(i for i in idx if i not in excluded)
        if policy.order == "rows_first" and idx:
            qc = initial_qc if idx == all_idx else triplet_qc(_subset(matched, idx))
            masks, mask_flags = _site_decisions(qc, opts, panel)
        if not idx:
            # No column has support after all observations have been excluded.
            masks = tuple(np.zeros_like(v) for v in panel)
            mask_flags = [{"no_observations": np.ones_like(v), "outside_panel": ~v} for v in panel]
            qc = initial_qc
        changes = {
            m.component: {
                "removed": [m.original_columns[i] + 1 for i in np.flatnonzero(old & ~new)],
                "reentered": [m.original_columns[i] + 1 for i in np.flatnonzero(~old & new)],
            }
            for m, old, new in zip(maps, old_masks, masks)
        }
        history.append(
            {
                "pass": turn,
                "input_rows": len(before),
                "excluded_rows": len(rejected),
                "retained_rows": len(idx),
                "column_changes": changes,
                "row_test_sites": {
                    c: int(mask.sum()) for c, mask in zip(COMPONENTS, decision_masks)
                },
                "column_test_observations": len(before)
                if policy.order == "columns_first"
                else len(idx),
            }
        )
        same_cols = all(np.array_equal(old, new) for old, new in zip(old_masks, masks))
        if not idx or (not rejected and (policy.order == "columns_first" or same_cols)):
            settled = True
        if policy.iteration == "single" or settled:
            break
    final_msa = _subset(matched, idx) if idx else None
    final_qc = (
        (qc if qc.n_observations == len(idx) else triplet_qc(final_msa)) if idx else initial_qc
    )
    final_metrics = (
        vectors
        if all(np.array_equal(a, b) for a, b in zip(decision_masks, masks))
        else row_metrics(masks)
    )
    blockers = []
    if not idx:
        blockers.append("QC excluded every matched observation.")
    if policy.iteration == "fixed_point" and not settled:
        blockers.append(f"QC did not converge within max_passes={limit}.")
    for c, mask, values in zip(COMPONENTS, masks, final_metrics):
        if not mask.any():
            blockers.append(
                f"QC removed every column in component {c}. Check the masking thresholds."
            )
        elif any(values["canonical_count"][i] == 0 for i in idx):
            blockers.append(f"Component {c} has retained observations with no canonical residues.")
    # A plan can inspect a panel with missing support without attempting inference.
    filtered = None
    retained_maps = ()
    if final_msa is not None and all(v.any() for v in masks):
        filtered = apply_triplet_masks(final_msa, *masks)
        retained_maps = tuple(
            subset_reference_map(m, np.flatnonzero(v)) for m, v in zip(maps, masks)
        )
    rows, dup, excluded_reasons, flagged = _row_report(
        matched,
        idx,
        masks,
        final_metrics,
        vectors,
        audits,
        decisions,
    )
    sites, site_reasons, reference_summary = _site_report(
        matched,
        maps,
        masks,
        panel,
        idx,
        initial_qc,
        final_qc,
        qc,
        mask_flags,
    )
    summary = {
        "schema_version": 1,
        "policy": policy.as_dict(),
        "effective_site_rules": opts,
        "row_scope": "Matched observations, synchronized across A/B/C. Exclusion at any component removes the observation.",
        "input_observations": matched.n_observations,
        "retained_observations": len(idx),
        "excluded_observations": matched.n_observations - len(idx),
        "passes": history,
        "fixed_point_confirmed": settled if policy.iteration == "fixed_point" else None,
        "row_exclusion_reason_counts": dict(sorted(excluded_reasons.items())),
        "site_exclusion_reason_counts": dict(sorted(site_reasons.items())),
        "retained_row_flags": dict(sorted(flagged.items())),
        "references": reference_summary,
        "component_support": _component_support(final_metrics, idx),
        "gap_presence_patterns": dict(
            sorted(
                Counter(
                    "".join(c for c, v in zip(COMPONENTS, final_metrics) if v["gap_count"][i] > 0)
                    or "none"
                    for i in idx
                ).items()
            )
        ),
        "panel_selection": "explicit" if panel_masks is not None else "all_input_columns",
        "panel_sha256": _mask_digest(maps, panel),
        "retained_mask_sha256": _mask_digest(maps, masks),
        "joint_support": _joint(final_metrics, idx),
        "exact_encoded_duplicates": dup,
        "metadata": _categories(table, idx, policy),
        "blockers": list(blockers),
    }
    return QCReport(
        idx, masks, filtered, retained_maps, final_qc, rows, sites, summary, tuple(blockers)
    )


def _row_report(matched, idx, masks, final_metrics, decision_metrics, audits, decisions):
    dup, memberships = _dup_frequencies(matched, idx, masks)
    rows = []
    kept = set(idx)
    excluded_reasons = Counter()
    flagged = Counter()
    tested_keys = (
        "site_count",
        "canonical_count",
        "coverage",
        "reference_coverage",
        "entropy_bits",
        "length_ratio",
    )
    for i, match in enumerate(matched.matches):
        for part, c in enumerate(COMPONENTS):
            retained = i in kept
            turn, why = decisions.get((i, c), (None, ()))
            values = (
                {k: _plain(v[i]) for k, v in final_metrics[part].items()}
                if retained
                else audits[i, c]
            )
            tested = (
                {key: _plain(decision_metrics[part][key][i]) for key in tested_keys}
                if retained
                else values
            )
            flags = []
            if values["canonical_count"] == 0:
                flags.append("no_canonical")
            elif values["state_count"] == 1:
                flags.append("single_canonical_state")
            if values["left_noncanonical"] or values["right_noncanonical"]:
                flags.append("terminal_uncovered")
            if values["internal_gap_count"]:
                flags.append("internal_gaps")
            if values["reference_sites"] and not values["reference_canonical"]:
                flags.append("no_reference_coverage")
            if retained:
                flagged.update(f"{c}.{flag}" for flag in flags)
            excluded_reasons.update(f"{c}.{key}" for key in why)
            cluster, size = memberships[c].get(i, (None, None))
            joint, joint_size = memberships["ABC"].get(i, (None, None))
            rows.append(
                {
                    "observation_id": match.observation_id,
                    "component": c,
                    "sequence_id": getattr(match, c.lower() + "_id"),
                    "retained_observation": retained,
                    "exclusion_pass": turn,
                    "scope": "final_sites" if retained else "row_decision_sites",
                    "exclusion_reasons": "|".join(why),
                    "flags": "|".join(flags),
                    **values,
                    **{f"decision_{k}": tested[k] for k in tested_keys},
                    "encoded_group_first_observation": cluster,
                    "encoded_group_frequency": size,
                    "abc_group_first_observation": joint,
                    "abc_group_frequency": joint_size,
                }
            )
    return rows, dup, excluded_reasons, flagged


def _site_report(matched, maps, masks, panel, idx, initial_qc, final_qc, decision_qc, site_flags):
    sites = []
    site_reasons = Counter()
    reference_summary = {}
    for k, (comp, m, keep, selected) in enumerate(zip(matched.components, maps, masks, panel)):
        q = final_qc.components[k]
        iq = initial_qc.components[k]
        dq = decision_qc.components[k]
        alphabet = CANONICAL[comp.molecule]
        retained_column = 0
        for i, chosen in enumerate(keep):
            if chosen:
                retained_column += 1
            why = [key for key, arr in site_flags[k].items() if arr[i]]
            site_reasons.update(f"{comp.name}.{key}" for key in why)
            hist = q.state_counts[:, i] if idx else np.zeros(len(alphabet), dtype=int)
            nvalid = int(hist.sum())
            ngaps = round(float(q.gap_fraction[i]) * len(idx)) if idx else 0
            nmissing = len(idx) - nvalid - ngaps
            sites.append(
                {
                    "component": comp.name,
                    "molecule": comp.molecule.value,
                    "canonical_alphabet": alphabet,
                    "reference_id": m.reference_id,
                    "reference_position": m.reference_positions[i],
                    "reference_symbol": m.symbols[i],
                    "strand": m.strand,
                    "source_column": m.original_columns[i] + 1,
                    "input_column": i + 1,
                    "retained_column": retained_column if chosen else None,
                    "panel_selected": bool(selected[i]),
                    "retained": bool(chosen),
                    "mask_reasons": "|".join(why),
                    "input_rows": iq.n_sequences,
                    "input_coverage": float(iq.coverage[i]),
                    "mask_decision_rows": dq.n_sequences if idx else 0,
                    "mask_decision_canonical": int(dq.n_canonical[i]) if idx else 0,
                    "mask_decision_state_count": int((dq.state_counts[:, i] > 0).sum())
                    if idx
                    else 0,
                    "mask_decision_entropy_bits": _plain(dq.entropy_bits[i]) if idx else None,
                    "mask_decision_gap_fraction": float(dq.gap_fraction[i]) if idx else None,
                    "mask_decision_missing_fraction": float(dq.missing_fraction[i])
                    if idx
                    else None,
                    "mask_decision_coverage": float(dq.coverage[i]) if idx else None,
                    "final_rows": len(idx),
                    "n_canonical": nvalid,
                    "n_gap": ngaps,
                    "n_missing": nmissing,
                    "coverage": nvalid / len(idx) if idx else None,
                    "gap_fraction": ngaps / len(idx) if idx else None,
                    "missing_fraction": nmissing / len(idx) if idx else None,
                    "state_count": int(np.count_nonzero(hist)),
                    "entropy_bits": _plain(q.entropy_bits[i]) if idx else None,
                    "conservation": _plain(q.conservation[i]) if idx else None,
                    "canonical_state_counts": json.dumps(
                        {s: int(n) for s, n in zip(alphabet, hist) if n}, separators=(",", ":")
                    ),
                }
            )
        numbered = np.array([pos is not None for pos in m.reference_positions])
        reference_summary[comp.name] = {
            "reference_id": m.reference_id,
            "strand": m.strand,
            "input_sites": comp.n_sites,
            "input_reference_sites": int(numbered.sum()),
            "panel_sites": int(selected.sum()),
            "retained_sites": int(keep.sum()),
            "retained_reference_sites": int((keep & numbered).sum()),
            "retained_reference_gaps": int((keep & ~numbered).sum()),
        }
    return sites, site_reasons, reference_summary


def _mask_digest(maps, masks):
    digest = hashlib.sha256()
    digest.update(b"component\toriginal_column\n")
    for m, keep in zip(maps, masks):
        for i in np.flatnonzero(keep):
            digest.update(f"{m.component}\t{(m.original_columns[i] + 1)}\n".encode())
    return digest.hexdigest()


def _component_support(metrics, idx):
    result = {}
    rows = np.asarray(idx, dtype=int)
    for component, values in zip(COMPONENTS, metrics):

        def total(name, values=values):
            return int(values[name][rows].sum())

        cells = total("site_count")
        ref = total("reference_sites")
        good = total("canonical_count")
        ref_good = total("reference_canonical")
        result[component] = {
            "input_full_canonical_length_median": float(np.median(values["full_canonical_length"])),
            "input_full_canonical_length_min": int(values["full_canonical_length"].min()),
            "input_full_canonical_length_max": int(values["full_canonical_length"].max()),
            "observation_cells": cells,
            "canonical_cells": good,
            "gap_cells": total("gap_count"),
            "missing_cells": total("missing_count"),
            "coverage": good / cells if cells else None,
            "numbered_reference_cells": ref,
            "numbered_reference_canonical": ref_good,
            "reference_coverage": ref_good / ref if ref else None,
        }
    return result


def write_qc(out_dir: str | Path, report: QCReport) -> dict[str, str]:
    """Write separate row, site and summary artifacts to caller-managed staging."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"rows": "qc_rows.tsv", "sites": "qc_sites.tsv", "summary": "qc_summary.json"}
    for key, data in (("rows", report.rows), ("sites", report.sites)):
        with (out / files[key]).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(data[0]), delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(data)
    (out / files["summary"]).write_text(
        json.dumps(report.summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return files
