"""Recover the rows eligible for ranking in complete, possibly sparse, v1 score views."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from triplet_coevolution.io.score_store import (
    FAMILIES,
    PAIRS,
    ExportedCoordinates,
    ScoreStore,
    _validate,
    file_digest,
)


def read_legacy_scores(directory: Path) -> ScoreStore:
    """Verify v1 declared row counts and export cap before recovering ranked cells and support."""
    summary_path = directory / "run_summary.json"
    summary_bytes = summary_path.read_bytes()
    summary_hash = hashlib.sha256(summary_bytes).hexdigest()
    meta = json.loads(summary_bytes)
    try:
        if meta["schema_version"] != 1:
            raise ValueError("legacy import requires a v1 run_summary.json")
        parts = meta["components"]
        settings = meta["settings"]
        cap = settings["max_pairs"]
        if cap is not None and (type(cap) is not int or cap < 1):
            raise ValueError("invalid export limit in legacy run summary")
        min_valid = settings["min_valid"]
        if (
            type(min_valid) is not int
            or min_valid < 1
            or type(settings["require_reference"]) is not bool
        ):
            raise ValueError("invalid eligibility settings in legacy run summary")
        selected_mi = settings["score"]
        if selected_mi not in FAMILIES["mi"]:
            raise ValueError("invalid selected MI statistic in legacy run summary")
        for c in "ABC":
            retained = parts[c]["retained_columns"]
            if (
                not isinstance(retained, list)
                or len(retained) != parts[c]["retained_sites"]
                or not retained
                or any(type(i) is not int or i < 1 for i in retained)
                or sorted(set(retained)) != retained
            ):
                raise ValueError(f"invalid retained-column map for legacy component {c}")
        wanted = {"mi": meta["outputs"]["tables"]}
        advanced = meta.get("advanced") or {}
        if advanced.get("settings", {}).get("dca"):
            for key in ("max_pairs", "min_valid", "require_reference"):
                if advanced["settings"].get(key, settings[key]) != settings[key]:
                    raise ValueError(f"inconsistent legacy export settings for {key}")
            wanted["dca"] = {p: advanced["outputs"][f"dca_{p.lower()}"] for p in PAIRS}
        counts = {
            p: np.full(tuple(parts[c]["retained_sites"] for c in p), -1, dtype=np.int64)
            for p in PAIRS
        }
        weights = {p: np.full(counts[p].shape, np.nan) for p in PAIRS}
        scores = {
            score: {p: np.full(counts[p].shape, np.nan) for p in PAIRS}
            for method in wanted
            for score in FAMILIES[method]
        }
        axes = {c: {} for c in "ABC"}
        hashes = {}
        for method, files in wanted.items():
            for pair in PAIRS:
                info = files[pair]
                if type(info["rows"]) is not int or info["rows"] < 0:
                    raise ValueError("invalid recorded legacy row count")
                location = Path(info["path"])
                if location.name != str(location) or location.suffix.lower() not in {
                    ".tsv",
                    ".csv",
                }:
                    raise ValueError("legacy score tables must be local TSV or CSV files")
                path = directory / location
                hashes[location.name] = file_digest(path)
                if "sha256" in info and info["sha256"] != hashes[location.name]:
                    raise ValueError(f"legacy score table checksum mismatch for {location.name}")
                row_count = 0
                shape = counts[pair].shape
                seen = np.zeros(shape, bool)
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    table = csv.DictReader(
                        handle,
                        delimiter="\t" if location.suffix.lower() == ".tsv" else ",",
                        strict=True,
                    )
                    header = table.fieldnames or []
                    needed = {"rank", "pair", "n_valid", "valid_weight", *FAMILIES[method]} | {
                        f"{side}_{field}"
                        for side in "xy"
                        for field in (
                            "component",
                            "reference_id",
                            "col",
                            "original_col",
                            "reference_pos",
                            "symbol",
                        )
                    }
                    if len(set(header)) != len(header) or not needed.issubset(header):
                        raise ValueError(
                            f"duplicate or missing required legacy header in {path.name}"
                        )
                    selected = selected_mi if method == "mi" else "dca_frobenius"
                    previous = float("inf")
                    for row in table:
                        if None in row or None in row.values() or row["pair"] != pair:
                            raise ValueError(f"malformed legacy score row in {path.name}")
                        idx = []
                        for side, c in zip("xy", pair, strict=True):
                            cur = int(row[f"{side}_col"]) - 1
                            original = int(row[f"{side}_original_col"]) - 1
                            ref = row[f"{side}_reference_pos"]
                            if settings["require_reference"] and not ref:
                                raise ValueError(
                                    f"missing required reference coordinate in {path.name}"
                                )
                            point = (original, int(ref) if ref else None, row[f"{side}_symbol"])
                            if (
                                row[f"{side}_component"] != c
                                or row[f"{side}_reference_id"] != parts[c]["reference_id"]
                                or not 0 <= cur < parts[c]["retained_sites"]
                                or parts[c]["retained_columns"][cur] != original + 1
                                or axes[c].setdefault(cur, point) != point
                            ):
                                raise ValueError(
                                    f"inconsistent legacy coordinate or reference in {path.name}"
                                )
                            idx.append(cur)
                        coord = tuple(idx)
                        if seen[coord]:
                            raise ValueError(f"duplicate coordinate in {path.name}")
                        seen[coord] = True
                        count, weight = int(row["n_valid"]), float(row["valid_weight"])
                        if count < min_valid or not np.isfinite(weight) or weight <= 0:
                            raise ValueError(
                                f"legacy score violates declared minimum support in {path.name}"
                            )
                        for field in FAMILIES[method]:
                            scores[field][pair][coord] = float(row[field])
                        score = scores[selected][pair][coord]
                        if not np.isfinite(score):
                            raise ValueError(f"nonfinite ranked legacy score in {path.name}")
                        if int(row["rank"]) != row_count + 1 or score > previous:
                            raise ValueError(f"invalid v1 rank order in {path.name}")
                        previous = score
                        if counts[pair][coord] >= 0 and (
                            counts[pair][coord] != count
                            or not np.isclose(weights[pair][coord], weight, rtol=1e-12, atol=1e-12)
                        ):
                            raise ValueError(
                                f"legacy methods do not share the same universe/support in {path.name}"
                            )
                        counts[pair][coord] = count
                        weights[pair][coord] = weight
                        row_count += 1
                if (
                    row_count != info["rows"]
                    or (cap is not None and row_count > cap)
                    or (cap is not None and row_count >= cap and row_count < int(np.prod(shape)))
                ):
                    raise ValueError(f"truncated or unprovably complete legacy table {path.name}")
                if file_digest(path) != hashes[location.name]:
                    raise ValueError(f"legacy score table changed while reading {path.name}")
        indices = {c: sorted(values) for c, values in axes.items()}
        maps = []
        for c in "ABC":
            points = [axes[c][i] for i in indices[c]]
            cols, refs, symbols = zip(*points, strict=True) if points else ((), (), ())
            maps.append(
                ExportedCoordinates(
                    c,
                    parts[c]["reference_id"],
                    cols,
                    refs,
                    symbols,
                    parts[c]["reference_start"],
                    parts[c].get("reference_strand", "+"),
                    tuple(x + 1 for x in indices[c]),
                )
            )
        for pair in PAIRS:
            subset = np.ix_(*(indices[c] for c in pair))
            counts[pair] = counts[pair][subset]
            weights[pair] = weights[pair][subset]
            for blocks in scores.values():
                blocks[pair] = blocks[pair][subset]
        provenance = {
            "origin": "legacy_uncapped_or_complete_views",
            "software_version": meta["software_version"],
            "inputs": {k: {"sha256": v["sha256"]} for k, v in meta["inputs"].items()},
            "n_observations": meta["matched_observations"],
            "weight_sum": meta["effective_depth"],
            "include_gaps": False,
            "apc_universe": "retained",
            "settings": settings,
            "ridge": advanced.get("settings", {}).get("ridge"),
            "legacy_current_columns": {c: [x + 1 for x in indices[c]] for c in "ABC"},
            "source_retained_columns": {c: parts[c]["retained_columns"] for c in "ABC"},
            "exported_rows": {
                method: {p: views[p]["rows"] for p in PAIRS} for method, views in wanted.items()
            },
            "unknown_support": {"n_valid": -1, "valid_weight": "NaN", "score": "NaN"},
            "legacy_selected_scores": {
                "mi": selected_mi,
                **({"dca": "dca_frobenius"} if "dca" in wanted else {}),
            },
            "source_hashes": {**hashes, "run_summary.json": summary_hash},
            "precision": "Imported decimal text as binary64; original in-memory precision cannot be recovered independently.",
            "universe": "All ranked rows of each originally selected statistic, verified against its declared count and export cap. An alternate score exists only where that method exported the row. A cell absent from every view has unknown support and unavailable scores. Unreported source columns retain alignment indices only; their reference positions and symbols are unknown. Reference-coordinate denominators use mapped axes and confirmed support.",
        }
        store = ScoreStore(
            tuple(maps),
            scores,
            counts,
            weights,
            provenance,
            min_valid,
            settings["require_reference"],
        )
        if file_digest(summary_path) != summary_hash:
            raise ValueError("legacy run summary changed while reading score tables")
        _validate(store)
        return store
    except (KeyError, IndexError, TypeError, csv.Error) as exc:
        raise ValueError(f"incomplete or malformed legacy score export: {exc}") from exc
