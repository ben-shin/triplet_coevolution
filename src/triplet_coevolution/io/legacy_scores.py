"""Import the provably complete Cartesian score views exported by v1."""

import csv
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
    """Check v1 summary counts, source identity and exact coverage of every represented axis."""
    summary_path = directory / "run_summary.json"
    meta = json.loads(summary_path.read_text(encoding="utf-8"))
    try:
        if meta["schema_version"] != 1:
            raise ValueError("legacy import requires a v1 run_summary.json")
        parts = meta["components"]
        settings = meta["settings"]
        cap = settings["max_pairs"]
        if cap is not None and (type(cap) is not int or cap < 1):
            raise ValueError("invalid export limit in legacy run summary")
        min_valid = settings["min_valid"]
        wanted = {"mi": meta["outputs"]["tables"]}
        advanced = meta.get("advanced") or {}
        if advanced.get("settings", {}).get("dca"):
            wanted["dca"] = {p: advanced["outputs"][f"dca_{p.lower()}"] for p in PAIRS}
        counts = {
            p: np.full(tuple(parts[c]["retained_sites"] for c in p), -1, dtype=np.int64)
            for p in PAIRS
        }
        weights = {p: np.zeros(counts[p].shape) for p in PAIRS}
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
                location = Path(info["path"])
                if location.name != str(location) or location.suffix.lower() not in {
                    ".tsv",
                    ".csv",
                }:
                    raise ValueError("legacy score tables must be local TSV or CSV files")
                path = directory / location
                hashes[location.name] = file_digest(path)
                row_count = 0
                shape = counts[pair].shape
                seen = np.zeros(shape, bool)
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    table = csv.DictReader(
                        handle,
                        delimiter="\t" if location.suffix.lower() == ".tsv" else ",",
                        strict=True,
                    )
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
                        if count < min_valid:
                            raise ValueError(
                                f"legacy score violates declared minimum support in {path.name}"
                            )
                        for field in FAMILIES[method]:
                            scores[field][pair][coord] = float(row[field])
                        if not any(np.isfinite(scores[f][pair][coord]) for f in FAMILIES[method]):
                            raise ValueError(f"nonfinite ranked legacy score in {path.name}")
                        if method != "mi" and (
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
                if not row_count:
                    raise ValueError(f"no recoverable score coordinates in {path.name}")
        indices = {c: sorted(values) for c, values in axes.items()}
        maps = []
        for c in "ABC":
            points = [axes[c][i] for i in indices[c]]
            cols, refs, symbols = zip(*points, strict=True)
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
            if (counts[pair] < 0).any():
                raise ValueError(f"partial legacy score rectangle for {pair}")
            for blocks in scores.values():
                blocks[pair] = blocks[pair][subset]
            for family in wanted:
                if not np.any(
                    np.isfinite(np.stack([scores[f][pair] for f in FAMILIES[family]])), axis=0
                ).all():
                    raise ValueError(f"incomplete legacy score coverage for {family} {pair}")
        provenance = {
            "origin": "legacy_complete_exported_rectangle",
            "software_version": meta["software_version"],
            "inputs": {k: {"sha256": v["sha256"]} for k, v in meta["inputs"].items()},
            "n_observations": meta["matched_observations"],
            "weight_sum": meta["effective_depth"],
            "include_gaps": False,
            "apc_universe": "retained",
            "settings": settings,
            "ridge": advanced.get("settings", {}).get("ridge"),
            "legacy_current_columns": {c: [x + 1 for x in indices[c]] for c in "ABC"},
            "source_hashes": {**hashes, "run_summary.json": file_digest(summary_path)},
            "precision": "Imported decimal text as binary64; original in-memory precision cannot be recovered independently.",
            "universe": "Represented Cartesian axes in complete legacy views. Unexported retained columns have unknown coordinate/support states. Original source indices are preserved in legacy_current_columns.",
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
        _validate(store)
        return store
    except (KeyError, IndexError, TypeError, csv.Error) as exc:
        raise ValueError(f"incomplete or malformed legacy score export: {exc}") from exc
