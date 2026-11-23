"""Load a reproducible matched-row subset from native prepared analysis inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from triplet_coevolution.core.coordinates import build_reference_map, subset_reference_map
from triplet_coevolution.io.alignments import read_msa
from triplet_coevolution.io.metadata import read_triplet_table
from triplet_coevolution.msa.masking import apply_triplet_masks, column_mask
from triplet_coevolution.msa.qc import triplet_qc
from triplet_coevolution.msa.synchronize import synchronize


def prepared_subset(source, n, largest, seed, strata_column=None):
    path = Path(source)
    if path.is_dir():
        manifest = path / "analyze_args.json"
        path = manifest if manifest.is_file() else path / "prepared/analyze_args.json"
    if not path.is_file():
        raise ValueError(f"could not find prepared analyze_args.json at {path}")
    words = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(words, list)
        or words[:1] != ["analyze"]
        or not all(isinstance(w, str) for w in words)
    ):
        raise ValueError("prepared analyze_args.json must be a native analyze argument list")

    options = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    for name in "abc":
        options.add_argument(f"--{name}", required=True)
        options.add_argument(f"--{name}-type", default="protein")
        options.add_argument(f"--{name}-ref", required=True)
        options.add_argument(f"--{name}-ref-row")
        options.add_argument(f"--{name}-start", type=int, default=1)
        options.add_argument(f"--{name}-strand", choices=["+", "-"], default="+")
    options.add_argument("--matches", required=True)
    options.add_argument("--strata-column")
    options.add_argument("--min-coverage", type=float, default=0.5)
    options.add_argument("--max-gap-fraction", type=float, default=0.5)
    options.add_argument("--max-missing-fraction", type=float, default=0.5)
    options.add_argument("--identity-threshold", type=float, default=0.8)
    options.add_argument("--min-overlap", type=float, default=0.5)
    options.add_argument("--min-valid", type=int, default=2)
    options.add_argument("--require-reference", action="store_true")
    options.add_argument("--score", choices=["mi_bits", "mi_apc"], default="mi_apc")
    # Reporting and optional model settings are intentionally set by the timing workload.
    config = vars(options.parse_known_args(words[1:])[0])
    table = read_triplet_table(path.parent / config["matches"])
    total = len(table.matches)
    if largest > total:
        raise ValueError(
            f"requested {largest} matched rows but the prepared table contains {total}"
        )
    indices = np.sort(np.random.default_rng(seed).permutation(total)[:n])
    matches = tuple(table.matches[i] for i in indices)
    column = strata_column if strata_column is not None else config["strata_column"]
    strata = None
    if column is not None:
        if column not in table.extra_columns:
            raise ValueError(f"missing permutation group column {column!r} in prepared matches")
        strata = [table.annotations[i][column] for i in indices]
        if any(not s.strip() for s in strata):
            raise ValueError(f"empty group in prepared matches column {column!r}")

    components = []
    maps = []
    for name in "abc":
        comp = read_msa(path.parent / config[name], name.upper(), config[f"{name}_type"])
        extra = {}
        if config[f"{name}_ref_row"] is not None:
            extra["row_id"] = config[f"{name}_ref_row"]
        if config[f"{name}_strand"] == "-":
            extra["strand"] = "-"
        maps.append(
            build_reference_map(comp, config[f"{name}_ref"], config[f"{name}_start"], **extra)
        )
        components.append(comp)
    msa = synchronize(*components, matches)
    keep = [
        column_mask(
            part,
            min_coverage=config["min_coverage"],
            max_gap_fraction=config["max_gap_fraction"],
            max_missing_fraction=config["max_missing_fraction"],
        )
        for part in triplet_qc(msa).components
    ]
    if not all(mask.any() for mask in keep):
        raise ValueError(f"QC removed a whole component in the {n}-observation subset")
    retained = apply_triplet_masks(msa, *keep)
    maps = tuple(
        subset_reference_map(m, c)
        for m, c in zip(
            maps, (retained.a_columns, retained.b_columns, retained.c_columns), strict=True
        )
    )
    sample_hash = hashlib.sha256(
        json.dumps([m.observation_id for m in matches]).encode()
    ).hexdigest()
    settings = {
        key: config[key]
        for key in ("identity_threshold", "min_overlap", "min_valid", "score", "require_reference")
    }
    return (
        retained.msa,
        maps,
        strata,
        {**settings, "strata_column": column, "observation_sha256": sample_hash},
    )
