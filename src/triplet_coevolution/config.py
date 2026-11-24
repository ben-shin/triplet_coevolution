"""Versioned local-file configuration for prepare and run."""

import json
import math
from pathlib import Path

COMPONENTS = ("A", "B", "C")

ANALYSIS_TYPES = {
    "min_coverage": float,
    "max_gap_fraction": float,
    "max_missing_fraction": float,
    "identity_threshold": float,
    "min_overlap": float,
    "min_valid": int,
    "min_valid_weight": float,
    "weight_policy": str,
    "weight_component": str,
    "clade_column": str,
    "weight_column": str,
    "memory_mib": float,
    "qc_policy": Path,
    "max_pairs": int,
    "score": str,
    "dca_score": str,
    "threebody_score": str,
    "apc_universe": str,
    "legacy": bool,
    "all_pairs": bool,
    "benchmark_bootstrap": int,
    "benchmark_k": list,
    "fmt": str,
    "require_reference": bool,
    "plots": bool,
    "dca": bool,
    "threebody": bool,
    "ridge": float,
    "pseudocount": float,
    "max_triplets": int,
    "triplet_candidates": Path,
    "candidate_policy": Path,
    "local_policy": Path,
    "null_policy": Path,
    "stability_policy": Path,
    "dca_policy": Path,
    "conditional_policy": Path,
    "dca_conditioning": str,
    "dca_backend": str,
    "candidate_mode": str,
    "site_panel": Path,
    "permutations": int,
    "seed": int,
    "strata_column": str,
    "contacts": Path,
    "threebody_model": str,
    "global_l2": float,
}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate configuration key {key!r}")
        result[key] = value
    return result


def _keys(value, required, allowed, where):
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be a JSON object")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(allowed)
    if missing or unknown:
        raise ValueError(f"{where}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")


def _typed(value, kind, name, base):
    if kind is Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} requires a local file path")
        path = (base / value).absolute()
        if not path.is_file():
            raise ValueError(f"{name} file not found: {path}")
        return path
    if kind is float and type(value) in (int, float) and math.isfinite(value):
        return float(value)
    if type(value) is not kind or (kind is str and not value.strip()) or kind is float:
        raise ValueError(
            f"{name} must be a finite {kind.__name__} value"
            if kind is float
            else f"{name} must be a {kind.__name__} value"
        )
    return value


def load_config(path: Path, mode: str) -> dict:
    from triplet_coevolution.search.discovery_config import SEARCH_KEYS, validate_search_component

    base = path.absolute().parent
    cfg = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique)
    required = {"schema_version", "components", "evidence"}
    _keys(cfg, required, required | {"paralogs"} | ({"analysis"} if mode == "run" else set()), mode)
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] not in (1, 2):
        raise ValueError("expected schema_version 1 or 2")
    cfg["evidence"] = _typed(cfg["evidence"], Path, "evidence", base)
    policy = cfg.setdefault("paralogs", "exclude")
    if policy not in ("exclude", "priority"):
        raise ValueError("paralogs must be exclude or priority")
    _keys(cfg["components"], COMPONENTS, COMPONENTS, "components")
    for name, comp in cfg["components"].items():
        fields = {"molecule", "reference_id"} | (
            {"alignment"} if mode == "prepare" else {"reference", "database"}
        )
        optional = {"reference_start"}
        if mode == "prepare":
            optional.update(("reference_row_id", "reference_strand", "format"))
        if mode == "run":
            optional.update(SEARCH_KEYS)
        _keys(comp, fields, fields | optional, name)
        for key in ("alignment", "reference", "database"):
            if key in comp:
                comp[key] = _typed(comp[key], Path, f"{name}.{key}", base)
        if comp["molecule"] not in ("protein", "dna", "rna"):
            raise ValueError(f"{name}.molecule must be protein, dna, or rna")
        _typed(comp["reference_id"], str, f"{name}.reference_id", base)
        if "reference_row_id" in comp:
            _typed(comp["reference_row_id"], str, f"{name}.reference_row_id", base)
        start = comp.setdefault("reference_start", 1)
        if type(start) is not int or start < 1:
            raise ValueError(f"{name}.reference_start must be a positive integer")
        if mode == "prepare":
            strand = comp.setdefault("reference_strand", "+")
            if strand not in ("+", "-") or strand == "-" and comp["molecule"] == "protein":
                raise ValueError(f"{name}.reference_strand requires + or a nucleotide - strand")
            if "format" in comp and comp["format"] not in ("fasta", "a3m", "stockholm"):
                raise ValueError(f"{name}.format must be fasta, a3m, or stockholm")
            continue
        old_keys = {"backend", "min_identity", "min_coverage", "profile", "evalue", "iterations"}
        legacy = (
            cfg["schema_version"] == 1
            and comp.get("backend", "reference") in ("reference", "hmmer", "infernal", "psiblast")
            and not (comp.keys() & (SEARCH_KEYS - old_keys))
        )
        validate_search_component(comp, name, base)
        comp["discovery_version"] = 1 if legacy else 2
    if mode == "run":
        analysis = cfg.setdefault("analysis", {})
        _keys(analysis, (), ANALYSIS_TYPES, "analysis")
        for key, value in analysis.items():
            analysis[key] = _typed(value, ANALYSIS_TYPES[key], f"analysis.{key}", base)
    return cfg
