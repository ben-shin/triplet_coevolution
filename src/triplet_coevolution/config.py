"""Versioned local-file configuration for prepare and run."""

import json
import math
from pathlib import Path

COMPONENTS = ("A", "B", "C")
BACKENDS = {"reference", "hmmer", "infernal", "psiblast"}
ANALYSIS_TYPES = {
    "min_coverage": float,
    "max_gap_fraction": float,
    "max_missing_fraction": float,
    "identity_threshold": float,
    "min_overlap": float,
    "min_valid": int,
    "max_pairs": int,
    "score": str,
    "fmt": str,
    "require_reference": bool,
    "plots": bool,
    "dca": bool,
    "threebody": bool,
    "ridge": float,
    "pseudocount": float,
    "max_triplets": int,
    "triplet_candidates": Path,
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
    base = path.absolute().parent
    cfg = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique)
    required = {"schema_version", "components", "evidence"}
    _keys(cfg, required, required | {"paralogs"} | ({"analysis"} if mode == "run" else set()), mode)
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ValueError("expected schema_version 1")
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
            optional.add("reference_row_id")
        if mode == "run":
            optional |= {
                "backend",
                "profile",
                "min_identity",
                "min_coverage",
                "evalue",
                "iterations",
            }
        _keys(comp, fields, fields | optional, name)
        for key in ("alignment", "reference", "database", "profile"):
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
            continue
        backend = comp.setdefault("backend", "reference")
        if not isinstance(backend, str) or backend not in BACKENDS:
            raise ValueError(f"{name}.backend must be one of {sorted(BACKENDS)}")
        if backend == "reference":
            if any(key in comp for key in ("profile", "evalue", "iterations")):
                raise ValueError(f"{name}: reference search uses min_identity and min_coverage")
            for key, default in (("min_identity", 0.5), ("min_coverage", 0.7)):
                value = _typed(comp.setdefault(key, default), float, f"{name}.{key}", base)
                if not 0 < value <= 1:
                    raise ValueError(f"{name}.{key} must be in (0, 1]")
        else:
            if "min_identity" in comp or "min_coverage" in comp:
                raise ValueError(f"{name}: profile searches use evalue, not identity or coverage")
            if backend in ("infernal", "hmmer") and "profile" not in comp:
                raise ValueError(f"{name}.{backend} requires profile")
            if backend == "infernal" and comp["molecule"] != "rna":
                raise ValueError("infernal requires an RNA component")
            if backend == "psiblast" and comp["molecule"] != "protein":
                raise ValueError("psiblast requires a protein component")
            if backend != "psiblast" and "iterations" in comp:
                raise ValueError("iterations applies only to psiblast")
            if backend == "psiblast":
                iterations = comp.setdefault("iterations", 3)
                if type(iterations) is not int or iterations < 1:
                    raise ValueError("iterations must be a positive integer")
            value = _typed(comp.setdefault("evalue", 0.01), float, f"{name}.evalue", base)
            if value <= 0:
                raise ValueError("evalue must be positive")
    if mode == "run":
        analysis = cfg.setdefault("analysis", {})
        _keys(analysis, (), ANALYSIS_TYPES, "analysis")
        for key, value in analysis.items():
            analysis[key] = _typed(value, ANALYSIS_TYPES[key], f"analysis.{key}", base)
    return cfg
