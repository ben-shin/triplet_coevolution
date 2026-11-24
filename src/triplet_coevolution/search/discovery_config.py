"""Strict local-first discovery component configuration."""

import math
import os
from pathlib import Path

from triplet_coevolution.search.hit_tables import FORMATS

BACKENDS = {"reference", "hmmer", "infernal", "psiblast", "jackhmmer", "import"}
SEARCH_KEYS = {
    "backend",
    "profile",
    "min_identity",
    "min_coverage",
    "evalue",
    "iterations",
    "hit_format",
    "hit_table",
    "hit_query",
    "reference_hits",
    "hit_model_length",
    "alignment_backend",
    "alignment_profile",
    "include_evalue",
    "domain_evalue",
    "include_domain_evalue",
    "seed",
    "cpu",
    "timeout_seconds",
    "memory_mb",
    "max_candidates",
    "min_model_coverage",
    "min_target_coverage",
    "min_reference_coverage",
    "min_domain_coverage",
    "min_entropy_bits",
    "database_release",
    "cache_dir",
}


def _path(comp, key, base, directory=False):
    val = comp[key]
    if not isinstance(val, (str, Path)) or not str(val).strip():
        raise ValueError(f"{key} requires a local path")
    path = (Path(base) / val).absolute()
    if not directory and not path.is_file():
        raise ValueError(f"{key} file not found: {path}")
    if directory and path.exists() and not path.is_dir():
        raise ValueError(f"{key} must be a directory")
    comp[key] = path


def validate_search_component(comp, name, base):
    backend = comp.setdefault("backend", "reference")
    if not isinstance(backend, str) or backend not in BACKENDS:
        raise ValueError(f"{name}: unknown local search backend")
    mol = comp["molecule"]
    if mol not in ("protein", "dna", "rna"):
        raise ValueError("explicit molecule type is required")
    if (
        backend in ("jackhmmer", "psiblast")
        and mol != "protein"
        or backend == "infernal"
        and mol != "rna"
    ):
        raise ValueError(f"incompatible {backend} search molecule {mol}")
    for key in ("profile", "alignment_profile", "hit_table", "reference_hits", "cache_dir"):
        if key in comp:
            _path(comp, key, base, key == "cache_dir")
    if backend == "reference" and set(comp) & {
        "profile",
        "evalue",
        "iterations",
        "include_evalue",
        "domain_evalue",
        "include_domain_evalue",
    }:
        raise ValueError("reference search uses min_identity and min_coverage")
    if backend in ("hmmer", "infernal") and "profile" not in comp:
        raise ValueError(f"{backend} requires a profile")
    if backend in ("jackhmmer", "import") and "profile" in comp:
        raise ValueError(
            f"{backend} takes a typed reference seed or table; use alignment_profile for a separate alignment model"
        )
    if (backend == "import") != ("hit_table" in comp and "hit_format" in comp):
        raise ValueError("import search requires hit_table and hit_format")
    if backend != "import" and set(comp) & {
        "hit_format",
        "hit_table",
        "hit_query",
        "reference_hits",
        "hit_model_length",
    }:
        raise ValueError("hit-table settings require the import backend")
    if backend == "import":
        fmt = comp["hit_format"]
        if (
            fmt not in FORMATS[:-1]
            or (fmt == "hmmsearch_domtblout" and mol != "protein")
            or (fmt == "nhmmer_tblout" and mol == "protein")
            or (fmt == "cmsearch_tblout" and mol != "rna")
        ):
            raise ValueError("unsupported hit format/molecule combination")
        if "hit_query" in comp and (
            not isinstance(comp["hit_query"], str) or not comp["hit_query"].strip()
        ):
            raise ValueError("hit_query requires an exact nonempty query identifier")
    if "iterations" in comp and backend not in ("jackhmmer", "psiblast"):
        raise ValueError("iterations requires jackhmmer or psiblast")
    if backend not in ("hmmer", "jackhmmer") and set(comp) & {
        "include_evalue",
        "domain_evalue",
        "include_domain_evalue",
    }:
        raise ValueError("iterative/domain thresholds require protein HMMER")
    if backend == "hmmer" and (
        set(comp) & {"include_evalue", "include_domain_evalue"}
        or ("domain_evalue" in comp and mol != "protein")
    ):
        raise ValueError(
            "inclusion thresholds require jackhmmer and domain_evalue requires protein"
        )
    if backend != "reference" and set(comp) & {"min_coverage", "min_identity"}:
        raise ValueError(
            "portable min_identity/min_coverage require reference search; use reported model/target filters"
        )

    def scalar(key, default, lo=0.0, hi=math.inf, strict=False):
        v = comp.setdefault(key, default)
        if (
            type(v) not in (int, float)
            or not math.isfinite(v)
            or v < lo
            or v > hi
            or strict
            and v == lo
        ):
            raise ValueError(f"invalid {name}.{key}")

    if backend == "reference":
        scalar("min_identity", 0.5, hi=1, strict=True)
        scalar("min_coverage", 0.7, hi=1, strict=True)
    else:
        scalar("evalue", 0.01, strict=True)
    if backend == "jackhmmer" or (backend == "hmmer" and mol == "protein"):
        scalar("domain_evalue", comp["evalue"], strict=True)
    if backend == "jackhmmer":
        scalar("include_evalue", min(0.001, comp["evalue"]), hi=comp["evalue"], strict=True)
        scalar(
            "include_domain_evalue",
            min(0.001, comp["domain_evalue"]),
            hi=comp["domain_evalue"],
            strict=True,
        )

    def integer(key, default, lo, hi):
        v = comp.setdefault(key, default)
        if type(v) is not int or not lo <= v <= hi:
            raise ValueError(f"{name}.{key} must be an integer in [{lo},{hi}]")

    if backend in ("jackhmmer", "psiblast"):
        integer("iterations", 3, 1, 100)
    integer("seed", 42, 1, 2**31 - 1)
    integer("cpu", 1, 1, 256)
    integer("timeout_seconds", 3600, 1, 7 * 86400)
    integer("max_candidates", 100000, 1, 1000000)
    if "hit_model_length" in comp:
        integer("hit_model_length", None, 1, 2**31 - 1)
    if "memory_mb" in comp:
        integer("memory_mb", None, 1, 2**30)
        if os.name != "posix" and backend not in ("import", "reference"):
            raise ValueError("hard child memory bounds require POSIX resource limits")
    scalar("min_model_coverage", 0, hi=1)
    scalar("min_target_coverage", 0, hi=1)
    scalar("min_reference_coverage", 0, hi=1)
    scalar("min_domain_coverage", 0, hi=1)
    scalar("min_entropy_bits", 0)
    if "database_release" in comp and (
        not isinstance(comp["database_release"], str) or not comp["database_release"].strip()
    ):
        raise ValueError("database_release requires a nonempty release label")
    align = comp.setdefault(
        "alignment_backend", "jackhmmer_stockholm" if backend == "jackhmmer" else "projection"
    )
    if align not in ("projection", "jackhmmer_stockholm", "hmmalign", "cmalign"):
        raise ValueError("unsupported alignment_backend")
    if align == "jackhmmer_stockholm" and backend != "jackhmmer":
        raise ValueError("jackhmmer_stockholm requires jackhmmer")
    if backend == "jackhmmer" and align == "projection":
        raise ValueError("use jackhmmer_stockholm for iterative model-to-seed coordinates")
    if "alignment_profile" in comp and align not in ("hmmalign", "cmalign"):
        raise ValueError("alignment_profile requires a profile alignment backend")
    if align in ("hmmalign", "cmalign"):
        if align == "cmalign" and mol != "rna":
            raise ValueError("cmalign requires RNA")
        if "alignment_profile" not in comp and not (
            align == "hmmalign"
            and backend == "hmmer"
            or align == "cmalign"
            and backend == "infernal"
        ):
            raise ValueError(f"{align} requires a compatible alignment_profile")
    return comp
