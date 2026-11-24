"""Versioned local configuration for independently mapping structures to three references."""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from triplet_coevolution.config import _keys, _unique


@dataclass(frozen=True)
class ReferenceInput:
    component: str
    fasta: Path
    reference_id: str
    row_id: str
    molecule: str
    start: int
    strand: str
    organism: str | None
    copy_id: str | None


@dataclass(frozen=True)
class StructureConfig:
    path: Path
    coordinate_file: Path | None
    pdb_id: str | None
    expected_sha256: str | None
    cache: Path
    assembly: str | None
    model: int | None
    source_kind: str
    references: tuple[ReferenceInput, ...]
    min_identity: float
    min_coverage: float
    explicit_map: Path | None
    metric: str
    close: float
    distant: float
    distant_metric: str | None
    thresholds: tuple[float, ...]
    negative_ratio: float | None
    seed: int
    scores: Path | None
    viewer_score: str | None
    viewer_limit: int
    triples: Path | None
    functional_labels: Path | None
    exposure_bins: Path | None
    hard_negative_upper: float | None


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} must be a nonempty string without control characters")
    return value


def _number(value, name, *, minimum=0.0, maximum=None):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"invalid numeric {name}: {value!r}")
    return float(value)


def _integer(value, name, *, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def load_structure_config(path: Path) -> StructureConfig:
    """Read schema v1 with strict keys and paths relative to the JSON file."""
    path = Path(path).expanduser().resolve(strict=True)
    obj = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique)
    allowed = {
        "schema_version",
        "structure",
        "pdb_id",
        "sha256",
        "cache",
        "assembly",
        "model",
        "source_kind",
        "references",
        "explicit_map",
        "min_identity",
        "min_coverage",
        "metric",
        "close",
        "distant",
        "distant_metric",
        "sensitivity_thresholds",
        "negative_ratio",
        "seed",
        "scores",
        "viewer_score",
        "viewer_limit",
        "triples",
        "functional_labels",
        "exposure_bins",
        "hard_negative_upper",
    }
    _keys(obj, ("schema_version", "references"), allowed, "structure config")
    if type(obj["schema_version"]) is not int or obj["schema_version"] != 1:
        raise ValueError("unsupported structure config schema_version")
    if ("structure" in obj) == ("pdb_id" in obj):
        raise ValueError("set exactly one of structure or pdb_id")

    def local(value, name, *, directory_ok=False):
        result = (path.parent / _text(value, name)).resolve(strict=True)
        if not result.is_file() and not (directory_ok and result.is_dir()):
            raise ValueError(f"{name} must identify an existing input")
        return result

    def optional(key):
        return _text(obj[key], key) if key in obj else None

    refs = obj["references"]
    _keys(refs, "ABC", "ABC", "references")
    references = []
    for component in "ABC":
        spec = refs[component]
        need = ("fasta", "id", "molecule")
        _keys(
            spec,
            need,
            (*need, "row_id", "start", "strand", "organism", "copy"),
            f"references.{component}",
        )
        identifier = _text(spec["id"], f"{component}.id")
        molecule = spec["molecule"]
        strand = spec.get("strand", "+")
        if (
            molecule not in ("protein", "dna", "rna")
            or strand not in ("+", "-")
            or (molecule == "protein" and strand == "-")
        ):
            raise ValueError(f"invalid molecule or strand for references.{component}")
        references.append(
            ReferenceInput(
                component,
                local(spec["fasta"], f"{component}.fasta"),
                identifier,
                _text(spec.get("row_id", identifier), f"{component}.row_id"),
                molecule,
                _integer(spec.get("start", 1), f"{component}.start", minimum=1),
                strand,
                _text(spec["organism"], f"{component}.organism") if "organism" in spec else None,
                _text(spec["copy"], f"{component}.copy") if "copy" in spec else None,
            )
        )
    pdb_id = optional("pdb_id")
    if pdb_id and not re.fullmatch(r"(?:[0-9][a-z0-9]{3}|pdb_[a-z0-9]{8})", pdb_id, re.IGNORECASE):
        raise ValueError("pdb_id requires a four-character PDB ID or extended pdb_ identifier")
    sha = optional("sha256")
    if sha and not re.fullmatch(r"[a-fA-F0-9]{64}", sha):
        raise ValueError("sha256 requires a 64-character hexadecimal digest")
    metric = obj.get("metric", "heavy")
    distant_metric = obj.get("distant_metric")
    if metric not in ("heavy", "ca", "cb") or distant_metric not in (None, "heavy", "ca", "cb"):
        raise ValueError("metric and distant_metric support heavy, ca or cb")
    close = _number(obj.get("close", 5), "close")
    distant = _number(obj.get("distant", 25), "distant")
    if close <= 0 or distant <= close:
        raise ValueError("distance thresholds require 0 < close < distant in angstroms")
    thresholds = obj.get("sensitivity_thresholds", [])
    if not isinstance(thresholds, list):
        raise TypeError("sensitivity_thresholds must be a list of positive cutoffs")
    values = tuple(sorted({_number(v, "sensitivity_threshold") for v in thresholds} | {close}))
    if values[0] <= 0:
        raise ValueError("sensitivity thresholds must be positive")
    kind = obj.get("source_kind", "unspecified")
    if kind not in ("experimental", "predicted", "synthetic", "unspecified"):
        raise ValueError("source_kind must be experimental, predicted, synthetic or unspecified")
    coverage = _number(obj.get("min_coverage", 0.7), "min_coverage", maximum=1)
    if coverage <= 0:
        raise ValueError("min_coverage must be positive")
    hard_upper = (
        _number(obj["hard_negative_upper"], "hard_negative_upper")
        if "hard_negative_upper" in obj
        else None
    )
    if (hard_upper is None) != ("exposure_bins" not in obj) or (
        hard_upper is not None and hard_upper <= close
    ):
        raise ValueError("hard controls require exposure_bins and hard_negative_upper above close")
    return StructureConfig(
        path,
        local(obj["structure"], "structure") if "structure" in obj else None,
        pdb_id.lower() if pdb_id else None,
        sha.lower() if sha else None,
        (path.parent / _text(obj.get("cache", ".triplet-cache/structures"), "cache")).resolve(),
        optional("assembly"),
        _integer(obj["model"], "model", minimum=1) if "model" in obj else None,
        kind,
        tuple(references),
        _number(obj.get("min_identity", 0.3), "min_identity", maximum=1),
        coverage,
        local(obj["explicit_map"], "explicit_map") if "explicit_map" in obj else None,
        metric,
        close,
        distant,
        distant_metric,
        values,
        _number(obj["negative_ratio"], "negative_ratio") if "negative_ratio" in obj else None,
        _integer(obj.get("seed", 0), "seed", minimum=0),
        local(obj["scores"], "scores", directory_ok=True) if "scores" in obj else None,
        optional("viewer_score"),
        _integer(obj.get("viewer_limit", 5), "viewer_limit", minimum=0),
        local(obj["triples"], "triples") if "triples" in obj else None,
        local(obj["functional_labels"], "functional_labels")
        if "functional_labels" in obj
        else None,
        local(obj["exposure_bins"], "exposure_bins") if "exposure_bins" in obj else None,
        hard_upper,
    )
