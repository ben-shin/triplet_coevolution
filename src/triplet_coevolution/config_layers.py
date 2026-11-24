"""Version-bound single-parent configuration inheritance with source-relative input paths."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from triplet_coevolution.config import _unique
from triplet_coevolution.parameters import analysis_types

COMPONENT_PATHS = {
    "alignment",
    "database",
    "reference",
    "profile",
    "alignment_profile",
    "hit_table",
    "reference_hits",
    "cache_dir",
}


def _absolutize(data, base):
    def fields(mapping, names):
        if isinstance(mapping, dict):
            for key in names & mapping.keys():
                val = mapping[key]
                if isinstance(val, str) and val.strip():
                    mapping[key] = str((base / val).absolute())

    fields(data, {"matches", "evidence"})
    comps = data.get("components")
    if isinstance(comps, dict):
        for value in comps.values():
            fields(value, COMPONENT_PATHS)
    fields(data.get("analysis"), {key for key, kind in analysis_types().items() if kind is Path})


def _merge(parent, child):
    merged = deepcopy(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_layers(path):
    """Return resolved JSON with its exact child-to-root chain. Lists replace inherited lists."""

    def visit(path, ancestry):
        path = Path(path).resolve()
        if path in ancestry or len(ancestry) >= 16:
            raise ValueError(f"{path}: extends cycle or inheritance deeper than 16 layers")
        try:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique)
        except (ValueError, OSError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        if not isinstance(data, dict):
            raise TypeError(f"{path}: expected configuration object")
        if "extends" not in data:
            _absolutize(data, path.parent)
            return data, ((path, digest),)
        parent = data.pop("extends")
        if not isinstance(parent, str) or not parent.strip():
            raise ValueError(f"{path}: extends must name a local configuration")
        base, chain = visit(path.parent / parent, ancestry | {path})
        for field in ("schema_version", "kind"):
            if field in data and data[field] != base.get(field):
                raise ValueError(f"{path}: extends cannot change {field}")
        _absolutize(data, path.parent)
        return _merge(base, data), ((path, digest), *chain)

    return visit(path, set())
