"""Explicit unweighted row and column decisions for synchronized triplets."""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Literal

COMPONENTS = ("A", "B", "C")


@dataclass(frozen=True)
class RowRule:
    min_coverage: float | None = None
    min_canonical: int | None = None
    max_canonical: int | None = None
    min_reference_coverage: float | None = None
    min_entropy_bits: float | None = None
    min_length_ratio: float | None = None
    max_length_ratio: float | None = None


@dataclass(frozen=True)
class SiteRule:
    min_coverage: float | None = None
    max_gap_fraction: float | None = None
    max_missing_fraction: float | None = None
    min_entropy_bits: float | None = None
    min_canonical: int | None = None
    min_states: int | None = None


def _thresholds(rule):
    for name, value in rule.items():
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
        if name in {"min_canonical", "max_canonical", "min_states"} and not isinstance(value, int):
            raise ValueError(f"{name} requires an integer count")
        if ("coverage" in name or "fraction" in name) and value > 1:
            raise ValueError(f"{name} must be between 0 and 1")
    for lo, hi in (("min_canonical", "max_canonical"), ("min_length_ratio", "max_length_ratio")):
        if rule.get(lo) is not None and rule.get(hi) is not None and rule[lo] > rule[hi]:
            raise ValueError(f"{lo} exceeds {hi}")


def _rules(obj, typ):
    if not isinstance(obj, Mapping) or set(obj) - set(COMPONENTS):
        raise ValueError("QC rules must be keyed by component A, B or C")
    result = {}
    for name, value in obj.items():
        if isinstance(value, typ):
            value = asdict(value)
        if not isinstance(value, Mapping) or set(value) - {f.name for f in fields(typ)}:
            raise ValueError(f"invalid {typ.__name__} options for component {name}")
        _thresholds(value)
        result[name] = typ(**value)
    return MappingProxyType(result)


@dataclass(frozen=True)
class QCPolicy:
    schema_version: int = 1
    order: Literal["columns_first", "rows_first"] = "columns_first"
    iteration: Literal["single", "fixed_point"] = "single"
    max_passes: int | None = None
    rows: Mapping[str, RowRule] = field(default_factory=dict)
    sites: Mapping[str, SiteRule] = field(default_factory=dict)
    taxon_column: str | None = None
    clade_column: str | None = None

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported QC policy schema_version")
        if self.order not in ("columns_first", "rows_first"):
            raise ValueError("QC order must be rows_first or columns_first")
        if self.iteration not in ("single", "fixed_point"):
            raise ValueError("QC iteration must be single or fixed_point")
        if self.max_passes is not None and (
            type(self.max_passes) is not int or self.max_passes < 1
        ):
            raise ValueError("max_passes must be a positive integer")
        if self.iteration == "single" and self.max_passes not in (None, 1):
            raise ValueError("max_passes above one requires fixed_point")
        for key in (self.taxon_column, self.clade_column):
            if key is not None and (not isinstance(key, str) or not key.strip()):
                raise ValueError("QC metadata column must be a nonempty string")
        object.__setattr__(self, "rows", _rules(self.rows, RowRule))
        object.__setattr__(self, "sites", _rules(self.sites, SiteRule))

    @classmethod
    def from_config(cls, value: str | Path | Mapping | None) -> "QCPolicy":
        """Read a validated policy from JSON, a mapping or the legacy defaults."""
        if isinstance(value, (str, Path)):

            def unique(items):
                result = {}
                for key, data in items:
                    if key in result:
                        raise ValueError(f"duplicate QC policy key {key}")
                    result[key] = data
                return result

            value = json.loads(
                Path(value).read_text(encoding="utf-8-sig"), object_pairs_hook=unique
            )
        if value is None:
            return cls()
        if not isinstance(value, Mapping) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError("invalid QC policy keys")
        return cls(**value)

    def as_dict(self):
        def options(rules):
            return {
                c: {k: v for k, v in asdict(r).items() if v is not None} for c, r in rules.items()
            }

        return {
            "schema_version": self.schema_version,
            "order": self.order,
            "iteration": self.iteration,
            "max_passes": self.max_passes,
            "rows": options(self.rows),
            "sites": options(self.sites),
            "taxon_column": self.taxon_column,
            "clade_column": self.clade_column,
        }
