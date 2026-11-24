"""Validated configuration for selecting ABC candidates and comparing local probability models."""

import math
from dataclasses import dataclass
from pathlib import Path

from triplet_coevolution.config import _unique


def _object(value, allowed, title):
    import json

    if isinstance(value, (str, Path)):
        with Path(value).open(encoding="utf-8-sig") as stream:
            value = json.load(stream, object_pairs_hook=_unique)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{title} must be a JSON object")
    value = dict(value)
    version = value.pop("schema_version", 1)
    if type(version) is not int or version != 1:
        raise ValueError(f"unsupported {title} schema_version")
    if unknown := value.keys() - allowed:
        raise ValueError(f"unknown {title} fields: {sorted(unknown)}")
    return value


def _training_ids(values, title):
    if values is None:
        return
    if (
        not isinstance(values, (list, tuple))
        or not values
        or not all(isinstance(x, str) and x.strip() for x in values)
    ):
        raise ValueError(f"{title} training_ids must contain explicit nonempty observation IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {title} training IDs")


CANDIDATE_MODES = {
    "exhaustive",
    "uniform",
    "explicit",
    "prior",
    "entropy",
    "triangle",
    "wedge",
    "hybrid",
}
SCREENED = {"entropy", "triangle", "wedge", "hybrid"}
PAIR_SCREENS = {"triangle", "wedge", "hybrid"}


def candidate_options(mode, policy, candidate_file, maximum):
    mode = mode or ("explicit" if candidate_file is not None else "uniform")
    if mode not in CANDIDATE_MODES:
        raise ValueError(f"unknown candidate mode {mode!r}")
    opts = _object(
        policy,
        {
            "prior_source",
            "pair_statistic",
            "min_entropy_bits",
            "min_site_coverage",
            "uniform_quota",
            "screen_budget",
            "training_ids",
            "file_origin",
        },
        "candidate policy",
    )
    if mode in {"explicit", "prior"} and candidate_file is None:
        raise ValueError(f"{mode} requires --triplet-candidates")
    if candidate_file is not None and mode not in {"explicit", "prior", "hybrid"}:
        raise ValueError(f"candidate file is incompatible with {mode}")
    if (mode == "prior" or (mode == "hybrid" and candidate_file)) and not opts.get("prior_source"):
        raise ValueError("supply prior_source for explicit prior evidence")
    if "prior_source" in opts and (
        candidate_file is None
        or mode not in {"prior", "hybrid"}
        or not isinstance(opts["prior_source"], str)
        or not opts["prior_source"].strip()
    ):
        raise ValueError(
            "prior_source requires independent prior evidence and its documented source"
        )
    if "pair_statistic" in opts and (
        mode not in PAIR_SCREENS
        or opts["pair_statistic"] not in {"mi_bits", "mi_apc", "dca_frobenius", "dca_apc"}
    ):
        raise ValueError(
            "pair_statistic must select a registered MI or DCA pair score for a pair screen"
        )
    if "file_origin" in opts and (
        mode != "explicit" or opts["file_origin"] not in ("prespecified", "same_data", "training")
    ):
        raise ValueError(
            "file_origin applies to explicit candidates and must be prespecified, same_data or training"
        )
    if "uniform_quota" in opts and mode != "hybrid":
        raise ValueError("uniform_quota applies only to hybrid screening")
    if any(k in opts for k in ("min_site_coverage", "min_entropy_bits")) and mode not in {
        "hybrid",
        "entropy",
    }:
        raise ValueError("entropy and coverage thresholds require entropy or hybrid selection")
    if opts.get("training_ids") and opts.get("pair_statistic", "mi_bits") != "mi_bits":
        raise ValueError("training-only pair screening requires internally fitted raw MI")
    for key in (
        "uniform_quota",
        "screen_budget",
        "min_entropy_bits",
        "min_site_coverage",
        "training_ids",
    ):
        if key in opts and mode not in SCREENED:
            raise ValueError(f"{key} requires a screened candidate mode")
    for key in ("min_entropy_bits", "min_site_coverage"):
        if key in opts and (
            type(opts[key]) not in (float, int) or not math.isfinite(opts[key]) or opts[key] < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative")
    if opts.get("min_site_coverage", 0) > 1:
        raise ValueError("min_site_coverage must be at most one")
    if "uniform_quota" in opts and (
        type(opts["uniform_quota"]) is not int or not 1 <= opts["uniform_quota"] <= maximum
    ):
        raise ValueError("uniform_quota must be an integer within the candidate budget")
    if "screen_budget" in opts and (
        type(opts["screen_budget"]) is not int or opts["screen_budget"] < 1
    ):
        raise ValueError("screen_budget must be a positive integer")
    _training_ids(opts.get("training_ids"), "candidate")
    return mode, opts


@dataclass(frozen=True)
class LocalSettings:
    options: tuple
    include_gaps: bool
    min_state_count: int
    training_ids: tuple | None
    structural_zeros: tuple

    @classmethod
    def parse(cls, policy):
        from triplet_coevolution.inference.maxent import MaxEntOptions

        keys = {"prior_mode", "prior_mass", "max_iterations", "max_cells", "tolerance"}
        raw = _object(
            policy,
            keys
            | {
                "sensitivity",
                "include_gaps",
                "training_ids",
                "min_state_count",
                "structural_zeros",
            },
            "local policy",
        )
        primary = MaxEntOptions(**{k: v for k, v in raw.items() if k in keys})
        grid = raw.get("sensitivity", [])
        if not isinstance(grid, list):
            raise TypeError("sensitivity must be a list of prior settings")
        options = [primary]
        from dataclasses import replace

        for value in grid:
            if (
                not isinstance(value, dict)
                or not value
                or value.keys() - {"prior_mode", "prior_mass"}
            ):
                raise ValueError(
                    "each sensitivity setting must specify prior_mode and/or prior_mass"
                )
            setting = replace(primary, **value)
            if setting not in options:
                options.append(setting)
        includes = raw.get("include_gaps", False)
        minimum = raw.get("min_state_count", 0)
        if type(includes) is not bool or type(minimum) is not int or minimum < 0:
            raise ValueError(
                "include_gaps must be boolean and min_state_count must be a nonnegative integer"
            )
        ids = raw.get("training_ids")
        _training_ids(ids, "local support")
        if minimum and not ids:
            raise ValueError("rare-state pooling requires training_ids")
        raw_zeros = raw.get("structural_zeros", [])
        if not isinstance(raw_zeros, list):
            raise TypeError(
                "structural_zeros must be an array of source columns and zero-based state bins"
            )
        zeros = []
        for rule in raw_zeros:
            if not isinstance(rule, dict) or rule.keys() != {"original_columns", "bins"}:
                raise ValueError(
                    "each structural zero requires original_columns and bins in A/B/C order"
                )
            entries = []
            for key, lo in (("original_columns", 1), ("bins", 0)):
                values = rule[key]
                if (
                    not isinstance(values, list)
                    or len(values) != 3
                    or any(type(i) is not int or i < lo for i in values)
                ):
                    raise ValueError(f"invalid structural zero {key}")
                entries.append(tuple(values))
            zeros.append(tuple(entries))
        if len(set(zeros)) != len(zeros):
            raise ValueError("duplicate structural zero")
        return cls(tuple(options), includes, minimum, tuple(ids) if ids else None, tuple(zeros))

    def zero_bins(self, maps, support, axes):
        lookups = [
            {m.original_columns[j] + 1: j for j in axis} for m, axis in zip(maps, axes, strict=True)
        ]
        rules = {}
        for original, bins in self.structural_zeros:
            try:
                triple = tuple(mapping[i] for mapping, i in zip(lookups, original, strict=True))
            except KeyError as exc:
                raise ValueError(
                    f"structural zero outside retained candidate axes: {original}"
                ) from exc
            shape = support.shape_for(triple)
            if any(b >= q for b, q in zip(bins, shape, strict=True)):
                raise ValueError(
                    f"structural zero state bin outside {shape} at original columns {original}"
                )
            if math.prod(shape) > self.options[0].max_cells:
                raise ValueError(f"structural zero at {original} exceeds max_cells")
            rules.setdefault(triple, []).append(bins)
            if len(rules[triple]) == math.prod(shape):
                raise ValueError(f"structural zeros exclude every state at {original}")
        return rules

    def support(self, msa):
        from triplet_coevolution.inference.maxent import freeze_support

        return freeze_support(
            msa,
            include_gaps=self.include_gaps,
            training_observations=self.training_ids,
            min_state_count=self.min_state_count,
        )
