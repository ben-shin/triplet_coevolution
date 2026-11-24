"""Prespecified calibration policy and finite Monte Carlo resolution."""

import json
from dataclasses import dataclass, fields
from math import ceil, fsum, isfinite, sqrt
from pathlib import Path

import numpy as np

from triplet_coevolution.inference.scores import get_score
from triplet_coevolution.validation.fdr import bh_adjust
from triplet_coevolution.validation.tree_design import TreeDesign

MATCHING = {"whole_component", "within_stratum", "preserve_ab", "preserve_ac", "preserve_bc"}
GENERATIVE = {"m2_bootstrap", "pair_panel"}


@dataclass(frozen=True)
class NullPolicy:
    null: str
    statistics: tuple[str, ...]
    draws: int
    schema_version: int = 1
    seed: int = 0
    planned_draws: int | None = None
    family: str = "statistic"
    alpha: float = 0.05
    max_t: bool = False
    workers: int = 1
    checkpoint: Path | None = None
    resume: bool = False
    memory_mib: int = 512
    checkpoint_mib: int = 1024
    strata_column: str | None = None
    weight_design: str = "uniform"
    tree: TreeDesign | None = None

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported null-policy schema version")
        if self.null not in MATCHING | GENERATIVE | {"independent_site_tree"}:
            raise ValueError(f"unknown null model {self.null!r}")
        if (self.tree is not None) != (self.null == "independent_site_tree"):
            raise ValueError("independent_site_tree requires its own supplied tree design")
        if self.tree is not None:
            object.__setattr__(self, "tree", TreeDesign.parse(self.tree))
        if not isinstance(self.statistics, (tuple, list)) or not self.statistics:
            raise ValueError("statistics must be a nonempty list of registered score names")
        for name in self.statistics:
            get_score(name)
        if len(set(self.statistics)) != len(self.statistics):
            raise ValueError("statistics must be unique")
        object.__setattr__(self, "statistics", tuple(self.statistics))
        for key in ("draws", "workers", "memory_mib", "checkpoint_mib"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.planned_draws is not None and (
            type(self.planned_draws) is not int or self.planned_draws < self.draws
        ):
            raise ValueError("planned_draws must cover requested draws")
        if self.family not in {"statistic", "interface"}:
            raise ValueError("family must be statistic or interface")
        if (
            type(self.alpha) not in (float, int)
            or not isfinite(self.alpha)
            or not 0 < self.alpha < 1
        ):
            raise ValueError("alpha must be in (0,1)")
        if type(self.max_t) is not bool or type(self.resume) is not bool:
            raise TypeError("max_t and resume must be Boolean")
        if self.max_t and self.null == "m2_bootstrap":
            raise ValueError(
                "maxT requires joint draws; local M2 candidates simulate independently"
            )
        if self.strata_column is not None and (
            not isinstance(self.strata_column, str) or not self.strata_column.strip()
        ):
            raise ValueError("strata_column must name a match-table field")
        if self.strata_column is not None and self.null in GENERATIVE | {
            "whole_component",
            "independent_site_tree",
        }:
            raise ValueError(f"{self.null} does not sample within named strata")
        if self.null == "within_stratum" and self.strata_column is None:
            raise ValueError("within_stratum requires strata_column")
        if self.weight_design not in {"uniform", "fixed_design"} or (
            self.null not in GENERATIVE and self.weight_design != "uniform"
        ):
            raise ValueError(
                "weight_design is uniform or, for an independent supplied design in an iid bootstrap, fixed_design"
            )
        if self.resume and self.checkpoint is None:
            raise ValueError("resume requires a persistent checkpoint directory")
        if self.checkpoint is not None:
            object.__setattr__(self, "checkpoint", Path(self.checkpoint).expanduser().resolve())

    @classmethod
    def from_config(cls, source, **overrides):
        base = Path.cwd()
        if isinstance(source, (str, Path)):
            path = Path(source)
            base = path.resolve().parent
            from triplet_coevolution.config import _unique

            source = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique)
        if not isinstance(source, dict):
            raise TypeError("null policy must be a JSON object")
        data = dict(source)
        if data.get("tree") is not None:
            data["tree"] = TreeDesign.parse(data["tree"], base)
        if data.get("checkpoint") is not None:
            data["checkpoint"] = base / data["checkpoint"]
        if (
            overrides.get("draws") is not None
            and data.get("planned_draws") is None
            and "draws" in data
        ):
            data["planned_draws"] = data["draws"]
        data.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown null policy fields: {sorted(unknown)}")
        return cls(**data)


def resolution(tests, draws, alpha):
    harmonic = fsum(1 / i for i in range(1, tests + 1))
    return {
        "tests": tests,
        "draws": draws,
        "alpha": alpha,
        "minimum_p": 1 / (draws + 1),
        "bh_first_threshold": alpha / tests,
        "by_first_threshold": alpha / (tests * harmonic),
        "draws_for_bh_rank_one": ceil(tests / alpha) - 1,
        "draws_for_by_rank_one": ceil(tests * harmonic / alpha) - 1,
        "bh_rank_one_attainable": 1 / (draws + 1) <= alpha / tests,
        "harmonic_factor": harmonic,
    }


def adjusted(p):
    values = np.where(np.isfinite(p), p, 1.0)
    bh = bh_adjust(values)
    return bh, np.minimum(1.0, bh * fsum(1 / i for i in range(1, len(values) + 1)))


def mc_interval(exceed, invalid, b):
    """95% Wilson limits, including both extremes for unresolved refits."""
    z = 1.959963984540054

    def limit(k, sign):
        p = k / b
        return (p + z * z / (2 * b) + sign * z * sqrt(p * (1 - p) / b + z * z / (4 * b * b))) / (
            1 + z * z / b
        )

    lower = 0.0 if exceed == 0 else max(0.0, limit(exceed, -1))
    upper = 1.0 if exceed + invalid == b else min(1.0, limit(exceed + invalid, 1))
    return lower, upper


def calibration_plan(
    policy, plan, *, require_reference, apc_universe, dca, threebody, threebody_model, weight_policy
):
    """Compute declared family sizes and Monte Carlo resolution before fitting."""
    families = {}
    for name in policy.statistics:
        spec = get_score(name)
        if (spec.method == "threebody" and not threebody) or (spec.method == "dca" and not dca):
            raise ValueError(f"null statistic {name} requires its observed method to be enabled")
        spec.metadata(threebody_model=threebody_model)
        if policy.null in GENERATIVE:
            if spec.method != "threebody" or threebody_model != "maxent":
                raise ValueError("generative calibration requires local maxent triple scores")
            if (
                weight_policy
                != {"uniform": "uniform", "fixed_design": "supplied"}[policy.weight_design]
            ):
                raise ValueError(
                    "iid calibration requires --weight-policy uniform or explicitly independent supplied weights"
                )
            if policy.null == "pair_panel" and sum(plan["retained_sites"].values()) > 8:
                raise ValueError(
                    "exact pair_panel requires the entire retained panel to have 3–8 sites"
                )
        if spec.method == "threebody":
            if plan["candidate_mode"] in {"triangle", "wedge", "entropy", "hybrid"}:
                raise ValueError(
                    "fixed-candidate calibration requires selection independent of the evaluation rows"
                )
            sizes = {
                name if policy.family == "statistic" else name + ".ABC": plan["candidate_count"]
            }
        else:
            mapped = require_reference or spec.correction == "APC" and apc_universe == "reference"
            block = plan["reference_pairs" if mapped else "retained_pairs"]
            sizes = (
                {name + "." + k: n for k, n in block.items()}
                if policy.family == "interface"
                else {name: sum(block.values())}
            )
        for group, n in sizes.items():
            families[group] = (
                resolution(n, policy.planned_draws or policy.draws, policy.alpha)
                if n
                else {"tests": 0}
            )
    return {
        "null": policy.null,
        "families": families,
        "workers": policy.workers,
        "requested_draws": policy.draws,
        "planned_draws": policy.planned_draws or policy.draws,
        "memory_mib": policy.memory_mib,
    }
