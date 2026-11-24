"""Explicit backend, feature support and conditioning for intercomponent coupling scores."""

import json
from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import isfinite
from pathlib import Path

import numpy as np

from triplet_coevolution.config import _unique
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.potts import (
    MAX_DUAL_ELEMENTS,
    MAX_SOLVE_DIMENSION,
    direct_couplings,
    factor_backend,
)


@dataclass(frozen=True)
class DCADesign:
    backend: str = "mf"
    conditioning: str = "joint"
    ridge: float = 0.1
    include_gaps: bool = False
    shrinkage: float = 0.0
    alphabet_support: str = "observed"
    plm: dict = field(default_factory=dict)
    factor_solver: str = "numpy_legacy"

    def __post_init__(self):
        if self.backend not in {"mf", "plm"} or self.conditioning not in {"joint", "pairwise"}:
            raise ValueError(
                "DCA backend must be mf or plm; conditioning must be joint or pairwise"
            )
        if type(self.ridge) not in (int, float) or not isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("ridge must be a finite positive covariance-unit penalty")
        if (
            type(self.shrinkage) not in (int, float)
            or not isfinite(self.shrinkage)
            or not 0 <= self.shrinkage <= 1
            or type(self.include_gaps) is not bool
        ):
            raise ValueError("shrinkage must be a finite fraction and include_gaps a Boolean")
        if self.alphabet_support not in {"observed", "fixed"}:
            raise ValueError("alphabet_support must be observed or fixed")
        if not isinstance(self.plm, dict):
            raise TypeError("plm options require an object")
        if self.backend == "mf" and self.plm:
            raise ValueError("plm optimizer options require backend=plm")
        if self.backend == "plm":
            if self.factor_solver != "numpy_legacy":
                raise ValueError("factor_solver applies only to mf DCA")
            if self.ridge != 0.1 or self.shrinkage or self.alphabet_support != "fixed":
                raise ValueError(
                    "PLM uses a fixed alphabet and its own lambda_h/lambda_j rather than mf ridge/shrinkage"
                )
            self.plm_options()
        elif self.factor_solver not in ("numpy_legacy", "cholesky"):
            raise ValueError("mf factor_solver must be numpy_legacy or cholesky")

    def plm_options(self):
        from triplet_coevolution.inference.plm import PLMOptions

        return PLMOptions(**self.plm, include_gaps=self.include_gaps)

    @classmethod
    def parse(cls, source=None, *, backend=None, conditioning=None, ridge=None):
        if isinstance(source, cls):
            source = source.metadata()
        if isinstance(source, (str, Path)):
            source = json.loads(
                Path(source).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique
            )
        if source is not None and not isinstance(source, dict):
            raise TypeError("DCA policy must be a JSON object")
        opts = dict(source or {})
        version = opts.pop("schema_version", 1)
        if type(version) is not int or version != 1:
            raise ValueError("unrecognized DCA policy schema")
        for k, v in (("backend", backend), ("conditioning", conditioning), ("ridge", ridge)):
            if v is not None:
                if k in opts and opts[k] != v:
                    raise ValueError(f"DCA option {k} conflicts with its policy file")
                opts[k] = v
        if opts.get("backend") == "plm":
            if "ridge" in opts or "shrinkage" in opts:
                raise ValueError(
                    "mf ridge/shrinkage are not PLM penalties; use lambda_h and lambda_j"
                )
            opts.setdefault("alphabet_support", "fixed")
        return cls(**opts)

    def metadata(self):
        data = asdict(self)
        if self.factor_solver == "numpy_legacy":
            del data["factor_solver"]
        if self.backend == "plm":
            data.pop("ridge")
            data.pop("shrinkage")
            data["plm"] = asdict(self.plm_options())
            del data["plm"]["include_gaps"]
        return data

    def fit(
        self, msa, weights, *, min_valid, min_valid_weight, diagnostics=None, mf=direct_couplings
    ):
        if self.backend == "mf":
            return mf(
                msa,
                weights,
                ridge=self.ridge,
                include_gaps=self.include_gaps,
                alphabet_support=self.alphabet_support,
                shrinkage=self.shrinkage,
                conditioning=self.conditioning,
                diagnostics=diagnostics,
                min_valid=min_valid,
                min_valid_weight=min_valid_weight,
                **(
                    {"factor_solver": self.factor_solver}
                    if self.factor_solver != "numpy_legacy"
                    else {}
                ),
            )
        from triplet_coevolution.inference.plm import fit_plm

        fitted = fit_plm(
            msa,
            weights,
            options=self.plm_options(),
            min_valid=min_valid,
            min_valid_weight=min_valid_weight,
            conditioning=self.conditioning,
        )
        if diagnostics is not None:
            diagnostics.update(fitted.diagnostics)
        return fitted.blocks

    def preflight(self, msa, positive=None):
        if self.backend == "plm":
            from triplet_coevolution.inference.plm import plm_preflight

            return plm_preflight(msa, self.plm_options(), conditioning=self.conditioning)
        n = msa.n_observations
        factor_backend(self.factor_solver)
        dims = {}
        for comp in msa.components:
            q = len(CANONICAL[comp.molecule]) + self.include_gaps
            if self.alphabet_support == "fixed":
                dims[comp.name] = q * comp.n_sites
            else:
                values = comp.matrix if positive is None else comp.matrix[positive, :]
                sizes = [len(np.unique(col[(col >= 0) & (col < q)])) for col in values.T]
                dims[comp.name] = sum(k for k in sizes if k >= 2)
        details = {}
        for roles in [tuple("ABC")] if self.conditioning == "joint" else combinations("ABC", 2):
            d = sum(dims[c] for c in roles)
            dual = n < d and n <= MAX_SOLVE_DIMENSION and n * d <= MAX_DUAL_ELEMENTS
            if d > MAX_SOLVE_DIMENSION and not dual:
                raise ValueError(
                    f"mf panel {''.join(roles)} exceeds both primal and dual solver budgets"
                )
            batch = max(1, min(n, 1024, 1_000_000 // max(1, d)))
            space = 8 * (3 * n * d + 3 * n * n) if dual else 8 * (4 * d * d + batch * d)
            details["".join(roles)] = {
                "features": d,
                "solver": "dual" if dual else "primal",
                "estimated_numeric_bytes": space,
                "factor_solver": self.factor_solver,
            }
        return {
            "scopes": details,
            "estimated_numeric_bytes": max(v["estimated_numeric_bytes"] for v in details.values()),
        }
