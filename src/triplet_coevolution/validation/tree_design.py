"""Source-bound tree scenarios for analysis, sensitivity and independent grouping."""

import hashlib
import json
from dataclasses import dataclass, replace
from itertools import combinations
from math import isfinite
from pathlib import Path


def _mapping(data, keys, label):
    if not isinstance(data, dict) or set(data) - set(keys):
        raise ValueError(f"{label} must be an object containing only {', '.join(sorted(keys))}")
    return data


def _label(text, what):
    if not isinstance(text, str) or not text.strip() or any(ord(c) < 32 for c in text):
        raise ValueError(f"{what} requires nonempty text without control characters")
    return text


@dataclass(frozen=True)
class TreeSource:
    newick: Path
    tip_map: Path | None = None

    @classmethod
    def parse(cls, data, base):
        data = {"newick": data} if isinstance(data, (str, Path)) else data
        _mapping(data, {"newick", "tip_map"}, "tree source")
        if not data.get("newick"):
            raise ValueError("tree source requires a Newick file")
        return cls(
            (base / data["newick"]).resolve(),
            (base / data["tip_map"]).resolve() if data.get("tip_map") else None,
        )

    def load(self, ids):
        from triplet_coevolution.validation.tree_model import load_tree

        return load_tree(self.newick, ids, tip_map=self.tip_map)


@dataclass(frozen=True)
class TreeDesign:
    label: str
    source: str
    origin: str
    sources: tuple[tuple[str, TreeSource], ...]
    frequency_policy: str = "empirical"
    frequencies: object = None
    prior_mass: float = 0.0
    rate_scale: float = 1.0
    alternatives: tuple = ()

    @classmethod
    def parse(cls, data, base=None, *, alternative=False):
        if isinstance(data, cls):
            return data
        base = Path.cwd() if base is None else Path(base)
        _mapping(
            data,
            {
                "label",
                "source",
                "origin",
                "newick",
                "tip_map",
                "components",
                "frequency_policy",
                "frequencies",
                "prior_mass",
                "rate_scale",
                "alternatives",
            },
            "tree",
        )
        label = _label(data.get("label", "primary" if not alternative else None), "tree label")
        source = _label(data.get("source"), "tree source description")
        origin = data.get("origin", "unspecified")
        if origin not in {"independent", "outcome_derived", "synthetic", "unspecified"}:
            raise ValueError(
                "tree origin must be independent, outcome_derived, synthetic or unspecified"
            )
        if "components" in data:
            if (
                "newick" in data
                or "tip_map" in data
                or not isinstance(data["components"], dict)
                or set(data["components"]) != set("ABC")
            ):
                raise ValueError(
                    "component-tree scenario requires exactly A, B, C and their own optional tip maps"
                )
            sources = tuple((c, TreeSource.parse(data["components"][c], base)) for c in "ABC")
        else:
            sources = (
                (
                    "shared",
                    TreeSource.parse(
                        {k: data[k] for k in ("newick", "tip_map") if k in data}, base
                    ),
                ),
            )
        freq = data.get("frequency_policy", "empirical")
        values = data.get("frequencies")
        if freq not in {"empirical", "uniform", "supplied"} or (freq == "supplied") != (
            values is not None
        ):
            raise ValueError("choose empirical, uniform, or supplied with explicit frequencies")
        values = (base / values).resolve() if isinstance(values, (str, Path)) else values
        prior, rate = data.get("prior_mass", 0.0), data.get("rate_scale", 1.0)
        if any(type(v) not in (float, int) or not isfinite(v) or v < 0 for v in (prior, rate)) or (
            freq != "empirical" and prior
        ):
            raise ValueError(
                "tree prior_mass and rate_scale must be finite and nonnegative; prior mass applies to empirical frequencies"
            )
        alts = data.get("alternatives", [])
        if not isinstance(alts, list) or len(alts) > 16 or (alternative and alts):
            raise ValueError(
                "supply up to 16 independent alternative tree scenarios without nesting"
            )
        settings = {
            "frequency_policy": freq,
            "frequencies": values,
            "prior_mass": prior,
            "rate_scale": rate,
            "origin": origin,
        }
        alternatives = tuple(cls.parse({**settings, **a}, base, alternative=True) for a in alts)
        if (
            len({s.casefold() for s in [label, *[a.label for a in alternatives]]})
            != len(alternatives) + 1
        ):
            raise ValueError("tree scenario labels must be unique including case variants")
        return cls(label, source, origin, sources, freq, values, prior, rate, alternatives)

    @property
    def scenarios(self):
        return (replace(self, alternatives=()), *self.alternatives)

    def load(self, ids):
        from triplet_coevolution.validation.tree_model import compare_trees

        trees = {key: source.load(ids) for key, source in self.sources}
        meta = {
            "label": self.label,
            "source": self.source,
            "origin": self.origin,
            "scope": "supplied topology and branch lengths; conditional on the fitted panel",
            "topologies": {key: item.metadata() for key, item in trees.items()},
            "discordance": {
                a + b: compare_trees(trees[a], trees[b]) for a, b in combinations(trees, 2)
            },
        }
        if self.origin in {"outcome_derived", "unspecified"}:
            meta["interpretation"] = (
                "Exploratory sensitivity conditional on tree choice; its independence from the scored alignment has not been established."
            )
        return trees.get("shared", trees), meta

    def fit(self, msa, weights, *, loaded=None):
        from triplet_coevolution.validation.tree_model import prepare_tree_null

        tree, meta = loaded if loaded is not None else self.load(weights.observation_ids)
        values = self.frequencies
        if isinstance(values, Path):
            content = values.read_bytes()
            from triplet_coevolution.config import _unique

            values = json.loads(content.decode("utf-8-sig"), object_pairs_hook=_unique)
            meta["supplied_frequency_sha256"] = hashlib.sha256(content).hexdigest()
        model = prepare_tree_null(
            msa,
            tree,
            weights=weights,
            frequency_policy=self.frequency_policy,
            frequencies=values,
            prior_mass=self.prior_mass,
            rate_scale=self.rate_scale,
        )
        meta["evolution"] = model.metadata()
        return model, meta
