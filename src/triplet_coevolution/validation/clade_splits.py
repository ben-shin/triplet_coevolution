"""Observation-keyed blocked splits and reproducible whole-block bootstrap samples."""

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.validation.null_runner import ReplicateStreams


def identity(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _ids(values, name):
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} requires a sequence of identifiers")
    items = tuple(values)
    if (
        not items
        or any(not isinstance(v, str) or not v.strip() for v in items)
        or len(set(items)) != len(items)
    ):
        raise ValueError(f"{name} requires unique nonempty identifiers")
    return items


def _validate_groups(ids, groups):
    if not isinstance(groups, Mapping) or set(groups) != set(ids):
        raise ValueError("clade groups must cover exactly the original observation IDs")
    if any(not isinstance(groups[v], str) or not groups[v].strip() for v in ids):
        raise ValueError("each observation requires a nonempty clade identifier")


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


@dataclass(frozen=True)
class SplitPlan:
    observation_ids: tuple[str, ...]
    groups: dict
    folds: tuple[Fold, ...]
    mode: str
    source: dict

    def metadata(self):
        data = {
            "schema": "triplet.clade-split.v1",
            "mode": self.mode,
            "source": self.source,
            "status": "available" if self.folds else "insufficient_independent_blocks",
            "observation_ids": list(self.observation_ids),
            "groups": self.groups,
            "block_sizes": dict(sorted(Counter(self.groups.values()).items())),
            "folds": [asdict(v) for v in self.folds],
        }
        return {**data, "plan_sha256": identity(data)}


def make_split_plan(observation_ids, groups, *, test_groups=None, source=None):
    """Fix whole clades; retain the source row order within each disjoint partition."""
    ids = _ids(observation_ids, "observations")
    _validate_groups(ids, groups)
    groups = {key: groups[key] for key in ids}
    all_groups = tuple(sorted(set(groups.values())))
    if source is not None and not isinstance(source, Mapping):
        raise ValueError("split source provenance requires a mapping")
    source = dict(source or {})
    identity(source)
    if test_groups is None:
        partitions = [(g,) for g in all_groups] if len(all_groups) >= 2 else []
        mode = "leave_one_clade_out"
    else:
        chosen = tuple(sorted(_ids(test_groups, "held-out clades")))
        if not set(chosen) < set(all_groups):
            raise ValueError("held-out clades must be known and leave at least one training clade")
        partitions = [chosen]
        mode = "prespecified_holdout"
    if len(ids) * len(partitions) > 2_000_000:
        raise ValueError("split manifest exceeds 2000000 row assignments; specify held-out clades")
    folds = []
    for chosen in partitions:
        test = set(chosen)
        train_ids = tuple(i for i in ids if groups[i] not in test)
        test_ids = tuple(i for i in ids if groups[i] in test)
        train_groups = tuple(g for g in all_groups if g not in test)
        # The ordered partition and block evidence identify the fold independently of sequence values.
        fold_id = identity(
            [
                "clade-fold.v1",
                [(i, groups[i]) for i in train_ids],
                [(i, groups[i]) for i in test_ids],
            ]
        )
        folds.append(Fold(fold_id, train_ids, test_ids, train_groups, chosen))
    return SplitPlan(ids, groups, tuple(folds), mode, source)


def load_split_plan(path, observation_ids=None):
    """Reload the exact ordered source partition, verifying its digest and disjoint folds."""

    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate keys in split manifest")
        return result

    data = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique)
    if not isinstance(data, dict) or data.get("schema") != "triplet.clade-split.v1":
        raise ValueError("unsupported clade split manifest")
    supplied_hash = data.get("plan_sha256")
    if identity({k: v for k, v in data.items() if k != "plan_sha256"}) != supplied_hash:
        raise ValueError("split manifest checksum mismatch")
    ids = tuple(data["observation_ids"])
    if observation_ids is not None and ids != tuple(observation_ids):
        raise ValueError("split manifest source observation IDs/order differ")
    if data["mode"] not in ("prespecified_holdout", "leave_one_clade_out"):
        raise ValueError("unsupported split mode")
    held_out = None
    if data["mode"] == "prespecified_holdout":
        if len(data["folds"]) != 1:
            raise ValueError("a prespecified holdout must contain one fold")
        held_out = data["folds"][0]["test_groups"]
    replay = make_split_plan(ids, data["groups"], test_groups=held_out, source=data["source"])
    if replay.metadata()["plan_sha256"] != supplied_hash:
        raise ValueError("split manifest contents disagree with reconstructed disjoint partition")
    return replay


def subset_rows(msa, observation_ids):
    ids = _ids(observation_ids, "source subset")
    by_id = {row.observation_id: row for row in msa.matches}
    if not set(ids) <= by_id.keys():
        raise ValueError("unknown source observation ID")
    matches = tuple(by_id[key] for key in ids)
    components = [
        component.take_ids([getattr(row, label) for row in matches])
        for component, label in zip(msa.components, ("a_id", "b_id", "c_id"), strict=True)
    ]
    return MatchedTripletMSA(*components, matches)


@dataclass(frozen=True)
class BootstrapSample:
    msa: MatchedTripletMSA
    source_ids: tuple[str, ...]
    groups: dict[str, str]
    sampled_groups: tuple[str, ...]


def bootstrap_sample(msa, groups, *, seed, replicate, fold_id, eligible_ids=None):
    """Sample G training blocks with replacement. Every occurrence retains all matched rows."""
    seed, replicate = integer(seed, "seed"), integer(replicate, "replicate")
    if not isinstance(fold_id, str) or not fold_id.strip():
        raise ValueError("fold_id must be nonempty")
    ids = tuple(row.observation_id for row in msa.matches)
    _validate_groups(ids, groups)
    allowed = _ids(eligible_ids, "eligible_ids") if eligible_ids is not None else ids
    if not set(allowed) <= set(ids):
        raise ValueError("bootstrap includes an unknown observation")
    if set(allowed) != set(ids):
        chosen = set(allowed)
        if {groups[i] for i in chosen} & {groups[i] for i in ids if i not in chosen}:
            raise ValueError("bootstrap eligibility splits a clade")
    row_index = {key: i for i, key in enumerate(ids)}
    blocks = defaultdict(list)
    for key in allowed:
        blocks[groups[key]].append(row_index[key])
    labels = sorted(blocks)
    rng = ReplicateStreams(seed, "clade_block_bootstrap.v1", replicate).generator(fold_id)
    sampled = tuple(labels[int(i)] for i in rng.integers(len(labels), size=len(labels)))
    n_rows = sum(len(blocks[label]) for label in sampled)
    if n_rows > 1_000_000 or n_rows * sum(c.n_sites for c in msa.components) > 50_000_000:
        raise ValueError("block draw exceeds the workspace of 1000000 rows or 50000000 symbols")
    indexes, aliases, instance_groups = [], [], {}
    for occurrence, label in enumerate(sampled):
        for member, index in enumerate(blocks[label]):
            alias = f"bs.{fold_id}.{replicate}.{occurrence}.{member}"
            aliases.append(alias)
            indexes.append(index)
            instance_groups[alias] = f"block.{occurrence}"
    components = [
        ComponentMSA(
            c.name,
            c.molecule,
            tuple(f"{c.name}.{a}" for a in aliases),
            tuple(c.sequences[i] for i in indexes),
        )
        for c in msa.components
    ]
    matches = tuple(TripletMatch(a, *(c.ids[i] for c in components)) for i, a in enumerate(aliases))
    return BootstrapSample(
        MatchedTripletMSA(*components, matches),
        tuple(ids[i] for i in indexes),
        instance_groups,
        sampled,
    )
