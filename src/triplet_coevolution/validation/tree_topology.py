"""Strict, observation-aligned Newick input and topology comparisons."""

import csv
import hashlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import hypot, isfinite
from pathlib import Path

import numpy as np

TOKEN = re.compile(r"\s+|\[[^\[\]]*\]|'(?:[^']|'')*'|[(),:;]|[^\s()\[\],:;']+|.", re.DOTALL)
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
PUNCTUATION = {"(", ")", ",", ":", ";", "[", "]", "'"}
MAX_NEWICK_BYTES = 64 * 1024 * 1024


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _tip_map(value, tips, ids):
    source_hash = None
    if isinstance(value, (str, Path)):
        path = Path(value)
        source = path.read_bytes()
        reader = csv.DictReader(io.StringIO(source.decode("utf-8-sig")), delimiter="\t")
        if reader.fieldnames != ["tip_label", "observation_id"]:
            raise ValueError("tip mapping TSV must have columns tip_label and observation_id")
        mapping = {}
        for row in reader:
            key, val = row["tip_label"], row["observation_id"]
            if None in row or not key or not val or key in mapping:
                raise ValueError("invalid or repeated entry in tip mapping TSV")
            mapping[key] = val
        source_hash = hashlib.sha256(source).hexdigest()
    elif value is None:
        mapping = {tip: tip for tip in tips}
    elif isinstance(value, Mapping):
        mapping = dict(value)
    else:
        raise TypeError("tip_map must be a mapping or local TSV path")
    if any(
        not isinstance(label, str) or not label.strip()
        for pair in mapping.items()
        for label in pair
    ):
        raise ValueError("tip mapping requires nonempty textual tip and observation IDs")
    if set(mapping) != set(tips) or set(mapping.values()) != set(ids) or len(mapping) != len(ids):
        raise ValueError(
            "tip mapping must be a bijection between all tree leaves and matched observations"
        )
    return mapping, source_hash


def _parse(text):
    # Internal numeric annotations remain labels; confidence is never interpreted as distance.
    tokens = (
        m.group()
        for m in TOKEN.finditer(text)
        if not m.group().isspace() and not (m.group().startswith("[") and m.group().endswith("]"))
    )
    cursor = next(tokens, None)
    parents: list[int] = []
    lengths: list[float | None] = []
    labels: list[str] = []
    child_counts: list[int] = []
    stack: list[int] = []

    def label(token):
        if token is None or token in PUNCTUATION:
            raise ValueError("expected a Newick terminal or internal label")
        return token[1:-1].replace("''", "'") if token.startswith("'") else token

    while True:
        node = len(parents)
        parents.append(stack[-1] if stack else -1)
        labels.append("")
        lengths.append(None)
        child_counts.append(0)
        if stack:
            child_counts[stack[-1]] += 1
        if cursor == "(":
            stack.append(node)
            cursor = next(tokens, None)
            continue
        labels[node] = label(cursor)
        if not labels[node].strip():
            raise ValueError("tree tip labels must be nonempty")
        cursor = next(tokens, None)
        while True:
            if cursor == ":":
                token = next(tokens, None)
                if token is None or NUMBER.fullmatch(token) is None:
                    raise ValueError("invalid Newick branch length")
                length = float(token)
                if not isfinite(length) or length < 0:
                    raise ValueError("Newick branch lengths must be finite and nonnegative")
                lengths[node] = length
                cursor = next(tokens, None)
            if parents[node] != -1 and lengths[node] is None:
                raise ValueError("every nonroot branch requires an explicit length")
            if cursor == "," and stack:
                cursor = next(tokens, None)
                break
            if cursor == ")" and stack:
                node = stack.pop()
                cursor = next(tokens, None)
                if cursor not in {None, ":", ",", ")", ";"}:
                    labels[node] = label(cursor)
                    cursor = next(tokens, None)
                continue
            if cursor == ";" and not stack and next(tokens, None) is None:
                return parents, lengths, labels, child_counts
            raise ValueError("expected a single complete semicolon-terminated Newick tree")


@dataclass(frozen=True)
class ValidatedTree:
    observation_ids: tuple[str, ...]
    tip_labels: tuple[str, ...]
    parents: np.ndarray = field(repr=False)
    lengths: np.ndarray = field(repr=False)
    tips: np.ndarray = field(repr=False)
    levels: tuple[np.ndarray, ...] = field(repr=False)
    outdegree: np.ndarray = field(repr=False)
    source_sha256: str
    tip_map_sha256: str | None
    root_stem: float | None
    source_root_marker: str | None

    def metadata(self) -> dict:
        branches = self.lengths[1:]
        return {
            "source_sha256": self.source_sha256,
            "tip_map_file_sha256": self.tip_map_sha256,
            "tip_map_sha256": _hash(list(zip(self.tip_labels, self.observation_ids))),
            "tip_labels_in_observation_order": list(self.tip_labels),
            "observation_ids": list(self.observation_ids),
            "tips": len(self.tips),
            "nodes": len(self.parents),
            "edges": len(branches),
            "zero_length_edges": int(np.count_nonzero(branches == 0)),
            "polytomies": int(np.count_nonzero(self.outdegree > 2)),
            "unary_nodes": int(np.count_nonzero(self.outdegree == 1)),
            "minimum_length": float(branches.min()),
            "maximum_length": float(branches.max()),
            "root_stem_ignored_at_stationarity": self.root_stem,
            "newick_root_marker": self.source_root_marker,
            "root": "first Newick node, stationary ancestral state at each independent site",
        }

    def clades_at_depth(self, depth: int) -> dict[str, str]:
        """Partition tips by topological edge depth from the supplied file root."""
        if type(depth) is not int or depth < 0:
            raise ValueError("clade depth must be a nonnegative integer")
        ancestors = np.zeros(len(self.parents), dtype=int)
        for level, nodes in enumerate(self.levels[1:], start=1):
            ancestors[nodes] = nodes if level <= depth else ancestors[self.parents[nodes]]
        groups: dict[int, list[str]] = {}
        for key, node in zip(self.observation_ids, ancestors[self.tips], strict=True):
            groups.setdefault(int(node), []).append(key)
        return {key: "clade-" + _hash(sorted(keys)) for keys in groups.values() for key in keys}


def load_tree(path: str | Path, observation_ids: Sequence[str], *, tip_map=None) -> ValidatedTree:
    """Read exactly one branch-length Newick tree and align every labeled tip to its observation."""
    path = Path(path)
    if path.stat().st_size > MAX_NEWICK_BYTES:
        raise ValueError("Newick tree exceeds the 64 MiB input limit")
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    ids = tuple(observation_ids)
    if (
        isinstance(observation_ids, (str, bytes))
        or len(ids) < 2
        or any(not isinstance(i, str) or not i.strip() for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("tree alignment requires at least two unique nonempty observation IDs")
    parent, length, labels, child_count = _parse(text)
    lookup = {}
    for i, kids in enumerate(child_count):
        if kids == 0:
            if labels[i] in lookup:
                raise ValueError(f"duplicate tree tip {labels[i]!r}")
            lookup[labels[i]] = i
    mapping, source_hash = _tip_map(tip_map, lookup, ids)
    reverse = {key: tip for tip, key in mapping.items()}
    ordered = tuple(reverse[key] for key in ids)
    levels: list[list[int]] = []
    depths: list[int] = []
    for i, p in enumerate(parent):
        d = 0 if p < 0 else depths[p] + 1
        depths.append(d)
        if len(levels) == d:
            levels.append([])
        levels[d].append(i)
    nodes = tuple(np.array(v, dtype=int) for v in levels)
    arrays = (
        np.array(parent, dtype=int),
        np.array([0.0, *length[1:]], dtype=float),
        np.array([lookup[t] for t in ordered], dtype=int),
        np.array(child_count, dtype=int),
    )
    for arr in (*arrays, *nodes):
        arr.setflags(write=False)
    root_flag = re.match(r"\s*\[&([RU])\]", text)
    return ValidatedTree(
        ids,
        ordered,
        arrays[0],
        arrays[1],
        arrays[2],
        nodes,
        arrays[3],
        hashlib.sha256(raw).hexdigest(),
        source_hash,
        length[0],
        root_flag.group(1) if root_flag else None,
    )


def _splits(tree):
    # Commutative SHA-256 leaf fingerprints avoid storing O(tips²) descendant sets on ladders.
    n = len(tree.tips)
    signature = [0] * len(tree.parents)
    size = [0] * len(signature)
    for node, key in zip(tree.tips, tree.observation_ids, strict=True):
        signature[node] = int.from_bytes(hashlib.sha256(key.encode()).digest(), "big")
        size[node] = 1
    for i in range(len(signature) - 1, 0, -1):
        p = tree.parents[i]
        signature[p] ^= signature[i]
        size[p] += size[i]
    rooted: set[tuple[int, int]] = set()
    internal: set[tuple[int, int]] = set()
    edges: dict[tuple[int, int], float] = {}
    for i in range(1, len(signature)):
        if size[i] == n:
            continue
        if 1 < size[i] < n:
            rooted.add((size[i], signature[i]))
        bipart = min((size[i], signature[i]), (n - size[i], signature[0] ^ signature[i]))
        edges[bipart] = edges.get(bipart, 0.0) + float(tree.lengths[i])
        if not isfinite(edges[bipart]):
            raise ValueError("combined unrooted branch length exceeds numerical range")
        if 1 < size[i] < n - 1:
            internal.add(bipart)
    return rooted, internal, edges


def compare_trees(left: ValidatedTree, right: ValidatedTree) -> dict:
    """Compare named-tip rooted clades, unrooted splits and branch-score distance."""
    if set(left.observation_ids) != set(right.observation_ids):
        raise ValueError("tree comparison requires identical mapped observation sets")
    r1, s1, e1 = _splits(left)
    r2, s2, e2 = _splits(right)
    changed = len(s1 ^ s2)
    positive1 = {s for s in s1 if e1[s] > 0}
    positive2 = {s for s in s2 if e2[s] > 0}
    distance = hypot(*(e1.get(s, 0) - e2.get(s, 0) for s in e1.keys() | e2.keys()))
    if not isfinite(distance):
        raise ValueError("branch-score distance exceeds numerical range")
    return {
        "left_sha256": left.source_sha256,
        "right_sha256": right.source_sha256,
        "rooted_clade_symmetric_difference": len(r1 ^ r2),
        "unrooted_rf": changed,
        "unrooted_splits_left": len(s1),
        "unrooted_splits_right": len(s2),
        "normalized_unrooted_rf": changed / (len(s1) + len(s2)) if s1 or s2 else 0.0,
        "unrooted_rf_positive_edges": len(positive1 ^ positive2),
        "branch_score_distance": distance,
        "branch_unit": "supplied Newick length; unrooted corresponding edges summed across artificial roots",
    }
