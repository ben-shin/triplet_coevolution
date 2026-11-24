"""Versioned storage of complete pair-score matrices on retained alignment coordinates."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from triplet_coevolution.core.coordinates import ReferenceCoordinates, SiteCoordinate
from triplet_coevolution.inference.mi import MIBlock, TripletMI
from triplet_coevolution.inference.potts import DCABlock
from triplet_coevolution.inference.scores import get_score

PAIRS = ("AB", "AC", "BC")
FAMILIES = {"mi": ("mi_bits", "mi_apc"), "dca": ("dca_frobenius", "dca_apc")}
MANIFEST_NAME = "score_manifest.json"
PAYLOAD_NAME = "pair_scores.npz"
SCHEMA = "triplet.pair_scores.v1"
CONVENTION = "Current and original exported columns and reference positions are one-based. Stored original_columns are zero-based. A null reference position denotes a reference gap."


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class ExportedCoordinates(ReferenceCoordinates):
    """A recovered axis retaining the fitted column numbers printed in a v1 score view."""

    source_current_columns: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not any(
            (
                self.original_columns,
                self.reference_positions,
                self.symbols,
                self.source_current_columns,
            )
        ):
            # A v1 ranked view can legitimately have no reported site in this component.
            # The source summary still identifies the reference and records retained columns.
            if (
                self.component not in tuple("ABC")
                or not isinstance(self.reference_id, str)
                or not self.reference_id
                or type(self.start) is not int
                or self.start < 1
                or self.strand not in ("+", "-")
            ):
                raise ValueError("invalid metadata for an empty exported coordinate axis")
            for key in (
                "original_columns",
                "reference_positions",
                "symbols",
                "source_current_columns",
            ):
                object.__setattr__(self, key, ())
            return
        super().__post_init__()
        cols = tuple(self.source_current_columns)
        if (
            len(cols) != self.n_sites
            or any(type(c) is not int or c < 1 for c in cols)
            or any(b <= a for a, b in pairwise(cols))
        ):
            raise ValueError("invalid source current-column map in legacy scores")
        object.__setattr__(self, "source_current_columns", cols)

    def site(self, index: int) -> SiteCoordinate:
        value = super().site(index)
        return replace(value, current_column=self.source_current_columns[index])


@dataclass(frozen=True)
class ScoreStore:
    """Complete retained-site scores, pair support, reference maps and replay metadata."""

    maps: tuple[ReferenceCoordinates, ...]
    scores: dict[str, dict[str, np.ndarray]]
    n_valid: dict[str, np.ndarray]
    valid_weight: dict[str, np.ndarray]
    provenance: dict
    min_valid: int
    require_reference: bool

    def eligible(self, pair: str, *, reference_only: bool = False) -> np.ndarray:
        """Return support and reference eligibility, independent of finite method scores."""
        keep = self.n_valid[pair] >= self.min_valid
        if self.require_reference or reference_only:
            a, b = (self.maps["ABC".index(c)] for c in pair)
            x, y = (
                np.array([p is not None for p in axis.reference_positions], dtype=bool)
                for axis in (a, b)
            )
            keep &= x[:, None] & y[None, :]
        return keep

    def to_results(self) -> tuple[TripletMI, dict[str, DCABlock]]:
        """Reconstruct ranking and plotting inputs without refitting the models."""
        blocks = [
            MIBlock(
                *p,
                self.scores["mi_bits"][p],
                self.scores["mi_apc"][p],
                self.n_valid[p],
                self.valid_weight[p],
            )
            for p in PAIRS
        ]
        mi = TripletMI(
            *blocks,
            self.provenance["n_observations"],
            self.provenance["weight_sum"],
            self.provenance["include_gaps"],
            self.min_valid,
        )
        dca = {
            p: DCABlock(
                self.scores["dca_frobenius"][p],
                self.scores["dca_apc"][p],
                self.n_valid[p],
                self.valid_weight[p],
            )
            for p in PAIRS
            if "dca_apc" in self.scores
        }
        return mi, dca


def _validate(store):
    if tuple(m.component for m in store.maps) != tuple("ABC"):
        raise ValueError("score store needs coordinate maps in A, B, C order")
    if set(store.scores) not in (set(FAMILIES["mi"]), set(FAMILIES["mi"] + FAMILIES["dca"])):
        raise ValueError("score store requires raw and corrected matrices for every fitted method")
    if (
        type(store.min_valid) is not int
        or store.min_valid < 1
        or type(store.require_reference) is not bool
    ):
        raise ValueError("invalid score-store eligibility policy")
    prov = store.provenance
    if (
        not isinstance(prov, dict)
        or not prov.get("software_version")
        or not isinstance(prov.get("inputs"), dict)
    ):
        raise ValueError("score provenance requires software_version and input hashes")
    for name in ("A", "B", "C", "matches"):
        obj = prov["inputs"].get(name)
        digest = obj.get("sha256") if isinstance(obj, dict) else obj
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"invalid input SHA-256 for {name}")
    n = prov.get("n_observations")
    weight = prov.get("weight_sum")
    if (
        type(n) is not int
        or n < 1
        or not isinstance(weight, (int, float))
        or not np.isfinite(weight)
        or weight <= 0
    ):
        raise ValueError("invalid recorded model observation count or effective depth")
    if type(prov.get("include_gaps")) is not bool:
        raise ValueError("missing MI gap policy")
    if prov.get("apc_universe", "retained") not in {"retained", "reference"}:
        raise ValueError("unknown APC universe")
    sparse_legacy = prov.get("origin") == "legacy_uncapped_or_complete_views"
    for name, blocks in {
        **store.scores,
        "n_valid": store.n_valid,
        "valid_weight": store.valid_weight,
    }.items():
        if set(blocks) != set(PAIRS):
            raise ValueError(f"missing interface in {name}")
        for pair, block in blocks.items():
            shape = tuple(store.maps["ABC".index(c)].n_sites for c in pair)
            if not isinstance(block, np.ndarray) or block.shape != shape:
                raise ValueError(f"invalid matrix shape for {name} {pair}")
            if name == "n_valid":
                if (
                    block.dtype.kind not in "iu"
                    or np.any(block < (-1 if sparse_legacy else 0))
                    or np.any(block > n)
                ):
                    raise ValueError(f"invalid pair counts for {pair}")
            elif block.dtype.kind != "f" or np.isinf(block).any():
                raise ValueError(f"invalid numeric matrix for {name} {pair}")
            elif name == "valid_weight" and (
                not np.array_equal(np.isnan(block), store.n_valid[pair] < 0)
                or np.any(block < 0)
                or np.any(block > weight + 1e-8 * max(1, weight))
            ):
                raise ValueError(f"invalid pair support weights for {pair}")
            elif name in store.scores and np.any((store.n_valid[pair] < 0) & np.isfinite(block)):
                raise ValueError(f"finite {name} on a legacy cell of unknown support in {pair}")


def from_results(
    maps: tuple[ReferenceCoordinates, ...],
    mi: TripletMI,
    dca: Mapping[str, DCABlock] | None = None,
    *,
    provenance: Mapping,
    require_reference: bool = False,
) -> ScoreStore:
    """Build a native store from full fitted matrices; supply original input SHA-256 identities."""
    scores = {
        field: {p: getattr(getattr(mi, p.lower()), field) for p in PAIRS}
        for field in FAMILIES["mi"]
    }
    counts = {p: getattr(mi, p.lower()).n_valid for p in PAIRS}
    weights = {p: getattr(mi, p.lower()).valid_weight for p in PAIRS}
    if dca:
        if set(dca) != set(PAIRS):
            raise ValueError("DCA needs all three interface blocks")
        for pair, block in dca.items():
            if not np.array_equal(block.n_valid, counts[pair]) or not np.allclose(
                block.valid_weight, weights[pair], rtol=1e-10, atol=1e-10
            ):
                raise ValueError(
                    "DCA and MI support differ; separate support definitions are required"
                )
        scores.update(
            dca_frobenius={p: b.frobenius for p, b in dca.items()},
            dca_apc={p: b.corrected for p, b in dca.items()},
        )
    prov = {
        **provenance,
        "n_observations": int(mi.n_observations),
        "weight_sum": float(mi.weight_sum),
        "include_gaps": bool(mi.include_gaps),
        "origin": "native_full_matrices",
    }
    # File contents identify sources independently of machine-specific path names.
    prov["inputs"] = {
        name: {"sha256": obj.get("sha256")} if isinstance(obj, dict) else obj
        for name, obj in prov.get("inputs", {}).items()
    }
    store = ScoreStore(
        tuple(maps), scores, counts, weights, prov, int(mi.min_valid), require_reference
    )
    _validate(store)
    return store


def _descriptors(store):
    return {
        name: get_score(name).metadata(
            apc_universe=store.provenance.get("apc_universe", "retained")
        )
        for name in store.scores
    }


def write_score_store(directory: Path, store: ScoreStore) -> Path:
    """Publish the payload and then its verified manifest into a new or staged run directory."""
    _validate(store)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / MANIFEST_NAME
    if manifest_path.exists() or (directory / PAYLOAD_NAME).exists():
        raise FileExistsError("a score artifact already exists in the destination")
    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        stage = Path(scratch)
        arrays = {
            f"{name}_{pair}": a
            for name, blocks in {
                **store.scores,
                "n_valid": store.n_valid,
                "valid_weight": store.valid_weight,
            }.items()
            for pair, a in blocks.items()
        }
        np.savez_compressed(stage / PAYLOAD_NAME, **arrays)
        manifest = {
            "schema": SCHEMA,
            "coordinate_convention": CONVENTION,
            "payload": PAYLOAD_NAME,
            "payload_sha256": file_digest(stage / PAYLOAD_NAME),
            "coordinates": [asdict(m) for m in store.maps],
            "scores": _descriptors(store),
            "min_valid": store.min_valid,
            "require_reference": store.require_reference,
            "provenance": store.provenance,
        }
        manifest["manifest_sha256"] = hashlib.sha256(_json_bytes(manifest)).hexdigest()
        (stage / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        load_score_store(stage)
        os.link(stage / PAYLOAD_NAME, directory / PAYLOAD_NAME)
        try:
            os.link(stage / MANIFEST_NAME, manifest_path)
        except BaseException:
            (directory / PAYLOAD_NAME).unlink()
            raise
    return manifest_path


def load_score_store(path: Path) -> ScoreStore:
    """Load a verified artifact. Reject incomplete, corrupt or incompatible score archives."""
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST_NAME
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        saved = meta.pop("manifest_sha256")
        if hashlib.sha256(_json_bytes(meta)).hexdigest() != saved:
            raise ValueError("score manifest checksum mismatch")
        if (
            meta["schema"] != SCHEMA
            or meta["payload"] != PAYLOAD_NAME
            or meta["coordinate_convention"] != CONVENTION
        ):
            raise ValueError("unsupported score schema or coordinate convention")
        payload = path.parent / PAYLOAD_NAME
        maps = tuple(
            ExportedCoordinates(**m) if "source_current_columns" in m else ReferenceCoordinates(**m)
            for m in meta["coordinates"]
        )
        fields = [*meta["scores"], "n_valid", "valid_weight"]
        with payload.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != meta["payload_sha256"]:
                raise ValueError("score payload checksum mismatch")
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as data:
                keys = {f"{field}_{p}" for field in fields for p in PAIRS}
                if set(data.files) != keys or len(data.files) != len(keys):
                    raise ValueError("unexpected or missing score matrices")
                blocks = {name: {p: data[f"{name}_{p}"] for p in PAIRS} for name in fields}
        store = ScoreStore(
            maps,
            {k: blocks[k] for k in meta["scores"]},
            blocks["n_valid"],
            blocks["valid_weight"],
            meta["provenance"],
            meta["min_valid"],
            meta["require_reference"],
        )
        _validate(store)
        if _descriptors(store) != meta["scores"]:
            raise ValueError(
                "score descriptors disagree with the registered method, units or correction"
            )
        return store
    except (KeyError, TypeError, BadZipFile) as exc:
        raise ValueError(f"invalid or incomplete score manifest: {exc}") from exc


def import_legacy_scores(directory: Path) -> ScoreStore:
    """Read complete v1 score tables with their original run summary and declared scope."""
    from triplet_coevolution.io.legacy_scores import read_legacy_scores

    return read_legacy_scores(Path(directory))
