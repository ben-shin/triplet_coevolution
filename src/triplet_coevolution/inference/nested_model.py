"""Frozen source-site encoding, fitted conditional model and portable checked representation."""

import hashlib
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.mi import _get_weights
from triplet_coevolution.inference.nested_solver import (
    Conditionals,
    Layout,
    NestedOptions,
    exact_law,
)
from triplet_coevolution.validation._null_checkpoint import canonical
from triplet_coevolution.validation.clade_splits import identity


@dataclass(frozen=True)
class PanelSite:
    component: str
    original_column: int
    reference_id: str
    reference_position: int | None
    reference_symbol: str
    molecule: str
    native_to_bin: tuple[int, ...]

    def __post_init__(self):
        bins = tuple(self.native_to_bin)
        if (
            self.component not in ("A", "B", "C")
            or type(self.original_column) is not int
            or self.original_column < 1
            or not isinstance(self.reference_id, str)
            or not self.reference_id
            or self.molecule not in CANONICAL
            or (
                self.reference_position is not None
                and (type(self.reference_position) is not int or self.reference_position < 1)
            )
            or not isinstance(self.reference_symbol, str)
            or len(self.reference_symbol) != 1
            or not bins
            or any(type(x) is not int or x < 0 for x in bins)
            or set(bins) != set(range(max(bins) + 1))
        ):
            raise ValueError("malformed nested source-site definition or state bins")
        object.__setattr__(self, "native_to_bin", bins)

    @property
    def key(self):
        return self.component, self.original_column


def encode_panel(msa, maps, sites):
    axes = {
        m.component: (c, m, {v + 1: i for i, v in enumerate(m.original_columns)})
        for c, m in zip(msa.components, maps, strict=True)
        if c.name == m.component and c.n_sites == m.n_sites
    }
    if set(axes) != {"A", "B", "C"}:
        raise ValueError("source coordinate maps differ from incoming alignment axes")
    rows = np.full((msa.n_observations, len(sites)), -1, dtype=int)
    for j, site in enumerate(sites):
        comp, m, by_pos = axes[site.component]
        i = by_pos.get(site.original_column)
        if (
            i is None
            or m.reference_id != site.reference_id
            or comp.molecule.value != site.molecule
            or m.reference_positions[i] != site.reference_position
            or m.symbols[i] != site.reference_symbol
        ):
            raise ValueError(
                f"unseen source panel differs at {site.component} original column {site.original_column}"
            )
        x = comp.matrix[:, i]
        keep = (x >= 0) & (x < len(site.native_to_bin))
        rows[keep, j] = np.asarray(site.native_to_bin)[x[keep]]
    return rows


@dataclass(frozen=True)
class NestedModel:
    kind: str
    sites: tuple[PanelSite, ...]
    edges: tuple
    triples: tuple
    options: NestedOptions
    theta: np.ndarray
    metadata_json: str

    def __post_init__(self):
        if self.kind not in ("pair_only", "pair_plus_k") or (
            self.kind == "pair_only" and self.triples
        ):
            raise ValueError("inconsistent nested model order")
        if len({s.key for s in self.sites}) != len(self.sites):
            raise ValueError("duplicate fitted panel site")
        if any(
            len(s.native_to_bin) != len(CANONICAL[s.molecule]) + self.options.include_gaps
            for s in self.sites
        ):
            raise ValueError("model state maps do not cover the declared native alphabet")
        layout = self.layout()
        x = np.array(self.theta, dtype="<f8", copy=True)
        if (
            x.shape != (layout.size,)
            or not np.isfinite(x).all()
            or not np.allclose(layout.projected(x), x, rtol=1e-10, atol=1e-10)
        ):
            raise ValueError("invalid fitted zero-sum coefficients")
        x.setflags(write=False)
        object.__setattr__(self, "theta", x)
        info = json.loads(self.metadata_json)
        if (
            not isinstance(info, dict)
            or not isinstance(info.get("diagnostics"), dict)
            or not info.get("training_design_sha256")
        ):
            raise ValueError("missing nested fit metadata")
        design = {
            k: v
            for k, v in info.items()
            if k not in {"diagnostics", "training_design_sha256", "fold_id", "split_sha256"}
        }
        if identity(design) != info["training_design_sha256"]:
            raise ValueError("nested training design checksum mismatch")
        if (
            info.get("schema") != "triplet.nested-training.v1"
            or canonical(info.get("sites")) != canonical([asdict(s) for s in self.sites])
            or canonical(info.get("edges")) != canonical(self.edges)
            or info.get("options") != asdict(self.options)
            or info["diagnostics"].get("coefficients") != layout.size
            or (
                self.kind == "pair_plus_k"
                and canonical(info.get("selected_triples")) != canonical(self.triples)
            )
        ):
            raise ValueError("nested model and bound training design disagree")

    def layout(self):
        return Layout(
            tuple(max(s.native_to_bin) + 1 for s in self.sites),
            self.edges,
            self.triples,
            self.options,
        )

    @property
    def metadata(self):
        return json.loads(self.metadata_json)

    def exact_law(self):
        return exact_law(self.layout(), self.theta)

    def header(self):
        return {
            "schema": "triplet.nested-model.v1",
            "kind": self.kind,
            "sites": [asdict(s) for s in self.sites],
            "edges": self.edges,
            "triples": self.triples,
            "options": asdict(self.options),
            "fit": self.metadata,
        }

    def digest(self):
        h = hashlib.sha256(canonical(self.header()))
        h.update(self.theta.tobytes(order="C"))
        return h.hexdigest()


def save_model(path, model):
    header = canonical({**model.header(), "sha256": model.digest()})
    if len(header) > 32 * 1024**2:
        raise ValueError("nested model metadata exceeds 32 MiB")
    with Path(path).open("wb") as f:
        np.savez_compressed(f, header=np.frombuffer(header, dtype=np.uint8), theta=model.theta)


def load_model(path):
    def vector(member, expected):
        # Check the declared shape against bytes present before allocating a numeric array.
        stream = io.BytesIO(member)
        version = np.lib.format.read_magic(stream)
        readers = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }
        if version not in readers:
            raise ValueError("unsupported nested array version")
        shape, _, dtype = readers[version](stream)
        if (
            len(shape) != 1
            or dtype != np.dtype(expected)
            or shape[0] * dtype.itemsize != len(member) - stream.tell()
        ):
            raise ValueError("invalid nested model array dimensions or format")
        return np.frombuffer(member, dtype, offset=stream.tell())

    try:
        with zipfile.ZipFile(path) as z:
            files = z.infolist()
            if (
                sorted(i.filename for i in files) != ["header.npy", "theta.npy"]
                or sum(i.file_size for i in files) > 50 * 1024**2
            ):
                raise ValueError("invalid nested model archive members/size")
            raw = vector(z.read("header.npy"), "u1")
            theta = vector(z.read("theta.npy"), "<f8")
    except zipfile.BadZipFile as exc:
        raise ValueError("corrupt nested model archive") from exc

    def unique(pairs):
        d = dict(pairs)
        if len(pairs) != len(d):
            raise ValueError("duplicate nested metadata key")
        return d

    obj = json.loads(raw.tobytes().decode(), object_pairs_hook=unique)
    if (
        not isinstance(obj, dict)
        or set(obj) != {"schema", "kind", "sites", "edges", "triples", "options", "fit", "sha256"}
        or obj["schema"] != "triplet.nested-model.v1"
    ):
        raise ValueError("unsupported nested model schema")
    model = NestedModel(
        obj["kind"],
        tuple(PanelSite(**s) for s in obj["sites"]),
        tuple(map(tuple, obj["edges"])),
        tuple(map(tuple, obj["triples"])),
        NestedOptions(**obj["options"]),
        theta,
        canonical(obj["fit"]).decode(),
    )
    if model.digest() != obj["sha256"]:
        raise ValueError("nested model checksum mismatch")
    return model


def evaluate_model(model, msa, maps, *, weights=None):
    """Score new matched observations on identical source coordinates and frozen native bins."""
    if model.metadata["diagnostics"]["status"] != "converged":
        raise ValueError("held-out evaluation requires a converged nested model")
    rows = encode_panel(msa, maps, model.sites)
    w = _get_weights(weights, msa.n_observations, msa)
    good = (rows >= 0).all(axis=1) & (w > 0)
    loss = np.full(rows.shape, np.nan)
    if good.any():
        loss[good] = Conditionals(model.layout(), rows[good]).losses(model.theta)
    return {
        "row_ids": tuple(m.observation_id for m in msa.matches),
        "complete_mask": good,
        "loss_per_row_site_nats": loss,
        "weights": w,
        "scoreable_rows": int(good.sum()),
        "scoreable_weight": float(w[good].sum()),
    }
