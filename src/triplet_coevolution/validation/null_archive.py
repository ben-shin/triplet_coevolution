"""Reconstruct the exact fitted cohort and declared hypothesis coordinates."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from triplet_coevolution.inference.candidates import candidate_id
from triplet_coevolution.inference.maxent import FrozenSupport, MaxEntOptions
from triplet_coevolution.inference.weights import SequenceWeightResult
from triplet_coevolution.io.alignments import read_msa
from triplet_coevolution.io.metadata import read_triplet_table
from triplet_coevolution.io.score_store import file_digest, load_score_store
from triplet_coevolution.msa.masking import apply_triplet_masks
from triplet_coevolution.msa.synchronize import synchronize


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="," if path.suffix == ".csv" else "\t")


def checked_file(root, name, hashes):
    p = (root / name).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file():
        raise ValueError(f"missing native run artifact: {name}")
    if name in hashes and file_digest(p) != hashes[name]:
        raise ValueError(f"run artifact checksum mismatch: {name}")
    return p


@dataclass
class RefitData:
    root: Path
    summary: dict
    store: object
    msa: object
    weights: SequenceWeightResult
    metadata: dict

    def artifact(self, name):
        return checked_file(self.root, name, self.summary.get("artifact_sha256", {}))

    @classmethod
    def load(cls, root, summary=None, *, inputs=None, live=None):
        root = Path(root).resolve()
        summary = summary if summary is not None else read_json(root / "run_summary.json")
        hashes = summary.get("artifact_sha256", {})
        for key in ("score_manifest", "score_payload", "weights"):
            if key not in summary.get("outputs", {}):
                raise ValueError(
                    "calibration requires a native score archive with row-level replay and weights"
                )
            checked_file(root, summary["outputs"][key], hashes)
        store = load_score_store(root / summary["outputs"]["score_manifest"])
        prov = store.provenance
        ids = tuple(prov["observation_ids"])
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("native score archive must identify distinct observations")
        if live is None:
            overrides = {} if inputs is None else inputs
            if not isinstance(overrides, dict) or set(overrides) - {"A", "B", "C", "matches"}:
                raise ValueError("input relocation accepts only A, B, C and matches")
            paths = {}
            for k in ("A", "B", "C", "matches"):
                paths[k] = Path(overrides.get(k, summary["inputs"][k]["path"]))
                if not paths[k].is_file() or file_digest(paths[k]) != prov["inputs"][k]["sha256"]:
                    raise ValueError(
                        f"source {k} unavailable or checksum mismatch; use --inputs for relocation"
                    )
            table = read_triplet_table(paths["matches"])
            records = {m.observation_id: m for m in table.matches}
            parts = [read_msa(paths[k], k, prov["components"][k]["molecule_type"]) for k in "ABC"]
            synced = synchronize(*parts, [records[key] for key in ids])
            masks = []
            for c, m in zip(parts, store.maps, strict=True):
                mask = np.zeros(c.n_sites, dtype=bool)
                mask[list(m.original_columns)] = True
                masks.append(mask)
            msa = apply_triplet_masks(synced, *masks).msa
            path = root / summary["outputs"]["weights"]
            if file_digest(path) != prov["weight_table_sha256"]:
                raise ValueError("weight table disagrees with the score archive")
            rows = list(read_rows(path))
            if tuple(r["observation_id"] for r in rows) != ids:
                raise ValueError("weight row order disagrees with score archive")
            w = np.array([float(r["weight"]) for r in rows])
            design = prov["weighting"]
            clades = tuple(r["clade"] for r in rows) if design["policy"] == "clade_balanced" else ()
            weights = SequenceWeightResult(
                w,
                None,
                float(w.sum()),
                len(ids),
                tuple("ABC"),
                prov["settings"]["identity_threshold"],
                prov["settings"]["min_overlap"],
                ids,
                design["policy"],
                design.get("component"),
                clades=clades,
            )
        else:
            msa, weights, table = live
            if tuple(m.observation_id for m in msa.matches) != ids:
                raise ValueError("live and archived observations disagree")
        meta = {m.observation_id: r for m, r in zip(table.matches, table.annotations, strict=True)}
        return cls(root, summary, store, msa, weights, {key: meta[key] for key in ids})

    def triples(self):
        advanced = self.summary["advanced"]
        if not advanced.get("candidate_manifest"):
            raise ValueError("triple calibration requires a native fit with a candidate manifest")
        manifest = read_json(self.artifact(advanced["candidate_manifest"]))
        if manifest["scope"] not in {
            "coordinate_only",
            "user_supplied_file",
            "user_declared_external_prior",
        }:
            raise ValueError(
                "fixed-candidate calibration requires selection independent of evaluation rows; adaptive selection needs held-out testing or selection-aware refits"
            )
        path = self.artifact(manifest["coordinate_table"])
        if file_digest(path) != manifest["coordinate_table_sha256"]:
            raise ValueError("candidate coordinate checksum mismatch")
        index = [{v: i for i, v in enumerate(m.original_positions)} for m in self.store.maps]
        triples = []
        for row in read_rows(path):
            t = tuple(
                idx[int(row[f"{role}_original_col"])]
                for idx, role in zip(index, "abc", strict=True)
            )
            if row["candidate_id"] != candidate_id(self.store.maps, t):
                raise ValueError("candidate identity disagrees with score axes")
            triples.append(t)
        if not triples or len(triples) != manifest["selected"] or len(set(triples)) != len(triples):
            raise ValueError("calibration requires distinct, nonempty candidate coordinates")
        return triples, manifest

    def local_settings(self, triples):
        info = self.summary["advanced"]["local_fit"]
        data = read_json(self.artifact(info["frozen_support"]))
        zeros = data.pop("structural_zeros")
        coords = data.pop("coordinates")
        for m in self.store.maps:
            if coords[m.component]["original_columns"] != list(m.original_positions):
                raise ValueError("local and pair archive coordinates disagree")
        support = FrozenSupport.from_metadata(data)
        lookup = {
            tuple(m.original_positions[i] for m, i in zip(self.store.maps, t)): t for t in triples
        }
        masks = {}
        for row in zeros:
            t = lookup[tuple(row["original_columns"])]
            if t not in masks:
                masks[t] = np.ones(support.shape_for(t), dtype=bool)
            masks[t][tuple(row["bins"])] = False
        return support, MaxEntOptions(**info["settings"][info["ranked_setting"]]), masks
