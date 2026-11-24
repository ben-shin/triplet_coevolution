"""Verify and copy native structural data before linking its physical coordinates to scores."""

import csv
import shutil
from pathlib import Path, PureWindowsPath

from triplet_coevolution.io.score_store import file_digest
from triplet_coevolution.operations import put_json, read_json
from triplet_coevolution.structure.mapping import comparison_spelling


def local_artifact(root, name):
    if not isinstance(name, str) or ":" in name:
        raise ValueError(f"invalid artifact path {name!r}")
    rel = Path(name)
    if (
        not name
        or rel.is_absolute()
        or PureWindowsPath(name).drive
        or "\\" in name
        or ".." in rel.parts
        or rel.as_posix() != name
    ):
        raise ValueError(f"invalid artifact path {name!r}")
    p = (root / rel).resolve(strict=True)
    if not p.is_relative_to(root.resolve(strict=True)) or not p.is_file():
        raise ValueError(f"artifact outside source directory: {name!r}")
    return p


def add_structure(directory, source, store, score_digest):
    root = Path(source).resolve(strict=True)
    if root.is_file():
        if root.name != "structure_summary.json":
            raise ValueError("structure requires a native structure_summary.json or its directory")
        root = root.parent
    manifest = local_artifact(root, "structure_summary.json")
    digest = file_digest(manifest)
    meta = read_json(manifest)
    hashes = meta.get("output_sha256")
    if not isinstance(hashes, dict) or not {"mapping.tsv", "distances.tsv"} <= hashes.keys():
        raise ValueError("structure has no verified mapping and distance tables")
    bound = meta.get("inputs", {}).get("scores", {}).get("sha256")
    if bound and bound != score_digest:
        raise ValueError("structure was fitted with a different score manifest")
    expected = {}
    for m in store.maps:
        role = m.component
        ref = meta.get("inputs", {}).get("references", {}).get(role, {})
        mol = store.provenance.get("components", {}).get(role, {}).get("molecule_type")
        if (
            not mol
            or ref.get("molecule") != mol
            or ref.get("strand") != m.strand
            or ref.get("reference_id") != m.reference_id
        ):
            raise ValueError(f"structure reference identity, strand or molecule differs for {role}")
        for pos, symbol in zip(m.reference_positions, m.symbols, strict=True):
            if pos is not None:
                expected[role, m.reference_id, pos] = (
                    mol,
                    comparison_spelling(symbol.upper(), mol),
                )
    dest = directory / "structure"
    if dest.exists():
        raise ValueError(
            "report already embeds a structural bundle; regenerate from the original native run to choose another"
        )
    dest.mkdir()
    copied = {}
    viewer_files = {
        "viewer/view.pml",
        "viewer/view_chimerax.py",
        *(f"viewer/{role}.cif" for role in "ABC"),
    }
    for name, sha in hashes.items():
        # Only verified native data and viewer scripts are bundled. The explorer renders
        # the measurements itself; externally provided pages never execute inside it.
        if name not in viewer_files and Path(name).suffix.lower() not in (".tsv", ".csv", ".json"):
            continue
        src = local_artifact(root, name)
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        copied[name] = sha
        if file_digest(target) != sha:
            raise ValueError(f"structural artifact checksum mismatch: {name}")
    with (dest / "mapping.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = row["component"], row["reference_id"], int(row["reference_pos"])
            value = expected.pop(key, None)
            if value and comparison_spelling(row["reference_symbol"].upper(), value[0]) != value[1]:
                raise ValueError(f"structural reference residue differs at {key}")
    if expected:
        raise ValueError("structural mapping omits reference positions in the stored score panel")
    source_manifest = dest / "source_structure_summary.json"
    shutil.copyfile(manifest, source_manifest)
    if file_digest(source_manifest) != digest:
        raise ValueError("structural manifest changed during report rendering")
    meta["output_sha256"] = copied
    meta["bundle"] = {
        "source_manifest": source_manifest.name,
        "source_manifest_sha256": digest,
        "omitted_files": sorted(set(hashes) - copied.keys()),
    }
    put_json(dest / manifest.name, meta)
    return "structure/structure_summary.json"
