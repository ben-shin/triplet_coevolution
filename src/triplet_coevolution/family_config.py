"""Read the three-folder Stockholm project or a portable explicit-path configuration."""

import csv
import json
from pathlib import Path

from triplet_coevolution.config import COMPONENTS, _keys, _typed, _unique
from triplet_coevolution.parameters import analysis_types

STOCKHOLM_SUFFIXES = (".sto", ".sth", ".stk", ".stockholm", ".seed", ".full")
COMPONENT_FIELDS = {
    "alignment",
    "name",
    "molecule",
    "reference_id",
    "reference_row_id",
    "reference_start",
    "reference_strand",
    "family",
    "targets",
    "scope",
}
PROJECT_FIELDS = {
    "components",
    "schema_version",
    "manifest",
    "cache",
    "nucleotide_cache",
    "uniprot",
    "expected_orthogroups",
    "analysis",
    "orthology_scope",
    "gff3",
    "providers",
    "analysis_limits",
}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique)


def _strings(value, field):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} requires a nonempty list")
    for item in value:
        if not isinstance(item, str) or not item or item != item.strip():
            raise ValueError(f"{field} requires nonempty strings without surrounding whitespace")
    return list(dict.fromkeys(value))


def _folder_components(directory, cfg, inputs, automatic):
    if not directory.is_dir():
        raise ValueError(f"input directory not found: {directory}")
    folders = sorted(
        (p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=lambda p: p.name,
    )
    if len(folders) != 3:
        raise ValueError(
            f"expected exactly three component folders under {directory}; found {len(folders)}"
        )
    order = cfg.pop("components", None)
    if order is None:
        order = dict(zip(COMPONENTS, [p.name for p in folders], strict=True))
    _keys(order, COMPONENTS, COMPONENTS, "components")
    if not all(isinstance(v, str) for v in order.values()) or set(order.values()) != {
        p.name for p in folders
    }:
        raise ValueError("project.json components must assign each of the three folder names once")
    components = {}
    for name in COMPONENTS:
        folder = directory / order[name]
        settings = folder / "component.json"
        comp = {}
        if settings.exists():
            comp = _json(settings)
            _keys(comp, (), COMPONENT_FIELDS - {"name"}, f"{name}.component.json")
            inputs[f"{name}.config"] = settings
        if "alignment" not in comp:
            candidates = sorted(
                p
                for p in folder.iterdir()
                if p.is_file() and p.name.lower().removesuffix(".gz").endswith(STOCKHOLM_SUFFIXES)
            )
            if automatic and len(candidates) > 1:
                for i, option in enumerate(candidates):
                    inputs[f"{name}.candidate.{i}"] = option
            elif len(candidates) != 1:
                raise ValueError(
                    f"{folder}: found {len(candidates)} Stockholm files; "
                    "set alignment in component.json"
                )
            comp["alignment"] = candidates[0].name
        if "targets" not in comp and "scope" not in comp and (folder / "targets.txt").is_file():
            comp["targets"] = "targets.txt"
        components[name] = (comp | {"name": folder.name}, folder)
    return components, order


def family_config(config, directory, *, uniprot=None, default_molecule=None, automatic=False):
    if config is not None and directory is not None:
        raise ValueError("choose either --input or --config")
    inputs = {}
    if config is None:
        base = (directory or Path("input")).expanduser().resolve()
        project = base / "project.json"
        cfg = {}
        if project.is_file():
            inputs["project"] = project
            cfg = _json(project)
        _keys(cfg, (), PROJECT_FIELDS, "project")
        components, order = _folder_components(base, cfg, inputs, automatic)
        for key, filename in (
            ("manifest", "mappings.tsv"),
            ("cache", "uniprot.sqlite"),
            ("nucleotide_cache", "nucleotide.sqlite"),
        ):
            if (base / filename).is_file():
                cfg.setdefault(key, filename)
    else:
        config = config.expanduser().resolve()
        base = config.parent
        inputs["project"] = config
        cfg = _json(config)
        _keys(cfg, ("components",), PROJECT_FIELDS, "configuration")
        _keys(cfg["components"], COMPONENTS, COMPONENTS, "components")
        components = {k: (cfg["components"][k], base) for k in COMPONENTS}
        order = None
    if type(cfg.setdefault("schema_version", 1)) is not int or cfg["schema_version"] != 1:
        raise ValueError("expected schema_version 1")
    for key in ("manifest", "cache", "nucleotide_cache"):
        if key in cfg:
            inputs[key] = cfg[key] = _typed(cfg[key], Path, key, base)
    from triplet_coevolution.evidence.gff3 import parse_specs

    cfg["gff3"] = parse_specs(cfg.get("gff3"), base)
    for i, source in enumerate(cfg["gff3"]):
        inputs[f"gff3.{i}"] = source["path"]
        if "system_windows" in source:
            inputs[f"gff3.{i}.system_windows"] = source["system_windows"]
    from dataclasses import asdict

    from triplet_coevolution.evidence.provider_policy import ProviderPolicy

    cfg["providers"] = asdict(ProviderPolicy.parse(cfg.get("providers")))
    configured_online = _typed(cfg.get("uniprot", automatic), bool, "uniprot", base)
    cfg["uniprot"] = configured_online if uniprot is None else uniprot
    if default_molecule is not None and default_molecule not in ("protein", "dna", "rna"):
        raise ValueError("--default-molecule must be protein, dna or rna")
    cfg["components"] = {}
    for name, (comp, folder) in components.items():
        _keys(comp, ("alignment",), COMPONENT_FIELDS, name)
        comp.setdefault("name", name)
        for key in ("name", "family", "reference_id", "reference_row_id"):
            if key in comp:
                _typed(comp[key], str, f"{name}.{key}", folder)
        if "reference_row_id" in comp and "reference_id" not in comp:
            raise ValueError(f"{name}.reference_row_id requires reference_id")
        if "reference_strand" in comp and (
            "reference_id" not in comp or comp["reference_strand"] not in ("+", "-")
        ):
            raise ValueError(f"{name}.reference_strand requires a reference and + or -")
        if "reference_start" in comp and (
            "reference_id" not in comp
            or type(comp["reference_start"]) is not int
            or comp["reference_start"] < 1
        ):
            raise ValueError(f"{name}.reference_start requires a reference and a positive integer")
        molecule = comp.setdefault("molecule", default_molecule)
        if molecule is not None and molecule not in ("protein", "dna", "rna"):
            raise ValueError(f"{name}.molecule must be protein, dna or rna")
        inputs[f"{name}.alignment"] = comp["alignment"] = _typed(
            comp["alignment"], Path, f"{name}.alignment", folder
        )
        if "targets" in comp:
            if "scope" in comp:
                raise ValueError(f"{name}: choose targets or scope")
            targets = comp["targets"]
            if isinstance(targets, str):
                path = _typed(targets, Path, f"{name}.targets", folder)
                inputs[f"{name}.targets"] = path
                targets = [
                    s.strip()
                    for s in path.read_text(encoding="utf-8-sig").splitlines()
                    if s.strip() and not s.lstrip().startswith("#")
                ]
            comp["targets"] = _strings(targets, f"{name}.targets")
            comp["scope"] = "targets"
        else:
            scope = comp.setdefault("scope", "manifest" if "manifest" in cfg else "family")
            if scope not in ("manifest", "family") or (
                scope == "manifest" and "manifest" not in cfg
            ):
                raise ValueError(f"{name}.scope must be family, or manifest with a mappings file")
        cfg["components"][name] = comp
    if len({comp["name"] for comp in cfg["components"].values()}) != 3:
        raise ValueError("component names must be unique")
    if (
        not automatic
        and cfg["uniprot"]
        and not any(c["molecule"] == "protein" for c in cfg["components"].values())
    ):
        raise ValueError(
            "UniProt requires protein components; set molecule in component.json "
            "or use --default-molecule protein for an all-protein project"
        )
    expected = cfg.setdefault("expected_orthogroups", {})
    if "orthology_scope" in cfg:
        _typed(cfg["orthology_scope"], str, "orthology_scope", base)
    _keys(expected, (), COMPONENTS, "expected_orthogroups")
    for key in expected:
        expected[key] = _strings(expected[key], f"expected_orthogroups.{key}")
    analysis = cfg.setdefault("analysis", {})
    limits = cfg.setdefault("analysis_limits", {})
    _keys(limits, (), ("pair_cells", "weight_cells"), "analysis_limits")
    if any(type(v) is not int or v < 1 for v in limits.values()):
        raise ValueError("analysis_limits entries must be positive integers")
    _keys(analysis, (), analysis_types(), "analysis")
    for key, value in analysis.items():
        analysis[key] = _typed(value, analysis_types()[key], f"analysis.{key}", base)
        if isinstance(analysis[key], Path):
            inputs[f"analysis.{key}"] = analysis[key]
    return cfg, inputs, order


def manifest_targets(path):
    """Obtain candidate identifiers without requiring a pre-paired triplet table."""
    result = {name: {} for name in COMPONENTS}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        fields = [field.strip() for field in reader.fieldnames or []]
        if len(set(fields)) != len(fields) or not {"component", "identifier"}.issubset(fields):
            raise ValueError("mapping manifest requires component and identifier columns")
        reader.fieldnames = fields
        for row in reader:
            if row["component"] not in result or not row["identifier"]:
                raise ValueError(f"invalid mapping manifest at line {reader.line_num}")
            result[row["component"]][row["identifier"]] = None
    return result
