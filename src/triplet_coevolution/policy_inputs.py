"""Bind the files consumed by native analysis policies, using their actual typed parsers."""

from pathlib import Path

from triplet_coevolution.operations import file_id

# These policies contain inline data, including observation IDs and source descriptions.
# The remaining three analysis inputs below are physical tabular data, regardless of suffix.
SELF_CONTAINED = {
    "candidate_policy",
    "local_policy",
    "qc_policy",
    "dca_policy",
    "site_panel",
    "triplet_candidates",
    "contacts",
}


def dependencies(kind, source):
    """Hash the policy itself and every input its schema reads. Checkpoint paths are outputs."""
    source = Path(source).resolve()
    original = file_id(source)
    if kind in SELF_CONTAINED:
        return {str(source): original}
    files = {source}

    def add_tree(tree):
        if tree:
            files.add(tree.newick)
            if tree.tip_map:
                files.add(tree.tip_map)

    def add_clades(design):
        if design.split_manifest:
            files.add(design.split_manifest)
        add_tree(design.clade_tree)

    if kind == "null_policy":
        from triplet_coevolution.validation.null_policy import NullPolicy

        design = NullPolicy.from_config(source)
        if design.tree:
            for scenario in design.tree.scenarios:
                for _, tree in scenario.sources:
                    add_tree(tree)
                if isinstance(scenario.frequencies, Path):
                    files.add(scenario.frequencies)
    elif kind == "stability_policy":
        from triplet_coevolution.clade_workflow import CladePolicy

        add_clades(CladePolicy.from_config(source))
    elif kind == "conditional_policy":
        from triplet_coevolution.conditional_workflow import conditional_design, read_policy

        raw, base = read_policy(source)
        if isinstance(raw.get("clades"), str):
            files.add(base / raw["clades"])
        design, _, opts = conditional_design(source)
        add_clades(design)
        if "panel_file" in opts:
            files.add(opts["panel_file"])
    else:
        raise ValueError(f"no input-dependency schema for {kind!r}")

    result = {str(p): file_id(p) for p in sorted({f.resolve() for f in files})}
    if result[str(source)] != original:
        raise ValueError(f"{kind} changed during dependency discovery: {source}")
    return result
