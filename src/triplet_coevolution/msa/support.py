"""Quality masks and synchronized observation support in original alignment coordinates."""

import csv

import numpy as np

from triplet_coevolution.msa.masking import column_mask


class MissingRowSupport(ValueError):
    """At least one row or component has no canonical residue among retained sites."""


def site_masks(path, components, maps=None):
    """Resolve an original-column panel or exact reference ID and residue-position panel."""
    masks = {c.name: np.zeros(c.n_sites, dtype=bool) for c in components}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", strict=True)
        fields = reader.fieldnames
        by_ref = fields == ["component", "reference_id", "reference_position"]
        if not by_ref and fields != ["component", "original_column"]:
            raise ValueError(
                "site panel requires component,original_column or "
                "component,reference_id,reference_position TSV columns"
            )
        if maps is None and by_ref:
            raise ValueError("reference-coordinate site panel requires all reference maps")
        if maps is not None:
            if len(maps) != len(components) or any(
                m.component != c.name or m.n_sites != c.n_sites for m, c in zip(maps, components)
            ):
                raise ValueError("site panel reference maps must match input components")
            originals = {
                m.component: {n + 1: i for i, n in enumerate(m.original_columns)} for m in maps
            }
            lookups = {
                m.component: {
                    (m.reference_id, pos): i
                    for i, pos in enumerate(m.reference_positions)
                    if pos is not None
                }
                for m in maps
            }
        for row in reader:
            name = row["component"]
            col = row["reference_position" if by_ref else "original_column"]
            if (
                None in row
                or name not in masks
                or not col
                or not col.isascii()
                or not col.isdigit()
            ):
                raise ValueError(f"invalid site panel record on line {reader.line_num}")
            index = (
                lookups[name].get((row["reference_id"], int(col)), -1)
                if by_ref
                else (int(col) - 1 if maps is None else originals[name].get(int(col), -1))
            )
            if not 0 <= index < len(masks[name]) or masks[name][index]:
                raise ValueError(f"duplicate or out-of-range site panel coordinate {name} {col}")
            masks[name][index] = True
    if any(not keep.any() for keep in masks.values()):
        raise ValueError("site panel must include at least one column per component")
    return tuple(masks[c.name] for c in components)


def retained_sites(qc, options, panel):
    masks = tuple(column_mask(part, **options) for part in qc.components)
    if panel is None:
        return masks
    return tuple(usable & selected for usable, selected in zip(masks, panel, strict=True))
