"""Quality masks and synchronized observation support in original alignment coordinates."""

import csv

import numpy as np

from triplet_coevolution.msa.masking import column_mask


class MissingRowSupport(ValueError):
    """At least one row or component has no canonical residue among retained sites."""


def site_masks(path, components):
    """Map a prespecified panel to the unchanged input alignment coordinate systems."""
    masks = {c.name: np.zeros(c.n_sites, dtype=bool) for c in components}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", strict=True)
        if reader.fieldnames != ["component", "original_column"]:
            raise ValueError("site panel requires a TSV header: component, original_column")
        for row in reader:
            name, col = row["component"], row["original_column"]
            if (
                None in row
                or name not in masks
                or not col
                or not col.isascii()
                or not col.isdigit()
            ):
                raise ValueError(f"invalid site panel record on line {reader.line_num}")
            index = int(col) - 1
            if not 0 <= index < len(masks[name]) or masks[name][index]:
                raise ValueError(f"duplicate or out-of-range site panel column {name} {col}")
            masks[name][index] = True
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("site panel must include at least one original column per component")
    return tuple(masks[c.name] for c in components)


def retained_sites(qc, options, panel):
    masks = tuple(column_mask(part, **options) for part in qc.components)
    if panel is None:
        return masks
    return tuple(usable & selected for usable, selected in zip(masks, panel, strict=True))
