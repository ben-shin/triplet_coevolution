"""Materialize matched records and coordinate references after inspecting source inventories."""

from contextlib import ExitStack
from dataclasses import dataclass


class InputBudgetError(ValueError):
    def __init__(self, details, limit):
        self.details = details
        cells = sum(v["selected_symbols"] for v in details.values())
        super().__init__(
            f"The requested matched/reference rows contain {cells} sequence symbols, exceeding the {limit}-byte encoding budget."
        )


@dataclass(frozen=True)
class SelectedSources:
    components: tuple
    maps: tuple
    statistics: dict


def matched_sources(
    paths,
    matches,
    molecules,
    references,
    *,
    row_references=None,
    starts=None,
    strands=None,
    formats=None,
    byte_limit=None,
):
    from triplet_coevolution.io.alignments import open_msa

    with ExitStack() as stack:
        inventories = {}
        wanted = {}
        statistics = {}
        for role in "ABC":
            inv = stack.enter_context(
                open_msa(paths[role], role, molecules[role], (formats or {}).get(role))
            )
            inventories[role] = inv
            row = (row_references or {}).get(role) or references[role]
            ids = tuple(dict.fromkeys([*(getattr(m, role.lower() + "_id") for m in matches), row]))
            if not inv.contains(row):
                raise ValueError(f"{role}: reference ID {row!r} not found in alignment")
            for key in ids:
                if not inv.contains(key):
                    raise ValueError(f"{role}: matched sequence ID {key!r} not found in alignment")
            wanted[role] = ids
            statistics[role] = {
                "format": inv.format,
                "source_rows": inv.n_rows,
                "selected_rows": len(ids),
                "alignment_width": inv.n_columns,
                "source_symbols": inv.raw_symbols,
                "selected_match_cells": len(ids) * inv.n_columns,
                "selected_symbols": inv.selected_raw_symbols(ids),
            }
        if (
            byte_limit is not None
            and sum(v["selected_symbols"] for v in statistics.values()) > byte_limit
        ):
            raise InputBudgetError(statistics, byte_limit)
        components = []
        maps = []
        for role, inv in inventories.items():
            result = inv.select(wanted[role])
            components.append(result.msa)
            maps.append(
                result.reference_map(
                    references[role],
                    (starts or {}).get(role, 1),
                    row_id=(row_references or {}).get(role),
                    strand=(strands or {}).get(role, "+"),
                )
            )
            statistics[role]["selection"] = result.provenance
    return SelectedSources(tuple(components), tuple(maps), statistics)
