"""Read explicit component membership in genome or orthology evidence groups."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    component: str
    sequence_id: str
    priority: float | None


def read_evidence(path: Path):
    delimiter = {".csv": ",", ".tsv": "\t"}.get(path.suffix.lower())
    if delimiter is None:
        raise ValueError("evidence must be a .csv or .tsv file")
    groups = {}
    metadata = {}
    seen = set()
    required = {"group_id", "component", "sequence_id"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter, strict=True)
        fields = [name.strip() for name in reader.fieldnames or []]
        if not all(fields) or len(set(fields)) != len(fields) or not required <= set(fields):
            raise ValueError("evidence requires unique columns group_id, component, sequence_id")
        reader.fieldnames = fields
        extras = tuple(name for name in fields if name not in required | {"priority"})
        if any(
            not name.strip() or name in {"observation_id", "a_id", "b_id", "c_id"}
            for name in extras
        ):
            raise ValueError("invalid or reserved evidence annotation header")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"malformed evidence at line {reader.line_num}")
            group, component, seq_id = (row[k] for k in ("group_id", "component", "sequence_id"))
            if (
                not group.strip()
                or group != group.strip()
                or component not in ("A", "B", "C")
                or not seq_id
                or any(c.isspace() for c in seq_id)
            ):
                raise ValueError(
                    f"invalid group, component, or sequence ID at line {reader.line_num}"
                )
            if (component, seq_id) in seen:
                raise ValueError(
                    f"reused evidence sequence {component}/{seq_id} at line {reader.line_num}"
                )
            seen.add((component, seq_id))
            priority = float(row["priority"]) if row.get("priority", "").strip() else None
            if priority is not None and not math.isfinite(priority):
                raise ValueError(f"nonfinite evidence priority at line {reader.line_num}")
            group_meta = {key: row[key] for key in extras}
            if group_meta != metadata.setdefault(group, group_meta):
                raise ValueError(f"group {group!r} has conflicting annotations")
            groups.setdefault(group, []).append(Candidate(component, seq_id, priority))
    if not groups:
        raise ValueError("evidence contains no candidate rows")
    return groups, metadata, extras
