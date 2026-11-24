"""Load one local structure model and an explicit biological assembly or ASU."""

import gzip
import hashlib
import io
from pathlib import Path

from triplet_coevolution.structure._coordinates import transform_copy
from triplet_coevolution.structure._mmcif import parse_cif
from triplet_coevolution.structure.types import (
    IDENTITY,
    ChoiceRequired,
    Structure,
    resolve_selection,
)

__all__ = ["ChoiceRequired", "read_structure"]


def read_structure(
    path: str | Path,
    *,
    assembly: str | None = None,
    model: int | None = None,
) -> Structure:
    """Materialize selected polymer copies. Model numbers are the deposited model IDs."""
    if assembly is not None and (not isinstance(assembly, str) or not assembly.strip()):
        raise ValueError("choose a biological assembly ID or explicitly select 'asu'")
    if model is not None and (not isinstance(model, int) or isinstance(model, bool) or model < 1):
        raise ValueError("model must be a positive deposited model number")
    if assembly is not None:
        assembly = assembly.strip()
        if assembly.lower() == "asu":
            assembly = "asu"
    path = Path(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if payload.startswith(b"\x1f\x8b"):
        payload = gzip.decompress(payload)
    stream = io.StringIO(payload.decode("utf-8-sig"))
    line = next((s.strip() for s in stream if s.strip() and not s.lstrip().startswith("#")), "")
    stream.seek(0)
    if line.lower().startswith("data_"):
        format_name, parse = "mmCIF", parse_cif
    else:
        from triplet_coevolution.structure._pdb import parse_pdb

        format_name, parse = "PDB", parse_pdb
    try:
        entry, method, chains, models, assemblies, copies, warnings = parse(stream, assembly, model)
    except KeyError as exc:
        raise ValueError(f"missing required {format_name} data item {exc.args[0]}") from exc
    assembly, model = resolve_selection(assembly, model, models, assemblies)
    if assembly == "asu":
        copies = [(label, (), IDENTITY) for label in chains]
    seen = {}
    locations = {}
    result = []
    for label, operations, matrix in copies:
        if label not in chains:
            continue
        if (label, operations) in seen:
            if seen[label, operations] != matrix:
                raise ValueError(f"conflicting transforms for assembly copy {label}/{operations}")
            continue
        seen[label, operations] = matrix
        if (label, matrix) in locations:
            warnings.append(
                f"Chain {label}: operation path {operations} duplicates "
                f"{locations[label, matrix]} at exactly the same position and was omitted."
            )
            continue
        locations[label, matrix] = operations
        result.append(transform_copy(chains[label], operations, matrix))
    if not result:
        raise ValueError(f"no supported polymer chains in {format_name} assembly {assembly!r}")
    return Structure(
        entry,
        format_name,
        model,
        tuple(sorted(models)),
        assembly,
        tuple(sorted(assemblies)),
        tuple(result),
        digest,
        method,
        tuple(warnings),
    )
