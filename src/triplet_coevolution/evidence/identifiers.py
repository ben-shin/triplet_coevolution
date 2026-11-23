"""Retain aligned row identity separately from identifiers used for annotation."""

import re
from dataclasses import dataclass

_AC = re.compile(
    r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[1-9][0-9]*)?"
)
_ENTRY = re.compile(r"[A-Z0-9]{1,16}_[A-Z0-9]{1,5}")
_DOMAIN = re.compile(r"(.+)/(\d+)-(\d+)$")


@dataclass(frozen=True)
class EvidenceRow:
    row_id: str
    accession: str | None = None
    start: int | None = None
    end: int | None = None
    present: bool = True
    selectors: tuple[str, ...] = ()


def accession(token: str) -> bool:
    return _AC.fullmatch(token) is not None


def lookup_id(token: str) -> str:
    """An exact accession, isoform accession or mnemonic suitable for a REST query."""
    token = token.upper()
    # Stockholm AC annotations sometimes carry a sequence version after a period.
    token = re.sub(r"\.\d+$", "", token)
    return token if accession(token) or _ENTRY.fullmatch(token) else ""


def identifiers(
    row: EvidenceRow,
) -> tuple[tuple[str, ...], tuple[str, ...], int | None, int | None]:
    aliases = {}
    queries = set()
    start, end = row.start, row.end
    for i, identifier in enumerate([row.row_id, row.accession, *getattr(row, "selectors", ())]):
        if not identifier:
            continue
        aliases[identifier] = None
        match = _DOMAIN.fullmatch(identifier)
        if match:
            identifier, first, last = match.groups()
            if i == 0:
                if start is not None and (start, end) != (int(first), int(last)):
                    raise ValueError(f"conflicting domain coordinates on {row.row_id!r}")
                start, end = int(first), int(last)
            aliases[identifier] = None
        parts = identifier.split("|")
        if len(parts) == 3 and parts[0].lower() in ("sp", "tr"):
            aliases.update(dict.fromkeys(x for x in parts[1:] if x))
            # The stable accession takes precedence over a possibly renamed mnemonic.
            identifier = parts[1]
        token = lookup_id(identifier)
        if token:
            queries.add(token)
            aliases[token] = None
    if (start is None) != (end is None) or (start is not None and min(start, end) < 1):
        raise ValueError(f"invalid domain coordinates on {row.row_id!r}")
    return tuple(aliases), tuple(sorted(queries)), start, end
