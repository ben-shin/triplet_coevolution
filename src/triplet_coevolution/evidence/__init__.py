"""Generate and audit candidate triplets from alignment identifiers and genomic evidence."""

from .generate import generate_evidence
from .identifiers import EvidenceRow

__all__ = ["EvidenceRow", "generate_evidence"]
