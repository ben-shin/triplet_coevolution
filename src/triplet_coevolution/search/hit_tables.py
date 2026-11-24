"""Declared tool-table coordinate conventions and explicit rejected records."""

import gzip
import math
from dataclasses import dataclass
from pathlib import Path

from triplet_coevolution.search.shared import Hit

FORMATS = ("hmmsearch_domtblout", "nhmmer_tblout", "cmsearch_tblout", "blast6", "psiblast7")


@dataclass(frozen=True)
class HitRecord:
    line: int
    text: str
    hit: Hit | None
    reason: str | None


def _positive(v):
    n = int(v)
    if n < 1:
        raise ValueError("coordinates and lengths must be positive integers")
    return n


def _number(v, positive=False):
    n = float(v)
    if not math.isfinite(n) or (positive and n < 0):
        raise ValueError("invalid score or E-value")
    return n


def _accession(v):
    return None if v == "-" else v


def _hit(fmt, r):
    env = None
    pct = aligned = mismatches = gaps = None
    tl = ml = seqe = ce = di = dc = acc = qacc = None
    qs = "+"
    if fmt == "hmmsearch_domtblout":
        if len(r) < 22:
            raise ValueError("HMMER domtblout requires at least 22 fields")
        target, query = r[0], r[3]
        acc, qacc = _accession(r[1]), _accession(r[4])
        tl, ml = _positive(r[2]), _positive(r[5])
        seqe, ce, evalue = (_number(r[i], True) for i in (6, 11, 12))
        score = _number(r[13])
        di, dc = _positive(r[9]), _positive(r[10])
        if di > dc:
            raise ValueError("domain index exceeds domain count")
        m, n, a, b, x, y = (_positive(r[i]) for i in range(15, 21))
        strand = "+"
        env = (x, y)
    elif fmt == "nhmmer_tblout":
        if len(r) < 15:
            raise ValueError("nhmmer tblout requires at least 15 fields")
        target, query = r[0], r[2]
        acc, qacc = _accession(r[1]), _accession(r[3])
        m, n, a, b, x, y = (_positive(r[i]) for i in range(4, 10))
        tl = _positive(r[10])
        strand = r[11]
        evalue, score = _number(r[12], True), _number(r[13])
        env = (min(x, y), max(x, y))
    elif fmt == "cmsearch_tblout":
        if len(r) < 17 or r[4] not in ("cm", "hmm"):
            raise ValueError("cmsearch requires Infernal format 1")
        target, query = r[0], r[2]
        acc, qacc = _accession(r[1]), _accession(r[3])
        m, n, a, b = (_positive(r[i]) for i in (5, 6, 7, 8))
        strand = r[9]
        score, evalue = _number(r[14]), _number(r[15], True)
    elif fmt == "blast6":
        if len(r) not in (12, 14):
            raise ValueError("BLAST6 requires standard 12 fields with optional trailing qlen slen")
        query, target = r[:2]
        identity = _number(r[2])
        aligned = _positive(r[3])
        mismatches, gaps = int(r[4]), int(r[5])
        if not 0 <= identity <= 100 or not (0 <= mismatches <= aligned and 0 <= gaps <= aligned):
            raise ValueError("invalid BLAST identity/alignment length")
        m, n, a, b = (_positive(v) for v in r[6:10])
        if max(abs(m - n), abs(a - b)) + 1 > aligned:
            raise ValueError(
                "translated or inconsistent BLAST interval for this same-alphabet search"
            )
        pct = identity / 100
        evalue, score = _number(r[10], True), _number(r[11])
        qs = "-" if m > n else "+"
        strand = "-" if a > b else "+"
        if len(r) == 14:
            ml, tl = _positive(r[12]), _positive(r[13])
    else:
        if len(r) != 7:
            raise ValueError("native PSI-BLAST output requires seven fields")
        target, query = r[0].removeprefix("lcl|"), "seed"
        a, b, score, evalue, m, n = (
            _positive(r[1]),
            _positive(r[2]),
            _number(r[3]),
            _number(r[4], True),
            _positive(r[5]),
            _positive(r[6]),
        )
        strand = "+"
    if (
        not target
        or not query
        or strand not in ("+", "-")
        or (strand == "+" and a > b)
        or (strand == "-" and a < b)
    ):
        raise ValueError("invalid target strand or interval")
    if (
        (qs == "+" and m > n)
        or (ml is not None and max(m, n) > ml)
        or (tl is not None and max(a, b) > tl)
    ):
        raise ValueError("alignment outside model/target coordinates")
    if env and (env[0] > min(a, b) or env[1] < max(a, b) or (tl is not None and env[1] > tl)):
        raise ValueError("invalid enclosing target envelope")
    relative = "+" if qs == strand else "-"
    return Hit(
        target,
        min(a, b),
        max(a, b),
        relative,
        score,
        evalue,
        model_strand=relative,
        model_first=min(m, n),
        model_last=max(m, n),
        query_id=query,
        target_length=tl,
        model_length=ml,
        sequence_evalue=seqe,
        conditional_evalue=ce,
        envelope_first=env[0] if env else None,
        envelope_last=env[1] if env else None,
        source_strand=strand,
        query_strand=qs,
        domain_index=di,
        domain_count=dc,
        source_accession=acc,
        query_accession=qacc,
        reported_identity=pct,
        aligned_length=aligned,
        mismatches=mismatches,
        gap_openings=gaps,
    )


def read_hit_table(path, fmt, *, max_rows=1_000_000):
    """Yield every data line with a typed positive-coordinate hit or an explicit rejection."""
    if fmt not in FORMATS:
        raise ValueError(f"unsupported search table {fmt}")
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as f:
        count = 0
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if text.lower().startswith("# program:"):
                name = text.split(":", 1)[1].strip().lower()
                allowed = {
                    "hmmsearch_domtblout": {"hmmsearch", "jackhmmer"},
                    "nhmmer_tblout": {"nhmmer"},
                    "cmsearch_tblout": {"cmsearch"},
                }
                if fmt in allowed and name not in allowed[fmt]:
                    raise ValueError(
                        "table program header conflicts with declared search direction/format"
                    )
            if fmt == "blast6" and text.upper().startswith(
                ("# BLASTX ", "# TBLASTN ", "# TBLASTX ")
            ):
                raise ValueError("translated BLAST tables require a separate coding-coordinate map")
            if not text or text.startswith("#") or text == "Search has CONVERGED!":
                continue
            count += 1
            if count > max_rows:
                raise ValueError(
                    "hit table exceeds max_candidates; increase the declared bound or tighten the search"
                )
            try:
                hit = _hit(fmt, text.split())
            except (ValueError, IndexError) as exc:
                yield HitRecord(line_no, text, None, str(exc))
            else:
                yield HitRecord(line_no, text, hit, None)
