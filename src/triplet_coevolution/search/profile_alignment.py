"""Retain profile/seed Stockholm columns while binding opaque tool names to source loci."""

import subprocess
from dataclasses import replace

from triplet_coevolution.core.alignment import ComponentMSA
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.io.profiles import check_profile
from triplet_coevolution.msa.builders import Placement, oriented
from triplet_coevolution.search.local_tools import checksum, copy_model, run_tool


def read_stockholm(path, names):
    parts = {k: [] for k in names}
    rf = []
    first = end = False
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s == "# STOCKHOLM 1.0":
                if first:
                    raise ValueError("profile aligner must return one Stockholm family")
                first = True
            elif s == "//":
                end = True
            elif not s or s.startswith("#"):
                if s.startswith("#=GC RF "):
                    rf.append(s.split()[-1])
            else:
                if not first or end:
                    raise ValueError("unexpected Stockholm content outside family")
                tokens = s.split()
                if len(tokens) != 2:
                    raise ValueError("invalid aligned Stockholm sequence row")
                if tokens[0] in parts:
                    parts[tokens[0]].append(tokens[1])
    out = {key: "".join(seq).upper().replace(".", "-") for key, seq in parts.items() if seq}
    widths = {len(seq) for seq in out.values()}
    if (
        not first
        or not end
        or len(widths) != 1
        or set(out) != set(names)
        or (rf and len("".join(rf)) != next(iter(widths)))
    ):
        raise ValueError("incomplete or inconsistent selected Stockholm alignment")
    return out, "".join(rf)


def stockholm_names(path, limit):
    names = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line != "//" and not line.startswith("#"):
                names.add(line.split(maxsplit=1)[0])
                if len(names) > limit + 1:
                    raise ValueError("Stockholm sequence count exceeds max_candidates")
    return names


def ref_id_for(ref_id, keys):
    key = ref_id
    i = 0
    while key in keys:
        key = ref_id + ".coordinate" + (f".{i}" if i else "")
        i += 1
    return key


def profile_align(name, comp, ref, selected, raw, aliases, ref_strand, root, programs):
    """Selected maps row ID -> (Hit, oriented source fragment). The seed is always independent."""
    profile_dir = root / "profile"
    profile_dir.mkdir()
    tool = comp["alignment_backend"]
    log = []
    seed = oriented(ref, comp["molecule"], ref_strand)
    if tool == "jackhmmer_stockholm":
        path = raw / "final.sto"
        bindings = {key: aliases[key] for key in selected}
        native_seed = "seed"
    else:
        path = profile_dir / "alignment.sto"
        bindings = {key: f"h{i:09d}" for i, key in enumerate(sorted(selected), 1)}
        native_seed = "seed"
        model = comp.get("alignment_profile", comp.get("profile"))
        check_profile(
            {
                "backend": "hmmer" if tool == "hmmalign" else "infernal",
                "profile": model,
                "molecule": comp["molecule"],
            }
        )
        with (profile_dir / "selected.fa").open("w", encoding="ascii") as f:
            f.write(">seed\n" + seed + "\n")
            for key, alias in bindings.items():
                f.write(f">{alias}\n{selected[key][1]}\n")
        args = [tool, "--outformat", "Stockholm", "-o", "alignment.sto"]
        if tool == "cmalign":
            args.extend(["--cpu", comp["cpu"]])
        copy_model(model, profile_dir / "alignment-model")
        args.extend(["alignment-model", "selected.fa"])
        try:
            run_tool(args, profile_dir, comp, programs, log)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"{tool} exceeded timeout_seconds") from exc
    rows, rf = read_stockholm(path, [native_seed, *bindings.values()])
    reference_row = rows[native_seed]
    if reference_row.replace("-", "") != seed:
        raise ValueError("aligned seed failed exact ungapped reference validation")
    native = {key: rows[a] for key, a in bindings.items()}
    for key, s in native.items():
        if s.replace("-", "") != selected[key][1]:
            raise ValueError(f"profile output sequence differs from source domain {key!r}")
    if ref_strand == "-":
        reference_row = oriented(reference_row, comp["molecule"], "-")
        native = {k: oriented(v, comp["molecule"], "-") for k, v in native.items()}
        rf = rf[::-1]
    columns = [i for i, c in enumerate(reference_row) if c != "-"]
    inserted = [i for i, c in enumerate(reference_row) if c == "-"]
    molecule = comp["molecule"]
    mapped = {}
    details = {}
    for key, row in native.items():
        projected = "".join(row[i] for i in columns)
        occupied = [i for i, c in enumerate(projected) if c != "-"]
        if not occupied:
            raise ValueError(f"{key}: no aligned residues at reference positions")
        a, b = occupied[0], occupied[-1] + 1
        hit = selected[key][0]
        strand = hit.strand if ref_strand == "+" else "-" if hit.strand == "+" else "+"
        placed = Placement(a + 1, projected[a:b], strand, hit.score)
        joint = [
            (x, y)
            for x, y in zip(ref, projected)
            if x in CANONICAL[molecule] and y in CANONICAL[molecule]
        ]
        mapped[key] = replace(
            hit,
            strand=strand,
            placement=placed,
            projection_method=tool,
            identity=sum(x == y for x, y in joint) / len(joint) if joint else None,
            coverage=len(joint) / sum(c in CANONICAL[molecule] for c in ref),
        )
        details[key] = {
            "insertions": sum(row[i] != "-" for i in inserted),
            "deletions": projected.count("-"),
            "domain_coverage": len(joint) / sum(c in CANONICAL[molecule] for c in selected[key][1]),
            "reference_coverage": mapped[key].coverage,
        }
    # Preserve source-coordinate insert columns and a distinct seed when the same ID names a partial domain.
    rid = comp["reference_id"]
    refrow = rid if rid not in native or native[rid] == reference_row else ref_id_for(rid, native)
    keys = tuple(k for k in sorted(native) if k != refrow)
    msa = ComponentMSA(name, molecule, (refrow, *keys), (reference_row, *(native[k] for k in keys)))
    with (profile_dir / "selection.tsv").open("w", encoding="utf-8") as f:
        f.write("tool_row\tsequence_id\tkind\n")
        f.write(f"{native_seed}\t{refrow}\tcoordinate_seed\n")
        for key, alias in bindings.items():
            f.write(f"{alias}\t{key}\tdomain\n")
    return (
        msa,
        refrow,
        mapped,
        details,
        {
            "backend": tool,
            "source_stockholm_sha256": checksum(path),
            "commands": log,
            "seed_orientation_vs_model": ref_strand,
            "aligned_width": len(reference_row),
            "reference_residues": len(columns),
            "retained_insertion_columns": len(inserted),
            "discarded_insertion_columns": 0,
            "rf_match_columns": sum(c not in ".-~" for c in rf) if rf else None,
        },
    )
