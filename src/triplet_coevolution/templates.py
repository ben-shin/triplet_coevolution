"""Generate small, completely local, versioned example projects."""

import csv
from pathlib import Path

from triplet_coevolution.operations import put_json
from triplet_coevolution.publication import _stage_run

TEMPLATES = {
    "protein-protein-protein": ("protein", "protein", "protein"),
    "protein-protein-rna": ("protein", "protein", "rna"),
    "protein-rna-rna": ("protein", "rna", "rna"),
    "rna-rna-rna": ("rna", "rna", "rna"),
    "protein-dna-rna": ("protein", "dna", "rna"),
}


def _tsv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def init_project(directory: Path, template: str = "protein-protein-rna") -> Path:
    """Write a self-contained training fixture; return its native analysis configuration."""
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {', '.join(TEMPLATES)}")
    directory = Path(directory).resolve()
    with _stage_run(directory) as out:
        inp = out / "input"
        inp.mkdir()
        roles = {}
        mappings = []
        matches = []
        for r, (role, molecule) in enumerate(zip("ABC", TEMPLATES[template], strict=True)):
            folder = f"{role}_{molecule}"
            component = inp / folder
            component.mkdir()
            alphabet = {"protein": "KEFG", "rna": "AUGC", "dna": "ATGC"}[molecule]
            ref = role + "01"
            start = (11, 51, 101)[r]
            base = {"molecule": molecule, "reference_id": ref, "reference_start": start}
            put_json(component / "component.json", base)
            roles[role] = {**base, "alignment": f"input/{folder}/family.sto", "format": "stockholm"}
            with (component / "family.sto").open("w", encoding="ascii", newline="\n") as handle:
                handle.write(f"# STOCKHOLM 1.0\n#=GF ID synthetic_{role}\n")
                for i in range(16):
                    sid = f"{role}{i + 1:02d}"
                    x, y = (i // 2) % 2, i % 2
                    value = (x, y, x ^ y)[r]
                    seq = (
                        alphabet[value]
                        + ("-" if i == 0 else alphabet[(i + r) % 4])
                        + alphabet[(i // 4 + r) % 4]
                        + alphabet[(i // 8 + r) % 4]
                    )
                    handle.write(f"{sid} {seq}\n")
                    mappings.append(
                        (
                            role,
                            sid,
                            f"locus_{sid}",
                            f"genome{i + 1:02d}",
                            f"fixture:{role}",
                            "synthetic generator",
                        )
                    )
                    if r == 0:
                        matches.append(
                            (
                                f"obs{i + 1:02d}",
                                *(f"{c}{i + 1:02d}" for c in "ABC"),
                                f"clade{i // 4 + 1}",
                            )
                        )
                handle.write("//\n")
        _tsv(
            inp / "mappings.tsv",
            ("component", "identifier", "protein_id", "genome_id", "orthogroup", "source"),
            mappings,
        )
        _tsv(out / "matches.tsv", ("observation_id", "a_id", "b_id", "c_id", "clade"), matches)
        put_json(
            inp / "project.json",
            {
                "schema_version": 1,
                "uniprot": False,
                "expected_orthogroups": {r: [f"fixture:{r}"] for r in "ABC"},
            },
        )
        _tsv(
            out / "labels.tsv",
            (
                "x_component",
                "x_reference_id",
                "x_reference_pos",
                "y_component",
                "y_reference_id",
                "y_reference_pos",
                "contact",
                "source",
            ),
            [
                (
                    a,
                    a + "01",
                    apos,
                    b,
                    b + "01",
                    bpos + i,
                    (1 if i == 0 else 0),
                    "synthetic illustrative labels",
                )
                for a, apos, b, bpos in [
                    ("A", 11, "B", 51),
                    ("A", 11, "C", 101),
                    ("B", 51, "C", 101),
                ]
                for i in range(2)
            ],
        )
        put_json(
            out / "analysis.json",
            {
                "kind": "analysis",
                "schema_version": 1,
                "matches": "matches.tsv",
                "components": roles,
                "analysis": {
                    "dca": True,
                    "threebody": True,
                    "max_triplets": 64,
                    "contacts": "labels.tsv",
                    "seed": 7,
                },
            },
        )
        (out / "README.md").write_text(
            f"# {template} synthetic project\n\nThe 16 observations contain a planted three-way dependence at the first column. "
            "The local matching and label tables are synthetic. Each reference has a gap at column 2 and its own numbering start.\n\n"
            "Run these commands from this directory.\n\n~~~sh\ntriplet run --input input --offline --out family-result\n"
            "triplet score --config analysis.json --dry-run --out planning\ntriplet score --config analysis.json --out scores\n"
            "triplet report --run scores --out regenerated\n~~~\n",
            encoding="utf-8",
        )
    return directory / "analysis.json"
