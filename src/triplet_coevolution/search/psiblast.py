"""Iterative PSI-BLAST from a protein reference or binary PSSM checkpoint."""

from triplet_coevolution.search.shared import execute, external_hit


def search(comp, database, work):
    index = work / "blastdb"
    table = work / "blast.tsv"
    execute(
        [
            "makeblastdb",
            "-in",
            database.name,
            "-dbtype",
            "prot",
            "-parse_seqids",
            "-out",
            index.name,
        ],
        cwd=work,
    )
    query = ["-in_pssm", comp["profile"]] if "profile" in comp else ["-query", comp["reference"]]
    # Explicitly request the full database so BLAST's default hit cap cannot drop evidence groups.
    with database.open(encoding="ascii") as handle:
        count = sum(line.startswith(">") for line in handle)
    execute(
        [
            "psiblast",
            *query,
            "-db",
            index.name,
            "-num_iterations",
            comp["iterations"],
            "-num_threads",
            "1",
            "-evalue",
            comp["evalue"],
            "-max_target_seqs",
            max(1, count),
            "-outfmt",
            "7 sseqid sstart send bitscore evalue",
            "-out",
            table.name,
        ],
        cwd=work,
    )
    # Retain the last reported iteration rather than pooling incompatible iterations.
    final = []
    with table.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("# Iteration:"):
                final.clear()
            elif (
                line.strip()
                and not line.startswith("#")
                and line.strip() != "Search has CONVERGED!"
            ):
                row = line.split()
                if len(row) != 5:
                    raise ValueError("malformed PSI-BLAST output")
                final.append(external_hit(row, row[0].removeprefix("lcl|"), 1, 2, 3, 4))
    yield from final
