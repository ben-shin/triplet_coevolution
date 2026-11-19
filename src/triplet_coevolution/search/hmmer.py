"""HMMER domain searches using its documented tabular formats."""

from triplet_coevolution.search.shared import execute, external_hit, table_rows


def search(comp, database, work):
    table = work / "hmmer.tbl"
    cutoff = comp["evalue"]
    if comp["molecule"] == "protein":
        execute(
            [
                "hmmsearch",
                "--cpu",
                "1",
                "--noali",
                "-E",
                cutoff,
                "--domE",
                cutoff,
                "--domtblout",
                table,
                comp["profile"],
                database,
            ]
        )
        for row in table_rows(table):
            if len(row) < 22:
                raise ValueError("malformed HMMER domain table")
            # Independent domain E-value and bit score, inclusive aligned target interval.
            yield external_hit(row, row[0], 17, 18, 13, 12)
    else:
        execute(
            [
                "nhmmer",
                "--cpu",
                "1",
                "--noali",
                "-E",
                cutoff,
                "--tblout",
                table,
                comp["profile"],
                database,
            ]
        )
        for row in table_rows(table):
            if len(row) < 15:
                raise ValueError("malformed nhmmer table")
            yield external_hit(row, row[0], 6, 7, 13, 12, row[11])
