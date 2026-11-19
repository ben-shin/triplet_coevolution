"""Infernal RNA covariance-model search, default format-1 target table."""

from triplet_coevolution.search.shared import execute, external_hit, table_rows


def search(comp, database, work):
    table = work / "infernal.tbl"
    execute(
        [
            "cmsearch",
            "--cpu",
            "1",
            "--noali",
            "-E",
            comp["evalue"],
            "--tblout",
            table,
            comp["profile"],
            database,
        ]
    )
    for row in table_rows(table):
        if len(row) < 17:
            raise ValueError("malformed Infernal hit table")
        yield external_hit(row, row[0], 7, 8, 14, 15, row[9])
