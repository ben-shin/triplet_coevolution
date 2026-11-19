"""Export ranked sites and three-site interactions as an undirected GraphML network."""

import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import islice
from pathlib import Path


@dataclass(frozen=True)
class RankedTable:
    label: str
    method: str
    path: str
    score: str
    prefixes: tuple[str, ...]
    rows: int

    def first_rows(self, directory: Path, limit: int):
        delimiter = "," if self.path.endswith(".csv") else "\t"
        with (directory / self.path).open(encoding="utf-8", newline="") as handle:
            return list(islice(csv.DictReader(handle, delimiter=delimiter), limit))


def ranked_tables(summary: dict) -> list[RankedTable]:
    tables = []
    advanced = summary.get("advanced", {}).get("outputs", {})
    for pair, meta in summary["outputs"]["tables"].items():
        tables.append(RankedTable(
            pair, "MI", meta["path"], summary["settings"]["score"], ("x", "y"), meta["rows"]
        ))
    for pair in ("AB", "AC", "BC"):
        if meta := advanced.get(f"dca_{pair.lower()}"):
            tables.append(RankedTable(
                pair, "DCA", meta["path"], "dca_frobenius", ("x", "y"), meta["rows"]
            ))
    if meta := advanced.get("threebody_abc"):
        tables.append(RankedTable(
            "ABC", "Three-body Potts", meta["path"], "threebody_rms", ("a", "b", "c"), meta["rows"]
        ))
    return tables


def write_network(directory: Path, tables: list[RankedTable], limit: int = 25) -> dict:
    """Keep the top entries of each ranking and represent triples with a factor node."""
    root = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
    keys = {
        "kind": "string", "component": "string", "original_column": "int",
        "reference_id": "string", "reference_position": "int", "label": "string",
        "method": "string", "score": "double", "rank": "int", "n_valid": "int",
    }
    for key, dtype in keys.items():
        ET.SubElement(root, "key", {
            "id": key, "for": "all", "attr.name": key, "attr.type": dtype,
        })
    graph = ET.SubElement(root, "graph", id="interactions", edgedefault="undirected")
    sites = set()
    edges = triples = 0

    def data(element, **values):
        for key, value in values.items():
            if value != "":
                # Sequence identifiers can contain characters excluded from XML 1.0.
                text = "".join(c for c in str(value) if c.isprintable())
                ET.SubElement(element, "data", key=key).text = text

    for table in tables:
        for row in table.first_rows(directory, limit):
            ids = []
            for prefix in table.prefixes:
                component = row[f"{prefix}_component"]
                col = row[f"{prefix}_original_col"]
                reference = row[f"{prefix}_reference_id"]
                position = row[f"{prefix}_reference_pos"]
                identifier = f"{component}_{col}"
                ids.append(identifier)
                if identifier not in sites:
                    sites.add(identifier)
                    element = ET.SubElement(graph, "node", id=identifier)
                    data(element, kind="site", component=component, original_column=col,
                         reference_id=reference, reference_position=position,
                         label=f"{component} {reference} {position or ('alignment column ' + col)}")
            evidence = {
                "method": table.score, "rank": row["rank"], "score": row[table.score],
                "n_valid": row["n_valid"],
            }
            if len(ids) == 2:
                edge = ET.SubElement(graph, "edge", source=ids[0], target=ids[1])
                data(edge, kind="pair", **evidence)
                edges += 1
            else:
                factor = "triple_" + "_".join(ids)
                node = ET.SubElement(graph, "node", id=factor)
                data(node, kind="three_site_interaction", label=" / ".join(ids), **evidence)
                triples += 1
                for site in ids:
                    data(ET.SubElement(graph, "edge", source=site, target=factor), kind="membership")
                    edges += 1
    path = directory / "network.graphml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return {"path": path.name, "site_nodes": len(sites), "triplet_nodes": triples,
            "edges": edges, "limit_per_table": limit}
