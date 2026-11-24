"""Export ranked sites and three-site interactions as an undirected GraphML network."""

import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from triplet_coevolution.reporting.explore_data import artifact


@dataclass(frozen=True)
class RankedTable:
    label: str
    method: str
    path: str
    score: str
    prefixes: tuple[str, ...]
    rows: int
    dca_design: dict | None = None

    def first_rows(self, directory: Path, limit: int):
        delimiter = "," if self.path.endswith(".csv") else "\t"
        with artifact(directory,self.path).open(encoding="utf-8", newline="") as handle:
            return list(islice(csv.DictReader(handle, delimiter=delimiter), limit))


def ranked_tables(summary: dict) -> list[RankedTable]:
    tables = []
    advanced = summary.get("advanced", {}).get("outputs", {})
    selection = summary.get("ranking", {})
    for pair, meta in summary["outputs"]["tables"].items():
        tables.append(RankedTable(
            pair, "MI", meta["path"], summary["settings"]["score"], ("x", "y"), meta["rows"]
        ))
    for pair in ("AB", "AC", "BC"):
        if meta := advanced.get(f"dca_{pair.lower()}"):
            tables.append(
                RankedTable(
                    pair,
                    "DCA",
                    meta["path"],
                    selection.get("dca") or "dca_frobenius",
                    ("x", "y"),
                    meta["rows"],
                    summary["settings"].get("dca_design"),
                )
            )
    if meta := advanced.get("threebody_abc"):
        tables.append(
            RankedTable(
                "ABC",
                "Local maximum entropy"
                if summary.get("advanced", {}).get("local_fit")
                else "Three-body Potts",
                meta["path"],
                selection.get("threebody") or "threebody_rms",
                ("a", "b", "c"),
                meta["rows"],
            )
        )
    return tables


def write_network(directory: Path, tables: list[RankedTable], limit: int = 25, *, structure_evidence=None) -> dict:
    """Keep the top entries of each ranking and represent triples with a factor node."""
    from triplet_coevolution.inference.scores import get_score

    root = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
    keys = {
        "kind": "string", "component": "string", "original_column": "int",
        "reference_id": "string", "reference_position": "int", "label": "string",
        "method": "string", "score": "double", "rank": "int", "n_valid": "int",
        "direction": "string", "units": "string", "competition_rank": "int",
        "evidence_kind":"string", "structure_source_kind":"string",
        "distance_angstrom":"double","structure_category":"string",
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
        spec = get_score(table.score)
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
                "method": table.score,
                "rank": row["rank"],
                "score": row[table.score],
                "n_valid": row["n_valid"],
                "direction": spec.direction,
                "units": spec.metadata(dca_design=table.dca_design)["units"],
                "competition_rank": row.get("competition_rank", ""),
            }
            if len(ids) == 2:
                edge = ET.SubElement(graph, "edge", source=ids[0], target=ids[1])
                joined={}
                if structure_evidence is not None:
                    key=(table.label, row["x_reference_id"],row["x_reference_pos"],row["y_reference_id"],row["y_reference_pos"])
                    joined=structure_evidence.get(key,{})
                data(edge, kind="pair", evidence_kind="statistical association", **evidence, **joined)
                edges += 1
            else:
                factor = "triple_" + "_".join(ids)
                node = ET.SubElement(graph, "node", id=factor)
                data(node, kind="three_site_interaction", evidence_kind="statistical three-site factor", label=" / ".join(ids), **evidence)
                triples += 1
                for site in ids:
                    data(ET.SubElement(graph, "edge", source=site, target=factor), kind="membership",evidence_kind="factor membership")
                    edges += 1
    path = artifact(directory, "network.graphml" if structure_evidence is None else "network_structure.graphml")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return {"path": path.name, "site_nodes": len(sites), "triplet_nodes": triples,
            "edges": edges, "limit_per_table": limit}
