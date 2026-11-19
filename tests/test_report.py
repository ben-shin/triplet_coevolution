import csv
import json
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

from typer.testing import CliRunner

from triplet_coevolution.cli import app


class ReportLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.tags = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append(tag)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a":
            self.hrefs.append(attrs["href"])


def demo_args(out, a=None, ref="prot_ref", matches=None):
    base = Path(__file__).resolve().parents[1] / "examples" / "mixed"
    return [
        "analyze", "--a", str(a or base / "protein.fasta"), "--a-ref", ref,
        "--b", str(base / "dna.fasta"), "--b-type", "dna", "--b-ref", "dna_ref",
        "--c", str(base / "rna.fasta"), "--c-type", "rna", "--c-ref", "rna_ref",
        "--matches", str(matches or base / "matches.tsv"), "--out", str(out),
    ]


def test_report_links_empty_results_and_escaped_reference(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples" / "mixed" / "protein.fasta"
    a = tmp_path / "a.fasta"
    ref = "<script>&site"
    a.write_text(source.read_text().replace("prot_ref", ref), encoding="utf-8")
    for minimum in (2, 100):
        out = tmp_path / str(minimum)
        result = CliRunner().invoke(app, [*demo_args(out, a, ref), "--min-valid", str(minimum)])
        assert result.exit_code == 0, result.output
        html = (out / "report.html").read_text(encoding="utf-8")
        parsed = ReportLinks()
        parsed.feed(html)
        assert "script" not in parsed.tags
        assert "&lt;script&gt;&amp;site" in html
        for href in parsed.hrefs:
            if href.startswith("#"):
                assert href[1:] in parsed.ids
            else:
                assert (out / href).is_file()
        graph = ET.parse(out / "network.graphml")
        namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
        nodes = {node.get("id") for node in graph.findall(".//g:node", namespace)}
        edges = graph.findall(".//g:edge", namespace)
        assert all(edge.get("source") in nodes and edge.get("target") in nodes for edge in edges)
        assert len(edges) == (40 if minimum == 2 else 0)
        if minimum == 100:
            assert "No site combinations passed" in html


def test_advanced_models_benchmarks_and_factor_network(tmp_path):
    out = tmp_path / "advanced"
    base = Path(__file__).resolve().parents[1] / "examples" / "mixed"
    groups = tmp_path / "matches.tsv"
    with (base / "matches.tsv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.reader(source, delimiter="\t"))
    with groups.open("w", encoding="utf-8", newline="") as dest:
        writer = csv.writer(dest, delimiter="\t")
        writer.writerow([*rows[0], "clade"])
        for idx, row in enumerate(rows[1:]):
            writer.writerow([*row, f"group{idx % 2}"])
    candidates = tmp_path / "candidates.tsv"
    candidates.write_text("a_original_col\tb_original_col\tc_original_col\n1\t1\t1\n")
    contacts = tmp_path / "contacts.tsv"
    contacts.write_text(
        "x_component\tx_reference_id\tx_reference_pos\ty_component\ty_reference_id\t"
        "y_reference_pos\tcontact\nA\tprot_ref\t1\tB\tdna_ref\t1\t1\n"
        "A\tprot_ref\t1\tB\tdna_ref\t2\t0\n"
    )
    result = CliRunner().invoke(app, [
        *demo_args(out, matches=groups), "--dca", "--threebody", "--permutations", "3",
        "--seed", "11", "--strata-column", "clade", "--triplet-candidates", str(candidates),
        "--contacts", str(contacts), "--fmt", "csv",
    ])
    assert result.exit_code == 0, result.output
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    html = (out / "report.html").read_text(encoding="utf-8")
    for stem in ("dca_ab", "threebody_abc", "permutation_threebody", "permutation_dca",
                 "permutation_mi", "contacts"):
        artifact = summary["advanced"]["outputs"][stem]["path"]
        assert (out / artifact).is_file() and artifact in html
    for phrase in ("AUROC", "Candidate coverage", "threebody_rms", "BH q",
                   "dca_frobenius", "Co-information (bits)", "AB MI (bits)"):
        assert phrase in html
    assert summary["advanced"]["settings"]["strata_column"] == "clade"
    assert summary["outputs"]["network"]["triplet_nodes"] == 1
    assert (out / "network.graphml").read_text(encoding="utf-8").count(">membership<") == 3


def test_coordinate_row_and_trimmed_observation_keep_distinct_roles(tmp_path):
    base = Path(__file__).resolve().parents[1] / "examples" / "mixed"
    a = tmp_path / "a.fasta"
    a.write_text((base / "protein.fasta").read_text().replace("prot_ref", "coordinate_only")
                 .replace(">prot01\nA-KLT", ">prot01\n--KLT"), encoding="utf-8")
    out = tmp_path / "report"
    contacts = tmp_path / "labels.tsv"
    contacts.write_text(
        "x_component\tx_reference_id\tx_reference_pos\ty_component\ty_reference_id\t"
        "y_reference_pos\tcontact\nA\tprot01\t101\tB\tdna_ref\t1\t1\n"
    )
    result = CliRunner().invoke(app, [
        *demo_args(out, a=a, ref="prot01"), "--a-ref-row", "coordinate_only", "--a-start", "101",
        "--contacts", str(contacts),
    ])
    assert result.exit_code == 0, result.output
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    comp = summary["components"]["A"]
    assert (comp["reference_id"], comp["reference_row_id"]) == ("prot01", "coordinate_only")
    with (out / "mi_ab.tsv").open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle, delimiter="\t") if r["x_original_col"] == "1"]
    assert len(rows) == 4
    assert all((r["x_reference_id"], r["x_reference_pos"], r["n_valid"]) == ("prot01", "101", "7")
               for r in rows)
    labels = json.loads((out / "contacts.json").read_text())
    assert labels["methods"]["mi"]["scored"] == 1
