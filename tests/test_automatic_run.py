"""Unattended CLI coverage using temporary synthetic families and simulated public metadata."""

import csv
import gzip
import json
import sqlite3
from contextlib import closing
from html.parser import HTMLParser
from http.client import IncompleteRead
from io import BytesIO
from urllib.parse import parse_qs, urlsplit

import pytest
from typer.testing import CliRunner

from triplet_coevolution.cli import app
from triplet_coevolution.evidence import nucleotide, uniprot
from triplet_coevolution.io.alignments import read_msa

UNIPROT_REQUEST = uniprot._request


def _no_request(*_args, **_kwargs):
    pytest.fail("unexpected public request in isolated fixture")


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    monkeypatch.setattr(uniprot, "_request", _no_request)
    monkeypatch.setattr(nucleotide, "_request_json", _no_request)


def _folders(base, n=4, local=True):
    for index, folder in enumerate(("ATPase", "Partner", "guide-RNA")):
        path = base / folder
        path.mkdir(parents=True)
        profile = ("PF98001", "PF98002", "RF98003")[index]
        lines = ["# STOCKHOLM 1.0", f"#=GF AC {profile}.2"]
        for i in range(1, n + 1):
            token = (f"P19{i:03}", f"Q19{i:03}", f"AB{i:06}.1")[index] + "/11-14"
            if local:
                lines += [
                    f"#=GS {token} ASSEMBLY GCA_{i:09}.1",
                    f"#=GS {token} LOCUS loc{index}_{i}",
                ]
            sequence = (("CKDY", "GWFA"), ("FKMD", "YDEH"), ("TCGA", "CGTT"))[index][i % 2]
            lines.append(f"{token} {sequence}")
        (path / "family.sto").write_text("\n".join([*lines, "//", ""]), encoding="utf-8")


def _run(base, out, *options):
    return CliRunner().invoke(app, ["run", "--input", str(base), "--out", str(out), *options])


def _json(file):
    return json.loads(file.read_text(encoding="utf-8"))


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.paths.append(dict(attrs)["href"])


def test_bare_folders_score_benchmark_and_repeat(tmp_path, monkeypatch):
    _folders(tmp_path / "input")
    monkeypatch.chdir(tmp_path)
    for _ in range(2):
        run = CliRunner().invoke(app, ["run"])
        assert run.exit_code == 0, (run.output, run.exception)
    outputs = list((tmp_path / "results").glob("run-*"))
    assert len(outputs) == 2
    out = outputs[0]
    summary = _json(out / "workflow_summary.json")
    assert summary["network_enabled"] and summary["status"] == "complete"
    assert summary["counts"]["resolved_groups"] == 4
    assert summary["role_evidence_counts"] == {"profile_membership_included": 4}
    assert (
        summary["components"]["C"]["molecule"] == "rna"
        and summary["molecules"]["C"]["transcribe_t"]
    )
    assert summary["components"]["C"]["reference_start"] == 11
    with (out / summary["reference_coordinates"]).open(newline="") as source:
        symbols = "".join(
            row["symbol"]
            for row in csv.DictReader(source, delimiter="\t")
            if row["component"] == "C"
        )
    assert symbols == "CGTT"
    assert read_msa(out / "prepared/C.sto", "C", "rna").sequences[0] == "CGUU"
    scores = _json(out / summary["analysis_summary"])
    assert scores["advanced"]["settings"]["permutations"] == 19
    for key in (
        "permutation_mi",
        "permutation_dca",
        "permutation_threebody",
        "threebody_abc",
        "dca_bc",
    ):
        table = scores["advanced"]["outputs"][key]
        assert table["rows"] > 0 and (out / "analysis" / table["path"]).is_file()
    report = (out / summary["report"]).read_text(encoding="utf-8")
    assert summary["score_report"] in report and summary["group_audit"] in report
    links = _Links()
    links.feed(report)
    assert all((out / target).is_file() for target in links.paths)
    assert all(
        (out / "analysis" / x["path"]).is_file() for x in scores["outputs"]["tables"].values()
    )
    assert all(".triplet-in-progress" not in d["path"] for d in scores["inputs"].values())


def test_identifier_only_mock_providers_then_cache_replay(tmp_path, monkeypatch):
    base, out = tmp_path / "inputs", tmp_path / "first"
    _folders(base, n=3, local=False)
    seen = []

    def proteins(url):
        parts = urlsplit(url)
        request = parse_qs(parts.query)["query"][0]
        seen.append(parts.path)
        entries = []
        for i in range(1, 4):
            proteome = f"UP{i:09}"
            if parts.path == "/proteomes/search" and proteome in request:
                entries.append({"id": proteome, "genomeAssembly": {"assemblyId": f"GCA_{i:09}.1"}})
            for ac in (f"P19{i:03}", f"Q19{i:03}"):
                if parts.path == "/uniprotkb/search" and f"accession:{ac}" in request:
                    entries.append(
                        {
                            "primaryAccession": ac,
                            "uniProtkbId": ac + "_MOCK",
                            "organism": {"taxonId": i},
                            "uniProtKBCrossReferences": [{"database": "Proteomes", "id": proteome}],
                        }
                    )
        return {"results": entries}, {
            "x-uniprot-release": "fixture",
            "x-uniprot-release-date": "test",
        }

    def nucleotides(url):
        assert "ebi.ac.uk" in url and "family.sto" not in url
        seen.append("ENA")
        return [
            {
                "accession": f"AB{i:06}",
                "sequence_version": 1,
                "assembly_accession": f"GCA_{i:09}.1",
                "tax_id": i,
                "base_count": 50,
            }
            for i in range(1, 4)
        ]

    monkeypatch.setattr(uniprot, "_request", proteins)
    monkeypatch.setattr(nucleotide, "_request_json", nucleotides)
    first = _run(base, out)
    assert first.exit_code == 0, (first.output, first.exception)
    assert {"/uniprotkb/search", "/proteomes/search", "ENA"} <= set(seen)
    monkeypatch.setattr(uniprot, "_request", _no_request)
    monkeypatch.setattr(nucleotide, "_request_json", _no_request)
    cached = tmp_path / "cached"
    second = _run(base, cached, "--offline")
    assert second.exit_code == 0, (second.output, second.exception)
    assert (cached / "prepared/matches.tsv").read_bytes() == (
        out / "prepared/matches.tsv"
    ).read_bytes()
    summary = _json(cached / "metadata/evidence_summary.json")
    assert summary["providers"]["nucleotide"]["counts"]["cache_hits"] >= 3
    assert summary["resolver"]["cache_hits"] >= 6


def test_explained_zero_and_partial_results(tmp_path, monkeypatch):
    base = tmp_path / "input"
    _folders(base, n=2, local=False)
    out = tmp_path / "unplaced"
    assert _run(base, out, "--offline").exit_code == 3
    summary = _json(out / "workflow_summary.json")
    assert summary["status"] == "no_matches" and (out / summary["evidence"]).is_file()
    assert "No matches" in (out / "report.html").read_text(encoding="utf-8")
    assert not (out / "analysis").exists()

    def outage(url):
        raise uniprot.UniProtError("simulated service outage")

    monkeypatch.setattr(uniprot, "_request", outage)
    monkeypatch.setattr(nucleotide, "_request_json", lambda url: [])
    failed = tmp_path / "outage"
    assert _run(base, failed).exit_code == 4
    summary = _json(failed / "workflow_summary.json")
    assert summary["status"] == "provider_failure" and summary["providers"]["failures"]
    assert (failed / summary["evidence"]).is_file() and not (failed / "analysis").exists()


def test_truncated_public_response_finishes_with_a_closed_cache_and_report(tmp_path, monkeypatch):
    class Truncated(BytesIO):
        def read(self, *args, **kwargs):
            raise IncompleteRead(b"{", 37)

    responses = []

    def transport(*args, **kwargs):
        response = Truncated()
        responses.append(response)
        return response

    monkeypatch.setattr(uniprot, "_request", UNIPROT_REQUEST)
    monkeypatch.setattr(uniprot, "urlopen", transport)
    monkeypatch.setattr(uniprot.time, "sleep", lambda *_: None)
    monkeypatch.setattr(nucleotide, "_request_json", lambda url: [])
    base, out = tmp_path / "input", tmp_path / "out"
    _folders(base, 3, local=False)
    result = _run(base, out)
    assert result.exit_code == 4, (result.output, result.exception)
    summary = _json(out / "workflow_summary.json")
    assert summary["status"] == "provider_failure" and not summary["ready_for_analysis"]
    assert summary["providers"]["failures"]["UniProtKB"] and (out / summary["report"]).is_file()
    assert (out / summary["evidence"]).is_file() and not (out / "analysis").exists()
    assert len(responses) == 4 and all(r.closed for r in responses)
    cache = out / "metadata/uniprot.sqlite"
    with closing(sqlite3.connect(cache)) as db:
        db.execute("begin exclusive")
        assert db.execute("select count(*) from lookups").fetchone() == (0,)


def test_input_choice_and_failure_report(tmp_path):
    base = tmp_path / "input"
    _folders(base)
    full = base / "ATPase/family.full"
    (base / "ATPase/family.sto").rename(full)
    (full.parent / "family.full.gz").write_bytes(gzip.compress(full.read_bytes()))
    seed = full.parent / "family.seed"
    seed.write_text(full.read_text().replace("CKDY", "CGDY"))
    out = tmp_path / "accepted"
    result = _run(base, out, "--offline")
    assert result.exit_code == 0, (result.output, result.exception)
    assert _json(out / "workflow_summary.json")["sources"]["A"]["path"] == str(full)
    seed.write_text(seed.read_text().replace("PF98001", "PF98009"))
    out = tmp_path / "refused"
    assert _run(base, out, "--offline").exit_code == 1
    summary = _json(out / "workflow_summary.json")
    assert summary["status"] == "invalid_input" and "competing Stockholm" in summary["reason"]
    assert (out / summary["report"]).is_file()
    before = (out / "workflow_summary.json").read_bytes()
    assert _run(base, out, "--offline").exit_code == 1
    assert (out / "workflow_summary.json").read_bytes() == before


def test_unavailable_optional_model_keeps_mi_and_benchmark(tmp_path, monkeypatch):
    from triplet_coevolution.inference import advanced

    def no_fit(*args, **kwargs):
        raise ValueError("simulated solver budget")

    monkeypatch.setattr(advanced, "direct_couplings", no_fit)
    base, out = tmp_path / "in", tmp_path / "out"
    _folders(base, n=3)
    assert _run(base, out, "--offline").exit_code == 3
    summary = _json(out / "workflow_summary.json")
    assert summary["status"] == "partial" and summary["analysis_plan"]["skipped"]["dca"]
    assert (out / "analysis/permutation_mi.tsv").is_file() and (
        out / "analysis/threebody_abc.tsv"
    ).is_file()
    assert "--dca" not in _json(out / summary["analyze_args"])


def test_project_limits_and_overrides(tmp_path):
    base = tmp_path / "in"
    _folders(base)
    (base / "project.json").write_text(json.dumps({"analysis_limits": {"pair_cells": 10}}))
    limited = tmp_path / "limited"
    assert _run(base, limited, "--offline").exit_code == 3
    summary = _json(limited / "workflow_summary.json")
    assert summary["status"] == "resource_limit" and (limited / summary["matches"]).is_file()
    (base / "project.json").write_text(
        json.dumps({"analysis": {"dca": False, "threebody": False, "permutations": 2, "seed": 12}})
    )
    override = tmp_path / "override"
    run = _run(base, override, "--offline")
    assert run.exit_code == 0, (run.output, run.exception)
    with (override / "analysis/permutation_mi.tsv").open(newline="") as fh:
        assert {r["null_draws"] for r in csv.DictReader(fh, delimiter="\t")} == {"2"}
    assert not (override / "analysis/dca_ab.tsv").exists()


def test_too_few_matches_still_has_a_report_and_evidence(tmp_path):
    base, out = tmp_path / "in", tmp_path / "out"
    _folders(base, n=1)
    run = _run(base, out, "--offline")
    assert run.exit_code == 3, (run.output, run.exception)
    summary = _json(out / "workflow_summary.json")
    assert summary["status"] == "insufficient_data"
    assert not summary["ready_for_analysis"] and (out / summary["evidence"]).is_file()
    assert (out / "prepared/C.sto").is_file() and (out / summary["report"]).is_file()


def test_invalid_selected_symbols_keep_finished_evidence(tmp_path):
    base, out = tmp_path / "in", tmp_path / "out"
    _folders(base)
    comp = base / "guide-RNA"
    (comp / "component.json").write_text('{"molecule":"rna"}')
    alignment = comp / "family.sto"
    alignment.write_text(alignment.read_text().replace("TCGA", "ECGA"))
    run = _run(base, out, "--offline")
    assert run.exit_code == 1, (run.output, run.exception)
    summary = _json(out / "workflow_summary.json")
    assert summary["status"] == "invalid_input" and "component C" in summary["reason"]
    assert (out / summary["evidence"]).is_file() and (out / summary["matches"]).is_file()
    assert summary["reference_coordinates"] is None and not list((out / "prepared").glob("*.sto"))


def test_prespecified_sites_replay_original_reference_coordinates(tmp_path, monkeypatch):
    base, out = tmp_path / "input", tmp_path / "scored"
    _folders(base)
    protein = base / "ATPase/family.sto"
    protein.write_text(protein.read_text().replace("CKDY", "C-KDY").replace("GWFA", "G-WFA"))
    sites = base / "sites.tsv"
    sites.write_text("component\toriginal_column\nA\t1\nA\t2\nA\t5\nB\t2\nC\t3\n")
    (base / "project.json").write_text(json.dumps({"analysis": {"site_panel": "sites.tsv"}}))
    result = _run(base, out, "--offline")
    assert result.exit_code == 0, (result.output, result.exception)
    analysis = _json(out / "analysis/run_summary.json")
    assert analysis["components"]["A"]["retained_columns"] == [1, 5]
    with (out / "analysis/mi_ab.tsv").open(newline="") as source:
        pairs = list(csv.DictReader(source, delimiter="\t"))
    assert {(r["x_original_col"], r["x_reference_pos"]) for r in pairs} == {
        ("1", "11"),
        ("5", "14"),
    }
    monkeypatch.chdir(out / "prepared")
    args = _json(out / "prepared/analyze_args.json")
    replay = tmp_path / "native"
    assert CliRunner().invoke(app, [*args, "--out", str(replay)]).exit_code == 0
    assert (replay / "mi_ab.tsv").read_bytes() == (out / "analysis/mi_ab.tsv").read_bytes()
    sites.write_text("component\toriginal_column\nA\t999\nB\t2\nC\t3\n")
    bad = tmp_path / "bad"
    assert _run(base, bad, "--offline").exit_code == 1
    failure = _json(bad / "workflow_summary.json")
    assert failure["status"] == "analysis_error" and "out-of-range" in failure["reason"]


def test_oversize_default_scores_bounded_panel_and_replays(tmp_path, monkeypatch):
    from triplet_coevolution import automatic_run

    base, out = tmp_path / "input", tmp_path / "bounded"
    _folders(base, 12)
    monkeypatch.setattr(automatic_run, "DEFAULT_LIMITS", {"pair_cells": 24, "weight_cells": 800})
    result = _run(base, out, "--offline")
    assert result.exit_code == 3, (result.output, result.exception)
    summary = _json(out / "workflow_summary.json")
    assert summary["status"] == "partial", summary["reason"]
    panel = summary["scoring_panel"]
    assert panel["n_input"] == 12 and summary["scored_scope"]["selection"] == "bounded_panel"
    assert summary["scored_scope"]["observations"] == panel["n_selected"]
    assert 8 <= panel["n_selected"] <= 12 and any(v < 4 for v in panel["selected_widths"].values())
    assert len(read_msa(out / "prepared/A.sto", "A", "protein").ids) == 12
    assert (out / summary["full_analyze_args"]).is_file()
    snapshot = _json(out / "analysis/run_summary.json")
    assert ".triplet-in-progress" not in snapshot["inputs"]["site_panel"]["path"]
    assert (out / "analysis/permutation_mi.tsv").is_file() and (
        out / "analysis/threebody_abc.tsv"
    ).is_file()
    replay_args = _json(out / summary["analyze_args"])
    assert "--site-panel" in replay_args
    monkeypatch.chdir(out / "prepared")
    replay = tmp_path / "replay"
    invoked = CliRunner().invoke(app, [*replay_args, "--out", str(replay)])
    assert invoked.exit_code == 0, invoked.output
    for name in ("mi_ab.tsv", "dca_bc.tsv", "permutation_threebody.tsv"):
        assert (out / "analysis" / name).read_bytes() == (replay / name).read_bytes()


def _gappy_project(base, sequences, analysis=None):
    expected = {}
    for name, seqs, folder, profile in zip(
        "ABC", sequences, ("Alpha", "Beta", "Gamma"), ("PF98001", "PF98002", "RF98003"), strict=True
    ):
        dest = base / folder
        dest.mkdir(parents=True)
        lines = ["# STOCKHOLM 1.0", f"#=GF AC {profile}.2"]
        expected[name] = [f"fixture:{name}"]
        for i, sequence in enumerate(seqs, 1):
            row = f"{name}_obs{i}"
            lines.extend(
                (
                    f"#=GS {row} ASSEMBLY GCA_{i:09}.1",
                    f"#=GS {row} LOCUS {row}",
                    f"#=GS {row} ORTHOGROUP fixture:{name}",
                    f"{row} {sequence}",
                )
            )
        (dest / "family.sto").write_text("\n".join([*lines, "//", ""]))
    (base / "Alpha/component.json").write_text(json.dumps({"reference_id": "A_obs1"}))
    (base / "project.json").write_text(
        json.dumps({"expected_orthogroups": expected, "analysis": analysis or {}})
    )


def test_post_mask_row_filter_stratified_replay(tmp_path, monkeypatch):
    base, out = tmp_path / "in", tmp_path / "out"
    _gappy_project(
        base,
        (
            ["?--A", "KKF-", "GGY-", "FKG-", "QWK-", "GWD-"],
            ["MEAE", "HEMI", "MIDY", "GMEI", "MWDW", "WEID"],
            ["ACUG", "ACGG", "GUAA", "UUAC", "UCGA", "AGCC"],
        ),
        {"strata_column": "role_evidence", "permutations": 3, "site_panel": "sites.tsv"},
    )
    (base / "sites.tsv").write_text(
        "component\toriginal_column\nA\t1\nA\t3\nA\t4\nB\t2\nB\t3\nC\t3\n"
    )
    run = _run(base, out, "--offline")
    assert run.exit_code == 3, (run.output, run.exception)
    workflow = _json(out / "workflow_summary.json")
    assert workflow["status"] == "partial" and workflow["ready_for_analysis"]
    assert workflow["counts"]["resolved_groups"] == 6 and workflow["analysis_blockers"] == []
    audit = workflow["observation_qc"]
    assert audit["eligible_observations"] == 5 and audit["excluded_observations"] == 1
    with (out / audit["table"]).open(newline="") as fh:
        (record,) = csv.DictReader(fh, delimiter="\t")
    assert (record["a_id"], record["components_without_canonical_residues"]) == ("A_obs1", "A")
    full = read_msa(out / "prepared/A.sto", "A", "protein")
    assert len(full.ids) == 6 and "A_obs1" in full.ids
    assert workflow["scored_scope"] == {
        "selection": "qc_supported",
        "observations": 5,
        "retained_columns": {"A": 2, "B": 2, "C": 1},
    }
    result = _json(out / workflow["analysis_summary"])
    assert result["components"]["A"]["reference_id"] == "A_obs1"
    assert result["components"]["A"]["retained_columns"] == [1, 3]
    assert result["advanced"]["settings"]["strata_column"] == "role_evidence"
    for report in (out / "report.html", out / "analysis/report.html"):
        html = report.read_text(encoding="utf-8")
        assert record["observation_id"] in html and "observation_qc.tsv" in html
    args = _json(out / workflow["analyze_args"])
    assert "scored_matches.tsv" in args and "scored_columns.tsv" in args
    monkeypatch.chdir(out / "prepared")
    direct = tmp_path / "replay"
    assert CliRunner().invoke(app, [*args, "--out", str(direct)]).exit_code == 0
    for file in ("mi_ab.tsv", "threebody_abc.tsv", "permutation_mi.tsv", "permutation_dca.tsv"):
        assert (direct / file).read_bytes() == (out / "analysis" / file).read_bytes()
    legacy = _json(out / "prepared/analyze_args.json")
    explicit = CliRunner().invoke(app, [*legacy, "--out", str(tmp_path / "legacy")])
    assert explicit.exit_code == 1 and "no canonical res" in explicit.output


@pytest.mark.parametrize(("b_good", "count"), [({4, 5, 6}, 0), ({3, 4, 5}, 1)])
def test_insufficient_after_row_qc_keeps_audit(tmp_path, b_good, count):
    base, out = tmp_path / "in", tmp_path / "out"
    _gappy_project(
        base,
        (
            ["KKFG" if i in {1, 2, 3} else "????" for i in range(1, 7)],
            ["QWDD" if i in b_good else "????" for i in range(1, 7)],
            ["ACUG"] * 6,
        ),
    )
    invoked = _run(base, out, "--offline")
    assert invoked.exit_code == 3, (invoked.output, invoked.exception)
    summary = _json(out / "workflow_summary.json")
    assert summary["status"] == "insufficient_data" and not summary["ready_for_analysis"]
    assert summary["reason"] in summary["analysis_blockers"]
    assert summary["analyze_args"] is None and not (out / "analysis").exists()
    qc = summary["observation_qc"]
    assert qc["eligible_observations"] == count and qc["excluded_observations"] == 6 - count
    assert (out / qc["matches"]).read_text().count("\n") == count + 1
    assert (out / qc["table"]).read_text().count("\n") == 7 - count
    assert (out / summary["matches"]).read_text().count("\n") == 7


def test_automatic_bounded_panel_filters_unsupported_candidates(tmp_path, monkeypatch):
    from triplet_coevolution import automatic_run

    base, out = tmp_path / "in", tmp_path / "out"
    _folders(base, 12)
    # The sampled source columns for width four are 2 and 4. Two candidates lack both.
    path = base / "ATPase/family.sto"
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if any(token in line for token in ("P19001/11-14", "P19002/11-14")):
            lines[i] = line.replace("/11-14", "/11-12")
            if not line.startswith("#"):
                lines[i] = lines[i].split()[0] + " K-E-"
    path.write_text("\n".join([*lines, ""]))
    monkeypatch.setattr(automatic_run, "DEFAULT_LIMITS", {"pair_cells": 24, "weight_cells": 800})
    run = _run(base, out, "--offline")
    assert run.exit_code == 3, (run.output, run.exception)
    summary = _json(out / "workflow_summary.json")
    qc = summary["observation_qc"]
    assert summary["scored_scope"]["selection"] == "bounded_panel"
    assert qc["input_observations"] == summary["scoring_panel"]["n_selected"]
    assert (
        qc["eligible_observations"]
        == summary["scored_scope"]["observations"]
        < qc["input_observations"]
    )
    assert (out / summary["full_analyze_args"]).is_file() and (
        out / "analysis/permutation_mi.tsv"
    ).is_file()


def _audit_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


@pytest.mark.parametrize(
    ("component", "folder", "residues"), [("A", "ATPase", "MEMH"), ("C", "guide-RNA", "ACGU")]
)
@pytest.mark.parametrize("wrong_role", [False, True])
def test_folder_run_keeps_finite_assembly_alternatives(
    tmp_path, component, folder, residues, wrong_role
):
    base, out = tmp_path / "input", tmp_path / "results"
    _folders(base, 4)
    expected = {name: [f"demo:{name}"] for name in "ABC"}
    (base / "project.json").write_text(json.dumps({"expected_orthogroups": expected}))
    for name, subfolder in zip("ABC", ("ATPase", "Partner", "guide-RNA"), strict=True):
        source = base / subfolder / "family.sto"
        lines = []
        for line in source.read_text().splitlines():
            if " LOCUS " in line:
                lines.append(f"#=GS {line.split()[1]} ORTHOGROUP demo:{name}")
            lines.append(line)
        if subfolder == folder:
            lines[-1:-1] = [
                "#=GS anonymous-row ASSEMBLY GCA_000000001.1",
                "#=GS anonymous-row ASSEMBLY GCA_000000002.1",
                f"#=GS anonymous-row ORTHOGROUP demo:{'unrelated' if wrong_role else name}",
                f"anonymous-row {residues}",
            ]
        source.write_text("\n".join([*lines, ""]))
    invoked = _run(base, out, "--offline")
    assert invoked.exit_code == 0, (invoked.output, invoked.exception)
    summary = _json(out / "workflow_summary.json")
    groups = {r["genome_id"]: r for r in _audit_rows(out / summary["group_audit"])}
    healthy = {"GCA_000000003.1", "GCA_000000004.1"}
    possible = {"GCA_000000001.1", "GCA_000000002.1"}
    selected = {r["observation_id"] for r in _audit_rows(out / summary["matches"])}
    assert selected == (healthy | possible if wrong_role else healthy)
    assert (
        summary["scored_scope"]["observations"]
        == len(selected)
        == summary["counts"]["resolved_groups"]
    )
    displayed = (out / summary["report"]).read_text(encoding="utf-8")
    assert f"{len(selected)} of 4 groups resolved" in displayed
    candidates = [
        r for r in _audit_rows(out / summary["evidence"]) if r["sequence_id"] == "anonymous-row"
    ]
    assert {r["genome_id"] for r in candidates} == possible
    assert not any(r["selected"] == "1" for r in candidates)
    if wrong_role:
        assert all(r["role_status"] == "orthogroup_mismatch" for r in candidates)
    else:
        assert all(
            groups[g]["status"] == "ambiguous"
            and groups[g][component.lower() + "_candidates"] == "2"
            for g in possible
        )


@pytest.mark.parametrize("protein", [False, True], ids=["rna", "protein"])
@pytest.mark.parametrize("local_refseq", [False, True], ids=["local_gca", "local_gcf"])
@pytest.mark.parametrize("identical", [False, True], ids=["distinct", "verified_alias"])
def test_folder_run_reconciles_reported_and_local_assemblies(
    tmp_path, monkeypatch, protein, local_refseq, identical
):
    base, out = tmp_path / "input", tmp_path / "result"
    _folders(base, 3)
    gca, gcf = "GCA_000000001.1", "GCF_000091123.2"
    local, reported = (gcf, gca) if local_refseq else (gca, gcf)
    original = ("P19001" if protein else "AB000001.1") + "/11-14"
    identifier = ("CHECK_YEAST" if protein else "URS000000000A_91") + "/11-14"
    source = base / ("ATPase" if protein else "guide-RNA") / "family.sto"
    source.write_text(
        "\n".join(
            [
                *(
                    line.replace(original, identifier).replace(gca, local)
                    if original in line
                    else line
                    for line in source.read_text().splitlines()
                    if not (original in line and " LOCUS " in line)
                ),
                "",
            ]
        )
    )
    calls = []

    def nucleotide_response(url):
        calls.append(url)
        if "/genome-locations/" in url:
            assert not protein
            return {
                "count": 1,
                "next": None,
                "results": [
                    {
                        "chromosome": "chrI",
                        "start": 24,
                        "end": 28,
                        "strand": 1,
                        "assembly": {"gca_accession": reported},
                    }
                ],
            }
        assert f"/{gcf}/dataset_report" in url
        return {
            "reports": [
                {
                    "accession": gcf,
                    "assembly_info": {
                        "paired_assembly": {
                            "accession": gca,
                            "refseq_genbank_are_different": not identical,
                        }
                    },
                }
            ]
        }

    def protein_response(url):
        assert protein
        calls.append(url)
        query = parse_qs(urlsplit(url).query)["query"][0]
        if "/proteomes/" in url:
            assert "UP123456789" in query
            rows = [{"id": "UP123456789", "genomeAssembly": {"assemblyId": reported}}]
        elif "id:CHECK_YEAST" in query:
            rows = [
                {
                    "primaryAccession": "P56789",
                    "uniProtkbId": "CHECK_YEAST",
                    "organism": {"taxonId": 91},
                    "uniProtKBCrossReferences": [{"database": "Proteomes", "id": "UP123456789"}],
                }
            ]
        else:
            pytest.fail(f"unexpected protein lookup: {query}")
        return {"results": rows}, {"x-uniprot-release": "fixture", "x-uniprot-release-date": "test"}

    monkeypatch.setattr(nucleotide, "_request_json", nucleotide_response)
    monkeypatch.setattr(uniprot, "_request", protein_response)
    result = _run(base, out)
    assert result.exit_code == 0, (result.output, result.exception)
    workflow = _json(out / "workflow_summary.json")
    assert workflow["providers"]["status"] == "complete"
    assert any("dataset_report" in url for url in calls)
    selected = {r["observation_id"] for r in _audit_rows(out / workflow["matches"])}
    assert selected == ({"GCA_000000002.1", "GCA_000000003.1"} | ({gca} if identical else set()))
    assert (
        workflow["scored_scope"]["observations"]
        == len(selected)
        == workflow["counts"]["resolved_groups"]
    )
    records = [r for r in _audit_rows(out / workflow["evidence"]) if r["sequence_id"] == identifier]
    if identical:
        (resolved,) = [r for r in records if r["selected"] == "1"]
        assert (resolved["assembly_id"], resolved["genome_id"]) == (reported, gca)
        assert ["ASSEMBLY", local] in json.loads(resolved["stockholm_annotations"])
        assert set(json.loads(resolved["assembly_aliases"])) == {gca, gcf}
        witness = next(
            x
            for x in json.loads(resolved["cross_references"])
            if x.get("source") == "NCBI Datasets"
        )
        assert witness["reported_assembly"] == gcf
        assert witness["paired_assembly"] == {
            "accession": gca,
            "refseq_genbank_are_different": False,
        }
        if local_refseq:
            replay = tmp_path / "cached"
            monkeypatch.setattr(nucleotide, "_request_json", _no_request)
            monkeypatch.setattr(uniprot, "_request", _no_request)
            rerun = _run(base, replay, "--offline")
            assert rerun.exit_code == 0, (rerun.output, rerun.exception)
            assert (out / workflow["matches"]).read_bytes() == (
                replay / workflow["matches"]
            ).read_bytes()
            cached = next(
                r
                for r in _audit_rows(replay / workflow["evidence"])
                if r["sequence_id"] == identifier and r["selected"] == "1"
            )
            for field in ("assembly_id", "genome_id", "assembly_aliases", "stockholm_annotations"):
                assert cached[field] == resolved[field]
    else:
        assert not any(r["selected"] == "1" for r in records)
        assert {r["genome_id"] for r in records} == {gca, gcf}


def test_project_qc_replays_from_published_analysis_and_reports_blockers(tmp_path, monkeypatch):
    base = tmp_path / "input"
    _folders(base)
    policy = base / "qc.json"
    policy.write_text(json.dumps({"rows": {"A": {"min_canonical": 5}}}))
    (base / "project.json").write_text(
        json.dumps(
            {
                "analysis": {
                    "qc_policy": "qc.json",
                    "weight_policy": "component",
                    "weight_component": "B",
                    "memory_mib": 8,
                    "permutations": 0,
                }
            }
        )
    )
    out = tmp_path / "blocked"
    result = _run(base, out, "--offline")
    assert result.exit_code == 3, (result.output, result.exception)
    summary = _json(out / "workflow_summary.json")
    assert summary["status"] == "insufficient_data" and not summary["ready_for_analysis"]
    assert summary["analyze_args"] is None and (out / summary["matches"]).is_file()
    plan_file = out / summary["qc_plan"]
    plan = _json(plan_file)
    assert _json(plan_file.parent / plan["diagnostics"]["summary"])["retained_observations"] == 0
    assert (plan_file.parent / plan["inputs"]["A"]["path"]).is_file()
    assert summary["qc_plan"] in (out / "report.html").read_text(encoding="utf8")

    policy.write_text(json.dumps({"rows": {"A": {"min_canonical": 2}}}))
    complete = tmp_path / "complete"
    run = _run(base, complete, "--offline")
    assert run.exit_code == 0, (run.output, run.exception)
    previous = complete / "analysis"
    scoring = _json(previous / "run_summary.json")
    assert scoring["weighting"]["policy"] == "component"
    assert scoring["weighting"]["component"] == "B"
    replay = _json(previous / "analysis_replay.json")["arguments"]
    assert all(".triplet-in-progress" not in value for value in replay)
    monkeypatch.chdir(previous)
    repeat = tmp_path / "replay"
    result = CliRunner().invoke(app, [*replay, "--out", str(repeat)])
    assert result.exit_code == 0, (result.output, result.exception)
    for name in ("qc_rows.tsv", "weights.tsv", "site_weight_support.tsv", "dca_ab.tsv"):
        assert (repeat / name).read_bytes() == (previous / name).read_bytes()
