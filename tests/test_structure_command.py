"""Exercise published structure-map inputs, coordinate joins, abstention and file integrity."""

import csv
import gzip
import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from typer.testing import CliRunner

from triplet_coevolution.cli import app
from triplet_coevolution.core.coordinates import ReferenceCoordinates
from triplet_coevolution.inference.mi import MIBlock, TripletMI
from triplet_coevolution.io.score_store import PAIRS, from_results, write_score_store
from triplet_coevolution.structure.workflow import run_structure

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/structure"


def setup_config(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "inputs")
    path = tmp_path / "inputs/config.json"
    return path, json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_example_geometry_scripts_and_manifest(tmp_path):
    out = tmp_path / "published"
    result = CliRunner().invoke(
        app, ["structure-map", "--config", str(EXAMPLE / "config.json"), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    data = json.loads((out / "structure_summary.json").read_text())
    assert data["status"] == "complete" and data["source_kind"] == "synthetic"
    assert [r["copy_id"] for r in data["roles"].values()] == ["P@I", "Q@I", "R@T"]
    assert [r["observable"] for r in data["interfaces"].values()] == [12, 16, 12]
    assert all(row["source"].startswith("synthetic ") for row in table(out / "labels.tsv"))
    assert all(
        not row["contact"] for row in table(out / "labels.tsv") if row["status"] == "unresolved"
    )
    triples = table(out / "triple_geometry.tsv")
    assert [r["geometry"] for r in triples] == ["triangle", "one_close_edge", "unresolved"]
    assert [
        float(triples[0][k]) for k in ("ab_distance", "ac_distance", "bc_distance", "span")
    ] == [4, 3, 5, 5]
    assert triples[-1]["span"] == ""
    assert all(
        hashlib.sha256((out / path).read_bytes()).hexdigest() == digest
        for path, digest in data["output_sha256"].items()
    )
    assert (out / "report.html").read_text().count("href=") >= 10
    index = json.loads((out / "viewer/viewer.json").read_text())
    lookup = {}
    for c in "ABC":
        block = MMCIF2Dict(out / "viewer" / (c + ".cif"))
        lookup[c] = list(
            zip(
                block["_atom_site.auth_asym_id"],
                block["_atom_site.auth_seq_id"],
                block["_atom_site.pdbx_PDB_ins_code"],
                strict=True,
            )
        )
        if c == "C":
            assert set(map(float, block["_atom_site.Cartn_y"])) == {3}
    for r in index["pairs"]:
        for s in (r["x"], r["y"]):
            assert (s["auth_chain"], s["auth_number"], s["insertion"] or "?") in lookup[
                s["component"]
            ]
    marker = (out / "structure_summary.json").read_bytes()
    again = CliRunner().invoke(
        app, ["contacts", "--config", str(EXAMPLE / "config.json"), "--out", str(out)]
    )
    assert again.exit_code == 1 and (out / "structure_summary.json").read_bytes() == marker


def test_sampling_and_honest_metric_partial(tmp_path):
    path, cfg = setup_config(tmp_path)
    out = tmp_path / "p"
    data = run_structure(path, out)
    repeat = tmp_path / "q"
    run_structure(path, repeat)
    assert (out / "sampled_labels.tsv").read_bytes() == (repeat / "sampled_labels.tsv").read_bytes()
    avail = table(out / "distant_labels.tsv")
    chosen = table(out / "sampled_labels.tsv")
    assert {tuple(r.values()) for r in chosen}.issubset(tuple(r.values()) for r in avail)
    assert data["sensitivity"]["negative_sample"]["sampled_distant"] == {"AB": 3, "AC": 4, "BC": 3}
    cfg.update(metric="ca", close=4)
    result = CliRunner().invoke(
        app, ["contacts", "--config", str(write(path, cfg)), "--out", str(tmp_path / "ca")]
    )
    assert result.exit_code == 3, result.output
    for pair, r in json.loads((tmp_path / "ca/structure_summary.json").read_text())[
        "interfaces"
    ].items():
        if "C" in pair:
            assert not r["observable"] and r["prevalence"] is None


def test_choice_required_ambiguous_copy_and_valid_overrides(tmp_path):
    path, cfg = setup_config(tmp_path)
    cif = path.parent / "mixed.cif"
    content = cif.read_text().replace("1 T R", "1 T R\n2 I P,Q,R")
    cif.write_text(content)
    data = run_structure(path, tmp_path / "abstain")
    assert data["status"] == "selection_required" and data["available_assemblies"] == ("1", "2")
    assert not (tmp_path / "abstain/distances.tsv").exists()
    cif.write_text(content.replace("1 T R", "1 T R\n1 I R"))
    cfg.update(assembly="1")
    data = run_structure(write(path, cfg), tmp_path / "copies")
    assert (
        data["roles"]["C"]["status"] == "ambiguous_chain"
        and data["interfaces"]["AB"]["observable"] == 12
    )
    cfg["references"]["C"]["copy"] = "R@T"
    data = run_structure(write(path, cfg), tmp_path / "explicit")
    assert data["status"] == "complete" and data["roles"]["C"]["copy_id"] == "R@T"


def _store_for_example(path, cfg, symbols=("AC", "FW", "AU")):
    maps = tuple(
        ReferenceCoordinates(c, r["id"], (1, 2), (1, 2), tuple(s))
        for (c, r), s in zip(cfg["references"].items(), symbols, strict=True)
    )
    raw = np.array([[0.1, 0.3], [0.2, 0.4]])
    n = np.full((2, 2), 8)
    mi = TripletMI(
        *(MIBlock(*pair, raw, raw - 0.25, n, n.astype(float)) for pair in PAIRS), 8, 8.0, False, 2
    )
    store = from_results(
        maps,
        mi,
        provenance={
            "software_version": "fixture",
            "inputs": {k: {"sha256": "0" * 64} for k in ("A", "B", "C", "matches")},
            "ranking": {"mi": "mi_apc"},
        },
    )
    write_score_store(path.parent / "scores", store)


def test_score_joins_and_standalone_replay(tmp_path):
    path, cfg = setup_config(tmp_path)
    _store_for_example(path, cfg)
    cfg.update(scores="scores")
    cfg.pop("negative_ratio")
    data = run_structure(write(path, cfg), tmp_path / "s")
    assert data["inputs"]["scores"]["viewer_score"] == "mi_apc"
    pairs = json.loads((tmp_path / "s/viewer/viewer.json").read_text())["pairs"]
    assert all(r["score"] == pytest.approx(0.15) for r in pairs[::4])
    assert [r["x"]["insertion"] for r in pairs[::4]] == ["A"] * 3
    direct = tmp_path / "b"
    ans = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--scores",
            str(path.parent / "scores"),
            "--labels",
            str(tmp_path / "s/labels.tsv"),
            "--out",
            str(direct),
        ],
    )
    assert ans.exit_code == 0, ans.output
    assert (direct / "benchmark.json").read_bytes() == (
        tmp_path / "s/benchmark_all_measurable.json"
    ).read_bytes()
    cfg["references"]["B"]["id"] = "other"
    cfg["references"]["B"]["row_id"] = "partner_fragment"
    with pytest.raises(ValueError, match="reference"):
        run_structure(write(path, cfg), tmp_path / "wrongid")
    assert not (tmp_path / "wrongid").exists()


def test_cached_fetch_with_integrity_and_offline_replay(tmp_path, monkeypatch):
    path, cfg = setup_config(tmp_path)
    cif = (path.parent / cfg.pop("structure")).read_text().replace("TRIPLET", "2XYZ")
    raw = gzip.compress(cif.encode())
    cfg.update(pdb_id="2xyz", cache="cache", sha256=hashlib.sha256(raw).hexdigest())
    write(path, cfg)
    with pytest.raises(ValueError, match="absent from the cache"):
        run_structure(path, tmp_path / "absent")
    calls = []

    def fake(request, timeout):
        calls.append(request.full_url)
        stream = io.BytesIO(raw)
        stream.headers = {"ETag": "test-release"}
        return stream

    monkeypatch.setattr("triplet_coevolution.structure.source.urlopen", fake)
    a = run_structure(path, tmp_path / "download", download=True)
    b = run_structure(path, tmp_path / "cache")
    assert calls == ["https://files.rcsb.org/download/2xyz.cif.gz"]
    assert (
        a["inputs"]["coordinates"]["source"] == "download"
        and b["inputs"]["coordinates"]["source"] == "cache"
    )
    assert b["inputs"]["coordinates"]["http_version"] == {"ETag": "test-release"}
    assert (tmp_path / "download/distances.tsv").read_bytes() == (
        tmp_path / "cache/distances.tsv"
    ).read_bytes()
    (path.parent / "cache/2xyz/coordinates.cif.gz").write_bytes(b"bad")
    with pytest.raises(ValueError, match="cached structure checksum mismatch"):
        run_structure(path, tmp_path / "corrupt")


def test_functional_measurement_retains_source_and_uncertainty(tmp_path):
    path, cfg = setup_config(tmp_path)
    cfg["functional_labels"] = "assays.tsv"
    fields = [
        "assay_id",
        "source",
        "evidence_type",
        "assay_context",
        "units",
        "value",
        "uncertainty",
        "a_reference_id",
        "a_reference_pos",
        "b_reference_id",
        "b_reference_pos",
        "c_reference_id",
        "c_reference_pos",
    ]
    example = [
        "fixture",
        "test fixture",
        "simulation",
        "validation only",
        "arbitrary",
        "2.5",
        ".2",
        "enzyme_fragment",
        "2",
        "partner_fragment",
        "3",
        "",
        "",
    ]
    (path.parent / "assays.tsv").write_text("\t".join(fields) + "\n" + "\t".join(example) + "\n")
    data = run_structure(write(path, cfg), tmp_path / "f")
    row = table(tmp_path / "f/functional_evidence.tsv")[0]
    assert (
        row["source"] == "test fixture"
        and row["uncertainty"] == ".2"
        and row["a_insertion_code"] == "A"
    )
    assert row["b_mapping_status"] == "missing_coordinates"
    assert data["functional_labels"]["observations"] == 1


def test_exposure_matched_neighbor_controls_remain_distinct(tmp_path):
    path, cfg = setup_config(tmp_path)
    cfg.update(exposure_bins="exposure.tsv", hard_negative_upper=12)
    records = ["component\treference_id\treference_pos\texposure_bin"]
    for c, r in cfg["references"].items():
        for n in (1, 2, 3):
            records.append(f"{c}\t{r['id']}\t{n}\touter")
    (path.parent / "exposure.tsv").write_text("\n".join(records) + "\n")
    out = tmp_path / "hard"
    s = run_structure(write(path, cfg), out)["sensitivity"]["exposure_matched_controls"]
    assert s["positive_missing_exposure"] == 3
    assert all(r["sampled_controls"] <= r["positive"] for r in s["strata"])
    assert all(r["positive_included"] for r in s["strata"])
    for r in table(out / "hard_labels.tsv"):
        assert 1 <= int(r["x_reference_pos"]) <= 3 and 1 <= int(r["y_reference_pos"]) <= 3
        assert (float(r["distance_angstrom"]) <= 5) == (r["contact"] == "1")
        assert float(r["distance_angstrom"]) <= 12


def test_incorrect_ambiguous_remote_entry_does_not_poison_cache(tmp_path, monkeypatch):
    path, cfg = setup_config(tmp_path)
    cif = (path.parent / cfg.pop("structure")).read_text().replace("1 T R", "1 T R\n2 I P,Q,R")
    cfg.update(pdb_id="2xyz", cache="cache")

    def fake(*args, **kwargs):
        stream = io.BytesIO(gzip.compress(cif.encode()))
        stream.headers = {}
        return stream

    monkeypatch.setattr("triplet_coevolution.structure.source.urlopen", fake)
    with pytest.raises(ValueError, match="entry ID"):
        run_structure(write(path, cfg), tmp_path / "untrusted", download=True)
    assert not (path.parent / "cache/2xyz").exists()
    assert not (tmp_path / "untrusted").exists()


def test_chimerax_script_resolves_each_real_atom_id_with_mock_session(tmp_path):
    import runpy
    from types import SimpleNamespace

    out = tmp_path / "view"
    run_structure(EXAMPLE / "config.json", out)
    observed = {}

    class Open:
        def open_data(self, filename):
            path = Path(filename)
            data = MMCIF2Dict(path)
            keys = set(
                zip(
                    data["_atom_site.auth_asym_id"],
                    data["_atom_site.auth_seq_id"],
                    data["_atom_site.pdbx_PDB_ins_code"],
                    strict=True,
                )
            )
            residues = []
            for chain, n, ins in keys:
                a = SimpleNamespace(selected=False, displays=False)
                ins = "" if ins == "?" else ins
                observed[path.stem, chain, n, ins] = a
                residues.append(
                    SimpleNamespace(chain_id=chain, number=int(n), insertion_code=ins, atoms=a)
                )
            return [SimpleNamespace(residues=residues)], "ok"

    models = []
    session = SimpleNamespace(open_command=Open(), models=SimpleNamespace(add=models.extend))
    runpy.run_path(str(out / "viewer/view_chimerax.py"), init_globals={"session": session})
    assert len(models) == 3
    for r in json.loads((out / "viewer/viewer.json").read_text())["pairs"]:
        for site in (r["x"], r["y"]):
            assert observed[
                site["component"], site["auth_chain"], site["auth_number"], site["insertion"]
            ].selected


def test_predicted_and_experimental_source_mismatch(tmp_path):
    path, cfg = setup_config(tmp_path)
    cif = path.parent / cfg["structure"]
    cif.write_text(cif.read_text().replace("SYNTHETIC EXAMPLE", "THEORETICAL MODEL"))
    cfg["source_kind"] = "experimental"
    with pytest.raises(ValueError, match="source_kind conflicts"):
        run_structure(write(path, cfg), tmp_path / "contradiction")
    assert not (tmp_path / "contradiction").exists()


@pytest.mark.parametrize(
    ("role", "source_sequence", "score_symbols", "compatible"),
    [
        ("C", "ATGC", ("AC", "FW", "AU"), True),
        ("C", "AGGC", ("AC", "FW", "AU"), False),
        ("A", "ATDE", ("AU", "FW", "AU"), False),
    ],
)
def test_score_join_respects_typed_polymer_spelling(
    tmp_path, role, source_sequence, score_symbols, compatible
):
    path, cfg = setup_config(tmp_path)
    refs = path.parent / "references.fasta"
    original = "ACDE" if role == "A" else "AUGC"
    refs.write_text(refs.read_text().replace(original, source_sequence))
    # Native RNA analysis transcribes T as U. The FASTA/structure uses the original spelling.
    if compatible:
        assert run_structure(path, tmp_path / "unscored")["status"] == "complete"
    _store_for_example(path, cfg, score_symbols)
    cfg["scores"] = "scores"
    output = tmp_path / "scored"
    write(path, cfg)
    if not compatible:
        with pytest.raises(ValueError, match=f"sequence disagrees at {role} 2"):
            run_structure(path, output)
        assert not output.exists()
        return
    summary = run_structure(path, output)
    assert summary["status"] == "complete"
    rows = table(output / "distances.tsv")
    assert any(
        r["pair"] == "AC"
        and r["y_reference_pos"] == "2"
        and r["y_reference_symbol"] == "T"
        and r["mi_bits"]
        for r in rows
    )
