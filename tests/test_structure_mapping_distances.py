"""Independent sequence and distance oracles for polymer-aware structural mapping."""

import csv
import math
import os
from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest
from Bio.Align import PairwiseAligner

from benchmarks.verify_gatabc_structure import verify
from triplet_coevolution.structure.distances import (
    ensemble_distances,
    iter_distances,
    write_structure_tables,
)
from triplet_coevolution.structure.mapping import ReferenceSequence, _align, map_components
from triplet_coevolution.structure.types import IDENTITY, Atom, Chain, Residue, Structure


def _chain(name, molecule, seq, offset):
    names = {"M": "MET", "A": "ALA", "G": "GLY"}
    residues = []
    for i, s in enumerate(seq):
        x = offset + i * 3.0
        atoms = [
            Atom(
                "P" if molecule != "protein" else "CA",
                "P" if molecule != "protein" else "C",
                (x, 1.0, 0.0),
                1.0,
                "",
            ),
            Atom("H1", "H", (2.0, 0.0, 0.0), 1.0, ""),
            Atom("D1", "D", (2.0, 0.0, 0.0), 1.0, ""),
            Atom("skip", "C", (2.0, 0.0, 0.0), 0.0, ""),
        ]
        if molecule == "protein" and s != "G":
            atoms.append(Atom("CB", "C", (x, 2.0, 0.0), 1.0, ""))
        if molecule != "protein":
            atoms.append(Atom("O3'", "O", (x, 0.5, 0.0), 1.0, ""))
        mon = names[s] if molecule == "protein" else ("PSU" if s == "U" else s)
        residues.append(
            Residue(
                i + 1,
                mon,
                s,
                str((4, 17, 90)[i]) if i < 3 else str(100 + i),
                "B" if i == 1 else "",
                "",
                tuple(atoms) if molecule != "rna" or i != 2 else (),
            )
        )
    return Chain(
        name + "@1",
        name,
        name + "auth",
        "e" + name,
        molecule,
        "example species",
        "",
        ("1",),
        IDENTITY,
        tuple(residues),
    )


@pytest.fixture
def complex_fixture():
    chains = (
        _chain("reverse", "dna", "GTC", 19.0),
        _chain("pol", "protein", "MAG", 0.0),
        _chain("mod", "rna", "AUG", 4.0),
    )
    structure = Structure("demo", "mmcif", 1, (1,), "1", ("1",), chains, "0" * 64, "synthetic", ())
    refs = (
        ReferenceSequence("A", "target1", "M-AG", "protein"),
        ReferenceSequence("B", "target2", "AU.G", "rna", start=8, strand="-"),
        ReferenceSequence("C", "target3", "GTC", "dna"),
    )
    return refs, structure


def test_all_optimal_correspondences_match_independent_tracebacks():
    aligner = PairwiseAligner(
        mode="global", match_score=2, mismatch_score=-2, open_gap_score=-4, extend_gap_score=-1
    )
    sequences = ["".join(s) for n in (1, 2, 3) for s in product("AC", repeat=n)]
    for x, y in product(sequences, repeat=2):
        matches = aligner.align(x, y)
        truth = [set() for _ in x]
        for alignment in matches:
            first, second = alignment.indices
            for i, j in zip(first, second, strict=True):
                if i >= 0:
                    truth[i].add(int(j) + 1 if j >= 0 else None)
        score, sites, _, _ = _align(x, y, "dna")
        assert score == matches.score
        assert [set(options) | ({None} if gap else set()) for options, gap in sites] == truth


def test_modified_polymer_missing_position_and_direct_distances(tmp_path, complex_fixture):
    refs, st = complex_fixture
    mapping = map_components(refs, st)
    assert [mapping.roles[c].chain.copy_id for c in "ABC"] == ["pol@1", "mod@1", "reverse@1"]
    assert [s.reference_position for s in mapping.roles["B"].sites] == [8, 7, 6]
    assert mapping.roles["B"].sites[1].residue.mon_id == "PSU"
    assert mapping.roles["B"].sites[1].residue.insertion_code == "B"
    assert mapping.roles["B"].sites[2].status == "missing_coordinates"
    rows = list(iter_distances(mapping, close=4, distant=8))
    assert len(rows) == 27
    for row in rows:
        endpoints = [
            mapping.roles[c]
            .sites[mapping.references[c].positions.index(row[s + "_reference_pos"])]
            .residue
            for c, s in zip(row["pair"], "xy", strict=True)
        ]
        coords = [
            [a.xyz for a in res.atoms if a.element not in {"H", "D", "T"} and a.occupancy > 0]
            for res in endpoints
        ]
        if not all(coords):
            assert row["contact"] is None and row["category"] == "unresolved"
            continue
        expected = min(math.dist(a, b) for a in coords[0] for b in coords[1])
        assert row["heavy_distance"] == pytest.approx(expected) and row["contact"] == int(
            expected <= 4
        )
        assert row["ca_distance"] is None and row["cb_distance"] is None
    summary = write_structure_tables(tmp_path, mapping, close=4, distant=8)
    assert summary["interfaces"]["AB"]["observable"] == 6
    assert summary["interfaces"]["AB"]["scoreable"] is None
    with (tmp_path / "labels.tsv").open() as handle:
        labels = list(csv.DictReader(handle, delimiter="\t"))
    assert len(labels) == 27 and {r["contact"] for r in labels} == {"0", "1", ""}
    with pytest.raises(FileExistsError):
        write_structure_tables(tmp_path, mapping)


def test_protein_atom_definitions_and_explicit_alternatives():
    pa = _chain("one", "protein", "MAG", 0.0)
    pb = _chain("two", "protein", "GAM", 5.0)
    pc = _chain("three", "protein", "MMM", 15.0)
    st = Structure("demo", "pdb", 1, (1, 2), "1", ("1",), (pc, pa, pb), "0" * 64, "synthetic", ())
    refs = [
        ReferenceSequence(c, "r" + c, seq, "protein")
        for c, seq in zip("ABC", ("MAG", "GAM", "MMM"), strict=True)
    ]
    m = map_components(refs, st)
    row = next(
        r
        for r in iter_distances(m, metric="cb", close=5.0, distant=10.0)
        if r["pair"] == "AB" and r["x_reference_pos"] == r["y_reference_pos"] == 1
    )
    assert row["ca_distance"] == 5 and row["cb_distance"] == pytest.approx(math.sqrt(26))
    assert row["contact"] == 0
    moved = replace(
        pb,
        residues=tuple(
            replace(r, atoms=tuple(replace(a, xyz=(a.xyz[0] + 10, *a.xyz[1:])) for a in r.atoms))
            for r in pb.residues
        ),
    )
    second = map_components(refs, replace(st, model=2, chains=(pa, moved, pc)))
    first = next(ensemble_distances([m, second], metric="ca", close=5.0, distant=10.0))
    assert (first["consensus_contact"], first["status"], first["observed_close_fraction"]) == (
        None,
        "discordant",
        0.5,
    )
    assert first["min_angstrom"] == 5 and first["max_angstrom"] == 15
    with pytest.raises(ValueError, match="duplicate structural"):
        list(ensemble_distances([m, m]))


def test_copy_ambiguity_alignment_anchors_and_conflicts(complex_fixture):
    refs, st = complex_fixture
    dna = st.chains[0]
    repeat = replace(dna, residues=_chain("repeat", "dna", "GGAAC", 0.0).residues)
    refs = (*refs[:2], ReferenceSequence("C", "target3", "GGAC", "dna"))
    st = replace(st, chains=(*st.chains[1:], repeat, replace(repeat, copy_id="alternative@1")))
    m = map_components(refs, st, min_coverage=0.5)
    assert m.roles["C"].status == "ambiguous_chain"
    assert {name for name, _, _ in m.roles["C"].sites[2].alternative_copies} == {
        repeat.copy_id,
        "alternative@1",
    }
    chosen = {"C": repeat.copy_id}
    m = map_components(refs, st, chain_ids=chosen, min_coverage=0.5)
    assert m.roles["C"].sites[2].status == "ambiguous_alignment"
    assert set(m.roles["C"].sites[2].alternatives) == {3, 4}
    anchored = map_components(
        refs, st, chain_ids=chosen, min_coverage=0.5, explicit_map={"C": {3: 4}}
    )
    assert anchored.roles["C"].sites[2].residue.seq_index == 4
    with pytest.raises(ValueError, match="conflict"):
        map_components(refs, st, chain_ids=chosen, explicit_map={"C": {3: 1}})
    with pytest.raises(ValueError, match="bounds"):
        map_components(refs, st, chain_ids=chosen, explicit_map={"C": {3: 9}})
    with pytest.raises(ValueError, match="explicit chain"):
        map_components(refs, st, explicit_map={"C": {3: 4}})


def test_private_original_structure_and_all_historical_labels():
    location = os.environ.get("TRIPLET_GATABC_COMPACT_ROOT")
    if not location:
        pytest.skip(
            "Set TRIPLET_GATABC_COMPACT_ROOT to verify the supplied original 2G5H structure and labels"
        )
    result = verify(Path(location))
    assert [result["roles"][c]["chain"] for c in "ABC"] == ["A@1", "B@1", "C@1"]
    totals = {
        name: sum(v.get(name, 0) for v in result["historical_comparison"].values())
        for name in (
            "positive_agrees",
            "positive_disagrees",
            "positive_unresolved",
            "distant_agrees",
            "distant_disagrees",
            "distant_unresolved",
        )
    }
    assert list(totals.values()) == [1406, 13, 21, 1424, 0, 16]
    assert result["max_heavy_error_against_direct_atom_pairs_angstrom"] < 2e-12
