"""Assembly and polymer-coordinate tests with independently specified Cartesian positions."""

import gzip
import hashlib

import numpy as np
import pytest

from triplet_coevolution.structure.parser import ChoiceRequired, read_structure


def _cif(path):
    # Deposited P differs from author chain S. Entity position 2 lacks coordinates.
    # (R,I)(T) first translates x by 2, then rotates 90 degrees about z (or leaves it).
    path.write_text(
        """data_TINY
_entry.id TINY
_exptl.method 'X-RAY DIFFRACTION'
loop_
_entity_poly.entity_id
_entity_poly.type
1 'polypeptide(L)'
2 polyribonucleotide
3 polydeoxyribonucleotide
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 ALA
1 2 GLY
1 3 MSE
2 1 PSU
3 1 DG
loop_
_entity_src_gen.entity_id
_entity_src_gen.pdbx_gene_src_scientific_name
_entity_src_gen.pdbx_gene_src_ncbi_taxonomy_id
_entity_src_gen.pdbx_host_org_scientific_name
1 'synthetic source' 999 'expression host'
loop_
_chem_comp.id
_chem_comp.mon_nstd_parent_comp_id
MSE MET
PSU U
loop_
_struct_asym.id
_struct_asym.entity_id
P 1
R 2
D 3
L 4
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_strand_id
_pdbx_poly_seq_scheme.pdb_ins_code
P 1 1 ALA 8 S .
P 1 2 GLY ? S .
P 1 3 MSE 34 S A
loop_
_pdbx_struct_assembly_gen.assembly_id
_pdbx_struct_assembly_gen.oper_expression
_pdbx_struct_assembly_gen.asym_id_list
1 '(R,I)(T)' P
1 'I' R,D,L
2 I P
loop_
_pdbx_struct_oper_list.id
_pdbx_struct_oper_list.matrix[1][1]
_pdbx_struct_oper_list.matrix[1][2]
_pdbx_struct_oper_list.matrix[1][3]
_pdbx_struct_oper_list.vector[1]
_pdbx_struct_oper_list.matrix[2][1]
_pdbx_struct_oper_list.matrix[2][2]
_pdbx_struct_oper_list.matrix[2][3]
_pdbx_struct_oper_list.vector[2]
_pdbx_struct_oper_list.matrix[3][1]
_pdbx_struct_oper_list.matrix[3][2]
_pdbx_struct_oper_list.matrix[3][3]
_pdbx_struct_oper_list.vector[3]
R 0 -1 0 0 1 0 0 0 0 0 1 0
T 1 0 0 2 0 1 0 0 0 0 1 0
I 1 0 0 0 0 1 0 0 0 0 1 0
loop_
_atom_site.group_PDB
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.label_comp_id
_atom_site.label_atom_id
_atom_site.type_symbol
_atom_site.label_alt_id
_atom_site.occupancy
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
ATOM P 1 1 S 8 . ALA CA C . 1 1 0 0 1
ATOM P 1 1 S 8 . ALA CB C A .4 100 0 0 1
ATOM P 1 1 S 8 . ALA CB C B .6 2 0 0 1
ATOM P 1 1 S 8 . ALA H H . 1 3 0 0 1
ATOM P 1 1 S 8 . ALA N N . 0 40 10 0 1
HETATM P 1 3 S 34 A MSE SE SE . 1 3 1 0 1
HETATM R 2 1 X -2 . PSU P P . 1 0 3 4 1
ATOM D 3 1 Z 71 . DG P P . 1 0 0 5 1
HETATM L 4 . S 90 . ZN ZN ZN . 1 1 0 0 1
ATOM P 1 1 S 8 . ALA CA C . 1 6 0 0 7
""",
        encoding="utf-8",
    )
    return path


def test_cif_cartesian_order_copies_and_polymer_identity(tmp_path):
    path = _cif(tmp_path / "experimental.cif")
    result = read_structure(path, assembly="1", model=1)
    assert result.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.available_models == (1, 7) and result.available_assemblies == ("1", "2")
    chains = {c.copy_id: c for c in result.chains}
    assert set(chains) == {"P@R,T", "P@I,T", "R@I", "D@I"}
    protein = chains["P@R,T"]
    assert (protein.label_asym_id, protein.auth_asym_id, protein.entity_id) == ("P", "S", "1")
    assert protein.operations == ("R", "T")
    np.testing.assert_array_equal(
        protein.transform, [[0, -1, 0, 0], [1, 0, 0, 2], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    assert protein.organism == "synthetic source" and protein.taxonomy_id == "999"
    assert protein.sequence == "AGM"
    a, g, m = protein.residues
    assert a.auth_seq_id == "8" and a.altloc == "B" and a.alternatives == ("ALA:A", "ALA:B")
    atoms = {at.name: at for at in a.atoms}
    assert set(atoms) == {"CA", "CB", "H"}
    assert atoms["CA"].xyz == (0, 3, 0) and atoms["CB"].xyz == (0, 4, 0)
    assert atoms["CB"].occupancy == 0.6
    assert (g.seq_index, g.atoms, g.auth_seq_id) == (2, (), None)
    assert (m.seq_index, m.auth_seq_id, m.insertion_code, m.mon_id, m.atoms[0].element) == (
        3,
        "34",
        "A",
        "MSE",
        "SE",
    )
    assert chains["P@I,T"].residues[0].atoms[0].xyz == (3, 0, 0)
    rna = chains["R@I"]
    assert (rna.molecule, rna.sequence, rna.residues[0].mon_id) == ("rna", "U", "PSU")
    assert (chains["D@I"].molecule, chains["D@I"].sequence) == ("dna", "G")
    assert np.linalg.norm(np.subtract(atoms["CA"].xyz, rna.residues[0].atoms[0].xyz)) == 4


def test_cif_assembly_model_selection_and_gzip(tmp_path):
    path = _cif(tmp_path / "source.cif")
    compressed = tmp_path / "source.cif.gz"
    compressed.write_bytes(gzip.compress(path.read_bytes()))
    with pytest.raises(ChoiceRequired) as exc:
        read_structure(compressed)
    assert exc.value.available_models == (1, 7) and exc.value.available_assemblies == ("1", "2")
    with pytest.raises(ChoiceRequired):
        read_structure(compressed, model=1)
    with pytest.raises(ValueError, match="model 4 unavailable"):
        read_structure(compressed, model=4)
    with pytest.raises(ValueError, match="assembly 'fake' unavailable"):
        read_structure(compressed, assembly="fake")
    result = read_structure(compressed, assembly="asu", model=7)
    assert result.sha256 == hashlib.sha256(compressed.read_bytes()).hexdigest()
    p = next(c for c in result.chains if c.copy_id == "P@asu")
    assert p.residues[0].atoms[0].xyz == (6, 0, 0)
    assert not p.residues[2].atoms


def _pdb_atom(serial, name, mon, chain, seq, xyz, element, alt="", occ=1, ins="", record="ATOM"):
    return (
        f"{record:6s}{serial:5d} {name:4s}{alt:1s}{mon:>3s} {chain}{seq:4d}{ins:1s}   "
        + "".join(f"{v:8.3f}" for v in xyz)
        + f"{occ:6.2f}{20.0:6.2f}          {element:>2s}"
    )


def _pdb_chain(tmp_path, *, assembly=True):
    path = tmp_path / "tiny.pdb"
    lines = [
        "HEADER".ljust(62) + "DEMO",
        "EXPDTA    X-RAY DIFFRACTION",
        "COMPND    MOL_ID: 7; CHAIN: X;",
        "SOURCE    MOL_ID: 7; ORGANISM_SCIENTIFIC: local fixture; ORGANISM_TAXID: 99;",
        "SEQRES   1 X    3  ALA GLY MSE",
        "SEQRES   1 Y    1   DG",
        "SEQRES   1 Z    1    U",
        "MODRES DEMO MSE X  101B MET".ljust(80),
        "REMARK 465     GLY X    89",
    ]
    if assembly:
        lines.extend(
            [
                "REMARK 350 BIOMOLECULE: 3",
                "REMARK 350 APPLY THE FOLLOWING TO CHAINS: X, Y,",
                "REMARK 350                    AND CHAINS: Z",
                "REMARK 350   BIOMT1   1   1.000000  0.000000  0.000000  2.00000",
                "REMARK 350   BIOMT2   1   0.000000  1.000000  0.000000  4.00000",
                "REMARK 350   BIOMT3   1   0.000000  0.000000  1.000000  6.00000",
            ]
        )
    lines.extend(
        [
            "MODEL        2",
            _pdb_atom(1, " CA ", "ALA", "X", 71, (1, 0, 0), ""),
            _pdb_atom(2, " CB ", "ALA", "X", 71, (2, 0, 0), "C", alt="A", occ=0.4),
            _pdb_atom(3, " CB ", "ALA", "X", 71, (3, 0, 0), "C", alt="B", occ=0.6),
            _pdb_atom(4, "SE  ", "MSE", "X", 101, (4, 0, 0), "SE", record="HETATM", ins="B"),
            _pdb_atom(5, " P  ", "DG", "Y", 3, (0, 0, 4), "P"),
            _pdb_atom(6, " P  ", "U", "Z", -1, (0, 0, 9), "P"),
            _pdb_atom(7, " O  ", "HOH", "X", 35, (0, 0, 0), "O", record="HETATM"),
            "ENDMDL",
            "END",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def test_pdb_seqres_missing_residue_and_nonsequential_insertion(tmp_path):
    result = read_structure(_pdb_chain(tmp_path))
    assert result.entry_id == "DEMO" and result.model == 2 and result.assembly == "3"
    x, y, z = result.chains
    assert (x.copy_id, x.entity_id, x.organism, x.taxonomy_id) == (
        "X@1",
        "7",
        "local fixture",
        "99",
    )
    assert x.sequence == "AGM" and y.molecule == "dna" and z.molecule == "rna"
    assert [(r.auth_seq_id, r.insertion_code) for r in x.residues] == [
        ("71", ""),
        ("89", ""),
        ("101", "B"),
    ]
    assert not x.residues[1].atoms
    assert x.residues[0].atoms[0].xyz == (3, 4, 6)
    assert x.residues[0].atoms[0].element == "C"
    assert x.residues[0].altloc == "B"
    assert x.residues[2].mon_id == "MSE" and x.residues[2].atoms[0].element == "SE"


def test_pdb_abstains_on_repeat_placement_and_requires_explicit_asu(tmp_path):
    path = tmp_path / "repeated.pdb"
    path.write_text(
        "SEQRES   1 X    3  GLY GLY ALA\n"
        + "\n".join(
            (
                _pdb_atom(1, " CA ", "GLY", "X", 10, (2, 0, 0), "C"),
                _pdb_atom(2, " CA ", "ALA", "X", 99, (3, 0, 0), "C"),
            )
        ),
        encoding="ascii",
    )
    with pytest.raises(ChoiceRequired, match="explicitly select 'asu'"):
        read_structure(path)
    s = read_structure(path, assembly="asu")
    (x,) = s.chains
    assert x.copy_id == "X@asu" and x.sequence == "GGA"
    assert [bool(r.atoms) for r in x.residues] == [False, False, True]
    assert any("ambiguous" in w for w in s.warnings)
    broken = tmp_path / "fragment.pdb"
    broken.write_text(_pdb_atom(1, " CA ", "ALA", "X", 3, (0, 0, 0), "C"))
    with pytest.raises(ValueError, match="SEQRES"):
        read_structure(broken, assembly="asu")


def test_invalid_or_unknown_operator_refused(tmp_path):
    path = _cif(tmp_path / "model.cif")
    path.write_text(path.read_text().replace("(R,I)(T)", "(I,absent)"))
    with pytest.raises(ValueError, match="unavailable assembly operator"):
        read_structure(path, assembly="1", model=1)


def test_cif_numeric_range_and_duplicate_physical_copy(tmp_path):
    path = _cif(tmp_path / "operations.cif")
    text = (
        path.read_text()
        .replace("(R,I)(T)", "(1-3)")
        .replace("R 0 -1 0 0", "1 0 -1 0 0")
        .replace("T 1 0 0 2", "2 1 0 0 2")
        .replace("I 1 0 0 0", "3 1 0 0 2")
        .replace("1 'I' R,D,L", "1 '3' R,D,L")
        .replace("2 I P", "2 3 P")
    )
    path.write_text(text)
    structure = read_structure(path, assembly="1", model=1)
    assert {c.copy_id for c in structure.chains} == {"P@1", "P@2", "R@3", "D@3"}
    assert any("duplicates" in warning for warning in structure.warnings)


def test_reject_compressed_or_singular_assembly_axis(tmp_path):
    path = _cif(tmp_path / "stretch.cif")
    path.write_text(path.read_text().replace("R 0 -1 0 0", "R 0 -2 0 0"))
    with pytest.raises(ValueError, match="preserve Cartesian distances"):
        read_structure(path, assembly="1", model=1)


def test_pdb_reject_incomplete_biomt(tmp_path):
    path = _pdb_chain(tmp_path)
    path.write_text("\n".join(s for s in path.read_text().splitlines() if "BIOMT3" not in s))
    with pytest.raises(ValueError, match="incomplete BIOMT"):
        read_structure(path)


def test_microheterogeneous_entity_stays_ambiguous_across_conformers(tmp_path):
    path = _cif(tmp_path / "microheterogeneous.cif")
    path.write_text(path.read_text().replace("1 1 ALA\n", "1 1 ALA\n1 1 GLY\n", 1))
    one = read_structure(path, assembly="asu", model=1)
    seven = read_structure(path, assembly="asu", model=7)
    for result in (one, seven):
        chain = next(c for c in result.chains if c.label_asym_id == "P")
        res = chain.residues[0]
        assert chain.sequence == "XGM"
        assert res.mon_id == "ALA" and res.declared_monomers == ("ALA", "GLY")
