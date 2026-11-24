"""Legacy PDB fixed-column polymer records and REMARK 350 transforms."""

import re
from collections import defaultdict

from Bio.Data.PDBData import protein_letters_3to1_extended

from triplet_coevolution.structure._coordinates import (
    NUCLEOTIDES,
    as_transform,
    letter,
    read_atom,
    residue,
)
from triplet_coevolution.structure.types import IDENTITY, Chain, resolve_selection


def _label(chain):
    return chain.strip() or "_"


def _labels(text):
    return [_label("" if s.strip() == "NULL" else s) for s in text.split(",") if s.strip()]


def _assemblies(lines, assembly):
    """Each BIOMT matrix applies only to the immediately preceding chain list."""
    groups = []
    active = ()
    for line in lines:
        if not line.startswith("REMARK 350"):
            continue
        text = line[10:].strip()
        if text.startswith("BIOMOLECULE:"):
            active = tuple(x.strip() for x in text.split(":", 1)[1].split(","))
            continue
        if text.startswith(
            ("APPLY THE FOLLOWING TO CHAINS:", "IN ADDITION APPLY THE FOLLOWING TO CHAINS:")
        ):
            groups.append((active, _labels(text.split(":", 1)[1]), {}))
        elif text.startswith("AND CHAINS:"):
            if not groups or groups[-1][0] != active:
                raise ValueError("REMARK 350 chain continuation without initial chain list")
            groups[-1][1].extend(_labels(text.split(":", 1)[1]))
        elif text.startswith("BIOMT"):
            if len(fields := text.split()) != 6 or not re.fullmatch(r"BIOMT[123]", fields[0]):
                raise ValueError("malformed REMARK 350 BIOMT record")
            if not groups or groups[-1][0] != active:
                raise ValueError("REMARK 350 BIOMT lacks an assembly and chain list")
            if assembly not in active:
                continue
            pos = int(fields[0][-1])
            op = groups[-1][2].setdefault(fields[1], {})
            if pos in op:
                raise ValueError(f"duplicate BIOMT{pos} component for operator {fields[1]}")
            op[pos] = list(map(float, fields[2:]))
    result = []
    for ids, chains, ops in groups:
        if assembly in ids:
            if not ops:
                raise ValueError(f"assembly {assembly} chain group lacks BIOMT operators")
            for key, rows in ops.items():
                if set(rows) != {1, 2, 3}:
                    raise ValueError(f"incomplete BIOMT operator {key}")
                matrix = as_transform([rows[1], rows[2], rows[3], [0, 0, 0, 1]])
                result.extend((chain, (key,), matrix) for chain in chains)
    return result


def _entities(lines, name):
    text = " ".join(line[10:].strip() for line in lines if line.startswith(name))
    out = defaultdict(dict)
    eid = None
    for field in text.split(";"):
        key, sep, value = field.partition(":")
        if sep:
            key, value = key.strip(), value.strip()
            if key == "MOL_ID":
                eid = value
            elif eid:
                out[eid][key] = value
    return out


def _location(raw):
    return int(raw[0]), raw[1]


def _place(sequence, observed, names, molecule, parents):
    """Unique exact monotone embeddings into SEQRES, without gap-length guesses."""

    def token(mon):
        base = letter(mon, molecule, parents)
        return mon if base == ("X" if molecule == "protein" else "N") else base

    full = [token(mon) for mon in sequence]
    pieces = [{token(mon) for mon in names[k]} for k in observed]

    def scan(indices, step, index):
        sites = {}
        for i in indices:
            while 0 <= index < len(full) and full[index] not in pieces[i]:
                index += step
            if not 0 <= index < len(full):
                raise ValueError("PDB coordinate residues cannot be placed in full SEQRES order")
            sites[i] = index
            index += step
        return sites

    forward = scan(range(len(pieces)), 1, 0)
    reverse = scan(reversed(range(len(pieces))), -1, len(full) - 1)
    return {forward[i]: key for i, key in enumerate(observed) if forward[i] == reverse[i]}


def _molecule(sequence, parents):
    monomers = {parents.get(m, m) for m in sequence} - {"UNK"}
    protein = monomers & protein_letters_3to1_extended.keys()
    nucleic = monomers & NUCLEOTIDES.keys()
    if protein and not nucleic:
        return "protein"
    if nucleic and not protein:
        deoxy = {"DA", "DC", "DT", "DG", "DI", "DU"}
        ribo = {"A", "G", "C", "U", "I"}
        if monomers & deoxy and not monomers & ribo:
            return "dna"
        if monomers & ribo and not monomers & deoxy:
            return "rna"
    return None


def _element(atom_name, text):
    if text.strip():
        return text.strip().upper()
    if not atom_name.strip():
        raise ValueError("missing PDB atom name")
    # A leading space or digit implies a one-letter element. Thus protein
    # ' CA ' is carbon and a left-justified 'CA  ' is calcium.
    if atom_name[0].isspace() or atom_name[0].isdigit():
        return atom_name.strip().lstrip("0123456789")[:1].upper()
    if match := re.match(r"[A-Za-z]{1,2}", atom_name):
        return match.group().upper()
    raise ValueError(f"could not infer element for PDB atom {atom_name!r}")


def parse_pdb(stream, selected, model):
    lines = [line.rstrip("\r\n") for line in stream]
    declared_models = [int(s[10:14]) for s in lines if s.startswith("MODEL ")]
    if len(set(declared_models)) != len(declared_models):
        raise ValueError("duplicate deposited PDB MODEL number")
    models = set(declared_models) or (
        {1} if any(s.startswith(("ATOM  ", "HETATM")) for s in lines) else set()
    )
    assemblies = {
        key.strip()
        for s in lines
        if s.startswith("REMARK 350") and s[10:].strip().startswith("BIOMOLECULE:")
        for key in s.split(":", 1)[1].split(",")
    }
    selected, model = resolve_selection(selected, model, models, assemblies)
    copies = _assemblies(lines, selected)
    compounds = _entities(lines, "COMPND")
    sources = _entities(lines, "SOURCE")
    entities = {}
    for eid, fields in compounds.items():
        for label in _labels(fields.get("CHAIN", "")):
            if label in entities:
                raise ValueError(f"multiple PDB entities declare chain {label!r}")
            entities[label] = eid
    sequences = defaultdict(list)
    sizes = {}
    parents = {}
    site_mods = defaultdict(dict)
    missing = defaultdict(dict)
    for row in lines:
        s = row.ljust(80)
        if s.startswith("SEQRES"):
            label = _label(s[11])
            count = int(s[13:17])
            if sizes.setdefault(label, count) != count:
                raise ValueError(f"conflicting SEQRES length for chain {label}")
            sequences[label].extend(s[19:70].split())
        elif s.startswith("MODRES"):
            label, key, mon, parent = (
                _label(s[16]),
                (s[18:22].strip(), s[22].strip()),
                s[12:15].strip(),
                s[24:27].strip(),
            )
            site_mods[label, key][mon] = parent
            if mon in parents and parents[mon] != parent:
                raise ValueError(f"conflicting PDB MODRES parents for {mon}")
            parents[mon] = parent
        elif s.startswith("REMARK 465"):
            auth = s[21:26].strip()
            number = s[11:14].strip()
            if re.fullmatch(r"-?\d+", auth) and (not number or number == str(model)):
                missing[_label(s[19])][(auth, s[26].strip())] = s[15:18].strip()
    if not sequences:
        raise ValueError("PDB requires a complete SEQRES polymer sequence for mapping")
    if any(len(seq) != sizes[label] for label, seq in sequences.items()):
        raise ValueError("PDB SEQRES count disagrees with full polymer record length")
    sites = defaultdict(lambda: defaultdict(list))
    names = defaultdict(lambda: defaultdict(set))
    warns = []
    active = None if declared_models else 1
    omitted = set()
    for row in lines:
        s = row.ljust(80)
        if s.startswith("MODEL "):
            active = int(s[10:14])
        elif s.startswith("ENDMDL"):
            active = None
        elif active == model and s.startswith(("ATOM  ", "HETATM")):
            label, mon = _label(s[21]), s[17:20].strip()
            key = (s[22:26].strip(), s[26].strip())
            if label not in sequences:
                if s.startswith("ATOM  "):
                    omitted.add(label)
                continue
            if (
                s.startswith("HETATM")
                and mon not in sequences[label]
                and mon not in site_mods[label, key]
            ):
                continue
            names[label][key].add(mon)
            atom = read_atom(
                s[12:16],
                _element(s[12:16], s[76:78]),
                (s[30:38], s[38:46], s[46:54]),
                s[54:60],
                s[16],
            )
            if atom is not None:
                sites[label][key].append((mon, atom))
    if omitted:
        warns.append(f"Excluded coordinate chains without SEQRES: {sorted(omitted)}.")
    chains = {}
    for label, seq in sequences.items():
        molecule = _molecule(seq, parents)
        if not molecule:
            warns.append(
                f"Excluded PDB chain {label} with undetermined or mixed polymer chemistry."
            )
            continue
        loc_names = names[label]
        observed = list(loc_names)
        if missing[label] and observed == sorted(observed, key=_location):
            for key, mon in missing[label].items():
                if key in loc_names:
                    raise ValueError(f"PDB position both missing and observed at {label}/{key}")
                loc_names[key].add(mon)
            observed = sorted(loc_names, key=_location)
        elif missing[label]:
            warns.append(
                f"Chain {label} has nonmonotone author numbering. Unresolved REMARK 465 positions could not be ordered safely."
            )
        positions = _place(seq, observed, loc_names, molecule, parents)
        if len(positions) != len(observed):
            warns.append(
                f"Chain {label}: {len(observed) - len(positions)} author positions have ambiguous SEQRES placement and were withheld."
            )
        residues = []
        for i, mon in enumerate(seq):
            key = positions.get(i)
            residues.append(
                residue(
                    i + 1, (mon,), molecule, parents, key or (None, ""), sites[label].get(key, [])
                )
            )
        eid = entities.get(label, "pdb-chain:" + label)
        if label not in entities:
            warns.append(
                f"Chain {label} has a synthetic entity identifier because COMPND lacks a MOL_ID/CHAIN link."
            )
        source = sources[eid]
        chains[label] = Chain(
            "",
            label,
            "" if label == "_" else label,
            eid,
            molecule,
            source.get("ORGANISM_SCIENTIFIC", ""),
            source.get("ORGANISM_TAXID", ""),
            (),
            IDENTITY,
            tuple(residues),
        )
    entry = next((line[62:66].strip() for line in lines if line.startswith("HEADER")), "")
    method = " ".join(line[10:].strip() for line in lines if line.startswith("EXPDTA"))
    return entry, method, chains, models, assemblies, copies, warns
