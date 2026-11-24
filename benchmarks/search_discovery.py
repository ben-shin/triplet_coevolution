"""Small generated local databases. Exercise real executables, alignments and verified cache reuse."""

import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

from Bio.Seq import Seq

from triplet_coevolution.search.discovery import discover_component


def exercise(root):
    rng = random.Random(773)
    report = {}

    def perform(label, comp, ref, wanted=None):
        start = perf_counter()
        native = discover_component(
            "A", comp, ref, root / label, wanted_ids=wanted, cache_dir=root / "cache"
        )
        report[label] = {
            "seconds": perf_counter() - start,
            "accepted": len(native.hits),
            "aligned": native.manifest["aligned_domains"],
            "width": native.msa.n_sites,
            "cache_hit": native.manifest["search_cache_hit"],
            "versions": {p["name"]: p["version"] for p in native.programs},
            "key": native.manifest["raw_key"],
            "insert_columns": native.manifest["alignment"].get("retained_insertion_columns"),
        }
        return native

    aa = "".join(rng.choices("ACDEFGHIKLMNPQRSTVWY", k=90))

    def input_set(mol, ref, dbrows):
        refp = root / f"seed-{mol}.fa"
        db = root / f"database-{mol}.fa"
        refp.write_text(">query\n" + ref + "\n", encoding="ascii")
        db.write_text(
            "".join(">" + key + "\n" + v + "\n" for key, v in dbrows.items()), encoding="ascii"
        )
        return {
            "reference": refp,
            "reference_id": "query",
            "database": db,
            "molecule": mol,
            "evalue": 0.1,
            "timeout_seconds": 120,
        }

    db = {
        "single.1": aa,
        "dup|2": "L" * 8 + aa + "GGGGGGGGGGG" + aa,
        "insert": aa[:27] + "SS" + aa[27:],
        "decoy": "Y" * 94,
    }
    c = input_set("protein", aa, db)
    jc = {**c, "backend": "jackhmmer", "iterations": 4, "memory_mb": 1024}
    r = perform("jack", jc, aa, {"insert", "dup|2/110-199"})
    assert {"dup|2/9-98", "dup|2/110-199"} <= r.hits.keys()
    assert (
        set(r.msa.ids) == {"query", "insert", "dup|2/110-199"}
        and r.msa.sequences[r.msa.ids.index("insert")].replace("-", "") == db["insert"]
    )
    raw = json.loads((root / "jack/raw/manifest.json").read_text())
    report["jack"]["rounds"] = len(raw["result"]["iterations"])
    report["jack"]["converged"] = raw["result"]["converged"]
    r2 = perform("jack_reuse", jc, aa, {"single.1"})
    assert r2.manifest["search_cache_hit"] and set(r2.msa.ids) == {"query", "single.1"}
    x = perform("jack_seed_change", {**jc, "seed": 49}, aa, {"single.1"})
    assert not x.manifest["search_cache_hit"]
    with c["database"].open("a") as f:
        f.write(">new\n" + "C" * 88 + "\n")
    x = perform("jack_db_change", jc, aa, {"single.1"})
    assert not x.manifest["search_cache_hit"]

    def build(tool, model, sto, kind, reference):
        ss = "(" * 12 + "." * (len(reference) - 24) + ")" * 12
        sto.write_text(
            "# STOCKHOLM 1.0\ns1 "
            + reference
            + "\ns2 "
            + reference
            + "\n#=GC SS_cons "
            + ss
            + "\n//\n",
            encoding="ascii",
        )
        args = [tool] + (["--" + kind] if tool == "hmmbuild" else []) + [str(model), str(sto)]
        subprocess.run(args, check=True, capture_output=True, timeout=90)

    model = root / "protein.hmm"
    build("hmmbuild", model, root / "protein.sto", "amino", aa)
    aligned = perform(
        "hmmalign",
        {**c, "backend": "hmmer", "profile": model, "alignment_backend": "hmmalign"},
        aa,
        {"insert"},
    )
    assert aligned.msa.sequences[aligned.msa.ids.index("insert")].replace("-", "") == db["insert"]
    psi = perform("psiblast", {**c, "backend": "psiblast", "iterations": 2}, aa, {"single.1"})
    assert "single.1" in psi.msa.ids
    rna = "".join(rng.choices("ACGU", k=64))
    rna += str(Seq(rna[:12]).reverse_complement_rna())
    rev = str(Seq(rna).reverse_complement_rna())
    c = input_set("rna", rna, {"plus": rna, "reverse": rev, "noise": "A" * 84})
    cm = root / "rna.cm"
    build("cmbuild", cm, root / "rna.sto", "rna", rna)
    subprocess.run(
        [
            "cmcalibrate",
            "--seed",
            "42",
            "--cpu",
            "1",
            "--noforecast",
            "-L",
            "0.05",
            "--tailp",
            ".1",
            str(cm),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    reverse_domain = f"reverse/{len(rna)}-1"
    cov = perform(
        "cmalign",
        {**c, "backend": "infernal", "profile": cm, "alignment_backend": "cmalign"},
        rna,
        {reverse_domain},
    )
    assert (
        cov.hits[reverse_domain].strand == "-"
        and cov.msa.sequences[cov.msa.ids.index(reverse_domain)].replace("-", "") == rna
    )
    return report


if __name__ == "__main__":
    for tool in (
        "jackhmmer",
        "hmmsearch",
        "hmmalign",
        "hmmbuild",
        "cmbuild",
        "cmcalibrate",
        "cmsearch",
        "cmalign",
        "psiblast",
        "makeblastdb",
    ):
        if not shutil.which(tool):
            raise SystemExit("Missing optional executable " + tool)
    with tempfile.TemporaryDirectory(prefix="triplet-discovery-benchmark-") as tmp:
        print(json.dumps(exercise(Path(tmp)), indent=2))
