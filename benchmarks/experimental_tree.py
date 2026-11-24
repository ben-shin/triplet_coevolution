"""Join local GatABC observations to an external marker tree and test a registered MI panel."""

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from Bio import SeqIO
from experimental_structures import (
    HERE,
    OUT,
    PAIRS,
    PLAN,
    PRIVATE,
    check,
    protocol_digest,
    read_inputs,
)

from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA, TripletMatch
from triplet_coevolution.core.alphabets import CANONICAL
from triplet_coevolution.inference.mi import mutual_information_block
from triplet_coevolution.validation._clade_scores import common_spearman, rank_scores
from triplet_coevolution.validation.fdr import bh_adjust
from triplet_coevolution.validation.tree_model import load_tree, prepare_tree_null
from triplet_coevolution.validation.tree_topology import _parse


def unique_tip_matches(private, prep, sources, tips):
    with (prep / "matches.tsv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    proteomes = defaultdict(set)
    manifest = json.loads((HERE / "gatabc_manifest.json").read_text())
    relative = "metadata/uniprot_proteomes_cache.tsv"
    path = check(private / "data" / relative, manifest["supplied_sha256"][relative])
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for r in csv.DictReader(handle, delimiter="\t"):
            if r["proteome_id"] and r["taxon_id"]:
                proteomes[r["accession"]].add((r["proteome_id"], r["taxon_id"]))
    targets = {row["observation_id"] for row in rows}
    witnesses = defaultdict(set)
    with gzip.open(
        sources / "uniprot_proteomes.tsv.gz", "rt", encoding="utf-8", newline=""
    ) as handle:
        for r in csv.DictReader(handle, delimiter="\t"):
            if r["Proteome Id"] in targets:
                witnesses[r["Proteome Id"]].add((r["Genome assembly ID"], r["Organism Id"]))
    assemblies = {acc for values in witnesses.values() for acc, _ in values}
    equivalent = defaultdict(set)
    with (sources / "assembly_summary_refseq.txt").open(encoding="utf-8", newline="") as handle:
        next(handle)
        for r in csv.DictReader(handle, delimiter="\t"):
            if r["paired_asm_comp"] == "identical":
                refseq, genbank = r["#assembly_accession"], r["gbrs_paired_asm"]
                if refseq in assemblies or genbank in assemblies:
                    equivalent[refseq].add(genbank)
                    equivalent[genbank].add(refseq)
    by_assembly = {}
    for tip in tips:
        code = tip.removeprefix("RS_").removeprefix("GB_")
        if code in by_assembly:
            raise ValueError("two public GTDB tips have the same assembly")
        by_assembly[code] = tip
    staged = []
    reasons = Counter()
    audit = []
    for row in rows:
        obs = row["observation_id"]
        expected = {(obs, row["taxon_id"])}
        ids = [row[c + "_id"] for c in "abc"]
        acs = [s.split("|")[1] if s.startswith(("sp|", "tr|")) else s.split("/")[0] for s in ids]
        if any(proteomes[ac] != expected for ac in acs):
            status = "three_accessions_lack_unique_local_proteome_taxon_witness"
        elif len(witnesses.get(obs, ())) != 1:
            status = "missing_or_nonunique_current_proteome"
        else:
            ((acc, taxon),) = witnesses[obs]
            alternatives = {acc, *equivalent.get(acc, ())}
            matched = {by_assembly[a] for a in alternatives if a in by_assembly}
            if not re.fullmatch(r"GC[AF]_[0-9]+\.[0-9]+", acc):
                status = "missing_or_malformed_versioned_assembly"
            elif taxon != row["taxon_id"]:
                status = "current_and_frozen_taxon_disagree"
            elif len(matched) != 1:
                status = "no_exact_representative_tip" if not matched else "multiple_candidate_tips"
            else:
                (tip,) = matched
                staged.append(
                    (
                        row,
                        tip,
                        acc,
                        "direct" if by_assembly.get(acc) == tip else "identical_NCBI_pair",
                    )
                )
                continue
        reasons[status] += 1
        audit.append((obs, "", "", status))
    counts = Counter(tip for _, tip, _, _ in staged)
    accepted = []
    for row, tip, acc, method in staged:
        if counts[tip] != 1:
            method = "multiple_observations_share_tip"
        else:
            accepted.append((row, tip))
        reasons[method] += 1
        audit.append((row["observation_id"], tip, acc, method))
    return (
        accepted,
        audit,
        {
            "input": len(rows),
            "tree_tips": len(tips),
            "admitted": len(accepted),
            "dispositions": dict(reasons),
        },
    )


def induced_tree(parsed, selected, path):
    """Collapse unary paths bottom up, accumulating lengths of every retained edge."""
    parent, length, name, children = parsed
    kept = defaultdict(list)
    for i in range(len(parent) - 1, -1, -1):
        if children[i] == 0 and name[i] in selected:
            expression, extra = name[i], 0.0
        elif not kept.get(i):
            continue
        else:
            cs = kept.pop(i)
            if len(cs) == 1:
                expression, extra = cs[0]
            else:
                expression = "(" + ",".join(f"{s}:{d:.17g}" for s, d in reversed(cs)) + ")"
                extra = 0.0
        distance = (length[i] or 0) + extra
        if parent[i] >= 0:
            kept[parent[i]].append((expression, distance))
        else:
            path.write_text(expression + ";\n", encoding="utf-8")


def bounded_msa(prep, store, accepted):
    matches = tuple(
        TripletMatch(row["observation_id"], *(row[c + "_id"] for c in "abc")) for row, _ in accepted
    )
    comps, columns = [], {}
    for coord in store.maps:
        comp = coord.component
        ids = tuple(getattr(m, comp.lower() + "_id") for m in matches)
        seqs = SeqIO.to_dict(SeqIO.parse(prep / f"{comp}.fasta", "fasta"))
        rows = np.asarray([list(str(seqs[name].seq).upper()) for name in ids])
        orig = np.asarray(coord.original_columns, dtype=int)
        retained = rows[:, orig]
        canonical = np.isin(retained, list(CANONICAL["protein"]))
        eligible = np.asarray(
            [
                i
                for i in range(retained.shape[1])
                if canonical[:, i].mean() >= PLAN["tree"]["min_nonmissing"]
                and len(set(retained[canonical[:, i], i])) >= 2
            ]
        )
        n = min(PLAN["tree"]["max_columns_per_component"], len(eligible))
        if not n:
            raise ValueError(f"no eligible columns for {comp}")
        chosen = orig[eligible[np.linspace(0, len(eligible) - 1, n, dtype=int)]]
        columns[comp] = {"eligible": len(eligible), "original_1based": (chosen + 1).tolist()}
        comps.append(ComponentMSA(comp, "protein", ids, tuple("".join(r) for r in rows[:, chosen])))
    return MatchedTripletMSA(*comps, matches), columns


def scores(msa):
    pairs = []
    for left, right in PAIRS:
        pair = mutual_information_block(getattr(msa, left.lower()), getattr(msa, right.lower()))
        pairs.append(pair.mi_bits.ravel())
    return np.concatenate(pairs)


def subset(msa, indices):
    matches = tuple(msa.matches[i] for i in indices)
    parts = [part.take_ids([part.ids[i] for i in indices]) for part in msa.components]
    return MatchedTripletMSA(*parts, matches)


def evaluate(msa, tree, phyla):
    plan = PLAN["tree"]
    law = prepare_tree_null(
        msa, tree, prior_mass=1.0, frequency_policy="empirical", rate_scale=plan["branch_scale"]
    )
    rng = np.random.default_rng(plan["seed"])
    observed = scores(msa)
    draws = np.asarray([scores(law.draw(rng)) for _ in range(plan["null_replicates"])])
    if not np.isfinite(draws).all() or not np.isfinite(observed).all():
        raise ValueError("nonfinite registered tree statistic")
    exceed = (draws >= observed).sum(axis=0)
    probs = (1 + exceed) / (1 + len(draws))
    maxobs = float(observed.max())
    maxnull = draws.max(axis=1)
    sig = bh_adjust(probs)
    splits = []
    phyla = np.asarray(phyla)
    for phylum, n in sorted(Counter(phyla).items()):
        test = np.flatnonzero(phyla == phylum)
        train = np.flatnonzero(phyla != phylum)
        if n < 8 or len(train) < 24:
            continue
        a, b = scores(subset(msa, train)), scores(subset(msa, test))
        common = np.isfinite(a) & np.isfinite(b)
        ar = rank_scores(a[common], "mi_bits", 10)
        br = rank_scores(b[common], "mi_bits", 10)
        splits.append(
            {
                "phylum": str(phylum),
                "heldout": n,
                "train": len(train),
                "common_pairs": int(common.sum()),
                "spearman": common_spearman(a, b, "mi_bits"),
                "overlap_top10": len(set(ar.top) & set(br.top)),
            }
        )
    return (
        {
            "replicates": len(draws),
            "seed": plan["seed"],
            "tested": len(observed),
            "observed_max_mi_bits": maxobs,
            "simulated_max_median": float(np.median(maxnull)),
            "simulated_max_95pct": float(np.quantile(maxnull, 0.95)),
            "global_max_p": (1 + int(np.sum(maxnull >= maxobs))) / (1 + len(draws)),
            "min_marginal_p": float(probs.min()),
            "nominal_below_or_equal_05": int(np.sum(probs <= 0.05)),
            "min_bh_q": float(sig.min()),
            "bh_below_or_equal_05": int(np.sum(sig <= 0.05)),
            "clades": splits,
            "phylum_size_distribution": dict(Counter(phyla)),
        },
        observed,
        draws,
        probs,
        sig,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--private", type=Path, default=PRIVATE)
    args = parser.parse_args()
    sources = args.out / "sources"
    hashes = json.loads((HERE / "experimental_sources.json").read_text())
    for name in (
        "bac120_r232.tree.gz",
        "bac120_taxonomy_r232.tsv.gz",
        "bac120_msa_marker_info_r232.tsv",
        "uniprot_proteomes.tsv.gz",
        "assembly_summary_refseq.txt",
    ):
        check(sources / name, hashes[name]["sha256"])
    with gzip.open(sources / "bac120_r232.tree.gz", "rt") as h:
        parsed = _parse(h.read())
    tips = [name for name, n in zip(parsed[2], parsed[3], strict=True) if not n]
    if len(set(tips)) != len(tips):
        raise ValueError("duplicate tree tips")
    _, store, prep = read_inputs(args.private)
    joined, audit, summary = unique_tip_matches(args.private, prep, sources, tips)
    with (args.out / "private-tip-join.tsv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.writer(h, delimiter="\t")
        writer.writerow(("observation_id", "tip_id", "source_assembly", "decision"))
        writer.writerows(audit)
    summary["protocol_sha256"] = protocol_digest()
    summary["distinct_phyla"] = None
    if len(joined) >= PLAN["tree"]["min_observations"]:
        tip_map = {tip: row["observation_id"] for row, tip in joined}
        taxonomy = {}
        with gzip.open(sources / "bac120_taxonomy_r232.tsv.gz", "rt") as h:
            for tip, lineage in csv.reader(h, delimiter="\t"):
                if tip in tip_map:
                    (phylum,) = [x for x in lineage.split(";") if x.startswith("p__")]
                    taxonomy[tip] = phylum
        tree_path = args.out / "admitted-bac120.nwk"
        induced_tree(parsed, tip_map, tree_path)
        msa, summary["sites"] = bounded_msa(prep, store, joined)
        tree = load_tree(tree_path, [m.observation_id for m in msa.matches], tip_map=tip_map)
        phyla = [taxonomy[tip] for row, tip in joined]
        summary["distinct_phyla"] = len(set(phyla))
        summary["induced_tree_sha256"] = hashlib.sha256(tree_path.read_bytes()).hexdigest()
        summary["tree_test"], observed, draws, p, q = evaluate(msa, tree, phyla)
        np.savez_compressed(
            args.out / "private-tree-statistics.npz", observed=observed, simulated=draws, p=p, q=q
        )
    path = args.out / "tree-summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
