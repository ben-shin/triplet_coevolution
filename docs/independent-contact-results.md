# Experimental contact rankings in RNA polymerase and the ribosome

The two registered public sequence cohorts yielded full MI and joint mean-field DCA rankings. On the polymerase structure, APC-corrected DCA had pooled AUROC near 0.61, but its first 50 predictions included one known contact for one alpha copy and zero for the other. On the mixed ribosome, it had AUROC 0.533 and one known contact in the first 50. Contact recovery varied by interface, copy and sequence weighting. [Evaluation registration](../benchmarks/independent_evaluation_policy.json) Â· [Primary results by interface and method](../benchmarks/independent_contact_metrics.tsv)

## Primary experimental comparison

An observable pair is positive when its minimum heavy-atom distance in assembly 1, model 1 is at most 8 Ã…. The reference columns, sequence weights and score settings were frozen before the experimental coordinates were assessed. All four complete native score archives were written before coordinate analysis. The two unit-weight and two equal-phylum fits used the full registered panels. The common denominator contains observable pairs with finite scores from all four methods. Average precision (AP) and prevalence in the table are percentages. AUROC is on its usual zero-to-one scale. [Polymerase deposition](https://www.rcsb.org/structure/4YG2) Â· [Ribosome deposition](https://www.rcsb.org/structure/4YBB)

| Structure, copy | Contacts / common pairs | Prevalence | MI AP | MI-APC AP | mfDCA AP | mfDCA-APC AP | mfDCA-APC AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4YG2 A/C/D | 1,888 / 1,591,248 | 0.119% | 0.098% | 0.171% | 0.097% | 0.183% | 0.608 |
| 4YG2 B/C/D | 1,681 / 1,409,794 | 0.119% | 0.099% | 0.180% | 0.095% | 0.178% | 0.610 |
| 4YBB WB/AC/KC | 320 / 45,328 | 0.706% | 0.469% | 0.698% | 0.483% | 0.825% | 0.533 |

Across all reference positions, 4YG2 A/C/D had 2,613 contacts among 2,361,854 observable pairs and 430,761 unknowns. B/C/D had 2,350 among 2,106,242 with 686,373 unknowns. In 4YBB the corresponding counts were 424 among 55,989 with 474 unknowns. Sequence masks were fixed before assessing observability. On the intersection of observable scored positions for both polymerase copies, 1,407,800 pairs remained. Their uniform-weight DCA-APC results were AP 0.200% and AUROC 0.612 for A/C/D, and AP 0.178% and AUROC 0.610 for B/C/D.

| Copy | Uniform DCA-APC top 10, positive / unknown | Uniform DCA-APC top 50, positive / unknown | Possible precision at 50 | Interface-macro MI-APC / DCA-APC AP |
| --- | ---: | ---: | ---: | ---: |
| 4YG2 A/C/D | 0 / 1 | 1 / 5 | 2%â€“12% | 0.149% / 0.194% |
| 4YG2 B/C/D | 0 / 2 | 0 / 10 | 0%â€“20% | 0.157% / 0.127% |
| 4YBB WB/AC/KC | 1 / 0 | 1 / 0 | 2% | 1.405% / 1.690% |

The polymerase beta/beta-prime interface contributed the same 1,431 contacts among 993,048 scored, observable positions in both copies. For A/C/D, the alpha/beta-prime interface had just 27 contacts among 308,400 and DCA-APC AUROC 0.463. In the ribosome, 5Sâ€“uL5 had 67 contacts among 16,600 and 5Sâ€“uL18 had 253 among 10,800. The uL5â€“uL18 interface yielded zero among 17,928 at 8 Ã…. Its AP and AUROC are undefined. The mixed-system macro average therefore uses two interfaces. [Full results, including undefined metrics and controls](../benchmarks/independent_contact_metrics.json.gz)

## Weighting and experimental sensitivity

Equalizing total mass across GTDB phyla reduced Kish effective depth from 461 to 112.88 in polymerase and from 163 to 61.91 in the ribosome. The underlying 461 and 163 assemblies represent 40 and 29 phyla, respectively. The score support cutoffs of 231 and 82 complete observations excluded zero registered pair positions. Equal-phylum pooled DCA-APC AP/AUROC was 0.212%/0.607 and 0.182%/0.608 for polymerase A/C/D and B/C/D, and 1.142%/0.531 for the ribosome. The ribosome top 50 then contained three known contacts. The equal-phylum rows in the primary TSV report the matching MI baselines.

Restricting the control group to observable negatives at least 25 Ã… apart gave uniform DCA-APC AUROC of 0.610, 0.612 and 0.533 for the three copies in table order. The primary 8 Ã… positives and natural-negative populations remain available alongside this control. The prespecified 4, 5, 8 and 10 Ã… evaluations and uncertainties from unresolved coordinates appear in the aggregate archive. Every selected entity also passed the stricter 30% identity and 70% coverage correspondence gate.

The earlier GatABC benchmark yielded interface-macro DCA-APC AP of 3.87%â€“4.38% on two deposits using one alignment. The added sequence systems have different prevalence, coverage and sampling depth. The two polymerase copies share two subunits, and 139 assemblies occur in both new cohorts. Two distinct sequence systems provide too few independent units for a population confidence interval on transfer to other complexes. Leave-phylum-out predictive evaluation uses a separate frozen panel and estimand. The structural fits themselves use all enrolled assemblies. [GatABC benchmark](experimental-benchmarks.md) Â· [Cohort provenance and exclusions](independent-cohort-staging.md)

## Coordinate and software provenance

4YG2 contains O157:H7 polymerase homologs. All three deposited sequences match the registered E. coli K-12 reference axes at every position. For 4YBB, the deposited atom table identifies assembly-1 label chains WB/AC/KC as author chains DB/DF/DP. The RCSB entry FASTA display assigns an inconsistent author alias to label AC. The registered label chain, entity ID and assembly operation resolve the correspondence in the mmCIF. The uL5 entity covers 177 of the 179 reference positions. [4YBB mmCIF](https://files.rcsb.org/download/4YBB.cif.gz)

The full native dual fits used 22,607 observed-state features in polymerase and 2,650 in the ribosome. The native mfDCA resource estimates were 255.2 and 11.0 MB. For the two weighting schemes, full polymerase MI took 9.65 and 9.64 s and mfDCA took 21.98 and 22.67 s. The ribosome fits took 0.12â€“0.13 s for MI and approximately 0.50 s for mfDCA. The reported estimates cover the numerical workspace.

From the repository root with the project installed, execute the commands below after staging the two cohorts. Score publication requires new destination directories. All full scores, distances and reference mapping tables live under `results/public-experimental/independent-evaluation/`. An independent check compared 5,625 deterministically sampled reference pairs against original mmCIF atoms, including 4,364 observable minima and 1,261 unknowns. It also reproduced all 24 primary pooled AUROCs from ascending midranks.

```sh
python benchmarks/independent_pair_scores.py rnap uniform equal_phylum
python benchmarks/independent_pair_scores.py ribosome uniform equal_phylum
python benchmarks/independent_contact_labels.py
python benchmarks/independent_contact_metrics.py
python benchmarks/verify_independent_contacts.py
python benchmarks/independent_contact_metrics.py --export-only results/public-experimental/independent-evaluation/export
```

The SHA-256 identifiers of the compressed source coordinate files are `6613a2f02c646ae2f3fffa2067f986ee5fd261837659149f27da23f6188a5d0a` for 4YG2 and `9ca44cd8449fb34b29cf3a84e58c798faee5102a8eefe1a09dca144915b72265` for 4YBB. The aggregate report preserves fit manifests, score payload hashes, reference mapping hashes, observed and unobserved counts and per-method top-K bounds. The TSV SHA-256 with LF endings is `15330cfe67f8b68041615bdd1e9627650c6a7cffb93bc6def12ac1010ad663cb`.
