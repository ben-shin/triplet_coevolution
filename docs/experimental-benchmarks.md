# GatABC structural replication and marker-tree benchmark

On two separately deposited transamidosomes, APC-corrected DCA achieved the highest interface-macro average precision of the four frozen rankings. Absolute average precision was 0.039–0.044. The same 888-observation alignment and its original pair scores were used in each evaluation. The structures vary the experimental labels and species. Independent sequence systems were evaluated separately in the [polymerase and ribosome study](independent-contact-results.md). [Registered protocol](experimental-benchmark-protocol.md) · [Thermotoga 3AL0](https://www.rcsb.org/structure/3AL0) · [Thermus 3KFU](https://www.rcsb.org/structure/3KFU)

## Experimental contacts

Values are equal-weight means across AB, AC and BC at a minimum deposited heavy-atom distance of 8 Å. Within each interface, all methods used exactly the same observable, finite-score positions. The two 3KFU copies belong to one deposition. A system-level confidence interval would require independently scored systems.

| Experimental copy | Jointly measurable and scored pairs | MI | MI-APC | DCA | DCA-APC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3AL0 A/B/C | 295,073 | 0.0173 | 0.0233 | 0.0140 | 0.0387 |
| 3KFU E/F/G | 268,129 | 0.0148 | 0.0235 | 0.0119 | 0.0424 |
| 3KFU H/I/J | 266,559 | 0.0152 | 0.0239 | 0.0122 | 0.0438 |

| Copy / interface | All observable | Common scoreable | Contacts at 8 Å | Prevalence | DCA-APC top 10, positives / unknowns |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3AL0 AB | 217,092 | 212,926 | 411 | 0.193% | 2 / 0 |
| 3AL0 AC | 40,762 | 40,406 | 618 | 1.529% | 9 / 0 |
| 3AL0 BC | 42,186 | 41,741 | 352 | 0.843% | 6 / 0 |
| 3KFU EFG AB | 195,650 | 192,621 | 392 | 0.204% | 2 / 1 |
| 3KFU EFG AC | 39,130 | 38,614 | 485 | 1.256% | 6 / 2 |
| 3KFU EFG BC | 36,980 | 36,894 | 374 | 1.014% | 8 / 0 |
| 3KFU HIJ AB | 197,944 | 193,519 | 387 | 0.200% | 2 / 1 |
| 3KFU HIJ AC | 37,682 | 37,267 | 484 | 1.299% | 6 / 2 |
| 3KFU HIJ BC | 36,188 | 35,773 | 373 | 1.043% | 8 / 1 |

Each copy started with 234,240 AB, 48,800 AC and 48,000 BC reference pairs. The shared intersection across all three copies contained 181,440 AB, 34,560 AC and 33,600 BC pairs with finite values for every ranking. Top-10 positions were selected before masking unknown labels. For example, the six confirmed contacts and two unknowns among 3KFU AC's top ten bound precision between 0.60 and 0.80. On each copy's jointly scoreable positions, 4/5/8/10 Å cutoffs produced 227/371/1,381/2,534 positives for 3AL0, 214/341/1,251/2,290 for 3KFU EFG and 202/346/1,244/2,263 for HIJ. Complete AP, AUROC, top-10/top-50 counts and the cross-copy-intersection sensitivity are archived in the aggregate output. [Aggregate metrics](../benchmarks/experimental_structure_metrics.tsv)

The deposited sequence annotation identifies 3AL0 GatC as residues 3–97 of a fusion chain. The linker and ligase were excluded. All three 3AL0 subunits pass the stricter 30% identity and 70% coverage sensitivity. In 3KFU, GatA and GatB pass both gates. GatC has 26.1% identity and 88.0% coverage, so only AB remains under the stricter gate. DCA-APC AP on AB is 0.00765 and 0.00770 for the two copies. The registered EFG/HIJ pairing had the lowest chain-centroid assignment cost, 224.5 Å versus 438.4 Å for the next assignment. [3AL0 entity composition](https://www.rcsb.org/structure/3AL0) · [3KFU assembly](https://www.rcsb.org/structure/3KFU)

## Externally inferred bacterial tree

Exactly 590 of 888 observations mapped to unique tips in GTDB R232's 189,801-tip bacterial marker tree. All three proteins had a single matching local proteome and taxonomy witness. The current UniProt proteome assembly directly identified 11 tips, and NCBI marked the GenBank and RefSeq assemblies identical for another 579. The other 298 observations lacked an exact representative tip. The 590 retained tips span 27 phyla. The marker-file descriptions list 120 markers and contain no GatA, GatB, GatC or amidotransferase annotation. The marker alignment was inferred externally and was not fitted from this three-component MSA. The topology, branch lengths and exact-representative sampling still condition the result. [GTDB R232](https://data.gtdb.ecogenomic.org/releases/release232/232.0/) · [Release marker identities](https://data.gtdb.ecogenomic.org/releases/release232/232.0/auxillary_files/bac120_msa_marker_info_r232.tsv) · [NCBI assembly-report definitions](https://ftp.ncbi.nlm.nih.gov/genomes/README_assembly_summary.txt)

On the 192 prespecified MI pairs from 24 columns, the largest observed MI was **0.71284 bits**. Across 99 independent-site F81 draws on the induced tree, the median simulated maximum was **0.72028** and its 95th percentile was **0.94415**. The global maximum test gave **p = 0.54**. Thirty individual pairs had nominal Monte Carlo p ≤ 0.05. None passed BH at 5% (minimum q = 0.192). With 99 draws the minimum p is 0.01. This test covers a bounded, outcome-blind panel with empirical stationary frequencies, fixed masks and one branch scale. It does not estimate an all-site false discovery rate.

Twelve phyla qualified for a held-out comparison. Spearman correlation of train and held-out raw-MI rankings ranged from 0.449 to 0.851, with zero to four shared pairs among their ten highest. Four or fewer shared pairs occurred even in the largest held-out groups. Column variability and coverage were evaluated on the full admitted cohort. Held-out estimates thus assess rank stability on that fixed panel. [All split sizes, correlations and overlaps](../benchmarks/experimental_tree_metrics.json)

## Reproduction

The machine protocol was frozen before new geometry, score values or labels were inspected. Its SHA-256, with LF newlines, is `14d58d27f6b6f4730b493a66ac3c3df1fac25ae01b5c1a4ab733b943d3f42b24`. Keep the exact downloaded bytes under `results/public-experimental/sources/` using the names and URLs in the source manifest, and the owner-supplied frozen inputs at `results/private-gatabc-validation/`. Changing public source hashes requires a separately identified rerun. UniProt's streaming endpoint and NCBI's assembly report are moving resources. Replay of this snapshot requires the recorded release and checksum. The private accession join, site panel, mappings and distance arrays stay in ignored local results. [Public source hashes](../benchmarks/experimental_sources.json) · [Machine protocol](../benchmarks/experimental_protocol.json)

```console
python benchmarks/experimental_structures.py
python benchmarks/experimental_tree.py
```

The runs produce `structural-summary.json`, per-copy correspondence and distance arrays, and `tree-summary.json` with its private join and induced tree. Source coordinates remain in the wwPDB archive under CC0. UniProt publishes copyrightable database content under CC BY 4.0. GTDB and NCBI data were retrieved as attributed source inputs. No source databases or private input files are redistributed with these results. [wwPDB usage](https://www.wwpdb.org/about/usage-policies) · [UniProt licensing](https://www.uniprot.org/help/license) · [GTDB attribution](https://gtdb.ecogenomic.org/about)

A separate Biopython mmCIF read and direct atom-by-atom calculation checked 45 measurable residue pairs across the nine copy interfaces. The largest difference was 0.000009 Å. Comparing 136 tip-pair path lengths before and after tree pruning gave a maximum absolute difference of 4.45 × 10⁻¹⁶.
