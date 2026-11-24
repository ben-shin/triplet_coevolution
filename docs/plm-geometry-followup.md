# PLM contact recovery and matched triplet geometry

On registered panels of RNA polymerase and the mixed 5S/uL5/uL18 ribosome, pseudolikelihood (PLM) had lower pooled average precision than mean-field DCA after average product correction (APC) on all three experimental copies. In a second comparison, high connected-information triples and lower-scoring controls could be closely matched using sequence data alone. The sampled structures contained zero fully observed triples with two or more contacts at 8 Å, leaving discrimination of that geometric endpoint unmeasurable. The earlier whole-system contact outcomes were known when this follow-up was registered. [Separate follow-up policy](../benchmarks/plm_geometry_followup_policy.json) · [Original study](independent-contact-results.md)

## Known-contact test of pseudolikelihood

Per cohort, 64 hash-selected, well-covered variable sites per component were distributed over 64 joint 24-site fits. An eight-group Latin square assigned every chosen AB, AC and BC coordinate pair to exactly one panel, yielding 12,288 unique pairs per system. For each fit, all six raw/APC methods used exactly the same panel-complete assemblies, with uniform weights. The RNA polymerase retained 422–456 of 461 assemblies per panel, the ribosome 152–163 of 163. Each panel conditioned on its own set of 24 sites. APC used its own 8-by-8 interface block. Pooling ranks across panels therefore describes a composite ranking. [PLM estimator](plm.md) · [Panel design and fits](../benchmarks/followup_plm.py)

| Experimental copy | Contacts / common observable pairs at 8 Å | Prevalence | MI-APC AP | Mean-field APC AP | PLM-APC AP | PLM-APC AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4YG2 A/C/D | 17 / 12,160 | 0.140% | 0.373% | 0.455% | 0.258% | 0.629 |
| 4YG2 B/C/D | 11 / 10,255 | 0.107% | 0.481% | 0.647% | 0.297% | 0.678 |
| 4YBB WB/AC/KC | 148 / 12,272 | 1.206% | 1.249% | 1.458% | 1.330% | 0.541 |

The first 50 PLM-APC pairs contained zero known contacts for each copy. Their unknown counts were 0, 10 and 0. The possible precision at 50 for B/C/D extends from 0% to 20%. The known contact counts by AB, AC and BC were 7/0/10, 1/0/10 and 37/111/0 in table order. AP and AUROC are undefined for the zero-positive interfaces. The two polymerase alpha copies share beta and beta-prime, so the BC results coincide. The archive gives raw PLM, raw mean-field and raw MI alongside their APC results for every interface and copy at 5, 8 and 10 Å. It includes lower and upper top-K precision bounds and equal-interface macro results. [Detailed results and target diagnostics](../benchmarks/plm_geometry_followup_results.json.gz)

Of 3,072 target fits, 3,063 converged in 9–15 iterations and nine targets were constant among their complete assemblies. Incident PLM scores for constant targets were unavailable. All converged targets met the registered gradient tolerance of 0.00001. The largest planned numeric and scratch workspaces per panel were 5.76 MB and 0.62 MB. A check recomputed 384 exported coupling-block norms and 18 APC blocks. An independent rank-sum calculation reproduced the defined AUROCs. The small numbers of known polymerase contacts and different conditioning contexts limit estimator comparisons.

## Geometry-blind matching

From 8,192 uniformly sampled triples in each cohort, the top tenth by connected information became candidates. A fixed, cohort-wide modal residue versus all other canonical residues defined the two-state alphabet at each position. The three-way statistic is the divergence in bits from the fitted maximum-entropy model constrained by the three pair marginals. The pair-only ranking used the mean percentile of the three frozen mean-field APC scores.

Each candidate received at most one unused control matched on its joint canonical support, the three site entropies and all three individual pair-score percentiles. The selections were hashed before distances were joined. Thirty-four polymerase and 25 ribosome triples had a constant binary axis on joint rows. Remaining failed matches came from the prespecified calipers. The maximum absolute standardized difference across seven matching variables decreased from 0.444 to 0.058 in polymerase, and from 0.643 to 0.034 in the ribosome. Median matched candidate/control connected information was 0.0248/0.00195 bits and 0.0388/0.00293 bits, respectively. Corresponding pair-only percentile medians were 0.484/0.488 and 0.493/0.495. [Selection and balance calculation](../benchmarks/followup_geometry.py)

| Copy | Matched cases / eligible cases | Observable vs unknown sampled triples | Matched pairs with both geometries observed | Mean candidate minus control maximum edge |
| --- | ---: | ---: | ---: | ---: |
| 4YG2 A/C/D | 731 / 815 | 7,878 / 280 | 706 | −2.25 Å |
| 4YG2 B/C/D | 731 / 815 | 5,397 / 2,761 | 330 | −1.82 Å |
| 4YBB WB/AC/KC | 723 / 816 | 8,120 / 47 | 714 | +0.008 Å |

Zero observed wedges or triangles at 8 Å made both the connected-information and pair-only geometric AP/AUROC undefined. The paired event-rate difference among fully observable matched pairs was zero for every copy. Missing geometry matters particularly for polymerase B/C/D. Of the 731 candidate/control pairs enrolled there, 245 candidates and 240 controls had unknown geometry. Their individual upper bounds on the fraction of wedges or triangles were 33.5% and 32.8%. The sampled maximum-edge differences are descriptive. The same residues and genomes contribute repeatedly, the two copies share subunits, and structural proximity alone cannot establish an energetic effect.

## Reproduce or reuse the comparison

The public cohorts and frozen structural distance archives are required for the biological study. The shared-contact conventions, registered K-12 coordinate references, assembly selections and source-accession caveats are documented in the earlier benchmark. [Inputs](independent-cohort-staging.md) · [Experimental structures](independent-contact-results.md)

```sh
python benchmarks/followup_plm.py rnap
python benchmarks/followup_plm.py ribosome
python benchmarks/followup_geometry.py study rnap
python benchmarks/followup_geometry.py study ribosome
python benchmarks/followup_verify.py results/public-experimental/plm-geometry-followup/verified.json.gz
```

For new data, the standalone matching command takes a tab-separated candidate table with the following columns in order. Use a unique integer ID, connected information in bits, support as a fraction, alphabet-normalized canonical entropies and three within-interface fractional midranks. Features range from zero to one. All rows must have already passed the intended cohort's eligibility rules. The matching command uses the registered upper decile, calipers, tie rule and no-replacement selection. The measurement command joins tab-separated distances with ordered columns `id`, `AB`, `AC`, `BC` in Å. Blank cells and absent rows represent unknown distances. Each output's parent directory must exist.

```text
id	connected_information_bits	support	entropy_A	entropy_B	entropy_C	AB_percentile	AC_percentile	BC_percentile
```

```sh
python benchmarks/followup_geometry.py match candidates.tsv selection.json
python benchmarks/followup_geometry.py measure selection.json distances.tsv geometry.json
python -m pytest -q tests/test_followup_geometry.py
```

The fixtures verify ties, exclusion by caliper, non-reuse, missing geometry, a known paired span difference, smoothed XOR information and a zero-information independent table. An atom-level check reproduced 276 measured sampled edges and confirmed 30 unknown edges from the archived mmCIF records. This study's compressed aggregate contains all declared sensitivity results and compact target diagnostics. Its uncompressed canonical SHA-256 is `349a5639b9979a0ce2682899cf658ddbdc4290391505712bd1d92614255617d8`.
