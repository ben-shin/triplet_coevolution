# GatABC v1.0.0 regression

On November 24, 2026, the original compact score archive reproduced all four historical AUROC and average precision (AP) benchmarks on all 2,880 labeled pairs. Refitting the frozen v1.0.0 source on the original prepared alignments reproduced all published reference ranks. One pooled metric fell outside the prespecified rounding tolerance.

## What was measured

| Statistic | Original pooled AUROC | Original pooled AP | Refit pooled AP |
| --- | ---: | ---: | ---: |
| MI | 0.503004678 | 0.563083807 | 0.563083185 |
| MI/APC | 0.601581790 | 0.636672048 | 0.636672048 |
| DCA Frobenius | 0.431892361 | 0.480900767 | 0.480900767 |
| DCA/APC | 0.634399595 | 0.655387163 | 0.655387163 |

The two calculations gave identical pooled AUROCs. All interface AUROC/AP results and macro AUROCs met the original printed tolerances. The refit recovered 888 observations, effective depth 505.49112803812943, 480/474/94 retained columns, and 226,098/44,838/44,556 eligible AB/AC/BC pairs. The original capped exports each had 1,000 rows. Every one of those coordinates occurred in the full refit. The maximum absolute difference across both score columns was 1.12e-16 for MI and 6.11e-16 for DCA on these rows.

The original raw-MI pooled AP was 0.5630838071254171 using grouped ties. The refit gave 0.5630831846447628. Six positive AB pairs shared one raw score near 1.28e-15 in the original. Five moved up one binary64 unit in the refit, splitting the tied group. This changes the precision at a recall increment while preserving the ordering of positives versus negatives. Across all 2,880 labeled pairs, the maximum absolute raw-MI change was 1.11e-16 to three significant digits. The validator retains its original pooled tolerance of 5.00001e-8, reports the discrepancy and returns exit code 1 for this refit.

The original and compact bundle checksums verify the prepared MSAs, matches, contact labels and PDB 2G5H coordinates. All 27 entries listed by the compact bundle's checksum document matched. The compact bundle contains both score families for all labeled pairs, 1,200 top-ranked rows, the complete 1,000-candidate local model and triple permutation tables, and 3,891 MI permutation rows. Its published AP and AUROC values were independently recalculated from the labeled scores. The six original full score-file hashes are known. Their bytes were absent from the supplied compact bundle. The six regenerated tables have different byte hashes.

All 1,200 top-list labels agreed with the original contact TSV, including 1,071 unknowns. At K=10 the positive-label counts were 1/3/2 for MI/APC and 2/9/7 for DCA/APC in AB/AC/BC. The original 1,000 triple scores matched their permutation observations exactly on original alignment columns. With 19 draws, 70 had p=0.05, 603 had p=1, and the minimum BH q was 0.7142857143. These are archived results. The refit covered only the pairwise models. The complete historical MI permutation table and upstream search alignments were unavailable. Structural distance remeasurement and independent systems are separate validation tasks.

## Reproduction

The score-table validator runs from the repository root after installing the package. Set `DATA` to the extracted compact bundle. It validates the original label and labeled-score hashes before computing metrics. An operator with the six complete original tables can omit `--compact` to check their individual hashes, score coverage and coordinate ranks.

```bash
python benchmarks/validate_gatabc.py "$DATA" --compact
```

For the historical refit, install repository commit `c511759810d6a1659ea60b3022913dd646ea5093` in a separate environment. Run the following against the prepared files, using an empty output directory.

```bash
PREP="$DATA/results/gatabc_prepared_20261123T182555Z"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
triplet analyze --a "$PREP/A.fasta" --b "$PREP/B.fasta" --c "$PREP/C.fasta" \
  --matches "$PREP/matches.tsv" --a-ref 'sp|Q97SE6|GATA_STRPN' \
  --b-ref 'sp|Q97SE7|GATB_STRPN' --c-ref 'sp|Q97SE5|GATC_STRPN' \
  --score mi_apc --dca --ridge 0.1 --max-pairs 250000 --min-valid 100 \
  --require-reference --contacts "$DATA/metadata/contacts_2G5H.tsv" --no-plots --out "$OUT"
python benchmarks/validate_gatabc.py "$DATA" --recomputed-scores "$OUT"
```

The v1.0.0 CLI used the three protein defaults, reference starts of 1, identity threshold 0.8, minimum overlap 0.5, minimum coverage 0.5, and maximum gap and missing fractions 0.5. The refit took 23.28 seconds wall time with 487,067,648 bytes peak process working set on Windows, Python 3.13.15, NumPy 2.5.5 and Biopython 1.88 with one BLAS thread. The original producer recorded Linux, Python 3.12.16, NumPy 2.5.5 and Biopython 1.88. This is one legacy timing. All raw alignments, scores, structures and local paths stay outside version control.

The label set has 1,440 positive contacts and 1,440 sampled distant controls. AP measures precision for this balanced sample. Shared-proteome evidence supports candidate groups and leaves cognate orthology uncertain. The [benchmark manifest](gatabc_manifest.json) records the complete historical targets, source paths, hashes, score definitions and label thresholds.
