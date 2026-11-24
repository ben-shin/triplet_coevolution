# GatABC structural regression on 2G5H

Of the 2,880 archived labels, 2,830 agreed with an independent mapping to the supplied crystal coordinates. Thirteen archived positives mapped to measurable distances greater than 8 Å. The remaining 37 depended on a residue opposite a sequence-alignment gap, so their geometry is unknown under this mapping.

## Source and correspondence

The checksummed 2G5H mmCIF declares one biological assembly, a trimer (assembly 1), and one X-ray model. The structure represents *Staphylococcus aureus* (taxonomy ID 1280). The prepared coordinate references belong to *Streptococcus pneumoniae*. Each complete reference was aligned against the full deposited polymer sequences before consulting contact labels. The three highest-scoring supported copies were unique. All numbers below refer to this homologous structure and its selected deposited conformers.

| Component and reference | Chain copy | Reference / polymer length | Identity | Coverage | Reference sites with mapped atoms |
|---|---|---:|---:|---:|---:|
| GatA Q97SE6 | A@1 | 488 / 485 | 55.69% | 98.98% | 483 |
| GatB Q97SE7 | B@1 | 480 / 483 | 63.45% | 99.17% | 397 |
| GatC Q97SE5 | C@1 | 100 / 100 | 44.44% | 99.00% | 98 |

The protein mapping uses global BLOSUM62 alignment with a gap opening penalty of 10 and extension penalty of 0.5. Identity is measured among unambiguously paired canonical residues. Coverage uses the entire reference length as its denominator. Per-position ambiguity is evaluated over all optimal alignments. One GatB position had multiple optimal correspondences. Another 79 mapped to polymer positions without deposited atoms. GatA, GatB and GatC had respectively five, three and one reference positions aligned to gaps.

## Archived positive contacts and distant controls

The original benchmark classified positives by minimum inter-residue heavy-atom distance ≤8 Å and selected distant controls by Cα distance ≥25 Å. A positive takes precedence when evaluating these two criteria. Each of the 2,880 original coordinate pairs was evaluated under those definitions.

| Interface | Original positives | Positives reproduced | Measured above 8 Å | Positive unresolved | Original distant controls | Distant reproduced | Distant unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|
| AB | 409 | 402 | 4 | 3 | 409 | 407 | 2 |
| AC | 610 | 607 | 0 | 3 | 610 | 599 | 11 |
| BC | 421 | 397 | 9 | 15 | 421 | 418 | 3 |

All 13 measured disagreements involve GatB reference residues 135–136 or 278–280. Their minimum heavy-atom distances range from 8.15 to 12.68 Å. The compact bundle did not specify the original residue-level alignment path. These discrepancies localize where correspondence choices merit inspection. Every measurable archived pair also underwent a separate direct atom-by-atom minimum-distance check. The largest absolute difference from the streamed distance calculation was 1.43 × 10⁻¹⁴ Å (rounded up).

| Reference pair | Author chain/residue in 2G5H | Heavy-atom minimum (Å) | Cα (Å) | Cβ with glycine Cα fallback (Å) | Archived label |
|---|---|---:|---:|---:|---|
| A K301 / C L40 | A LYS303 / C GLN41 | 4.575 | 8.836 | 7.513 | Positive |
| A T381 / C T52 | A THR383 / C THR53 | 2.596 | 5.422 | 4.621 | Positive |
| B K53 / C P64 | B LYS53 / C GLU65 | 2.886 | 5.531 | 5.951 | Positive |
| A R321 / B D276 | A ARG323 / B ASP274 | 6.551 | 15.570 | 13.125 | Positive |
| A G175 / B D276 | A GLY177 / B ASP274 | 17.026 | 20.300 | 20.572 | Unlabeled |
| A G175 / C S15 | A GLY177 / C ALA16 | 23.810 | 26.039 | 24.946 | Unlabeled |
| B D276 / C S15 | B ASP274 / C ALA16 | 18.317 | 21.264 | 20.614 | Unlabeled |

## Complete measurable interfaces

The full reference map spans 331,040 inter-component pairs. Of these, 277,991 have a measurable minimum between deposited heavy atoms. A structurally observable pair enters a score comparison when the retained legacy model also provides a finite score at those exact reference coordinates. The final column gives prevalence within the observable, scoreable pairs.

| Interface | Full reference pairs | Observable | Heavy ≤4 Å | Heavy ≤5 Å | Heavy ≤8 Å | Cβ ≤8 Å | Observable and scoreable | Heavy ≤8 Å prevalence among observable, scoreable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AB | 234,240 | 191,751 | 63 | 107 | 405 | 92 | 189,369 | 0.214% |
| AC | 48,800 | 47,334 | 94 | 162 | 630 | 170 | 44,361 | 1.375% |
| BC | 48,000 | 38,906 | 75 | 121 | 419 | 108 | 36,921 | 1.119% |

At the 8 Å heavy-atom cutoff, 1,428 of the 270,651 pairs with measurable geometry and all four legacy scores were positive (0.528%). Every other measurable pair in this comparison received a negative label. Macromolecular contacts omitted by missing coordinates remained unknown.

| Statistic | Pooled AUROC | Pooled noninterpolated AP |
|---|---:|---:|
| MI | 0.530672 | 0.008728 |
| MI/APC | 0.606995 | 0.014287 |
| DCA Frobenius | 0.485561 | 0.012077 |
| DCA/APC | 0.635551 | 0.032277 |

These figures describe a single homologous assembly using GatABC scores from the local v1.0.0 refit. The earlier balanced sample had a positive prevalence of 50%. Neither benchmark establishes three-body energetic dependence, generalization to other complexes or a confidence interval over independent structures.

## Replay

From the installed checkout, run the following against the supplied local bundle. The verifier checks the original labels, three prepared FASTAs, and crystal file against the expected hashes before calculating. Its JSON output contains source identities, per-role mapping coverage, threshold counts, named reference pairs and the direct-distance comparison.

```bash
python benchmarks/verify_gatabc_structure.py "$GATABC_ROOT"
python benchmarks/verify_gatabc_structure.py "$GATABC_ROOT" --scores "$GATABC_SCORE_ARTIFACT"
```

The mmCIF SHA-256 is `4bda3244cd2927691d66827c5f551ed81f6e24fcb49e6fafc0e16711fb7723c8`. The original contact TSV SHA-256 is `01041f913c8aa13e7be79b02b7f97d84d52d9ec10d1841fa0b16e1b2d688137b`. The full scoreable-pair comparison additionally requires the four complete score matrices of the refit. [Score regression and refit instructions](gatabc-regression.md)
