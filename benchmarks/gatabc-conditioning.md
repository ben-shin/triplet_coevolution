# GatABC coupling sensitivity on two bounded panels

The 2G5H association between fitted scores and structure depended on how residues entered the comparison. Both panels use the previously supplied GatABC matched observations, unweighted for this prespecified experiment. All protein source FASTAs, the observation table and 2G5H labels were verified against the published historical digest list. No structural label, protein sequence or record was submitted to a service. This complex contributed to development. It provides no independent-system test of a tuned estimator. [Source identities](gatabc_manifest.json) · [Structural label reconstruction](gatabc-structure.md)

**Availability-only panel.** Before inspecting model scores, choose the six reference positions per component with the largest number of labeled incident pairs, with ties in reference/position order. This rule uses label availability and ignores class. The 18 sites gave 108 intercomponent pairs. Requiring one canonical character per role retained 860 of 888 rows. Mean-field scored 29 positive labels and zero negatives. Complete-case joint PLM included 830 observations and scored 27 positives. One constant target was withheld. AUROC is undefined and AP equals one on either all-positive labeled subset. These labels cannot discriminate estimator performance. Across the 36 pairs in each interface the joint APC rank correlations of mean-field ridge 1 against .001 were −.12, .07 and .39 for AB, AC and BC.

**Class-ascertained contrast.** To obtain observable negative controls after the availability-only panel returned none, take two labeled pairs per class per interface by ascending SHA-256 of the coordinate tuple. Include their endpoints without looking at any fitted score. The resulting source panel has eight sites per component, yielding 192 potential pairs. All 888 matched rows entered with unit weight. Among those pairs 11 contacts and six distant controls had class labels. The other 175 are unlabelled. All 17 entered both raw/APC rankings and every labeled comparison below. The same labels helped select this panel, which identifies a retrospective conditional comparison. Its contact prevalence differs from that of the full system.

| Prespecified estimator | Complete or endpoint support | Raw AUROC / AP | APC AUROC / AP | Fit time |
| --- | --- | --- | --- | --- |
| Joint mean-field, ridge .1 | 717–886 by pair | .500 / .725 | .667 / .820 | .009 s |
| Separate AB/AC/BC mean-field, ridge .1 | 717–886 by pair | .576 / .795 | .652 / .811 | .011 s |
| Joint PLM, both penalties .01 | 607 complete observations | .212 / .528 | .576 / .774 | 2.516 s |
| Separate AB/AC/BC PLM, both penalties .01 | 627 / 638 / 799 complete | .242 / .544 | .561 / .769 | 3.955 s |

All 24 joint and 48 pairwise PLM target fits converged. The numeric workspace estimates for joint mean field and joint PLM were 5,011,200 and 6,097,152 bytes. Changing the conditioning set changes complete-row attrition as well as model coefficients. On the fixed 17-label subset, six prespecified joint ridge values (.001, .01, .03, .1, .3, 1) gave APC AUROCs .576, .682, .682, .667, .515, .455. The separate mean-field fits gave .636, .697, .682, .652, .530, .439. Settings are reported together and none was selected using these outcomes.

Each timing is a single local fit, with one BLAS thread and one target worker, excluding disk I/O, benchmarking and planning. It is a scale indication for these panels. This single-complex, shared-residue, ascertained comparison has no independently replicated molecular systems or declared independent label clusters. A pair-bootstrap confidence interval would overcount independent evidence. The linked synthetic and public-software studies quantify numerical and known-truth behavior separately. [Native PLM and licensed comparison](../docs/plm.md) · [Synthetic held-out controls](nested-models.md)

```console
python benchmarks/gatabc_conditioning.py PATH_TO_VERIFIED_INPUTS NEW_OUTPUT_1
python benchmarks/gatabc_conditioning.py PATH_TO_VERIFIED_INPUTS NEW_OUTPUT_2 --panel balanced
```

The command writes a source-coordinate panel, native score archive and fitting diagnostics for all 14 configurations. Every score row includes reference coordinates, observed support, state count, entropy, coverage and gap fraction. [Reproduction script](gatabc_conditioning.py)
