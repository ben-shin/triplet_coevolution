# Candidate discovery and tests on known laws

On exact uniform binary XOR, every pair distribution is independent and the connected three-site information is one bit. In 48 simulated panels a hybrid screen retained that triple nine times at a 12-of-64 budget. Exhaustive held-out scoring ranked it first in all 48. An independent quota gives pure synergy a calculable chance of inclusion. The measured recall depends on budget and on the alternatives in the panel.

## Candidate coordinates and selection

The selector operates on three eligible axes of retained columns in increasing original alignment order. All files carry the three reference IDs, including when numbering runs in reverse. A reference-gap column retains its source position. A run can explicitly exclude unnumbered sites. The manifest records per-role retained, reference-numbered and eligible site counts. The full retained product and the product after coordinate exclusions are separate denominators.

| Mode | Selection rule |
| --- | --- |
| `exhaustive` | Visit every eligible triple. The request must fit the candidate budget. |
| `uniform` | Sample without replacement from the mixed-radix eligible product. The default reproduces v1 seeded sets and order. |
| `explicit` | Use the exact three source columns or three matched reference IDs and positions in a TSV. The default prespecified origin is an assumption retained for compatibility. A file can declare same-data or training-derived selection. |
| `prior` | Use an explicit coordinate list with a declared structural or functional evidence source. The file hash and source accompany the selected hypotheses. Assess independent evidence separately. |
| `triangle` | Rank by the weakest of three finite pair scores. |
| `wedge` | Rank by the second highest of at least two finite pair scores, allowing a missing third edge. |
| `entropy` | Apply minimum canonical site coverage and minimum Shannon entropy in bits. Rank each passing triple by the smallest of its three site entropies multiplied by their respective canonical weight fractions. |
| `hybrid` | Reserve a uniform sample first. Admit prior proposals in original-coordinate order, then alternate wedge and entropy proposals until the budget or available candidates are exhausted. |

An explicit file takes the three headers `a_original_col`, `b_original_col` and `c_original_col`, or all six headers `a_reference_id`, `a_reference_pos`, `b_reference_id`, `b_reference_pos`, `c_reference_id` and `c_reference_pos`. Coordinates are one-based. A duplicate, malformed or excluded coordinate rejects the file and reports counts from the parsed records. Inputs that exceed a hard bound stop at the first violating record. A hybrid prior list can be longer than the remaining selection budget. The manifest counts prior rows that failed to enter the final set.

Hybrid reserves at least one score-independent entry whenever the eligible product is nonempty. The default quota is the ceiling of one quarter of the smaller of the fit budget and the eligible product. A requested quota must lie between one and the fit budget. A given eligible triple has probability `quota / eligible` of entering through that stage, regardless of its edge scores. Rankings break ties by increasing source coordinates. The output order remains source-coordinate order and each fitted score has its own rank.

Data-dependent screens inspect a second, separately seeded uniform probe pool when the cheap-screen budget is smaller than the product. Each inspection costs three edge lookups or three site lookups. The selector retains bounded top lists, with O(P log K) ranking work for P probe triples and K fitted slots. A direct raw-MI edge fit costs O(N + qX qY); an 8,192-entry memo limits its storage. Supplied pair matrices are hashed once, streaming over rows in O(sum of pair matrix sizes) time. The sampling indices and top lists take O(P + K) working entries in the sampled case. Full eligible scans iterate integer codes directly. These bounds exclude input alignments and precomputed score matrices. The implementation caps effective fitted candidates at 100,000, probes at 1,000,000, and physical input records at 1,000,000. Restrict the model panel or either budget when a requested scan exceeds these supported limits. Exact source tuples and stable IDs link the manifest, support table and fit results.

The library can restrict screening to an explicit set of training observation IDs. It recomputes sequence weights and raw canonical pair MI on those rows. Held-out sequence edits leave that selection and its hashes unchanged under fixed axes. The training path rejects externally fitted pair scores, which could encode held-out information. A confirmatory study must also define or fit coordinate masks on training and score a separate evaluation cohort. A screen computed on the same rows as evaluation is exploratory. A structural annotation used to select a triple has already contributed to that hypothesis and needs an independent validation source.

## Fixed-budget benchmark

The script generated 48 independent panels per scenario using root seed 34309 and independent NumPy SeedSequence substreams indexed by scenario, role and replicate. Each panel contained four sites in each of three roles and therefore 64 triples. Each role used a distinct two-symbol subset of a protein, DNA or RNA alphabet. In every replicate the designated target moved to a random coordinate. A disjoint distractor followed a pair-only triangle law and the remaining sites were independent binary noise. Each screen saw 192 training observations, retained at most 12 candidates, and used raw MI or marginal entropy from training. The hybrid used three reserved uniform positions. A separately simulated set of 192 observations supplied all ranking scores. IPF used a total prior mass of 0.5 over eight cells. The exhaustive evaluation fit all 64 triples to give an exact-small comparator. Every held-out candidate score was finite.

The following cells report target inclusions out of 48, followed by 95% Wilson intervals for the inclusion probability. The first three rows are designated controls with zero true I3. Their inclusion measures attraction to first- or second-order properties.

| Target law | Uniform | Triangle | Wedge | Entropy | Hybrid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent fields | 8 (9–30%) | 6 (6–25%) | 5 (5–22%) | 2 (1–14%) | 1 (0–11%) |
| Pair-only chain | 7 (7–27%) | 48 (93–100%) | 48 (93–100%) | 2 (1–14%) | 48 (93–100%) |
| Pair-only triangle | 8 (9–30%) | 48 (93–100%) | 48 (93–100%) | 1 (0–11%) | 48 (93–100%) |
| Hidden clade mixture | 7 (7–27%) | 48 (93–100%) | 48 (93–100%) | 0 (0–7%) | 48 (93–100%) |
| Pure XOR | 5 (5–22%) | 8 (9–30%) | 4 (3–20%) | 16 (22–47%) | 9 (10–32%) |
| Biased, noisy XOR | 4 (3–20%) | 26 (40–67%) | 48 (93–100%) | 0 (0–7%) | 47 (89–100%) |
| Pair and three-site terms | 10 (12–34%) | 45 (83–98%) | 48 (93–100%) | 0 (0–7%) | 48 (93–100%) |

These counts record how often the designated target ranked first by held-out connected information. They use all 48 replicates in the denominator, including any replicate where the screen excluded the target. The script emits Wilson intervals for these rates too.

| Target law | Uniform | Triangle | Wedge | Entropy | Hybrid | Exhaustive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent fields | 1 | 0 | 1 | 0 | 1 | 0 |
| Pair-only chain | 1 | 7 | 5 | 0 | 5 | 0 |
| Pair-only triangle | 0 | 3 | 2 | 0 | 2 | 0 |
| Hidden clade mixture | 3 | 22 | 20 | 0 | 21 | 10 |
| Pure XOR | 5 | 8 | 4 | 16 | 9 | 48 |
| Biased, noisy XOR | 4 | 26 | 48 | 0 | 47 | 48 |
| Pair and three-site terms | 10 | 45 | 48 | 0 | 48 | 48 |

Pairwise screens preferentially admitted the pair-only controls. Biased inputs and noise made pairwise associations visible under noisy XOR, which favored the wedge. The entropy rule performed poorly on several imbalanced targets. For unbiased pure XOR, the population pair scores give the pair screens no specific information about the target. The trial counts show finite-sample variation, and the intervals overlap for several modes.

## Calibration at a fixed coordinate

Each positive binary law has one free degree of freedom after fixing all its pair marginals. Moving along the signed parity contrast preserves those marginals. An independent bisection oracle set the products of the four even and four odd cell probabilities equal, the condition for zero three-way log interaction. Across the seven exact laws its M2 cell probabilities differed from the package IPF results by at most 5.8 × 10⁻¹⁰.

For each replicate the fixed target was tested against its **known** M2 projection. We generated 39 independent null samples of size 192, refitted the score on each, and used the inclusive upper-tail Monte Carlo p-value with one added to numerator and denominator. A new null bank was generated for every scenario and replicate. The nominal cutoff was 0.05. The first three rows measure false positives for the generating marginal null. Rejections for the other laws measure power against their known projections. These calculations do not calibrate a selected winner or a composite null fitted from the same data.

| Target law | True I3 (bits) | True signed co-information (bits) | Rejections / 48 (95% interval) |
| --- | ---: | ---: | ---: |
| Independent fields | 0.000000 | 0.000000 | 2 (1–14%) |
| Pair-only chain | 0.000000 | 0.083406 | 5 (5–22%) |
| Pair-only triangle | 0.000000 | 0.197367 | 0 (0–7%) |
| Hidden clade mixture | 0.006313 | 0.090079 | 19 (27–54%) |
| Pure XOR | 1.000000 | -1.000000 | 48 (93–100%) |
| Biased, noisy XOR | 0.284265 | -0.290608 | 48 (93–100%) |
| Pair and three-site terms | 0.170152 | -0.208976 | 48 (93–100%) |

The latent scenario mixed two clades with distinct single-site frequencies. In each clade, the three target sites were conditionally independent, while the pooled law had I3 of 0.006313 bits. In a separate leave-one-clade-out stress test, the hybrid selector trained on the first clade and admitted the designated coordinate in four of 48 replicates (95% interval 3.3–19.6%). The known target I3 in either individual clade was zero to numerical precision. Its mean held-clade estimate was 0.00312 bits (95% replicate-bootstrap interval 0.00193–0.00466). The bootstrap used 4,000 resamples with seed 1541. A positive finite-sample estimate can occur within either clade.

## Depth and prior sensitivity

The same fixed evaluation target was scored with 24 or 192 rows and several total prior masses. Each cell reports mean fitted I3 in bits among finite fits and their count. In sparse, unsmoothed cases the finite-fit denominator changes. All 48 replicate outcomes contribute to the availability count. The machine-readable script output includes all six depth/prior combinations, mean occupied cells and the prior fraction.

| Law | N=24, α=0 | N=24, α=0.5 | N=24, α=8 | N=192, α=0.5 |
| --- | ---: | ---: | ---: | ---: |
| Independent fields | 0.046 (48/48) | 0.040 (48/48) | 0.019 (48/48) | 0.003 (48/48) |
| Pair-only chain | 0.033 (29/48) | 0.017 (48/48) | 0.010 (48/48) | 0.005 (48/48) |
| Pair-only triangle | 0.036 (35/48) | 0.022 (48/48) | 0.009 (48/48) | 0.003 (48/48) |
| Hidden clade mixture | 0.061 (38/48) | 0.041 (48/48) | 0.014 (48/48) | 0.013 (48/48) |
| Pure XOR | 0.908 (46/48) | 0.798 (48/48) | 0.405 (48/48) | 0.972 (48/48) |
| Biased, noisy XOR | 0.273 (47/48) | 0.247 (48/48) | 0.143 (48/48) | 0.287 (48/48) |
| Pair and three-site terms | 0.221 (40/48) | 0.174 (48/48) | 0.116 (48/48) | 0.178 (48/48) |

The three binary mixed-alphabet roles in this discovery benchmark each used two observed states. Its numerical rates apply to this panel size, depth, generating laws and distractor design. Larger alphabets, shared descendants and additional molecular systems need separate calibration.

## Reproduction

From the repository root, run the following in the package environment. The run took 29.84 seconds with a peak process working set of 39,841,792 bytes on Windows, Python 3.13.15 and NumPy 2.5.5, with BLAS and OpenMP set to one thread. The script prints aggregate JSON and requires no external data.

```bash
python benchmarks/candidate_discovery.py
```
