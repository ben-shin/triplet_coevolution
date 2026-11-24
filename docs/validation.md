# Validation record

## Phase 0 · November 24, 2026

**The unmodified 1.2.0 wheel passed all 741 baseline tests on Linux.** Native Windows passed 731 tests and skipped 10 external-search cases. The owner’s earlier 462 passes and 9 skips describe a different revision. Testing used the archived commit `4d5ae65b57e20fd93b483c2d8825481076eecd55` (Git tree `af9ce750808a13afa5988329512fbe01fd9b3050`). The dedicated local branch began at that commit following inspection of status, log, tracked files, existing tests and instructions. [Specification audit](specification-audit.md) · [Phase gates](implementation-ledger.md)

### Installation and regression

Both distributions were built from the Git archive. The source distribution supplied the source for the wheel in the default two-step build. On Windows, a new CPython virtual environment installed the wheel and development/plotting dependencies from a downloaded local wheelhouse with the package index disabled. A second new base-only environment built and installed the sdist with the index disabled. Both passed `pip check` and printed `triplet 1.2.0`. Matplotlib was absent from the base install. On Linux, a fresh wheel install directory and tests extracted from the sdist used an existing scientific environment and external binaries. In both setups the imported package path was checked outside the repository. No private input folders or live metadata providers were used in these software checks.

| Check | Actual outcome |
|---|---|
| Windows 11 Pro 10.0.26300, CPython 3.13.15 AMD64 | `python -m pytest -q -rs -p no:cacheprovider tests` passed 731, skipped 10 in 27.70 s (28.14 s process wall). Skips were one protein hmmsearch, two nhmmer, one Infernal, one PSI-BLAST and five HMMER domain/reference tests. |
| Windows dependency floor, CPython 3.11.16 | A separate fresh wheel target with NumPy 1.26.0, Biopython 1.86, Typer 0.26.0, Matplotlib 3.8.0 and pytest 8.0.0 passed 731, skipped the same 10 in 26.44 s. There were 13 Matplotlib/pyparsing deprecation warnings. |
| WSL2 Linux 6.18.58.1, CPython 3.14.4 GCC 15.2.0 | Same pytest command passed 741 in 21.77 s. Installation, suite and environment capture together took 22.94 s. Local generated databases exercised HMMER 3.4 (hmmbuild, hmmsearch, nhmmer), Infernal 1.1.5 (cmbuild, cmcalibrate, cmsearch), NCBI BLAST+ 2.16.0+ (makeblastdb, psiblast). |
| Static checks and build | Ruff 0.17.6 `ruff check src tests benchmarks` passed (0.04 s repeat). Advisory `ruff format --check src tests benchmarks` reported 32 files needing formatting and 60 already formatted. The baseline has no configured type checker or format gate. A timed repeat of `python -m build --no-isolation` took 2.84 s. |
| Synthetic mixed aligned demo | The installed `triplet analyze` quickstart produced 8 observations, weight sum 8.00 and retained 4/4/3 protein/DNA/RNA columns. SHA-256 of `mi_ab.tsv` on Windows was `4bdc4024afa1401f9028461c2b9c6c523c6df8165a44bd45521dddc98313e22a`. |
| Synthetic raw database demo | Installed `triplet run --config examples/workflow/run.json` matched 8 groups, excluded one, tested all 800 triplets and obtained finite estimates for all 800. It wrote the offline report, paired scores, synthetic contact benchmark and seeded 19-draw calibration. Windows `threebody_abc.tsv` SHA-256 was `e31169a8503ed3bbda35988c9267048987c24afb4ec30fec5c94e6767fffdc23`. Version/help and both examples took 3.10 s together. |

The clean base-only sdist installation also ran the full synthetic workflow and reproduced the wheel’s three-body table byte for byte.

Branch-aware coverage used coverage.py 7.17.5 against the installed Windows wheel. A fresh `coverage run --branch --source=triplet_coevolution -m pytest -q -p no:cacheprovider tests` produced 731 passes and 10 skips. Whole-package statement coverage was 5,064/5,494 (92.17%) and branch coverage was 2,082/2,470 (84.29%). The `core/` package covered 224/236 statements and 86/98 branches, and `validation/` covered 94/104 and 32/42 respectively. Core and validation branch coverage are both below the suggested 90% target. The measurement covers the main Python process.

**Environment.** The host has an AMD Ryzen 7 9800X3D, 8 physical cores, 16 logical processors and 31.11 GiB physical RAM (33,409,032,192 bytes). Windows used NumPy 2.5.5 and its scipy-openblas 0.3.34.237.2 ILP64 build for BLAS/LAPACK. Both hosts used Biopython 1.88, Typer 0.29.2, Matplotlib 3.11.2 and pytest 9.2.0. Windows build tools were build 1.6.1, setuptools 84.2.1 and wheel 0.48.0. Git was 2.55.0.windows.5. Windows tests, coverage and timings set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`.

<details>
<summary>Remaining Windows environment versions</summary>

```text
annotated-doc 0.0.7        annotated-types 0.8.0     colorama 0.4.6
contourpy 1.4.0            cycler 0.12.1             fonttools 4.68.0
iniconfig 2.3.1            kiwisolver 1.5.1          markdown-it-py 4.3.0
mdurl 0.1.2                packaging 26.3            pillow 13.0.0
pip 26.2.1                 pluggy 1.7.0              pydantic 2.14.4
pydantic_core 2.50.5       Pygments 2.21.0           pyparsing 3.3.3
pyproject_hooks 1.3.3      python-dateutil 2.9.0.post0
rich 15.0.0                shellingham 1.5.4         typing_extensions 4.16.0
typing-inspection 0.4.5    tzdata 2026.6
```

</details>

Reproduce from an empty scratch directory using an archive of the SHA above and an activated Python 3.13 environment. Set the two thread variables above to `1` before timing. Use the same dependency versions for numerical and timing comparison. To replay without network access, populate a platform-specific wheelhouse first, then pass `--no-index --find-links <wheelhouse>` to pip for all installs, including build tools. Use fresh output directories for the examples.

```shell
python -m pip install build==1.6.1 setuptools==84.2.1 wheel==0.48.0
python -m build --no-isolation
python -m pip install "dist/triplet_coevolution-1.2.0-py3-none-any.whl[dev,plots]"
python -m pip check
triplet --version
triplet --help
python -m pytest -q -rs -p no:cacheprovider tests
ruff check src tests benchmarks
triplet analyze --a examples/mixed/protein.fasta --b examples/mixed/dna.fasta --c examples/mixed/rna.fasta --b-type dna --c-type rna --matches examples/mixed/matches.tsv --a-ref prot_ref --b-ref dna_ref --c-ref rna_ref --out scratch/mixed
triplet run --config examples/workflow/run.json --out scratch/full
python benchmarks/matched_scaling.py --sizes 256 2048 4096 --lengths 72 30 24 --permutations 5 --repeats 5 --seed 2026 > scaling.json
```

The tested baseline Windows wheel SHA-256 was `ce96af1d2a82c4a6fd97d9350574efae4c8961dbc0033d0b10b775528041da2b` and the sdist SHA-256 was `b9d7ed29f2983033d46f7a397c2cf6378a57956713f2b59a6fe725ba74825f6c`. These identify the tested build. Generated archive timestamps can differ on a rebuild.

### Synthetic performance before the research-platform changes

The benchmark generates nested cohorts from 64 founder triplets over 8 strata. It redraws 10% of residues, assigns 6% gaps and 2% unknowns, and guarantees one canonical anchor column per molecule. All retained widths were 72/30/24. Identity threshold was 0.8, per-component overlap 0.5 and minimum valid observations 2. Calibration covered 4,608 MI/APC pairs and 16 local three-body candidates with five whole-component shuffles within strata. The calibration timer includes fresh matching weights and table writes, with original weights and MI computed before timing. There was one fresh process per size and job, with five timed repetitions within each. Reported ranges describe short-run variation on this machine. Peak RSS is the process high-water mark and includes imports. The complete driver took 35.07 s. [Benchmark methodology](performance.md)

| Matched rows | Weights, median s (min–max) | Calibration, median s (min–max) | Peak RSS MiB, weights / calibration |
|---:|---:|---:|---:|
| 256 | 0.00509 (0.00497–0.00518) | 0.1422 (0.1408–0.1443) | 40.3 / 42.3 |
| 2,048 | 0.1369 (0.1246–0.1599) | 0.7655 (0.7593–0.8812) | 45.1 / 67.3 |
| 4,096 | 0.5139 (0.4879–0.5217) | 4.6929 (4.5113–4.7898) | 48.0 / 51.6 |

At 4,096 observations the sum of weights was 90.07435573143195. The ordered neighbor counts hashed to `ee5a07bdc2c49878388e044c2bf16d3dced3313dea584452e2d354bc657ff52b`. The ordered MI p/q array hashed to `6ee0e06286743c5f8616e24159249ec3d2c5e8ccf6a78daf21561cc41ec536bb`, and the triple p/q array to `acfc7c36fce590b14adfc8a479e0dcd297e910a7212694c976c8b4284d34c144`. These record baseline output for parity testing, with no claim of a performance improvement on this branch or of calibration on biological data.

### Legacy table replay and independent refit

The owner subsequently supplied private alignments, cohort evidence, original scores on each of the 2,880 structural labels and the structure coordinates. The compact bundle passed its supplied checksums. Its original labeled scores reproduce the historical contact metrics. The frozen v1.0.0 refit produced pooled raw-MI average precision of 0.5630831846447628, compared with 0.5630838071254171 on the original scores. Six scores shared a near-zero raw-MI tie in the original data. Five shifted by one representable floating-point step during refitting. Noninterpolated AP depends on tied rank groups, including for scores indistinguishable at a coarser numeric tolerance. The replay retains the original tolerances and records this discrepancy. The private refit took 23.28 s with one BLAS thread and peaked at 464.50 MiB RSS. This comparison reuses supplied labels and a balanced set of distant controls. Independently verifying residue correspondence, distances and the complete observable universe belongs to Phase 2. [Legacy dataset manifest and replay instructions](../benchmarks/gatabc_manifest.json)
