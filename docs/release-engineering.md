# Release engineering

The distribution is named `triplet-coevolution`, imports as `triplet_coevolution` and installs the `triplet` console command. The version comes from package metadata and `triplet --version`. Python 3.11 is the declared floor. The CI matrix requests CPython 3.11 through 3.15 on Linux and both endpoints on macOS and Windows. A separate Python 3.11 lane pins the minimum NumPy, Biopython and Typer versions plus older compatible plotting and optimization extras. The other lanes resolve the latest compatible dependencies for that interpreter. [Python version status](https://devguide.python.org/versions/)

## Quality checks

Use a clean virtual environment. The release requirements pin direct build and validation tools. Runtime dependency ranges remain in pyproject.toml. The source contains older modules with a different indentation style, so formatting currently covers the validation and structure packages, release checks and their new regressions. Lint checks the entire library and test tree. Mypy checks function bodies in alignment, coordinate, structure type, FDR and Newick validation modules.

```sh
python -m pip install -r requirements/release.txt -e ".[dev,plots,optimize]"
python -m pip check
python -m ruff check src tests tools/release
python -m ruff format --check src/triplet_coevolution/validation src/triplet_coevolution/structure tools/release tests/test_release_validation.py
python -m mypy
python -m coverage run -m pytest -q -rs
python -m coverage combine
python -m coverage report
python -m coverage json -o runner-coverage.json
python tools/release/coverage_summary.py runner-coverage.json
python tools/release/audit_dependencies.py
```

Coverage counts distinct conditional exits separately from executable statements and includes child Python processes. The goal is at least 90% branch coverage for each of core and validation. The summary prints the measured numerator, denominator and regression floor for each. A high combined statement-and-branch percentage alone cannot establish that the branch target was met. The suite also checks analytical truth, seed and ordering invariance, unsafe input rejection, code/data identity, and actual interrupted processes.

The workflow grants read-only repository contents, disables checkout credential persistence and pins the two official actions by full commit. Routine PR jobs run isolated editable, quality and distribution checks. A scheduled or manually dispatched Linux job installs HMMER, Infernal and BLAST+, records package versions, requires their commands on PATH and exercises real adapters and fixed-seed simulations. Private GatABC tests require explicitly supplied paths. Hosted CI does not request those paths. No workflow publishes a package or sends source code or sequence data to an audit service. [GitHub Actions secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use) · [Pinned checkout](https://github.com/actions/checkout/tree/3d3c42e5aac5ba805825da76410c181273ba90b1) · [Pinned Python setup](https://github.com/actions/setup-python/tree/6d4fd22b20465ccc0f7354476588e8df7c9ca0e0)

The dependency audit selects the transitively installed public runtime, optional and release tools from a clean environment. It sends exact package names and versions to the PyPI advisory service. Installed project metadata, local paths, repositories, private packages and biological inputs are outside the request. A new advisory produces a failing job for triage. The helper makes no automatic package upgrades. [PyPA pip-audit documentation](https://github.com/pypa/pip-audit)

## Build and install outside the checkout

Build from a clean source tree into a new artifact directory. The standard frontend builds the sdist and then the wheel from the sdist. The build hook records Git revision and working-tree state when available. For the first check, retrieve the public dependency wheels for your interpreter and platform. The validator then uses a fresh virtual environment per artifact, disables the pip index, validates local report assets, exercises the installed console entry point and runs bundled mixed and raw workflows without plotting extras.

```sh
python -m build --no-isolation --outdir release-dist
python tools/release/check_distributions.py --dist release-dist --wheelhouse release-wheelhouse
python tools/release/check_distributions.py --dist release-dist --wheelhouse release-wheelhouse --offline
```

The verifier prints SHA-256 hashes for the downloaded wheels. Retain that inventory, the wheelhouse, the application artifacts and their hashes together for same-platform offline replay. Building for another Python, OS or architecture can require a different wheelhouse. Version ranges and version pins alone do not identify the bytes of downloaded artifacts. Preserve the resolved dependency list with `python -m pip freeze --all`.

The wheel includes the local HTML explorer's JS and CSS, synthetic examples, documentation, the optional Slurm template, and benchmark code and provenance. The resource checker imports established modules, regenerates an offline report and checks the original score payload for mutation. Packaging declares each demonstration input explicitly. Input workspaces and result trees are excluded from the sdist. Use a clean source checkout and inspect archive members before any owner-approved distribution.

## Local qualification on 24 November 2026

Windows CPython 3.13.15 measured 98/98 core branch exits and 236/236 executable statements (both 100%). Validation exercised 925/1,110 branch exits (83.33%) and 2,732/2,946 statements (92.74%). Reaching 90% on this measured validation tree would require exercising 74 additional exits. CI enforces regression floors of 90% for core and 82% for validation. The suite includes additional cases for invalid coordinate maps, train/test isolation, tree provenance, degenerate model fits and local shard recovery.

The wheel and source distribution each installed into a fresh base-only environment from a local wheelhouse. Both ran the two bundled eight-observation workflows and regenerated the offline report and explorer. The SHA-256 of the original mixed workflow score payload after regeneration was `421730e9d75beea9819f917302dfbd571e2bdea8730546db530afb3ff3a5f5d2` in each case. On WSL Linux, all 24 external adapter and indexed workflow tests passed with HMMER 3.4, Infernal 1.1.5 and BLAST+ 2.16.0+. Actionlint 1.7.12 checked the workflow syntax locally. Hosted jobs, including macOS, still need a hosted run.

The minimum dependency audit identified a Biopython advisory affecting the previous floor. The revised floor is 1.87. After that change the audit reported zero known vulnerabilities among 21 public package/version pairs in the minimum scientific environment and 62 in the current release-tool environment. This records the advisory service's responses on the test date. [PyPA advisory PYSEC-2026-1221](https://osv.dev/vulnerability/PYSEC-2026-1221)

## Attribution, licensing and publication

- **Project software.** The owner has not supplied a project license, SPDX expression or verified citation authors. Package metadata intentionally contains no project license or author assertions. A valid CITATION.cff requires author information. To complete it, the owner should confirm the credited individuals or organization, citation title and preferred version or persistent identifier. ORCID and DOI can be included when verified. The repository URL and package version can already be recorded in a methods section. [Citation File Format schema](https://github.com/citation-file-format/citation-file-format/blob/main/schema-guide.md)
- **Dependencies.** The installed Windows distributions declared `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` for NumPy 2.5.5, `LicenseRef-Biopython-License-Agreement` for Biopython 1.88 and `MIT` for Typer 0.29.2. SciPy 1.18.1 had a BSD classifier and Matplotlib 3.11.2 had a Python Software Foundation license classifier. Included notices identify additional terms for bundled components. Review the notices for the exact files and platforms before redistributing a wheelhouse or environment image. The independent plmc comparison identifies the upstream commit and its license separately. [plmDCA comparison](plm.md)
- **Public and private data.** The independent cohort records identify GTDB R232 and NCBI RefSeq source accessions. GTDB publishes its database under CC BY-SA 4.0. The PDB archive uses CC0 and requests attribution; integrated third-party resources can have separate terms. Raw private GatABC inputs require separate owner authorization. Review attribution and the terms applicable to each distributed table, modification and source before publishing. [GTDB licensing](https://gtdb.ecogenomic.org/licenses) · [RCSB policies](https://www.rcsb.org/pages/policies) · [NCBI policies](https://www.ncbi.nlm.nih.gov/home/about/policies/)

Owner decisions on software licensing, credited authors, third-party redistribution and an actual release remain open. A successful local build or hosted check is technical evidence only.
