# Contributing

Install from a local checkout using Python 3.11 or newer. The development test suite uses plotting and optimizer extras. Work from a dedicated virtual environment.

```sh
python -m venv .venv
# POSIX: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements/release.txt
python -m pip install -e ".[dev,plots,optimize]"
python -m pytest -q -rs
python -m ruff check src tests tools/release
python -m mypy
```

Include a small reproducer and the expected scientific or user-visible behavior with a change. Preserve source-coordinate mappings, exact observation IDs, score units and rank direction. For changes to scores or nulls, include an independent mathematical reference or known-truth fixture and retain failed or uncertain outcomes. Supply synthetic fixtures or material with documented permission to redistribute. Keep raw user data and access tokens outside patches and public discussions.

The release guide contains the format-check scope, coverage measurement, minimum-dependency lane, installed-distribution verification and platform matrix. The owner decides license, citation credit and publication. [Release engineering](docs/release-engineering.md)
