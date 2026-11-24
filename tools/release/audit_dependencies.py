"""Audit resolved public dependencies without sending local project metadata or paths."""

import argparse
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def resolved_public_dependencies(roots):
    found = {}
    visited = set()
    pending = list(roots)
    while pending:
        req = Requirement(pending.pop())
        name = canonicalize_name(req.name)
        state = (name, frozenset(req.extras))
        if state in visited:
            continue
        if req.url or name == "triplet-coevolution":
            raise ValueError("audit requires named public dependencies")
        dist = metadata.distribution(name)
        found[name] = dist.version
        visited.add(state)
        for item in dist.requires or ():
            child = Requirement(item)
            if child.marker is None or any(
                child.marker.evaluate({"extra": extra}) for extra in {"", *req.extras}
            ):
                pending.append(item)
    return [f"{name}=={found[name]}" for name in sorted(found)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional local pip-audit JSON report")
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Check the resolved runtime and scientific extras.",
    )
    args = parser.parse_args()
    runtime = ("numpy", "biopython", "typer", "scipy", "matplotlib")
    tools = (
        "pytest",
        "coverage",
        "mypy",
        "ruff",
        "packaging",
        "pip-audit",
        "build",
        "setuptools",
        "wheel",
        "pip",
    )
    requirements = resolved_public_dependencies(
        runtime if args.runtime_only else (*runtime, *tools)
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "public-requirements.txt"
        path.write_text("\n".join(requirements) + "\n", encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "pip_audit",
            "--disable-pip",
            "--no-deps",
            "--strict",
            "-r",
            str(path),
        ]
        if args.output:
            cmd.extend(["--format", "json", "--output", str(args.output.resolve())])
        print(f"Auditing {len(requirements)} public package/version pairs.", flush=True)
        subprocess.run(cmd, cwd=tmp, check=True)


if __name__ == "__main__":
    main()
